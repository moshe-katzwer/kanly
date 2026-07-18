"""Mixin base class for resolving cluster/group identifiers for robust standard errors.

All kanly regression models that support cluster-robust, HAC-panel, or SURE
(Seemingly Unrelated Regression Equations) estimation inherit from
``CovarianceClusterGroupsModelBase``.  It provides a unified interface for
resolving a "groups" specification – which may be a column name string, a
patsy formula expression, a raw array, or a tuple of any of the above – into
a concrete integer or string array of cluster labels.

The resolution logic lives entirely in the static
``get_covariance_groups_internal`` method, keeping model state management
(``get_covariance_groups`` / ``set_covariance_groups``) separate from the
resolution algorithm itself.
"""

from __future__ import absolute_import, print_function

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import Series
from patsy.missing import NAAction
from patsy.highlevel import dmatrix
from scipy.sparse import isspmatrix
from scipy.sparse import spmatrix

from kanly import _IN_NOTEBOOK
from kanly.dill_object import DillObject
from kanly.formula.formula_design_info import FormulaDesignInfo
from kanly.utils.util import get_eval_env_depth


class CovarianceClusterGroupsModelBase(DillObject):
    """Mixin that manages cluster-group arrays for robust covariance estimation.

    Stores the resolved cluster-group array (``cov_groups``) and its label
    (``cov_groups_name``) as instance attributes and exposes convenience
    methods for resolving user-supplied group specifications against the
    model's data at fit time.  All resolution paths – column lookup, patsy
    formula evaluation, and direct array passthrough – are handled here so
    that subclasses do not need to duplicate the logic.

    This class is the root of the kanly model inheritance chain (just above
    ``DillObject``).
    """

    def __init__(self, nobs, index=None, valid_obs_rows=None,
                 formula_design_info=None, specification_name=None,
                 cov_groups=None, cov_groups_name=None, is_sure=False, parent_model=None):
        """Initialise cluster-group state and common model metadata.

        Args:
            nobs: Number of observations after null-row removal.
            data: Source DataFrame used for column lookups and formula
                evaluation; ``None`` for array-only models.
            index: Row-selector applied to ``data`` before null-row removal.
            valid_obs_rows: Boolean mask or integer indices of valid rows
                within ``data.loc[index]``; defaults to ``np.arange(nobs)``
                when ``None``.
            specification_name: Human-readable label used in result summaries.
            cov_groups: Pre-resolved cluster-group array, or ``None``.
            cov_groups_name: Column name or label for ``cov_groups``; set to
                ``'<cov_groups>'`` automatically when ``cov_groups`` is
                non-``None`` and no name is given.
            is_sure: Whether this is a SURE stacked model whose data and
                index are lists (one element per equation).
            parent_model: Reference to a parent model for sub-model contexts.
        """
        self.nobs = nobs
        self.index = index

        self.formula_design_info: FormulaDesignInfo = formula_design_info
        self.from_formula = formula_design_info is not None
        self.formula = formula_design_info.formula if self.from_formula else None

        self.specification_name = specification_name

        if valid_obs_rows is None:
            valid_obs_rows = np.arange(self.nobs)
        self.valid_obs_rows = valid_obs_rows

        self.cov_groups = cov_groups
        if cov_groups_name is None and cov_groups is not None:
            cov_groups_name = '<cov_groups>'
        self.cov_groups_name = cov_groups_name

        self.is_sure = is_sure

        self.parent_model = parent_model

    def get_covariance_groups(self, cov_groups, debug=False):
        """Resolve ``cov_groups`` against the model's stored data and valid-row mask.

        Delegates to ``get_covariance_groups_internal`` using the instance's
        ``nobs``, ``data``, ``index``, ``valid_obs_rows``, and current
        ``cov_groups`` / ``cov_groups_name`` state.

        Args:
            cov_groups: Group specification; see
                ``get_covariance_groups_internal`` for accepted types.
            debug: If ``True``, enable debug output in the resolution step.

        Returns:
            Tuple of (cov_groups_array, cov_groups_name) as produced by
            ``get_covariance_groups_internal``.
        """
        if self.formula_design_info is not None:
            data = self.formula_design_info.data
        else:
            data = None
        return self.get_covariance_groups_internal(
            self.nobs, cov_groups, data, self.index, self.valid_obs_rows,
            current_cov_groups=self.cov_groups, current_cov_groups_name=self.cov_groups_name,
            is_sure=self.is_sure, debug=debug)

    def set_covariance_groups(self, cov_groups):
        """Resolve ``cov_groups`` and store the result on the instance.

        Updates ``self.cov_groups`` and ``self.cov_groups_name`` in-place by
        calling ``get_covariance_groups`` and unpacking the resulting tuple.

        Args:
            cov_groups: Group specification; see
                ``get_covariance_groups_internal`` for accepted types.
        """
        self.cov_groups, self.cov_groups_name = self.get_covariance_groups(cov_groups)

    @staticmethod
    def get_cov_group_keyword(cov_kwds):
        """Extract the ``'groups'`` value from a ``cov_kwds`` dict.

        Provides a safe accessor that handles the multiple types a user may
        pass for cluster groups: a column name string, a tuple of column
        names (multi-way clustering), or a raw array/Series.

        Args:
            cov_kwds: Optional dict of covariance keyword arguments (as
                produced by ``format_cov_kwds``).

        Returns:
            The value associated with ``'groups'`` in ``cov_kwds`` if it is
            a str, tuple-of-strings, ``pd.Series``, ``np.ndarray``, or
            ``list``; ``None`` otherwise (including when ``cov_kwds`` is
            ``None`` or ``'groups'`` is absent or ``None``).
        """

        if cov_kwds is None:
            return None

        if 'groups' in cov_kwds:
            if cov_kwds['groups'] is None:
                return None
            elif isinstance(cov_kwds['groups'], str):
                return cov_kwds['groups']
            elif isinstance(cov_kwds['groups'], tuple) and np.all(isinstance(x, str) for x in cov_kwds['groups']):
                return tuple(cov_kwds['groups'])
            elif np.any([isinstance(cov_kwds['groups'], t) for t in [Series, ndarray, list]]):
                return cov_kwds['groups']
            else:
                return None  # TODO should this None, or Exception?

        return None

    @staticmethod
    def get_covariance_groups_internal(
            nobs, cov_groups, data=None, index=None, valid_obs_rows=None, debug=False,
            current_cov_groups=None, current_cov_groups_name=None, is_sure=False):
        """Resolve a group specification into a concrete cluster-label array.

        Handles four resolution paths:

        1. **Tuple**: Multiple clustering variables; each element is resolved
           recursively and the results are collected into a list (multi-way
           clustering).
        2. **SURE stacked model** (``is_sure=True``): Resolves group arrays
           for each equation and horizontally stacks them.
        3. **String**: Column-name or patsy formula.

           - If the name matches the already-stored ``current_cov_groups_name``
             the existing array is returned directly (cache hit).
           - Otherwise the column is looked up in ``data.columns``; if absent,
             it is evaluated as a patsy formula expression.
        4. **Array/Series**: Passed through after a shape check.

        Args:
            nobs: Expected number of observations in the resolved array.
            cov_groups: One of: ``None``, a column-name string, a patsy
                expression string, a tuple of strings/arrays (multi-way
                clustering), a ``pd.Series``, an ``np.ndarray``, or a
                ``scipy.sparse.spmatrix``.
            data: Source DataFrame for column lookup and formula evaluation.
            index: Row-selector applied to ``data``.
            valid_obs_rows: Mask of valid rows within ``data.loc[index]``.
            debug: Reserved for future diagnostic output.
            current_cov_groups: Previously resolved array (used for cache
                hit on string path).
            current_cov_groups_name: Label for ``current_cov_groups``.
            is_sure: Whether the model is a SURE stacked model.

        Returns:
            Tuple of (cov_groups_array_or_list, cov_groups_name).  Both
            elements are ``None`` when ``cov_groups`` is ``None``.

        Raises:
            NotImplementedError: If ``is_sure=True`` and a tuple is passed
                (multi-way SURE grouping is not yet implemented).
            Exception: If resolution fails (e.g. string not found in data,
                shape mismatch, unsupported type).
        """

        if isinstance(cov_groups, tuple):
            if is_sure:
                raise NotImplementedError("need to implement multiple cov groups for sure")
            cov_groups = [CovarianceClusterGroupsModelBase.get_covariance_groups_internal(
                nobs, cg, data=data, index=index, valid_obs_rows=valid_obs_rows,
                current_cov_groups=current_cov_groups, current_cov_groups_name=current_cov_groups_name, is_sure=is_sure)
                for cg in cov_groups
            ]
            cov_groups_name = str(tuple([c[1] for c in cov_groups]))
            cov_groups = [c[0] for c in cov_groups]
            return cov_groups, cov_groups_name

        if is_sure:
            # Resolve cluster groups for each SURE equation independently,
            # then stack horizontally so the resulting array has one entry
            # per stacked observation.
            temp = [
                CovarianceClusterGroupsModelBase.get_covariance_groups_internal(
                    n, cov_groups, data=d, index=ii, valid_obs_rows=vor,
                    current_cov_groups=current_cov_groups, current_cov_groups_name=current_cov_groups_name,
                    is_sure=False)
                for d, ii, vor, n
                in zip(data, index, valid_obs_rows, [len(v) for v in valid_obs_rows])
            ]
            cov_groups_list = np.array([x[0] for x in temp])
            cov_groups_name_list = np.array([x[1] for x in temp])
            if (cov_groups_list == None).any():
                return None, None
            else:
                if len(set(cov_groups_name_list)) > 1:
                    raise Exception
                return np.hstack([x[0] for x in temp]), cov_groups_name_list[0]

        if cov_groups is None:
            cov_groups_name = None
            cov_groups = None
            return cov_groups, cov_groups_name

        else:
            groups_val = cov_groups

            if isinstance(groups_val, str):

                # If the string matches the name in the model, and there is data,
                # don't change anything
                if groups_val == current_cov_groups_name and current_cov_groups is not None:
                    return current_cov_groups, current_cov_groups_name

                # Otherwise, need to check in the model's data
                if data is not None:

                    if index is None:
                        index = np.arange(len(data))
                    if valid_obs_rows is None:
                        valid_obs_rows = [True] * len(index)

                    if groups_val in data.columns:

                        val_arr = data[groups_val]
                        if isinstance(val_arr, pd.Series):
                            val_arr = val_arr.values

                        cov_groups = np.asarray(
                            val_arr[index][valid_obs_rows]).ravel()
                        # # Kind of weird index manipulation needed because final
                        # # model `valid_obs_rows` is ints relative to data index
                        # # and in this step we need bools relative to `index`
                        # if valid_obs_rows.dtype.name == 'bool':
                        #     cov_groups = np.asarray(
                        #         val_arr[index][valid_obs_rows]).ravel()
                        # else:
                        #     valid_obs_rows_temp = np.array([False] * len(data))
                        #     valid_obs_rows_temp[valid_obs_rows] = True
                        #     valid_obs_rows = valid_obs_rows_temp[index]
                        #     cov_groups = np.asarray(
                        #         val_arr[index][valid_obs_rows]).ravel()

                        if cov_groups.shape[0] != nobs:
                            raise Exception(
                                f"Covariance groups have shape {cov_groups.shape} but there 'are {nobs} nobs!")
                        cov_groups_name = groups_val
                        return cov_groups, cov_groups_name
                    else:
                        try:
                            if _IN_NOTEBOOK:
                                vals = dmatrix(f'I({groups_val})-1', data.loc[index],
                                               NA_action=NAAction(NA_types=[])).flatten()
                            else:
                                vals = dmatrix(f'I({groups_val})-1', data.loc[index],
                                               NA_action=NAAction(NA_types=[]),
                                               eval_env=get_eval_env_depth()).flatten()
                            vals = vals[valid_obs_rows]

                        except Exception:
                            raise Exception(f"Could not parse {groups_val} as valid formula for data!")
                        cov_groups_name = groups_val
                        cov_groups = vals
                        if cov_groups.shape[0] != nobs:
                            raise Exception(
                                f"Covariance groups have shape {cov_groups.shape} but there 'are {nobs} nobs!")
                        return cov_groups, cov_groups_name
                else:
                    raise Exception(
                        "Model has null `data` attribute, cannot use a `groups` as string!")

            elif isinstance(groups_val, (ndarray, Series, spmatrix)):
                if isspmatrix(groups_val):
                    groups_val = groups_val.toarray().flatten()
                if groups_val.shape[0] != nobs:
                    raise Exception(
                        f"Covariance groups have shape {groups_val.shape} but there are {nobs} nobs!")
                cov_groups = groups_val
                cov_groups_name = None
                return cov_groups, cov_groups_name

            else:
                raise Exception(f"`groups` arg must be one of {{str, scipy.sparse.spmatrix, np.ndarray, pd.Series}},'"
                                f"f' not {type(groups_val)}!")
