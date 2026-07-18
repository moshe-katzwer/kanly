"""Abstract model base combining cluster-group state with fit and build protocols.

``ModelBase`` sits one level above ``CovarianceClusterGroupsModelBase`` in the
inheritance chain and adds the response data (``endog``), formula metadata, and
timing information that all concrete regression models share.  It declares three
abstract methods that every subclass must implement: ``fit``,
``build_model_from_formula``, and ``build_model``.
"""

from __future__ import absolute_import, print_function

from abc import abstractmethod

import numpy as np
from scipy.sparse import isspmatrix

from kanly import _IN_NOTEBOOK
from kanly.regression.covariance_cluster_groups_model_base import CovarianceClusterGroupsModelBase


class ModelBase(CovarianceClusterGroupsModelBase):
    """Abstract base class for all kanly regression models.

    Extends ``CovarianceClusterGroupsModelBase`` with the response array,
    formula string, ``from_formula`` flag, and model-construction timing.
    Concrete subclasses must implement ``fit``, ``build_model_from_formula``,
    and ``build_model``.
    """

    def __init__(self, nobs, endog, index=None, valid_obs_rows=None, specification_name=None,
                 formula_design_info=None, model_elapsed=0, cov_groups=None, cov_groups_name=None,
                 is_sure=False, parent_model=None):
        """Initialise shared model state.

        Args:
            nobs: Number of observations retained after null-row removal.
            endog: Response array (1-D or 2-D for multi-outcome models).
            data: Source DataFrame used for formula-based models; ``None`` for
                array-only models.
            index: Row selector applied to ``data`` before null removal.
            valid_obs_rows: Boolean or integer mask of rows retained after
                null-row processing.
            specification_name: Optional human-readable label used in
                summaries.
            formula: Formula string describing the model (may be ``None`` for
                array-based construction).
            from_formula: Whether the model was constructed from a formula;
                controls ``build_model`` behaviour.
            model_elapsed: Wall-clock seconds taken to parse the formula and
                build design matrices.
            cov_groups: Pre-resolved cluster-group array, or ``None``.
            cov_groups_name: Column name or label for ``cov_groups``.
            is_sure: Whether this is a SURE (Seemingly Unrelated Regression
                Equations) stacked model.
            parent_model: Reference to a parent model when this model is a
                sub-component of a larger system.
        """

        super().__init__(nobs, index=index, valid_obs_rows=valid_obs_rows,
                         formula_design_info=formula_design_info,
                         specification_name=specification_name, cov_groups=cov_groups, cov_groups_name=cov_groups_name,
                         is_sure=is_sure, parent_model=parent_model)

        if not isspmatrix(endog):
            endog = np.asarray(endog)
        self.endog = endog
        self.model_elapsed = model_elapsed

    @abstractmethod
    def fit(self, *args, **kwargs):
        """Fit the model and return a results object.

        Every concrete subclass must implement this method to estimate model
        parameters and produce a ``RegressionResultsBase`` (or subclass)
        instance.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def build_model_from_formula(formula, data, index=None, debug=False, *args, **kwargs):
        """Construct and fit a model from a patsy formula string.

        Args:
            formula: Patsy formula string describing the model
                (e.g. ``'y ~ x1 + x2'``).
            data: DataFrame containing all variables referenced by ``formula``.
            index: Optional row-selector applied to ``data`` before fitting.
            debug: If ``True``, emit extra diagnostic output during model
                construction.
            *args: Additional positional arguments passed to the concrete
                implementation.
            **kwargs: Additional keyword arguments passed to the concrete
                implementation.
        """
        raise NotImplementedError()

    @abstractmethod
    def build_model(self, data, index=None, debug=False, *args, **kwargs):
        """Rebuild the model on new data using the stored formula.

        Args:
            data: DataFrame containing all variables referenced by the stored
                formula.
            index: Optional row-selector applied to ``data`` before fitting.
            debug: If ``True``, emit extra diagnostic output during model
                construction.
            *args: Additional positional arguments passed to the concrete
                implementation.
            **kwargs: Additional keyword arguments passed to the concrete
                implementation.
        """
        raise NotImplementedError()
