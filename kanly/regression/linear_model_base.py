"""Abstract base class for linear regression models.

``LinearModelBase`` extends ``ModelBase`` with all the infrastructure that is
specific to models with a linear predictor: handling of an intercept column,
instrumental variables, weighted observations, fixed-effects absorption arrays,
variable naming, formula construction, and a concrete prediction pipeline.

Sub-classes must implement the abstract methods ``fit``, ``predict``,
``build_model_from_formula``, and ``accepts_multi_outcome``.
"""

from __future__ import absolute_import, print_function

from abc import abstractmethod

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.sparse import csc_matrix, isspmatrix, hstack as sp_hstack

from kanly.formula.keys import EXOG_KEY
from kanly.formula.formula_design_info import FormulaDesignInfo
from kanly.formula.keys import RETURN_CONSTANT_COLUMN_TERM_NAME
from kanly.regression.model_base import ModelBase
from kanly.sparse_data_frame import SparseDataFrame
from kanly.utils.linalg_utils import none_convert_2_sparse, gram_matrix


def add_constant_function(exog, instruments):
    """Prepend a column of ones to ``exog`` and (if present) ``instruments``.

    Used when ``add_constant=True`` is requested at model construction time.
    Supports both dense (``numpy.ndarray``) and sparse (``scipy.sparse``)
    input matrices; ``None`` instrument arrays are passed through unchanged.

    Args:
        exog: Design matrix (n_obs × k).
        instruments: Instrument matrix (n_obs × m), or ``None`` for non-IV
            models.

    Returns:
        Tuple of (exog_with_const, instruments_with_const) where each array
        has a leading column of ones appended.  If ``instruments`` is
        ``None``, the second element of the tuple is also ``None``.
    """
    ones = np.ones((exog.shape[0], 1))
    res = []
    for X in [exog, instruments]:
        if X is not None:
            if isspmatrix(X):
                X = sp_hstack([csc_matrix(ones), X])
            else:
                X = np.hstack([ones, X])
        res.append(X)

    return tuple(res)


class LinearModelBase(ModelBase):
    """Abstract base class for all kanly linear regression models.

    Stores and pre-processes all data arrays needed for a linear model:
    response vector, regressor matrix, optional instruments, observation
    weights, and an optional fixed-effects absorption matrix.  Also provides:

    - Placeholder variable-name generation (``make_var_names``).
    - Formula string synthesis (``make_formula``).
    - A concrete ``get_linear_predictor`` that supports dense, sparse, and
      DataFrame inputs.
    - Design-matrix condition number (``cond``) and rank (``rank``) helpers.
    - Sparsity fraction reporting (``sparsity``).
    - Formula term → design-column index maps (``exog_term_to_indices``,
      ``instrument_term_to_indices``) for RHS sparse_terms that expand to multiple
      columns (e.g. ``C(g)`` → ``[3, 4, 5, 6]``).

    Sub-classes must implement ``fit``, ``predict``, ``build_model_from_formula``,
    and the boolean query ``accepts_multi_outcome``.
    """

    def __init__(self, endog, exog, add_constant, has_intercept, has_implicit_constant, formula_design_info, weights=None,
                 instruments=None, endog_name=None, absorb=None, absorb_names=None, absorb_term_name=None,
                 cov_groups=None, cov_groups_name=None, exog_names=None, weights_name=None,
                 instrument_names=None, index=None, valid_obs_rows=None,
                 null_rows_info_dict=None, method=None, specification_name=None, endog_regressors=None,
                 model_elapsed=None, is_sure=False, parent_model=None):
        """Initialise all linear model arrays, names, and metadata.

        Args:
            endog: Response array (1-D or 2-D for multi-outcome models).
            exog: Regressor design matrix (n_obs × k).
            add_constant: If ``True``, prepend a column of ones to ``exog``
                and ``instruments`` via ``add_constant_function``.
            has_intercept: Whether the design matrix contains an explicit
                intercept column (set to ``True`` after constant addition).
            has_implicit_constant: Whether the model contains a constant
                through a sum-to-one constraint (e.g. saturated dummies).
            formula_design_info: FormulaDesignInfo object
            weights: Optional 1-D weight array.  Converted to a flat ndarray.
            instruments: Optional instrument matrix for IV estimation.
            endog_name: Column name for the response; defaults to
                ``'<y>'`` (or ``['<y0>', '<y1>', …]`` for multi-outcome).
            absorb: Optional sparse fixed-effects indicator matrix
                (n_obs × n_levels) to be absorbed.
            absorb_names: Column names of the absorbed FE levels.
            absorb_term_name: Human-readable label for the absorption term.
            cov_groups: Pre-resolved cluster group array.
            cov_groups_name: Label for ``cov_groups``.
            exog_names: Column names for ``exog``; defaults to
                ``['<x0>', '<x1>', …]``.
            exog_term_names: Term-level names for ``exog`` (may differ from
                ``exog_names`` for interaction sparse_terms); defaults to
                ``exog_names``.
            exog_term_to_indices: Dict mapping each formula RHS term name to
                the 0-based column indices of that term in ``exog``.  For
                example, ``{'C(g)': [3, 4, 5, 6], 'x': [1]}``.  One term can
                span several columns (dummy expansion, interactions, splines).
                Populated from formula parsing when available; otherwise filled
                in by :meth:`make_var_names` when ``None``.
            weights_name: Name of the weight variable; defaults to
                ``'<weights>'`` or ``'-'`` when absent.
            instrument_names: Column names for ``instruments``.
            instrument_term_names: Term-level names for ``instruments``.
            instrument_term_to_indices: Same as ``exog_term_to_indices``, but
                for the instrument design matrix (IV models only).
            index: Row-selector applied to ``data`` before null-row removal.
            valid_obs_rows: Boolean or integer mask of retained rows.
            null_rows_info_dict: Optional dict recording which rows were
                dropped and why.
            method: Estimation method label (e.g. ``'OLS'``, ``'2SLS'``).
            specification_name: Human-readable model label.
            endog_regressors: Endogenous regressors for IV models.
            model_elapsed: Wall-clock time to build the model.
            is_sure: Whether this is a SURE stacked model.
            parent_model: Reference to a parent model.
        """

        if np.ndim(exog) == 1:
            exog = exog.reshape((-1, 1))

        if add_constant:
            exog, instruments = add_constant_function(exog, instruments)
            has_intercept = True
            self.add_constant = add_constant

        super().__init__(exog.shape[0], endog, index, valid_obs_rows, specification_name,
                         formula_design_info, model_elapsed, cov_groups, cov_groups_name, is_sure, parent_model)


        self.has_intercept = has_intercept
        self.has_implicit_constant = has_implicit_constant

        # If any of the core matrices is sparse, promote all to sparse so that
        # downstream computations use a consistent code path.
        if (isspmatrix(exog)
                or (instruments is not None and isspmatrix(instruments))
                or (absorb is not None and isspmatrix(absorb))
        ):
            self.is_sparse_model = True
            exog = none_convert_2_sparse(exog)
            instruments = none_convert_2_sparse(instruments)
            absorb = none_convert_2_sparse(absorb)
        else:
            self.is_sparse_model = False

        self.exog = exog

        self.instruments = instruments
        self.is_iv = instruments is not None

        self.null_rows_info_dict = null_rows_info_dict

        if weights is not None and isspmatrix(weights):
            weights = weights.toarray()
        if weights is not None:
            weights = np.asarray(weights).flatten()
        self.weights = weights
        self.is_weighted = weights is not None

        (self.endog_name, self.exog_names, self.weights_name, self.instrument_names,
         self.exog_term_names, self.instrument_term_names, self.exog_term_to_indices, self.instrument_term_to_indices) \
            = LinearModelBase.make_var_names(endog, exog, weights, instruments, endog_name, exog_names, weights_name,
                                             instrument_names,add_constant, formula_design_info)


        if self.formula is None:
            self.formula = LinearModelBase.make_formula(
                self.endog_name, self.exog_names, self.instrument_names, self.weights_name)

        self.method = method
        self.specification_name = str(specification_name) if specification_name is not None else None

        self.is_absorb = absorb is not None
        self.absorb = absorb
        if absorb is not None:
            self.num_absorbed = absorb.shape[1]
            self.absorb_names = (absorb_names if absorb_names is not None
                                 else [f'<absorb{j}>' for j in range(absorb.shape[1])])
            self.absorb_term_name = absorb_term_name if absorb_term_name is not None else '<absorb>'
        else:
            self.num_absorbed = 0
            self.absorb_names = None
            self.absorb_term_name = None

        self.endog_regressors = endog_regressors
        self.is_endog_regressor = np.array([c not in self.instrument_names for c in self.exog_names])
        if self.endog_regressors is None:
            self.endog_regressors = {c not in self.instrument_names for c in self.exog_names}
        # if self.endog_regressors is None:
        #     print("BBBBBB")
        #     self.endog_regressors = self.exog_names
        #     self.is_endog_regressor = np.array([True] * len(self.exog_names))
        # else:
        #     print("AAAAA")
        #     self.is_endog_regressor = np.array([c not in self.instrument_names for c in self.exog_names])
        #     # self.is_endog_regressor = np.array([(c in self.endog_regressors)
        #     #                                     for i, c in enumerate(self.exog_names)])

        self.is_multi_outcome = np.ndim(self.endog) > 1 and self.endog.shape[1] > 1
        if isinstance(self.endog_name, list) and not self.is_multi_outcome:
            self.endog_name = endog_name[0]
        if self.is_multi_outcome and not self.accepts_multi_outcome():
            raise Exception(f"model of type {type(self)} does not allow multiple outcomes")

    @staticmethod
    def make_var_names(endog, exog, weights, instruments, endog_name, exog_names, weights_name, instrument_names,
                       add_constant, formula_design_info: FormulaDesignInfo):
        """Generate placeholder variable names for any array that lacks explicit names.

        Fills in ``'<y>'``, ``'<x0>'``, ``'<z0>'``, etc. for unnamed arrays,
        leaving caller-supplied names untouched.

        Args:
            endog: Response array; its rank determines whether a scalar or
                list name is generated.
            exog: Regressor array; its second dimension determines how many
                ``'<xj>'`` names to generate.
            weights: Weight array or ``None``; triggers generation of
                ``'<weights>'`` when non-``None``.
            instruments: Instrument array or ``None``; triggers generation of
                ``'<zj>'`` names when non-``None``.
            endog_name: Existing name or ``None``.
            exog_names: Existing column names list or ``None``.
            weights_name: Existing weight name or ``None``.
            instrument_names: Existing instrument names list or ``None``.
            formula_design_info: FormulaDesignInfo object

        Returns:
            Tuple of (endog_name, exog_names, weights_name, instrument_names,
            exog_term_names, instrument_term_names, exog_term_to_indices,
            instrument_term_to_indices).
        """
        if endog_name is None:
            if np.ndim(endog) == 1 or endog.shape[1] == 1:
                endog_name = '<y>'
            else:
                endog_name = [f'<y{j}>' for j in range(endog.shape[1])]
        if exog_names is None:
            exog_names = [f'<x{j}>' for j in range(exog.shape[1])]
        elif isinstance(exog_names, dict):
            exog_names = [exog_names.get(j, f'<x{j}>') for j in range(exog.shape[1])]
        else:
            assert len(exog_names) == exog.shape[1]
        if instrument_names is None:
            if instruments is not None:
                instrument_names = [f'<z{j}>' for j in range(instruments.shape[1])]
            else:
                instrument_names = []
        if weights_name is None:
            if weights is not None:
                weights_name = '<weights>'
            else:
                weights_name = '-'

        if formula_design_info is None:
            exog_term_names = exog_names[add_constant:]
            instrument_term_names = instrument_names
            exog_term_to_indices = {name: [i] for i, name in enumerate(exog_names)}
            instrument_term_to_indices = {name: [i] for i, name in enumerate(instrument_names)}
        else:
            exog_term_names = formula_design_info.exog_term_names
            instrument_term_names = formula_design_info.exog_term_names
            exog_term_to_indices = formula_design_info.exog_var_2_col_indices
            instrument_term_to_indices = formula_design_info.instruments_var_2_col_indices

        return (
            endog_name, exog_names, weights_name, instrument_names,
            exog_term_names, instrument_term_names, exog_term_to_indices, instrument_term_to_indices
        )

    @staticmethod
    def make_formula(endog_name, exog_names, instrument_names, weights_name):
        """Build a formula string from variable names.

        Constructs the standard kanly formula notation:
        ``y ~ x1 + x2 [| z1 + z2] [$ weights]`` where ``|`` separates
        instruments and ``$`` separates the weight variable.

        Args:
            endog_name: Response variable name (str) or list of names for
                multi-outcome models.
            exog_names: List of regressor column names.
            instrument_names: List of instrument column names; omitted when
                empty or ``None``.
            weights_name: Weight variable name; omitted when ``'-'`` or
                ``None``.

        Returns:
            str: Assembled formula string.
        """
        if not isinstance(endog_name, str):
            endog_name = ' + '.join(endog_name)
        formula = endog_name + " ~ " + " + ".join(exog_names)
        if instrument_names is not None and instrument_names:
            formula += " | " + " + ".join(instrument_names)
        if weights_name is not None and weights_name != '-':
            formula += " $ " + weights_name
        return formula

    def __repr__(self):
        """Return the string representation (delegates to ``__str__``)."""
        return str(self)

    def __str__(self):
        """Return a human-readable summary of the model's key dimensions.

        Includes observation count, response name, regressor term names, and
        optional sections for instruments, weights, absorbed FE, cluster
        groups, and the source data shape.  Sections are omitted when the
        corresponding feature is absent.
        """
        return (
                "\n===================================================" +
                "\nKanly Regression Model Base" +
                "\n---------------------------------------------------" +
                ("\n" + self.specification_name if self.specification_name is not None else "") +
                f'\nNobs: {self.nobs}' +
                '\nEndog: %s' % str(self.endog_name) +
                '\nExog: %s' % str(self.exog_term_names) +
                ('\nInstruments: %s' % str(self.instrument_term_names)
                 if len(self.instrument_names) else "") +
                ('\nWeights: %s' % str(self.weights_name)
                 if self.weights_name != '-' else "") +
                (f"\nAbsorbed: '{self.absorb_term_name}', num={self.absorb.shape[1]}"
                 if self.absorb is not None else "") +
                (f"\nVar-Covar Groups: '{self.cov_groups_name}', num={len(np.unique(self.cov_groups))}"
                 if self.absorb is not None else "") # +
                # ('\nData: shape=%d x %d,\n      type=%s' % (self.data.shape[0], self.data.shape[1], type(self.data))
                #  if self.data is not None and (not self.is_sure if hasattr(self, 'is_sure') else True) else "") +
                # "\n---------------------------------------------------"
        )

    @abstractmethod
    def predict(self, data=None, params=None, index=None, debug=False, *args, **kwargs):
        """Generate predictions from the fitted model.

        Sub-classes must implement the full prediction contract, handling
        both in-sample (``data=None``) and out-of-sample (DataFrame/array)
        scenarios.

        Args:
            data: New data for out-of-sample prediction, or ``None`` to use
                the model's own ``exog``.
            params: Optional parameter vector to use instead of the estimated
                coefficients.
            index: Row-selector applied to ``data``.
            debug: If ``True``, enable verbose output.
        """
        raise NotImplementedError()

    @staticmethod
    def strip_formula_to_just_exog_terms(formula, strip_non_exog=False):
        """Optionally remove instrument and weight suffixes from a formula.

        When ``strip_non_exog=True``, removes the ``'| instruments'`` suffix
        (IV instruments) and the ``'$ weights'`` suffix, replacing the
        left-hand side with a constant-column placeholder so that the formula
        can be used purely to rebuild ``exog`` for prediction.

        Args:
            formula: Full formula string (e.g. ``'y ~ x | z $ w'``).
            strip_non_exog: If ``True``, strip instrument and weight sections.

        Returns:
            str: Stripped (or unmodified) formula string.
        """
        if strip_non_exog:
            if '$' in formula:
                formula = ''.join(formula.split('$')[:-1])
            if '|' in formula:
                formula = ''.join(formula.split('|')[:-1])
            formula = RETURN_CONSTANT_COLUMN_TERM_NAME + ' ~ ' + formula.split('~')[1]
        return formula

    def build_model(self, data, index=None, debug=False, strip_non_exog=False, check_constant_cols=True,
                    drop_1_for_FE=True, *args, **kwargs):
        """Rebuild the model on new data using the stored formula.

        Only supported when the original model was built from a formula
        (``self.from_formula=True``).  Optionally strips the instrument and
        weight suffixes from the formula for prediction-only rebuilds.

        Args:
            data: DataFrame containing all variables referenced by the stored
                formula.
            index: Optional row-selector applied to ``data``.
            debug: If ``True``, emit verbose output during construction.
            strip_non_exog: If ``True``, pass the formula through
                ``strip_formula_to_just_exog_terms`` before rebuilding.
            check_constant_cols: Whether to check for constant columns in the
                design matrix; overridden to ``False`` when
                ``params`` is a ``pd.Series`` in ``get_linear_predictor``.
            drop_1_for_FE: Whether to drop one level per fixed-effect group;
                overridden to ``False`` when ``params`` is a ``pd.Series``
                in ``get_linear_predictor``.
            **kwargs: Additional keyword arguments forwarded to
                ``build_model_from_formula``.

        Returns:
            New fitted model object.

        Raises:
            Exception: If the model was not built from a formula.
        """
        if self.from_formula:
            formula = self.strip_formula_to_just_exog_terms(self.formula, strip_non_exog)
            model = self.build_model_from_formula(formula, data, index=index, debug=debug,
                                                  check_constant_cols=check_constant_cols,
                                                  drop_1_for_FE=drop_1_for_FE, **kwargs)
            return model
        else:
            raise Exception("Can only rebuild a model from a formula!")

    def get_linear_predictor(self, params, data=None, debug=False, index=None,
                             ignore_column_mismatch=False,
                             check_constant_cols=False, drop_1_for_FE=False):
        """Compute the linear predictor ``X @ params`` for in-sample or new data.

        Dispatches on the type of ``data`` to select the appropriate design
        matrix, then on the type of ``params`` to handle Series/dict
        (column-aligned), array (positional), and None (use model ``exog``).

        - ``data=None``: uses the model's own ``exog`` directly.
        - ``data`` is a sparse/dense array: used directly; params must be
          positional.
        - ``data`` is a DataFrame/dict: calls ``build_model`` to construct
          the design matrix, aligning params by column name.

        When ``params`` is a ``pd.Series`` or dict, missing column names in
        the new design receive a coefficient of 0.

        Args:
            params: Coefficient vector or mapping.  Accepted types:
                ``pd.Series``, ``dict``, ``np.ndarray``, ``list``.
            data: New data; ``None``, sparse matrix, dense array, dict, or
                ``pd.DataFrame``.
            debug: If ``True``, enable verbose output during model building.
            index: Row-selector applied to ``data`` (DataFrame only).
            check_constant_cols: Forwarded to ``build_model`` when
                ``data`` is a DataFrame.
            drop_1_for_FE: Forwarded to ``build_model`` when ``data`` is
                a DataFrame.
            ignore_column_mismatch (bool): When ``True``, allow prediction when
                the design matrix built from ``data`` has **fewer columns** than
                the fitted coefficient vector (typical case: out-of-sample rows
                missing some fixed-effect levels so those dummy columns are
                absent). Extra coefficients not present in the new design are
                dropped; missing names in ``params`` still default to zero when
                aligning to ``column_names``. When ``False`` (default), raises
                if ``params`` contains names that do not appear in the new
                design.

        Returns:
            1-D numpy array of predicted values.

        Raises:
            Exception: On shape mismatches, unsupported types, or an
                ``index`` supplied with a non-DataFrame ``data``.
        """

        def _X_dot_beta(_X, _params, _is_sparse_data):
            """Internal method used for doing Xb calculation"""
            if _is_sparse_data:
                return _X.dot(csc_matrix(_params).reshape((-1, 1))).toarray().flatten()
            else:
                return _X.dot(params)

        if (data is None or isinstance(data, np.ndarray) or isspmatrix(data)) and index is not None:
            raise Exception("Can only supply an `index` if `data` is for building a new model!")

        # check data type
        is_sparse_data = True
        column_names = None
        has_column_data = True
        if data is None:
            X = self.exog
            column_names = self.exog_names
        elif isspmatrix(data):
            X = data
            has_column_data = False
        elif isinstance(data, np.ndarray):
            X = data
            is_sparse_data = False
            has_column_data = False
        elif isinstance(data, (DataFrame, SparseDataFrame, dict)):

            if isinstance(data, dict):
                data = DataFrame(data)
            # When params is already a named Series the caller is responsible
            # for alignment; skip constant-col checks and FE dummy-dropping so
            # the new design matrix columns match the Series index.
            check_constant_cols, drop_1_for_FE = (False, False) if isinstance(params, pd.Series) else (True, True)

            ret_val = self.formula_design_info.get_design_data_exog(data=data, index=index, debug=debug,
                                                                    check_constant_cols=check_constant_cols,
                                                                    drop_1_for_FE=drop_1_for_FE)
            X = ret_val[EXOG_KEY].values
            column_names = ret_val[EXOG_KEY].column_names
            # model = self.build_model(data=data, index=index, debug=debug, strip_non_exog=True,
            #                          check_constant_cols=check_constant_cols,
            #                          drop_1_for_FE=drop_1_for_FE)
            # X = model.exog
            # column_names = model.exog_names

        else:
            raise Exception(f"data type not valid: {type(data)}")

        # work on param types
        if isinstance(params, (pd.Series, dict)):

            if isinstance(params, dict):
                params_new = pd.Series(index=list(self.exog_names), data=[0.0] * len(self.exog_names))
                for k, v in params.items():
                    params_new[k] = v
                params = params_new

            if isinstance(params, pd.Series):
                params_new = pd.Series(index=list(self.exog_names), data=[0.0] * len(self.exog_names))
                params_new.loc[params.index] = params
                params = params_new

            if not has_column_data:
                if len(params) != X.shape[1]:
                    raise Exception(f"params has shape {params.shape} but X has shape {X.shape}")
                else:
                    return _X_dot_beta(X, params, is_sparse_data)

            else:
                # in this case we see column names of what was supplied
                if set(params.index) > set(column_names):
                    if ignore_column_mismatch:
                        params = params.loc[list(set(params.index) & set(column_names))]
                    else:
                        raise Exception(f"Supplied params has index {params.index} "
                                        f", but column names for data are {column_names}!")
                params_new = pd.Series(index=column_names, data=np.zeros(len(column_names)))
                params_new[params.index] = params
                return _X_dot_beta(X, params_new, is_sparse_data)

        elif isinstance(params, (np.ndarray, list)):
            params = np.asarray(params).flatten()
            if len(params) != X.shape[1]:
                raise Exception(f"params has shape {params.shape} but X has shape {X.shape}")
            else:
                return _X_dot_beta(X, params, is_sparse_data)

        else:
            raise Exception(f"Unsupported params type {type(params)}!")

    @abstractmethod
    def accepts_multi_outcome(self):
        """Return whether this model supports a multi-column response.

        Sub-classes must implement this method.  Return ``True`` if the
        model can handle a 2-D ``endog`` with more than one column (e.g. SURE
        or seemingly-unrelated systems), or ``False`` for single-outcome-only
        models.
        """
        pass

    def sparsity(self):
        """Return the fraction of non-zero elements in ``exog``.

        For sparse matrices this is ``nnz / (n_obs * n_cols)``; for dense
        arrays it always returns ``1.0``.

        Returns:
            float: Sparsity fraction in (0, 1].
        """
        if isspmatrix(self.exog):
            return self.exog.nnz / np.prod(self.exog.shape)
        else:
            return 1.0

    def cond(self, debug=False, p=None):
        """Compute the condition number of the weighted Gram matrix ``X'WX``.

        A large condition number indicates near-multicollinearity.

        Args:
            debug: If ``True``, print timing diagnostics for Gram matrix
                construction.
            p: Order of the norm passed to ``numpy.linalg.cond``; ``None``
                uses the 2-norm.

        Returns:
            float: Condition number.
        """
        XtX = gram_matrix(self.exog, weights=self.weights, debug=debug)
        return np.linalg.cond(XtX, p=p)

    def rank(self, debug=False, tol=None):
        """Compute the numerical rank of the weighted Gram matrix ``X'WX``.

        Args:
            debug: If ``True``, print timing diagnostics for Gram matrix
                construction.
            tol: Singular-value threshold passed to
                ``numpy.linalg.matrix_rank``; ``None`` uses the default
                machine-precision-based threshold.

        Returns:
            int: Numerical rank of ``X'WX``.
        """
        XtX = gram_matrix(self.exog, weights=self.weights, debug=debug)
        return np.linalg.matrix_rank(XtX, tol=tol, hermitian=True)
