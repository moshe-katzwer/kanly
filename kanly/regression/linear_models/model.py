"""Primary public API for kanly linear regression models.

``SparseLinearModel`` is the main entry point.  It supports:

- **Formula API** (``lm`` / ``lm_fast``): Patsy-style formula strings with
  automatic sparse matrix construction.
- **Matrix API** (``LM`` / ``LM_fast``): Raw endog/exog ndarray or CSC matrix
  inputs for maximum flexibility.
- **OLS / WLS / IV (2SLS)**: via the ``instruments`` or ``|`` formula syntax.
- **Absorbed fixed effects**: via ``absorb`` argument or ``C(...)`` absorb
  formula syntax.
- **FGLS**: iterative re-weighting (``do_fgls=True``).
- **Ridge regression**: via ``ridge_kwds={'alpha': ...}``.
- **SURE** (Seemingly Unrelated Regression): static ``sure(specifications)``
  method.
- **Multiple outcomes**: automatically detected from a multi-column endog.
- **Fast path** (``fit_lsmr`` / ``lm_fast``): no matrix inverse, LSMR solver.
- **Quadratic form / LLF** (``get_quadratic_form_and_llf``): for use with
  external optimisers and Bayesian samplers.

Typical formula syntax::

    y ~ x + z              # OLS
    y ~ x + z $ w          # WLS (w = weight variable)
    y ~ x + z | z1 + z2    # IV  (z1, z2 = excluded instruments)
    lm('y ~ x', df, absorb='grp')  # absorbed FE
"""

from __future__ import absolute_import, print_function

import time

import numpy as np
from pandas import Series
from scipy.sparse import csc_matrix, block_diag as sp_block_diag, vstack as sp_vstack, isspmatrix

from kanly.formula.sparse_formula_data_object import SparseFormulaDataObj
from kanly.bootstrap.bootstrap import do_bootstrap2, get_bootstrap_weights2, DEFAULT_BB_ALPHA, DEFAULT_BB_METHOD, \
    DEFAULT_BB_MAX_PROCESSES
from kanly.formula.data_getter import (
    SparseDataGetter, ENDOG_KEY, EXOG_KEY, WEIGHTS_KEY, INSTRUMENTS_KEY, ABSORB_KEY, HAS_IMPLICIT_CONSTANT_KEY,
    HAS_INTERCEPT_KEY, ABSORB_NAME_KEY, NULL_ROWS_INFO_DICT_KEY, VALID_OBS_ROWS_KEY,
    INSTRUMENT_REGRESSORS_KEY, ENDOG_REGRESSORS_KEY)
from kanly.formula.formula_design_info import FormulaDesignInfo
from kanly.formula.keys import FORMULA_DESIGN_INFO_KEY
from kanly.regression.cov_types import BOOTSTRAP, check_cov_kwds, format_cov_kwds, NONROBUST, OLS_SMALL
from kanly.regression.linear_model_base import LinearModelBase
from kanly.regression.linear_models.constants import (
    DEFAULT_LM_TEST_LEVEL, DEFAULT_LM_COV_TYPE, DEFAULT_LM_USE_T, DEFAULT_LM_BOOTSTRAP_N_SAMPLES,
    DEFAULT_LM_TEST_FORMULA_ON_DUMMY, DEFAULT_LM_FORCE_IV_PROJECTION, DEFAULT_LM_CHECK_CONST_COLS,
    DEFAULT_LM_SCALE_DESIGN_MATRIX, DEFAULT_LM_COMPUTE_EIGENVALUES, DEFAULT_LM_COMPUTE_EIGENVALUES_INSTRUMENTS,
    DEFAULT_LM_INVERSE_METHOD)
from kanly.regression.linear_models.fast_lm_internal import fit_lsmr_internal
from kanly.regression.linear_models.linear_model_2_quadratic_form import \
    _linear_model_components_2_quadratic_form_and_likelihood
from kanly.regression.linear_models.lm_internal import (
    lm_internal, loglike_internal, get_fit_summary_stats, fgls_internal, lin_mod_get_method,
    lin_mod_get_mean_exog_columns, LinearModelRegressionResultsRaw, )
from kanly.regression.linear_models.regression_results import AbsorbInfo2, InstrumentInfo, \
    SparseLinearRegressionResults, GLSARInfo
from kanly.regression.linear_models.variance_covariance2 import SparseVarianceCovariance2
from kanly.time_series.autoregression.estimate_ar import estimate_ar
from kanly.time_series.autoregression.glsar_helper import make_ar_full_information_W
from kanly.utils.linalg_utils import DEFAULT_DENSE_THRESHOLD_MB
from kanly.utils.util import dict_2_dataframe


class DuplicateDataException(Exception):
    """Raised when a ``data`` argument is supplied at both the SURE top level and inside individual specifications.

    SURE requires each specification to carry its own data dict, or the
    caller to supply a single ``data`` argument at the top level — but not
    both simultaneously.
    """
    pass


class SparseLinearModel(LinearModelBase):
    """OLS, WLS, IV, FGLS, ridge, SURE, multi-outcome, and fast-path linear models.

    ``SparseLinearModel`` extends :class:`~kanly.regression.linear_model_base.LinearModelBase`
    and is the central model class in the ``kanly.regression.linear_models``
    package.  It provides two entry points:

    **Formula API** (recommended)::

        from kanly.api import lm
        fit = lm('y ~ x + C(grp)', df, absorb='period')

    **Matrix API**::

        fit = SparseLinearModel.LM(y, X, has_constant=True)

    Estimation modes are selected via keyword arguments to :meth:`fit`:

    - ``cov_type='ols'`` / ``'hc1'`` / ``'cluster'`` / ``'hac'`` / etc.
    - ``do_fgls=True`` for iterative heteroscedasticity correction.
    - ``ridge_kwds={'alpha': ...}`` for L2-penalised estimation.

    The static :meth:`sure` method fits Seemingly Unrelated Regressions by
    building a block-diagonal design and running a joint estimation pass.

    Inherits ``is_sparse_model``, ``is_iv``, ``is_absorb``, ``is_weighted``,
    ``has_intercept``, ``has_implicit_constant``, ``exog_names``,
    ``exog_term_names``, ``exog_term_to_indices``, ``instrument_term_to_indices``,
    etc. from :class:`~kanly.regression.linear_model_base.LinearModelBase`.

    Examples
    --------
    Fit OLS via the formula entry point exposed on ``kanly.api``:

    >>> import numpy as np, pandas as pd
    >>> from kanly.api import lm
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> df = pd.DataFrame({'x': rng.normal(size=n),
    ...                    'grp': rng.integers(0, 5, n)})
    >>> df['y'] = 1.2 - 0.3 * df['x'] + 0.2 * rng.normal(size=n)
    >>> fit = lm('y ~ x + C(grp)', df)
    >>> print(fit.summary())                              # doctest: +SKIP
    ==========================================================================
    Linear Model Results
    ==========================================================================
    ...
    Intercept     1.21   ****  0.029   42.0  <0.001  ...
    x            -0.30   ****  0.014  -21.4  <0.001  ...
    ...

    Build (but don't fit) a model from a formula, then reuse the model object
    to try several covariance types without re-parsing:

    >>> from kanly.regression.linear_models.model import SparseLinearModel
    >>> model = SparseLinearModel.build_model_from_formula('y ~ x + C(grp)', df)
    >>> ols_fit  = model.fit(cov_type='ols')
    >>> hc1_fit  = model.fit(cov_type='hc1')

    Matrix-form (no formula parsing):

    >>> X = np.column_stack([np.ones(n), df['x'].values])
    >>> fit_mat = SparseLinearModel.LM(df['y'].values, X,
    ...                                exog_names=['Intercept', 'x'])

    Aliases on ``kanly.api`` for the formula entry point: ``reg``, ``ols``,
    ``wls`` (all point at :meth:`lm`); ``REG``, ``OLS``, ``WLS`` point at
    :meth:`LM`. See ``examples/regression/linear_models/`` for runnable
    scripts (OLS, WLS, IV, clustered SEs, bootstrap, absorbed FEs, ridge,
    SURE, multiple outcomes, fast LSMR path, etc.).
    """

    def __init__(self, endog, exog, add_constant=False, has_intercept=False, has_implicit_constant=False,
                 formula_design_info: FormulaDesignInfo = None,
                 weights=None, instruments=None,
                 endog_name=None,
                 exog_names=None,
                 weights_name=None, instrument_names=None,
                 specification_name=None, method=None, valid_obs_rows=None,
                 index=None, absorb=None, absorb_names=None,
                 absorb_term_name=None, cov_groups=None, cov_groups_name=None,
                 # ridge_kwds=None,
                 endog_regressors=None, null_rows_info_dict=dict(), model_elapsed=0, is_sure=False,
                 parent_model=None,
                 sigma=None, sigma_inv=None,
                 ):
        """Construct a SparseLinearModel from arrays and metadata.

        Typically called indirectly via :meth:`build_model_from_formula`,
        :meth:`lm`, or :meth:`LM`.  Direct instantiation is useful when
        arrays are already prepared and formula parsing overhead must be
        avoided.

        After calling the parent ``LinearModelBase.__init__``, this method:
        - Ensures ``endog`` is a CSC sparse column vector for sparse models
          or a flat dense array for dense models.
        - Computes weighted column means of ``exog`` for use in lift tests.
        - Stores GLS covariance matrices (``sigma``, ``sigma_inv``) and sets
          ``is_gls``.

        Args:
            endog (array-like or sparse): Dependent variable, shape ``(n,)``
                or ``(n, 1)``; for multi-outcome ``(n, k)``.
            exog (array-like or sparse): Design matrix, shape ``(n, p)``.
            add_constant (bool): If ``True``, prepend an intercept column to
                ``exog``.
            has_intercept (bool): True if ``exog`` already includes an explicit
                intercept column (e.g. ``Intercept`` from patsy).
            has_implicit_constant (bool): True if the model implicitly contains
                a constant (e.g. from a full set of dummy variables).
            formula (str or list of str, optional): Original formula string(s).
            from_formula (bool): True when the model was constructed via a
                formula (affects ``is_sparse_model`` default).
            weights (array-like, optional): Observation weights.
            instruments (array-like or sparse, optional): Instrument matrix.
            endog_name (str or list of str, optional): Dependent variable
                name(s).
            exog_names (list of str, optional): Regressor names.
            exog_term_names (list of str, optional): Patsy RHS term names
                (one per formula term; may be fewer than ``exog_names``).
            exog_term_to_indices (dict, optional): Maps each entry in
                ``exog_term_names`` to the 0-based column indices of that term
                in ``exog`` (e.g. ``{'C(g)': [3, 4, 5, 6], 'x': [1]}``).
                Set automatically when building from a formula via
                :meth:`build_model_from_formula`; pass explicitly for the
                matrix API when sparse_terms span multiple columns.
            weights_name (str, optional): Name of the weight variable.
            instrument_names (list of str, optional): Instrument column names.
            instrument_term_names (list of str, optional): Patsy term names for
                instruments.
            instrument_term_to_indices (dict, optional): Term → column-index
                map for ``instruments``, analogous to ``exog_term_to_indices``.
            data (pd.DataFrame or list, optional): Original data (retained for
                prediction).
            specification_name (str, optional): Label for this specification
                in output tables.
            method (str, optional): Method label (e.g. ``'OLS'``).
            valid_obs_rows (array-like, optional): Boolean mask of rows with
                valid (non-null) observations.
            index (array-like, optional): Row index for result alignment.
            absorb: Fixed-effects absorption specification.
            absorb_names (list of str, optional): Names of the absorbed
                variable columns.
            absorb_term_name (str, optional): Name of the absorb term.
            cov_groups (array-like, optional): Cluster labels.
            cov_groups_name (str, optional): Name of the cluster variable.
            endog_regressors (list of str, optional): Names of endogenous
                regressors (for IV models).
            null_rows_info_dict (dict): Metadata about null-row dropping.
            model_elapsed (float): Time spent building the model (seconds).
            is_sure (bool): True for SURE models.
            parent_model (SparseLinearModel, optional): For multi-outcome
                sub-models, the parent joint model.
            sigma (ndarray, optional): GLS error covariance matrix.
            sigma_inv (ndarray, optional): Pre-computed GLS covariance inverse.
        """

        if exog_names is None and add_constant:
            exog_names = {0: 'Intercept'}

        super().__init__(endog, exog, add_constant, has_intercept, has_implicit_constant, formula_design_info,
                         weights=weights,
                         instruments=instruments,
                         endog_name=endog_name, exog_names=exog_names,
                         weights_name=weights_name, instrument_names=instrument_names,
                         specification_name=specification_name, method=method, valid_obs_rows=valid_obs_rows,
                         index=index, absorb=absorb,
                         absorb_names=absorb_names, absorb_term_name=absorb_term_name,
                         cov_groups=cov_groups, cov_groups_name=cov_groups_name,
                         endog_regressors=endog_regressors,
                         null_rows_info_dict=null_rows_info_dict, model_elapsed=model_elapsed, is_sure=is_sure,
                         parent_model=parent_model)

        # Enforce sparse vs dense endog consistency.
        if self.is_sparse_model:
            if not isspmatrix(endog):
                endog = csc_matrix(self.endog).reshape((self.nobs, -1))
        else:
            if isspmatrix(endog):
                endog = endog.toarray().flatten()

        if not isspmatrix(endog):
            endog = np.asarray(endog)

        self.endog = endog

        self.wexog_instrumented_means, self.sum_weights = lin_mod_get_mean_exog_columns(
            self.is_weighted, weights, self.exog, debug=False, _time=None)

        # GLS mode is active whenever either covariance matrix is supplied.
        self.sigma = sigma
        self.sigma_inv = sigma_inv
        self.is_gls = self.sigma is not None or self.sigma_inv is not None
        assert self.sigma is None or self.sigma_inv is None

    @staticmethod
    def build_model_from_formula(formula, data, absorb=None, index=None, debug=False,
                                 check_constant_cols=DEFAULT_LM_CHECK_CONST_COLS, specification_name=None,
                                 fail_on_missing=False, cache_intermediate=True,
                                 sum_to_n=False, test_formula_on_dummy=DEFAULT_LM_TEST_FORMULA_ON_DUMMY,
                                 cov_groups=None, drop_1_for_FE=True, drop_endog=False, dense=False):
        """Parse a formula and data into a ``SparseLinearModel`` without fitting.

        Calls ``SparseDataGetter.get_data`` to build sparse matrices for
        endog, exog, instruments, weights, and absorb from the formula string,
        performs dimension checks, and instantiates a ``SparseLinearModel``.

        The resulting model object can be reused across multiple calls to
        :meth:`fit` (e.g. to test different covariance types) without
        re-parsing the formula.

        Args:
            formula (str): Patsy-style formula, e.g. ``'y ~ x + z | z1 $ w'``.
                - ``|`` separates instruments from exogenous regressors.
                - ``$`` separates the weight variable.
            data (pd.DataFrame or dict): Dataset.  Dicts are converted to
                DataFrames via ``dict_2_dataframe``.
            absorb (str or list of str, optional): Name(s) of categorical
                column(s) to absorb via within-group demeaning.
            index (array-like, optional): Row index for alignment.
            debug (bool): Print detailed parsing and dimension info.
            check_constant_cols (bool): Drop constant columns from the design
                matrix after formula evaluation.
            specification_name (str, optional): Label for output tables.
            fail_on_missing (bool): Raise if any formula variable has NaN rows.
            cache_intermediate (bool or dict): Whether to cache intermediate
                patsy term computations for repeated formula use.
            sum_to_n (bool): Normalise sparse weights to sum to n.
            test_formula_on_dummy (bool): Test formula on a dummy DataFrame
                first (cheap error checking).
            cov_groups (str, optional): Column name for cluster-robust SEs.
            drop_1_for_FE (bool): Drop one dummy per fixed-effect group to
                avoid perfect multicollinearity.
            dense (bool): If ``True``, convert all sparse matrices to dense
                before building the model.
            drop_endog (bool): Don't build LHS (only used for prediction)

        Returns:
            SparseLinearModel: Unfitted model object ready for :meth:`fit`.
            The model's ``exog_term_to_indices`` and
            ``instrument_term_to_indices`` attributes record which design
            columns belong to each parsed formula term (e.g. all dummy
            columns for ``C(grp)``).

        Examples
        --------
        Build a sparse OLS model once and fit it with two different
        covariance types without re-parsing the formula:

        >>> import numpy as np, pandas as pd
        >>> from kanly.regression.linear_models.model import SparseLinearModel
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({
        ...     'x':   rng.normal(size=500),
        ...     'grp': rng.integers(0, 10, size=500),
        ... })
        >>> df['y'] = 1.0 + 0.5 * df['x'] + rng.normal(size=500)
        >>> model = SparseLinearModel.build_model_from_formula(
        ...     'y ~ x + C(grp)', df, debug=False)
        >>> fit_ols     = model.fit(cov_type='ols')
        >>> fit_cluster = model.fit(cov_type='cluster',
        ...                         cov_kwds={'groups': 'grp'})

        IV model (instruments after ``|``), absorbed fixed effects, or
        weights (``$``) are all expressed in the formula:

        >>> # IV regression: x is endogenous, z1/z2 are excluded instruments
        >>> # model = SparseLinearModel.build_model_from_formula(
        >>> #     'y ~ x + w | z1 + z2', df)
        >>> # Absorbed FEs (one-way within transform):
        >>> # model = SparseLinearModel.build_model_from_formula(
        >>> #     'y ~ x', df, absorb='grp')
        """

        _time = time.time()

        data = dict_2_dataframe(data)

        # ------------------------
        # Parse the absorb keyword
        absorb = SparseLinearModel._get_absorb(absorb)

        # TODO maybe remove?
        assert isinstance(cache_intermediate, bool) or isinstance(cache_intermediate, dict)

        build_data_result = SparseDataGetter.get_data(
            data, formula, check_constant_cols=check_constant_cols, absorb=absorb, debug=debug, _time=_time,
            fail_on_missing=fail_on_missing, cache_intermediate=cache_intermediate, sum_to_n=sum_to_n,
            index=index, test_formula_on_dummy=test_formula_on_dummy,
            drop_1_for_FE=drop_1_for_FE
            # , cov_groups=cov_groups)
        )

        endog_obj: SparseFormulaDataObj = build_data_result[ENDOG_KEY]
        exog_obj: SparseFormulaDataObj = build_data_result[EXOG_KEY]
        instr_obj: SparseFormulaDataObj = build_data_result[INSTRUMENTS_KEY]
        absorb_obj: SparseFormulaDataObj = build_data_result[ABSORB_KEY]
        absorb_term_name: SparseFormulaDataObj = build_data_result[ABSORB_NAME_KEY]
        weights_obj: SparseFormulaDataObj = build_data_result[WEIGHTS_KEY]

        # cov_groups_obj = build_data_result[COV_GROUPS_KEY]
        # if cov_groups_obj is not None:
        #     cov_groups = cov_groups_obj.values
        # else:
        #     cov_groups = None
        # cov_groups_name = build_data_result[COV_GROUPS_NAME_KEY]

        endog_regressors = build_data_result[ENDOG_REGRESSORS_KEY]
        # instrument_regressors = build_data_result[INSTRUMENT_REGRESSORS_KEY] TODO drop?
        has_implicit_constant = build_data_result[HAS_IMPLICIT_CONSTANT_KEY]
        has_intercept = build_data_result[HAS_INTERCEPT_KEY]

        valid_obs_rows = build_data_result[VALID_OBS_ROWS_KEY]
        null_rows_info_dict = build_data_result[NULL_ROWS_INFO_DICT_KEY]

        if debug:
            print("\nAfter parsing inputs:")
            if specification_name is not None:
                print("\n\tspecification:    %s" % specification_name)
            print("\n\tformula:          %s" % formula)
            print("\n\tendog_name:       %s" % endog_obj.column_names[0])
            print("\n\texog_names:       %s" % exog_obj.column_names[:100]
                  + (f"... of {len(exog_obj.column_names)}" if len(exog_obj.column_names) > 100 else ''))
            if instr_obj is not None:
                print("\n\tinstrument_names: %s" % instr_obj.column_names[:100]
                      + (f"... of {len(instr_obj.column_names)}" if len(instr_obj.column_names) > 100 else ''))
            if weights_obj is not None:
                print("\n\tweights:          %s" % weights_obj.column_names[0])
            if absorb is not None:
                print("\n\tabsorb:           %s" % absorb)
            print()

        # ----------------
        # Check dimensions
        SparseLinearModel._check_dimensions(endog_obj, exog_obj, instr_obj, absorb_obj, weights_obj)

        if debug:

            def get_dims(x):
                """Return (nrows, ncols, nnz, sparsity_pct) for a sparse matrix."""
                return (x.shape[0], x.shape[1], x.nnz,
                        100.0 * x.nnz / (x.shape[0] * x.shape[1]))

            dim_string = "\texog matrix is (%d x %d) with %d non-zero entries (%.2f%%)" % get_dims(exog_obj.values)
            dim_string += ",\n\tendog matrix is (%d x %d) with %d non-zero entries (%.2f%%)" \
                          % get_dims(endog_obj.values)

            if instr_obj is not None:
                dim_string += ",\n\tinstrument matrix is (%d x %d) with %d non-zero entries (%.2f%%)" \
                              % get_dims(instr_obj.values)

            print("\n" + dim_string)
            print("\n...Sparse Formula complete! (%.3f s)\n\n" % (time.time() - _time))

        if weights_obj is not None:
            weights, weights_name = weights_obj.values, weights_obj.column_names[0]
        else:
            weights, weights_name = None, None

        if absorb_obj is not None:
            absorb, absorb_names = absorb_obj.values, absorb_obj.column_names
        else:
            absorb, absorb_names = None, None

        if instr_obj is not None:
            instruments, instrument_names, instrument_term_names, instrument_term_to_indices \
                = instr_obj.values, instr_obj.column_names, instr_obj.term_names, instr_obj.var_2_col_indices
        else:
            instruments, instrument_names, instrument_term_names, instrument_term_to_indices = None, None, None, None

        method = lin_mod_get_method(is_iv=instr_obj is not None, is_weighted=weights_obj is not None)

        cov_groups, cov_groups_name = LinearModelBase.get_covariance_groups_internal(
            exog_obj.values.shape[0], cov_groups, data=data, index=index,
            valid_obs_rows=valid_obs_rows, current_cov_groups=None, current_cov_groups_name=None)

        y = endog_obj.values
        X = exog_obj.values
        Z = instruments
        if dense:
            y = y.toarray().flatten()
            X = X.toarray()
            if Z is not None:
                Z = Z.toarray()

        formula_design_info = build_data_result[FORMULA_DESIGN_INFO_KEY]

        model = SparseLinearModel(
            y, X, False, has_intercept, has_implicit_constant, formula_design_info,
            weights=weights, instruments=Z,
            endog_name=endog_obj.column_names,
            exog_names=exog_obj.column_names,
            weights_name=weights_name, instrument_names=instrument_names,
            specification_name=specification_name, method=method, valid_obs_rows=valid_obs_rows,
            index=index,
            absorb=absorb, absorb_names=absorb_names, absorb_term_name=absorb_term_name,
            cov_groups=cov_groups, cov_groups_name=cov_groups_name,
            null_rows_info_dict=dict(), model_elapsed=time.time() - _time,
            endog_regressors=endog_regressors,
        )

        if debug:
            print(model)

        return model

    @staticmethod
    def _check_dimensions(endog_obj, exog_obj, instr_obj, absorb_obj, weights_obj):
        """Assert that all formula-parsed matrix objects have the same row count.

        Args:
            endog_obj: Parsed endog data object with a ``.values`` matrix.
            exog_obj: Parsed exog data object.
            instr_obj: Parsed instrument data object, or ``None``.
            absorb_obj: Parsed absorb data object, or ``None``.
            weights_obj: Parsed weights data object, or ``None``.

        Raises:
            AssertionError: If any non-None object has a different number of
                rows than ``endog_obj``.
        """
        assert endog_obj.values.shape[0] == exog_obj.values.shape[0]
        if instr_obj is not None:
            assert endog_obj.values.shape[0] == instr_obj.values.shape[0]
        if absorb_obj is not None:
            assert endog_obj.values.shape[0] == absorb_obj.values.shape[0]
        if weights_obj is not None:
            assert endog_obj.values.shape[0] == weights_obj.values.shape[0]

    @staticmethod
    def _get_absorb(absorb):
        """Normalise the ``absorb`` argument to a string or sorted list.

        Accepts a single string, a single-element iterable (unwrapped), or a
        multi-element iterable (sorted for reproducibility).

        Args:
            absorb (str, list of str, or None): Absorption specification.

        Returns:
            str, list of str, or None: Normalised form.

        Raises:
            Exception: If the input is not a string or iterable of strings.
        """
        if absorb is not None:
            if not isinstance(absorb, str):
                try:
                    if len(absorb) == 1:
                        return absorb[0]
                    else:
                        return sorted(list(absorb))
                except Exception:
                    raise Exception("`absorb` must be string or iterable of strings!")
            else:
                return absorb
        return None

    @staticmethod
    def lm(formula: str, data, absorb=None, index=None, debug=False, check_constant_cols=DEFAULT_LM_CHECK_CONST_COLS,
           specification_name=None, scale_design_matrix=DEFAULT_LM_SCALE_DESIGN_MATRIX,
           fail_on_missing=False, cache_intermediate=True, sum_to_n: bool = False,
           test_formula_on_dummy=DEFAULT_LM_TEST_FORMULA_ON_DUMMY, use_t=DEFAULT_LM_USE_T,
           cov_type=DEFAULT_LM_COV_TYPE, cov_kwds=None, test_level=DEFAULT_LM_TEST_LEVEL, compute_cov=True,
           keep_model=True, do_fgls: bool = False, fgls_kwds=None, force_iv_projection=DEFAULT_LM_FORCE_IV_PROJECTION,
           compute_eigenvalues=DEFAULT_LM_COMPUTE_EIGENVALUES,
           compute_eigenvalues_instruments=DEFAULT_LM_COMPUTE_EIGENVALUES_INSTRUMENTS,
           dense_threshold_mb=DEFAULT_DENSE_THRESHOLD_MB, inverse_method=DEFAULT_LM_INVERSE_METHOD,
           ridge_kwds=None, dense=False) -> SparseLinearRegressionResults:
        """One-shot formula API: parse, build, and fit a linear model.

        Convenience wrapper that calls :meth:`build_model_from_formula` and
        :meth:`fit` in sequence.  This is the primary entry point exposed
        by ``kanly.api.lm``.

        Formula syntax::

            'y ~ x + z'            # OLS
            'y ~ x + z $ w'        # WLS (w is the weight column)
            'y ~ x + z | z1 + z2'  # IV (z1, z2 are excluded instruments)
            'y ~ x + z | z1 $ w'   # IV + WLS

        Args:
            formula (str): Patsy-style formula string.
            data (pd.DataFrame or dict): Dataset.
            absorb (str or list of str, optional): Categorical variable(s) to
                absorb as fixed effects.
            index (array-like, optional): Row index for output alignment.
            debug (bool): Verbose parsing and estimation output.
            check_constant_cols (bool): Drop constant design-matrix columns.
            specification_name (str, optional): Label for result tables.
            scale_design_matrix (bool): Scale columns before solving.
            fail_on_missing (bool): Raise on NaN rows in formula variables.
            cache_intermediate (bool or dict): Cache patsy intermediates.
            sum_to_n (bool): Normalise sparse weights to sum to n.
            test_formula_on_dummy (bool): Pre-validate formula on a dummy DF.
            use_t (bool): t-distribution (``True``) vs normal.
            cov_type (str): Covariance type (e.g. ``'ols'``, ``'hc1'``,
                ``'cluster'``).
            cov_kwds (dict, optional): Covariance keyword arguments.
            test_level (float): Significance level for CIs.
            compute_cov (bool): Whether to compute the covariance matrix.
            keep_model (bool): Store the model object in results for
                prediction and recomputation.
            do_fgls (bool): Perform iterative FGLS.
            fgls_kwds (dict, optional): FGLS options (``'maxiter'``, ``'tol'``).
            force_iv_projection (bool): Project all columns through IV stage.
            compute_eigenvalues: ``None`` (auto), ``True``, or ``False``.
            compute_eigenvalues_instruments (bool): Compute instrument
                eigenvalues.
            dense_threshold_mb (float): Sparse/dense threshold.
            inverse_method: Matrix inversion algorithm selector.
            ridge_kwds (dict, optional): Ridge penalty options
                ``{'alpha': ..., 'normalize': bool, 'penalize_intercept': bool}``.
            dense (bool): Convert to dense arrays before fitting.

        Returns:
            SparseLinearRegressionResults: Fitted regression results including
                parameter estimates, standard errors, and fit statistics.

        Examples
        --------
        Simple OLS on a pandas DataFrame:

        >>> import numpy as np, pandas as pd
        >>> from kanly.api import lm
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({'x': rng.normal(size=100)})
        >>> df['y'] = 1.0 + 2.0 * df['x'] + rng.normal(size=100)
        >>> fit = lm('y ~ x', df)
        >>> round(float(fit.params['x']), 2)
        2.04
        >>> print(fit.summary())                              # doctest: +SKIP
        ==========================================================================
        Linear Model Results
        ==========================================================================
        ...

        Cluster-robust standard errors:

        >>> df['firm'] = rng.integers(0, 20, size=100)
        >>> fit = lm('y ~ x', df, cov_type='cluster',
        ...          cov_kwds={'groups': 'firm'})

        Two-way clustering (pass a tuple of column names):

        >>> df['period'] = rng.integers(0, 5, size=100)
        >>> fit = lm('y ~ x', df, cov_type='cluster',
        ...          cov_kwds={'groups': ('firm', 'period')})

        Heteroskedasticity-consistent (HC1) SEs:

        >>> fit_hc1 = lm('y ~ x', df, cov_type='HC1')

        Bayesian / classical / block bootstrap covariance (see
        ``kanly.bootstrap`` for details):

        >>> fit_bs = lm('y ~ x', df, cov_type='bootstrap',
        ...             cov_kwds={'n_samples': 200, 'method': 'bayesian'})

        Instrumental-variables regression (``|`` separates instruments) and
        absorbed fixed effects:

        >>> # fit_iv     = lm('y ~ x | z1 + z2', df)
        >>> # fit_absorb = lm('y ~ x', df, absorb='firm')

        Weighted least squares (``$`` introduces a weight variable in the
        formula), feasible GLS, and ridge:

        >>> df['w'] = np.abs(rng.normal(size=100)) + 0.1
        >>> fit_wls   = lm('y ~ x $ w', df)
        >>> fit_fgls  = lm('y ~ x', df, do_fgls=True,
        ...                fgls_kwds={'maxiter': 20, 'tol': 1e-8})
        >>> fit_ridge = lm('y ~ x', df, ridge_kwds={'alpha': 0.5})

        See Also
        --------
        :meth:`LM` : matrix-form entry point taking numpy/sparse arrays.
        :meth:`lm_fast` : coefficient-only LSMR fast path for huge designs.
        :meth:`sure` : seemingly unrelated regressions.
        Aliases on ``kanly.api``: ``reg``, ``ols``, ``wls``.
        """

        cov_kwds = format_cov_kwds(cov_kwds)
        if fgls_kwds is None:
            fgls_kwds = dict()

        model = SparseLinearModel.build_model_from_formula(
            formula, data, absorb=absorb, index=index, debug=debug,
            check_constant_cols=check_constant_cols,
            specification_name=specification_name, fail_on_missing=fail_on_missing,
            cache_intermediate=cache_intermediate, sum_to_n=sum_to_n, test_formula_on_dummy=test_formula_on_dummy,
            cov_groups=LinearModelBase.get_cov_group_keyword(cov_kwds),
            dense=dense)

        return model.fit(use_t=use_t, cov_type=cov_type, cov_kwds=cov_kwds, debug=debug, test_level=test_level,
                         compute_cov=compute_cov, keep_model=keep_model, specification_name=specification_name,
                         do_fgls=do_fgls, fgls_kwds=fgls_kwds, force_iv_projection=force_iv_projection,
                         scale_design_matrix=scale_design_matrix, compute_eigenvalues=compute_eigenvalues,
                         ridge_kwds=ridge_kwds, compute_eigenvalues_instruments=compute_eigenvalues_instruments,
                         dense_threshold_mb=dense_threshold_mb, inverse_method=inverse_method
                         )

    @staticmethod
    def LM(endog, exog, add_constant=False, has_constant: bool = True, weights=None, instruments=None,
           instrument_names=None,
           endog_name=None, exog_names=None, exog_term_names=None, weights_name=None,
           scale_design_matrix=DEFAULT_LM_SCALE_DESIGN_MATRIX,
           absorb=None, absorb_term_name=None, absorb_names=None, debug: bool = False,
           specification_name: [str, None] = None,
           invert_XpX=True, normalized_cov_params=None, instruments_normalized_cov_params=None, ridge_kwds=None,
           use_t=DEFAULT_LM_USE_T, cov_type=DEFAULT_LM_COV_TYPE, cov_kwds=None, test_level=DEFAULT_LM_TEST_LEVEL,
           compute_cov=True, keep_model=True, do_fgls=False, fgls_kwds=dict(),
           compute_eigenvalues_instruments=DEFAULT_LM_COMPUTE_EIGENVALUES_INSTRUMENTS,
           dense_threshold_mb=DEFAULT_DENSE_THRESHOLD_MB, inverse_method=DEFAULT_LM_INVERSE_METHOD,
           force_iv_projection=DEFAULT_LM_FORCE_IV_PROJECTION, compute_eigenvalues=DEFAULT_LM_COMPUTE_EIGENVALUES,
           sigma=None, sigma_inv=None,
           ) -> SparseLinearRegressionResults:
        """One-shot matrix API: build and fit from raw arrays.

        Accepts pre-built endog, exog, and optional arrays instead of a
        formula string.  Equivalent to calling :meth:`lm` when your data
        are already in matrix form and formula parsing overhead is undesirable.

        Args:
            endog (array-like or sparse, shape (n,) or (n,1)): Dependent
                variable.
            exog (array-like or sparse, shape (n, p)): Design matrix.
            add_constant (bool): Prepend an intercept column.
            has_constant (bool): True if the model already contains a
                constant.  Used for R² and F-stat computation.
            weights (array-like, optional): Observation weights, length n.
            instruments (array-like or sparse, optional): Instrument matrix,
                shape ``(n, q)``.
            instrument_names (list of str, optional): Instrument column names.
            endog_name (str, optional): Name for the dependent variable.
            exog_names (list of str, optional): Regressor names.
            exog_term_names (list of str, optional): Patsy term names.
            weights_name (str, optional): Weight variable name.
            scale_design_matrix (bool): Scale before solving.
            absorb: Fixed-effects absorption specification.
            absorb_term_name (str, optional): Absorb term label.
            absorb_names (list of str, optional): Absorb column names.
            debug (bool): Verbose output.
            specification_name (str, optional): Label for results.
            invert_XpX (bool): Unused parameter retained for API compatibility.
            normalized_cov_params: Unused; retained for API compatibility.
            instruments_normalized_cov_params: Unused; retained for API
                compatibility.
            ridge_kwds (dict, optional): Ridge penalty options.
            use_t, cov_type, cov_kwds, test_level, compute_cov, keep_model,
            do_fgls, fgls_kwds, compute_eigenvalues_instruments,
            dense_threshold_mb, inverse_method, force_iv_projection,
            compute_eigenvalues: Forwarded to :meth:`fit`.
            sigma (ndarray, optional): GLS covariance matrix.
            sigma_inv (ndarray, optional): Pre-computed GLS inverse.

        Returns:
            SparseLinearRegressionResults: Fitted regression results.

        Examples
        --------
        Use :meth:`LM` (uppercase) when you already have numpy arrays and
        want to skip the formula parser:

        >>> import numpy as np
        >>> from kanly.api import LM
        >>> rng = np.random.default_rng(0)
        >>> n = 200
        >>> X = np.column_stack([np.ones(n),                       # intercept
        ...                      rng.normal(size=n),               # x1
        ...                      rng.normal(size=n)])              # x2
        >>> beta_true = np.array([1.0, 2.0, -0.5])
        >>> y = X @ beta_true + rng.normal(size=n)
        >>> fit = LM(y, X, has_constant=True,
        ...          exog_names=['Intercept', 'x1', 'x2'])
        >>> [round(float(p), 2) for p in fit.params]              # doctest: +SKIP
        [1.04, 1.98, -0.47]

        Matrix-form IV: pass an instrument matrix ``Z`` for the endogenous
        columns of ``X``:

        >>> # fit_iv = LM(y, X, instruments=Z,
        >>> #             instrument_names=['z1', 'z2', ...])

        See Also
        --------
        :meth:`lm` : formula entry point taking a Patsy-style formula and DF.
        Aliases on ``kanly.api``: ``REG``, ``OLS``, ``WLS``.
        """

        model = SparseLinearModel(
            endog, exog, add_constant, has_constant, has_constant, None,
            weights=weights, instruments=instruments,
            endog_name=endog_name, exog_names=exog_names,
            weights_name=weights_name, instrument_names=instrument_names,
            method=lin_mod_get_method(instruments is not None, weights is not None, None),
            specification_name=specification_name, valid_obs_rows=None,
            index=None, absorb=absorb, absorb_names=absorb_names,
            absorb_term_name=absorb_term_name,
            sigma=sigma, sigma_inv=sigma_inv,
        )
        return model.fit(
            use_t=use_t, cov_type=cov_type, cov_kwds=cov_kwds, debug=debug, test_level=test_level,
            compute_cov=compute_cov, keep_model=keep_model, specification_name=specification_name,
            do_fgls=do_fgls, fgls_kwds=fgls_kwds, force_iv_projection=force_iv_projection,
            scale_design_matrix=scale_design_matrix, compute_eigenvalues=compute_eigenvalues,
            ridge_kwds=ridge_kwds, compute_eigenvalues_instruments=compute_eigenvalues_instruments,
            dense_threshold_mb=dense_threshold_mb, inverse_method=inverse_method
        )

    def overwrite_weights_sigma(self, weights, sigma, sigma_inv):
        """Resolve sentinel values for weights, sigma, and sigma_inv.

        When :meth:`fit` is called with default sentinel ``""`` values for
        these arguments, this method falls back to the values stored on the
        model object (set during construction).  Non-sentinel values are
        passed through unchanged.

        Args:
            weights: Observation weights, or ``""`` to use ``self.weights``.
            sigma: GLS covariance, or ``""`` to use ``self.sigma``.
            sigma_inv: GLS inverse, or ``""`` to use ``self.sigma_inv``.

        Returns:
            tuple: ``(weights, sigma, sigma_inv)`` with sentinels resolved.

        Raises:
            AssertionError: If both ``sigma`` and ``sigma_inv`` are non-None,
                or if ``weights`` is non-None alongside GLS arrays.
        """
        if isinstance(weights, str) and weights == "":
            weights = self.weights
        if isinstance(sigma, str) and sigma == "":
            sigma = self.sigma
        if isinstance(sigma_inv, str) and sigma_inv == "":
            sigma_inv = self.sigma_inv

        assert sigma_inv is None or sigma is None
        assert weights is None or (sigma_inv is None and sigma is None)

        if sigma is not None:
            assert isinstance(sigma, np.ndarray) or isspmatrix(sigma)
        if sigma_inv is not None:
            assert isinstance(sigma_inv, np.ndarray) or isspmatrix(sigma)

        return weights, sigma, sigma_inv

    def check_gls_valid(self, sigma, sigma_inv, do_fgls):
        """Validate that GLS configuration is compatible with the model.

        GLS (``sigma`` / ``sigma_inv`` supplied) has several restrictions:
        - The model must not be sparse (call ``to_dense()`` first).
        - The model must not include instruments or absorb.
        - ``do_fgls=True`` cannot be combined with GLS.

        Args:
            sigma (ndarray or None): GLS covariance matrix.
            sigma_inv (ndarray or None): Pre-computed GLS inverse.
            do_fgls (bool): True if FGLS re-weighting was requested.

        Raises:
            Exception: If any of the GLS restrictions are violated.
        """
        is_gls = sigma is not None or sigma_inv is not None
        if is_gls:
            if self.is_sparse_model:
                raise Exception(f"Cannot do GLS on a sparse model currently...use model.to_dense() first...")
            if self.instruments is not None:
                raise Exception(f"Cannot do GLS on a model with instruments...")
            if self.absorb is not None:
                raise Exception(f"Cannot do GLS on a model with absorb...")
            if do_fgls:
                raise Exception(f"Cannot do GLS and specify `do_fgls=True`!")

    def fit(self, use_t=True, cov_type=DEFAULT_LM_COV_TYPE, cov_kwds=None, debug=False,
            test_level=DEFAULT_LM_TEST_LEVEL, compute_cov=True, keep_model=True, specification_name=None, do_fgls=False,
            fgls_kwds=dict(), force_iv_projection=DEFAULT_LM_FORCE_IV_PROJECTION,
            dense_threshold_mb=DEFAULT_DENSE_THRESHOLD_MB,
            scale_design_matrix=DEFAULT_LM_SCALE_DESIGN_MATRIX, compute_eigenvalues=DEFAULT_LM_COMPUTE_EIGENVALUES,
            ridge_kwds=None, compute_eigenvalues_instruments=DEFAULT_LM_COMPUTE_EIGENVALUES_INSTRUMENTS,
            inverse_method=DEFAULT_LM_INVERSE_METHOD, weights="", sigma="", sigma_inv=""):
        """Estimate model parameters and compute inference statistics.

        This is the core estimation method.  It:

        1. Resolves sentinel weight/sigma values (:meth:`overwrite_weights_sigma`).
        2. Validates GLS constraints (:meth:`check_gls_valid`).
        3. Processes ridge penalty keyword dict (:meth:`_get_ridge_parameters`).
        4. Calls :func:`fgls_internal` (which calls :func:`lm_internal`) to
           estimate β, the normalised covariance, and raw residuals.
        5. Loops over outcomes (multi-outcome support) to compute summary
           statistics via :func:`get_fit_summary_stats`.
        6. Computes the parameter variance-covariance matrix via
           :class:`~kanly.regression.linear_models.variance_covariance2.SparseVarianceCovariance2`.
        7. When ``cov_type='bootstrap'``, runs
           :func:`~kanly.bootstrap.bootstrap.do_bootstrap2` and attaches
           bootstrapped SEs.
        8. Packages everything into :class:`~kanly.regression.linear_models.regression_results.SparseLinearRegressionResults`.

        Args:
            use_t (bool): Use t-distribution for inference.
            cov_type (str): Covariance estimator type.
            cov_kwds (dict, optional): Estimator-specific keywords.
            debug (bool): Verbose output.
            test_level (float): Significance level for CIs and tests.
            compute_cov (bool): Compute covariance matrix.
            keep_model (bool): Attach model to results for later prediction.
            specification_name (str, optional): Label for output tables.
            do_fgls (bool): Perform FGLS iteration.
            fgls_kwds (dict): FGLS options.
            force_iv_projection (bool): Force IV projection for all columns.
            dense_threshold_mb (float): Sparse/dense threshold.
            scale_design_matrix (bool): Column-scale before solving.
            compute_eigenvalues: ``None`` (auto), ``True``, or ``False``.
            ridge_kwds (dict, optional): Ridge penalty options.
            compute_eigenvalues_instruments (bool): Compute instrument
                eigenvalues.
            inverse_method: Inversion algorithm selector.
            weights (array-like or ""): Override instance weights.
            sigma (ndarray or ""): Override instance GLS covariance.
            sigma_inv (ndarray or ""): Override instance GLS inverse.

        Returns:
            SparseLinearRegressionResults or dict: A single result object
                for single-outcome models; a ``{y_name: result}`` dict for
                multi-outcome models.
        """

        _time = time.time()

        weights, sigma, sigma_inv = self.overwrite_weights_sigma(weights, sigma, sigma_inv)
        self.check_gls_valid(sigma, sigma_inv, do_fgls)

        compute_eigenvalues = self.default_compute_eigenvalues(compute_eigenvalues)

        cov_kwds = format_cov_kwds(cov_kwds)

        if cov_type is None:
            cov_type = DEFAULT_LM_COV_TYPE
        if cov_type is not None:
            cov_type = cov_type.replace('-', '_').replace(' ', '').upper()

        if sigma is not None or sigma_inv is not None:
            # only non-robust for GLS
            if cov_type not in {NONROBUST, OLS_SMALL}:
                raise Exception(f"For GLS, `cov_type` must be in {set([NONROBUST, OLS_SMALL])}, not {cov_type}!")

        check_cov_kwds(cov_type, cov_kwds)

        ridge_parameters = self._get_ridge_parameters(ridge_kwds, debug=debug, _time=_time)

        # ------------------------------------
        # Fit the parameters of the regression (FGLS or standard).
        result_fgls = fgls_internal(
            self.nobs, self.endog, self.exog, do_fgls=do_fgls, fgls_kwds=fgls_kwds, absorb=self.absorb,
            is_endog_regressor=self.is_endog_regressor, weights=weights, instruments=self.instruments,
            debug=debug, force_iv_projection=force_iv_projection, scale_design_matrix=scale_design_matrix,
            compute_eigenvalues=compute_eigenvalues, ridge_parameters=ridge_parameters,
            compute_eigenvalues_instruments=compute_eigenvalues_instruments, dense_threshold_mb=dense_threshold_mb,
            inverse_method=inverse_method, sigma=sigma, sigma_inv=sigma_inv
        )
        result, weights, fgls_info = result_fgls['result'], result_fgls['fgls_weights'], result_fgls['fgls_info']

        fgls_weights = weights if do_fgls else None
        weights_name = self.weights_name if not do_fgls else (
                ((self.weights_name + ":") if self.is_weighted else '') + f"FGLS[{fgls_info['n_iter']}]")

        if isspmatrix(result.params):
            params_sparse = result.params.copy()
            params = params_sparse.toarray()
        else:
            params = result.params.copy()
            if np.ndim(params) == 1:
                params = params.reshape((-1, 1))
            params_sparse = csc_matrix(result.params)
            if params_sparse.shape[0] == 1:
                params_sparse = params_sparse.transpose()

        exog_absorb_instrumented = result.exog_absorb_instrumented
        normalized_cov_params = result.normalized_cov_params
        condition_number = result.condition_number
        eigenvals = result.eigenvals

        # ----------------------------------------------------
        # Record info for instrument fits and absorbed FE fits
        if self.is_iv:
            instrument_info = InstrumentInfo(
                result.instrument_info.instrument_params,
                self.endog_regressors,
                set(self.exog_names) - set(self.endog_regressors),
                set(self.instrument_names) - set(self.exog_names),
                self.is_endog_regressor,
                result.instrument_info.instrument_normalized_cov_params
            )

        else:
            instrument_info = None

        if self.is_multi_outcome:
            y_names = self.endog_name
        else:
            y_names = [self.endog_name]

        fits = dict()
        for i, y_name in enumerate(y_names):
            if debug:
                print(f"\nComputing summary stats for outcome {y_name} ({i + 1}/{len(y_names)})")

            # -------------------------------------------
            # Get some key stats related to the model fit
            (df_resid, df_model, rsquared, rsquared_adj, wssr, ssr, wsst, sst, uncentered_tss, resid, wresid,
             fittedvalues, llf,
             absorbed_y_baselines, rsquared_within, rsquared_between) = get_fit_summary_stats(
                self.nobs, self.num_absorbed, params_sparse.getcol(i),
                self.endog.getcol(i) if self.is_multi_outcome else self.endog,
                self.exog,
                result.exog_absorb_instrumented,
                result.rsquared_within_raw[i] if self.is_multi_outcome else result.rsquared_within_raw,
                weights, do_fgls, self.is_weighted, self.has_implicit_constant,
                self.has_intercept, self.is_absorb,
                result.absorb_info.get_absorb_info_column_i(i) if self.is_absorb else None,
                sigma, sigma_inv,
                _time=_time, debug=debug)

            # ------------------
            # Record absorb info
            if self.is_absorb:
                absorb_info = AbsorbInfo2(self.absorb_term_name, self.num_absorbed, absorbed_y_baselines, None, None,
                                          rsquared_within,  # [i] if self.is_multi_outcome else rsquared_within,
                                          rsquared_between)
            else:
                absorb_info = None

            fit_elapsed = time.time() - _time

            # ------------------------------------------------------------------
            # Compute the variance-covariance of the parameters if not Bootstrap.
            # Bootstrap is handled after all outcomes are collected below.

            if compute_cov:
                self.set_covariance_groups(self.get_cov_group_keyword(cov_kwds))

            if compute_cov and BOOTSTRAP not in cov_type:

                cov_params, num_groups, df_t_dist, small_samp_correct, cov_elapsed \
                    = SparseVarianceCovariance2.compute_cov_params(
                    cov_type, cov_kwds, use_t, df_resid, wssr, resid,  # TODO WHICH RESID?
                    normalized_cov_params, self.is_sure, exog_absorb_instrumented,
                    groups=self.cov_groups,
                    debug=debug, _time=_time, weights=weights, param_name=f'{y_name} params',
                    sigma=sigma, sigma_inv=sigma_inv,
                )

                cov_string = SparseVarianceCovariance2.get_cov_string(self.nobs, cov_type, cov_kwds,
                                                                      self.cov_groups_name)

            else:
                df_t_dist = None
                small_samp_correct = None
                cov_elapsed = 0.0
                if BOOTSTRAP not in cov_type:
                    cov_type = 'NOT COMPUTED'
                cov_params = None
                cov_string = None

            if self.is_multi_outcome:
                fdi_clone: FormulaDesignInfo = self.formula_design_info.clone()
                fdi_clone.endog_terms = [y_name]
                fdi_clone.formula = f"{y_name} ~ {'~'.join(self.formula.split('~')[1:])}"
                model_copy_i = SparseLinearModel(
                    self.endog.getcol(i), self.exog, False, self.has_intercept, self.has_implicit_constant,
                    fdi_clone,
                    weights=self.weights, instruments=self.instruments,
                    endog_name=y_name, exog_names=self.exog_names,
                    weights_name=self.weights_name, instrument_names=self.instrument_names,
                    specification_name=self.specification_name,
                    method=self.method, valid_obs_rows=self.valid_obs_rows,
                    index=self.index, absorb=self.absorb, absorb_names=self.absorb_names,
                    absorb_term_name=self.absorb_term_name, cov_groups=self.cov_groups,
                    cov_groups_name=self.cov_groups_name,
                    endog_regressors=self.endog_regressors, null_rows_info_dict=dict(),
                    model_elapsed=self.model_elapsed, is_sure=self.is_sure, parent_model=self)
            else:
                model_copy_i = self

            fits[y_name] = SparseLinearRegressionResults(
                use_t, model_copy_i, self.nobs, df_resid, df_model, df_t_dist, params[:, i],
                cov_params, rsquared, rsquared_adj,
                ssr, wssr, sst, wsst, uncentered_tss, self.endog_name, self.exog_names, self.exog_term_names, llf,
                fit_elapsed,
                cov_elapsed,
                cov_type, cov_kwds.copy(), self.has_implicit_constant or self.has_intercept,
                self.has_implicit_constant,
                self.method if ridge_kwds is None else (('WTD ' if self.is_weighted else '') + 'RIDGE'),
                resid, wresid, fittedvalues,
                eigenvals, condition_number, normalized_cov_params,
                absorb_info=absorb_info, instrument_info=instrument_info,
                weights_name=weights_name, cov_string=cov_string, test_level=test_level,
                small_samp_correct=small_samp_correct, wexog_instrumented_means=self.wexog_instrumented_means,
                keep_model=keep_model, null_rows_info_dict=self.null_rows_info_dict.copy(),
                valid_obs_rows=self.valid_obs_rows.copy(), specification_name=specification_name,
                do_fgls=do_fgls, fgls_weights=fgls_weights if do_fgls else None,
                fgls_info=fgls_info if do_fgls else dict(),
                ridge_kwds=ridge_kwds
            )

        if BOOTSTRAP in cov_type and compute_cov:

            def param_estimation_func(bootstrap_weights):
                """Re-fit the model with a new set of bootstrap weights and return the raw result."""
                bootstrap_weights = get_bootstrap_weights2(bootstrap_weights, self.weights)
                result_idx = fgls_internal(
                    self.nobs, self.endog, self.exog, do_fgls=do_fgls, fgls_kwds=fgls_kwds, absorb=self.absorb,
                    is_endog_regressor=self.is_endog_regressor, weights=bootstrap_weights,
                    instruments=self.instruments, debug=False, force_iv_projection=force_iv_projection,
                    scale_design_matrix=scale_design_matrix,
                    compute_eigenvalues=False,
                    compute_eigenvalues_instruments=False,
                    dense_threshold_mb=dense_threshold_mb,
                    inverse_method=inverse_method, ridge_parameters=ridge_parameters
                )
                p = result_idx['result'].params
                if isspmatrix(p):
                    p = p.toarray()
                if self.is_multi_outcome:
                    return p
                else:
                    return p.ravel()

            do_bootstrap2(self.nobs, fits, param_estimation_func,
                          groups=self.cov_groups, group_name=self.cov_groups_name,
                          n_samples=cov_kwds.get('n_samples', DEFAULT_LM_BOOTSTRAP_N_SAMPLES),
                          seed=cov_kwds.get('seed', 0), debug=debug, method=cov_kwds.get('method', DEFAULT_BB_METHOD),
                          alpha=cov_kwds.get('alpha', DEFAULT_BB_ALPHA),
                          max_processes=cov_kwds.get('max_processes', DEFAULT_BB_MAX_PROCESSES),
                          use_correction=cov_kwds.get('use_correction', True),
                          test_level=test_level)

        if self.is_multi_outcome:
            return fits
        else:
            return list(fits.values())[0]

    def loglike(self, params):
        """Evaluate the Gaussian log-likelihood at arbitrary parameters.

        Computes residuals as ``y − X params`` and evaluates the Gaussian
        log-likelihood via :func:`loglike_internal`.

        Args:
            params (array-like, shape (p,)): Parameter vector to evaluate.

        Returns:
            float: Gaussian log-likelihood value.
        """
        if self.is_sparse_model:
            wresid = (self.endog - self.exog.dot(csc_matrix(params).reshape((-1, 1)))).toarray().flatten()
        else:
            wresid = (self.endog - self.exog.dot(params)).flatten()
        return loglike_internal(wresid, self.nobs, self.weights)

    @staticmethod
    def sure(specifications, debug=False, cache_intermediate=True, specification_name=None, data=None,
             cov_type=DEFAULT_LM_COV_TYPE, cov_kwds=None, compute_cov=True, test_level=DEFAULT_LM_TEST_LEVEL,
             use_t=DEFAULT_LM_USE_T, index=None, keep_model=True, inverse_method=DEFAULT_LM_INVERSE_METHOD,
             compute_eigenvalues=DEFAULT_LM_COMPUTE_EIGENVALUES,
             compute_eigenvalues_instruments=DEFAULT_LM_COMPUTE_EIGENVALUES_INSTRUMENTS,
             scale_design_matrix=DEFAULT_LM_SCALE_DESIGN_MATRIX):
        """Fit a Seemingly Unrelated Regressions (SURE) system.

        Builds a joint block-diagonal model from multiple individual
        specifications, estimates all equations jointly, and computes a
        shared covariance matrix over the stacked parameter vector.  This
        allows cross-equation Wald tests and efficiency gains when equations
        have correlated errors.

        .. note::
            Absorbed fixed effects (``absorb``) are **not yet supported** for
            SURE.  Bootstrap covariance is also not yet implemented.

        The block-diagonal construction:
        - ``endog``: vertical stack of individual endogs.
        - ``exog``: ``scipy.sparse.block_diag`` of individual exogs.
        - ``instruments`` (if any equation uses IV): block-diagonal of each
          equation's instruments (or exog for non-IV equations).

        Args:
            specifications (list of dict): One dict per equation.  Each must
                have ``'formula'`` and ``'data'`` keys (unless a top-level
                ``data`` is supplied).  Optional keys: ``'index'``,
                ``'absorb'`` (raises if non-None).
            debug (bool): Verbose output.
            cache_intermediate (bool or dict): Patsy term cache.
            specification_name (str, optional): Label for the joint result.
            data (pd.DataFrame, optional): Single shared dataset.  Cannot be
                combined with per-specification data.
            cov_type (str): Covariance estimator type.
            cov_kwds (dict, optional): Estimator keywords.
            compute_cov (bool): Compute covariance matrix.
            test_level (float): Significance level.
            use_t (bool): t-distribution vs normal.
            index (array-like, optional): Shared row index.  Cannot be
                combined with per-specification indices.
            keep_model (bool): Attach model to results.
            inverse_method: Inversion algorithm.
            compute_eigenvalues: ``None`` (auto), ``True``, or ``False``.
            compute_eigenvalues_instruments (bool): Instrument eigenvalues.
            scale_design_matrix (bool): Column-scale before solving.

        Returns:
            SparseLinearRegressionResults: Joint SURE result.

        Raises:
            DuplicateDataException: If ``data`` is supplied at the top level
                and also inside individual specifications.
            Exception: If ``absorb`` is non-None for any specification.
            NotImplementedError: If ``cov_type='bootstrap'`` is requested.

        Examples
        --------
        Two-equation SURE with a shared regressor ``x`` and one weighted
        equation, clustering both equations on ``user_id``:

        >>> import numpy as np, pandas as pd
        >>> from kanly.api import sure
        >>> rng = np.random.default_rng(0)
        >>> n = 500
        >>> df = pd.DataFrame({
        ...     'x':       rng.normal(size=n),
        ...     'user_id': np.arange(n),
        ...     'e':       rng.normal(size=n),
        ... })
        >>> df['wts1'] = np.exp(0.2 * rng.normal(size=n) + np.abs(df['x']))
        >>> df['y1'] = 1.2 - 0.3 * df['x'] + 0.2 * rng.normal(size=n) - 0.3 * df['e']
        >>> df['y2'] = 0.4 + 0.1 * df['x'] + 0.5 * rng.normal(size=n) + 0.2 * df['e']
        >>> fit = sure(
        ...     [
        ...         {'formula': 'y1 ~ x $ wts1', 'data': df,
        ...          'specification_name': 'y1'},
        ...         {'formula': 'y2 ~ x',        'data': df,
        ...          'specification_name': 'y2'},
        ...     ],
        ...     cov_type='cluster', cov_kwds={'groups': 'user_id'},
        ... )
        >>> print(fit)                                       # doctest: +SKIP
        ==========================================================================
        Linear Model Results
        ==========================================================================
        ...
        0_Intercept    1.218  ****   0.0208  58.57  <0.001  ...
        0_x          -0.3121  ****  0.01878 -16.62  <0.001  ...
        1_Intercept   0.3723  ****  0.02311  16.11  <0.001  ...
        1_x           0.1005  ****  0.02163   4.65  <0.001  ...
        """

        cov_type = cov_type.upper()
        cov_kwds = format_cov_kwds(cov_kwds)

        _time_model = time.time()

        specifications = [d.copy() for d in specifications]  # since we modify in method
        formulas = [s['formula'] for s in specifications]
        for s in specifications:
            s['index'] = s.get('index', None)

        if data is not None:
            for spec in specifications:
                if 'data' in spec.keys():
                    raise DuplicateDataException("Cannot specify overall `data` param and specification "
                                                 "specific data sets!")
                spec['data'] = data

        if index is not None:
            for spec in specifications:
                if 'index' in spec.keys():
                    raise DuplicateDataException("Cannot specify overall `index` param and specification "
                                                 "specific index!")
                spec['index'] = index

        term_dict = dict()
        models = []
        for spec in specifications:
            assert {'formula', 'data'} <= spec.keys()
            if spec.get('absorb', None) is not None:
                raise Exception("SURE does not support absorbing FEs yet!")
            spec['cache_intermediate'] = term_dict if cache_intermediate else None
            models.append(SparseLinearModel.build_model_from_formula(
                formula=spec['formula'], data=spec['data'],
                debug=debug, cache_intermediate=cache_intermediate,
                index=spec['index'],
            ))

        # Stack endog and build block-diagonal exog / instruments for the joint SURE model.
        endog = sp_vstack([m.endog for m in models])
        endog_names = [m.endog_name for m in models]

        def add_prefix(var_names, prefix):
            """Prepend *prefix* to every name in *var_names* to distinguish SURE equation columns."""
            return [prefix + v for v in var_names]

        exog_names = np.hstack([add_prefix(m.exog_names, f'{i}_') for i, m in enumerate(models)])
        wexog_instrumented_means = np.hstack([m.wexog_instrumented_means for m in models])
        exog_term_names = np.hstack([add_prefix(m.exog_term_names, f'{i}_') for i, m in enumerate(models)])
        exog = sp_block_diag([m.exog for m in models])

        is_weighted = np.any([m.is_weighted for m in models])
        if is_weighted:
            weights_list = [m.weights if m.is_weighted else np.ones(m.nobs) for m in models]
            weights = np.hstack(weights_list)
            weights_name = '<weights>'
        else:
            weights = None
            weights_name = None

        is_iv = np.any([m.is_iv for m in models])
        if is_iv:
            instrument_list = [m.instruments if m.is_iv else m.exog for m in models]
            instrument_names = np.hstack([add_prefix(m.instrument_names, f'{i}_')
                                          if m.is_iv
                                          else add_prefix(m.exog_names, f'{i}_')
                                          for i, m in enumerate(models)])
            instrument_term_names = np.hstack([add_prefix(m.instrument_term_names, f'{i}_')
                                               for i, m in enumerate(models)])
            instruments = sp_block_diag(instrument_list)
            endog_regressors = np.hstack(
                [add_prefix(m.endog_regressors, f'{i}_') for i, m in enumerate(models) if m.is_iv])
        else:
            instruments = None
            instrument_names = None
            instrument_term_names = None
            endog_regressors = None

        has_intercept = np.all([m.has_intercept for m in models])
        has_implicit_constant = np.all([m.has_implicit_constant or m.has_intercept for m in models])

        valid_obs_rows = [m.valid_obs_rows for m in models]
        # data = [m.formula_design_info.data for m in models]
        index = [m.index for m in models]

        model_elapsed = time.time() - _time_model

        model = SparseLinearModel(
            endog, exog, False, has_intercept, has_implicit_constant, models[0].formula_design_info,
            weights=weights, instruments=instruments,
            endog_name=endog_names, exog_names=exog_names,
            weights_name=weights_name, instrument_names=instrument_names,
            specification_name=specification_name, method='SURE', valid_obs_rows=valid_obs_rows,
            index=index, absorb=None, absorb_names=None, absorb_term_name=None,
            cov_groups=None, cov_groups_name=None,  # ridge_kwds=None,
            endog_regressors=endog_regressors, null_rows_info_dict=dict(), model_elapsed=model_elapsed, is_sure=True,
        )

        _time_fit = time.time()

        fits = [lm_internal(m.endog, m.exog, m.weights, m.instruments, None, m.is_endog_regressor, debug=debug,
                            scale_design_matrix=scale_design_matrix, inverse_method=inverse_method,
                            compute_eigenvalues_instruments=compute_eigenvalues_instruments,
                            compute_eigenvalues=compute_eigenvalues)
                for m in models]

        if compute_eigenvalues:
            eigenvals = np.hstack([f.eigenvals for f in fits])
            eigenvals = sorted(eigenvals)[::-1]
            condition_number = np.sqrt(eigenvals[0] / eigenvals[-1])
        else:
            eigenvals, condition_number = None, None

        normalized_cov_params = sp_block_diag([f.normalized_cov_params for f in fits])
        exog_absorb_instrumented = sp_block_diag([f.exog_absorb_instrumented for f in fits])
        params_sparse = sp_vstack([f.params for f in fits])
        params = params_sparse.toarray().flatten()
        instrument_params = [f.instrument_info.instrument_params if m.is_iv else np.eye(m.exog.shape[1])
                             for f, m in zip(fits, models)]

        # -------------------------------------------
        # Get some key stats related to the model fit
        (df_resid, df_model, rsquared, rsquared_adj, wssr, ssr, wsst, sst, uncentered_tss, resid, wresid, fittedvalues,
         llf, absorbed_y_baselines, rsquared_within, rsquared_between) = get_fit_summary_stats(
            model.nobs, 0, params_sparse, model.endog, model.exog, exog_absorb_instrumented,
            np.nan, weights, False, model.is_weighted, model.has_implicit_constant,
            model.has_intercept, False, None,
            sigma_inv=None, sigma=None,
            _time=_time_fit, debug=debug)

        # ----------------------------------------------------
        # Record info for instrument fits and absorbed FE fits
        if is_iv:
            instrument_info = InstrumentInfo(instrument_params,
                                             endog_regressors,
                                             set(exog_names) - set(endog_regressors),
                                             set(instrument_names) - set(exog_names),
                                             model.is_endog_regressor)
        else:
            instrument_info = None

        fit_elapsed = time.time() - _time_fit

        cluster_name = LinearModelBase.get_cov_group_keyword(cov_kwds)
        if cluster_name is not None:
            cov_groups_list = [m.get_covariance_groups(cluster_name)[0] for m in models]
            cov_groups = np.hstack(cov_groups_list)
        else:
            cov_groups = None

        if compute_cov and BOOTSTRAP not in cov_type:

            cov_params, num_groups, df_t_dist, small_samp_correct, cov_elapsed \
                = SparseVarianceCovariance2.compute_cov_params(
                cov_type, cov_kwds, use_t, df_resid, wssr, resid,  # TODO WHICH RESID?
                normalized_cov_params, True, exog_absorb_instrumented,
                groups=cov_groups, debug=debug, _time=None, weights=weights)
            cov_string = SparseVarianceCovariance2.get_cov_string(model.nobs, cov_type, cov_kwds, cluster_name)

        else:
            df_t_dist = None
            small_samp_correct = None
            cov_elapsed = 0.0
            if BOOTSTRAP not in cov_type:
                cov_type = 'NOT COMPUTED'
            cov_params = None
            cov_string = None

        fit = SparseLinearRegressionResults(
            use_t, model, model.nobs, df_resid, df_model, df_t_dist, params, cov_params, rsquared, rsquared_adj,
            ssr, wssr, sst, wsst, uncentered_tss, endog_names, exog_names, exog_term_names, llf, fit_elapsed, cov_elapsed,
            cov_type, cov_kwds.copy(), has_intercept, has_implicit_constant, model.method, resid, wresid, fittedvalues,
            eigenvals, condition_number, normalized_cov_params,
            absorb_info=None, instrument_info=instrument_info,
            weights_name=weights_name, cov_string=cov_string, test_level=test_level,
            small_samp_correct=small_samp_correct, wexog_instrumented_means=wexog_instrumented_means,
            keep_model=keep_model, null_rows_info_dict=dict(), valid_obs_rows=model.valid_obs_rows,
            specification_name=specification_name, do_fgls=False, fgls_weights=None, fgls_info=dict(),
        )

        if compute_cov and BOOTSTRAP in cov_type:
            raise NotImplementedError('Bootstrap not yet available for SURE')

        return fit

    def predict(self, params, data=None, index=None, debug=False, ignore_column_mismatch=False, *args, **kwargs):
        """Generate in-sample or out-of-sample predictions.

        Delegates to :meth:`~kanly.regression.linear_model_base.LinearModelBase.get_linear_predictor`,
        which handles both array and DataFrame inputs with optional formula
        re-evaluation.

        Args:
            params (array-like): Coefficient vector to use for prediction.
            data (pd.DataFrame, optional): Out-of-sample data.  If ``None``,
                returns in-sample predictions using ``self.exog``.
            index (array-like, optional): Row index for output alignment.
            debug (bool): Verbose output.
            ignore_column_mismatch (bool): When ``True``, allow prediction when
                the new design matrix has fewer columns than ``params`` (e.g.
                prediction data missing some fixed-effect dummies). See
                :meth:`~kanly.regression.linear_model_base.LinearModelBase.get_linear_predictor`.
            *args, **kwargs: Unused; accepted for interface compatibility.

        Returns:
            ndarray: Predicted values.
        """
        return self.get_linear_predictor(params, data=data, index=index, debug=debug,
                                         ignore_column_mismatch=ignore_column_mismatch)

    def build_model(self, data, index=None, debug=False, strip_non_exog=False, check_constant_cols=True,
                    drop_endog=False, drop_1_for_FE=True):
        """Build a new model from out-of-sample data using this model's formula.

        Passes the stored ``absorb_term_name`` through to the parent
        :meth:`~kanly.regression.linear_model_base.LinearModelBase.build_model`
        so that out-of-sample prediction data is handled consistently.

        Args:
            data (pd.DataFrame): New data conforming to the original formula.
            index (array-like, optional): Row index.
            debug (bool): Verbose output.
            strip_non_exog (bool): Remove non-exog columns after building.
            check_constant_cols (bool): Drop constant columns.
            drop_1_for_FE (bool): Drop one FE dummy per group.

        Returns:
            SparseLinearModel: New model built on ``data``.
        """
        return super().build_model(data, index=index, debug=debug, absorb=self.absorb_term_name,
                                   strip_non_exog=strip_non_exog, check_constant_cols=check_constant_cols,
                                   drop_endog=drop_endog,
                                   drop_1_for_FE=drop_1_for_FE)

    @staticmethod
    def LM_fast(endog, exog, instruments=None, weights=None, bootstrap_weights=None, endog_name=None, exog_names=None,
                weights_name=None,
                debug=False, specification_name=None, **kwargs):
        """One-shot matrix API via the fast LSMR path (no matrix inverse).

        Wrapper around :func:`~kanly.regression.linear_models.fast_lm_internal.fit_lsmr_internal`
        that adds metadata and parameter Series to the result dict.

        .. note::
            Does not compute standard errors, covariance, or R².
            Use :meth:`LM` for full inference.

        Args:
            endog (array-like or sparse): Dependent variable.
            exog (array-like or sparse): Design matrix.
            instruments (None): Reserved; raises if not ``None``.
            weights (array-like, optional): Observation weights.
            bootstrap_weights (array-like, optional): Bootstrap weights.
            endog_name (str, optional): Dependent variable name.
            exog_names (list of str, optional): Regressor names.
            weights_name (str, optional): Weight variable name.
            debug (bool): Verbose output.
            specification_name (str, optional): Label for the result.
            **kwargs: Forwarded to ``lsmr`` (e.g. ``damp``, ``maxiter``).

        Returns:
            dict: LSMR result dict plus ``'endog_name'``, ``'exog_names'``,
                ``'weights_name'``, ``'params'`` (pandas Series), and
                ``'specification_name'``.

        Examples
        --------
        Coefficient-only matrix-form fit on a very wide sparse design:

        >>> import numpy as np
        >>> from scipy.sparse import random as sp_random
        >>> from kanly.api import LM_fast
        >>> rng = np.random.default_rng(0)
        >>> n, p = 100_000, 5_000
        >>> X = sp_random(n, p, density=0.001, format='csr',
        ...               random_state=rng).tocsc()             # doctest: +SKIP
        >>> beta = rng.normal(size=p) * (rng.uniform(size=p) < 0.05)
        >>> y = X @ beta + rng.normal(size=n)                   # doctest: +SKIP
        >>> result = LM_fast(y, X,
        ...                  exog_names=[f'x{j}' for j in range(p)])  # doctest: +SKIP
        >>> result['params'].head()                              # doctest: +SKIP
        """

        result = fit_lsmr_internal(endog, exog, instruments, weights, bootstrap_weights, **kwargs)

        if exog_names is None:
            exog_names = ['<x%d>' % d for d in range(exog.shape[1])]
        if endog_name is None:
            endog_name = '<y>'
        if weights is not None:
            if weights_name is None:
                weights_name = '<weights>'
        else:
            weights_name = None

        result.update({
            'endog_name': endog_name, 'exog_names': exog_names, 'weights_name': weights_name,
            'params': Series(index=exog_names, data=result['x'].copy()),
            'specification_name': specification_name
        })

        return result

    @staticmethod
    def lm_fast(formula, data, index=None, debug=False, check_constant_cols=DEFAULT_LM_CHECK_CONST_COLS,
                specification_name=None, fail_on_missing=False, cache_intermediate=True, sum_to_n=False,
                test_formula_on_dummy=DEFAULT_LM_TEST_FORMULA_ON_DUMMY,
                keep_model=True, dense=False, **kwargs):
        """One-shot formula API via the fast LSMR path.

        Parses the formula, builds a ``SparseLinearModel``, and calls
        :meth:`fit_lsmr`.  No matrix inverse is computed.

        Args:
            formula (str): Patsy-style formula string.
            data (pd.DataFrame or dict): Dataset.
            index (array-like, optional): Row index.
            debug (bool): Verbose output.
            check_constant_cols (bool): Drop constant design-matrix columns.
            specification_name (str, optional): Label for results.
            fail_on_missing (bool): Raise on NaN rows.
            cache_intermediate (bool or dict): Cache patsy intermediates.
            sum_to_n (bool): Normalise sparse weights to sum to n.
            test_formula_on_dummy (bool): Pre-validate formula.
            keep_model (bool): Attach model to results.
            dense (bool): Convert to dense before fitting.
            **kwargs: Forwarded to ``lsmr``.

        Returns:
            SparseLinearRegressionResults: Results with parameters but no
                standard errors or covariance matrix.

        Examples
        --------
        Fit a wide formula on a 2M-row dataset, skipping covariance
        computation for speed:

        >>> import numpy as np, pandas as pd
        >>> from kanly.api import lm, lm_fast
        >>> rng = np.random.default_rng(0)
        >>> n = 2_000_000                                        # doctest: +SKIP
        >>> x = rng.normal(size=n)                               # doctest: +SKIP
        >>> df = pd.DataFrame({                                  # doctest: +SKIP
        ...     'x': x,
        ...     'z': rng.normal(size=n),
        ...     'g': rng.integers(0, 1_500, n),
        ...     'y': 10 + x * 1.2 + rng.normal(size=n),
        ... })
        >>> fit_fast = lm_fast(                                  # doctest: +SKIP
        ...     'y ~ x*C(g) + I(x**2) + poly(z, 3)', df)
        >>> fit_fast.params.head()                               # doctest: +SKIP
        """

        model = SparseLinearModel.build_model_from_formula(
            formula, data, index=index, debug=debug,
            check_constant_cols=check_constant_cols,
            specification_name=specification_name, fail_on_missing=fail_on_missing,
            cache_intermediate=cache_intermediate, sum_to_n=sum_to_n, test_formula_on_dummy=test_formula_on_dummy,
            dense=dense
        )

        result = model.fit_lsmr(keep_model=keep_model, specification_name=specification_name,
                                **kwargs)

        return result

    def fit_lsmr(self, bootstrap_weights=None, keep_model=True, specification_name=None, **kwargs):
        """Estimate parameters using the LSMR iterative solver.

        Calls :func:`~kanly.regression.linear_models.fast_lm_internal.fit_lsmr_internal`
        on the model's stored arrays, then computes summary statistics
        (R², SSR, LLF, df) and packages the result into a
        :class:`~kanly.regression.linear_models.regression_results.SparseLinearRegressionResults`
        object without a covariance matrix.

        Note: no absorb or IV support in the LSMR path.

        Args:
            bootstrap_weights (array-like, optional): Bootstrap resample
                weights to overlay on the standard observation weights.
            keep_model (bool): Attach the model to results for prediction.
            specification_name (str, optional): Label for the result.
            **kwargs: Forwarded to ``lsmr`` (e.g. ``damp``, ``maxiter``).

        Returns:
            SparseLinearRegressionResults: Results with parameters and fit
                statistics but without standard errors or covariance.
        """

        _time = time.time()

        result = fit_lsmr_internal(self.endog, self.exog, self.instruments, self.weights, bootstrap_weights,
                                   self.is_endog_regressor, **kwargs)
        result.update({
            'endog_name': self.endog_name, 'exog_names': self.exog_names, 'weights_name': self.weights_name,
            'params': Series(index=self.exog_names, data=result['x'].copy())
        })

        eigenvals, condition_number = None, result['conda']
        params_sparse = csc_matrix(result['params']).reshape((-1, 1))

        (
            df_resid, df_model, rsquared, rsquared_adj, wssr, ssr, wsst, sst, uncentered_tss,
            resid, wresid, fittedvalues, llf,
            absorbed_y_baselines, rsquared_within, rsquared_between
        ) = get_fit_summary_stats(
            self.nobs, self.num_absorbed, params_sparse, self.endog, self.exog,
            self.exog,  # result.exog_absorb_instrumented,
            None,  # result.rsquared_within_raw,
            self.weights, False, self.is_weighted, self.has_implicit_constant,
            self.has_intercept, self.is_absorb,
            None,  # result.absorb_info,
            sigma_inv=None, sigma=None,
            _time=None, debug=False)

        df_t_dist = df_resid

        fit_elapsed = time.time() - _time

        """
        use_t, model, nobs, df_resid, df_model, df_t_dist, params, cov_params, rsquared, rsquared_adj,
                 ssr, wssr, sst, wsst, uncentered_tss, endog_name, exog_names, exog_term_names, llf, fit_elapsed, cov_elapsed,
                 cov_type, cov_kwds, has_const, has_implicit_constant, method, resid, wresid, fittedvalues,
                 eigenvals, condition_number, normalized_cov_params,
                 absorb_info=None, instrument_info=None,
                 weights_name=None, cov_string=None, test_level=DEFAULT_LM_TEST_LEVEL,
                 small_samp_correct=None, wexog_instrumented_means=None, keep_model=True,
                 null_rows_info_dict=None, valid_obs_rows=None, specification_name=None,
                 do_fgls=False, fgls_weights=None, fgls_info=dict(), ridge_kwds=None,
        """

        results = SparseLinearRegressionResults(use_t=True, model=self, nobs=self.nobs, df_resid=df_resid,
                                                df_model=df_model, df_t_dist=df_t_dist, params=result['params'],
                                                cov_params=None, rsquared=rsquared, rsquared_adj=rsquared_adj, ssr=ssr,
                                                wssr=wssr, sst=sst, wsst=wsst, uncentered_tss=uncentered_tss,
                                                endog_name=self.endog_name,
                                                exog_names=self.exog_names, exog_term_names=self.exog_term_names,
                                                llf=llf, fit_elapsed=fit_elapsed, cov_elapsed=0, cov_type=None,
                                                has_const=self.has_implicit_constant or self.has_intercept,
                                                has_implicit_constant=self.has_implicit_constant, method=self.method,
                                                resid=resid, wresid=wresid, fittedvalues=fittedvalues,
                                                eigenvals=eigenvals, condition_number=condition_number,
                                                normalized_cov_params=None, absorb_info=None, instrument_info=None,
                                                weights_name=self.weights_name, cov_string=None,
                                                cov_kwds=None,
                                                test_level=.05,
                                                small_samp_correct=1.,
                                                wexog_instrumented_means=self.wexog_instrumented_means,
                                                keep_model=keep_model,
                                                null_rows_info_dict=self.null_rows_info_dict.copy(),
                                                valid_obs_rows=self.valid_obs_rows.copy(),
                                                specification_name=specification_name, do_fgls=None, fgls_weights=None,
                                                fgls_info=None)
        return results

    @staticmethod
    def quadratic_form_and_llf_from_formula(formula, data, index=None, debug=False):
        """Parse a formula and return the quadratic form and log-likelihood callable.

        Convenience wrapper that builds a model from the formula and calls
        :meth:`get_quadratic_form_and_llf`.

        Args:
            formula (str): Patsy-style formula string.
            data (pd.DataFrame or dict): Dataset.
            index (array-like, optional): Row index.
            debug (bool): Verbose output.

        Returns:
            tuple: ``(log_likelihood_func, ssr_quad_form)`` — see
                :meth:`get_quadratic_form_and_llf`.
        """
        model = SparseLinearModel.build_model_from_formula(formula, data, index=index, debug=debug)
        return model.get_quadratic_form_and_llf()

    def get_quadratic_form_and_llf(self):
        """Return the SSR quadratic form and Gaussian log-likelihood callable.

        Delegates to
        :func:`~kanly.regression.linear_models.linear_model_2_quadratic_form._linear_model_components_2_quadratic_form_and_likelihood`.

        .. note::
            The log-likelihood signature is ``llf(beta, sigma)`` where
            ``sigma`` is the **standard deviation** (not the variance).

        Returns:
            tuple:
                - **log_likelihood_func** (callable): ``llf(beta, sigma)``.
                - **ssr_quad_form** (QuadraticForm): Callable SSR quadratic
                  form object.

        Raises:
            Exception: If the model uses instrumental variables.
        """
        if self.is_iv:
            raise Exception
        llf, quad_form = _linear_model_components_2_quadratic_form_and_likelihood(self.endog, self.exog, self.weights)
        return llf, quad_form

    def accepts_multi_outcome(self):
        """Return True: SparseLinearModel supports multi-column endog.

        When ``endog`` has multiple columns, :meth:`fit` loops over each
        outcome, computing separate summary statistics and result objects
        while sharing the joint parameter estimation pass.

        Returns:
            bool: Always ``True``.
        """
        return True

    # def bayesian_linear_regression_TO_DELETE(self, beta_0, Ainv_0, nu_0, tau2_0):
    #     """
    #     prior on (beta | sigma^2) is Normal(beta0, sigma^2 * Ainv_0)
    #     prior on sigma^2 is InvGamma(nu0/2, tau2_0 * nu0/2)
    #     """
    #
    #     if self.is_absorb or self.is_iv:
    #         raise Exception
    #
    #     from scipy.stats import invgamma, multivariate_normal, t as t_dist
    #
    #     beta_0 = np.asarray(beta_0)
    #     Ainv_0 = np.asarray(Ainv_0)
    #
    #     n = self.nobs
    #     k = self.exog.shape[1]
    #     XtX = self.exog.transpose().dot(self.exog).toarray()
    #     Xty = self.exog.transpose().dot(self.endog).toarray().flatten()
    #     yty = self.endog.power(2).sum()
    #
    #     A = np.linalg.inv(Ainv_0)
    #     XtX_plus_A = XtX + A
    #
    #     An_inv = np.linalg.inv(XtX_plus_A)
    #
    #     beta_hat = (An_inv @ (Xty + A @ beta_0)).flatten()
    #     nu_n = nu_0 + n
    #     tau2_n = (nu_0 * tau2_0 + yty + beta_0 @ A @ beta_0 - beta_hat @ XtX_plus_A @ beta_hat) / nu_n
    #
    #     sigma2_hat = tau2_n * nu_n / (nu_n - 2)
    #
    #     posterior_sigma2 = invgamma(nu_n / 2, 0, tau2_n * nu_n / 2)
    #     posterior_pdf = lambda theta: posterior_sigma2.pdf(theta[-1]) * multivariate_normal.pdf(
    #         theta[:-1], beta_hat, theta[-1] * An_inv)
    #
    #     marginal_pdf = {
    #         name: t_dist(n - k, _b, _s)
    #         for name, _b, _s in zip(self.exog_names, beta_hat, np.sqrt(sigma2_hat * np.diag(An_inv)))
    #     }
    #     marginal_pdf['sigma2'] = posterior_sigma2
    #
    #     return BayesianLinearRegressionResults()
    #

    def _get_ridge_parameters(self, ridge_kwds, debug=False, _time=None):
        """Convert ``ridge_kwds`` to a per-column ridge penalty array.

        Validates ``ridge_kwds``, scales the alpha value(s) to match the
        model's ``nobs`` (or sum of weights for WLS), and optionally
        normalises by the column L2 norms of the design matrix.

        Args:
            ridge_kwds (dict or None): Ridge options:
                - ``'alpha'`` (float, list, or dict): Penalty value(s).
                  A float applies the same penalty to all columns; a list
                  applies per-column; a dict maps variable names to penalties.
                - ``'normalize'`` (bool, default True): If True, scale alpha
                  by the column L2 norm of X so the penalty is unit-invariant.
                - ``'penalize_intercept'`` (bool, default False): If False,
                  zero out the penalty for the intercept column.
            debug (bool): Print timing message.
            _time (float, optional): Elapsed-time baseline.

        Returns:
            ndarray or None: Per-column penalty array of length ``p``, or
                ``None`` when ``ridge_kwds`` is ``None``.

        Raises:
            Exception: If ``ridge_kwds`` does not contain ``'alpha'``, or
                contains unrecognised keys.
            AssertionError: If any penalty is negative.
        """
        if _time is None:
            _time = time.time()

        if ridge_kwds is None:
            return None

        else:

            if 'alpha' not in ridge_kwds:
                raise Exception("`ridge_kwds` must specify alpha penalty!")
            if not set(ridge_kwds.keys()) <= {'alpha', 'normalize', 'penalize_intercept'}:
                raise Exception("Acceptable `ridge_kwds` keys are `alpha`, `normalize`, and `penalize_intercept`!")

            # if self.is_weighted:
            #     self.weights = self.weights * (self.nobs / self.weights.sum())

            if debug:
                print('Processing ridge parameters....', end='')

            if isinstance(ridge_kwds['alpha'], (float, int)):
                ridge_parameters = np.array([ridge_kwds['alpha']] * self.exog.shape[1]).astype(float)
            elif isinstance(ridge_kwds['alpha'], dict):
                ridge_parameters = np.array([ridge_kwds['alpha'].get(e, 0.0) for e in self.exog_names])
            else:
                ridge_parameters = np.array(ridge_kwds['alpha']).astype(float)

            if self.is_weighted and not ridge_kwds.get('normalize', True):
                ridge_parameters *= self.weights.sum()
            else:
                ridge_parameters *= self.nobs

            assert np.all(ridge_parameters >= 0)

            if 'Intercept' in self.exog_names:
                if self.exog_names[0] != 'Intercept':
                    raise Exception('Intercept should always be the first exog variable!')
                if not ridge_kwds.get('penalize_intercept', False):
                    ridge_parameters[0] = 0.0

            if ridge_kwds.get('normalize', True):
                from kanly.regression.linear_models.penalized.sparse_elastic_net_internal import \
                    _get_normalizing_factors
                _, _, l2_norm_X = _get_normalizing_factors(self.exog, self.is_weighted, self.weights)
                ridge_parameters *= l2_norm_X ** 2

            if debug:
                print("%.4fs" % (time.time() - _time))

            return ridge_parameters

    def default_compute_eigenvalues(self, compute_eigenvalues):
        """Resolve the eigenvalue computation flag with an automatic default.

        When ``compute_eigenvalues`` is ``None``, eigenvalues are computed
        automatically for small models (< 500 regressors) and skipped for
        larger ones (where computing the full spectrum of a 500×500 Gram
        matrix would be slow).

        Args:
            compute_eigenvalues (bool or None): Explicit flag, or ``None``
                to trigger automatic selection.

        Returns:
            bool: Whether eigenvalues should be computed for this model.
        """
        if compute_eigenvalues is not None:
            return compute_eigenvalues
        else:
            return self.exog.shape[1] < 500

    def to_dense(self, inplace=False):
        """Convert all sparse model arrays to dense (ndarray) equivalents.

        Required before running GLS estimation (``sigma`` / ``sigma_inv``),
        which does not support sparse matrices.  Can also improve performance
        for very small models where dense BLAS operations outperform sparse
        arithmetic.

        Args:
            inplace (bool): If ``True``, convert all arrays on the current
                model object and set ``self.is_sparse_model = False``.
                If ``False`` (default), return a new ``SparseLinearModel``
                with dense arrays, leaving the original unchanged.

        Returns:
            SparseLinearModel or None: A new dense model when
                ``inplace=False``; ``None`` (modifies self) when
                ``inplace=True``.  Returns ``self`` unchanged if the model
                is already dense.
        """

        if not self.is_sparse_model:
            return self

        def to_dense(v, flatten):
            """Convert a sparse matrix or DataFrame column to a dense ndarray.

            Args:
                v: Value to convert (None, ndarray, sparse matrix, or DataFrame).
                flatten (bool): If True, reshape the result to a 1-D array.

            Returns:
                np.ndarray or None: Dense representation, or None if *v* is None.
            """
            if v is None or isinstance(v, np.ndarray):
                return v

            if isspmatrix(v):
                v = v.toarray()
                if flatten:
                    v = v.flatten()
                return v
            else:
                raise Exception

        dense_dict = {
            k: to_dense(getattr(self, k), flatten)
            for k, flatten in zip(
                ['endog', 'exog', 'instruments', 'weights', 'absorb'],
                [True, False, False, True, False]
            )
        }

        if inplace:
            self.exog, self.endog, self.instruments, self.absorb, self.weights = (
                dense_dict['exog'], dense_dict['endog'], dense_dict['instruments'], dense_dict['absorb'],
                dense_dict['weights'],
            )
            self.is_sparse_model = False
        else:
            return SparseLinearModel(
                endog=dense_dict['endog'], exog=dense_dict['exog'],
                add_constant=False,
                has_intercept=self.has_intercept, has_implicit_constant=self.has_implicit_constant,
                formula=self.formula,
                from_formula=self.from_formula,
                weights=dense_dict['weights'], instruments=dense_dict['instruments'],
                endog_name=self.endog_name, exog_names=self.exog_names, exog_term_names=self.exog_term_names,
                weights_name=self.weights_name, instrument_names=self.instrument_names,
                instrument_term_names=self.instrument_term_names,
                data=self.data, specification_name=self.specification_name, method=self.method,
                valid_obs_rows=self.valid_obs_rows,
                index=self.index, absorb=dense_dict['absorb'],
                absorb_names=self.absorb_names,
                absorb_term_name=self.absorb_term_name, cov_groups=self.cov_groups,
                cov_groups_name=self.cov_groups_name,
                # ridge_kwds=None,
                endog_regressors=self.endog_regressors, null_rows_info_dict=dict(), model_elapsed=self.model_elapsed,
                is_sure=self.is_sure,
                parent_model=self.parent_model,
                sigma=self.sigma, sigma_inv=self.sigma_inv,
            )

    @staticmethod
    def GLSAR(endog, exog, nlags, add_constant=False, has_constant=True, ar_method='yw', debug=False, maxiter=10,
              tol=1e-6,
              compute_eigenvalues=True,
              use_t=DEFAULT_LM_USE_T, test_level=DEFAULT_LM_TEST_LEVEL, keep_model=True, specification_name=None,
              exog_names=None, endog_name=None, exog_term_names=None, full_information=True, compute_cov=True):
        """
        Feasible GLS with AR(p) errors (array API).

        Estimates regression coefficients when residuals follow an autoregressive
        process of order ``nlags``. The estimator iterates: (1) fit OLS/GLS,
        (2) estimate AR parameters on residuals, (3) whiten ``y`` and ``X``,
        (4) refit on whitened data until AR coefficients stabilize. See
        :meth:`fit_glsar` for the algorithm and whitening details.

        **Whitening and ``full_information``**

        * ``full_information=True`` (default): **Prais-Winsten** — uses the
          stationary initial covariance for the first ``nlags`` observations
          (see :func:`~kanly.time_series.regression.regression.make_ar_full_information_W`).
        * ``full_information=False``: **Cochrane-Orcutt** — only innovation rows
          whiten the data; summaries adjust ``nobs`` and residual df by
          ``nlags`` and apply a small-sample covariance correction. This mode is
          closer to :class:`statsmodels.regression.linear_model.GLSAR`, whose
          ``whiten()`` drops the first ``rho`` observations.

        **Comparison to statsmodels**

        `statsmodels GLSAR <https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.GLSAR.html>`_
        is experimental and implements Cochrane-Orcutt-style whitening only.
        kanly defaults to Prais-Winsten; set ``full_information=False`` for a
        Cochrane-Orcutt analogue. statsmodels typically requires an outer loop
        over ``fit()`` and ``yule_walker``; kanly runs the iteration inside
        :meth:`fit_glsar`.

        **Restrictions**

        Not supported with absorbed fixed effects, IV, WLS weights, known
        ``sigma``/``sigma_inv`` (GLS), or SURE.

        Parameters
        ----------
        endog, exog : array-like
            Outcome and design matrix.
        nlags : int
            AR order ``p`` (must be >= 1).
        ar_method : str, default='yw'
            Passed to :func:`~kanly.time_series.autoregression.estimate_ar` on
            each iteration's residuals (e.g. ``'yw'`` / ``'yule-walker'``,
            ``'css'``).
        maxiter : int, default=10
            Maximum AR/GLS iterations.
        tol : float, default=1e-6
            Stop when ``max |ar_params - ar_params_previous| < tol``.
        full_information : bool, default=True
            Prais-Winsten (True) vs Cochrane-Orcutt (False) whitening.
        use_t, test_level, keep_model, specification_name, compute_eigenvalues
            Same role as in :meth:`LM` / :meth:`lm`.

        Returns
        -------
        SparseLinearRegressionResults
            ``method`` is ``'GLSAR[nlags]'``; AR diagnostics in ``fit.glsar_info``.

        See Also
        --------
        glsar : formula interface.
        fit_glsar : core iterative implementation.
        """
        model: SparseLinearModel = SparseLinearModel(
            endog, exog, add_constant, has_constant, has_constant, None, False,
            weights=None, instruments=None,
            endog_name=endog_name, exog_names=exog_names, exog_term_names=exog_term_names,
            weights_name=None, instrument_names=None,
            instrument_term_names=None,
            method='GLSAR',
            data=None, specification_name=specification_name, valid_obs_rows=None,
            index=None, absorb=None, absorb_names=None,
            absorb_term_name=None,
            sigma=None, sigma_inv=None)

        assert isinstance(nlags, int) and nlags >= 1
        assert model.is_absorb == False
        assert model.is_iv == False
        assert model.is_gls == False
        assert model.is_weighted == False
        assert model.is_sure == False

        return model.fit_glsar(nlags=nlags, ar_method=ar_method, debug=debug, maxiter=maxiter, tol=tol,
                               compute_eigenvalues=compute_eigenvalues, use_t=use_t,
                               test_level=test_level, keep_model=keep_model,
                               specification_name=specification_name, full_information=full_information,
                               compute_cov=compute_cov)

    @staticmethod
    def glsar(formula, data, nlags, ar_method='yw', debug=False, maxiter=10, tol=1e-6, compute_eigenvalues=True,
              use_t=DEFAULT_LM_USE_T, test_level=DEFAULT_LM_TEST_LEVEL, keep_model=True, specification_name=None,
              full_information=True, compute_cov=True):
        """
        Feasible GLS with AR(p) errors (formula API).

        Same estimator as :meth:`GLSAR`, but builds the design from a patsy-like
        formula and ``data`` (e.g. ``'y ~ x + C(grp)'``).

        Examples
        --------
        ::

            from kanly.api import glsar

            fit = glsar('y ~ x + C(grp)', df, nlags=2)
            print(fit.glsar_info)  # AR coefficients, iterations, full_information flag

        Use ``full_information=False`` for Cochrane-Orcutt whitening (compare to
        statsmodels ``GLSAR``); default is Prais-Winsten.

        Parameters
        ----------
        formula : str
            Linear model formula (same syntax as :meth:`lm`).
        data : DataFrame
            Data source for the formula.
        nlags : int
            AR order ``p``.
        ar_method, maxiter, tol, full_information, debug, compute_eigenvalues,
        use_t, test_level, keep_model, specification_name
            See :meth:`GLSAR` and :meth:`fit_glsar`.

        Returns
        -------
        SparseLinearRegressionResults

        See Also
        --------
        GLSAR, fit_glsar
        """
        model: SparseLinearModel = SparseLinearModel.build_model_from_formula(formula, data, debug=debug)
        assert isinstance(nlags, int) and nlags >= 1
        assert model.is_absorb == False
        assert model.is_iv == False
        assert model.is_gls == False
        assert model.is_weighted == False
        assert model.is_sure == False

        return model.fit_glsar(nlags=nlags, ar_method=ar_method, debug=debug, maxiter=maxiter, tol=tol,
                               compute_eigenvalues=compute_eigenvalues, use_t=use_t,
                               test_level=test_level, keep_model=keep_model,
                               specification_name=specification_name, full_information=full_information,
                               compute_cov=compute_cov)

    def fit_glsar(self, nlags=1, ar_method='yw', debug=False, maxiter=10, tol=1e-6, compute_eigenvalues=True,
                  use_t=True, test_level=.05, keep_model=True, specification_name=None, full_information=True,
                  compute_cov=True):
        """
        Fit GLSAR on an existing :class:`SparseLinearModel` (AR errors).

        Algorithm
        ---------
        1. **Initial OLS** — :meth:`fit_lsmr` to get starting residuals ``e``.
        2. **Iterate** up to ``maxiter`` times:

           a. **Estimate AR(p)** on ``e`` via
              :func:`~kanly.time_series.autoregression.estimate_ar`
              (``ar_method``, order ``nlags``) → ``ar_params``, ``scale``.
           b. **Whitening** — build ``W`` with
              :func:`~kanly.time_series.regression.regression.make_ar_full_information_W`
              (Prais-Winsten if ``full_information=True``, else Cochrane-Orcutt).
           c. **GLS step** — regress ``Wy`` on ``WX`` (:func:`lm_internal`).
           d. **Convergence** — if AR coefficients changed by less than ``tol``
              since the previous iteration, stop.
           e. **Update residuals** — ``e = y - X @ beta`` for the next AR fit.

        3. **Inference** — ``OLS_SMALL`` covariance on the final whitened fit;
           if ``full_information=False``, apply a Cochrane-Orcutt small-sample
           scale to ``cov_params`` and reduce ``nobs`` / ``df_resid`` by ``nlags``.

        Results store AR diagnostics in :class:`~kanly.regression.linear_models.regression_results.GLSARInfo`
        as ``fit.glsar_info`` (``ar_params``, ``nlags``, ``numiter``, ``full_information``, etc.).

        Parameters
        ----------
        nlags : int, default=1
            AR order ``p``.
        ar_method : str, default='yw'
            AR estimation method for :func:`~kanly.time_series.autoregression.estimate_ar`.
        maxiter : int, default=10
            Maximum iterations of the AR → whiten → GLS loop.
        tol : float, default=1e-6
            Convergence tolerance on AR coefficient changes.
        full_information : bool, default=True
            ``True``: Prais-Winsten (full initial-information whitening).
            ``False``: Cochrane-Orcutt (innovation rows only; see helper docs).
        use_t, test_level, keep_model, specification_name, compute_eigenvalues, debug
            Standard result and solver options.

        Returns
        -------
        SparseLinearRegressionResults
            ``method`` attribute ``'GLSAR[nlags]'``.

        See Also
        --------
        glsar, GLSAR : user-facing constructors.
        make_ar_full_information_W : whitening matrix construction.
        """
        assert isinstance(nlags, int) and nlags >= 1
        assert self.is_absorb == False
        assert self.is_iv == False
        assert self.is_gls == False
        assert self.is_weighted == False
        assert self.is_sure == False

        _time = time.time()
        is_sparse = self.is_sparse_model
        n = self.nobs

        fit0 = self.fit_lsmr()

        e = fit0.resid
        ar_last = None

        for itr in range(maxiter):

            ar_estimate = estimate_ar(e, nlags, method=ar_method)
            ar_params, scale = ar_estimate['arparams'], ar_estimate['sigma2']

            # Whitening: Prais-Winsten (full_information=True) or Cochrane-Orcutt (False)
            W = make_ar_full_information_W(ar_params, n, scale=1.0, full_information=full_information)
            if not is_sparse:
                W = W.toarray()
            WX = W.dot(self.exog)
            Wy = W.dot(self.endog)
            result: LinearModelRegressionResultsRaw = lm_internal(
                Wy, WX, compute_eigenvalues=compute_eigenvalues)

            beta = result.params
            if not is_sparse:
                beta = beta.toarray().flatten()
            beta = beta.reshape((-1, 1))

            if ar_last is not None:
                ar_error = np.max(np.abs(ar_params - ar_last))
                if debug:
                    print(f'{itr=}, {ar_error=}, {ar_params=}, {scale=}')
                if ar_error < tol:
                    break

            e = self.endog - self.exog.dot(beta)
            if is_sparse:
                e = e.toarray().flatten()
            ar_last = ar_params

        if isspmatrix(beta):
            beta = beta.toarray().flatten()

        ncp = result.normalized_cov_params

        (df_resid, df_model, rsquared, rsquared_adj, wssr, ssr, wsst, sst, uncentered_tss, resid, wresid, fittedvalues,
         llf,
         absorbed_y_baselines, rsquared_within, rsquared_between) = get_fit_summary_stats(
            self.nobs, self.num_absorbed, beta,
            Wy,
            WX,
            result.exog_absorb_instrumented,
            result.rsquared_within_raw,
            None, False, self.is_weighted, self.has_implicit_constant,
            self.has_intercept, self.is_absorb,
            None,
            None, None,
            _time=_time, debug=debug)

        condition_number = result.condition_number
        eigenvals = result.eigenvals

        fit_elapsed = time.time() - _time

        if compute_cov:
            _time = time.time()
            cov_type = 'OLS_SMALL'
            cov_kwds = dict()
            cov_params, num_groups, df_t_dist, small_samp_correct, cov_elapsed \
                = SparseVarianceCovariance2.compute_cov_params(
                cov_type, cov_kwds, use_t, df_resid, wssr, resid,
                ncp, False, result.exog_absorb_instrumented,
                groups=None,
                debug=debug, _time=_time, weights=None, param_name=f'{self.endog_name} params',
                sigma=None, sigma_inv=None,
            )

            # Cochrane-Orcutt: adjust covariance and effective sample for dropped initial obs
            if not full_information:
                cov_params *= (n - (nlags + 1)) / (n - (2 * nlags + 1))
                small_samp_correct = (n - (nlags + 1), n - (2 * nlags + 1))

            cov_string = SparseVarianceCovariance2.get_cov_string(
                self.nobs, cov_type, cov_kwds, self.cov_groups_name)
        else:
            df_t_dist = None
            small_samp_correct = None
            cov_elapsed = 0.0
            cov_type = 'NOT COMPUTED'
            cov_params = None
            cov_string = None
            cov_kwds = dict()

        method = f'GLSAR[{nlags}]'

        glsar_info = GLSARInfo(nlags, ar_params, scale, itr + 2, ar_error, ar_method, full_information)

        do_fgls = False
        nobs = self.nobs
        if not full_information:
            nobs -= nlags
            df_resid -= nlags

        fit = SparseLinearRegressionResults(
            use_t, self, nobs, df_resid, df_model, df_t_dist, beta,
            cov_params, rsquared, rsquared_adj,
            ssr, wssr, sst, wsst, uncentered_tss, self.endog_name, self.exog_names, self.exog_term_names,
            llf, fit_elapsed,
            cov_elapsed,
            cov_type, cov_kwds.copy(), self.has_implicit_constant or self.has_intercept,
            self.has_implicit_constant,
            method,
            resid, wresid, fittedvalues,
            eigenvals, condition_number, ncp,
            absorb_info=None, instrument_info=None,
            weights_name=None, cov_string=cov_string, test_level=test_level,
            small_samp_correct=small_samp_correct,
            wexog_instrumented_means=self.wexog_instrumented_means,
            keep_model=keep_model, null_rows_info_dict=self.null_rows_info_dict.copy(),
            valid_obs_rows=self.valid_obs_rows.copy(), specification_name=specification_name,
            do_fgls=do_fgls, fgls_weights=None,
            fgls_info=dict(), ridge_kwds=None, glsar_info=glsar_info
        )

        return fit

# if __name__ == '__main__':
#
#     from kanly.api import simulate_sarima, autoreg
#     import pandas as pd
#     n = 200
#     y = simulate_sarima(n=n, seed=0, ar=[.5])
#     y -= y.mean()
#     y += 3 + .35 * np.arange(n)
#     fit = SparseLinearModel.AUTOREG(y, lags=2, trend='ct', seasonal_periods=3)
#     print(fit)
#
#     df = pd.DataFrame({'y': y, 't': np.arange(n), 'period': np.arange(n) % 3})
#     print(autoreg('y~1', df, trend='ct', lags=2, seasonal_periods=3))


# from kanly.regression_results_base import RegressionResultsBase
# class BayesianLinearRegressionResults(RegressionResultsBase):
#
#     def __init__(self):
#         pass
#
#
# if __name__ == '__main__':
#
#     import pandas as pd
#     from kanly.api import lm
#
#     n = 12
#     np.random.seed(0)
#     df = pd.DataFrame({
#         'x': np.random.randn(n),
#         'grp': np.random.randint(0, 12, n),
#         'obs': np.random.randint(1, 6, n),
#     })
#     df['y'] = 1.2 - 0.3 * df['x'] + np.sqrt(.2) * np.random.randn(n)
#
#     model = SparseLinearModel.build_model_from_formula('y ~ x', df)
#     print(model.fit())
#     result = model.bayesian_linear_regression_TO_DELETE([0,0], np.eye(2)*.01, .2, 1.1)
#     for k, d in result.items():
#         print(k, d.mean(), d.std())

# if __name__ == '__main__':
# #
#     import pandas as pd
#     from kanly.api import lm
#     from pathos.pools import ProcessingPool
#
#     n = 1200
#     np.random.seed(0)
#     df = pd.DataFrame({
#         'x': np.random.randn(n),
#         'grp': np.random.randint(0, 12, n),
#         'obs': np.random.randint(1, 6, n),
#     })
#     df['y'] = 1.2 - 0.3 * df['x'] + np.sqrt(.2) * np.random.randn(n)
#
#     lm('y ~ x + C(grp)', df, cov_type='bootstrap', cov_kwds={'n_samples': 500}, debug=True)
