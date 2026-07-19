"""Public API for sparse robust linear regression (M-estimation).

``SparseRobustLinearModel`` wraps ``rlm_internal`` and exposes two entry
points:

- ``rlm(formula, data, ...)``  — Patsy formula string API (most common).
- ``RLM(endog, exog, ...)``    — Raw matrix API for programmatic use.

Bootstrap covariance is handled here (``cov_type='BOOTSTRAP'``) via
``do_bootstrap2``; all analytic covariance types are delegated to
``variance_covariance.py``.

Note: IV and fixed-effects absorption are not supported for RLM.
"""
from __future__ import absolute_import, print_function

import time

from pandas import DataFrame

from kanly.bootstrap.bootstrap import do_bootstrap2, DEFAULT_BB_ALPHA, DEFAULT_BB_METHOD, get_bootstrap_weights2, DEFAULT_BB_MAX_PROCESSES
from kanly.formula.data_getter import (SparseDataGetter, ENDOG_KEY, EXOG_KEY, HAS_INTERCEPT_KEY, INDEX_KEY,
                                       VALID_OBS_ROWS_KEY, WEIGHTS_KEY,
                                       HAS_IMPLICIT_CONSTANT_KEY, NULL_ROWS_INFO_DICT_KEY)
from kanly.formula.keys import FORMULA_DESIGN_INFO_KEY
from kanly.regression.linear_models.robust.constants import (DEFAULT_RLM_COV_TYPE, DEFAULT_RLM_M, DEFAULT_RLM_MAX_ITER,
                                                  DEFAULT_RLM_X_TOL, DEFAULT_RLM_BOOTSTRAP_N_SAMPLES,
                                                  DEFAULT_RLM_TEST_LEVEL)
from kanly.regression.linear_models.robust.regression_results import SparseRobustLinearRegressionResults
from kanly.regression.linear_models.robust.rlm_internal import rlm_internal
from kanly.regression.linear_models.robust.variance_covariance import BOOTSTRAP
from kanly.regression.linear_model_base import LinearModelBase
from kanly.regression.cov_types import check_cov_kwds


def get_variable_names(endog_name, exog_names, num_params, weights_name, is_weighted):
    """Resolve display names for the response, regressors, and weights.

    Fills in placeholder strings for any name that was not supplied by the
    caller, so that result tables always have human-readable labels.

    Args:
        endog_name (str or None): Name of the response variable.  If None,
            the placeholder ``'<y>'`` is used.
        exog_names (list of str or None): Names of the regressor columns.  If
            None, placeholders ``'<x0>'``, ``'<x1>'``, … are generated.
        num_params (int): Number of regressor columns; used only when
            ``exog_names`` is None.
        weights_name (str or None): Name of the weights column.  Ignored when
            ``is_weighted`` is False.
        is_weighted (bool): Whether observation weights are present.

    Returns:
        tuple: Three-element tuple ``(endog_name, exog_names, weights_name)``
            with placeholders filled in where needed.  ``weights_name`` is
            set to None when ``is_weighted`` is False.
    """
    if endog_name is None:
        endog_name = '<y>'
    if exog_names is None:
        exog_names = [f'<x{j}>' for j in range(num_params)]
    if is_weighted:
        if weights_name is None:
            weights_name = '<weights>'
    else:
        weights_name = None
    return endog_name, exog_names, weights_name


class SparseRobustLinearModel(LinearModelBase):
    """Sparse robust linear model (M-estimation) wrapping the IRLS core.

    Extends ``LinearModelBase`` with the ``'RLM'`` method.  The two main
    user-facing entry points are the static methods:

    - ``rlm(formula, data, ...)``  — Patsy formula string API.
    - ``RLM(endog, exog, ...)``    — Raw matrix API.

    Both ultimately call ``fit``, which delegates the numerical work to
    ``rlm_internal`` and wraps the results in
    ``SparseRobustLinearRegressionResults``.

    Limitations:
        - Single outcome (``accepts_multi_outcome`` returns False).
        - IV (instrumental variables) and fixed-effects absorption are not
          supported (``fail_on_iv=True``, ``fail_on_absorb=True``).

    Examples
    --------
    Huber M-estimation on data contaminated with outliers:

    >>> import numpy as np, pandas as pd
    >>> from kanly.api import rlm
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> df = pd.DataFrame({'x1': rng.normal(size=n),
    ...                    'x2': rng.normal(size=n)})
    >>> df['y'] = 1.0 + 2.0*df['x1'] - 0.5*df['x2'] + rng.normal(size=n)
    >>> df.loc[df.sample(20, random_state=1).index, 'y'] += 10  # outliers
    >>> fit = rlm('y ~ x1 + x2', df)                       # doctest: +SKIP

    Different M-functions (``'HuberT'``, ``'TukeyBiweight'``,
    ``'AndrewWave'``, ``'Hampel'``, ``'RamsayE'``):

    >>> fit_tukey = rlm('y ~ x1 + x2', df, M='TukeyBiweight')   # doctest: +SKIP

    Bootstrap covariance (recommended for robust SEs):

    >>> fit_bs = rlm('y ~ x1 + x2', df,                    # doctest: +SKIP
    ...              cov_type='bootstrap',
    ...              cov_kwds={'n_samples': 500, 'method': 'bayesian'})

    Matrix-form entry point: ``RLM(y, X, ...)``.

    See Also
    --------
    :meth:`rlm`, :meth:`RLM`.
    """

    def __init__(self, endog, exog, add_constant, has_intercept, has_implicit_constant, formula_design_info, weights=None,
                 endog_name=None, exog_names=None, weights_name=None, valid_obs_rows=None, index=None,
                 specification_name=None, null_rows_info_dict=None, model_elapsed=0):
        """Initialise the model, hardcoding ``method='RLM'``.

        All arguments are forwarded to ``LinearModelBase.__init__``.
        The ``method`` is always ``'RLM'``; ``instruments`` and ``absorb``
        are always None (not supported for robust regression).

        Args:
            endog (array-like or sparse): Response vector y, shape (n,).
            exog (array-like or sparse): Design matrix X, shape (n, p).
            add_constant (bool): Whether a constant column was prepended.
            has_intercept (bool): Whether X contains an intercept.
            has_implicit_constant (bool): Whether a constant is implicit in X.
            formula_design_info:
            weights (array-like, optional): WLS observation weights.
            endog_name (str, optional): Display name for the response.
            exog_names (list of str, optional): Display names for regressors.
            weights_name (str, optional): Display name for the weights column.
            valid_obs_rows (array-like, optional): Boolean/integer row selector
                for valid (non-null) observations.
            index (array-like, optional): Row index labels for predictions.
            data (DataFrame, optional): Original data (retained for
                ``predict`` and ``build_model`` re-use).
            specification_name (str, optional): Human-readable model label.
            null_rows_info_dict (dict, optional): Metadata about dropped rows.
            model_elapsed (float): Wall-clock time taken to build the model.
        """
        super().__init__(endog, exog, add_constant, has_intercept, has_implicit_constant, formula_design_info, weights=weights,
                         instruments=None, endog_name=endog_name, absorb=None, exog_names=exog_names,
                         weights_name=weights_name, instrument_names=None,
                         index=index, valid_obs_rows=valid_obs_rows,
                         model_elapsed=model_elapsed,
                         method='RLM', specification_name=specification_name, null_rows_info_dict=null_rows_info_dict)

    def fit(self, start_params=None, M=DEFAULT_RLM_M, x_tol=DEFAULT_RLM_X_TOL, max_iter=DEFAULT_RLM_MAX_ITER,
            cov_type=DEFAULT_RLM_COV_TYPE, cov_kwds=None, keep_model=True, force_scale=None,
            test_level=DEFAULT_RLM_TEST_LEVEL, debug=False, compute_cov=True):
        """Fit the robust linear model and return a results object.

        Calls ``rlm_internal`` (IRLS) to estimate coefficients, then wraps the
        output in ``SparseRobustLinearRegressionResults``.  Bootstrap covariance
        is computed here if ``cov_type='BOOTSTRAP'``.

        Args:
            start_params (array-like, optional): Initial β vector for IRLS.
                If None, an OLS warm-start is used.
            M (str, type, or RobustNormFunction): Norm (influence) function.
                Default ``DEFAULT_RLM_M`` (``'HuberT'``).
            x_tol (float): IRLS convergence tolerance on max β change.
            max_iter (int): Maximum IRLS iterations.
            cov_type (str): Covariance type: ``'H1'``, ``'H2'``, ``'H3'``,
                ``'SANDWICH'``, or ``'BOOTSTRAP'``.
            cov_kwds (dict, optional): Extra keywords for covariance estimation.
                For ``BOOTSTRAP``, supported keys include:

                - ``'n_samples'`` (int): Number of bootstrap draws.
                - ``'method'`` (str): Bootstrap method (Bayesian, block, …).
                - ``'alpha'`` (float): Significance level for CI.
                - ``'groups'`` (array-like): Group labels for block bootstrap.
                - ``'seed'`` (int): Random seed for reproducibility.
                - ``'max_processes'`` (int): Parallel workers.

            keep_model (bool): If True, attach the model object to results.
            force_scale (float, optional): Fix σ̂ at this value instead of
                estimating via MAD.
            test_level (float): Significance level for hypothesis tests.
            debug (bool): Print IRLS iteration diagnostics.
            compute_cov (bool): If False, skip covariance computation.

        Returns:
            SparseRobustLinearRegressionResults: Fitted model results,
                including coefficients, standard errors, scale, pseudo-R²,
                and convergence information.
        """
        cov_type = cov_type.upper()
        if cov_kwds is None:
            cov_kwds = dict()

        if BOOTSTRAP in cov_type:
            check_cov_kwds(cov_type, cov_kwds)

        fit_dict = rlm_internal(
            self.endog, self.exog, self.has_intercept, weights=self.weights, start_params=start_params, M=M,
            x_tol=x_tol, max_iter=max_iter, force_scale=force_scale,
            cov_type=cov_type, debug=debug, compute_cov=compute_cov and (BOOTSTRAP not in cov_type),
            cov_kwds=cov_kwds)

        fit = SparseRobustLinearRegressionResults(
            self.exog.shape[0], fit_dict['coef'].toarray().flatten(), fit_dict['fittedvalues'], compute_cov,
            fit_dict['var_covar'], cov_type, fit_dict['resid'], fit_dict['df_resid'], fit_dict['df_model'], fit_dict['scale'],
            fit_dict['pseudo_rsquared'], fit_dict['cost'],
            fit_dict['irls_weights'], self, M, fit_elapsed=fit_dict['fit_elapsed'], keep_model=keep_model, iteration_info=fit_dict['iteration_info'],
            specification_name=self.specification_name)

        if compute_cov and BOOTSTRAP in cov_type:
            # Bootstrap branch: re-run IRLS on each resampled dataset, warm-
            # starting from the point estimate, then accumulate the bootstrap
            # distribution of β̂ for covariance and CI estimation.
            def param_estimation_func(wts_bootstrap):
                """Run a single bootstrap IRLS replicate.

                Combines the bootstrap resampling weights with any WLS
                observation weights, re-fits the model, and returns the
                converged coefficient vector.  Returns None if IRLS did not
                converge on this replicate.

                Args:
                    wts_bootstrap (ndarray): Raw bootstrap multiplicity weights
                        for this replicate, shape (n,).

                Returns:
                    ndarray or None: Converged coefficient vector of shape (p,),
                        or None if this replicate failed to converge.
                """
                wts_bootstrap = get_bootstrap_weights2(wts_bootstrap, self.weights)
                result = rlm_internal(self.endog, self.exog, self.has_intercept, weights=wts_bootstrap,
                                      start_params=fit.params, M=M,
                                      x_tol=x_tol, max_iter=max_iter, force_scale=force_scale,
                                      cov_type=cov_type, debug=False,
                                      compute_cov=False, cov_kwds=cov_kwds)
                if result['iteration_info']['converged']:
                    return result['coef'].toarray().flatten()
                else:
                    return None

            do_bootstrap2(self.nobs, fit, param_estimation_func, groups=cov_kwds.get('groups', None),
                          n_samples=cov_kwds.get('n_samples', DEFAULT_RLM_BOOTSTRAP_N_SAMPLES),
                          method=cov_kwds.get('method', DEFAULT_BB_METHOD), alpha=cov_kwds.get('alpha', DEFAULT_BB_ALPHA),
                          seed=cov_kwds.get('seed', 0), debug=debug, use_correction=True, test_level=test_level,
                          max_processes=cov_kwds.get('max_processes', DEFAULT_BB_MAX_PROCESSES)
                          )

        return fit

    def build_model(self, data, index=None, debug=False, strip_non_exog=False, check_constant_cols=True, drop_1_for_FE=True):
        """Rebuild the model on new data using the stored formula.

        A thin wrapper around ``build_model_from_formula`` that reuses
        ``self.formula`` so that callers do not need to supply it again.

        Args:
            data (DataFrame): New dataset to build the model on.
            index (array-like, optional): Row index for predictions.
            debug (bool): Pass-through to ``build_model_from_formula``.
            strip_non_exog (bool): Drop sparse_terms that are not exogenous.
            check_constant_cols (bool): Detect and warn about near-constant
                columns.
            drop_1_for_FE (bool): Drop one level per fixed-effect category.

        Returns:
            SparseRobustLinearModel: A new model instance built on ``data``.
        """
        return self.build_model_from_formula(self.formula, data, index=index, debug=debug,
                                             strip_non_exog=strip_non_exog,
                                             check_constant_cols=check_constant_cols, drop_1_for_FE=drop_1_for_FE)

    def predict(self, params, data=None, index=None, debug=False):
        """Dead-code stub — overridden by the concrete ``predict`` definition below.

        This definition is never called at runtime; Python resolves the name
        ``predict`` to the second definition in the class body.  Retained as
        an artefact of the class's development history.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError()

    @staticmethod
    def build_model_from_formula(formula, data, index=None, debug=False, test_formula_on_dummy=True,
                                 check_constant_cols=True, drop_1_for_FE=True,
                                 specification_name=None, strip_non_exog=False):
        """Build a ``SparseRobustLinearModel`` from a Patsy formula and a DataFrame.

        Parses the formula via ``SparseDataGetter.get_data``, extracts the
        response, design matrix, and optional weights, then constructs and
        returns a ``SparseRobustLinearModel`` ready for ``fit``.

        Note: IV (``fail_on_iv=True``) and fixed-effects absorption
        (``fail_on_absorb=True``) are explicitly disallowed for robust
        regression.

        Args:
            formula (str): Patsy-style formula, e.g.
                ``'y ~ x1 + x2'`` or ``'y ~ x1 + x2 | fe_var'`` (FE not
                absorbed; the ``|`` syntax raises an error).
            data (DataFrame or dict): Data containing all formula variables.
            index (array-like, optional): Row index labels for predictions.
            debug (bool): Print formula-parsing diagnostics.
            test_formula_on_dummy (bool): Validate formula on a dummy dataset
                before applying to the real data.
            check_constant_cols (bool): Detect and warn about near-constant
                columns in the design matrix.
            drop_1_for_FE (bool): Drop one level per fixed-effect category
                (only applies if FE columns are included directly in formula).
            specification_name (str, optional): Human-readable label for this
                model specification.
            strip_non_exog (bool): Remove non-exogenous sparse_terms from the formula
                before parsing.

        Returns:
            SparseRobustLinearModel: A new model instance ready to call
                ``fit`` on.
        """
        _t = time.time()

        if isinstance(data, dict):
            data = DataFrame(data, copy=False)

        formula = SparseRobustLinearModel.strip_formula_to_just_exog_terms(formula, strip_non_exog=strip_non_exog)

        data_result = SparseDataGetter.get_data(
            data, formula, fail_on_iv=True, fail_on_absorb=True, fail_on_weights=False, debug=debug,
            test_formula_on_dummy=test_formula_on_dummy, index=index,
            check_constant_cols=check_constant_cols, drop_1_for_FE=check_constant_cols)

        endog, endog_name = data_result[ENDOG_KEY].values, data_result[ENDOG_KEY].column_names[0]
        exog, exog_names = data_result[EXOG_KEY].values, data_result[EXOG_KEY].column_names
        if data_result[WEIGHTS_KEY] is not None:
            weights, weights_name = data_result[WEIGHTS_KEY].values, data_result[WEIGHTS_KEY].column_names[0]
        else:
            weights, weights_name = None, None
        has_intercept = data_result[HAS_INTERCEPT_KEY]
        has_implicit_constant = data_result[HAS_IMPLICIT_CONSTANT_KEY]
        index = data_result[INDEX_KEY]
        valid_obs_rows = data_result[VALID_OBS_ROWS_KEY]
        null_rows_info_dict = data_result[NULL_ROWS_INFO_DICT_KEY]
        formula_design_info = data_result[FORMULA_DESIGN_INFO_KEY]

        return SparseRobustLinearModel(
            endog, exog, False, has_intercept, has_implicit_constant, formula_design_info, weights=weights, endog_name=endog_name,
            exog_names=exog_names, model_elapsed=time.time() - _t,
            weights_name=weights_name, index=index, valid_obs_rows=valid_obs_rows,
            specification_name=specification_name, null_rows_info_dict=null_rows_info_dict)

    @staticmethod
    def rlm(formula, data, start_params=None, M=DEFAULT_RLM_M, x_tol=DEFAULT_RLM_X_TOL, max_iter=DEFAULT_RLM_MAX_ITER,
            debug=False, cov_type=DEFAULT_RLM_COV_TYPE, test_formula_on_dummy=True, index=None, keep_model=True,
            compute_cov=True, cov_kwds=None, specification_name=None, test_level=DEFAULT_RLM_TEST_LEVEL,
            force_scale=None, residual_inclusion=False):
        """Formula-based convenience entry point for robust linear regression.

        Builds the model from a Patsy formula string, fits it with M-estimation,
        and returns a results object.  This is the most common user-facing API.

        Note: ``cov_type='BOOTSTRAP'`` with an integer ``index`` is not
        supported and raises an ``Exception``.

        Args:
            formula (str): Patsy formula, e.g. ``'y ~ x1 + x2'``.
            data (DataFrame or dict): Data containing all formula variables.
            start_params (array-like, optional): Initial β for IRLS warm-start.
            M (str, type, or RobustNormFunction): Norm function.  Default
                ``DEFAULT_RLM_M`` (``'HuberT'``).
            x_tol (float): IRLS convergence tolerance.
            max_iter (int): Maximum IRLS iterations.
            debug (bool): Print IRLS and formula-parsing diagnostics.
            cov_type (str): Covariance type.  Default ``DEFAULT_RLM_COV_TYPE``
                (``'H1'``).
            test_formula_on_dummy (bool): Validate formula on dummy data first.
            index (array-like, optional): Row index for predictions.
            keep_model (bool): Attach the model to the results object.
            compute_cov (bool): Compute the covariance matrix.
            cov_kwds (dict, optional): Extra keywords for covariance estimation
                (e.g. bootstrap ``n_samples``, ``method``, ``alpha``).
            specification_name (str, optional): Human-readable model label.
            test_level (float): Significance level for hypothesis tests.
            force_scale (float, optional): Fix σ̂ bypassing MAD estimation.
            residual_inclusion (bool): Reserved; not currently used for RLM.

        Returns:
            SparseRobustLinearRegressionResults: Fitted results object.

        Examples
        --------
        Basic robust linear model with Huber M-function (default):

        >>> import numpy as np, pandas as pd
        >>> from kanly.api import rlm
        >>> rng = np.random.default_rng(0)
        >>> n = 500
        >>> df = pd.DataFrame({'x1': rng.normal(size=n),
        ...                    'x2': rng.normal(size=n)})
        >>> df['y'] = 1.0 + 2.0*df['x1'] - 0.5*df['x2'] + rng.normal(size=n)
        >>> df.loc[df.sample(25, random_state=1).index, 'y'] += 10  # outliers
        >>> fit = rlm('y ~ x1 + x2', df)                   # doctest: +SKIP
        >>> fit.params.round(2)                             # doctest: +SKIP
        Intercept     0.99
        x1            2.01
        x2           -0.51
        dtype: float64

        Choose a different M-function:

        >>> fit_tukey = rlm('y ~ x1 + x2', df,             # doctest: +SKIP
        ...                 M='TukeyBiweight')

        Bootstrap covariance (recommended for inference):

        >>> fit_bs = rlm('y ~ x1 + x2', df,                # doctest: +SKIP
        ...              cov_type='bootstrap',
        ...              cov_kwds={'n_samples': 500})
        """
        if cov_type is None:
            cov_type = DEFAULT_RLM_COV_TYPE
        cov_type = cov_type.upper()

        if index is not None and cov_type.upper() == BOOTSTRAP:
            raise Exception("Bootstrap covariance not supported with integer index!")

        model = SparseRobustLinearModel.build_model_from_formula(
            formula, data, index=index, debug=debug, test_formula_on_dummy=test_formula_on_dummy,
            specification_name=specification_name)

        return model.fit(
            start_params=start_params, M=M, x_tol=x_tol, max_iter=max_iter, keep_model=keep_model,
            debug=debug, compute_cov=compute_cov, cov_type=cov_type, cov_kwds=cov_kwds, test_level=test_level,
            force_scale=force_scale)

    @staticmethod
    def RLM(endog, exog, add_constant=False, has_constant=False, weights=None, x_tol=DEFAULT_RLM_X_TOL, max_iter=DEFAULT_RLM_MAX_ITER,
            cov_type=DEFAULT_RLM_COV_TYPE, start_params=None, endog_name=None, exog_names=None, weights_name=None,
            M=DEFAULT_RLM_M, debug=False, cov_kwds=None, compute_cov=True, keep_model=True,
            specification_name=None, force_scale=None, test_level=DEFAULT_RLM_TEST_LEVEL):
        """Matrix-based convenience entry point for robust linear regression.

        Constructs a ``SparseRobustLinearModel`` directly from NumPy/sparse
        arrays (no formula parsing) and immediately calls ``fit``.

        Args:
            endog (array-like): Response vector y, shape (n,).
            exog (array-like or sparse): Design matrix X, shape (n, p).
            add_constant (bool): Prepend a column of ones to X.
            has_constant (bool): Whether X already contains a constant column.
            weights (array-like, optional): WLS observation weights.
            x_tol (float): IRLS convergence tolerance.
            max_iter (int): Maximum IRLS iterations.
            cov_type (str): Covariance type.  Default ``DEFAULT_RLM_COV_TYPE``
                (``'H1'``).
            start_params (array-like, optional): Initial β for IRLS warm-start.
            endog_name (str, optional): Display name for the response variable.
            exog_names (list of str, optional): Display names for regressors.
            weights_name (str, optional): Display name for the weights column.
            M (str, type, or RobustNormFunction): Norm function.
            debug (bool): Print IRLS diagnostics.
            cov_kwds (dict, optional): Extra covariance keywords.
            compute_cov (bool): Compute the covariance matrix.
            keep_model (bool): Attach the model to the results object.
            specification_name (str, optional): Human-readable model label.
            force_scale (float, optional): Fix σ̂ bypassing MAD estimation.
            test_level (float): Significance level for hypothesis tests.

        Returns:
            SparseRobustLinearRegressionResults: Fitted results object.

        Examples
        --------
        Robust regression directly from numpy arrays:

        >>> import numpy as np
        >>> from kanly.api import RLM
        >>> rng = np.random.default_rng(0)
        >>> n = 500
        >>> X = np.column_stack([np.ones(n),
        ...                      rng.normal(size=n),
        ...                      rng.normal(size=n)])
        >>> y = X @ np.array([1.0, 2.0, -0.5]) + rng.normal(size=n)
        >>> y[rng.choice(n, 20, replace=False)] += 10       # outliers
        >>> fit = RLM(y, X, has_constant=True,             # doctest: +SKIP
        ...           exog_names=['Intercept', 'x1', 'x2'],
        ...           M='HuberT')
        """
        model = SparseRobustLinearModel(
            endog, exog, add_constant, has_constant, has_implicit_constant=False, formula_design_info=None, weights=weights,
            endog_name=endog_name, exog_names=exog_names, weights_name=weights_name,
            valid_obs_rows=range(exog.shape[0]), index=None,
            null_rows_info_dict=dict(), specification_name=specification_name)
        return model.fit(
            start_params=start_params, M=M, x_tol=x_tol, max_iter=max_iter, keep_model=keep_model,
            debug=debug, compute_cov=compute_cov, cov_type=cov_type,
            cov_kwds=cov_kwds, force_scale=force_scale, test_level=test_level)

    def predict(self, params, data=None, index=None, debug=False,
                ignore_column_mismatch=False, *args, **kwargs):
        """Compute linear predictions Xβ for given parameters.

        Delegates to ``LinearModelBase.get_linear_predictor`` which applies
        the stored design matrix (or rebuilds one from ``data``) and returns
        ŷ = Xβ.

        Args:
            params (array-like): Coefficient vector β, shape (p,).
            data (DataFrame, optional): New data for out-of-sample prediction.
                If None, uses the training design matrix.
            index (array-like, optional): Row index for the returned Series.
            debug (bool): Print prediction diagnostics.
            ignore_column_mismatch (bool): When ``True``, allow prediction when
                the new design has fewer columns than ``params`` (e.g. missing
                fixed-effect levels). See
                :meth:`~kanly.regression.linear_model_base.LinearModelBase.get_linear_predictor`.
            *args: Ignored (accepted for interface compatibility).
            **kwargs: Ignored (accepted for interface compatibility).

        Returns:
            ndarray or Series: Predicted values ŷ = Xβ, shape (n,).
        """
        return self.get_linear_predictor(params, data=data, index=index, debug=debug,
                                         ignore_column_mismatch=ignore_column_mismatch)

    def accepts_multi_outcome(self):
        """Return False — robust regression supports only a single response.

        Returns:
            bool: Always False.
        """
        return False
