from __future__ import absolute_import, print_function

import time

import numpy as np
from pandas import DataFrame
from scipy.sparse import csc_matrix, isspmatrix

from kanly.bootstrap.bootstrap import (do_bootstrap2, get_bootstrap_weights2, DEFAULT_BB_METHOD, DEFAULT_BB_ALPHA,
                                       DEFAULT_BB_SEED, DEFAULT_BB_MAX_PROCESSES)
from kanly.formula.data_getter import (SparseDataGetter, ENDOG_KEY, EXOG_KEY, WEIGHTS_KEY, INSTRUMENTS_KEY,
                                       HAS_INTERCEPT_KEY, VALID_OBS_ROWS_KEY, HAS_IMPLICIT_CONSTANT_KEY,
                                       NULL_ROWS_INFO_DICT_KEY)
from kanly.formula.keys import FORMULA_DESIGN_INFO_KEY
from kanly.regression.generalized_linear_models.constants import (DEFAULT_GLM_FAMILY, DEFAULT_GLM_TOL,
                                                                  DEFAULT_GLM_MAX_ITER,
                                                                  DEFAULT_GLM_COV_TYPE, DEFAULT_GLM_RESIDUAL_INCLUSION,
                                                                  NONROBUST,
                                                                  HC1, BOOTSTRAP, DEFAULT_GLM_RESIDUAL_INCLUSION_ORDER,
                                                                  DEFAULT_GLM_ALPHA, DEFAULT_GLM_USE_T,
                                                                  DEFAULT_GLM_L1_RATIO,
                                                                  DEFAULT_GLM_FORCE_IV_PROJECTION,
                                                                  DEFAULT_GLM_PROMPT_USER_FOR_MORE_ITERS,
                                                                  DEFAULT_GLM_CHECK_CONSTANT_COLS,
                                                                  DEFAULT_GLM_TEST_LEVEL)
from kanly.regression.generalized_linear_models.families import _get_family_and_link
from kanly.regression.generalized_linear_models.links import _get_link
from kanly.regression.generalized_linear_models.regression_results import SparseGLMRegressionResults
from kanly.regression.generalized_linear_models.sparse_glm_internal import glm_internal, _get_opt_method, GLMRawFitData
from kanly.regression.generalized_linear_models.sparse_glm_var_covar_internal import get_robust_glm_covariance
from kanly.regression.linear_model_base import LinearModelBase
from kanly.regression.linear_models.penalized import _check_penalties
from kanly.regression.cov_types import check_cov_kwds, format_cov_kwds, check_cov_kwds
from kanly.regression.linear_models.sparse_iv_first_stage2 import convert_exog_col_map_to_col_names2
from kanly.utils.util import dict_2_dataframe
from kanly.utils.dict_2_array import dict_2_array


SHRINK_INTERCEPT = .5
LINE_SEARCH_SHRINK = .5
MAX_LINE_SEARCH = 10
GLM_COV_TYPES = [NONROBUST, HC1, BOOTSTRAP]
DEFAULT_GLM_BOOTSTRAP_N_SAMPLES = 100

DEBUG_INTERNAL = False

METHOD_IRLS = 'IRLS'
METHOD_COORD_DESC = 'COORDINATE_DESCENT'
METHOD_COORD_DESCENT_1_ITER = 'COORDINATE_DESCENT_1_ITER'


class SparseGeneralizedLinearModel(LinearModelBase):
    """
    Utility class for some parity with kanly RegressionResults
    in output

    Sparse GLM model wrapper that stores parsed response/design data,
    optional variance weights, and optional instruments.  The ``fit`` method
    delegates numerical estimation to ``glm_internal`` and wraps the result in
    ``SparseGLMRegressionResults``.

    Examples
    --------
    Logistic regression (``binomial`` family, default logit link):

    >>> import numpy as np, pandas as pd
    >>> from kanly.api import glm
    >>> rng = np.random.default_rng(0)
    >>> n = 1_000
    >>> df = pd.DataFrame({'x': rng.normal(size=n)})
    >>> df['p'] = 1 / (1 + np.exp(-(0.4 + 0.9 * df['x'])))
    >>> df['y'] = (rng.uniform(size=n) < df['p']).astype(float)
    >>> fit = glm('y ~ x', df, family='binomial')          # doctest: +SKIP

    Poisson regression with cluster-robust standard errors:

    >>> df['firm'] = rng.integers(0, 50, n)
    >>> df['count'] = rng.poisson(np.exp(0.5 + 0.3 * df['x']))
    >>> fit_pois = glm('count ~ x', df, family='poisson',  # doctest: +SKIP
    ...                cov_type='HC1')

    Penalised binomial GLM (elastic net via coordinate descent):

    >>> fit_pen = glm('y ~ x', df, family='binomial',      # doctest: +SKIP
    ...               alpha=0.1, l1_ratio=0.5)

    Bootstrap covariance:

    >>> fit_bs = glm('y ~ x', df, family='binomial',       # doctest: +SKIP
    ...              cov_type='bootstrap',
    ...              cov_kwds={'n_samples': 500, 'method': 'bayesian'})

    Matrix-form entry point on ``kanly.api`` is uppercase ``GLM``.

    See Also
    --------
    :meth:`glm`, :meth:`GLM`.
    """

    def __init__(self, exog, endog, add_constant, has_intercept, has_implicit_constant, formula_design_info,
                 is_gam, L2_penalty_matrix, regularize_to_values=None, weights=None, instruments=None, endog_name=None,
                 exog_names=None, weights_name=None,
                 instrument_names=None, specification_name=None, index=None, valid_obs_rows=None,
                 null_rows_info_dict=None, model_elapsed=None):
        """Initialize a sparse generalized linear model.

        Args:
            exog: Sparse or dense design matrix.
            endog: Response vector.
            add_constant: Whether the base class should add a constant column.
            has_intercept: Whether the formula/design already includes an intercept.
            has_implicit_constant: Whether the design implicitly contains a constant.
            formula_design_info: Parsed formula metadata (terms, spline knots, etc.).
            is_gam: Whether the model was constructed as a generalized additive
                model.
            L2_penalty_matrix: Optional symmetric ``p × p`` L2 penalty matrix
                added to ``X'WX`` in IRLS and to the covariance bread matrix.
                This may be a general ridge matrix supplied through :meth:`GLM`
                or the roughness matrix constructed for a GAM. ``None`` disables
                matrix penalization.
            regularize_to_values: Optional scalar or length-``p`` ridge target.
                The quadratic penalty is centered on these values; ``None``
                centers it at zero.
            weights: Optional variance weights.
            instruments: Optional instrument matrix for IV-style GLM.
            endog_name: Optional response name.
            exog_names: Optional exogenous column names.
            weights_name: Optional variance-weight column name.
            instrument_names: Optional instrument column names.
            data: Source data used for formula construction.
            instrument_term_names: Original instrument term names.
            exog_term_names: Original exogenous term names.
            specification_name: Optional label for summaries.
            index: Optional row indexer used during construction.
            valid_obs_rows: Rows retained after formula/data validation.
            null_rows_info_dict: Metadata about rows dropped due to null values.
            model_elapsed: Model-construction elapsed time in seconds.
        """
        super().__init__(
            endog, exog, add_constant, has_intercept, has_implicit_constant, formula_design_info, weights=weights, instruments=instruments,
            endog_name=endog_name, absorb=None, exog_names=exog_names,
            weights_name=weights_name, instrument_names=instrument_names,
            method='GLM', specification_name=specification_name, model_elapsed=model_elapsed,
            index=index, valid_obs_rows=valid_obs_rows, null_rows_info_dict=null_rows_info_dict,
        )
        self.is_gam = is_gam
        self.L2_penalty_matrix = None
        self._set_L2_penalty_matrix(L2_penalty_matrix, regularize_to_values)

    def _set_L2_penalty_matrix(self, L2_penalty_matrix, regularize_to_values):
        """Store an IRLS L2 penalty matrix and its target values.

        ``L2_penalty_matrix`` is passed through to :func:`~kanly.regression.generalized_linear_models.sparse_glm_internal.glm_internal`
        and added to ``X'WX`` each IRLS iteration. Covariance computation adds the
        same matrix to the bread (see
        :func:`~kanly.regression.generalized_linear_models.sparse_glm_var_covar_internal.get_robust_glm_covariance`).

        Args:
            L2_penalty_matrix: Symmetric ``p × p`` ridge or GAM roughness
                matrix, or ``None`` for no matrix penalty.
            regularize_to_values: Scalar or length-``p`` vector giving the
                center of the quadratic penalty. ``None`` uses zero.
        """
        if L2_penalty_matrix is None:
            self.L2_penalty_matrix = None
        else:
            if self.is_sparse_model:
                if not isspmatrix(L2_penalty_matrix):
                    L2_penalty_matrix = csc_matrix(L2_penalty_matrix)
            else:
                if isspmatrix(L2_penalty_matrix):
                    L2_penalty_matrix = L2_penalty_matrix.toarray()
            self.L2_penalty_matrix = L2_penalty_matrix

        if regularize_to_values is None:
            self.regularize_to_values = np.zeros((self.exog.shape[1],1))
        else:
            if isinstance(regularize_to_values, (float, int, np.integer)):
                self.regularize_to_values = np.array([regularize_to_values] * self.exog[0].shape[1])
            else:
                self.regularize_to_values = np.array(regularize_to_values)
            self.regularize_to_values = self.regularize_to_values.reshape((self.exog.shape[1], 1))

    def fit(self, family=DEFAULT_GLM_FAMILY, link=None, start_params=None, tol=DEFAULT_GLM_TOL, max_iter=DEFAULT_GLM_MAX_ITER,
            alpha=0.0, l1_ratio=0.0, L2_penalty_matrix=None, regularize_to_values=None, debug=False, normalize=True, penalize_scale=False, specification_name=None,
            use_t=True, test_level=DEFAULT_GLM_TEST_LEVEL, compute_cov=True, store_convergence_path=False,
            residual_inclusion=DEFAULT_GLM_RESIDUAL_INCLUSION, cov_kwds=None, cov_type=DEFAULT_GLM_COV_TYPE,
            fit_intercept=True, keep_model=True, prompt_user_for_more_iters=DEFAULT_GLM_PROMPT_USER_FOR_MORE_ITERS,
            line_search_fallback=True, pick_default_start=True, opt_method=None, first_column_constant=True,
            residual_inclusion_order=DEFAULT_GLM_RESIDUAL_INCLUSION_ORDER,
            force_iv_projection=DEFAULT_GLM_FORCE_IV_PROJECTION, ) -> SparseGLMRegressionResults:
        """Fit the GLM by IRLS or penalized coordinate descent.

        Args:
            family: GLM family name, class, or instance.
            link: Optional link name, class, or instance. Defaults to the
                family default/canonical link.
            start_params: Optional starting coefficient vector or dict keyed by
                exogenous variable name.
            tol: Convergence tolerance.
            max_iter: Maximum optimizer iterations.
            alpha: Elastic-net regularization strength.
            l1_ratio: Fraction of regularization assigned to L1 penalty.
            regularize_to_values: Optional scalar or length-``p`` center for a
                pure ridge penalty fitted with IRLS. If supplied here, it
                overrides the target stored when the model was constructed.
            debug: Whether to print optimizer diagnostics.
            normalize: Whether to normalize predictors for penalized fitting.
            penalize_scale: Whether penalties are multiplied by estimated scale.
            specification_name: Optional result label.
            use_t: Whether inference uses t-distribution critical values.
            test_level: Significance level for inference.
            compute_cov: Whether to compute covariance estimates.
            store_convergence_path: Whether to keep per-iteration diagnostics.
            residual_inclusion: Whether to include first-stage residuals for IV GLM.
            cov_kwds: Covariance/bootstrap options.
            cov_type: Covariance type: nonrobust, HC1, or bootstrap.
            fit_intercept: Whether to fit an explicit intercept.
            keep_model: Whether to retain the model on the result object.
            prompt_user_for_more_iters: Whether/how to prompt after max iterations.
            line_search_fallback: Whether to use fallback line search in GLM fitting.
            pick_default_start: Whether to choose family-based starting values.
            opt_method: Optimizer method override.
            first_column_constant: Whether the first design column is a constant.
            residual_inclusion_order: Polynomial order for IV residual inclusion.
            force_iv_projection: Whether to force first-stage IV projection.

        Returns:
            ``SparseGLMRegressionResults`` containing estimates and diagnostics.
        """

        if regularize_to_values is not None:
            self._set_L2_penalty_matrix(self.L2_penalty_matrix, regularize_to_values)

        opt_method = _get_opt_method(opt_method, alpha, l1_ratio)

        start_params = dict_2_array(start_params, self.exog_names, ignore_extra_keys=True, default_value=0.0)

        cov_kwds = format_cov_kwds(cov_kwds)

        cov_type = cov_type.upper()
        if cov_type not in GLM_COV_TYPES:
            if BOOTSTRAP not in cov_type:
                raise Exception("`cov_type` must be one of %s!" % str(GLM_COV_TYPES))
        check_cov_kwds(cov_type, cov_kwds)
        _check_penalties(alpha, l1_ratio)

        if L2_penalty_matrix is None:
            L2_penalty_matrix = self.L2_penalty_matrix

        fit_object: GLMRawFitData = glm_internal(
            self.endog, self.exog, L2_penalty_matrix=L2_penalty_matrix, regularize_to_values=self.regularize_to_values, 
            var_weights=self.weights, instruments=self.instruments, start_params=start_params,
            tol=tol, max_iter=max_iter, alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, debug=debug,
            family=family, link=link, normalize=normalize, penalize_scale=penalize_scale,
            store_convergence_path=store_convergence_path, force_iv_projection=force_iv_projection,
            residual_inclusion=residual_inclusion, residual_inclusion_order=residual_inclusion_order,
            line_search_fallback=line_search_fallback, pick_default_start=pick_default_start,
            opt_method=opt_method, first_column_constant=first_column_constant,
            prompt_user_for_more_iters=prompt_user_for_more_iters,
            is_endog_regressor=self.is_endog_regressor)

        if compute_cov and BOOTSTRAP not in cov_type:
            var_covar, cov_time = get_robust_glm_covariance(
                self.endog, fit_object.exog, fit_object.endog_predicted, self.weights, fit_object.irls_weights,
                fit_object.family, fit_object.link, fit_object.scale, cov_type, fit_intercept, first_column_constant,
                fit_object.alpha, fit_object.l2s, self.L2_penalty_matrix)
        else:
            var_covar = None
            cov_time = 0

        exog_names = convert_exog_col_map_to_col_names2(fit_object.exog_col_map, self.exog_names)

        fit = SparseGLMRegressionResults(fit_object.params, var_covar, self, exog_names, self.endog_name, self.nobs,
                                         fit_object.df_model, fit_object.df_resid, fit_object.family, fit_object.link,
                                         fit_object.alpha, fit_object.l1_ratio, fit_object.penalize_scale,
                                         fit_intercept, fit_object.normalize, fit_object.converged, fit_object.num_iter,
                                         fit_object.abs_error, fit_object.rel_error, fit_object.llf, fit_object.scale,
                                         fit_object.irls_weights, fit_object.normalized_cov_params, fit_object.edf,
                                         fit_object.endog_predicted, test_level, use_t,
                                         fit_object.llnull, fit_object.deviance, fit_object.pearson_chi2,
                                         fit_object.convergence_path, start_params, max_iter,
                                         fit_object.instrument_params, residual_inclusion, fit_object.resid,
                                         fit_object.g_prime, fit_object.lin_pred, cov_type, cov_kwds, self.weights_name,
                                         fit_object.fit_time, cov_time, opt_method, keep_model=keep_model,
                                         specification_name=specification_name)

        if compute_cov and BOOTSTRAP in cov_type:

            def param_estimation_func(bootstrap_weights):
                """Refit GLM coefficients for one bootstrap sample.

                Args:
                    bootstrap_weights: Bootstrap weights/frequencies.

                Returns:
                    Fitted parameter vector, or ``None`` if the refit fails to
                    converge.
                """
                bootstrap_weights = get_bootstrap_weights2(bootstrap_weights, self.weights)
                fit_object_temp = glm_internal(
                    self.endog, self.exog, L2_penalty_matrix=self.L2_penalty_matrix, regularize_to_values=self.regularize_to_values,
                    var_weights=bootstrap_weights,
                    instruments=self.instruments, start_params=fit_object.params, tol=tol, max_iter=max_iter,
                    alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, debug=False, family=family, link=link,
                    normalize=normalize, penalize_scale=penalize_scale, store_convergence_path=store_convergence_path,
                    residual_inclusion=residual_inclusion, residual_inclusion_order=residual_inclusion_order,
                    line_search_fallback=line_search_fallback, pick_default_start=pick_default_start,
                    opt_method=opt_method, first_column_constant=first_column_constant,
                    force_iv_projection=force_iv_projection)
                if fit_object_temp.converged:
                    return fit_object_temp.params.copy()
                else:
                    return None

            do_bootstrap2(self.nobs, fit, param_estimation_func, groups=None,
                          n_samples=cov_kwds.get('n_samples', DEFAULT_GLM_BOOTSTRAP_N_SAMPLES),
                          seed=cov_kwds.get('seed', DEFAULT_BB_SEED), debug=debug,
                          method=cov_kwds.get('method', DEFAULT_BB_METHOD), alpha=cov_kwds.get('alpha', DEFAULT_BB_ALPHA),
                          use_correction=cov_kwds.get('use_correction', True), test_level=test_level,
                          max_processes=cov_kwds.get('max_processes', DEFAULT_BB_MAX_PROCESSES),
                          group_name=None)

        return fit

    @staticmethod
    def log_likelihood_function_base(endog, exog, params, family, link=None, scale=1., var_weights=1.):
        """Evaluate the summed GLM log-likelihood for a parameter vector.

        Args:
            endog: Response vector.
            exog: Design matrix.
            params: Coefficient vector, or 2-D array of coefficient vectors.
            family: GLM family instance.
            link: Optional link instance; defaults to family canonical link.
            scale: Dispersion/scale parameter.
            var_weights: Variance weights.

        Returns:
            Scalar log-likelihood, or an array for 2-D ``params``.
        """
        return SparseGeneralizedLinearModel._log_likelihood_function_base_internal(
            endog, exog, params, family, False, link=link, scale=scale, var_weights=var_weights)

    @staticmethod
    def log_likelihood_function_base_obs(endog, exog, params, family, link=None, scale=1., var_weights=1.):
        """Evaluate observation-level GLM log-likelihood contributions.

        Args:
            endog: Response vector.
            exog: Design matrix.
            params: Coefficient vector.
            family: GLM family instance.
            link: Optional link instance; defaults to family canonical link.
            scale: Dispersion/scale parameter.
            var_weights: Variance weights.

        Returns:
            Vector of observation-level log-likelihood contributions.
        """
        return SparseGeneralizedLinearModel._log_likelihood_function_base_internal(
            endog, exog, params, family, True, link=link, scale=scale, var_weights=var_weights)

    @staticmethod
    def _log_likelihood_function_base_internal(endog, exog, params, family, obs_level, link=None, scale=1., var_weights=1.):
        """Evaluate summed or observation-level GLM log-likelihood.

        `obs_level` is true for observation level (a vector), otherwise sum

        Args:
            endog: Response vector.
            exog: Design matrix.
            params: Coefficient vector or matrix of coefficient vectors.
            family: GLM family instance.
            obs_level: Whether to return observation-level contributions.
            link: Optional link instance.
            scale: Dispersion/scale parameter.
            var_weights: Variance weights.

        Returns:
            Scalar, vector, or array of log-likelihood values.
        """

        if link is None:
            link = family.canonical_link

        if np.ndim(params) > 1:
            return np.array([
                SparseGeneralizedLinearModel.log_likelihood_function_base(
                    endog, exog, p, family, link, scale, var_weights) for p in params])

        if isspmatrix(exog):
            lin_pred = exog.dot(csc_matrix(params).reshape((-1, 1))).toarray().flatten()
        else:
            lin_pred = exog.dot(params)

        endog_predicted0 = link.inverse_link(lin_pred)
        theta0 = family.b_deriv_inv(endog_predicted0)

        func = family.log_likelihood_obs if obs_level else family.log_likelihood

        return func(endog, theta0, scale=scale, var_weights=var_weights)

    def get_log_likelihood_function(self, family, link=None):
        """Return a closure for the model's summed GLM log-likelihood.

        Args:
            family: GLM family instance.
            link: Optional link instance.

        Returns:
            Callable ``llf(params, scale=1., var_weights=1.)``.
        """
        family, link = _get_family_and_link(family, link)

        def llf(params, scale=1., var_weights=1):
            """Evaluate the model log-likelihood at ``params``.

            Args:
                params: Coefficient vector.
                scale: Dispersion/scale parameter.
                var_weights: Variance weights.

            Returns:
                Scalar log-likelihood.
            """
            return SparseGeneralizedLinearModel.log_likelihood_function_base(
                self.endog, self.exog, params, family, link=link, scale=scale, var_weights=var_weights)

        return llf

    def get_log_likelihood_function_obs(self, family, link=None):
        """Return a closure for observation-level GLM log-likelihood values.

        Args:
            family: GLM family instance.
            link: Optional link instance.

        Returns:
            Callable ``llf(params, scale=1., var_weights=1.)``.
        """
        if link is None:
            link = family.canonical_link

        def llf(params, scale=1., var_weights=1):
            """Evaluate observation-level log-likelihood contributions.

            Args:
                params: Coefficient vector.
                scale: Dispersion/scale parameter.
                var_weights: Variance weights.

            Returns:
                Vector of log-likelihood contributions.
            """
            return SparseGeneralizedLinearModel.log_likelihood_function_base_obs(
                self.endog, self.exog, params, family, link=link, scale=scale, var_weights=var_weights)

        return llf

    @staticmethod
    def build_model_from_formula(formula, data, debug=False, index=None, specification_name=None,
                                 drop_1_for_FE=True, cov_groups=None,
                                 check_constant_cols=DEFAULT_GLM_CHECK_CONSTANT_COLS):
        """Build a sparse GLM model from formula syntax and data.

        Args:
            formula: Formula string understood by ``SparseDataGetter``. Supports
                response/exog sparse_terms, optional weights, and optional instruments.
            data: pandas ``DataFrame`` or dict-like data.
            debug: Whether to print formula parsing diagnostics.
            index: Optional row subset/indexer.
            specification_name: Optional model label.
            drop_1_for_FE: Whether to drop one level from categorical fixed effects.
            cov_groups: Reserved for covariance grouping metadata.
            check_constant_cols: Whether to detect constant design columns.

        Returns:
            ``SparseGeneralizedLinearModel``.
        """

        _t = time.time()

        data = dict_2_dataframe(data)

        result = SparseDataGetter.get_data(data=data, formula=formula, debug=debug, index=index,
                                           check_constant_cols=check_constant_cols, drop_1_for_FE=drop_1_for_FE)

        endog, endog_name = result[ENDOG_KEY].values.toarray().flatten(), result[ENDOG_KEY].column_names[0]
        exog, exog_names, exog_term_names \
            = result[EXOG_KEY].values, result[EXOG_KEY].column_names, result[EXOG_KEY].term_names

        if result[INSTRUMENTS_KEY] is not None:
            instruments, instrument_names, instrument_term_names = (
                result[INSTRUMENTS_KEY].values, result[INSTRUMENTS_KEY].column_names,
                result[INSTRUMENTS_KEY].term_names
            )
        else:
            instruments, instrument_names, instrument_term_names = None, None, None

        if result[WEIGHTS_KEY] is not None:
            var_weights, var_weights_name = result[WEIGHTS_KEY].values, result[WEIGHTS_KEY].column_names[0]
        else:
            var_weights, var_weights_name = None, None

        fit_intercept = result[HAS_INTERCEPT_KEY]
        has_implicit_constant = result[HAS_IMPLICIT_CONSTANT_KEY]
        valid_obs_rows = result[VALID_OBS_ROWS_KEY]
        null_rows_info_dict = result[NULL_ROWS_INFO_DICT_KEY]
        formula_design_info = result[FORMULA_DESIGN_INFO_KEY]

        model = SparseGeneralizedLinearModel(
            exog, endog, False, fit_intercept, has_implicit_constant, formula_design_info, False, weights=var_weights,
            endog_name=endog_name, exog_names=exog_names, weights_name=var_weights_name, instruments=instruments,
            instrument_names=instrument_names, valid_obs_rows=valid_obs_rows, index=index,
            null_rows_info_dict=null_rows_info_dict, model_elapsed=time.time() - _t,
            specification_name=specification_name, L2_penalty_matrix=None
        )

        if debug:
            print(model)

        return model

    @staticmethod
    def GLM(endog, exog, L2_penalty_matrix=None, regularize_to_values=None, add_constant=False, instruments=None, start_params=None, tol=DEFAULT_GLM_TOL, max_iter=DEFAULT_GLM_MAX_ITER,
            var_weights=None, alpha=DEFAULT_GLM_ALPHA, l1_ratio=DEFAULT_GLM_L1_RATIO, debug=False,
            family=DEFAULT_GLM_FAMILY, link=None, exog_names=None,
            endog_name=None, fit_intercept=False, normalize=True, penalize_scale=False, use_t=DEFAULT_GLM_USE_T,
            test_level=DEFAULT_GLM_TEST_LEVEL, residual_inclusion_order=DEFAULT_GLM_RESIDUAL_INCLUSION_ORDER,
            force_iv_projection=DEFAULT_GLM_FORCE_IV_PROJECTION,
            compute_cov=True, store_convergence_path=False, instrument_names=None,
            residual_inclusion=DEFAULT_GLM_RESIDUAL_INCLUSION,
            cov_kwds=None, cov_type=DEFAULT_GLM_COV_TYPE, var_weights_name=None, line_search_fallback=True,
            pick_default_start=True, opt_method=None, first_column_constant=False,
            prompt_user_for_more_iters=DEFAULT_GLM_PROMPT_USER_FOR_MORE_ITERS):
        """Construct and fit a GLM from arrays rather than formula syntax.

        Args:
            endog: Response vector.
            exog: Design matrix.
            L2_penalty_matrix: Optional symmetric ``p × p`` matrix defining a
                quadratic penalty for IRLS. It is added directly to ``X'WX``,
                so the caller controls its scaling. Use ``alpha=0`` when
                supplying this matrix; a pure-ridge ``alpha`` fit builds and
                uses its own diagonal matrix. GAMs use the same mechanism with
                a spline roughness matrix.
            regularize_to_values: Optional scalar or length-``p`` target vector
                ``r`` for the matrix penalty, which is centered on ``r``.
                ``None`` uses zero. This argument applies to IRLS ridge fitting.
            add_constant: Whether to add a constant in the base model.
            instruments: Optional instrument matrix.
            start_params: Optional starting coefficient vector.
            tol: Convergence tolerance.
            max_iter: Maximum optimizer iterations.
            var_weights: Optional variance weights.
            alpha: Elastic-net regularization strength.
            l1_ratio: L1 share of regularization. With ``l1_ratio=0``, ridge
                can be fitted by coordinate descent or by setting
                ``opt_method='IRLS'``.
            debug: Whether to print optimizer diagnostics.
            family: GLM family name, class, or instance.
            link: Optional link name, class, or instance.
            exog_names: Optional exogenous column names.
            endog_name: Optional response name.
            fit_intercept: Whether to fit an intercept separately.
            normalize: Whether to normalize predictors for penalization.
            penalize_scale: Whether penalties are multiplied by estimated scale.
            use_t: Whether inference uses t critical values.
            test_level: Significance level for inference.
            residual_inclusion_order: Polynomial order for IV residual inclusion.
            force_iv_projection: Whether to force first-stage IV projection.
            compute_cov: Whether to compute covariance estimates.
            store_convergence_path: Whether to store per-iteration diagnostics.
            instrument_names: Optional instrument column names.
            residual_inclusion: Whether to include first-stage residuals.
            cov_kwds: Covariance/bootstrap options.
            cov_type: Covariance type.
            var_weights_name: Optional variance-weight label.
            line_search_fallback: Whether to use fallback line search.
            pick_default_start: Whether to choose family-based starting values.
            opt_method: Optimizer method override.
            first_column_constant: Whether the first design column is constant.
            prompt_user_for_more_iters: Whether/how to prompt after max iterations.

        Returns:
            ``SparseGLMRegressionResults``.

        Examples
        --------
        Logistic regression from numpy arrays (no formula parsing):

        >>> import numpy as np
        >>> from kanly.api import GLM
        >>> rng = np.random.default_rng(0)
        >>> n = 500
        >>> X = np.column_stack([np.ones(n),
        ...                      rng.normal(size=n),
        ...                      rng.normal(size=n)])
        >>> logits = X @ np.array([0.5, 1.0, -0.3])
        >>> y = (rng.uniform(size=n) < 1/(1+np.exp(-logits))).astype(float)
        >>> fit = GLM(y, X, family='binomial', fit_intercept=False,    # doctest: +SKIP
        ...           exog_names=['Intercept', 'x1', 'x2'])

        Poisson regression with an offset (passed via the design matrix):

        >>> y_counts = rng.poisson(np.exp(X @ np.array([0.2, 0.4, -0.1])))
        >>> fit_p = GLM(y_counts, X, family='poisson',                  # doctest: +SKIP
        ...             fit_intercept=False)

        Ridge pseudo-MLE using the IRLS path:

        >>> fit_ridge = GLM(y, X, family='binomial', alpha=0.1,         # doctest: +SKIP
        ...                 l1_ratio=0, opt_method='IRLS',
        ...                 fit_intercept=True, first_column_constant=True)

        See Also
        --------
        :meth:`glm` : formula entry point taking a Patsy-style formula.
        """
        return SparseGeneralizedLinearModel(
            exog, endog, add_constant, fit_intercept, has_implicit_constant=False, formula_design_info=None, is_gam=False,
            weights=var_weights, L2_penalty_matrix=L2_penalty_matrix, regularize_to_values=regularize_to_values,
            endog_name=endog_name, exog_names=exog_names, weights_name=var_weights_name, instruments=instruments,
            instrument_names=instrument_names, valid_obs_rows=None, index=None,
            null_rows_info_dict=None, model_elapsed=0,
        ).fit(family=family, link=link, start_params=start_params, tol=tol, max_iter=max_iter, alpha=alpha,
              l1_ratio=l1_ratio, debug=debug, normalize=normalize, penalize_scale=penalize_scale,
              use_t=use_t, test_level=test_level, compute_cov=compute_cov,
              residual_inclusion=residual_inclusion, cov_kwds=cov_kwds, cov_type=cov_type, fit_intercept=fit_intercept,
              line_search_fallback=line_search_fallback, pick_default_start=pick_default_start, opt_method=opt_method,
              first_column_constant=first_column_constant, store_convergence_path=store_convergence_path,
              residual_inclusion_order=residual_inclusion_order, force_iv_projection=force_iv_projection,
              prompt_user_for_more_iters=prompt_user_for_more_iters)

    @staticmethod
    def glm(formula, data, start_params=None, tol=DEFAULT_GLM_TOL, max_iter=DEFAULT_GLM_MAX_ITER, alpha=0.0,
            l1_ratio=0.0, L2_penalty_matrix=None, regularize_to_values=None,
            debug=False, family=DEFAULT_GLM_FAMILY, link=None, normalize=True, penalize_scale=False,
            use_t=True, test_level=DEFAULT_GLM_TEST_LEVEL, compute_cov=True, store_convergence_path=False,
            residual_inclusion=DEFAULT_GLM_RESIDUAL_INCLUSION, cov_kwds=None, cov_type=DEFAULT_GLM_COV_TYPE,
            line_search_fallback=True, pick_default_start=True, opt_method=None, index=None,
            specification_name=None, residual_inclusion_order=DEFAULT_GLM_RESIDUAL_INCLUSION_ORDER,
            prompt_user_for_more_iters=DEFAULT_GLM_PROMPT_USER_FOR_MORE_ITERS,
            force_iv_projection=DEFAULT_GLM_FORCE_IV_PROJECTION, check_constant_cols=DEFAULT_GLM_CHECK_CONSTANT_COLS):
        """
        CURRENTLY
            minimized -llf.mean() + nobs * alpha * (l1 + l2)

        Args:
            formula: GLM formula string parsed by ``SparseDataGetter``.
            data: pandas ``DataFrame`` or dict-like data.
            start_params: Optional starting coefficient vector or dict.
            tol: Convergence tolerance.
            max_iter: Maximum optimizer iterations.
            alpha: Elastic-net regularization strength.
            l1_ratio: L1 share of regularization. Pure ridge
                (``l1_ratio=0``) may use coordinate descent or IRLS.
            regularize_to_values: Optional scalar or length-``p`` target for a
                pure ridge penalty fitted with ``opt_method='IRLS'``. ``None``
                centers the penalty at zero.
            debug: Whether to print optimizer diagnostics.
            family: GLM family name, class, or instance.
            link: Optional link name, class, or instance.
            normalize: Whether to normalize predictors for penalization.
            penalize_scale: Whether penalties are multiplied by estimated scale.
            use_t: Whether inference uses t critical values.
            test_level: Significance level for inference.
            compute_cov: Whether to compute covariance estimates.
            store_convergence_path: Whether to store per-iteration diagnostics.
            residual_inclusion: Whether to include first-stage residuals for IV GLM.
            cov_kwds: Covariance/bootstrap options.
            cov_type: Covariance type.
            line_search_fallback: Whether to use fallback line search.
            pick_default_start: Whether to choose family-based starting values.
            opt_method: Optimizer method override.
            index: Optional row subset/indexer.
            specification_name: Optional result label.
            residual_inclusion_order: Polynomial order for residual inclusion.
            prompt_user_for_more_iters: Whether/how to prompt after max iterations.
            force_iv_projection: Whether to force first-stage IV projection.
            check_constant_cols: Whether to detect constant design columns.

        Returns:
            ``SparseGLMRegressionResults``.

        Examples
        --------
        Logistic regression:

        >>> import numpy as np, pandas as pd
        >>> from kanly.api import glm
        >>> rng = np.random.default_rng(0)
        >>> n = 1_000
        >>> df = pd.DataFrame({'x': rng.normal(size=n)})
        >>> df['y'] = (rng.uniform(size=n) <
        ...            1/(1+np.exp(-(0.4 + 0.9 * df['x'])))).astype(float)
        >>> fit = glm('y ~ x', df, family='binomial')      # doctest: +SKIP
        >>> print(fit.summary())                            # doctest: +SKIP

        Poisson regression with an exposure offset via ``np.log``:

        >>> df['exposure'] = rng.uniform(0.5, 2.0, n)
        >>> df['count']    = rng.poisson(df['exposure'] *
        ...                              np.exp(0.5 + 0.3 * df['x']))
        >>> fit_p = glm('count ~ x + offset(np.log(exposure))', df,  # doctest: +SKIP
        ...             family='poisson')

        Gaussian GLM with an identity link reproduces OLS coefficients:

        >>> df['ynum'] = 1.0 + 2.0 * df['x'] + rng.normal(size=n)
        >>> fit_g = glm('ynum ~ x', df, family='gaussian',  # doctest: +SKIP
        ...             link='identity')

        Negative binomial with elastic-net penalty and a cluster-bootstrap
        covariance grouped by ``firm``:

        >>> df['firm'] = rng.integers(0, 20, n)
        >>> df['count2'] = rng.negative_binomial(5, 0.5, n)
        >>> fit_nb = glm('count2 ~ x', df, family='negativebinomial', # doctest: +SKIP
        ...              alpha=0.05, l1_ratio=0.5,
        ...              cov_type='bootstrap',
        ...              cov_kwds={'n_samples': 500,
        ...                        'groups': 'firm', 'method': 'bayesian'})

        IV GLM (control-function / residual inclusion):

        >>> # fit_iv = glm('y ~ x | z1 + z2', df, family='binomial',
        >>> #              residual_inclusion=True)

        Supported families include ``'binomial'``, ``'poisson'``, ``'gaussian'``,
        ``'gamma'``, ``'inversegaussian'``, ``'negativebinomial'`` (case
        insensitive); links default to each family's canonical link.

        See Also
        --------
        :meth:`GLM` : matrix-form entry point taking numpy/sparse arrays.
        """

        cov_kwds = format_cov_kwds(cov_kwds)
        opt_method = _get_opt_method(opt_method, alpha, l1_ratio)

        cov_type = cov_type.upper()
        if cov_type not in GLM_COV_TYPES:
            raise Exception("`cov_type` must be one of %s!" % str(GLM_COV_TYPES))

        _check_penalties(alpha, l1_ratio)

        model = SparseGeneralizedLinearModel.build_model_from_formula(
            formula, data, debug=debug, index=index, specification_name=specification_name,
            check_constant_cols=check_constant_cols)

        fit = model.fit(
            family=family, link=link,
            start_params=start_params, tol=tol, max_iter=max_iter,
            alpha=alpha, l1_ratio=l1_ratio, L2_penalty_matrix=L2_penalty_matrix, debug=debug,
            regularize_to_values=regularize_to_values,
            normalize=normalize, penalize_scale=penalize_scale, use_t=use_t, test_level=test_level,
            compute_cov=compute_cov,
            store_convergence_path=store_convergence_path,
            residual_inclusion=residual_inclusion, cov_type=cov_type, cov_kwds=cov_kwds,
            line_search_fallback=line_search_fallback, pick_default_start=pick_default_start,
            opt_method=opt_method, fit_intercept=model.has_intercept, first_column_constant=model.has_intercept,
            specification_name=specification_name, residual_inclusion_order=residual_inclusion_order,
            force_iv_projection=force_iv_projection, prompt_user_for_more_iters=prompt_user_for_more_iters,
        )

        return fit

    def predict(self, params, data=None, index=None, debug=False, ignore_column_mismatch=False, *args, **kwargs):
        """Predict linear predictors or response-scale means.

        Args:
            params: Coefficient vector or dict/Series accepted by the base model.
            data: Optional new data for formula-based prediction.
            index: Optional row subset/indexer for new data.
            debug: Whether to print data parsing diagnostics.
            ignore_column_mismatch (bool): When ``True``, allow prediction when
                the new design has fewer columns than ``params`` (e.g. missing
                fixed-effect levels). See
                :meth:`~kanly.regression.linear_model_base.LinearModelBase.get_linear_predictor`.
            *args: Ignored; retained for API compatibility.
            **kwargs: If ``link`` is supplied, predictions are transformed by
                the inverse link onto the response scale.

        Returns:
            NumPy array of predictions.
        """
        y_hat = self.get_linear_predictor(params, data=data, index=index, debug=debug,
                                          ignore_column_mismatch=ignore_column_mismatch)

        if kwargs.get('link', None) is not None:
            link = _get_link(kwargs['link'])
            y_hat = link.inverse_link(y_hat)

        return y_hat

    def accepts_multi_outcome(self):
        """Return whether GLM supports multiple response columns.

        Returns:
            ``False``; this GLM implementation expects a single response.
        """
        return False
