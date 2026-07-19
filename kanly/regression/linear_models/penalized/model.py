from __future__ import absolute_import, print_function

import time
from typing import Iterable


import numpy as np
from pandas import DataFrame
from scipy.sparse import csc_matrix, hstack as sphstack

from kanly.formula.data_getter import (SparseDataGetter, EXOG_KEY, ENDOG_KEY, WEIGHTS_KEY, NULL_ROWS_INFO_DICT_KEY,
                                       VALID_OBS_ROWS_KEY, HAS_IMPLICIT_CONSTANT_KEY, HAS_INTERCEPT_KEY,
                                       parse_formula)
from kanly.formula.keys import FORMULA_DESIGN_INFO_KEY
from kanly.regression.linear_model_base import LinearModelBase
from kanly.regression.linear_models.constants import DEFAULT_LM_TEST_LEVEL, DEFAULT_LM_USE_T
from kanly.regression.linear_models.model import SparseLinearModel
from kanly.regression.linear_models.penalized import _check_penalties
from kanly.regression.linear_models.penalized.constants import (
    DEFAULT_EN_X_TOL, DEFAULT_EN_G_TOL, DEFAULT_EN_F_TOL, DEFAULT_EN_MAX_ITER, DEFAULT_EN_L1_RATIO,
    DEFAULT_EN_POSITIVE, DEFAULT_EN_ALPHA, DEFAULT_EN_NORMALIZE,
    DEFAULT_EN_APPLY_SCALING, DEFAULT_EN_ACTIVE_SET,
    DEFAULT_EN_FIT_INTERCEPT, DEFAULT_EN_PROMPT_USER_FOR_MORE_ITERS,
    DEFAULT_EN_SELECTION, DEFAULT_EN_RELAXATION_PARAMETER,
    DEFAULT_EN_ONE_DIM_SEARCH_CADENCE, DEFAULT_EN_ONE_DIM_SEARCH_MULTIPLIER, DEFAULT_EN_ONE_DIM_SEARCH_INIT_VAL,
    ELASTIC_NET, W_ELASTIC_NET, OLS, WLS, RIDGE, W_RIDGE, LASSO, W_LASSO
)
from kanly.regression.linear_models.penalized.sparse_elastic_net_internal import _elastic_net_internal
from kanly.regression.linear_models.penalized.regression_results import SparsePenalizedLinearRegressionResults
from kanly.sparse_data_frame import SparseDataFrame


class RefitRidgeWarning(Exception):
    """Raised when post-selection refitting is requested for a ridge-only model.

    Refitting without penalization is only meaningful when the L1 component can set
    coefficients exactly to zero and therefore perform variable selection."""
    pass


class SparsePenalizedLinearModel(LinearModelBase):
    """Sparse linear model wrapper for elastic-net estimation.

    The model stores parsed design/response data and exposes formula and array APIs
    that delegate to the sparse coordinate-descent solver, then wraps outputs in
    ``SparsePenalizedLinearRegressionResults``.

    Examples
    --------
    LASSO (``l1_ratio=1.0``) on a high-dimensional design:

    >>> import numpy as np, pandas as pd
    >>> from kanly.api import elastic_net
    >>> rng = np.random.default_rng(0)
    >>> n, p = 200, 20
    >>> X = rng.normal(size=(n, p))
    >>> beta_true = np.zeros(p); beta_true[:3] = [2.0, -1.5, 0.5]
    >>> y = X @ beta_true + 0.5 * rng.normal(size=n)
    >>> df = pd.DataFrame(X, columns=[f'x{j}' for j in range(p)])
    >>> df['y'] = y
    >>> fit = elastic_net(                                  # doctest: +SKIP
    ...     'y ~ ' + ' + '.join(f'x{j}' for j in range(p)),
    ...     df, alpha=0.1, l1_ratio=1.0)

    Ridge (``l1_ratio=0.0``) and elastic net (mid-mixing):

    >>> fit_ridge = elastic_net('y ~ x1 + x2 + x3', df,    # doctest: +SKIP
    ...                         alpha=0.5, l1_ratio=0.0)
    >>> fit_en    = elastic_net('y ~ x1 + x2 + x3', df,    # doctest: +SKIP
    ...                         alpha=0.1, l1_ratio=0.5)

    Variable-specific penalties via dict-valued ``alpha``:

    >>> fit_d = elastic_net('y ~ x1 + x2 + x3', df,        # doctest: +SKIP
    ...                     alpha={'x1': 0.0, 'x2': 0.2, 'x3': 0.5},
    ...                     l1_ratio=1.0)

    Aliases on ``kanly.api``: ``en``. Matrix form: ``EN_sk`` / ``ELASTIC_NET``.
    """

    def __init__(self, exog, endog, has_intercept, has_implicit_constant, formula_design_info, weights, endog_name,
                 exog_names, weights_name, model_elapsed, valid_obs_rows, index,
                 null_row_info_dict, specification_name):
        """Initialize a sparse penalized linear model from parsed design data.

        Args:
            exog: Sparse design matrix.
            endog: Response vector.
            has_intercept: Whether the formula/design includes an intercept.
            has_implicit_constant: Whether the design implicitly contains a constant.
            formula_design_info:
            weights: Optional observation weights.
            endog_name: Response variable name.
            exog_names: Design column names.
            weights_name: Optional weights column name.
            model_elapsed: Model construction elapsed time.
            valid_obs_rows: Row indices retained after parsing/drop-null processing.
            index: Optional original row indexer.
            null_row_info_dict: Metadata about rows dropped for null values.
            specification_name: Optional label for summaries."""
        super().__init__(
            endog, exog, False, has_intercept, has_implicit_constant, formula_design_info, weights=weights,
            endog_name=endog_name,
            exog_names=exog_names, weights_name=weights_name,
            index=index, valid_obs_rows=valid_obs_rows, null_rows_info_dict=null_row_info_dict,
            method='ElasticNet', specification_name=specification_name,
            model_elapsed=model_elapsed,
        )

    def  fit(self, alpha=DEFAULT_EN_ALPHA, l1_ratio=DEFAULT_EN_L1_RATIO, fit_intercept=DEFAULT_EN_FIT_INTERCEPT,
            normalize=DEFAULT_EN_NORMALIZE, max_iter=DEFAULT_EN_MAX_ITER,
            xtol=DEFAULT_EN_X_TOL, gtol=DEFAULT_EN_G_TOL, ftol=DEFAULT_EN_F_TOL,
            positive=DEFAULT_EN_POSITIVE, debug=False, active_set=DEFAULT_EN_ACTIVE_SET, use_t=True, cov_type=None,
            test_level=DEFAULT_LM_TEST_LEVEL, compute_cov=True, keep_model=True, cov_kwds=None,
            apply_scaling=DEFAULT_EN_APPLY_SCALING, prompt_user_for_more_iters=DEFAULT_EN_PROMPT_USER_FOR_MORE_ITERS,
            start_coef=None, start_intercept=None, selection=DEFAULT_EN_SELECTION, seed=0, penalty_intensities=None,
            override_weights=None, ssr_quad_form=None, regularize_to_values=None,
            relaxation_parameter=DEFAULT_EN_RELAXATION_PARAMETER,
            one_dim_search_cadence=DEFAULT_EN_ONE_DIM_SEARCH_CADENCE,
            one_dim_search_multiplier=DEFAULT_EN_ONE_DIM_SEARCH_MULTIPLIER,
            one_dim_search_init_value=DEFAULT_EN_ONE_DIM_SEARCH_INIT_VAL,
            specification_name=None
            ):
        """Fit elastic-net coefficients for this parsed linear model.

        Args:
            alpha: Scalar, vector, or dict of total penalty strengths.
            l1_ratio: Scalar, vector, or dict mixing L1 versus L2 penalty.
            fit_intercept: Whether to estimate an unpenalized intercept.
            normalize: When True (default), scale each ``alpha`` by the column
                standard deviation so penalties are comparable across predictors
                on different scales (consistent with ``StandardScaler``).
            max_iter: Maximum coordinate-descent iterations.
            xtol: Coefficient-change convergence tolerance.
            gtol: Subgradient convergence tolerance.
            ftol: Relative objective_function-change convergence tolerance.
            positive: Boolean/vector/dict non-negativity constraints on coefficients.
            debug: Whether to print solver progress.
            active_set: Whether to restrict updates to recently changing coordinates.
            use_t: Reserved for compatibility with regression results APIs.
            cov_type: Reserved; penalized inference is not computed directly.
            test_level: Reserved test level for result compatibility.
            compute_cov: Reserved covariance flag for result compatibility.
            keep_model: Whether to retain this model on the result object.
            cov_kwds: Reserved covariance options.
            apply_scaling: Legacy optional coefficient rescaling after fit.
            prompt_user_for_more_iters: Whether/how to ask for more iterations.
            start_coef: Optional starting coefficients.
            start_intercept: Optional starting intercept.
            selection: Coordinate selection strategy: cyclic, random, or greedy.
            seed: Random seed for random coordinate selection.
            penalty_intensities: Optional descending penalty scales for warm-start path.
            override_weights: Optional weights replacing model weights for this fit.
            ssr_quad_form: Optional precomputed SSR quadratic form.
            regularize_to_values: Optional coefficient targets for penalties.
            relaxation_parameter: Optional second-pass penalty relaxation factor.
            one_dim_search_cadence: Optional cadence for full-direction line search.
            one_dim_search_multiplier: Expansion factor for full-direction search.
            one_dim_search_init_value: Initial full-direction search step.
            specification_name: Optional result label override.

        Returns:
            ``SparsePenalizedLinearRegressionResults``."""

        if isinstance(alpha, dict):
            # Dict inputs let callers target named coefficients while the solver
            # works with dense vectors aligned to the parsed sparse design.
            alpha = self.dict_2_vector(alpha, self)
        if isinstance(l1_ratio, dict):
            l1_ratio = self.dict_2_vector(l1_ratio, self)
        if isinstance(positive, dict):
            positive = self.dict_2_vector(positive, self).astype(bool)
        if isinstance(regularize_to_values, dict):
            regularize_to_values = self.dict_2_vector(regularize_to_values, self)

        _check_penalties(alpha, l1_ratio)

        fit_dict, params, cost = _elastic_net_internal(
            self.exog, self.endog, fit_intercept=fit_intercept, normalize=normalize, alpha=alpha,
            l1_ratio=l1_ratio, max_iter=max_iter, xtol=xtol, ftol=ftol, gtol=gtol,
            positive=positive,
            weights=self.weights if override_weights is None else override_weights,
            debug=debug,
            active_set=active_set,  prompt_user_for_more_iters=prompt_user_for_more_iters,
            # use_t=use_t, cov_type=cov_type, cov_kwds=cov_kwds, test_level=test_level,
            # compute_cov=compute_cov, endog_name=self.endog_name, exog_names=self.exog_names,
            #exog_term_names=self.exog_term_names,
            apply_scaling=apply_scaling,
            start_coef=start_coef, start_intercept=start_intercept, selection=selection, seed=seed,
            penalty_intensities=penalty_intensities, ssr_quad_form=ssr_quad_form,
            regularize_to_values=regularize_to_values, relaxation_parameter=relaxation_parameter,
            one_dim_search_cadence=one_dim_search_cadence,
            one_dim_search_multiplier=one_dim_search_multiplier,
            one_dim_search_init_value=one_dim_search_init_value,
        )

        method = self._get_method(alpha, l1_ratio, self.is_weighted)

        return SparsePenalizedLinearRegressionResults(
            self.nobs, params, self, method, fit_dict['fittedvalues'], fit_dict['resid'],
            fit_dict['positive'], fit_dict['normalize'],
            fit_dict['fit_intercept'], l1_ratio, alpha,
            fit_dict['l1_penalties'], fit_dict['l2_penalties'],
            apply_scaling, fit_dict['fit_time'], fit_dict['rsquared'],
            fit_dict['coef_'], fit_dict['intercept_'], fit_dict['iters'], fit_dict['converged'],
            fit_dict['x_error'], fit_dict['f_error'], fit_dict['g_error'], fit_dict['message'],
            cost, fit_dict['ssr'], fit_dict['penalty'], fit_dict['objective_function_'],
            fit_dict['objective_function'], fit_dict['solver_settings'], relaxation_parameter,
            specification_name=self.specification_name if specification_name is None else specification_name,
            keep_model=keep_model,
        )

    @staticmethod
    def _get_method(alpha, l1_ratio, is_weighted):
        """Choose a human-readable method label from penalties and weighting.

        Args:
            alpha: Penalty strength(s).
            l1_ratio: Elastic-net mixing value(s).
            is_weighted: Whether the model used observation weights.

        Returns:
            Method string such as LASSO, RIDGE, OLS, or WTD ELASTIC NET."""
        if np.all(np.array(alpha) == 0):
            return WLS if is_weighted else OLS
        else:
            if np.all(np.array(l1_ratio) == 1):
                return W_LASSO if is_weighted else LASSO
            elif np.all(np.array(l1_ratio) == 0):
                return W_RIDGE if is_weighted else RIDGE
            else:
                return W_ELASTIC_NET if is_weighted else ELASTIC_NET

    @staticmethod
    def build_model_from_formula(formula, data, debug=False, index=None, specification_name=None,
                                 check_constant_cols=False, drop_1_for_FE=False):
        """Parse formula/data into a ``SparsePenalizedLinearModel``.

        Args:
            formula: Linear model formula accepted by the sparse formula parser.
            data: DataFrame, SparseDataFrame, or dict-like data source.
            debug: Whether to print parsing diagnostics.
            index: Optional row subset/indexer.
            specification_name: Optional model label.
            check_constant_cols: Whether to check for constant design columns.
            drop_1_for_FE: Whether to drop a baseline level for fixed effects.

        Returns:
            Parsed ``SparsePenalizedLinearModel`` with sparse design matrix."""

        _t = time.time()

        if isinstance(data, dict):
            data = DataFrame(data, copy=False)

        result = parse_formula(formula)
        endog_name, exog_term_names = result['ENDOG'], result['EXOG']
        if '-1' not in exog_term_names:
            exog_term_names.append('-1')

        # Always drop the formula intercept from X; the solver handles the
        # intercept separately as an unpenalized scalar parameter.
        original_formula = formula
        formula = formula.replace('~', ' ~ -1 + ')

        data_obj = SparseDataGetter.get_data(data, formula, index=index, debug=debug,
                                             fail_on_iv=True, fail_on_absorb=True,
                                             check_constant_cols=check_constant_cols,
                                             drop_1_for_FE=drop_1_for_FE)
        exog, exog_names = data_obj[EXOG_KEY].values, data_obj[EXOG_KEY].column_names
        endog, endog_name = data_obj[ENDOG_KEY].values, data_obj[ENDOG_KEY].column_names[0]
        if data_obj[WEIGHTS_KEY] is not None:
            weights, weights_name = data_obj[WEIGHTS_KEY].values, data_obj[WEIGHTS_KEY].column_names[0]
        else:
            weights, weights_name = None, None
        null_row_info_dict = data_obj[NULL_ROWS_INFO_DICT_KEY]
        valid_obs_rows = data_obj[VALID_OBS_ROWS_KEY]
        has_intercept, has_implicit_constant = data_obj[HAS_INTERCEPT_KEY], data_obj[HAS_IMPLICIT_CONSTANT_KEY]
        fdi = data_obj[FORMULA_DESIGN_INFO_KEY]

        model_elapsed = time.time() - _t

        model = SparsePenalizedLinearModel(
            exog, endog, has_intercept, has_implicit_constant,
            fdi, # TODO originally had "original formula here...
            weights, endog_name, exog_names,
            weights_name, model_elapsed, valid_obs_rows, index, null_row_info_dict,
            specification_name
        )

        if debug:
            print(model)

        return model

    def predict(self, intercept, coef, data=None, index=None, debug=False, ignore_column_mismatch=False,
                *args, **kwargs):
        """Predict responses from intercept and coefficients.

        Args:
            intercept: Intercept value to add to the linear predictor.
            coef: Coefficient vector/series aligned to model columns.
            data: Optional new data for formula-based prediction.
            index: Optional row subset for new data.
            debug: Whether to print parsing diagnostics.
            ignore_column_mismatch (bool): When ``True``, allow prediction when
                the new design has fewer columns than ``coef`` (e.g. missing
                fixed-effect levels). See
                :meth:`~kanly.regression.linear_model_base.LinearModelBase.get_linear_predictor`.
            *args: Ignored compatibility arguments.
            **kwargs: Ignored compatibility keyword arguments.

        Returns:
            Predicted response vector."""
        y_hat = self.get_linear_predictor(coef, data=data, debug=debug, index=index, ignore_column_mismatch=ignore_column_mismatch)
        return y_hat + intercept

    def accepts_multi_outcome(self):
        """Return whether this model accepts multiple response columns.

        Returns:
            ``False``; elastic-net linear models expect a single response."""
        return False

    # @staticmethod
    # def lasso(
    #         formula, data, alpha=1.0, fit_intercept=True, normalize=False,
    #         max_iter=200, tol=1e-4, positive=False, specification_name=None, weights=None, debug=False,
    #         pause_coeff_updates=False, sample_updates=0, seed=0, order_list=False, refit=False,
    #         use_t=DEFAULT_LM_USE_T, cov_type=None, cov_kwds=dict(), test_level=DEFAULT_LM_TEST_LEVEL,
    #         prompt_user_for_more_iters=DEFAULT_EN_PROMPT_USER_FOR_MORE_ITERS, compute_cov=True,
    #         selection=DEFAULT_EN_SELECTION
    #     ):
    #     return SparsePenalizedLinearModel.elastic_net(
    #         formula, data, alpha=alpha, l1_ratio=1, fit_intercept=fit_intercept, normalize=normalize, max_iter=max_iter,
    #         tol=tol, positive=positive, specification_name=specification_name, weights=weights, debug=debug,
    #         pause_coeff_updates=pause_coeff_updates, sample_updates=sample_updates, seed=seed, order_list=order_list,
    #         refit=refit, use_t=use_t, cov_type=cov_type, cov_kwds=cov_kwds, test_level=test_level,
    #         compute_cov=compute_cov, prompt_user_for_more_iters=prompt_user_for_more_iters, selection=selection)
    #
    # @staticmethod
    # def ridge(
    #         formula, data, alpha=1.0, fit_intercept=True, normalize=False,
    #         max_iter=200, tol=1e-4, positive=False, specification_name=None, weights=None, debug=False,
    #         pause_coeff_updates=False, sample_updates=0, seed=0, order_list=False, refit=False,
    #         use_t=DEFAULT_LM_USE_T, cov_type=None, cov_kwds=dict(), test_level=DEFAULT_LM_TEST_LEVEL,
    #         prompt_user_for_more_iters=DEFAULT_EN_PROMPT_USER_FOR_MORE_ITERS, compute_cov=True,
    #         selection=DEFAULT_EN_SELECTION
    # ):
    #     return SparsePenalizedLinearModel.elastic_net(
    #         formula, data, alpha=alpha, l1_ratio=0.0, fit_intercept=fit_intercept, normalize=normalize,
    #         max_iter=max_iter, tol=tol, positive=positive, specification_name=specification_name, weights=weights, debug=debug,
    #         pause_coeff_updates=pause_coeff_updates, sample_updates=sample_updates, seed=seed, order_list=order_list,
    #         refit=refit, use_t=use_t, cov_type=cov_type, cov_kwds=cov_kwds, test_level=test_level,
    #         compute_cov=compute_cov, prompt_user_for_more_iters=prompt_user_for_more_iters, selection=selection)

    @staticmethod
    def ELASTIC_NET(
            endog, exog, weights=None, fit_intercept=DEFAULT_EN_FIT_INTERCEPT, normalize=DEFAULT_EN_NORMALIZE,
            alpha=DEFAULT_EN_ALPHA, l1_ratio=DEFAULT_EN_L1_RATIO,
            max_iter=DEFAULT_EN_MAX_ITER, xtol=DEFAULT_EN_X_TOL, gtol=DEFAULT_EN_G_TOL, ftol=DEFAULT_EN_F_TOL,
            positive=DEFAULT_EN_POSITIVE, specification_name=None,
            debug=False, active_set=DEFAULT_EN_ACTIVE_SET, endog_name=None, exog_names=None, weights_name=None,
            use_t=DEFAULT_LM_USE_T, cov_type=None, cov_kwds=None, test_level=DEFAULT_LM_TEST_LEVEL,
            compute_cov=True, keep_model=True, prompt_user_for_more_iters=DEFAULT_EN_PROMPT_USER_FOR_MORE_ITERS,
            selection=DEFAULT_EN_SELECTION, seed=0, regularize_to_values=None,
            one_dim_search_cadence=DEFAULT_EN_ONE_DIM_SEARCH_CADENCE,
            one_dim_search_multiplier=DEFAULT_EN_ONE_DIM_SEARCH_MULTIPLIER,
            one_dim_search_init_value=DEFAULT_EN_ONE_DIM_SEARCH_INIT_VAL,
    ):
        """Fit elastic net from array inputs.

        Args:
            endog: Response vector.
            exog: Design matrix.
            weights: Optional observation weights.
            fit_intercept: Whether to estimate an intercept.
            normalize: When True (default), scale penalties by column std dev.
            alpha: Penalty strength(s).
            l1_ratio: L1/L2 mixing value(s).
            max_iter: Maximum coordinate-descent iterations.
            xtol: Coefficient-change tolerance.
            gtol: Subgradient tolerance.
            ftol: Objective-change tolerance.
            positive: Non-negativity constraints.
            specification_name: Optional result label.
            debug: Whether to print progress.
            active_set: Whether to use active-set updates.
            endog_name: Optional response name.
            exog_names: Optional coefficient names.
            weights_name: Optional weights name.
            use_t: Compatibility flag.
            cov_type: Reserved covariance type.
            cov_kwds: Reserved covariance options.
            test_level: Compatibility test level.
            compute_cov: Reserved covariance flag.
            keep_model: Whether to retain the model on the result.
            prompt_user_for_more_iters: Whether/how to ask for more iterations.
            selection: Coordinate selection strategy.
            seed: Random seed.
            regularize_to_values: Optional coefficient targets.
            one_dim_search_cadence: Optional full-direction search cadence.
            one_dim_search_multiplier: Full-direction search expansion factor.
            one_dim_search_init_value: Initial full-direction search step.

        Returns:
            ``SparsePenalizedLinearRegressionResults``.

        Examples
        --------
        LASSO directly on numpy arrays:

        >>> import numpy as np
        >>> from kanly.api import ELASTIC_NET
        >>> rng = np.random.default_rng(0)
        >>> n, p = 200, 20
        >>> X = rng.normal(size=(n, p))
        >>> beta = np.zeros(p); beta[:3] = [2.0, -1.5, 0.5]
        >>> y = X @ beta + 0.5 * rng.normal(size=n)
        >>> fit = ELASTIC_NET(y, X, alpha=0.1, l1_ratio=1.0,    # doctest: +SKIP
        ...                   exog_names=[f'x{j}' for j in range(p)])

        Aliases on ``kanly.api``: ``EN_sk``.
        """

        model = SparsePenalizedLinearModel(
            exog, endog, fit_intercept, False, None, weights, endog_name,
            exog_names, weights_name, 0, None, None, dict(),
            specification_name
        )
        return model.fit(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, normalize=normalize,
                         max_iter=max_iter, xtol=xtol, ftol=ftol, gtol=gtol, positive=positive, debug=debug,
                         active_set=active_set, prompt_user_for_more_iters=prompt_user_for_more_iters,
                         use_t=use_t, cov_type=cov_type, cov_kwds=cov_kwds, test_level=test_level,
                         compute_cov=compute_cov, keep_model=keep_model, selection=selection, seed=seed,
                         regularize_to_values=regularize_to_values,
                         one_dim_search_cadence=one_dim_search_cadence,
                         one_dim_search_multiplier=one_dim_search_multiplier,
                         one_dim_search_init_value=one_dim_search_init_value,
                         )

    @staticmethod
    def elastic_net(
            formula: str,
            data: [DataFrame, SparseDataFrame, dict],
            alpha: [float, dict, Iterable[float]] = DEFAULT_EN_ALPHA,
            l1_ratio: [float, dict, Iterable[float]] = DEFAULT_EN_L1_RATIO,
            fit_intercept: bool = DEFAULT_EN_FIT_INTERCEPT,
            normalize: bool = DEFAULT_EN_NORMALIZE,
            apply_scaling: bool = DEFAULT_EN_APPLY_SCALING,
            max_iter: int = DEFAULT_EN_MAX_ITER,
            xtol: float = DEFAULT_EN_X_TOL,
            gtol: float = DEFAULT_EN_G_TOL,
            ftol: float = DEFAULT_EN_F_TOL,
            positive: [float, dict, Iterable[float]] = DEFAULT_EN_POSITIVE,
            specification_name: [None, str] = None,
            active_set: bool = DEFAULT_EN_ACTIVE_SET,
            prompt_user_for_more_iters: bool = DEFAULT_EN_PROMPT_USER_FOR_MORE_ITERS,
            debug: bool = False,
            refit: bool = False,
            use_t: bool = DEFAULT_LM_USE_T,
            cov_type: str = None,
            cov_kwds: dict = None,
            test_level: float = DEFAULT_LM_TEST_LEVEL,
            compute_cov: bool = True,
            index: Iterable = None,
            selection: str = DEFAULT_EN_SELECTION,
            seed: int = 0,
            penalty_intensities=None,
            regularize_to_values: [None, dict, Iterable[float]] = None,
            relaxation_parameter: float = DEFAULT_EN_RELAXATION_PARAMETER,
            one_dim_search_cadence: int = DEFAULT_EN_ONE_DIM_SEARCH_CADENCE,
            one_dim_search_multiplier: float = DEFAULT_EN_ONE_DIM_SEARCH_MULTIPLIER,
            one_dim_search_init_value: float = DEFAULT_EN_ONE_DIM_SEARCH_INIT_VAL,
    ):
        """Fit elastic net from a formula and data.

        Args:
            formula: Linear model formula parsed into sparse design matrices.
            data: DataFrame, SparseDataFrame, or dict-like data source.
            alpha: Penalty strength(s), optionally keyed by parameter name.
            l1_ratio: L1/L2 mixing value(s), optionally keyed by parameter name.
            fit_intercept: Whether to estimate an intercept.
            normalize: When True (default), scale penalties by column std dev.
            apply_scaling: Legacy optional coefficient rescaling after fit.
            max_iter: Maximum coordinate-descent iterations.
            xtol: Coefficient-change tolerance.
            gtol: Subgradient tolerance.
            ftol: Objective-change tolerance.
            positive: Non-negativity constraints, optionally keyed by name.
            specification_name: Optional result label.
            active_set: Whether to update only active coordinates after a full pass.
            prompt_user_for_more_iters: Whether/how to ask for more iterations.
            debug: Whether to print parsing and solver diagnostics.
            refit: Whether to run post-selection OLS on nonzero selected variables.
            use_t: Forwarded to optional OLS refit.
            cov_type: Covariance type for optional OLS refit.
            cov_kwds: Covariance options for optional OLS refit.
            test_level: Test level for optional OLS refit.
            compute_cov: Whether optional OLS refit computes covariance.
            index: Optional row subset/indexer.
            selection: Coordinate selection strategy.
            seed: Random seed for random selection.
            penalty_intensities: Optional warm-start sequence of penalty scales.
            regularize_to_values: Optional coefficient target values.
            relaxation_parameter: Optional second-pass penalty relaxation factor.
            one_dim_search_cadence: Optional full-direction search cadence.
            one_dim_search_multiplier: Full-direction search expansion factor.
            one_dim_search_init_value: Initial full-direction search step.

        Returns:
            Penalized fit, or ``(penalized_fit, ols_refit)`` when ``refit=True``.

        Examples
        --------
        LASSO (``l1_ratio=1``) with sparse formula support:

        >>> import numpy as np, pandas as pd
        >>> from kanly.api import elastic_net
        >>> rng = np.random.default_rng(0)
        >>> n, p = 200, 20
        >>> X = rng.normal(size=(n, p))
        >>> beta = np.zeros(p); beta[:3] = [2.0, -1.5, 0.5]
        >>> y = X @ beta + 0.5 * rng.normal(size=n)
        >>> df = pd.DataFrame(X, columns=[f'x{j}' for j in range(p)])
        >>> df['y'] = y
        >>> formula = 'y ~ ' + ' + '.join(df.columns[:-1])
        >>> fit_lasso = elastic_net(formula, df,            # doctest: +SKIP
        ...                         alpha=0.1, l1_ratio=1.0)

        Ridge (``l1_ratio=0``) or elastic net (mid-mixing):

        >>> fit_ridge = elastic_net(formula, df,            # doctest: +SKIP
        ...                         alpha=0.5, l1_ratio=0.0)
        >>> fit_en    = elastic_net(formula, df,            # doctest: +SKIP
        ...                         alpha=0.1, l1_ratio=0.5)

        Post-selection OLS refit on the LASSO-selected support (only valid
        for ``l1_ratio > 0``):

        >>> lasso_fit, refit = elastic_net(formula, df,    # doctest: +SKIP
        ...                                alpha=0.1, l1_ratio=1.0,
        ...                                refit=True)

        Coefficient-specific penalties, positivity, and shrinkage targets:

        >>> fit = elastic_net(formula, df,                  # doctest: +SKIP
        ...                   alpha={'x0': 0.0, 'x1': 0.05},
        ...                   positive={'x2': True},
        ...                   regularize_to_values={'x1': 0.0})

        See Also
        --------
        :meth:`ELASTIC_NET` : matrix-form entry point. Alias on ``kanly.api``: ``en``.
        """

        _t = time.time()

        if refit and l1_ratio == 0:
            raise RefitRidgeWarning(
                "Can only set `refit=True` for `l1_ratio` > 0.  "
                "Unpenalized refitting makes no sense without variable selection")

        if debug:
            print("Building design matrices with sparse patsy...\n\n", end='')

        model = SparsePenalizedLinearModel.build_model_from_formula(
                formula, data, debug=debug, index=index, specification_name=specification_name)

        if debug:

            def get_dims(x):
                """Summarize sparse design matrix shape and density for debug output.

                Args:
                    x: Sparse design matrix.

                Returns:
                    Tuple ``(n_rows, n_cols, nnz, percent_nonzero)``."""
                return (x.shape[0], x.shape[1], x.nnz,
                        100.0 * x.nnz / (x.shape[0] * x.shape[1]))

            dim_string = "exog matrix is (%d x %d) with %d non-zero entries (%.2f%%)" % get_dims(model.exog)

            print("\n" + dim_string)
            print("\n...Sparse Patsy complete! (%.3f s)\n\n" % (time.time() - _t))

        fit = model.fit(
            fit_intercept=fit_intercept, normalize=normalize, alpha=alpha,
            l1_ratio=l1_ratio, max_iter=max_iter, xtol=xtol, ftol=ftol, gtol=gtol, positive=positive, debug=debug,
            active_set=active_set, use_t=use_t, cov_type=cov_type, cov_kwds=cov_kwds, test_level=test_level,
            compute_cov=compute_cov, apply_scaling=apply_scaling, prompt_user_for_more_iters=prompt_user_for_more_iters,
            selection=selection, seed=seed, penalty_intensities=penalty_intensities,
            regularize_to_values=regularize_to_values, relaxation_parameter=relaxation_parameter,
            one_dim_search_cadence=one_dim_search_cadence,
            one_dim_search_multiplier=one_dim_search_multiplier,
            one_dim_search_init_value=one_dim_search_init_value
        )

        if not refit:
            return fit
        else:
            # Post-selection refit removes elastic-net shrinkage by running OLS
            # on variables whose penalized coefficients are nonzero.
            fit_ols = SparsePenalizedLinearModel.refit_model_without_penalization(
                fit, model, fit_intercept, cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t,
                specification_name=specification_name, debug=debug, compute_cov=compute_cov)
            return fit, fit_ols

    @staticmethod
    def dict_2_vector(dict_instance, model):
        """Convert parameter-keyed dict input to a coefficient vector aligned to model columns.

        Args:
            dict_instance: Dict mapping coefficient names to values.
            model: ``SparsePenalizedLinearModel`` whose ``exog_names`` define ordering.

        Returns:
            Dense vector of length ``model.exog.shape[1]`` with missing entries set to 0."""
        vec = np.zeros(model.exog.shape[1])
        var_2_idx = dict(zip(model.exog_names, range(model.exog.shape[1])))
        for nm, val in dict_instance.items():
            if nm in var_2_idx:
                vec[var_2_idx[nm]] = val
        return vec

    @staticmethod
    def refit_model_without_penalization(fit, model, fit_intercept, cov_type='hc1', cov_kwds=None, use_t=True,
                                         specification_name=None, debug=False, compute_cov=True):
        """Refit OLS on the support selected by an elastic-net fit.

        Args:
            fit: Penalized fit whose nonzero coefficients define the selected support.
            model: Original penalized model with full design matrix.
            fit_intercept: Whether the penalized model included an intercept.
            cov_type: Covariance type for the OLS refit.
            cov_kwds: Covariance options for the OLS refit.
            use_t: Whether the OLS refit uses t-based inference.
            specification_name: Optional refit label.
            debug: Whether to print refit diagnostics.
            compute_cov: Whether to compute covariance for the OLS refit.

        Returns:
            ``SparseLinearModel.LM`` result fit on selected variables only."""
        variables_selected = fit.params.values != 0
        exog_names_selected = fit.params.index[variables_selected]
        if fit_intercept:
            # The intercept lives in params but not in model.exog, so skip it in
            # the selected design and prepend an explicit constant column.
            exog_selected = model.exog[:, variables_selected[1:]]
            exog_selected = sphstack((csc_matrix(np.ones((model.nobs, 1))), exog_selected))
        else:
            exog_selected = model.exog[:, variables_selected]

        fit_ols = SparseLinearModel.LM(
            model.endog, exog_selected, weights=model.weights, endog_name=model.endog_name,
            exog_names=exog_names_selected, exog_term_names=model.exog_term_names, weights_name=model.weights_name,
            debug=debug, specification_name=specification_name, use_t=use_t, cov_type=cov_type, cov_kwds=cov_kwds,
            test_level=DEFAULT_LM_TEST_LEVEL, compute_cov=compute_cov, keep_model=True,
            #valid_obs_rows=model.valid_obs_rows, null_rows_info_dict=model.null_rows_info_dict
        )

        return fit_ols

