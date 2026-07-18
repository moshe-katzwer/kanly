from __future__ import absolute_import, print_function

import time

import numpy as np

from kanly.optimize.bfgs_bounded_quasi_newton import bfgs_pqn
from kanly.stats.distributions.nopython_logpdf import logpdf_norm
from kanly.time_series.sarimax.constants import \
    DEFAULT_SARIMAX_STEADY_STATE_TOL, DEFAULT_SARIMAX_ENFORCE_STATIONARITY, \
    DEFAULT_SARIMAX_ENFORCE_INVERTIBILITY, DEFAULT_SARIMAX_CONCENTRATE_SCALE, DEFAULT_SARIMAX_SIMPLE_DIFFERENCING, \
    DEFAULT_SARIMAX_INITIAL_DIFFUSE_VARIANCE, DEFAULT_SARIMAX_MULTIPLICATIVE, DEFAULT_SARIMAX_DIFFUSE, \
    DEFAULT_SARIMAX_STANDARDIZE_ENDOG
from kanly.time_series.sarimax.difference import difference
from kanly.time_series.sarimax.sarimax_internal_helper_functions import get_sarimax_start_params, none_tuple, \
    split_params, combine_seasonal_lag_params_into_one_vector, residualize, scale_params, format_trend_scale
from kanly.time_series.sarimax.state_space_arima import get_arima_statespace_matrices, do_arima_kalman_recursion
from kanly.time_series.sarimax.state_space_arma import get_arma_statespace_matrices, do_arma_kalman_recursion
from kanly.time_series.sarimax.validate_sarima_orders import validate_orders
from kanly.utils.print_options import print_options


def loglike_internal(endog, arparams, maparams,
                     exog=None, exogparams=None,
                     trend=None, trendparams=None,
                     trend_offset=1, trend_scale=1.0,
                     steady_state_tol=DEFAULT_SARIMAX_STEADY_STATE_TOL,
                     d=0, D=0, s=2,
                     sigma2=1.0,
                     simple_differencing=DEFAULT_SARIMAX_SIMPLE_DIFFERENCING,
                     initial_diffuse_variance=DEFAULT_SARIMAX_INITIAL_DIFFUSE_VARIANCE,
                     return_type='llf',
                     diffuse=DEFAULT_SARIMAX_DIFFUSE,
                     ):

    # Remove exog and trend
    """
    Evaluate the SARIMAX log-likelihood for given ARMA parameters.

    This is the innermost computational kernel called on every optimizer
    iteration.  It performs the following steps in order:

    1. **Residualize**: subtract exogenous (``exog @ exogparams``) and
       deterministic trend contributions from ``endog``.
    2. **Differencing strategy**: if ``simple_differencing=True``, apply the
       combined ``(1-L)^d * (1-L^s)^D`` filter to the residualized series
       and use the plain ARMA Kalman path.  Otherwise use the augmented ARIMA
       state-space that carries the integration in-state so the full
       undifferenced series length is retained.
    3. **Kalman recursion**: run either ``do_arma_kalman_recursion`` or
       ``do_arima_kalman_recursion`` to obtain one-step-ahead residuals
       ``v_k`` and forecast error variances ``S_k``.
    4. **Log-likelihood**: evaluate ``sum_k log N(v_k; 0, S_k)``.  In the
       integrated path the first ``d + D*s`` observations are burned.

    Args:
        endog: Observed endogenous time-series values.
        arparams: Autoregressive coefficients ordered by active AR lags.
        maparams: Moving-average coefficients ordered by active MA lags.
        exog: Optional exogenous regressor matrix aligned to ``endog``.
        exogparams: Coefficient vector for exogenous regressors.
        trend: Trend specification, such as None, ``c``, ``t``, ``ct``, ``n``, or an indicator list.
        trendparams: Coefficient vector for deterministic trend sparse_terms.
        trend_offset: Starting offset for deterministic trend time indices.
        trend_scale: Scale used to normalize deterministic trend time indices.
        steady_state_tol: Tolerance for detecting steady-state Kalman forecast error variance.
        d: Nonseasonal differencing order.
        D: Seasonal differencing order.
        s: Seasonal period length.
        sigma2: Innovation variance parameter.
        simple_differencing: Whether to difference data before the state-space likelihood instead of integrating in the state.
        initial_diffuse_variance: Initial variance assigned to diffuse state components.
        return_type: ``'llf'`` returns a scalar total log-likelihood; any
            other value returns a dict with keys ``llf``, ``llf_obs``,
            ``resid``, ``fittedinnovations``, ``state_vec``,
            ``forecasts_error_cov``, ``steady_state_iter``,
            ``exog_dot_beta``, ``trend_dot_beta``.
        diffuse: Whether to use diffuse initialization for integrated state components.

    Returns:
        Scalar total log-likelihood when ``return_type='llf'``, or a
        dictionary of Kalman filter internals otherwise.
    """
    endog_residual, exog_dot_beta, trend_dot_beta = residualize(
        endog, exog, trend, exogparams, trendparams, trend_offset, trend_scale)
    if simple_differencing:
        # Simple differencing shortens the series before Kalman filtering; the
        # result is padded back to the original length after fitting.
        endog_residual = difference(endog_residual, d, D, s)

    # Form state space for Kalman recursion
    llb = d + s * D
    if simple_differencing or llb == 0:
        transition, observation, var_innovation, unconditional_state_variance = \
            get_arma_statespace_matrices(arparams, maparams, sigma2)
        difference_coefs = None
        kalman_recursion_function = do_arma_kalman_recursion

    else:
        transition, observation, var_innovation, difference_coefs, unconditional_state_variance = \
            get_arima_statespace_matrices(arparams, maparams, d=d, D=D, s=s, sigma2=sigma2)
        kalman_recursion_function = do_arima_kalman_recursion

    # Do Kalman recursion
    resid, fittedinnovations, forecasts_error_cov, state_vec, steady_state_iter = \
        kalman_recursion_function(
            endog_residual, d, D, s, unconditional_state_variance, transition, observation,
            var_innovation, difference_coefs, steady_state_tol, initial_diffuse_variance, diffuse)

    llf_obs = logpdf_norm(resid, 0, np.sqrt(forecasts_error_cov))
    if simple_differencing:
        llf = llf_obs.sum()
    else:
        # Integrated state-space likelihood keeps the original series length,
        # so burn the observations needed to initialize differencing lags.
        llf = llf_obs[d + s * D:].sum()

    if return_type == 'llf':
        return llf
    else:
        return dict(
            llf=llf, llf_obs=llf_obs, resid=resid,
            fittedinnovations=fittedinnovations,
            state_vec=state_vec,
            forecasts_error_cov=forecasts_error_cov,
            steady_state_iter=steady_state_iter,
            exog_dot_beta=exog_dot_beta, trend_dot_beta=trend_dot_beta,
        )


def get_loglike_function(endog, exog, trend, trend_offset, trend_scale,
                         d, D, s, k_trend, k_exog, k_ar, k_ma, k_sar, k_sma,
                         ar_lags, ma_lags, sar_lags, sma_lags,
                         simple_differencing, initial_diffuse_variance,
                         diffuse
                         ):

    """
    Create a log-likelihood callable bound to fixed model structure and data.

    Captures all model-structure arguments (``endog``, ``exog``, ``trend``,
    order metadata, lag arrays, differencing strategy) via closure and returns
    a single-argument function ``loglike(params, ...)`` that the optimizer can
    call directly.  Inside that closure the packed parameter vector is unpacked
    via ``split_params``, the nonseasonal and seasonal AR/MA vectors are
    merged into full-length lag arrays via
    ``combine_seasonal_lag_params_into_one_vector``, and ``loglike_internal``
    is invoked for the actual Kalman evaluation.

    Args:
        endog: Observed endogenous time-series values.
        exog: Optional exogenous regressor matrix aligned to ``endog``.
        trend: Trend specification, such as None, ``c``, ``t``, ``ct``, ``n``, or an indicator list.
        trend_offset: Starting offset for deterministic trend time indices.
        trend_scale: Scale used to normalize deterministic trend time indices.
        d: Nonseasonal differencing order.
        D: Seasonal differencing order.
        s: Seasonal period length.
        k_trend: Number of deterministic trend parameters.
        k_exog: Number of exogenous-regressor parameters.
        k_ar: Number of nonseasonal autoregressive parameters.
        k_ma: Number of nonseasonal moving-average parameters.
        k_sar: Number of seasonal autoregressive parameters.
        k_sma: Number of seasonal moving-average parameters.
        ar_lags: Active nonseasonal autoregressive lags.
        ma_lags: Active nonseasonal moving-average lags.
        sar_lags: Active seasonal autoregressive lags.
        sma_lags: Active seasonal moving-average lags.
        simple_differencing: Whether to difference data before the state-space likelihood instead of integrating in the state.
        initial_diffuse_variance: Initial variance assigned to diffuse state components.
        diffuse: Whether to use diffuse initialization for integrated state components.

    Returns:
        Callable ``loglike_temp(params, return_type='llf', ...)`` that
        evaluates the SARIMAX log-likelihood for a packed parameter vector.
    """
    def loglike_temp(params, return_type='llf', steady_state_tol=DEFAULT_SARIMAX_STEADY_STATE_TOL,
                     simple_differencing=simple_differencing,
                     initial_diffuse_variance=initial_diffuse_variance,
                     diffuse=diffuse):
        """
        Evaluate the SARIMAX log-likelihood for a packed parameter vector.

        Unpacks ``params`` into its constituent sub-vectors (trend, exog, AR,
        MA, seasonal AR, seasonal MA, sigma2) via ``split_params``, expands
        the nonseasonal and seasonal AR and MA blocks into full-length lag
        arrays, then delegates to ``loglike_internal`` for Kalman evaluation.

        Args:
            params: Packed parameter vector in model parameter order.
            return_type: ``'llf'`` for a scalar total log-likelihood; any
                other value returns the full dict of Kalman internals.
            steady_state_tol: Tolerance for detecting steady-state Kalman forecast error variance.
            simple_differencing: Whether to difference data before the state-space likelihood instead of integrating in the state.
            initial_diffuse_variance: Initial variance assigned to diffuse state components.
            diffuse: Whether to use diffuse initialization for integrated state components.

        Returns:
            Scalar total log-likelihood when ``return_type='llf'``, or a
            dictionary of Kalman filter internals (``llf``, ``llf_obs``,
            ``resid``, ``fittedinnovations``, ``state_vec``,
            ``forecasts_error_cov``, ``steady_state_iter``,
            ``exog_dot_beta``, ``trend_dot_beta``) otherwise.
        """
        trendparams, exogparams, arparams, maparams, sarparams, smaparams, sigma2 \
            = split_params(params, k_trend, k_exog, k_ar, k_ma, k_sar, k_sma)

        arparams_expanded = combine_seasonal_lag_params_into_one_vector(
            ar_lags, arparams, sar_lags, sarparams, s)
        maparams_expanded = combine_seasonal_lag_params_into_one_vector(
            ma_lags, maparams, sma_lags, smaparams, s)

        return loglike_internal(
            endog, arparams=arparams_expanded,
            maparams=maparams_expanded,
            exog=exog, exogparams=exogparams,
            sigma2=sigma2, d=d, D=D, s=s,
            return_type=return_type,
            steady_state_tol=steady_state_tol,
            trend=trend, trendparams=trendparams,
            trend_offset=trend_offset, trend_scale=trend_scale,
            simple_differencing=simple_differencing,
            initial_diffuse_variance=initial_diffuse_variance,
            diffuse=diffuse
        )

    return loglike_temp


def sarimax_internal(endog, order, seasonal_order=None, exog=None, trend=None, trend_offset=1, trend_scale=1,
                     debug=False, maxiter=100, gtol=1e-4, ftol=1e-15, xtol=1e-15, B0=1.0, do_numba=None,
                     start_params=None, do_hannan_rissanen=True, steady_state_tol=DEFAULT_SARIMAX_STEADY_STATE_TOL,
                     enforce_stationarity=DEFAULT_SARIMAX_ENFORCE_STATIONARITY,
                     enforce_invertibility=DEFAULT_SARIMAX_ENFORCE_INVERTIBILITY,
                     concentrate_scale=DEFAULT_SARIMAX_CONCENTRATE_SCALE,
                     simple_differencing=DEFAULT_SARIMAX_SIMPLE_DIFFERENCING,
                     initial_diffuse_variance=DEFAULT_SARIMAX_INITIAL_DIFFUSE_VARIANCE,
                     multiplicative=DEFAULT_SARIMAX_MULTIPLICATIVE,
                     diffuse=DEFAULT_SARIMAX_DIFFUSE,
                     standardize_endog=DEFAULT_SARIMAX_STANDARDIZE_ENDOG,
                     **kwargs):
    """
    Fit a SARIMAX model by maximum likelihood using the BFGS projected quasi-Newton optimizer.

    This is the core estimation routine called by ``SarimaxModel.fit``.  It
    orchestrates the full fitting pipeline:

    1. **Parse and validate** the ``(p,d,q)`` and ``(P,D,Q,s)`` order specs
       into explicit lag lists and count metadata.
    2. **Optionally standardize** ``endog`` by the standard deviation of the
       differenced series so the optimizer works on a numerically stable scale.
    3. **Build the log-likelihood closure** via ``get_loglike_function``, which
       binds the fixed model structure and data.
    4. **Compute starting parameters** via Hannan-Rissanen (or a user-supplied
       vector, or all-zeros if ``start_params=0.0``).
    5. **Optimize** by maximizing the log-likelihood with ``bfgs_pqn``.
    6. **Recover scale**: if ``standardize_endog=True``, the optimizer ran on
       the standardized series; trend/exog/variance parameters are mapped back
       to the original scale by ``scale_params``.  If ``concentrate_scale=True``,
       the innovation variance ``sigma2`` was optimized out and is recovered
       analytically as the mean normalized squared residual.
    7. **Final evaluation** on the original-scale series to get all Kalman
       filter outputs (residuals, fitted values, per-obs log-likelihood, etc.).
    8. **Compute information criteria** (AIC, AICc, BIC, HQIC) and assemble
       the result dictionary.

    Args:
        endog: Observed endogenous time-series values.
        order: Nonseasonal ARIMA order specification ``(p, d, q)``.
        seasonal_order: Optional seasonal order specification ``(P, D, Q, s)``.
        exog: Optional exogenous regressor matrix aligned to ``endog``.
        trend: Trend specification, such as None, ``c``, ``t``, ``ct``, ``n``, or an indicator list.
        trend_offset: Starting offset for deterministic trend time indices.
        trend_scale: Scale used to normalize deterministic trend time indices.
        debug: Whether to print fitting diagnostics.
        maxiter: Maximum optimizer iterations.
        gtol: Optimizer projected-gradient tolerance.
        ftol: Optimizer objective_function-change tolerance.
        xtol: Optimizer parameter-change tolerance.
        B0: Initial Hessian scale or approximation passed to BFGS/PQN.
        do_numba: Compatibility flag for numba execution paths.
        start_params: Optional initial parameter vector.
        do_hannan_rissanen: Whether to estimate starting ARMA parameters with Hannan-Rissanen.
        steady_state_tol: Tolerance for detecting steady-state Kalman forecast error variance.
        enforce_stationarity: Whether to transform AR parameters into the stationary region.
        enforce_invertibility: Whether to transform MA parameters into the invertible region.
        concentrate_scale: Whether to optimize with innovation variance concentrated out.
        simple_differencing: Whether to difference data before the state-space likelihood instead of integrating in the state.
        initial_diffuse_variance: Initial variance assigned to diffuse state components.
        multiplicative: Whether to request multiplicative seasonal dynamics.
        diffuse: Whether to use diffuse initialization for integrated state components.
        standardize_endog: Whether to standardize endogenous data during optimization.

    Returns:
        Dictionary of fitted parameters, likelihood outputs, covariance inputs, and timing details.
    """
    t0 = time.time()

    k_trend = np.count_nonzero(trend) if trend is not None else 0
    k_exog = exog.shape[1] if exog is not None else 0

    (
        (order, seasonal_order),
        is_seasonal,
        (has_ar_terms, has_ma_terms),
        (((p, ar_lags, k_ar), d, (q, ma_lags, k_ma)),
         ((P, sar_lags, k_sar), D, (Q, sma_lags, k_sma), s))
    ) = validate_orders(order, seasonal_order)

    if multiplicative:
        raise NotImplementedError("`multiplicative = True`")

    sarimax_options = dict(
        **dict(order=none_tuple(order),
               seasonal_order=none_tuple(seasonal_order),
               is_seasonal=is_seasonal,
               ar_lags=ar_lags,
               ma_lags=ma_lags,
               sar_lags=sar_lags,
               sma_lags=sma_lags,
               maxiter=maxiter,
               B0=B0,
               xtol=xtol,
               ftol=ftol,
               gtol=gtol,
               start_params=start_params,
               do_hannan_rissanen=do_hannan_rissanen,
               steady_state_tol=steady_state_tol,
               enforce_stationarity=enforce_stationarity,
               enforce_invertibility=enforce_invertibility,
               trend_scale=trend_scale,
               trend=none_tuple(trend),
               trend_offset=trend_offset,
               simple_differencing=simple_differencing,
               initial_diffuse_variance=initial_diffuse_variance,
               diffuse=diffuse,
               multiplicative=multiplicative,
               standardize_endog=standardize_endog,
               ),
        **kwargs
    )

    trend_scale = format_trend_scale(trend_scale, len(endog))

    if exog is not None and np.ndim(exog) == 1:
        exog = exog.reshape((-1, 1))

    if standardize_endog:
        if d or D:
            endog_diff = difference(endog, d, D, s)
        else:
            endog_diff = endog
        # Standardize by the differenced scale so integrated models optimize on
        # a numerically stable innovation scale.
        mean_y = endog.mean()
        std_y = endog_diff.std()
        endog_std = (endog - mean_y) / std_y
    else:
        endog_std, std_y, mean_y = endog, 1.0, 0.0

    if debug:
        print_options(sarimax_options, 'SARIMAX OPTIONS')

    def get_loglike_function_temp(y):
        """
        Build a log-likelihood closure bound to the given endogenous series.

        Thin wrapper around ``get_loglike_function`` that captures all model-
        structure arguments (exog, trend, order, lag arrays, etc.) via closure,
        so the optimizer only needs to pass a parameter vector.

        Args:
            y: Endogenous time-series array (original or standardized).

        Returns:
            Callable ``loglike(params, return_type='llf', ...)`` that evaluates
            the SARIMAX log-likelihood for a packed parameter vector.
        """
        return get_loglike_function(
            y, exog, trend, trend_offset, trend_scale,
            d, D, s, k_trend, k_exog, k_ar, k_ma, k_sar, k_sma,
            ar_lags, ma_lags, sar_lags, sma_lags,
            simple_differencing, initial_diffuse_variance, diffuse)

    loglike = get_loglike_function_temp(endog_std)

    if debug:
        t1 = time.time()
        print("Computing start params" + (" using Hannan-Rissanen" if do_hannan_rissanen else "") + "...", end="")

    start_params = get_sarimax_start_params(start_params, ar_lags, ma_lags, sar_lags, sma_lags, d, D, s,
                                            endog_std, exog, trend, trend_offset=trend_offset, trend_scale=trend_scale,
                                            do_hannan_rissanen=do_hannan_rissanen,
                                            enforce_stationarity=enforce_stationarity,
                                            concentrate_scale=concentrate_scale,
                                            enforce_invertibility=enforce_invertibility, debug=debug)

    if debug:
        print("%.2fs" % (time.time() - t1))
        print(f'Start Params: {start_params}\n')
        print(f'llf at `start_params`: {loglike(start_params):.4f}', end='')
        if k_trend + k_exog + k_ar + k_ma + k_sar + k_sma == len(start_params):
            print(f' (concentrated scale)')
        else:
            print()

    optimization_result = bfgs_pqn(loglike, x0=start_params,
                                   B0=B0, xtol=xtol, ftol=ftol, gtol=gtol, maxiter=maxiter, debug=debug,
                                   maximize=True)

    llf_call_full = loglike(optimization_result.x, return_type='all')

    if concentrate_scale:
        if simple_differencing:
            start_idx = 0
        else:
            # State-space integration leaves burn-in observations in the arrays;
            # exclude them from the concentrated variance estimate.
            start_idx = d + D * s
        sigma2 = np.mean((llf_call_full['resid'] ** 2 / llf_call_full['forecasts_error_cov'])[start_idx:])
        params = np.hstack([optimization_result.x, sigma2])
    else:
        params = optimization_result.x.copy()

    params = scale_params(params, k_trend, k_exog, k_ar, k_ma, k_sar, k_sma,
                          trend=trend, scale=std_y, mean=mean_y)

    loglike = get_loglike_function_temp(endog)
    llf_call_full = loglike(params, return_type='all')
    # if concentrate_scale: TODO DELETE
    #     llf_call_full = loglike(params, return_type='all')

    if debug:
        print(f'llf at maximand: {llf_call_full["llf"]:.4f}', end='')
        if k_trend + k_exog + k_ar + k_ma + k_sar + k_sma == len(start_params):
            print(f' (concentrated scale)')
        else:
            print()

    trendparams, exogparams, arparams, maparams, seasonalarparams, seasonalmaparams, sigma2 \
        = split_params(params, k_trend, k_exog, k_ar, k_ma, k_sar, k_sma)

    loglikelihood_burn = d + s * D
    if loglikelihood_burn and simple_differencing:
        # Pad differenced-space Kalman outputs so result arrays align to the
        # original endogenous series.
        for key in ['llf_obs', 'resid', 'fittedinnovations', 'forecasts_error_cov', 'state_vec']:
            llf_call_full[key] = np.hstack([[np.nan] * loglikelihood_burn, llf_call_full[key]])

    fittedvalues = np.zeros(len(endog))
    if loglikelihood_burn and simple_differencing:
        fittedvalues[loglikelihood_burn:] = endog[loglikelihood_burn:] - llf_call_full['resid'][loglikelihood_burn:]
    else:
        fittedvalues[:] = endog - llf_call_full['resid']
    llf_call_full['fittedvalues'] = fittedvalues

    if debug:
        print("Final params...\n", params)

    def loglike_obs(params, steady_state_tol=DEFAULT_SARIMAX_STEADY_STATE_TOL):
        """
        Evaluate per-observation model log-likelihood values.

        Runs the full Kalman recursion and returns just the observation-level
        log-likelihood array.  Used by the covariance estimators in
        ``var_covar.py`` to compute score outer products and Hessians.

        Args:
            params: Packed parameter vector in model parameter order.
            steady_state_tol: Tolerance for detecting steady-state Kalman forecast error variance.

        Returns:
            1-D NumPy array of length ``nobs`` containing the per-observation
            Gaussian log-likelihood contributions.
        """
        return loglike(params, return_type='full', steady_state_tol=steady_state_tol)['llf_obs']

    num_params = k_ar + k_ma + k_sar + k_sma + k_exog + k_trend + 1
    T = len(endog) - loglikelihood_burn
    llf = llf_call_full['llf']
    aic = -2 * llf + 2 * num_params

    aicc = aic + 2 * (num_params ** 2 + num_params) / (T - num_params - 1 - d)

    bic = -2 * llf + (np.log(T - d)) * num_params
    hqic = -2 * llf + 2 * (np.log(np.log(T - d))) * num_params

    fit_elapsed = time.time() - t0

    llf_call_full.update(dict(
        params=params,
        fit_elapsed=fit_elapsed,
        sigma2=sigma2, trendparams=trendparams, exogparams=exogparams,
        arparams=arparams, maparams=maparams,
        seasonalarparams=seasonalarparams, seasonalmaparams=seasonalmaparams,
        optimization_result=optimization_result,
        loglike=loglike, loglike_obs=loglike_obs,
        converged=optimization_result.converged,
        iter=optimization_result.iter,
        grad_norm=optimization_result.gnorm,
        nobs=len(endog),
        loglikelihood_burn=loglikelihood_burn,
        aic=aic, aicc=aicc, bic=bic, hqic=hqic,
        sarimax_options=sarimax_options,
    ))

    return llf_call_full
