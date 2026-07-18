from __future__ import absolute_import, print_function

import numpy as np
from numba import njit
from scipy.linalg import solve_discrete_lyapunov

from kanly.time_series.sarimax.polynomial import get_combined_differencing_coefs
from kanly.time_series.sarimax.state_space_arma import get_arma_statespace_matrices

# The augmented ARIMA state can be large and often near non-stationary, so the
# implementation initializes the stationary ARMA block by Lyapunov solution and
# gives integrated states diffuse variance instead of solving the full system.
FULL_LYAPUNOV = False  # DON'T CHANGE


def get_arima_statespace_matrices(arparams, maparams, d=0, D=0, s=2, sigma2=1.0):
    """
    Build augmented ARIMA state-space matrices for integrated models.

    Extends the Brockwell-Davis ARMA companion form (``get_arma_statespace_matrices``)
    to handle non-seasonal and seasonal integration orders ``d`` and ``D``.
    The augmented state has dimension ``r + d + D*s`` and is structured as:

    * **Top-left block** (``r×r``): the ARMA companion transition matrix.
    * **Bottom block** (``(d+D*s) × (d+D*s)``): a companion block ``B`` that
      propagates the differenced past values; its last row encodes the
      combined differencing coefficients so the integrator correctly
      reconstructs the undifferenced level.
    * **Off-diagonal coupling block**: ``A ⊗ H_arma`` links the ARMA
      state forecast to the integrator state.

    The augmented observation vector is ``[H_arma | B[-1]]``, so the
    predicted observation includes both the ARMA signal and the lagged
    levels needed to invert the differencing.

    When ``d=0`` and ``D=0`` the function falls through to the plain ARMA
    matrices with an empty ``difference_coefs`` array.

    The initial state covariance is assembled by block-structure: the ARMA
    block receives the Lyapunov stationary covariance, and each integrated
    component receives ``initial_diffuse_variance`` on its diagonal (set
    during the recursion, not here).

    Args:
        arparams: Autoregressive coefficients ordered by active AR lags.
        maparams: Moving-average coefficients ordered by active MA lags.
        d: Nonseasonal differencing order.
        D: Seasonal differencing order.
        s: Seasonal period length.
        sigma2: Innovation variance parameter.

    Returns:
        5-tuple ``(transition, observation, var_innovation, difference_coefs,
        unconditional_state_variance)`` where:

        - ``transition``: ``(r+d+D*s, r+d+D*s)`` augmented transition matrix.
        - ``observation``: ``(r+d+D*s,)`` augmented observation vector.
        - ``var_innovation``: ``(r+d+D*s, r+d+D*s)`` innovation covariance
          (sigma2 at the ``(r-1, r-1)`` ARMA noise entry).
        - ``difference_coefs``: 1-D array of combined differencing lag
          coefficients for the steady-state integrator path in
          ``do_arima_kalman_recursion``.
        - ``unconditional_state_variance``: ``(r, r)`` stationary covariance
          for the ARMA block only (used to initialize the top-left block of
          the full covariance matrix in the recursion).
    """
    p = len(arparams)
    q = len(maparams)
    r = max(p, q + 1)

    transition_arma, observation_arma, var_innovation_arma, unconditional_state_variance = \
        get_arma_statespace_matrices(arparams, maparams, sigma2)

    if d == 0 and D == 0:
        transition, observation, difference_coefs = transition_arma, observation_arma, np.array([])

    else:

        difference_coefs = get_combined_differencing_coefs(d, D, s)

        A = np.zeros(d + D * s)
        A[-1] = 1

        if d + D * s > 1:
            B = np.zeros((d + D * s, d + D * s))
            B[:-1, 1:] = np.eye(d + D * s - 1)
            B[-1] = np.flip(difference_coefs)
        else:
            B = np.array([[1.0]])

        transition = np.array(np.bmat([
            [transition_arma, np.zeros((r, d + D * s))],
            [np.outer(A, observation_arma), B]
        ]))

        observation = np.hstack([observation_arma, B[-1]])

    var_innovation = np.zeros((d + s * D + r, d + s * D + r))  # Q
    var_innovation[r - 1, r - 1] = sigma2

    if FULL_LYAPUNOV:
        try:
            unconditional_state_variance = solve_discrete_lyapunov(transition, var_innovation)
        except:
            # Fall back to diagonal initialization when the Lyapunov solve is
            # singular or numerically unstable.
            unconditional_state_variance = np.eye(r + d + s * D) * sigma2
    else:
        try:
            unconditional_state_variance = solve_discrete_lyapunov(transition_arma, var_innovation_arma)
        except:
            # Only the ARMA block gets a stationary covariance in the default
            # path; integrated states are handled below with diffuse variance.
            unconditional_state_variance = np.eye(r) * sigma2

    return transition, observation, var_innovation, difference_coefs, unconditional_state_variance


@njit(cache=True)
def do_arima_kalman_recursion(endog, d, D, s, unconditional_state_variance, transition, observation, var_innovation,
                              difference_coefs, steady_state_tol, initial_diffuse_variance, diffuse):
    # endog0 = endog[0]
    # endog = endog - endog0

    """
    Run the integrated ARIMA Kalman filter on the original (undifferenced) series.

    Unlike ``do_arma_kalman_recursion``, this function operates directly on the
    level series.  The augmented state carries both the ARMA state and a
    buffer of the ``d + D*s`` lagged levels needed to undo the differencing
    in-state, so no pre-differencing of ``endog`` is required.

    **Initialization**: the full ``(r+d+D*s, r+d+D*s)`` initial covariance
    ``P_0`` is built as a zero matrix, then:

    * The top-left ``(r×r)`` block is set to the Lyapunov stationary covariance
      of the ARMA component.
    * Each of the ``d + D*s`` integrated state slots gets ``initial_diffuse_variance``
      on its diagonal, reflecting the lack of a finite unconditional variance
      for non-stationary components.

    **Predict-update loop**: identical to ``do_arma_kalman_recursion``.

    **Steady-state transition**: when ``|P_{k|k}|_max < steady_state_tol``
    (after at least ``2r`` steps) the covariance has converged.  The integrated
    state slots are dropped and the differencing contribution is computed
    directly from lagged ``endog`` values using ``difference_coefs``, avoiding
    the need to advance the full augmented state.

    The ``state_vec`` stores the scalar ARMA innovation state
    (index ``r - d - D*s - 1`` of the full state) at every time step.

    Args:
        endog: Observed endogenous time-series values at the original
            (undifferenced) scale, after exogenous and trend residualization.
        d: Nonseasonal differencing order.
        D: Seasonal differencing order.
        s: Seasonal period length.
        unconditional_state_variance: ``(r×r)`` stationary covariance for the
            ARMA block, used to fill the top-left block of ``P_0``.
        transition: Augmented ``(r+d+D*s, r+d+D*s)`` transition matrix.
        observation: Augmented ``(r+d+D*s,)`` observation vector.
        var_innovation: Augmented innovation covariance matrix.
        difference_coefs: 1-D array of combined differencing lag coefficients,
            used during the steady-state phase to subtract lagged ``endog``
            values from the predicted observation.
        steady_state_tol: Max absolute entry in ``P_{k|k}`` below which the
            filter is considered converged.
        initial_diffuse_variance: Diagonal variance assigned to integrated
            state slots in ``P_0``.
        diffuse: Accepted for call-signature compatibility; diffuse
            initialization is always applied here for the integrated slots.

    Returns:
        5-tuple ``(resid, fittedinnovations, forecasts_error_cov, state_vec,
        steady_state_iter)`` where each of the first four is a 1-D array of
        length ``T``, and ``steady_state_iter`` is the time index at which
        the filter reached steady state.  The first ``d + D*s`` entries of
        ``resid`` and ``llf_obs`` correspond to the diffuse initialization
        period and are typically excluded from the likelihood sum.
    """
    r = len(observation)
    P_k_k = np.zeros((r, r))

    if d + s * D:
        if FULL_LYAPUNOV:
            P_k_k = unconditional_state_variance
        else:
            # The leading block is the stationary ARMA covariance. The trailing
            # integrated components are initialized diffusely because they do
            # not have a finite unconditional variance.
            P_k_k[:-(d + s * D), :-(d + s * D)] = unconditional_state_variance
        for j in range(d + s * D):
            P_k_k[-(j + 1), -(j + 1)] = initial_diffuse_variance
    else:
        P_k_k = unconditional_state_variance.copy()

    x_k_k = np.zeros(r)

    T = len(endog)
    resid = np.zeros(T)
    fittedinnovations = np.zeros(T)
    forecasts_error_cov = np.zeros(T)
    state_vec = np.zeros(T)

    llb = d + D * s

    stopped = False

    for k in range(T):

        if not stopped:
            x_k_k_min_1 = transition.dot(x_k_k)
            P_k_k_min_1 = transition.dot(P_k_k).dot(transition.T) + var_innovation

            # Update
            fittedinnovations[k] = observation.dot(x_k_k_min_1)
            resid[k] = endog[k] - fittedinnovations[k]

            S_k = observation.dot(P_k_k_min_1).dot(observation.T)  # + R
            forecasts_error_cov[k] = S_k

            K_k = P_k_k_min_1.dot(observation.T) / S_k

            x_k_k = x_k_k_min_1 + K_k * resid[k]

            P_k_k = P_k_k_min_1 - np.outer(K_k, observation).dot(P_k_k_min_1)

            if k > 2 * r and np.max(np.abs(P_k_k)) < steady_state_tol:
                stopped = True
                steady_state_iter = k
                if llb:
                    x_k_k = x_k_k[:-(d + s * D)]
                    forecasts_error_cov[k:] = S_k
                    observation_arma = observation[:-(d + s * D)].copy()
                    transition_arma = transition[:-(d + s * D), :-(d + s * D)].copy()
                else:
                    observation_arma = observation
                    transition_arma = transition

        else:

            x_k_k_min_1 = transition_arma.dot(x_k_k)

            # Update
            fittedinnovations[k] = observation_arma.dot(x_k_k_min_1)
            ed = endog[k]
            # Once the ARMA covariance has reached steady state, subtract the
            # differencing lag terms manually rather than carrying the full
            # augmented integrated state forward.
            for l, j in enumerate(difference_coefs):
                ed -= j * endog[k - (l + 1)]
            resid[k] = ed - fittedinnovations[k]

            x_k_k = x_k_k_min_1
            x_k_k[-1] += resid[k]

        state_vec[k] = x_k_k[r - (d + D * s) - 1]

    #resid[0] += endog0
    return resid, fittedinnovations, forecasts_error_cov, state_vec, steady_state_iter

# def loglike_internal(endog, arparams, maparams,
#                      exog=None, exogparams=None,
#                      trend=None, trendparams=None,
#                      trend_offset=1, trend_scale=1.0,
#                      steady_state_tol=DEFAULT_SARIMAX_STEADY_STATE_TOL,
#                      d=0, D=0, s=2,
#                      sigma2=1.0,
#                      initial_diffuse_variance=DEFAULT_SARIMAX_INITIAL_DIFFUSE_VARIANCE,
#                      return_type='llf'):
#     # Remove exog and trend
#     endog_min_covar, exog_dot_beta, trend_dot_beta = residualize(
#         endog, exog, trend, exogparams, trendparams, trend_offset, trend_scale)
#
#     # Form state space for Kalman recursion
#     transition_arma, observation_arma, transition, observation, var_innovation, difference_coefs, P0 = \
#         get_arima_statespace_matrices(arparams, maparams, d=d, D=D, s=s, sigma2=sigma2)
#
#     # Do Kalman recursion
#     resid, fittedinnovations, forecasts_error_cov, state_vec, steady_state_iter = \
#         do_arima_kalman_recursion(endog_min_covar, d, D, s, transition_arma, observation_arma, transition, observation, var_innovation,
#                                   difference_coefs, P0, steady_state_tol, initial_diffuse_variance)
#
#     # Compute LLF from residuals
#     llf_obs = logpdf_norm(resid, 0, forecasts_error_cov ** .5)
#     llf = llf_obs[d + D * s:].sum()
#
#     if return_type == 'llf':
#         return llf
#     else:
#         return dict(
#             llf=llf, llf_obs=llf_obs, resid=resid,
#             fittedinnovations=fittedinnovations,
#             state_vec=state_vec,
#             forecasts_error_cov=forecasts_error_cov,
#             steady_state_iter=steady_state_iter,
#             exog_dot_beta=exog_dot_beta, trend_dot_beta=trend_dot_beta,
#         )
#
#
# def get_loglike_function(endog, exog, trend, trend_offset, trend_scale,
#                          d, D, s, k_trend, k_exog, k_ar, k_ma, k_sar, k_sma,
#                          ar_lags, ma_lags, sar_lags, sma_lags):
#
#     def loglike_temp(params, return_type='llf', steady_state_tol=DEFAULT_SARIMAX_STEADY_STATE_TOL):
#         trendparams, exogparams, arparams, maparams, sarparams, smaparams, sigma2 \
#             = split_params(params, k_trend, k_exog, k_ar, k_ma, k_sar, k_sma)
#
#         arparams_expanded = combine_seasonal_lag_params_into_one_vector(
#             ar_lags, arparams, sar_lags, sarparams, s)
#         maparams_expanded = combine_seasonal_lag_params_into_one_vector(
#             ma_lags, maparams, sma_lags, smaparams, s)
#
#         return loglike_internal(
#             endog, arparams=arparams_expanded,
#             maparams=maparams_expanded,
#             exog=exog, exogparams=exogparams,
#             sigma2=sigma2, d=d, D=D, s=s,
#             return_type=return_type,
#             steady_state_tol=steady_state_tol,
#             trend=trend, trendparams=trendparams,
#             trend_offset=trend_offset, trend_scale=trend_scale,
#         )
#
#     return loglike_temp
