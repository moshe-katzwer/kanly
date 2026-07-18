from __future__ import absolute_import, print_function

import numpy as np
from numba import njit
from scipy.linalg import solve_discrete_lyapunov


def get_arma_statespace_matrices(arparams, maparams, sigma2):
    """
    Build Brockwell-Davis companion-form ARMA state-space matrices.

    Constructs the state dimension ``r = max(p, q+1)`` and assembles four
    matrices that define the linear Gaussian state-space model:

    * **transition** ``(r×r)``: companion matrix whose last row holds the
      AR coefficients (reversed) and whose super-diagonal is the identity;
      this propagates the state one step forward.
    * **observation** ``(r,)``: row vector with ``1`` at position ``r-1`` and
      the MA coefficients (reversed) filling positions ``r-q-1`` to ``r-2``;
      extracting the observable from the state.
    * **var_innovation** ``(r×r)``: innovation covariance matrix with
      ``sigma2`` at the ``(r-1, r-1)`` entry only.
    * **unconditional_state_variance** ``(r×r)``: Lyapunov solution
      ``P = T P T' + Q`` giving the stationary state covariance, falling back
      to a ``sigma2``-scaled diagonal when the solve fails (e.g. near
      unit-root parameters).

    Args:
        arparams: Autoregressive coefficients ordered by active AR lags.
        maparams: Moving-average coefficients ordered by active MA lags.
        sigma2: Innovation variance parameter.

    Returns:
        4-tuple ``(transition_arma, observation_arma, var_innovation,
        unconditional_state_variance)`` of NumPy arrays with shapes
        ``(r,r)``, ``(r,)``, ``(r,r)``, and ``(r,r)`` respectively.
    """
    p = len(arparams)
    q = len(maparams)
    r = max(p, q + 1)

    transition_arma = np.zeros((r, r))
    if p:
        transition_arma[-1, -p:] = np.flip(arparams)
    transition_arma[:-1, 1:] = np.eye(r - 1)

    observation_arma = np.zeros(r)
    observation_arma[-1] = 1.0
    observation_arma[r - q - 1:-1] = np.flip(maparams)

    var_innovation = np.zeros((r, r))  # Q
    var_innovation[- 1, - 1] = sigma2

    try:
        unconditional_state_variance = solve_discrete_lyapunov(transition_arma, var_innovation)
    except:
        # Non-stationary or nearly singular starts can make the Lyapunov solve
        # fail; use a simple diagonal covariance so optimization can continue.
        unconditional_state_variance = np.ones(r) * sigma2

    return transition_arma, observation_arma, var_innovation, unconditional_state_variance


@njit(inline='always', cache=True)
def max_abs_matrix(A):
    """
    Compute the maximum absolute element of a matrix.

    Equivalent to ``np.abs(A).max()`` but avoids allocating a temporary
    array and is friendly to Numba compilation and optimization.

    Args:
        A: Two-dimensional array.

    Returns:
        Largest absolute entry in ``A``.
    """
    m = 0.0

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            v = abs(A[i, j])
            if v > m:
                m = v

    return m


@njit(inline='always', cache=True)
def kalman_predict_update(
        x_k_k,
        P_k_k,
        y,
        transition,
        observation,
        var_innovation):
    """
    Execute a single Kalman predict-update step.

    Given the filtered state and covariance from time ``k-1``, performs:

        1. State and covariance prediction.
        2. Innovation and forecast variance computation.
        3. Kalman gain calculation.
        4. State and covariance update.

    Args:
        x_k_k: Filtered state mean at time ``k-1``.
        P_k_k: Filtered state covariance at time ``k-1``.
        y: Observation at time ``k``.
        transition: State transition matrix.
        observation: Observation vector.
        var_innovation: State innovation covariance matrix.

    Returns:
        Tuple ``(x_upd, P_upd, resid, fitted, S)`` containing the updated
        state mean, updated covariance, innovation, fitted value, and
        forecast error variance.
    """
    # Predict state and covariance.
    x_pred = transition @ x_k_k

    tmp = transition @ P_k_k
    P_pred = tmp @ transition.T
    P_pred += var_innovation

    # Innovation and forecast variance.
    fitted = observation @ x_pred
    resid = y - fitted

    Hp = P_pred @ observation
    S = observation @ Hp

    # Kalman update.
    K = Hp / S

    x_upd = x_pred + K * resid

    KH = np.outer(K, observation)
    P_upd = P_pred - KH @ P_pred

    return x_upd, P_upd, resid, fitted, S


@njit(inline='always', cache=True)
def steady_state_step(
        x_k_k,
        x_pred,
        transition,
        observation,
        y):
    """
    Execute the lightweight steady-state recursion.

    Once covariance convergence has been detected, the covariance and
    forecast error variance are treated as fixed and only the state
    recursion is propagated.

    Args:
        x_k_k: Current filtered state vector (updated in place).
        x_pred: Scratch workspace used for the predicted state.
        transition: State transition matrix.
        observation: Observation vector.
        y: Observation at the current time step.

    Returns:
        Tuple ``(resid, fitted)`` containing the innovation and fitted
        value for the observation.
    """
    # Companion-form state prediction.
    x_pred[:-1] = x_k_k[1:]
    x_pred[-1] = transition[-1] @ x_k_k

    fitted = observation @ x_pred
    resid = y - fitted

    # State update under fixed steady-state gain.
    x_k_k[:] = x_pred
    x_k_k[-1] += resid

    return resid, fitted


@njit(cache=True)
def do_arma_kalman_recursion(
        endog, d,
        D,
        s,
        unconditional_state_variance,
        transition,
        observation,
        var_innovation,
        difference_coefs,
        steady_state_tol,
        initial_diffuse_variance,
        diffuse):
    """
    Run the ARMA Kalman filter and collect one-step-ahead innovations and
    filtered states.

    Implements the standard predict-update Kalman recursion:

        1. Predict state mean and covariance.
        2. Compute innovation and forecast error variance.
        3. Update state mean and covariance using the Kalman gain.

    Once the filtered covariance has effectively converged, the recursion
    switches to a lightweight steady-state implementation that avoids
    repeatedly updating the covariance matrix.

    When ``diffuse=True`` and the original model contains integration
    (``d + D*s > 0``), the initial covariance is taken as
    ``I * initial_diffuse_variance`` rather than the stationary covariance.

    Args:
        endog: Observed time-series values.
        d: Nonseasonal differencing order.
        D: Seasonal differencing order.
        s: Seasonal period.
        unconditional_state_variance: Stationary initial state covariance.
        transition: State transition matrix.
        observation: Observation vector.
        var_innovation: State innovation covariance matrix.
        difference_coefs: Unused. Included for API compatibility with the
            ARIMA recursion implementation.
        steady_state_tol: Covariance convergence tolerance.
        initial_diffuse_variance: Variance used for diffuse initialization.
        diffuse: Whether diffuse initialization is enabled.

    Returns:
        Tuple containing

            resid:
                One-step-ahead innovations.

            fittedinnovations:
                One-step-ahead predictions.

            forecasts_error_cov:
                Forecast error variances.

            state_vec:
                Final state component at each iteration.

            steady_state_iter:
                Iteration at which steady state was detected, or
                ``np.inf`` if never reached.
    """
    r = observation.shape[0]
    T_eff = endog.shape[0]

    resid = np.empty(T_eff)
    fittedinnovations = np.empty(T_eff)
    forecasts_error_cov = np.empty(T_eff)
    state_vec = np.empty(T_eff)

    if diffuse and (d + D * s):
        P_k_k = np.eye(r) * initial_diffuse_variance
    else:
        P_k_k = unconditional_state_variance.copy()

    x_k_k = np.zeros(r)

    stopped = False
    steady_state_iter = np.inf

    # Scratch workspace reused after covariance convergence.
    x_pred_ss = np.empty(r)

    for k in range(T_eff):

        if not stopped:

            x_k_k, P_k_k, resid_k, fitted_k, S_k = kalman_predict_update(
                x_k_k,
                P_k_k,
                endog[k],
                transition,
                observation,
                var_innovation
            )

            resid[k] = resid_k
            fittedinnovations[k] = fitted_k
            forecasts_error_cov[k] = S_k

            # Switch to the lightweight steady-state recursion once the
            # covariance matrix has effectively converged.
            if k > r and max_abs_matrix(P_k_k) <= steady_state_tol:
                stopped = True
                steady_state_iter = k
                forecasts_error_cov[k:] = S_k

        else:

            resid_k, fitted_k = steady_state_step(
                x_k_k,
                x_pred_ss,
                transition,
                observation,
                endog[k]
            )

            resid[k] = resid_k
            fittedinnovations[k] = fitted_k

        state_vec[k] = x_k_k[-1]

    return (
        resid,
        fittedinnovations,
        forecasts_error_cov,
        state_vec,
        steady_state_iter
    )

# @njit(cache=True)
# def do_arma_kalman_recursion_deprecated(
#         endog, d, D, s, unconditional_state_variance, transition, observation, var_innovation,
#         difference_coefs, steady_state_tol, initial_diffuse_variance, diffuse):
#     """
#     Run the ARMA Kalman filter and collect one-step-ahead innovations and states.
#
#     Implements the standard predict-update Kalman loop over the full series.
#     At each step:
#
#     1. **Predict**: advance the state mean ``x_{k|k-1} = T x_{k-1|k-1}`` and
#        covariance ``P_{k|k-1} = T P T' + Q``.
#     2. **Update**: compute the innovation ``v_k = y_k - H x_{k|k-1}``,
#        forecast error variance ``S_k = H P_{k|k-1} H'``, Kalman gain
#        ``K_k = P_{k|k-1} H' / S_k``, and apply the gain to update state mean
#        and covariance.
#
#     Once ``|P_{k|k}|_max <= steady_state_tol`` (after at least ``r`` steps)
#     the covariance has converged and only the lightweight state recursion is
#     run for the remaining observations, with ``S_k`` held fixed.
#
#     When ``diffuse=True`` and the model has integration (``d+D*s > 0``) the
#     initial covariance is set to ``I * initial_diffuse_variance`` instead of
#     the unconditional stationary covariance.
#
#     Args:
#         endog: Observed endogenous time-series values (after residualization
#             and, if ``simple_differencing``, after explicit differencing).
#         d: Nonseasonal differencing order (used only to detect if diffuse
#             initialization is needed).
#         D: Seasonal differencing order.
#         s: Seasonal period length.
#         unconditional_state_variance: Initial state covariance (Lyapunov solution
#             from ``get_arma_statespace_matrices``).
#         transition: ``(r×r)`` companion state transition matrix.
#         observation: ``(r,)`` observation row vector.
#         var_innovation: ``(r×r)`` innovation covariance matrix.
#         difference_coefs: Unused in the ARMA path (accepted for a uniform call
#             signature with ``do_arima_kalman_recursion``).
#         steady_state_tol: Max absolute entry in ``P_{k|k}`` below which the
#             filter is considered to have reached steady state.
#         initial_diffuse_variance: Diagonal variance for diffuse initialization.
#         diffuse: If True and the model has integration, initialize ``P_0`` to
#             ``I * initial_diffuse_variance`` instead of the stationary Lyapunov
#             solution.
#
#     Returns:
#         5-tuple ``(resid, fittedinnovations, forecasts_error_cov, state_vec,
#         steady_state_iter)`` where each of the first four is a 1-D array of
#         length ``T``, and ``steady_state_iter`` is the time index at which
#         the filter reached steady state (``np.inf`` if it never did).
#     """
#     r = observation.shape[0]
#
#     T_eff = endog.shape[0]
#     resid = np.zeros(T_eff)
#     fittedinnovations = np.zeros(T_eff)
#     forecasts_error_cov = np.zeros(T_eff)
#     state_vec = np.zeros(T_eff)
#
#     P_k_k = unconditional_state_variance
#     if diffuse and d + D*s:
#         # This ARMA path is used after simple differencing; a diffuse request
#         # keeps startup uncertainty large when the original model had integration.
#         P_k_k = np.eye(r) * initial_diffuse_variance
#
#     x_k_k = np.zeros(r)
#
#     stopped = False
#     steady_state_iter = np.inf
#
#     for k in range(T_eff):
#
#         if not stopped:
#
#             # Predict
#             x_k_k_min_1 = transition.dot(x_k_k)
#             P_k_k_min_1 = transition.dot(P_k_k).dot(transition.T) + var_innovation
#
#             # Update
#             fittedinnovations[k] = observation.dot(x_k_k_min_1)
#             resid[k] = endog[k] - fittedinnovations[k]
#
#             S_k = observation.dot(P_k_k_min_1).dot(observation.T)  # + R
#             forecasts_error_cov[k] = S_k
#
#             K_k = P_k_k_min_1.dot(observation.T) / S_k
#
#             x_k_k = x_k_k_min_1 + K_k * resid[k]
#
#             P_k_k = P_k_k_min_1 - np.outer(K_k, observation).dot(P_k_k_min_1)
#
#             if k > r and np.abs(P_k_k).max() <= steady_state_tol:
#                 # After steady state the covariance and forecast error variance
#                 # are fixed, so the loop can update only the state recursion.
#                 stopped = True
#                 steady_state_iter = k
#                 x_k_k = x_k_k[:r]
#                 x_k_k_min_1 = x_k_k_min_1[:r]
#                 forecasts_error_cov[k:] = S_k
#
#         else:
#
#             x_k_k_min_1[:-1] = x_k_k[1:]
#             x_k_k_min_1[-1] = transition[-1].dot(x_k_k)
#
#             fittedinnovations[k] = observation.dot(x_k_k_min_1)
#             resid[k] = endog[k] - fittedinnovations[k]
#
#             x_k_k[:] = x_k_k_min_1
#             x_k_k[-1] += resid[k]
#
#         state_vec[k] = x_k_k[-1]
#
#     return resid, fittedinnovations, forecasts_error_cov, state_vec, steady_state_iter
