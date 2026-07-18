"""
Autoregressive parameter estimation utilities.

This module provides multiple estimators for AR(lags) models, including:

    1. Conditional Sum of Squares (CSS / OLS)
    2. Yule-Walker (Durbin–Levinson recursion)
    3. Burg's method (forward/backward prediction error minimization)
    4. Maximum Likelihood Estimation (MLE) via exact or conditional likelihood  

The underlying data-generating process assumed across methods is:

    y[t] = const + φ1*y[t-1] + ... + φL*y[t-L] + e[t]
    e[t] ~ iid Normal(0, sigma^2)

The intercept `const` is a structural mean parameter and differs from the
parameterization used in SARIMAX state-space models, where the AR process is
embedded inside the error dynamics.  In kanly's SARIMAX, the model is

    y[t] = const + e[t]
    e[t] = φ1*y[t-1] + ... + φL*y[t-L] + u[t]
    u[t] ~ iid Normal(0, sigma^2)

``φ`` will be the same regardless of the treatment, but the relationship between intercept parameterizations is:

    const (this module) = (SARIMAX intercept) * (1 - sum(φ_i))

---

# Estimation methods

## 1. Conditional Sum of Squares (CSS / OLS), method='css'

Fits AR parameters via linear regression of y[t] on lagged values and an
intercept, using observations t >= lags.

Equivalent to OLS on the lagged design matrix.

Returns:
    - intercept
    - AR coefficients
    - innovation variance
    - covariance matrix based on (X'X)^(-1)

---

## 2. Yule-Walker (Durbin–Levinson), method='yule-walker'

Estimates AR coefficients using sample autocovariances and solves the
Yule-Walker equations using the Durbin–Levinson recursion.

Properties:
    - No intercept estimated (assumes centered data)
    - Fast O(lags^2) recursion
    - Asymptotically equivalent to MLE under stationarity

Returns:
    - AR coefficients
    - innovation variance
    - large-sample Toeplitz covariance approximation

---

## 3. Burg's method, method='burg'

Estimates AR coefficients by recursively minimizing forward and backward
one-step prediction errors.

Key properties:
    - Uses both forward and backward residuals
    - Produces reflection coefficients (PACF)
    - Enforces stability implicitly via |κ_k| < 1 at each recursion
    - Does NOT require autocovariance estimation
    - Typically lower bias than Yule-Walker in small samples

Returns:
    - intercept (sample mean)
    - AR coefficients
    - innovation variance (from final forward residuals)

Note:
    Burg is not a likelihood method; standard errors are not intrinsic and
    must be computed externally (e.g., OLS approximation or bootstrap).

---

## 4. Exact Gaussian MLE, method='mle_exact' / 'mle_conditional'

Maximizes the exact Gaussian log-likelihood of the AR(p) process.

Key features:
    - PACF reparameterization via tanh for stationarity
    - Exact initial distribution (optional diffuse switch)
    - Conditional likelihood via AR filtering (lfilter)
    - Full likelihood includes Toeplitz covariance for first p observations

Returns:
    - intercept
    - AR coefficients
    - innovation variance
    - full likelihood-based parameter estimates

---

# Covariance estimation

- CSS: OLS covariance from (X'X)^(-1)
- Yule-Walker: Toeplitz asymptotic covariance approximation
- Burg: no intrinsic covariance (requires external approximation)
- MLE: Fisher-information / inverse Hessian (implicit via optimizer or bootstrap)

---

# Notes

- All methods assume stationarity of the AR process.
- Burg and Yule-Walker estimate AR structure directly; CSS/MLE use regression
  or likelihood formulations.
- MLE and CSS are asymptotically equivalent under correct specification.
"""

from __future__ import absolute_import, print_function

import numpy as np
from scipy.sparse import isspmatrix

from kanly.time_series.auto_correlation_function import autocovariance_function
from kanly.time_series.autoregression.ar_mle import ar_mle as mle
from kanly.time_series.autoregression.burg import burg


def estimate_ar(y, lags=1, small_sample_correction=True, method='css', **kwargs):
    """
    Estimate autoregressive parameters using CSS or Yule-Walker.

    Parameters
    ----------
    y : array_like
        One-dimensional time series observations.
    lags : int, default=1
        Number of autoregressive lags to include.
    small_sample_correction : bool, default=True
        Whether to apply a small-sample degrees-of-freedom correction to the
        parameter covariance matrix.
    method : {'css', 'yule-walker'}, default='css'
        Estimation method.

        ``'css'`` estimates the AR model by conditional sum of squares, which is
        equivalent to OLS regression of ``y[t]`` on an intercept and its lagged
        values.

        ``'yule-walker'`` estimates the AR coefficients by solving the
        Yule-Walker equations using the Durbin-Levinson recursion.

    Returns
    -------
    params : ndarray
        Estimated parameters. For ``method='css'``, this includes the intercept
        followed by AR coefficients. For ``method='yule-walker'``, this contains
        only AR coefficients.
    cov_params : ndarray
        Estimated variance-covariance matrix of the parameter estimates.
    sigma_innov_sq : float
        Estimated innovation variance.
    """
    method = method.lower().strip()
    if isspmatrix(y):
        y = y.toarray().flatten()

    if method == 'css':
        values = css(y, lags, small_sample_correction, **kwargs)
    elif method == 'burg':
        values = burg(y, lags, **kwargs)
    elif method in ('yule-walker', 'yw', 'yulewalker'):
        values = yule_walker(y, lags, small_sample_correction)
    elif method == 'mle_exact':
        values = mle(y, lags, exact=True, **kwargs)
    elif method == 'mle_conditional':
        values = mle(y, lags, exact=False,  **kwargs)
    else:
        raise Exception("`method` must be 'yule-walker', 'mle_exact', 'mle_conditional', or 'css'")

    return values


def yule_walker(y, lags=1, small_sample_correction=True):
    """
    Estimate AR coefficients using the Yule-Walker equations.

    This function estimates an AR(lags) model from sample autocovariances. The
    Yule-Walker linear system has a Toeplitz autocovariance matrix, and this
    implementation solves it efficiently using the Durbin-Levinson recursion.

    The fitted model is

    ``y[t] = phi[0] * y[t-1] + ... + phi[lags-1] * y[t-lags] + e[t]``

    No intercept is estimated by this function. If an intercept is desired, use
    ``css`` or demean the series before calling this function.

    Parameters
    ----------
    y : array_like
        One-dimensional time series observations.
    lags : int, default=1
        Order of the autoregressive model. Must be an integer greater than or
        equal to 1.
    small_sample_correction : bool, default=True
        Whether to apply a small-sample adjustment to the covariance matrix.

    Returns
    -------
    phi : ndarray
        Estimated AR coefficients of shape ``(lags,)``. The coefficient
        ``phi[i]`` corresponds to lag ``i + 1``.
    cov_params : ndarray
        Estimated variance-covariance matrix of the AR coefficient estimates.
    sigma_innov_sq : float
        Estimated innovation variance for the fitted AR(lags) model.

    Notes
    -----
    The function first computes autocovariances ``gamma[0], ..., gamma[lags]``.
    It then recursively solves the Yule-Walker equations without explicitly
    forming or inverting the full Toeplitz matrix during coefficient estimation.

    At recursion step ``m``, ``kappa`` is the reflection coefficient, also known
    as the partial autocorrelation at lag ``m``. The AR coefficients from the
    previous order are updated in place to obtain the coefficients for the AR(m)
    model.

    The Durbin-Levinson recursion has time complexity ``O(lags^2)``, compared
    with ``O(lags^3)`` for a generic dense linear solve.

    After estimating the coefficients, this function constructs the Toeplitz
    autocovariance matrix and uses its pseudo-inverse to compute an approximate
    parameter covariance matrix.
    """
    assert isinstance(lags, int) and lags >= 1

    n = len(y)

    # Store AR coefficients using one-based indexing internally:
    # phi[1] is the lag-1 coefficient, ..., phi[lags] is the lag-lags
    # coefficient. phi[0] is unused.
    phi = np.zeros(lags + 1)

    # Sample autocovariances gamma[0], ..., gamma[lags].
    const = y.mean()
    y = y - const

    # DO NOT adjust for sample size - it breaks the Toeplitz structure!
    gamma = autocovariance_function(y, nlags=lags, adjusted=False)

    # Initial prediction error variance for an AR(0) model.
    sigma_innov_sq = gamma[0]

    # Durbin-Levinson recursion. Each iteration updates the AR solution from
    # order m - 1 to order m.
    for m in range(1, lags + 1):
        # Compute the numerator of the reflection coefficient.
        numerator = gamma[m]
        for j in range(1, m):
            numerator -= phi[j] * gamma[m - j]

        # Reflection coefficient / partial autocorrelation at lag m.
        kappa = numerator / sigma_innov_sq

        # Copy the old coefficients because the update for phi[j] depends on
        # both phi[j] and phi[m-j] from the previous recursion order.
        old_phi = phi.copy()

        # Update lower-order AR coefficients for the AR(m) solution.
        for j in range(1, m):
            phi[j] = old_phi[j] - kappa * old_phi[m - j]

        # The newest coefficient is the reflection coefficient.
        phi[m] = kappa

        # Update the innovation variance for the AR(m) model.
        sigma_innov_sq *= (1 - kappa ** 2)

    # Build the Toeplitz autocovariance matrix used for the large-sample
    # covariance approximation of the Yule-Walker AR coefficient estimates.
    #
    # XX[1:, 1:] is the lags x lags matrix:
    #
    #   gamma[0] gamma[1] ... gamma[lags-1]
    #   gamma[1] gamma[0] ... gamma[lags-2]
    #   ...
    #
    # XX is allocated as (lags + 1) x (lags + 1) so that slicing [1:, 1:]
    # lines up with the one-based coefficient indexing used above.
    XX = np.zeros((lags + 1, lags + 1))
    for i in range(lags + 1):
        for j in range(lags + 1):
            XX[i, j] = gamma[abs(i - j)]

    # Pseudo-inverse of the Toeplitz autocovariance matrix. This is used rather
    # than np.linalg.inv so that the function is more tolerant of singular or
    # nearly singular autocovariance estimates.
    ncp = np.linalg.pinv(XX[1:, 1:])

    # Large-sample covariance estimate for the Yule-Walker AR coefficients.
    # The denominator applies the requested small-sample adjustment.
    cov_params = (
            ncp
            * sigma_innov_sq
            / (n - lags + small_sample_correction * (n - (lags + 1)))
    )

    arparams = phi[1:]
    const *= (1 - sum(arparams))
    params = np.hstack([const, arparams, sigma_innov_sq])
    param_names = ['Intercept'] + [f'L{j}' for j in range(1, lags + 1)] + ['sigma2']

    return {
        'params': params,
        'cov_params': cov_params,
        'param_names': param_names,
        'arparams': arparams,
        'sigma2': sigma_innov_sq,
        'const': const
    }


def css(y, lags=1, small_sample_correction=True):
    """
    Estimate an autoregressive model by conditional sum of squares.

    This function fits an AR(lags) model by ordinary least squares, using the
    observations from ``lags`` onward as the regression target and the previous
    ``lags`` observations as predictors. An intercept is included as the first
    coefficient.

    The fitted model is

    ``y[t] = beta[0] + beta[1] * y[t-1] + ... + beta[lags] * y[t-lags] + e[t]``

    for ``t = lags, ..., n - 1``.

    Parameters
    ----------
    y : array_like
        One-dimensional time series observations.
    lags : int, default=1
        Number of autoregressive lags to include in the model.
    small_sample_correction : bool, default=True
        Whether to apply a degrees-of-freedom correction to the estimated
        parameter covariance matrix.

    Returns
    -------
    beta : ndarray
        Estimated regression coefficients of shape ``(lags + 1,)``.
        ``beta[0]`` is the intercept, and ``beta[l]`` is the coefficient on
        lag ``l``.
    cov_params : ndarray
        Estimated variance-covariance matrix of the coefficients, equal to the
        pseudo-inverse of the normal-equation matrix multiplied by the estimated
        residual variance and the requested small-sample adjustment.
    sigma_innov_sq : float
        Estimated innovation variance, computed from the conditional residual
        sum of squares.

    Notes
    -----
    This implementation avoids explicitly constructing the full lagged design
    matrix. Instead, it directly computes the sufficient statistics ``X'X`` and
    ``X'y`` for the OLS normal equations.

    The estimator is called "conditional" because the sum of squares is computed
    conditional on the first ``lags`` observations.
    """
    assert isinstance(lags, int) and lags >= 1

    n = len(y)

    # Compute X'y, where X contains an intercept column followed by lagged
    # values y[t-1], ..., y[t-lags], and y[lags:] is the regression target.
    Xy = np.zeros(lags + 1)
    Xy[0] = y[lags:].sum()
    for l in range(1, lags + 1):
        Xy[l] = np.dot(y[lags:], y[lags - l:-l])

    # Compute X'X without explicitly forming the lag matrix X.
    # The first row and first column correspond to the intercept.
    XX = np.zeros((lags + 1, lags + 1))
    XX[0, 0] = n - lags

    # Cross-products between the intercept column and each lag column.
    for l in range(1, lags + 1):
        XX[l, 0] = y[lags - l:-l].sum()
        XX[0, l] = XX[l, 0]

    # Cross-products between lag columns. Only the lower triangle is computed
    # directly, then copied to the upper triangle by symmetry.
    for l1 in range(1, lags + 1):
        for l2 in range(1, l1 + 1):
            XX[l1, l2] = np.dot(y[lags - l1:-l1], y[lags - l2:-l2])
            XX[l2, l1] = XX[l1, l2]

    # Solve the normal equations using the Moore-Penrose pseudo-inverse.
    # This is more forgiving than np.linalg.inv when XX is singular or nearly
    # singular, though it still relies on the normal-equation formulation.
    ncp = np.linalg.pinv(XX)
    beta = ncp.dot(Xy)

    # Compute residual sum of squares using the quadratic form:
    #
    #   RSS = y'y - 2 beta'X'y + beta'X'X beta
    #
    # This avoids explicitly forming fitted values or residuals.
    rss = (
            beta.dot(XX).dot(beta)
            - 2 * Xy.dot(beta)
            + np.dot(y[lags:], y[lags:])
    )

    # Estimate innovation variance from the conditional residual sum of squares.
    # This quantity is reported separately from cov_params and is not itself
    # small-sample corrected here.
    sigma_innov_sq = rss / (n - lags)

    # Cov(beta) = sigma^2 * (X'X)^(-1), using the pseudo-inverse here.
    #
    # The multiplicative adjustment changes the covariance scaling when
    # small_sample_correction=True.
    cov_params = (
            ncp
            * sigma_innov_sq
            * (n - lags)
            / (n - lags + small_sample_correction * (n - (lags + 1)))
    )
    params = np.hstack([beta, sigma_innov_sq])
    param_names = ['Intercept'] + [f'L{j}' for j in range(1,lags+1)] + ['sigma2']

    return {'params': params,
            'param_names': param_names,
            'arparams': params[1:-1],
            'const': params[0],
            'sigma2': params[-1],
            'cov_params': cov_params,
            }


# if __name__ == '__main__':
#     from kanly.api import simulate_sarima, SARIMAX, AUTOREG
#     from kanly.time_series.sarimax.sarimax_internal import sarimax_internal
#
#     n = 50
#     y = -5.0 + simulate_sarima(n=n, ar=[.5, .1], seed=0, burnin=1000, sigma2=2)
#     lags = 2
#
#     print('burg     ', burg(y, lags)['params'].round(4))
#
#     #print('mle_cond ', ar_mle(y, lags, conditional=True)['params'].round(4))
#     print('css      ', css(y, lags)['params'].round(4))
#     print('autoreg   ', np.hstack([(fit_auto:=AUTOREG(y, lags=lags)).params.values, fit_auto.scale]).round(4))
#     print('mle-cond ', mle(y, 2, exact=False, maxiter=100)['params'].values.round(4))
#
#     #print('mle_exact ', ar_mle(y, lags, conditional=False)['params'].round(4))
#     print('yw/dl   ', yule_walker(y, lags)['params'].round(4))
#     print('sarimax ', (fit_sarimax:=SARIMAX(y, order=(lags, 0, 0), trend='c', nlags=100, xtol=1e-8, ftol=1e-12)).params.values.round(4))
#     print('mle ', (fit_mle_full:=mle(y, 2, exact=True, xtol=1e-8, ftol=1e-12))['params'].values.round(4))
#
#     print(fit_sarimax.llf, fit_mle_full['llf'])
#
#     from kanly.api import timer, clear_timers
#     clear_timers()
#
#     T = 200
#
#     css(y, lags)
#     timer('burg')
#     for i in range(T):
#         burg(y, lags)
#     timer('burg')
#
#     css(y, lags)
#     timer('css')
#     for i in range(T):
#         css(y, lags)
#     timer('css')
#
#     yule_walker(y, lags)
#     timer('yw')
#     for i in range(T):
#         yule_walker(y, lags)
#     timer('yw')
#
#     mle(y, lags)
#     timer('mle-exact')
#     for i in range(T):
#         mle(y, lags, exact=True)
#     timer('mle-exact')
#
#     mle(y, lags)
#     timer('mle-conditional')
#     for i in range(T):
#         mle(y, lags, exact=False)
#     timer('mle-conditional')
#
#     timer('sarimax')
#     for i in range(T):
#         SARIMAX(y, order=(lags,0,0), conditional=True)
#     timer('sarimax')
#
#     timer('sarimax_internal')
#     for i in range(T):
#         sarimax_internal(y, order=(lags,0,0))
#     timer('sarimax_internal')
