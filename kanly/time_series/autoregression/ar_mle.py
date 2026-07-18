"""
Gaussian maximum-likelihood estimation for stationary AR(p) models.

This module provides utilities for estimating autoregressive models via
conditional or exact Gaussian maximum likelihood. The implementation supports:

    1. Exact Gaussian likelihood using the stationary distribution of the
       initial observations.
    2. Conditional likelihood (conditioning on the first p observations).
    3. Concentrated likelihood estimation for the innovation variance.
    4. Numerical covariance estimation via the observed information matrix.
    5. Stable unconstrained optimization using partial autocorrelation
       parameterization.

The optimizer operates in an unconstrained parameter space by mapping
autoregressive coefficients to partial autocorrelations (PACF), then applying
a hyperbolic tangent transform:

    optimizer space <-> PACF <-> stationary AR coefficients

This guarantees that all iterates correspond to stationary AR processes.

Model
-----
For an AR(p) process:

.. math::

    y_t = \\mu + \\rho_1 y_{t-1} + \\cdots + \\rho_p y_{t-p} + \\varepsilon_t

with

.. math::

    \\varepsilon_t \\sim N(0, \\sigma^2)

the exact Gaussian likelihood uses the stationary covariance structure of the
first ``p`` observations, while the conditional likelihood treats the first
``p`` observations as fixed.

References
----------
Brockwell, P. J., & Davis, R. A. (1991).
    *Time Series: Theory and Methods*.

Hamilton, J. D. (1994).
    *Time Series Analysis*.
"""

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm

from kanly.optimize.bfgs_bounded_quasi_newton import BFGSPQNResults, bfgs_pqn
from kanly.time_series.autoregression.burg import burg
from kanly.time_series.autoregression.stationary_ar_covariance import ar_initial_gamma


def pacf_to_ar(kappa):
    """
    Convert partial autocorrelations to AR coefficients.

    Uses the Levinson-Durbin recursion to map partial autocorrelations
    (reflection coefficients) to stationary AR coefficients.

    Parameters
    ----------
    kappa : array_like, shape (p,)
        Partial autocorrelations. Stationarity requires each element to lie
        strictly inside ``(-1, 1)``.

    Returns
    -------
    ar : ndarray, shape (p,)
        Autoregressive coefficients corresponding to ``kappa``.

    Notes
    -----
    The recursion is:

    .. math::

        \\phi_j^{(m)}
        =
        \\phi_j^{(m-1)}
        -
        \\kappa_m \\phi_{m-j}^{(m-1)}

    with

    .. math::

        \\phi_m^{(m)} = \\kappa_m

    This mapping is bijective between stationary AR coefficients and PACF
    coefficients inside ``(-1,1)``.
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    p = kappa.shape[0]

    if p == 0:
        return np.empty(0, dtype=np.float64)

    ar = np.empty((p, p), dtype=np.float64)
    ar[0, 0] = kappa[0]

    for m in range(1, p):
        km = kappa[m]
        prev = ar[m - 1, :m]

        ar[m, :m] = prev - km * prev[::-1]
        ar[m, m] = km

    return ar[p - 1, :p].copy()


def ar_to_pacf(ar):
    """
    Convert AR coefficients to partial autocorrelations.

    Uses the reverse Levinson-Durbin recursion to recover reflection
    coefficients from stationary AR coefficients.

    Parameters
    ----------
    ar : array_like, shape (p,)
        Stationary autoregressive coefficients.

    Returns
    -------
    kappa : ndarray, shape (p,)
        Partial autocorrelations (reflection coefficients).

    Notes
    -----
    The final AR coefficient at each recursion level equals the partial
    autocorrelation:

    .. math::

        \\kappa_m = \\phi_m^{(m)}

    The inverse recursion recovers lower-order AR coefficients recursively.
    """
    ar = np.asarray(ar, dtype=np.float64)
    p = ar.shape[0]

    if p == 0:
        return np.empty(0, dtype=np.float64)

    work = ar.copy()
    kappa = np.empty(p, dtype=np.float64)

    for m in range(p - 1, -1, -1):
        km = work[m]
        kappa[m] = km

        if m == 0:
            break

        denom = 1.0 - km * km

        work[:m] = (
            work[:m] + km * work[m - 1::-1]
        ) / denom

    return kappa


# -------------------------------------------------------------------
# unconstrained optimization space <-> stationary AR coefficients
# -------------------------------------------------------------------

def opt_to_ar(theta):
    """
    Map unconstrained optimizer parameters to stationary AR coefficients.

    The transformation proceeds in two steps:

        1. ``theta`` is mapped from :math:`\\mathbb{R}` to ``(-1,1)``
           using ``tanh``.
        2. The resulting PACF coefficients are converted to stationary
           AR coefficients using the Levinson-Durbin recursion.

    Parameters
    ----------
    theta : array_like
        Unconstrained optimizer parameters.

    Returns
    -------
    ar : ndarray
        Stationary autoregressive coefficients.
    """
    theta = np.asarray(theta, dtype=np.float64)

    # map R -> (-1, 1)
    kappa = np.tanh(theta)

    # PACF -> AR
    return pacf_to_ar(kappa)


def ar_to_opt(ar, eps=1e-12):
    """
    Map stationary AR coefficients to unconstrained optimizer coordinates.

    The transformation proceeds in two steps:

        1. Convert AR coefficients to PACF coefficients.
        2. Apply ``arctanh`` to map ``(-1,1)`` to :math:`\\mathbb{R}`.

    Parameters
    ----------
    ar : array_like
        Stationary autoregressive coefficients.

    eps : float, default=1e-12
        Numerical clipping parameter used to avoid overflow in
        ``arctanh``.

    Returns
    -------
    theta : ndarray
        Unconstrained optimizer parameters.
    """
    ar = np.asarray(ar, dtype=np.float64)

    # AR -> PACF
    kappa = ar_to_pacf(ar)

    # numerical safety
    kappa = np.clip(kappa, -1.0 + eps, 1.0 - eps)

    # map (-1,1) -> R
    return np.arctanh(kappa)


def ar_log_likelihood(y, mu, rho, sigma2=1.0,
                      exog=None, exog_params=None,
                      exact=True, return_full=False):
    """
    Gaussian log-likelihood for a stationary AR(p) process.

    Parameters
    ----------
    y : array_like
        Observed time series.

    mu : float or None
        Unconditional mean parameter. If ``None``, the intercept is
        initialized using the sample mean and AR coefficients.

    rho : array_like
        Autoregressive coefficients.

    sigma2 : float, default=1.0
        Innovation variance.

    exact : bool, default=True
        If True, compute the exact Gaussian likelihood using the
        stationary covariance of the first ``p`` observations.

        If False, compute the conditional likelihood conditioning on
        the first ``p`` observations.

    return_full : bool, default=False
        If True, additionally return intermediate quadratic forms and
        likelihood contributions.

    Returns
    -------
    llf : float
        Gaussian log-likelihood.

    OR

    llf : float
    ssr : float
    mu : float
    quadratic_forms : tuple
    likelihood_terms : tuple

        Additional intermediate quantities when ``return_full=True``.

    Notes
    -----
    The exact likelihood decomposes into:

    .. math::

        \\ell = \\ell_1 + \\ell_2

    where:

    * ``llf1`` is the likelihood contribution of the first ``p``
      observations under stationarity.
    * ``llf2`` is the conditional Gaussian likelihood of the remaining
      observations.

    Cholesky factorization is used for numerical stability when evaluating
    the stationary covariance contribution.
    """
    y = np.asarray(y)
    if exog is not None:
        y = y - exog.dot(exog_params)
    if mu is None:
        mu = np.mean(y)
    y = y - mu

    p = len(rho)
    n = len(y)
    qf1 = 0.0
    llf1 = 0.0

    if exact:
        # covariance matrix of first p observations
        Gamma = ar_initial_gamma(rho, p, sigma2)

        y0 = y[:p]

        try:
            U, is_lower = cho_factor(Gamma)
            z = cho_solve((U, is_lower), y0)
        except Exception:
            return -1.0e12

        qf1 = y0.dot(z)

        llf1 = -0.5 * (
            p * np.log(2.0 * np.pi)
            + 2.0 * np.log(np.diag(U)).sum()
            + qf1
        )

    # conditional innovations
    innov = y[p:].copy()
    for l in range(1, p + 1):
        innov -= rho[l - 1] * y[p - l:-l]

    qf2 = innov.dot(innov) / sigma2
    log_sigma2 = np.log(sigma2)

    llf2 = -0.5 * ((n - p) * (np.log(2 * np.pi) + log_sigma2) + qf2)

    ssr = qf1 + qf2
    llf = llf1 + llf2

    if return_full:
        return llf, ssr, mu, (qf1, qf2), (llf1, llf2)
    else:
        return llf


def ar_mle(y, lags=1, exact=True,
           exog=None,
           concentrate_scale=True,
           estimate_const=True,
           compute_cov=False,
           unconditional_mean=False,
           maxiter=100,
           xtol=1e-6,
           ftol=1e-8,
           gtol=1e-4,
           B0=100,
           onesided_fd=True,
           x0=None,
           debug=False,
           test_level=.95):
    """
    Estimate a stationary AR(p) model by Gaussian maximum likelihood.

    Parameters
    ----------
    y : array_like
        Observed time series.

    lags : int, default=1
        Number of autoregressive lags.

    exog : np.ndarray-like
        exogenous regressors

    exact : bool, default=True
        If True, maximize the exact Gaussian likelihood using the
        stationary distribution of the initial observations.

        If False, maximize the conditional likelihood conditioning on
        the first ``p`` observations.

    concentrate_scale : bool, default=True
        If True, profile out the innovation variance parameter from the
        optimization problem.

    estimate_const : bool, default=True
        If True, estimate the conditional intercept parameter directly.

        If False, recover it from the sample mean and AR coefficients.

    compute_cov : bool, default=True
        If True, compute the observed information matrix and parameter
        covariance estimates numerically.

    maxiter : int, default=20
        Maximum number of optimizer iterations.

    xtol : float, default=1e-6
        Parameter convergence tolerance.

    ftol : float, default=1e-8
        Objective convergence tolerance.

    gtol : float, default=1e-4
        Gradient convergence tolerance.

    B0 : float, default=100
        Initial Hessian scaling for the quasi-Newton optimizer.

    x0 : array_like or None, default=None
        Optional starting values in unconstrained optimizer coordinates.

    Returns
    -------
    results : dict
        Dictionary containing:

        * ``params`` : estimated parameters
        * ``arparams`` : AR coefficients
        * ``sigma2`` : innovation variance estimate
        * ``llf`` : maximized log-likelihood
        * ``cov_params`` : covariance matrix
        * ``bse`` : standard errors
        * ``result`` : optimizer output
        * ``loglike`` : callable likelihood function

    Notes
    -----
    Stationarity is enforced through PACF parameterization.

    When ``concentrate_scale=True``, the innovation variance is recovered
    analytically after optimization.

    Covariance estimation uses the observed Fisher information matrix:

    .. math::

        I(\\theta) = -H(\\theta)

    where ``H`` is the Hessian of the log-likelihood.
    """
    y = np.asarray(y)

    has_exog = exog is not None
    k_exog = 0
    if has_exog:
        # todo sparse?
        if np.ndim(exog) == 1:
            exog = exog.reshape((-1,1))
        k_exog = exog.shape[1]

    if concentrate_scale:
        if estimate_const:
            def obj_func(params):
                rho = opt_to_ar(params[1:lags + 1])
                exog_params = params[1 + lags:1 + lags + k_exog]
                return ar_log_likelihood(
                    y, params[0], rho,
                    exog=exog, exog_params=exog_params,
                    sigma2=1.0,
                    exact=exact,
                    return_full=False
                )
        else:
            def obj_func(params):
                rho = opt_to_ar(params[:lags])
                exog_params = params[lags:lags + k_exog]
                return ar_log_likelihood(
                    y, None, rho,
                    exog=exog, exog_params=exog_params,
                    sigma2=1.0,
                    exact=exact,
                    return_full=False
                )
    else:
        if estimate_const:
            def obj_func(params):
                rho = opt_to_ar(params[1:1 + lags])
                exog_params = params[1 + lags:1 + lags + k_exog]
                return ar_log_likelihood(
                    y,
                    params[0],
                    rho,
                    exog=exog, exog_params=exog_params,
                    sigma2=np.exp(params[-1]),
                    exact=exact,
                    return_full=False
                )
        else:
            def obj_func(params):
                rho = opt_to_ar(params[:lags])
                exog_params = params[lags:lags+k_exog]
                return ar_log_likelihood(
                    y,
                    None,
                    rho,
                    exog=exog, exog_params=exog_params,
                    sigma2=np.exp(params[-1]),
                    exact=exact,
                    return_full=False
                )

    if x0 is None:

        # starting guess
        if has_exog:
            beta0 = np.linalg.lstsq(exog, y - y.mean(), rcond=None)[0]
            mu0 = (y - exog.dot(beta0)).mean()
            ar0 = burg(y, lags)['arparams']
        else:
            mu0 = y.mean()
            ar0 = burg(y, lags)['arparams']
            beta0 = []

        # transform AR -> unconstrained optimizer coordinates
        theta0 = ar_to_opt(ar0)

        x0 = np.array(
            ([mu0] if estimate_const else [])
            + list(theta0)
            + (list(beta0) if has_exog else [])
            + ([np.log(np.var(y))] if not concentrate_scale else [])
        )
    else:
        x0 = np.asarray(x0)

    result: BFGSPQNResults = bfgs_pqn(
        obj_func,
        x0,
        maximize=True,
        maxiter=maxiter,
        xtol=xtol,
        ftol=ftol,
        gtol=gtol,
        B0=B0,
        onesided_fd=onesided_fd,
    )

    if debug:
        print(f'llf {result.fun=}')
        print(f'{result.x=}')
        print(f'{result.xerr=}, {result.gnorm=}, {result.ferr=}, {result.converged=}')

    opt_params = result.x.copy()

    if concentrate_scale:
        opt_params = np.hstack([opt_params, 0.0])

    # recover AR coefficients from optimizer coordinates
    if estimate_const:
        theta_hat = opt_params[1:1 + lags]
    else:
        theta_hat = opt_params[:lags]

    arparams = opt_to_ar(theta_hat)
    exog_params = opt_params[estimate_const + lags:estimate_const + lags + k_exog]

    if estimate_const:
        mu_hat = opt_params[0]
    else:
        mu_hat = y.mean()

    if concentrate_scale:
        sigma2_hat = 1.0
    else:
        sigma2_hat = np.exp(opt_params[-1])

    mu_unconditional = mu_hat
    mu_intercept = mu_hat * (1 - sum(arparams))
    mu_hat = mu_unconditional if unconditional_mean else mu_intercept

    params = np.hstack([mu_hat, arparams, exog_params, sigma2_hat])

    llf, ssr, *_ = ar_log_likelihood(
        y,
        mu_unconditional,
        rho=params[1:lags + 1],
        sigma2=params[-1],
        exog=exog, exog_params=params[1 + lags:1 + lags + k_exog],
        exact=exact,
        return_full=True
    )

    if concentrate_scale:
        params[-1] = ssr / (len(y) - (not exact) * lags)

        llf, *_ = ar_log_likelihood(
            y,
            mu_unconditional,
            rho=params[1:1+lags],
            sigma2=params[-1],
            exog=exog, exog_params=params[1 + lags:1 + lags + k_exog],
            exact=exact,
            return_full=True
        )

    param_names = (['const'] + [f'ar.L{j}' for j in range(1, lags + 1)]
                   + [f'x{j}' for j in range(k_exog)] + ['sigma2'])
    df_summary = pd.DataFrame(index=param_names, data={'param': params})

    cov_params, bse, _cov_params, _bse = None, None, None, None

    if compute_cov:
        I = information_matrix(
            lambda x: ar_log_likelihood(
                y,
                mu=(
                    (x[0] if unconditional_mean else x[0] / (1 - sum(x[1:-1])))
                    if estimate_const else
                    (params[0] if unconditional_mean else params[0] / (1 - sum(x[1:-1])))
                ),
                rho=x[1:1 + lags],
                exog=exog, exog_params=x[1 + lags:1 + lags + k_exog],
                sigma2=params[-1] if concentrate_scale else x[-1],
                exact=exact
            ),
            params,
            eps=1e-6
        )

        cov_param_names = param_names

        # if concentrate_scale:
        #     I = I[
        #         1 - estimate_const:-1,
        #         1 - estimate_const:-1
        #     ]
        #
        #     cov_param_names = cov_param_names[
        #         1 - estimate_const:-1
        #     ]

        cov_params_ = np.linalg.pinv(I)

        cov_params = pd.DataFrame(
            cov_params_,
            columns=cov_param_names,
            index=cov_param_names
        )

        bse_ = np.sqrt(np.diag(cov_params_))
        bse = pd.DataFrame(bse_, index=cov_param_names)
        df_summary['std err'] = bse_
        df_summary['z'] = params / np.where(bse_ != 0, bse_, np.inf)
        df_summary['p>|z|'] = 2.0 * norm.sf(df_summary['z'].abs())
        cv = norm.ppf(1 - test_level / 2)
        df_summary['[.025, '] = params - cv * bse_
        df_summary[' 0.975]'] = params + cv * bse_

    def loglike(params, exact=exact, return_full=False):
        """
        Evaluate the AR log-likelihood at arbitrary parameters.
        """
        params = np.asarray(params)

        return ar_log_likelihood(
            y, mu=params[0], rho=params[1:1 + lags],
            sigma2=params[-1], exog=exog, exog_params=params[1 + lags:1 + lags + k_exog],
            exact=exact, return_full=return_full,
        )

    return {
        'params': pd.Series(params, index=param_names),
        '_params': params,
        'const_unconditional': mu_unconditional,
        'const_regression': mu_intercept,
        'arparams': arparams,
        'exog_params': exog_params,
        'result': result,
        'sigma2': params[-1],
        'cov_params': cov_params,
        '_cov_params': _cov_params,
        'bse': bse,
        '_bse': _bse,
        'llf': llf,
        'loglike': loglike,
        'summary_df': df_summary,

        'settings': {
            'compute_cov': compute_cov,
            'exact': exact,
            'concentrate_scale': concentrate_scale,
            'estimate_const': estimate_const,
            'maxiter': maxiter,
            'xtol': xtol,
            'ftol': ftol,
            'gtol': gtol,
            'B0': B0,
            'x0': x0,
        }
    }


def numerical_hessian(f, theta, eps=1e-5):
    """
    Compute a finite-difference Hessian matrix.

    Parameters
    ----------
    f : callable
        Scalar-valued function.

    theta : array_like
        Parameter vector at which to evaluate the Hessian.

    eps : float, default=1e-5
        Finite-difference step size.

    Returns
    -------
    H : ndarray
        Numerical Hessian matrix.

    Notes
    -----
    Central finite differences are used:

    .. math::

        H_{ij}
        =
        \\frac{
            f(x+h_i+h_j)
            -
            f(x+h_i-h_j)
            -
            f(x-h_i+h_j)
            +
            f(x-h_i-h_j)
        }{4h^2}
    """
    theta = np.asarray(theta, dtype=float)
    n = len(theta)
    H = np.zeros((n, n))

    f0 = f(theta)

    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n)
            ej = np.zeros(n)

            ei[i] = eps
            ej[j] = eps

            fpp = f(theta + ei + ej)
            fpm = f(theta + ei - ej)
            fmp = f(theta - ei + ej)
            fmm = f(theta - ei - ej)

            H[i, j] = (
                fpp - fpm - fmp + fmm
            ) / (4 * eps * eps)

            H[j, i] = H[i, j]

    return H


def information_matrix(loglik, theta_hat, eps=1e-5):
    """
    Compute the observed Fisher information matrix.

    Parameters
    ----------
    loglik : callable
        Log-likelihood function.

    theta_hat : array_like
        Parameter vector at which to evaluate the information matrix.

    eps : float, default=1e-5
        Finite-difference step size used for the Hessian calculation.

    Returns
    -------
    I : ndarray
        Observed Fisher information matrix.

    Notes
    -----
    The observed information matrix is:

    .. math::

        I(\\theta)
        =
        -H(\\theta)

    where ``H`` is the Hessian of the log-likelihood.
    """
    H = numerical_hessian(loglik, theta_hat, eps=eps)
    return -H
