from __future__ import absolute_import, print_function

import numpy as np
import time

from kanly.time_series.sarimax.constants import \
    COV_TYPE_OPG, COV_TYPE_APPROX, COV_TYPE_ROBUST_APPROX, COV_TYPE_NONE, \
    DEFAULT_SARIMAX_COV_TYPE, ARIMA_COV_TYPES


def get_cov_params(
        loglike_obs, params, cov_type=DEFAULT_SARIMAX_COV_TYPE, debug=False,
):
    """
    Dispatch to the requested covariance estimator for fitted parameters.

    Args:
        loglike_obs: Per-observation log-likelihood callable.
        params: Packed parameter vector in model parameter order.
        cov_type: Covariance estimator name.
        debug: Whether to print fitting diagnostics.

    Returns:
        2-tuple ``(cov_params, cov_elapsed)`` where ``cov_params`` is the
        ``(k, k)`` estimated parameter covariance matrix and ``cov_elapsed``
        is the wall-clock time in seconds spent computing it.
    """
    _time = time.time()
    if debug:
        print("Computing parameter covariance...", end='')
    if cov_type.lower() == COV_TYPE_OPG:
        cov_params = get_cov_params_opg(
            loglike_obs, params
        )
    elif cov_type.lower() == COV_TYPE_APPROX:
        cov_params = get_cov_params_approx(
            lambda x: loglike_obs(x).sum(), params
        )
    elif cov_type.lower() == COV_TYPE_ROBUST_APPROX:
        cov_params = get_cov_params_robust_approx(
            loglike_obs, params
        )
    elif cov_type.lower() == COV_TYPE_NONE:
        k = len(params)
        cov_params = np.full((k, k), np.nan)
    else:
        raise Exception(f'cov_type must be in {ARIMA_COV_TYPES}, you gave "{cov_type}"!')

    cov_elapsed = time.time() - _time
    if debug:
        print("%.2fs" % cov_elapsed)

    return cov_params, cov_elapsed


def get_cov_params_robust_approx(func, x, h=1e-6):
    """
    Compute robust covariance from an approximate Hessian and OPG score covariance.

    Args:
        func: Callable objective_function or log-likelihood function.
        x: Input array or parameter vector.
        h: Finite-difference step size.

    Returns:
        ``(k, k)`` parameter covariance matrix estimated via the sandwich
        estimator ``H^{-1} S H^{-1}``, where ``H`` is the approximate Hessian
        of the total log-likelihood and ``S`` is the OPG outer-product matrix.
    """
    bread = get_approx_hessian(lambda v: func(v).sum(), x)
    meat, n = get_cov_params_opg(func, x, h=h, return_n=True)
    return np.linalg.pinv(bread @ meat @ bread)


def get_cov_params_opg(func, x, h=1e-6, return_n=False):
    """
    Compute outer-product-of-gradients covariance estimates.

    Args:
        func: Callable objective_function or log-likelihood function.
        x: Input array or parameter vector.
        h: Finite-difference step size.
        return_n: Whether to also return the number of likelihood contributions.

    Returns:
        ``(k, k)`` covariance matrix estimated as the pseudo-inverse of the
        outer product of per-observation score vectors, or, if ``return_n=True``,
        a 2-tuple ``(cov, n)`` where ``n`` is the number of likelihood
        contributions.
    """
    x = np.asarray(x)
    num_x = len(x)
    cov = np.zeros((num_x, num_x))
    f_0 = func(x)
    n = len(f_0)
    for i in range(num_x):
        xi = x.copy()
        dx = max(abs(xi[i]), 1) * h
        xi[i] += dx
        f_i = (func(xi) - f_0) / dx
        for j in range(i + 1):
            if i == j:
                f_j = f_i
            else:
                xj = x.copy()
                dx = max(abs(xj[j]), 1) * h
                xj[j] += dx
                f_j = (func(xj) - f_0) / dx
            cov[i, j] = cov[j, i] = np.dot(f_j, f_i)
    cov = np.linalg.pinv(cov)
    if return_n:
        return cov, n
    else:
        return cov


def get_cov_params_approx(func, x):
    """
    Compute covariance from an approximate Hessian.

    Args:
        func: Callable objective_function or log-likelihood function.
        x: Input array or parameter vector.

    Returns:
        ``(k, k)`` covariance matrix estimated as the pseudo-inverse of the
        negative approximate Hessian.
    """
    hess = get_approx_hessian(func, x)
    cov = np.linalg.pinv(-hess)
    return cov


def get_approx_hessian(func, x):
    """
    Approximate the Hessian of a scalar function by finite differences.

    Args:
        func: Callable objective_function or log-likelihood function.
        x: Input array or parameter vector.

    Returns:
        Symmetric ``(nx, nx)`` array containing the finite-difference Hessian
        approximation, using central differences on the diagonal and a
        cross-difference formula for off-diagonal entries.
    """
    nx = len(x)
    f0 = func(x)
    h = 1e-4
    hess = np.zeros((nx, nx))
    for i in range(nx):
        for j in range(i + 1):
            if i == j:
                x0 = x.copy();
                x0[i] += h;
                x1 = x.copy();
                x1[i] -= h;
                hess[i, j] = (func(x0) - 2 * f0 + func(x1)) / (h ** 2)
            else:
                x0 = x.copy();
                x0[i] += h;
                x0[j] += h
                x1 = x.copy();
                x1[i] -= h;
                x1[j] += h
                x2 = x.copy();
                x2[i] += h;
                x2[j] -= h
                x3 = x.copy();
                x3[i] -= h;
                x3[j] -= h
                hess[i, j] = hess[j, i] = (
                        (func(x0) - func(x1) - func(x2) + func(x3)) / (4 * h ** 2)
                )
    return hess
