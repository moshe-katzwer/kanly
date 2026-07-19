"""Numerical core of robust linear regression via M-estimation.

Implements the IRLS (Iteratively Reweighted Least Squares) loop for
M-estimators:

1. Warm-start from an OLS solution (or caller-supplied ``start_params``).
2. At each iteration, compute scaled residuals r/σ̂, update IRLS weights
   via the norm function, and solve a sparse WLS system.
3. Check convergence using a combined absolute/relative β change criterion.
4. Delegate final covariance estimation to ``variance_covariance.py``.

Key functions
-------------
``get_mad_scale``   — MAD-based scale estimator (σ̂ = median|r|/Φ⁻¹(0.75)).
``rlm_internal``    — Main IRLS driver; returns a results dict.
``compute_naive_cost`` — Baseline cost for pseudo-R² (robust location model).
"""
from __future__ import absolute_import, print_function

import time

import numpy as np
from scipy.sparse import csc_matrix, identity
from scipy.sparse.linalg import spsolve
from scipy.stats import norm as norm_dist

from kanly.regression.linear_models.robust.constants import (
    DEFAULT_RLM_COV_TYPE, DEFAULT_RLM_MAX_ITER, DEFAULT_RLM_X_TOL, DEFAULT_RLM_M)
from kanly.regression.linear_models.robust.robust_norm_functions import get_norm
from kanly.regression.linear_models.robust.variance_covariance import get_rlm_variance_covariance, NOT_COMPUTED
from kanly.utils.linalg_utils import csc_matrix_by_column_array_broadcast
from kanly.utils.util import print_iter_info

# Φ⁻¹(0.75) ≈ 0.6745 — the normal-distribution quantile used to make the MAD
# a consistent estimator of σ under Gaussian errors: σ̂ = median|r| / 0.6745.
MAD_DENOM = norm_dist.ppf(.75)


def get_mad_scale(resid, center=0.0):
    """Estimate the error scale using the Median Absolute Deviation (MAD).

    Returns σ̂ = median(|rᵢ − center|) / Φ⁻¹(0.75), which is a consistent
    and robust estimator of the standard deviation σ when residuals are
    approximately Gaussian.  The denominator Φ⁻¹(0.75) ≈ 0.6745 makes the
    estimate unbiased under normality.

    Args:
        resid (ndarray): Residual vector, shape (n,).
        center (float): Location around which to measure deviations.
            Defaults to 0.0 (no recentering).

    Returns:
        float: Positive scale estimate σ̂.
    """
    #med_resid = np.median(resid) # TODO?
    scale = np.median(np.abs(resid - center)) / MAD_DENOM
    return scale


def rlm_internal(endog, exog, has_intercept, weights=None, x_tol=DEFAULT_RLM_X_TOL, max_iter=DEFAULT_RLM_MAX_ITER,
                 cov_type=DEFAULT_RLM_COV_TYPE, cov_kwds=None, start_params=None,
                 M=DEFAULT_RLM_M, debug=False, compute_cov=True, force_scale=None):
    """Run IRLS to fit a robust linear model (M-estimator).

    Initialises from an OLS warm-start (or ``start_params``), then iterates:

    1. Compute current residuals and MAD scale.
    2. Update IRLS weights: wᵢ = M.weights(rᵢ/σ̂).
    3. Combine with optional observation weights and solve the sparse WLS.
    4. Check convergence on the max absolute/relative change in β.

    After convergence (or ``max_iter`` exhaustion), delegates covariance
    estimation to ``get_rlm_variance_covariance`` and computes a pseudo-R²
    relative to a robust location-only baseline.

    Args:
        endog (array-like or sparse): Response vector y, shape (n,) or (n, 1).
        exog (array-like or sparse): Design matrix X, shape (n, p).
        has_intercept (bool): Whether X contains an intercept column.  Used
            to compute ``df_model`` (= p − 1 when True, p when False).
        weights (array-like, optional): Observation-level WLS weights.  If
            None, all observations are treated equally.
        x_tol (float): Convergence threshold on max elementwise β change.
            Default ``DEFAULT_RLM_X_TOL``.
        max_iter (int): Maximum IRLS iterations.  Default ``DEFAULT_RLM_MAX_ITER``.
        cov_type (str): Covariance type: ``'H1'``, ``'H2'``, ``'H3'``,
            ``'SANDWICH'``, or ``'BOOTSTRAP'`` (BOOTSTRAP raises
            ``NotImplementedError`` here; use ``model.py`` instead).
        cov_kwds (dict, optional): Reserved for future covariance keywords.
        start_params (array-like, optional): Initial β vector.  If None, an
            OLS solve is used as the starting point.
        M (str, type, or RobustNormFunction): Norm function.  Default
            ``DEFAULT_RLM_M`` (``'HuberT'``).  Resolved via ``get_norm``.
        debug (bool): If True, print iteration-level convergence diagnostics.
        compute_cov (bool): If False, skip covariance estimation (faster).
        force_scale (float, optional): Fix the scale at this value instead of
            re-estimating with MAD at each iteration.  Useful for simulation
            studies or when an external scale estimate is available.

    Returns:
        dict: Results dictionary with the following keys:

            - ``'coef'`` (csc_matrix): Estimated coefficient vector β̂,
              shape (p, 1).
            - ``'cost'`` (float): Final M-estimator cost Σρ(rᵢ/σ̂).
            - ``'pseudo_rsquared'`` (float): 1 − cost/naive_cost, analogous
              to R²; NaN if naive_cost is zero or undefined.
            - ``'irls_weights'`` (ndarray): Final IRLS weights, shape (n,).
            - ``'var_covar'`` (ndarray or None): Covariance matrix (p, p), or
              None if ``compute_cov=False``.
            - ``'df_model'`` (int): Model degrees of freedom.
            - ``'df_resid'`` (int): Residual degrees of freedom.
            - ``'iteration_info'`` (dict): Convergence diagnostics
              (``num_iters``, ``max_iter``, ``tol``, ``error``,
              ``converged``, ``force_scale``).
            - ``'scale'`` (float): Final σ̂ estimate.
            - ``'resid'`` (ndarray): Final residuals, shape (n,).
            - ``'fittedvalues'`` (ndarray): Fitted values ŷ, shape (n,).
            - ``'fit_elapsed'`` (float): Wall-clock time in seconds.
    """

    _t = time.time()

    if cov_kwds is None:
        cov_kwds = dict()

    M = get_norm(M)

    endog = csc_matrix(endog).reshape((-1,1))
    exog = csc_matrix(exog)

    if not compute_cov:
        cov_type = NOT_COMPUTED

    is_weighted = weights is not None
    if not is_weighted:
        weights = 1.0

    I = identity(exog.shape[1], format='csc')
    if start_params is None:
        # OLS warm-start: β₀ = (XᵀX)⁻¹Xᵀy via sparse direct solve.
        ncv = spsolve(exog.transpose().dot(exog).transpose(), I)
        ncv = csc_matrix(ncv.reshape((exog.shape[1], exog.shape[1])))
        beta = ncv.dot(exog.transpose().dot(endog)).toarray().flatten()
    else:
        beta = np.array(start_params).flatten()

    beta = csc_matrix(beta).transpose()

    if debug:
        width = 40
        print(
            f'\n' +
            f'=' * width + "\n" +
            f'Robust Regression (M-Estimation)\n' +
            f'-' * width + "\n" +
            f'\n' +
            f'loss ("M"):     {str(M)}\n' +
            f'\n' +
            f'nobs:           {exog.shape[0]}\n' +
            f'params:         {exog.shape[1]}\n' +
            f'\n' +
            f'max_iter:       {max_iter}\n'
            f'x_tol:          {"%.1e" % x_tol}\n' +
            f'=' * width + "\n\n"
        )

    converged = False
    cost_old = np.nan
    err = np.inf
    scale = 1.0
    for itr in range(max_iter):

        try:

            resid = (endog - exog.dot(beta)).toarray().flatten()
            cost = M.rho(resid / scale).sum()

            if force_scale is None:
                scale = get_mad_scale(resid)
            else:
                scale = force_scale

            if debug:
                iter_info = [
                    {'name': 'iter', 'len': 6, 'format': '%6d', 'value': itr},
                    {'name': 'cost', 'len': 14, 'format': '%14.4e', 'value': cost},
                    {'name': '% dCost', 'len': 12, 'format': '%12.1e', 'value': cost / cost_old - 1},
                    {'name': '|dx|', 'len': 12, 'format': '%12.2e', 'value': err},
                    {'name': 'scale', 'len': 13, 'format': '%13.4e', 'value': scale},
                    {'name': 'time', 'len': 9, 'format': '%8.2fs', 'value': (time.time() - _t)},
                ]
                if itr == 0:
                    print_iter_info(iter_info, is_header=True)
                print_iter_info(iter_info)

            # IRLS weight update: wᵢ = M.weights(rᵢ/σ̂) = ψ(rᵢ/σ̂)/(rᵢ/σ̂).
            irls_weights = M.weights(resid / scale)
            # Combine IRLS weights with optional user-supplied WLS weights;
            # take the square root because spsolve solves X̃ᵀX̃β = X̃ᵀỹ where
            # X̃ = diag(√w)·X and ỹ = diag(√w)·y.
            rt_wts = (weights * irls_weights) ** .5

            Xw = csc_matrix_by_column_array_broadcast(exog, rt_wts)
            yw = csc_matrix_by_column_array_broadcast(endog, rt_wts)

            ncv = spsolve(Xw.transpose().dot(Xw).transpose(), I)
            ncv = csc_matrix(ncv.reshape((exog.shape[1], exog.shape[1])))
            beta_new = ncv.dot(Xw.transpose().dot(yw))

            beta_arr, beta_new_arr = beta.toarray().flatten(), beta_new.toarray().flatten()
            errors = np.abs(beta_arr - beta_new_arr)
            # Switch to relative error for coefficients with |β| > 1
            # so that large-scale parameters are not penalised unfairly.
            idx = np.abs(beta_arr) > 1.0
            errors[idx] = np.abs(beta_new_arr[idx] / beta_arr[idx] - 1)
            err = max(errors)

            beta = beta_new.copy()
            cost_old = cost

            if err < x_tol:
                converged = True
                if debug:
                    print_iter_info(iter_info, is_footer=True)
                break

        except KeyboardInterrupt:
            print("\nProcess interrupted, breaking...\n")
            break

        except Exception as e:
            raise e

    iteration_info = {
        'num_iters': itr + 1,
        'max_iter': max_iter,
        'tol': x_tol,
        'error': err,
        'converged': converged,
        'force_scale': force_scale,
    }

    nobs = exog.shape[0]
    df_model, df_resid = exog.shape[1] - int(has_intercept), nobs - exog.shape[1]
    if compute_cov:
        var_covar = get_rlm_variance_covariance(df_model, df_resid, nobs, exog, resid, scale, M, cov_type)
    else:
        var_covar = None

    cost = np.sum(M.rho(resid / scale))
    naive_cost = compute_naive_cost(endog.toarray().flatten(), M, scale)
    try:
        pseudo_rsquared = 1.0 - cost / naive_cost
    except:
        pseudo_rsquared = np.nan

    return {
        'coef': beta,
        'cost': cost,
        'pseudo_rsquared': pseudo_rsquared,
        'irls_weights': irls_weights,
        'var_covar': var_covar,
        'df_model': df_model,
        'df_resid': df_resid,
        'iteration_info': iteration_info,
        'scale': scale,
        'resid': resid,
        'fittedvalues': endog.toarray().flatten() - resid,
        'fit_elapsed': time.time() - _t
    }


def compute_naive_cost(y, M, scale):
    """Compute the baseline M-estimator cost for a robust location model.

    Finds the M-estimator of location (iteratively reweighted mean) for y
    using the norm ``M`` and the provided ``scale``, then returns the resulting
    cost Σρ((yᵢ − μ̂)/σ̂).  This baseline cost is used to compute the
    pseudo-R²:

        pseudo_R² = 1 − cost(full model) / naive_cost

    The iterative loop converges when the location estimate changes by less
    than 1e-15 (effectively machine precision).

    Args:
        y (ndarray): Response vector, shape (n,).
        M (RobustNormFunction): Norm object providing ``rho`` and ``weights``.
        scale (float): Fixed scale estimate σ̂ (same value used in the full
            model fit, so the ratio is meaningful).

    Returns:
        float: Baseline M-estimator cost at the converged location estimate,
            or None if the loop does not converge within 100 iterations.
    """
    mu0 = y.mean()
    for _ in range(100):
        resid = y - mu0
        w = M.weights(resid / scale)
        mu = np.average(y, weights=w)
        if abs(mu - mu0) < 1e-15:
            return sum(M.rho(resid / scale))
        mu0 = mu
