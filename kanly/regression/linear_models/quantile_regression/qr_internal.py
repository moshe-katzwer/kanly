"""Numerical core of quantile regression — IRLS loop, line search, and covariance estimators.

This module implements:

- ``qr_internal``  — the main IRLS (Iteratively Reweighted Least Squares) driver
  that estimates the tau-quantile regression coefficients, optionally with IV
  via a control-function approach.
- Helper functions for the IRLS loop: MAD scale estimation, line search,
  quadratic approximation solves, residual/error computation.
- ``get_var_covar`` — analytical covariance estimators (IID and heteroscedastic-
  robust) based on a kernel density estimate of the residual sparsity at zero.

For bootstrap covariance, see ``model.py``.
"""
from __future__ import absolute_import, print_function

import time

import numpy as np
from scipy.sparse import csc_matrix, isspmatrix_csc, isspmatrix
from scipy.sparse import diags
from scipy.stats import gaussian_kde
from scipy.stats import norm as norm_dist

from kanly.utils.linalg_utils import get_matrix_inverse_internal, DenseThreshold, csc_matrix_by_column_array_broadcast
from kanly.regression.linear_models.sparse_iv_first_stage2 import iv_first_stage2
from kanly.regression.linear_models.quantile_regression.constants import (
    QR_COV_TYPE_IID, DEFAULT_QR_LOSS_FUNCTION, DEFAULT_QR_MAX_ITER, DEFAULT_QR_X_TOL, DEFAULT_QR_SMOOTHING_K,
    DEFAULT_QR_MIN_RESID_CLIP, DEFAULT_QR_F_TOL, DEFAULT_QR_LINE_SEARCH, QR_COV_TYPE_ROBUST,
    DEFAULT_QR_RESIDUAL_INCLUSION, DEFAULT_QR_RESIDUAL_INCLUSION_ORDER,
    DEFAULT_QR_SCALE_DESIGN_MATRIX, DEFAULT_QR_STEP_GRID, DEFAULT_QR_DENSE_THRESHOLD_MB)
from kanly.regression.linear_models.quantile_regression.loss_functions import get_loss_deriv
from kanly.utils.util import print_iter_info

# Φ⁻¹(0.75) ≈ 0.6745 — the standard denominator for converting MAD to an
# approximately unbiased standard-deviation estimate under normality.
DEFAULT_MAD_DENOM = norm_dist.ppf(.75)


def get_mad_scale(resid, center=0.0, mad_denom=DEFAULT_MAD_DENOM):
    """Estimate the scale of a residual distribution using the Median Absolute Deviation.

    Computes  MAD / Φ⁻¹(0.75)  so that the result is a consistent estimator
    of the standard deviation when residuals are normally distributed.

    Args:
        resid (ndarray): Residual vector, shape (n,).
        center (float): Centre value subtracted before computing absolute
            deviations.  Defaults to 0.0.
        mad_denom (float): Normalising constant.  Defaults to
            ``DEFAULT_MAD_DENOM`` (= Φ⁻¹(0.75) ≈ 0.6745).

    Returns:
        float: MAD-based scale estimate.
    """
    # med_resid = np.median(resid) # TODO?
    scale = np.median(np.abs(resid - center)) / mad_denom
    return scale


def true_cost_func(z, tau):
    """Evaluate the exact (non-smoothed) quantile check-function cost.

    Computes  Σ ρ_τ(zᵢ) / 2  where  ρ_τ(z) = z·(τ − 𝟙{z<0}).
    The factor of 2 matches the convention used in the surrogate losses so
    that the true cost is comparable to the smoothed cost during convergence
    monitoring.

    Args:
        z (ndarray): Residual vector (y − Xβ), shape (n,).
        tau (float): Quantile level in (0, 1).

    Returns:
        float: Sum of check-function losses divided by 2.
    """
    idx_neg = z < 0
    return ((tau - 1) * np.sum(z[idx_neg]) + tau * np.sum(z[~idx_neg])) / 2


def get_line_search_step(y, X, cost_func, beta, beta_last, resid_prev, a_grid=DEFAULT_QR_STEP_GRID):
    """Perform a grid line search to find the best convex step between two iterates.

    Evaluates the cost at positions  β_last + a·(β − β_last)  for each
    step size ``a`` in ``a_grid`` and returns the parameters and residuals
    corresponding to the minimum.  This guarantees (in practice) a decrease
    in the quantile objective_function at each IRLS step.

    Args:
        y (ndarray or sparse): Response vector/matrix, shape (n,) or (n, 1).
        X (ndarray or sparse): Design matrix, shape (n, p).
        cost_func (callable): Function mapping a residual vector to a scalar
            cost (e.g. the check-function or weighted check-function cost).
        beta (ndarray or sparse): Newly proposed coefficient vector from the
            current IRLS step, shape (p,) or (p, 1).
        beta_last (ndarray or sparse): Coefficient vector from the previous
            IRLS step, shape (p,) or (p, 1).
        resid_prev (ndarray): Residuals at ``beta_last``, shape (n,).
        a_grid (list of float): Grid of step-size candidates.  Defaults to
            ``DEFAULT_QR_STEP_GRID``.

    Returns:
        tuple:
            - **beta** (ndarray or sparse): Best-step coefficient vector.
            - **best_step_size** (float): Selected step size from ``a_grid``.
            - **resid** (ndarray): Residuals at the selected step, shape (n,).
            - **fittedvalues** (ndarray): Fitted values at the selected step,
              shape (n,).
            - **cost** (float): Objective value at the selected step.
    """
    resid_new, fittedvalues_new = _get_resid_and_fittedvalues(X, y, beta)
    # Evaluate cost for each candidate step (interpolating residuals is exact for linear models).
    costs = [cost_func(resid_prev + _a * (resid_new - resid_prev)) for _a in a_grid]
    best = np.argmin(costs)
    best_step_size = a_grid[best]
    beta = beta_last + best_step_size * (beta - beta_last)

    cost = costs[best]
    resid = resid_prev + best_step_size * (resid_new - resid_prev)
    fittedvalues = (y.toarray().flatten() if isspmatrix(y) else y) - resid

    return beta, best_step_size, resid, fittedvalues, cost


def _format_data_inputs(X, y, dense_threshold_mb=DEFAULT_QR_DENSE_THRESHOLD_MB):
    """Normalise design matrix and response to a consistent sparse or dense format.

    If the design matrix is small enough to fit in memory as a dense array
    (determined by ``dense_threshold_mb``), it is converted to avoid the
    overhead of sparse arithmetic in the IRLS loop.  For sparse matrices,
    both ``X`` and ``y`` are coerced to CSC format.

    Args:
        X (ndarray or sparse): Design matrix, shape (n, p).
        y (ndarray or sparse): Response vector, shape (n,) or (n, 1).
        dense_threshold_mb (float or None): Memory ceiling in megabytes for
            converting ``X`` to dense.  Pass ``None`` to always keep sparse.
            Defaults to ``DEFAULT_QR_DENSE_THRESHOLD_MB``.

    Returns:
        tuple:
            - **X** (ndarray or csc_matrix): Formatted design matrix.
            - **y** (ndarray or csc_matrix): Formatted response.
            - **is_sparse** (bool): True if ``X`` remains sparse after formatting.
            - **nobs** (int): Number of observations (rows of ``X``).
    """
    is_sparse = isspmatrix(X)
    if is_sparse and DenseThreshold.is_below_threshold(X, dense_threshold_mb):
        X = X.toarray()

    if is_sparse:
        if not isspmatrix_csc(X):
            X = csc_matrix(X)
        if not isspmatrix(y):
            y = csc_matrix(y).reshape((-1, 1)).tocsc()
    else:
        if isspmatrix(y):
            y = y.toarray().flatten()

    nobs = X.shape[0]

    return X, y, is_sparse, nobs


def _solve_quadratic_approx(X, y, weights=None):
    """Solve the weighted least squares normal equations (XᵀWX)β = XᵀWy.

    This is the inner solve of each IRLS iteration.  When ``weights`` is
    None, the standard (unweighted) OLS solution is computed.

    Args:
        X (ndarray or csc_matrix): Design matrix, shape (n, p).
        y (ndarray or csc_matrix): Response, shape (n,) or (n, 1).
        weights (ndarray or None): Non-negative observation weights, shape
            (n,).  If None, all weights are treated as 1.

    Returns:
        tuple:
            - **ncp** (ndarray): Normalised covariance matrix (XᵀWX)⁻¹,
              shape (p, p).
            - **beta** (ndarray or csc_matrix): Coefficient estimate, shape
              (p,) or (p, 1) matching the sparsity of ``X``.
    """
    is_sparse = isspmatrix(X)

    if weights is None:
        XtX = X.transpose().dot(X)
        Xty = X.transpose().dot(y)
    else:
        # Pre-multiply each row of X by its weight to form Xw = diag(w)·X.
        if is_sparse:
            Xw = csc_matrix_by_column_array_broadcast(X, weights)
        else:
            Xw = X * np.array(weights).reshape((-1, 1))
        XtX = Xw.transpose().dot(X)
        Xty = Xw.transpose().dot(y)

    if is_sparse:
        XtX = XtX.toarray()
        Xty = Xty.toarray().flatten()

    ncp = get_matrix_inverse_internal(XtX)

    beta = ncp.dot(Xty)
    if is_sparse:
        beta = csc_matrix(beta).reshape((-1, 1))

    return ncp, beta


def _get_resid_and_fittedvalues(X, y, beta):
    """Compute residuals and fitted values from current coefficient estimates.

    Args:
        X (ndarray or sparse): Design matrix, shape (n, p).
        y (ndarray or sparse): Response vector, shape (n,) or (n, 1).
        beta (ndarray or sparse): Coefficient vector, shape (p,) or (p, 1).

    Returns:
        tuple:
            - **resid** (ndarray): Residual vector y − Xβ, shape (n,).
            - **fittedvalues** (ndarray): Fitted values Xβ, shape (n,).
    """
    fittedvalues = X.dot(beta)
    resid = y - fittedvalues
    # Flatten sparse result matrices to 1-D dense arrays for downstream use.
    if isspmatrix(resid):
        resid = resid.toarray().flatten()
        fittedvalues = fittedvalues.toarray().flatten()
    return resid, fittedvalues


def _get_error(beta, beta_last):
    """Compute the max elementwise convergence error between two iterate vectors.

    Uses a relative error  |β/β_last − 1|  for components where |β_last| > 1,
    and an absolute error  |β − β_last|  otherwise.  This mixed criterion
    avoids false convergence on large-magnitude coefficients and avoids
    numerical issues on near-zero ones.

    Args:
        beta (ndarray or sparse): Current coefficient vector, shape (p,).
        beta_last (ndarray or sparse): Previous coefficient vector, shape (p,).

    Returns:
        float: Maximum convergence error across all components.
    """
    if isspmatrix(beta_last):
        beta_last = beta_last.toarray().flatten()
    if isspmatrix(beta):
        beta = beta.toarray().flatten()

    errors = np.abs(beta - beta_last)
    # Switch to relative error for large-magnitude coefficients.
    errors[np.abs(beta_last) > 1] = np.abs(beta / beta_last - 1.0)[np.abs(beta_last) > 1]
    err = np.max(errors)
    return err


def qr_internal(y, X, tau, debug=False, start_params=None, Z=None, is_endog_regressor=None,
                smoothing_k=DEFAULT_QR_SMOOTHING_K, min_resid_clip=DEFAULT_QR_MIN_RESID_CLIP,
                xtol=DEFAULT_QR_X_TOL, ftol=DEFAULT_QR_F_TOL, max_iter=DEFAULT_QR_MAX_ITER,
                line_search=DEFAULT_QR_LINE_SEARCH, loss=DEFAULT_QR_LOSS_FUNCTION,
                residual_inclusion=DEFAULT_QR_RESIDUAL_INCLUSION,
                residual_inclusion_order=DEFAULT_QR_RESIDUAL_INCLUSION_ORDER,
                weights=None, scale_design_matrix=DEFAULT_QR_SCALE_DESIGN_MATRIX,
                dense_threshold_mb=DEFAULT_QR_DENSE_THRESHOLD_MB):
    """Run IRLS to estimate the tau-quantile regression coefficients.

    Implements the Iteratively Reweighted Least Squares algorithm for quantile
    regression.  At each step the smooth surrogate loss is used to form IRLS
    weights  w_i = ψ(r_i)/r_i  and the next iterate is obtained by solving a
    WLS problem (XᵀWX)β = XᵀWy.  Optionally a grid line search is performed
    to guarantee a decrease in the quantile objective_function.

    If instruments ``Z`` are provided, a control-function first stage is run
    via ``iv_first_stage2`` before the IRLS loop (IV for quantile regression
    is experimental).

    Convergence is declared when either:
    - The max elementwise β change is below ``xtol``, **or**
    - The relative change in the quantile objective_function is below ``ftol``.

    Args:
        y (ndarray or sparse): Response vector, shape (n,) or (n, 1).
        X (sparse matrix): Design matrix, shape (n, p).
        tau (float): Quantile level strictly in (0, 1).
        debug (bool): If True, print an iteration table and convergence
            message.  Defaults to False.
        start_params (array-like or None): Initial coefficient vector of
            length p.  If None, the OLS solution is used as the starting
            point.
        Z (sparse matrix or None): Instrument matrix for IV quantile
            regression.  If provided, a control-function first stage replaces
            ``X`` with the instrumented design matrix before the IRLS loop.
        is_endog_regressor (array-like or None): Boolean mask of length p
            indicating which columns of ``X`` are endogenous.  Required when
            ``Z`` is not None.
        smoothing_k (float): Half-width of the smooth quadratic region in the
            surrogate loss.  Smaller values give a closer approximation to the
            true check loss.  Defaults to ``DEFAULT_QR_SMOOTHING_K``.
        min_resid_clip (float): Minimum absolute residual magnitude used in
            the IRLS weight denominator.  Defaults to
            ``DEFAULT_QR_MIN_RESID_CLIP``.
        xtol (float): Convergence threshold on the max elementwise β change.
            Defaults to ``DEFAULT_QR_X_TOL``.
        ftol (float): Convergence threshold on the relative change in the
            quantile objective_function.  Defaults to ``DEFAULT_QR_F_TOL``.
        max_iter (int): Maximum number of IRLS iterations.  Defaults to
            ``DEFAULT_QR_MAX_ITER``.
        line_search (bool): If True, run a grid line search at each step.
            Defaults to ``DEFAULT_QR_LINE_SEARCH``.
        loss (str or type): Surrogate loss function.  String values
            ``'huber'``, ``'softl1'``, ``'smoothcup'`` are accepted, or pass
            a ``QuantileRegressionLossFunction`` subclass directly.  Defaults
            to ``DEFAULT_QR_LOSS_FUNCTION``.
        residual_inclusion (bool): If True and instruments are provided,
            include IV residual powers in the augmented design matrix
            (control-function approach).  Defaults to
            ``DEFAULT_QR_RESIDUAL_INCLUSION``.
        residual_inclusion_order (int): Polynomial order of residual powers
            to include when ``residual_inclusion`` is True.  Defaults to
            ``DEFAULT_QR_RESIDUAL_INCLUSION_ORDER``.
        weights (ndarray or None): Non-negative observation weights, shape
            (n,).  Used for WLS-quantile regression (e.g. bootstrap
            resampling).
        scale_design_matrix (bool): Reserved for future column scaling of X;
            currently unused.  Defaults to ``DEFAULT_QR_SCALE_DESIGN_MATRIX``.
        dense_threshold_mb (float or None): Memory threshold in MB above
            which ``X`` is kept sparse.  Defaults to
            ``DEFAULT_QR_DENSE_THRESHOLD_MB``.

    Returns:
        dict: Result dictionary with the following keys:

        - ``'params'`` (ndarray): Estimated coefficient vector, shape (p,).
        - ``'resid'`` (ndarray): Final residuals y − Xβ̂, shape (n,).
        - ``'fittedvalues'`` (ndarray): Final fitted values Xβ̂, shape (n,).
        - ``'iterations'`` (int): Number of IRLS iterations performed.
        - ``'converged'`` (bool): Whether a convergence criterion was met.
        - ``'error'`` (float): Final convergence error value.
        - ``'cost'`` (float): Final smoothed objective_function value (halved).
        - ``'true_cost'`` (float): Final exact (non-smoothed) check-function
          cost (halved).
        - ``'pseudo_rsquared'`` (float): Koenker-Machado pseudo-R² measuring
          improvement over the unconditional quantile.
        - ``'weights'`` (ndarray): Final IRLS weights ψ(r)/r, shape (n,).
        - ``'normalized_covariance_parameters'`` (ndarray): (XᵀWX)⁻¹ from
          the final IRLS iteration, shape (p, p).
        - ``'message'`` (str): Human-readable convergence message.
        - ``'exog_col_map'`` (list of tuple): Column index mapping from the
          (possibly augmented) design matrix back to the original columns;
          used by ``convert_exog_col_map_to_col_names2``.
        - ``'exog_instrumented'`` (ndarray or sparse): The (possibly
          augmented) design matrix used in the final IRLS iteration.
    """
    _t = time.time()

    # IV branch: replace X with the instrumented design matrix via the control-function approach.
    if Z is not None:
        iv_info = iv_first_stage2(
            X, Z, is_endog_regressor=is_endog_regressor, debug=debug, _time=_t, residual_inclusion=residual_inclusion,
            weights=weights, residual_inclusion_order=residual_inclusion_order)
        X = iv_info.exog_instrumented
        exog_col_map = iv_info.exog_col_map
    else:
        # Identity mapping: each output column corresponds to one input column.
        exog_col_map = [(i,) for i in range(X.shape[1])]

    assert 0 < tau < 1
    assert smoothing_k >= min_resid_clip >= 0

    # Build the quantile cost function; weighted version for bootstrap/WLS resampling.
    if weights is None:
        cost_func = lambda z: np.sum(np.where(z < 0, z * (tau - 1), z * tau)) / 2
    else:
        cost_func = lambda z: np.sum(weights * np.where(z < 0, z * (tau - 1), z * tau)) / 2

    loss_func = get_loss_deriv(loss)

    X, y, is_sparse, nobs = _format_data_inputs(X, y, dense_threshold_mb)

    # scale_design_matrix = False
    # from kanly.utils.linalg_utils import scale_in_place
    # if scale_design_matrix:
    #     x_scales = scale_in_place(X)

    if start_params is None:
        ncp, beta = _solve_quadratic_approx(X, y, weights)

    else:
        start_params = np.asarray(start_params).flatten()
        if len(start_params) != X.shape[1]:
            raise Exception(f"`start_params` must have length {X.shape[1]}")
        beta = np.asarray(start_params)
        if is_sparse:
            beta = csc_matrix(beta).reshape((-1, 1))

    resid, fittedvalues = _get_resid_and_fittedvalues(X, y, beta)
    cost = cost_func(resid)

    if debug:
        width = 40
        print(
            f'\n' +
            f'=' * width + "\n" +
            f'Quantile Regression\n' +
            f'-' * width + "\n" +
            f'tau:           {"%.3f" % tau}\n' +
            f'\n' +
            f'loss:          {str(loss)}\n' +
            f'smoothing k:   {"%.6f" % smoothing_k if smoothing_k >= 1e-6 else "%.1e" % smoothing_k}\n' +
            f'clip:          {"%.6f" % min_resid_clip if min_resid_clip >= 1e-6 else "%.1e" % min_resid_clip}\n' +
            f'\n' +
            f'nobs:          {X.shape[0]}\n' +
            f'params:        {X.shape[1]}\n' +
            f'cost:          {"%.4e" % cost}\n' +
            f'\n' +
            f'max_iter:      {max_iter}\n'
            f'ftol:          {"%.1e" % ftol}\n' +
            f'xtol:          {"%.1e" % xtol}\n' +
            f'line search:   {line_search}\n' +
            f'=' * width + "\n\n"
        )

    message = 'Did not converged!'
    converged = False

    for itr in range(max_iter):

        try:
            # if itr == 0:
            #     c0 *= np.median(np.mean(resid ** 2))
            # scale = get_mad_scale(resid)
            # w = loss_func.weights(resid / scale, smoothing_k, tau, min_resid_clip)
            # Compute IRLS weights w_i = ψ(r_i)/r_i from the smooth surrogate loss.
            w = loss_func.weights(resid, smoothing_k, tau, min_resid_clip)
            if weights is not None:
                # Multiply in external observation weights (e.g. bootstrap resampling weights).
                w *= weights
            # gradient = -0.5 * (X.transpose().dot(csc_matrix(w * resid).reshape((-1, 1)))).toarray().flatten()

            # W = diags(w, shape=(X.shape[0], X.shape[0]))
            # XpX = X.transpose().dot(W).dot(X).transpose()
            # ncp = get_matrix_inverse_internal(XpX.toarray(), return_csc=True)
            # Xty = X.transpose().dot(W).dot(y)

            beta_last = beta.copy()
            # if X.shape[1] > 1:
            #     beta = ncp.dot(Xty)
            # else:
            #     b = Xty * ncp
            #     beta = csc_matrix(b)

            ncp, beta = _solve_quadratic_approx(X, y, w)
            cost_old = cost

            if line_search:
                beta, best_step_size, resid, fittedvalues, cost \
                    = get_line_search_step(y, X, cost_func, beta, beta_last, resid)

            else:
                best_step_size = None
                resid, fittedvalues = _get_resid_and_fittedvalues(X, y, beta)
                cost = cost_func(resid)

            err = _get_error(beta, beta_last)

            if debug:
                iter_info = [
                    {'name': 'iter', 'len': 6, 'format': '%6d', 'value': itr},
                    {'name': 'cost', 'len': 14, 'format': '%14.4e', 'value': cost},
                    {'name': '% dCost', 'len': 12, 'format': '%12.1e', 'value': cost / cost_old - 1},
                    # {'name': '|grad|', 'len': 12, 'format': '%12.1e', 'value': np.max(np.abs(gradient))},
                    {'name': '|dx|', 'len': 12, 'format': '%12.2e', 'value': err},
                    {'name': '% (y-Xb)<0', 'len': 14, 'format': '%13.2f%%',
                     'value': 100 * np.count_nonzero(resid < 0) / nobs},
                    {'name': 'step', 'len': 8, 'format': '%8s',
                     'value': 'NA' if best_step_size is None else "%.2f" % best_step_size},
                    {'name': 'time', 'len': 9, 'format': '%8.2fs', 'value': (time.time() - _t)},
                    # {'name': '', 'len': 15, 'format': '%15.2f',
                    # 'value': 100*np.count_nonzero((resid < smoothing_k) & (resid > -smoothing_k)) / nobs},
                    # {'name': '', 'len': 15, 'format': '%15.4f',
                    # 'value': 100*np.count_nonzero((resid < min_resid_clip) & (resid > -min_resid_clip)) / nobs},
                ]
                if itr == 0:
                    print_iter_info(iter_info, is_header=True)
                print_iter_info(iter_info)

            if cost >= cost_old:
                # beta = beta_last
                # fittedvalues = fittedvalues_old
                # resid = resid_old
                # cost = cost_old
                # converged = True
                # message = 'Converged: IRLS iterate did not yield a lower objective_function value.'
                # if debug:
                #    print('IRLS iterate did not yield a lower objective_function value.')
                pass

            # x-tolerance: stop when the max coefficient change falls below xtol.
            if itr >= 1 and err < xtol:
                message = 'Converged: x error %.2e less than `x_tol` %.2e' % (err, xtol)
                converged = True

            # f-tolerance: stop when the relative change in the objective_function falls below ftol.
            rel_cost_change = np.abs(cost - cost_old) / max(1.0, np.abs(cost_old))
            if itr >= 1 and rel_cost_change < ftol:
                message = 'Converged: relative change in cost %.2e less than `f_tol` %.2e' \
                          % (rel_cost_change, ftol)
                converged = True
            if debug and (converged or itr == max_iter - 1):
                print_iter_info(iter_info, is_footer=True)
            if converged:
                break

        except KeyboardInterrupt:
            print("\nProcess interrupted, breaking...\n")
            break

        except Exception as e:
            raise e

    if debug:
        print(f'\n{message}\n')

    #fittedvalues = fittedvalues.toarray().flatten()
    if isspmatrix(y):
        y_flat = y.toarray().flatten()
    else:
        y_flat = y

    true_cost = true_cost_func(resid, tau)
    # Koenker-Machado pseudo-R²: 1 − cost(fitted) / cost(unconditional tau-quantile).
    pseudo_rsquared = 1.0 - true_cost / true_cost_func(y_flat - np.quantile(y_flat, tau), tau)

    # resid, fittedvalues = _get_resid_and_fittedvalues(X, y, beta)
    # cost = cost_func(resid)

    return {
        'params': beta.toarray().flatten() if isspmatrix(beta) else beta,
        'resid': resid,
        'fittedvalues': fittedvalues,
        'iterations': itr,
        'converged': converged,
        'error': err,
        'cost': cost,
        'true_cost': true_cost,
        'pseudo_rsquared': pseudo_rsquared,
        'weights': w,
        'normalized_covariance_parameters': ncp,
        'message': message,
        'exog_col_map': exog_col_map,
        # 'gradient': gradient
        'exog_instrumented': X,
    }


def get_var_covar(cov_type, exog, resid, tau, small_sample_correct=1.0):
    """Compute an analytical covariance matrix for quantile regression coefficients.

    Uses a Gaussian kernel density estimate of the residual sparsity
    (conditional density at zero)  f̂₀ = KDE(ε)(0)  with Silverman's
    bandwidth  h = 0.9·σ̂/n^{1/5}.  Two estimators are supported:

    - **IID** (homoscedastic):  Var(β̂) = τ(1−τ)/f₀² · (XᵀX)⁻¹ · small_sample_correct
    - **ROBUST** (heteroscedastic sandwich):
      Var(β̂) = (XᵀX)⁻¹ · (XᵀDX) · (XᵀX)⁻¹ / f₀²
      where D = diag(τ · 𝟙{r≥0} + (1−τ) · 𝟙{r<0}).

    For bootstrap covariance estimation, use the ``BOOTSTRAP`` path in
    ``model.py`` instead of calling this function.

    Args:
        cov_type (str): Covariance type; one of ``QR_COV_TYPE_IID`` or
            ``QR_COV_TYPE_ROBUST``.
        exog (sparse matrix): (Possibly instrumented) design matrix used in
            the final IRLS iteration, shape (n, p).
        resid (ndarray): Final IRLS residuals y − Xβ̂, shape (n,).
        tau (float): Quantile level in (0, 1).
        small_sample_correct (float): Small-sample correction factor applied
            to the IID estimator only (typically n / df_resid).  Defaults to
            1.0 (no correction).

    Returns:
        ndarray: Estimated covariance matrix of β̂, shape (p, p).

    Raises:
        Exception: If ``cov_type`` is not ``'IID'`` or ``'ROBUST'``.
    """
    n = exog.shape[0]

    scale = get_mad_scale(resid)
    # Silverman's rule-of-thumb bandwidth for KDE of the residual distribution.
    h_bw = .9 * scale / n ** 0.2

    # kappa = np.median(np.abs(resid)) / norm_dist.ppf(.75)
    #
    # h = n ** (-1.0 / 3) \
    #     * norm_dist.ppf(1.0 - .05 / 2) ** (2.0 / 3) \
    #     * (1.5 * norm_dist.pdf(norm_dist.ppf(tau)) ** 2
    #        / (2 * norm_dist.ppf(tau) ** 2 + 1)
    #        ) ** (1.0 / 3)
    #
    # delta = kappa * (norm_dist.ppf(min(tau + h, .9999)) - norm_dist.ppf(max(tau - h, .0001)))
    # h_bw = delta

    f0 = float(gaussian_kde(resid, bw_method=h_bw)(0.0).item())

    A = get_matrix_inverse_internal(exog.transpose().dot(exog).toarray())
    if cov_type == QR_COV_TYPE_IID:
        A *= tau * (1 - tau) * f0 ** -2 * small_sample_correct
        return A
    elif cov_type == QR_COV_TYPE_ROBUST:
        D = diags(np.where(resid < 0, 1 - tau, tau))
        B = exog.transpose().dot(D).dot(exog).toarray()
        V = A.dot(B).dot(A)
        V /= f0 ** 2
        return V
    else:
        raise Exception(f"Cov type {cov_type} not supported")

    # n = exog.shape[0]
    #
    # a_diag = (tau - (resid < 0).astype(float))
    # A = csc_matrix_by_column_array_broadcast(exog, a_diag)
    # A = A.transpose().dot(A).toarray() / n
    #
    # kappa = np.median(np.abs(resid)) / norm_dist.ppf(.75)
    #
    # h = n ** (-1.0 / 3) \
    #     * norm_dist.ppf(1.0 - .05 / 2) ** (2.0 / 3) \
    #     * (1.5 * norm_dist.pdf(norm_dist.ppf(tau)) ** 2
    #        / (2 * norm_dist.ppf(tau) ** 2 + 1)
    #        ) ** (1.0 / 3)
    #
    # delta = kappa * (norm_dist.ppf(min(tau + h, .9999)) - norm_dist.ppf(max(tau - h, .0001)))
    #
    # for _ in range(100):
    #     d_diag = ((-delta < resid) & (resid < delta)).astype(float)
    #     if np.count_nonzero(d_diag) == 0:
    #         delta *= 1.1
    #     else:
    #         break
    #
    # D = csc_matrix_by_column_array_broadcast(exog, d_diag)
    # D = D.transpose().dot(D).toarray() / (2.0 * n * delta)
    #
    # try:
    #     Dinv = np.linalg.inv(D)
    # except:
    #     return None
    #
    # var_covar = Dinv.dot(A).dot(Dinv) / n
    # var_covar *= small_sample_correct
    #
    # return var_covar
