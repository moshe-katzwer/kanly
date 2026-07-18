"""Optimization internals for generalized method of moments fitting.

The functions here minimize the GMM quadratic objective_function for a fixed weighting
matrix and optionally update that weighting matrix for two-step or iterative
GMM.  They are called by ``SparseGeneralizedMethodOfMomentsModel.fit`` and are
not usually imported directly by users.
"""

from __future__ import absolute_import, print_function

import time

import numpy as np

from kanly.regression.generalized_method_of_moments.constants import (
    DEFAULT_GMM_F_TOL, DEFAULT_GMM_X_TOL, DEFAULT_GMM_G_TOL, DEFAULT_GMM_MAX_ITER, DEFAULT_GMM_DELTA,
    TWO_STEP, ITERATIVE
)
from kanly.regression.generalized_method_of_moments.gmm_variance_covariance import get_Omega
from kanly.utils.linalg_utils import get_matrix_inverse_internal
from kanly.regression.nonlinear_least_squares.optimize.quadratic_approx_subproblem_functions import steihaug
from kanly.utils.util import print_iter_info


def fit_gmm_outer_loop(nobs, moment_func_mean, moment_func_mean_jacobian, moment_func_obs, num_params, start_params, W,
                       max_iter, xtol, ftol, gtol, Delta, debug, method, iterative_gmm_max_iter, iterative_gmm_x_tol,
                       weights=None, _time=None):
    """Run the outer GMM loop that updates the weighting matrix.

    Args:
        nobs: Number of observations.
        moment_func_mean: Callable returning sample-average moments.
        moment_func_mean_jacobian: Callable returning the Jacobian of average
            moments.
        moment_func_obs: Callable returning observation-level moments.
        num_params: Number of parameters.
        start_params: Initial parameter vector.
        W: Initial weighting matrix.
        max_iter: Maximum inner optimizer iterations.
        xtol: Parameter-step convergence tolerance.
        ftol: Moment/objective_function convergence tolerance.
        gtol: Gradient convergence tolerance.
        Delta: Initial trust-region radius.
        debug: Whether to print optimization progress.
        method: One-step, two-step, or iterative GMM method.
        iterative_gmm_max_iter: Maximum number of weighting-matrix updates.
        iterative_gmm_x_tol: Stop tolerance for parameter changes across outer
            iterations.
        weights: Optional observation weights, used by bootstrap refits.
        _time: Optional start time for debug timing.

    Returns:
        Tuple ``(params, opt_result, n_iters, W)`` containing final parameters,
        the last inner optimization result, total inner iterations, and the
        final weighting matrix.
    """

    if _time is None:
        _time = time.time()

    if debug:
        print("\nEstimating parameters given initial weighting matrix...", end='')

    opt_result = gmm_minimize_internal(
        moment_func_mean, moment_func_mean_jacobian, W, num_params,
        start_params=start_params,
        max_iter=max_iter, xtol=xtol, ftol=ftol, gtol=gtol,
        Delta=Delta,
        debug=debug, weights=weights
    )
    if debug:
        print(f'done ({"%.2fs" % (time.time() - _time)})')

    params = opt_result['params']
    n_iters = opt_result['n_iters']

    if method in (TWO_STEP, ITERATIVE):
        # Two-step GMM updates W once; iterative GMM repeats until parameter
        # changes are small or the configured outer-loop limit is reached.
        for j in range(iterative_gmm_max_iter):
            if debug:
                print(f"\nRe-estimating parameters (run {j}) given new weighting matrix...", end='')
            Omega_temp, _ = get_Omega(moment_func_obs, nobs, params)
            W = get_matrix_inverse_internal(Omega_temp)
            opt_result = gmm_minimize_internal(
                moment_func_mean, moment_func_mean_jacobian, W, num_params,
                start_params=params,
                max_iter=max_iter, xtol=xtol, ftol=ftol, gtol=gtol,
                Delta=Delta, debug=debug, weights=weights
            )
            err = np.max(np.where(
                np.abs(params) > 1,
                np.abs(opt_result['params'] / params - 1),
                np.abs(params - opt_result['params'])
            ))

            if debug:
                print(f'done ({"%.2fs" % (time.time() - _time)}) [|dx| =  {"%.2e" % err} between iterates]\n')
            n_iters += opt_result['n_iters']
            params = opt_result['params']

            if method == TWO_STEP:
                break
            elif err < iterative_gmm_x_tol:
                if debug:
                    print('\nIterative outer loop complete!\n')
                break

    return params, opt_result, n_iters, W


def quadratic_approx(g, B, x_step):
    """Evaluate the local quadratic model for a trust-region step.

    Args:
        g: Gradient of the GMM objective_function at the current point.
        B: Gauss-Newton Hessian approximation.
        x_step: Proposed parameter step.

    Returns:
        Scalar predicted objective_function change under the quadratic approximation.
    """
    return (g.dot(x_step) + 0.5 * np.dot(x_step, B).dot(x_step)) * 2


def wtd_moment_quadratic_obj(moment_vals, W):
    """Evaluate the weighted GMM moment objective_function.

    Args:
        moment_vals: Sample-average moments.
        W: Weighting matrix.

    Returns:
        Scalar ``moment_vals' W moment_vals``.
    """
    return moment_vals.dot(W).dot(moment_vals)


def gmm_minimize_internal(
        avg_moment_func, jac_avg_moment_func, W=None, num_params=None, start_params=None, max_iter=DEFAULT_GMM_MAX_ITER,
        xtol=DEFAULT_GMM_X_TOL, ftol=DEFAULT_GMM_F_TOL, gtol=DEFAULT_GMM_G_TOL, Delta=DEFAULT_GMM_DELTA,
        debug=True, weights=None):
    """Minimize the GMM objective_function for a fixed weighting matrix.

    Args:
        avg_moment_func: Callable returning sample-average moments.
        jac_avg_moment_func: Callable returning the moment Jacobian.
        W: Weighting matrix for the quadratic objective_function.
        num_params: Number of parameters; required when ``start_params`` is
            omitted.
        start_params: Optional starting parameter vector.
        max_iter: Maximum trust-region iterations.
        xtol: Parameter-step convergence tolerance.
        ftol: Moment/objective_function convergence tolerance.
        gtol: Gradient convergence tolerance.
        Delta: Initial trust-region radius. If ``None``, a dimension-based
            default is used.
        debug: Whether to print per-iteration progress.
        weights: Optional observation weights, used by bootstrap refits.

    Returns:
        Dictionary containing final parameters, objective_function value, convergence
        flags, moments, gradient, tolerances, and iteration metadata.
    """
    if num_params is None and start_params is None:
        raise Exception

    if start_params is None:
        x = np.zeros(num_params).astype(float)
    else:
        x = np.asarray(start_params).astype(float).flatten()
        num_params = x.shape[0]

    _time = time.time()

    m = avg_moment_func(x, weights)

    if Delta is None:
        Delta = np.sqrt(num_params)

    f = wtd_moment_quadratic_obj(m, W)
    G = jac_avg_moment_func(x, weights)
    g = G.transpose().dot(W).dot(m)
    B = G.transpose().dot(W).dot(G)

    converged = False
    message = f'Did not converge in {max_iter} iterations'
    for n_iter in range(max_iter):

        try:
            # Solve the trust-region subproblem for the local quadratic
            # approximation of the weighted moment objective_function.
            x_step = steihaug(g, B, Delta)[0]

            m_new = avg_moment_func(x + x_step, weights)
            dM = np.max(np.where(np.abs(m) > 1, np.abs(m_new / m - 1), np.abs(m_new - m)))

            f_new = wtd_moment_quadratic_obj(m_new, W)
            predicted_decrease = quadratic_approx(g, B, x_step)
            actual_decrease = f_new - f
            rho = (actual_decrease / predicted_decrease
                   if abs(predicted_decrease) > 0 else np.nan)

            # Accept only objective_function-improving steps with enough agreement
            # between actual and predicted objective_function change.
            if f_new < f and rho > .1:
                f = f_new
                x += x_step
                G = jac_avg_moment_func(x, weights)
                m = m_new
                g = G.transpose().dot(W).dot(m)
                B = G.transpose().dot(W).dot(G)

            if rho > .75 and np.linalg.norm(x_step) >= Delta:
                Delta *= 2
            elif rho < .25:
                Delta /= 3

            # Track several convergence diagnostics because GMM specifications
            # may stabilize in parameters, moments, or gradient first.
            norm_dx = max(np.where(np.abs(x) > 1, np.abs(x_step),
                                   np.abs(x_step) / (np.abs(x) + 1e-8)))
            dF = actual_decrease / max(abs(f), 1.)
            max_grad = np.abs(g).max()

            if debug:
                iter_info = [
                    ({'name': 'iter', 'value': n_iter, 'format': '%6d', 'len': 6}),
                    ({'name': 'fval', 'value': f, 'format': '%16.4e', 'len': 16}),
                    ({'name': '|dx|', 'value': norm_dx, 'format': '%12.2e', 'len': 12}),
                    ({'name': '|dF|', 'value': dF, 'format': '%12.2e', 'len': 12}),
                    ({'name': '|dM|', 'value': dM, 'format': '%12.2e', 'len': 12}),
                    ({'name': '|grad|', 'value': max_grad, 'format': '%12.2e', 'len': 12}),
                    ({'name': 'rho', 'value': rho, 'format': '%10.4f', 'len': 10}),
                    ({'name': 'Delta', 'value': Delta, 'format': '%12.2e', 'len': 12}),
                    ({'name': 'Time', 'value': time.time() - _time, 'format': '%9.2fs', 'len': 10}),
                ]
                if n_iter == 0:
                    print_iter_info(iter_info, is_header=True)
                print_iter_info(iter_info)

            if -ftol < predicted_decrease <= 0.0:
                message = 'Converged: predicted decrease in objective_function too close to zero'
                converged = True
            if rho > .1 and predicted_decrease < 0:
                if norm_dx < xtol:
                    message = f'Converged: step size norm {"%.1e" % norm_dx} ' \
                              f'below x_tol={"%.1e" % xtol}'
                    converged = True
                if dM < ftol:
                    message = f'Converged: max relative change in moments {"%.1e" % dM}' \
                              f' size below f_tol={"%.1e" % ftol}'
                    converged = True
                if max_grad < gtol:
                    message = f'Converged: max gradient {"%.1e" % max_grad}' \
                              f' below g_tol={"%.1e" % gtol}'
                    converged = True

            if debug:
                if converged:
                    print_iter_info(iter_info, is_footer=True)

            if converged:
                break

        except KeyboardInterrupt:
            message = 'Did not converge: keyboard interrupt'
            print("\nProcess interrupted, breaking...\n")
            break

        except Exception as e:
            raise e

    if debug:
        print(message)

    return {
        'params': x,
        'fval': f,
        'converged': converged,
        'norm_dx': norm_dx,
        'dF': dF,
        'n_iters': n_iter + 1,
        'moments': m,
        'gradient': g,
        'moment_func': avg_moment_func,
        'message': message,
        'max_iter': max_iter,
        'g_tol': gtol,
        'f_tol': ftol,
        'x_tol': xtol,
    }
