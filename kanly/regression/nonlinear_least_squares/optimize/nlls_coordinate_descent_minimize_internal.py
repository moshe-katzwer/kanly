from __future__ import absolute_import, print_function

import time

import numpy as np
import pandas as pd

from kanly.regression.nonlinear_least_squares.constants import (
    DEFAULT_NLLS_EN_GTOL, DEFAULT_NLLS_EN_FTOL, DEFAULT_NLLS_EN_XTOL, DEFAULT_NLLS_EN_NUM_SHRINKAGE,
    DEFAULT_NLLS_EN_SHRINK_FACTOR, DEFAULT_NLLS_EN_MAX_ITER, DEFAULT_NLLS_EN_ACTIVE_SET, DEFAULT_NLLS_EN_ALPHA,
    DEFAULT_NLLS_EN_L1_RATIO, DEFAULT_NLLS_EN_NORMALIZE,
    DEFAULT_NLLS_PROMPT_USER_FOR_MORE_ITERS, DEFAULT_NLLS_KEEP_OPTIMIZATION_PATH,
    DEFAULT_NLLS_EN_JAC_METHOD, NLLS_EN_JAC_METHODS, NLLS_EN_JAC_METHOD_ANALYTIC,
    DEFAULT_NLLS_EN_SCALE_PENALTIES, DEFAULT_NLLS_EN_ONE_DIM_SEARCH_MULTIPLIER, DEFAULT_NLLS_EN_ONE_DIM_SEARCH_CADENCE,
    DEFAULT_NLLS_EN_ONE_DIM_SEARCH_INIT_VAL,
    DEFAULT_NLLS_EN_SELECTION, EN_SELECTION_TYPES
)
from kanly.utils.util import print_iter_info
from kanly.utils.user_prompt_for_more_iters import user_prompt_for_more_iters_method


def cost_func(r, weights=None):
    """Compute mean squared residual cost divided by two.

    Args:
        r: Residual vector.
        weights: Optional observation weights.

    Returns:
        Scalar cost ``mean(r**2)/2`` or weighted analogue.
    """
    if weights is None:
        return (r ** 2).mean() / 2
    else:
        return (weights * r ** 2).mean() / 2


def penalty_func(p, l1_penalty, l2_penalty):
    """Compute elastic-net penalty for a parameter vector.

    Args:
        p: Parameter vector on the regularised scale.
        l1_penalty: Per-parameter L1 penalty weights.
        l2_penalty: Per-parameter L2 penalty weights.

    Returns:
        Scalar penalty ``sum(l1*abs(p)) + sum(l2*p**2)``.
    """
    return sum(l1_penalty * np.abs(p)) + sum(l2_penalty * p ** 2)


def objective_func(r, p, l1_penalty, l2_penalty, weights):
    """Compute penalised coordinate-descent objective_function components.

    Args:
        r: Residual vector.
        p: Parameter vector on the regularised scale.
        l1_penalty: Per-parameter L1 penalty weights.
        l2_penalty: Per-parameter L2 penalty weights.
        weights: Optional observation weights.

    Returns:
        Tuple ``(objective_function, cost, penalty)``.
    """
    cst, pen = cost_func(r, weights), penalty_func(p, l1_penalty, l2_penalty)
    return cst + pen, cst, pen


def jacobian_1d(func, idx, x, regularize_to_vals, f0=None, dx=1e-6, method='fwd', jac_funcs=None):
    """Compute one residual-Jacobian column for coordinate descent.

    Uses an analytic partial derivative when ``jac_funcs`` is supplied,
    otherwise finite-differences only coordinate ``idx``.

    Args:
        func: Residual function evaluated at actual parameters.
        idx: Coordinate index to differentiate.
        x: Current centred parameter vector.
        regularize_to_vals: Offset added back to get actual parameters.
        f0: Optional baseline residual vector.
        dx: Relative finite-difference step.
        method: Finite-difference method, ``'fwd'`` or ``'mid'``.
        jac_funcs: Optional list of analytic partial derivative callables.

    Returns:
        Residual derivative vector for coordinate ``idx``.
    """
    if jac_funcs is not None:
        return jac_funcs[idx](x + regularize_to_vals)
    dx = dx * max(1, abs(x[idx]))
    if method == 'fwd':
        if f0 is None:
            f0 = func(x + regularize_to_vals)
        x_copy = np.asarray(x).copy()
        x_copy[idx] += dx
        return (func(x_copy + regularize_to_vals) - f0) / dx
    elif method == 'mid':
        x_copy_l = np.asarray(x).copy()
        x_copy_l[idx] -= dx
        x_copy_r = np.asarray(x).copy()
        x_copy_r[idx] += dx
        return (func(x_copy_r + regularize_to_vals) - func(x_copy_l + regularize_to_vals)) / (2 * dx)
    else:
        raise Exception('`method` must be "fwd" or "mid"!')


def solve_one_coord_problem(resid_func, r0, x0, idx, l1_penalty, l2_penalty, weights, regularize_to_values,
                            finite_difference_method='fwd', jac_funcs=None):
    """
    :param resid_func:
    :param r0: current residual vector
    :param x0: current param vector
    :param idx: index of param we are updating
    :param l1_penalty:
    :param l2_penalty:
    :param weights:
    :param regularize_to_val: penalize (param - regularize_to_val) instead of param
    :return:

    Args:
        resid_func: Callable residual function evaluated at actual parameters.
        r0: Current residual vector.
        x0: Current centred parameter vector.
        idx: Coordinate index to update.
        l1_penalty: L1 penalty for this coordinate.
        l2_penalty: L2 penalty for this coordinate.
        weights: Optional observation weights.
        regularize_to_values: Offset added to centred parameters before
            residual evaluation.
        finite_difference_method: ``'fwd'`` or ``'mid'`` for finite differences.
        jac_funcs: Optional analytic partial derivative callables.

    Returns:
        Tuple ``(x0_j, grad_j, std_jac_j)`` with the coordinate update,
        subgradient diagnostic, and Jacobian-column scale.
    """

    jac_j = jacobian_1d(resid_func, idx, x0, regularize_to_values, f0=r0, method=finite_difference_method,
                        jac_funcs=jac_funcs)

    if np.all(jac_j == 0):
        return x0[idx], 0, 0

    resid_x_jac = r0 * jac_j

    denom = jac_j ** 2
    numer = resid_x_jac - x0[idx] * denom

    if weights is not None:
        numer *= weights
        denom *= weights

    numer = numer.mean()
    denom = denom.mean() + 2.0 * l2_penalty

    # Closed-form 1-D elastic-net/proximal update: soft-threshold the linear
    # term and divide by the local quadratic curvature.
    x0_j = -(np.sign(numer) * max(abs(numer) - l1_penalty, 0.0)) / denom

    grad_j = np.average(resid_x_jac, weights=weights) + 2 * l2_penalty * x0[idx]

    if abs(x0[idx]) > 1e-8:
        grad_j += l1_penalty * np.sign(x0[idx])
    else:
        grad_j = (grad_j - l1_penalty, grad_j + l1_penalty)

    mean_jac_j = np.average(jac_j, weights=weights)
    std_jac_j = np.sqrt(np.average((jac_j - mean_jac_j) ** 2, weights=weights))

    return x0_j, grad_j, std_jac_j


def truncate_step(x0_j, x0, j, bounds, positive, regularize_to_val):
    """Clip a coordinate update to positivity and box constraints.

    Args:
        x0_j: Proposed new centred value for coordinate ``j``.
        x0: Current centred parameter vector.
        j: Coordinate index.
        bounds: Optional actual-parameter lower/upper bounds.
        positive: Boolean mask for non-negative actual parameters.
        regularize_to_val: Offset added to the centred coordinate.

    Returns:
        Feasible coordinate step ``new_value - x0[j]``.
    """
    if positive[j]:
        x0_j = max(x0_j, -regularize_to_val)
    if bounds is not None:
        x0_j = max(min(x0_j, bounds[j, 1] - regularize_to_val), bounds[j, 0] - regularize_to_val)
    return x0_j - x0[j]


def check_starting_point_input(x0, num_params):
    """Validate or construct the coordinate-descent starting vector.

    Args:
        x0: Optional starting parameter vector.
        num_params: Optional parameter count.

    Returns:
        Tuple ``(x0, num_params)``.
    """
    if x0 is None and num_params is None:
        raise Exception

    if x0 is None:
        x0 = np.zeros(num_params)
    else:
        x0 = np.asarray(x0).flatten()

    if num_params is None:
        num_params = len(x0)
    else:
        if len(x0) != num_params:
            raise Exception

    return x0, num_params


def check_regularization_penalty_inputs(num_params, alpha, l1_ratio):
    """Convert elastic-net ``alpha``/``l1_ratio`` inputs to L1 and L2 penalties.

    Args:
        num_params: Number of parameters.
        alpha: Scalar or vector total regularisation strength.
        l1_ratio: Fraction of ``alpha`` allocated to L1 penalty.

    Returns:
        Tuple ``(alpha, l1_ratio, l1_penalty, l2_penalty)``.
    """
    if isinstance(alpha, (float, int)):
        alpha = np.ones(num_params) * alpha
    else:
        alpha = np.asarray(alpha)
        assert len(alpha) == num_params

    assert np.all(alpha >= 0.)
    assert 0 <= l1_ratio <= 1

    l1_penalty = l1_ratio * alpha
    l2_penalty = (1.0 - l1_ratio) * alpha / 2

    return alpha, l1_ratio, l1_penalty, l2_penalty


def check_positivity_input(num_params, positive):
    """Normalise positivity constraints to a boolean mask.

    Args:
        num_params: Number of parameters.
        positive: Boolean or per-parameter boolean array.

    Returns:
        Boolean NumPy array of length ``num_params``.
    """
    if isinstance(positive, bool):
        positive = np.array([positive] * num_params)
    else:
        positive = np.asarray(positive).astype(bool)
    return positive


def check_regularize_to_values_input(num_params, regularize_to_vals):
    """Validate parameter target values used for centred regularisation.

    Args:
        num_params: Number of parameters.
        regularize_to_vals: ``None``, scalar, or vector of target values.

    Returns:
        Float vector of length ``num_params``.
    """
    if regularize_to_vals is None:
        regularize_to_vals = np.zeros(num_params)
    else:
        if isinstance(regularize_to_vals, (float, int)):
            regularize_to_vals = np.full(num_params, float(regularize_to_vals))
        else:
            regularize_to_vals = np.array(regularize_to_vals, copy=True, dtype=float).flatten()
            assert len(regularize_to_vals) == num_params
            assert np.all(np.isfinite(regularize_to_vals))
    return regularize_to_vals


def check_bound_and_adjust_start_point(x0, bounds, num_params, regularize_to_vals):
    """Validate bounds and clip the centred starting point to feasibility.

    Args:
        x0: Centred starting parameter vector.
        bounds: Optional actual-parameter lower/upper bounds.
        num_params: Number of parameters.
        regularize_to_vals: Offsets added to centred parameters.

    Returns:
        Tuple ``(is_bounded, x0, bounds)``.
    """

    if bounds is not None:
        bounds = np.asarray(bounds)
        assert bounds.shape == (num_params, 2)
        if np.all(bounds[:, 0] == -np.inf) and np.all(bounds[:, 1] == np.inf):
            bounds = None
            is_bounded = False
        else:
            is_bounded = True
            x0 = np.clip(x0, bounds[:, 0] - regularize_to_vals, bounds[:, 1] - regularize_to_vals)
    else:
        is_bounded = False

    return is_bounded, x0, bounds


def check_inputs(x0, num_params, alpha, l1_ratio, positive, bounds, regularize_to_vals):
    """Run all coordinate-descent input validation and normalisation helpers.

    Args:
        x0: Optional starting parameter vector.
        num_params: Optional parameter count.
        alpha: Elastic-net penalty strength.
        l1_ratio: L1 share of the elastic-net penalty.
        positive: Positivity constraint input.
        bounds: Optional lower/upper bounds.
        regularize_to_vals: Optional penalty target values.

    Returns:
        Tuple of normalised starting values, penalties, constraints, bounds, and
        regularisation offsets.
    """

    x0, num_params = check_starting_point_input(x0, num_params)
    alpha, l1_ratio, l1_penalty, l2_penalty = check_regularization_penalty_inputs(num_params, alpha, l1_ratio)
    regularize_to_vals = check_regularize_to_values_input(num_params, regularize_to_vals)
    is_bounded, x0, bounds = check_bound_and_adjust_start_point(x0, bounds, num_params, regularize_to_vals)
    positive = check_positivity_input(num_params, positive)

    return x0, num_params, l1_penalty, l2_penalty, positive, bounds, is_bounded, regularize_to_vals


def check_convergence(xerr, xtol, gerr, gtol, ferr, ftol, full_update, active_set, do_1d_search):
    """Check coordinate-descent convergence criteria.

    Args:
        xerr: Maximum relative coordinate change.
        xtol: Coordinate-change tolerance.
        gerr: Maximum absolute subgradient.
        gtol: Gradient tolerance.
        ferr: Relative objective_function change.
        ftol: Objective-change tolerance.
        full_update: Whether the current pass visited the full coordinate set.
        active_set: Whether active-set cycling is enabled.
        do_1d_search: Whether a one-dimensional search step was attempted.

    Returns:
        Tuple ``(converged, message, full_update, status)``.
    """
    status = 0
    met_convergence_criterion = False
    message = 'did not converge'

    if not do_1d_search:
        if xerr < xtol:
            met_convergence_criterion = True
            message = f'converged, change in params < xtol: ({xerr} < {xtol})'
        elif gerr < gtol:
            met_convergence_criterion = True
            message = f'converged, maximum abs (sub)gradient < gtol: ({gerr} < {gtol})'
        elif ferr < 0 and -ferr < ftol:
            met_convergence_criterion = True
            message = f'converged, relative change in objective_function < ftol: ({ferr} < {ftol})'

    converged = met_convergence_criterion and full_update
    if not converged:
        message = 'did not converge'
    else:
        status = 1

    full_update = (not active_set) or met_convergence_criterion

    return converged, message, full_update, status


def nlls_elastic_net_minimize_internal_coordinate_descent(
        resid_func, x0=None, num_params=None, weights=None, bounds=None, positive=False, debug=False,
        alpha=DEFAULT_NLLS_EN_ALPHA, l1_ratio=DEFAULT_NLLS_EN_L1_RATIO, max_iter=DEFAULT_NLLS_EN_MAX_ITER,
        xtol=DEFAULT_NLLS_EN_XTOL, ftol=DEFAULT_NLLS_EN_FTOL, gtol=DEFAULT_NLLS_EN_GTOL,
        num_shrinkage=DEFAULT_NLLS_EN_NUM_SHRINKAGE, shrink_factor=DEFAULT_NLLS_EN_SHRINK_FACTOR,
        active_set=DEFAULT_NLLS_EN_ACTIVE_SET, normalize=DEFAULT_NLLS_EN_NORMALIZE,
        prompt_user_for_more_iters=DEFAULT_NLLS_PROMPT_USER_FOR_MORE_ITERS,
        keep_optimization_path=DEFAULT_NLLS_KEEP_OPTIMIZATION_PATH,
        selection=DEFAULT_NLLS_EN_SELECTION,
        regularize_to_values=None, jac_method=DEFAULT_NLLS_EN_JAC_METHOD,
        scale_penalties=DEFAULT_NLLS_EN_SCALE_PENALTIES,
        one_dim_search_cadence=DEFAULT_NLLS_EN_ONE_DIM_SEARCH_CADENCE,
        one_dim_search_multiplier=DEFAULT_NLLS_EN_ONE_DIM_SEARCH_MULTIPLIER,
        one_dim_search_init_value=DEFAULT_NLLS_EN_ONE_DIM_SEARCH_INIT_VAL,
        seed=0,
):
    """Fit an elastic-net penalised NLLS objective_function by coordinate descent.

    The algorithm works on centred parameters ``x`` where actual parameters are
    ``x + regularize_to_values``.  Each coordinate is updated using a local
    one-dimensional quadratic approximation with soft-thresholding, optional
    bounds/positivity constraints, and shrinkage when the objective_function does not
    improve.

    Args:
        resid_func: Callable returning residuals for actual parameters.
        x0: Optional starting parameter vector.
        num_params: Number of parameters when ``x0`` is absent.
        weights: Optional observation weights.
        bounds: Optional actual-parameter lower/upper bounds.
        positive: Boolean or mask enforcing non-negative actual parameters.
        debug: Whether to print iteration progress.
        alpha: Scalar or vector total elastic-net penalty strength.
        l1_ratio: Share of ``alpha`` assigned to L1 penalty.
        max_iter: Maximum coordinate-descent passes.
        xtol: Coordinate-change tolerance.
        ftol: Objective-change tolerance.
        gtol: Subgradient tolerance.
        num_shrinkage: Maximum backtracking shrinkage attempts per coordinate.
        shrink_factor: Step multiplier applied after failed coordinate updates.
        active_set: Whether to restrict passes to recently improving coordinates.
        normalize: Whether to scale penalties by coordinate Jacobian norms.
        prompt_user_for_more_iters: Whether/how to prompt after ``max_iter``.
        keep_optimization_path: Whether to store objective_function/parameter history.
        selection: Coordinate order: ``'cyclic'``, ``'greedy'``, or ``'random'``.
        regularize_to_values: Target values for centred regularisation.
        jac_method: ``'analytic'``, ``'fwd'``, or ``'mid'`` derivative method.
        scale_penalties: Whether penalties scale naturally with mean cost.
        one_dim_search_cadence: Optional cadence for line search along the full
            update direction (currently guarded as experimental).
        one_dim_search_multiplier: Multiplicative line-search expansion factor.
        one_dim_search_init_value: Initial line-search scalar.
        seed: Random seed used for random coordinate selection.

    Returns:
        Dict containing final parameters, diagnostics, penalties, residuals,
        active constraints, and solver settings.
    """
    if one_dim_search_cadence is not None:
        raise Exception('`one_dim_search_cadence` not fully working yet!')

    if debug:
        options_series = np.array([
            ['setting', 'value'],
            ['max_iter', max_iter],
            ['ftol', ftol],
            ['xtol', xtol],
            ['gtol', gtol],
            ['jac_method', jac_method],
            ['active_set', active_set],
            ['seed', seed],
            ['selection', selection],
            ['one_dim_search_cadence', one_dim_search_cadence],
            ['one_dim_search_multiplier', one_dim_search_multiplier],
            ['one_dim_search_init_value', one_dim_search_init_value],
            ['num_shrinkage', num_shrinkage],
            ['shrink_factor', shrink_factor],
        ])
        options_series = pd.Series(options_series[:, 1], index=[c + ": " for c in options_series[:, 0]])
        option_strs = options_series.to_string().split('\n')
        len_option_strs = len(option_strs[0])
        print('=' * len_option_strs + "\n" + option_strs[0] + "\n" + '-' * len_option_strs + '\n' +
              '\n'.join( option_strs[1:]) + '\n' + '-' * len_option_strs)

    assert jac_method in NLLS_EN_JAC_METHODS

    if jac_method == NLLS_EN_JAC_METHOD_ANALYTIC:
        if debug:
            tjac = time.time()
            print("\nComputing jacobian analytical function...", end='')
        jac_funcs = resid_func.get_analytical_partial_derivatives(return_info=False)
        if debug:
            print('%.3fs' % (time.time() - tjac))
    else:
        jac_funcs = None

    selection = str(selection).lower()
    assert selection in EN_SELECTION_TYPES

    rand = np.random.RandomState(seed)

    x0, num_params, l1_penalty0, l2_penalty0, positive, bounds, is_bounded, regularize_to_values \
        = check_inputs(x0, num_params, alpha, l1_ratio, positive, bounds, regularize_to_values)

    x0_start = x0.copy()

    _t = time.time()

    r0 = resid_func(x0 + regularize_to_values)
    nobs = len(r0)

    if not scale_penalties:
        # cost function is mean residual squared, so penalties naturally
        # scale with nobs.  To undo this and have an 'absolute' penalty,
        # we divide by nobs
        if weights is None:
            l1_penalty0 /= nobs
            l2_penalty0 /= nobs
        else:
            l1_penalty0 /= np.sum(weights)
            l2_penalty0 /= np.sum(weights)

    if weights is not None:
        # Normalise weights so the mean-squared objective_function stays on the same
        # scale as the unweighted objective_function.
        weights = weights * len(weights) / sum(weights)

    objective, cost, penalty = objective_func(r0, x0, l1_penalty0, l2_penalty0, weights)
    index_rng = np.array(range(num_params))
    full_update = True

    converged = False
    idx_last = -1

    subgrad = [None] * num_params

    std_jac = np.ones(num_params)

    if keep_optimization_path:
        optimization_path = []
    else:
        optimization_path = None

    n_iter = 0
    while True:

        if keep_optimization_path:
            optimization_path.append({'iter': n_iter, 'x': np.array(x0, copy=True), 'objective_function': objective,
                                      'penalty': penalty, 'cost': cost, 'regularize_to_vals': regularize_to_values})

        try:
            n_iter += 1

            count_failed = 0.0

            if normalize:
                l1_penalty = l1_penalty0 / std_jac
                l2_penalty = l1_penalty0 / std_jac ** 2
            else:
                l1_penalty, l2_penalty = l1_penalty0, l2_penalty0

            diff_record = np.zeros(num_params)
            diff_obj = np.zeros(num_params)

            objective_old = objective
            x0_old = x0.copy()

            cnt = 0

            if len(index_rng) == 0:
                break

            last_idx = index_rng[-1]
            for idx in index_rng:

                if idx == idx_last:
                    continue

                idx_last = idx

                x0_j_new, g_j, std_jac_j \
                    = solve_one_coord_problem(resid_func, r0, x0, idx, l1_penalty[idx], l2_penalty[idx], weights,
                                              regularize_to_values, jac_method, jac_funcs)

                if normalize:
                    std_jac[idx] = std_jac_j if std_jac_j >= 1e-12 else 1

                subgrad[idx] = g_j
                x0_j_step = truncate_step(x0_j_new, x0, idx, bounds, positive, regularize_to_values[idx])

                if x0_j_step == 0.0:
                    continue

                Delta = 1.0
                count_failed += 1

                for k in range(num_shrinkage):

                    cnt += 1

                    x0_copy = x0.copy()
                    x0_copy[idx] += Delta * x0_j_step

                    r_new = resid_func(x0_copy + regularize_to_values)
                    objective_new, cost_new, penalty_new = objective_func(
                        r_new, x0_copy, l1_penalty, l2_penalty, weights)

                    if objective_new > objective:
                        Delta *= shrink_factor

                    else:
                        count_failed -= 1

                        r0 = r_new
                        diff_record[idx] = abs(x0[idx] - x0_copy[idx])
                        x0 = x0_copy

                        diff_obj[idx] = objective - objective_new
                        objective, cost, penalty = objective_new, cost_new, penalty_new
                        break


            # try going in 1d direction implied between iterates
            do_1d_search = (not (
                    one_dim_search_init_value is None or one_dim_search_cadence is None or one_dim_search_multiplier is None)
                            and (n_iter - 1) % one_dim_search_cadence == 1)

            s_best = np.nan
            # if do_1d_search:
            #     rng = [-(.005 * 2 ** kk) for kk in range(1, 3)] + [(.005 * 2 ** kk) for kk in range(1, 3)]
            #     dir = x0 - x0_old
            #
            #     def _obj(_x):
            #         p = _x + regularize_to_vals
            #         r = resid_func(p)
            #         return objective_func(r, p, l1_penalty, l2_penalty, weights)
            #
            #     v = np.array([_obj(x0 + r * dir) for r in rng])
            #     best = np.argmin(v[:, 0])
            #     s_best = rng[best]
            #     print(s_best, objective_function, v[best][0], "%.3f" % (objective_function / v[best][0] - 1))
            #     if v[best, 0] < objective_function:
            #         objective_function, cost, penalty = v[best]
            #         x0 = x0 + dir * s_best
            #
            # diff_record = x0 - x0_old


            # do_1d_search = (not (
            #         one_dim_search_init_value is None or one_dim_search_cadence is None or one_dim_search_multiplier is None)
            #                 and (n_iter-1) % one_dim_search_cadence == 1)
            # s_best = np.nan
            # if do_1d_search:
            #     x0, objective_function, cost, penalty, s_best = _do_one_dimensional_search(
            #         objective_func, resid_func, x0, x0_old, l1_penalty, l2_penalty, regularize_to_vals, weights,
            #         one_dim_search_init_value, one_dim_search_multiplier, objective_function, cost, penalty)
            #     diff_record = x0 - x0_old

            with np.errstate(divide='ignore', invalid='ignore'):
                xerr = max(np.where(np.abs(x0) < 1, np.abs(diff_record), np.abs(diff_record / x0)))

            ferr = objective / objective_old - 1

            grad = np.array([
                (0.0 if x[0] <= 0 <= x[1] else x[np.argmax(np.abs(x))])
                if isinstance(x, tuple) else x
                for x in subgrad
            ])
            gerr = max(np.abs(grad))

            if debug:
                iter_info_dict = [
                    {'name': 'iter', 'len': 6, 'format': '%6d', 'value': n_iter},
                    {'name': 'cost', 'len': 10, 'format': '%10.2e', 'value': cost * nobs},
                    {'name': 'penalty', 'len': 10, 'format': '%10.2e', 'value': penalty * nobs},
                    {'name': 'F', 'len': 12, 'format': '%12.4e', 'value': objective * nobs},
                    {'name': 'dF/F', 'len': 10, 'format': '%10.1e', 'value': ferr},
                    {'name': 'max(dx/x)', 'len': 10, 'format': '%10.1e', 'value': xerr},
                    {'name': 'max(grad)', 'len': 10, 'format': '%10.1e', 'value': gerr},
                    {'name': 'full_update', 'len': 12, 'format': '%12s', 'value': full_update},
                    {'name': '|active x|', 'len': 12, 'format': '%12d', 'value': np.count_nonzero(diff_record > 0)},
                    {'name': '#failed', 'len': 10, 'format': '%10d', 'value': count_failed},
                    {'name': 's_best', 'len': 10, 'format': '%10.1e', 'value': s_best},
                    {'name': 'time', 'len': 11, 'format': '%10.2fs', 'value': time.time() - _t},
                ]
                if n_iter == 1:
                    print_iter_info(iter_info_dict, is_header=True)
                print_iter_info(iter_info_dict)

            converged, message, full_update, status \
                = check_convergence(xerr, xtol, gerr, gtol, ferr, ftol, full_update, active_set, do_1d_search)

            if converged:
                break

            if full_update:
                index_rng = np.arange(num_params)
            else:
                index_rng = np.arange(num_params)[diff_obj > 0]
                if len(index_rng) == num_params:
                    full_update = True

            if selection == 'greedy':
                grad_sub = np.abs(grad[index_rng])
                index_rng = index_rng[np.argsort(grad_sub)][::-1]
            elif selection == 'random':
                index_rng = rand.permutation(index_rng)
                if index_rng[0] == last_idx:
                    index_rng = np.hstack([index_rng[1:], index_rng[0]])

            if n_iter == max_iter:
                incremental_iters = user_prompt_for_more_iters_method(
                    f"\n\t[iteration = {'%d' % n_iter}, |dx| = {'%.2e' % xerr}, "
                    f"optimality = {'%.2e' %  np.linalg.norm(grad, ord=np.inf)}, "
                    f"dF/F = {'%.2e' % ferr}]",
                    prompt_user_for_more_iters
                )
                if incremental_iters == 0:
                    break
                else:
                    max_iter += incremental_iters

        except KeyboardInterrupt:
            message = 'Did not converge: keyboard interrupt'
            print("\nProcess interrupted, breaking...\n")
            break

        except Exception as e:
            raise e

    optimality = np.linalg.norm(grad, ord=np.inf)

    active_mask = np.zeros(num_params)
    active_mask[positive] = -(np.abs(x0[positive]) < xtol).astype(int)
    if is_bounded:
        active_mask[np.abs(bounds[:, 1] - x0) < xtol] = 1
        active_mask[np.abs(bounds[:, 0] - x0) < xtol] = -1

    if debug:
        print_iter_info(iter_info_dict, is_footer=True)
        print(f'Message: {message}')

    # import matplotlib.pyplot as plt
    # for i, x in enumerate(x0[:10]):
    #     def obj_i(x_i):
    #         x0_copy = x0.copy()
    #         x0_copy[i] = x_i
    #         r_new = resid_func(x0_copy + regularize_to_vals)
    #         return nobs * objective_func(
    #             r_new, x0_copy, l1_penalty, l2_penalty, weights)[0]
    #
    #     XX = np.linspace(x - .01, x + .01, 200)
    #     plt.figure(dpi=100)
    #     plt.title((i, x, subgrad[i]))
    #     plt.plot(XX, [obj_i(xx) for xx in XX])
    #     plt.scatter([x], [nobs * objective_function])
    #     plt.show()
    # raise Exception

    return {
        'params': x0 + regularize_to_values,
        'message': message,
        'converged': converged,
        'status': status,
        'cost': cost,
        'penalty': penalty,
        'objective_function': objective,
        'fit_elapsed': time.time() - _t,
        'iterations': n_iter,
        'xerr': xerr,
        'ferr': ferr,
        'gerr': gerr,
        'alpha': alpha,
        'l1_ratio': l1_ratio,
        'l1_penalty': l1_penalty,
        'l2_penalty': l2_penalty,
        'regularize_to_vals': np.array(regularize_to_values),
        'l1_penalty_start': l1_penalty0,
        'l2_penalty_start': l2_penalty0,
        'bounds': bounds,
        'is_bounded': is_bounded,
        'positive': positive,
        'active_set': active_set,
        'active_mask': active_mask,
        'subgrad': subgrad,
        'grad': grad,
        'optimality': optimality,
        'jac': None,
        'norm_dx': np.linalg.norm(diff_record),
        'start_params': x0_start,
        'resid': r0,
        'wresid': r0 if weights is None else np.sqrt(weights) * r0,
        'normalize': normalize,
        'std_jac': std_jac,
        'optimization_path': optimization_path,
        'scale_penalties': scale_penalties,
        'solver_setting': {
            'max_iter': max_iter,
            'ftol': ftol,
            'xtol': xtol,
            'finite_difference_method': jac_method,
            'max_iter': max_iter,
            'one_dim_search_cadence': one_dim_search_cadence,
            'one_dim_search_multiplier': one_dim_search_multiplier,
            'one_dim_search_init_value': one_dim_search_init_value,
            'num_shrinkage': num_shrinkage,
            'shrink_factor': shrink_factor,
            'seed': seed,
            'selection': selection,
        }
    }


def _do_one_dimensional_search(
        objective_func, resid_func, x0, x0_old, l1_penalty, l2_penalty, regularize_to_vals, weights,
        one_dim_search_init_value, one_dim_search_multiplier, objective, cost, penalty):
    """Search along the full coordinate-descent update direction.

    Args:
        objective_func: Callable returning objective_function/cost/penalty components.
        resid_func: Residual function.
        x0: Current centred parameter vector.
        x0_old: Previous centred parameter vector.
        l1_penalty: Current L1 penalties.
        l2_penalty: Current L2 penalties.
        regularize_to_vals: Offsets added to centred parameters.
        weights: Optional observation weights.
        one_dim_search_init_value: Initial scalar step away from ``x0_old``.
        one_dim_search_multiplier: Expansion multiplier.
        objective: Current objective_function.
        cost: Current cost.
        penalty: Current penalty.

    Returns:
        Tuple ``(x0, obj_best, cost_best, penalty_best, s_best)``.
    """

    def _obj(_x):
        """Evaluate objective_function components at a centred parameter vector.

        Args:
            _x: Centred parameter vector.

        Returns:
            Tuple ``(objective_function, cost, penalty)``.
        """
        p = _x + regularize_to_vals
        r = resid_func(p)
        return objective_func(r, p, l1_penalty, l2_penalty, weights)

    direction = x0 - x0_old
    s_best = 0

    obj_best = objective
    penalty_best = penalty
    cost_best = cost

    s = one_dim_search_init_value
    while True:
        obj_new, cost_new, penalty_new = _obj(x0_old + (1 + s) * direction)
        if obj_new < obj_best:
            obj_best = obj_new
            cost_best = cost_new
            penalty_best = penalty_new
            s_best = s
            s *= one_dim_search_multiplier
        else:
            break

    if s_best == 0:
        s = -one_dim_search_init_value
        while True:
            obj_new, cost_new, penalty_new = _obj(x0_old + (1 + s) * direction)
            if obj_new < obj_best:
                obj_best = obj_new
                cost_best = cost_new
                penalty_best = penalty_new
                s_best = s
                s *= one_dim_search_multiplier
            else:
                break

    if s_best != 0:
        x0 = x0_old + (1.0 + s_best) * direction

    return x0, obj_best, cost_best, penalty_best, s_best