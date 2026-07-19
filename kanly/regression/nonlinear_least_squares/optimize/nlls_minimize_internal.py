from __future__ import absolute_import, print_function

import time
import traceback

import numpy as np
from scipy.sparse import csc_matrix, isspmatrix, csr_matrix
from scipy.sparse.linalg import norm as spnorm
from scipy.stats import norm as norm_dist

from kanly.regression.nonlinear_least_squares.constants import (
    DEFAULT_NLLS_XTOL, DEFAULT_NLLS_GTOL, DEFAULT_NLLS_FTOL, DEFAULT_NLLS_DELTA, DEFAULT_NLLS_DELTA_FLOOR,
    DEFAULT_NLLS_MAX_ITER, DEFAULT_NLLS_X_SCALE, DEFAULT_NLLS_RHO_QUAD_MODEL_ACCEPT, DEFAULT_NLLS_RHO_QUAD_MODEL_REJECT,
    DEFAULT_NLLS_DELTA_INCREASE_FACTOR, DEFAULT_NLLS_DELTA_DECREASE_FACTOR, DEFAULT_NLLS_RHO_STEP_ACCEPT_FLOOR,
    DEFAULT_NLLS_NUM_REFLECTIONS, DEFAULT_NLLS_DO_BROYDEN_JAC_UPDATE, DEFAULT_NLLS_BROYDEN_JAC_UPDATE_CADENCE,
    DEFAULT_NLLS_DO_LINE_SEARCH, DEFAULT_NLLS_KEEP_OPTIMIZATION_PATH, DEFAULT_NLLS_PROMPT_USER_FOR_MORE_ITERS,
    DEFAULT_NLLS_JAC_METHOD, DEFAULT_NLLS_TRY_NEWTON_STEP, DEFAULT_NLLS_REFLECTION_THETA,
    DEFAULT_NLLS_DO_ANALYTIC_JAC_JIT, DEFAULT_NLLS_SCALE_L2_PENALTIES)
from kanly.regression.nonlinear_least_squares.function_callables.jacobian import get_finite_diff_jacobian
from kanly.regression.nonlinear_least_squares.function_callables.loss_functions import get_root_loss_func
from kanly.regression.nonlinear_least_squares.function_callables.residual_function import ResidualFunction
from kanly.regression.nonlinear_least_squares.optimize.nlls_minimize_internal_result import NllsMinimizeInternalResult
from kanly.regression.nonlinear_least_squares.optimize.quadratic_approx_subproblem_functions import (
    steihaug, cauchy_point, get_max_step_to_bound, quadratic_form, newton)
from kanly.regression.nonlinear_least_squares.optimize.reflect import get_reflect_path
from kanly.utils.linalg_utils import DEFAULT_DENSE_THRESHOLD_MB
from kanly.utils.linalg_utils import csc_matrix_by_column_array_broadcast
from kanly.utils.linalg_utils import get_matrix_inverse_internal
from kanly.utils.user_prompt_for_more_iters import user_prompt_for_more_iters_method
from kanly.utils.util import print_iter_info

MAD_DENOM = norm_dist.ppf(.75)



def get_mad_scale(resid, center=0.0):
    """Estimate residual scale using the median absolute deviation.

    Args:
        resid: Residual vector.
        center: Center value subtracted before taking absolute deviations.

    Returns:
        Robust normal-consistent scale estimate ``median(abs(resid-center)) / Φ^-1(.75)``.
    """
    # med_resid = np.median(resid) # TODO?
    scale = np.median(np.abs(resid - center)) / MAD_DENOM
    return scale


def cost_func_sum_squares(resid, weights=None):
    """Compute the NLLS cost ``0.5 * sum(resid**2)``.

    Args:
        resid: Residual vector.
        weights: Optional observation weights.

    Returns:
        Scalar weighted or unweighted sum-of-squares cost.
    """
    if weights is None:
        val = np.linalg.norm(resid) ** 2 / 2
    else:
        val = (weights * resid ** 2).sum() / 2
    return val


def penalty_func(params, l2_penalties, regularize_to_values):
    """Compute the L2 regularisation penalty.

    Args:
        params: Current parameter vector.
        l2_penalties: Optional per-parameter L2 penalty weights.
        regularize_to_values: Parameter target values for the L2 penalty.

    Returns:
        Scalar penalty value, or ``0.0`` when no L2 penalties are supplied.
    """
    if l2_penalties is None:
        return 0.0
    else:
        return np.dot(l2_penalties, (params - regularize_to_values) ** 2) / 2


def objective_func(resid, params, l2_penalties, regularize_to_values, weights=None):
    """
    1/2 * sum_of_squares + 1/2 * l2_penalty * (params - regularize_param_to_values)**2

    Args:
        resid: Residual vector.
        params: Current parameter vector.
        l2_penalties: Optional per-parameter L2 penalties.
        regularize_to_values: Target parameter values for regularisation.
        weights: Optional observation weights.

    Returns:
        Scalar penalised objective_function value.
    """
    return cost_func_sum_squares(resid, weights) + penalty_func(params, l2_penalties, regularize_to_values)


def check_convergenge(optimality, actual_decrease, nobs, cost, x0_scaled, dx_scaled, gtol, ftol, xtol):
    """Check gradient, function, and parameter convergence criteria.

    Args:
        optimality: Infinity-norm optimality measure.
        actual_decrease: Change in objective_function from the last accepted step.
        nobs: Number of observations.
        cost: Previous objective_function/cost value.
        x0_scaled: Current scaled parameter vector.
        dx_scaled: Last scaled parameter step.
        gtol: Gradient tolerance.
        ftol: Relative function-change tolerance.
        xtol: Relative parameter-change tolerance.

    Returns:
        Tuple ``(converged, message, status)``.
    """
    converged = False
    status = 0
    message = None

    gtol_satis, ftol_satis, xtol_satis = False, False, False
    if optimality < gtol * nobs:
        converged = True
        message = "Converged: supnorm(gradient) < gtol * nobs"
        gtol_satis = True
    if abs(actual_decrease) < ftol * max(1.0, cost):
        converged = True
        message = "Converged: |dF| < ftol * max(1, |F|)"
        ftol_satis = True
    if np.linalg.norm(dx_scaled) < xtol * max(1, np.linalg.norm(x0_scaled)):
        converged = True
        message = "Converged: |dx| < xtol * max(1, |x|)"
        xtol_satis = True

    if gtol_satis:
        status = 1
    elif ftol_satis and xtol_satis:
        status = 4
    elif ftol_satis:
        status = 2
    elif xtol_satis:
        status = 3

    return converged, message, status


def compute_jacobian(jacobian_function, x0, d_root_loss=None):
    """Evaluate and optionally chain-rule scale a Jacobian.

    Args:
        jacobian_function: Callable mapping parameters to a residual Jacobian.
        x0: Parameter vector at which to evaluate.
        d_root_loss: Optional derivative of a root-loss transformation applied
            row-wise to residuals.

    Returns:
        Dense or sparse Jacobian, modified by ``d_root_loss`` when provided.
    """
    J = jacobian_function(x0)
    if d_root_loss is not None:
        if d_root_loss is not None:
            if isspmatrix(J):
                J.data *= d_root_loss[J.indices]
            else:
                J *= d_root_loss.reshape((-1, 1))
    return J


def compute_v(g, x0, bounds, x_scale=None):
    """
    v[j] = { x0[j]-lb[j] if g[j] > 0 and lb[j] is finite
           { ub[j]-x0[j] if g[j] < 0 and ub[j] is finite
           { 1           else

    Args:
        g: Gradient vector.
        x0: Current parameter vector.
        bounds: ``(num_params, 2)`` lower/upper bound array.
        x_scale: Optional parameter scaling vector.

    Returns:
        Tuple ``(v, Jv)`` where ``v`` is the Coleman-Li distance-to-bound scale
        and ``Jv`` is its diagonal derivative.
    """

    num_params = g.shape[0]
    if x_scale is None:
        x_scale = np.ones(num_params)

    v = np.ones(num_params)
    Jv = np.zeros(num_params)

    ub_idx = (g < 0) & (bounds[:, 1] < np.inf)
    v[ub_idx] = bounds[:, 1][ub_idx] - x0[ub_idx]
    Jv[ub_idx] = -1.0

    lb_idx = (g > 0) & (bounds[:, 0] > -np.inf)
    v[lb_idx] = x0[lb_idx] - bounds[:, 0][lb_idx]
    Jv[lb_idx] = 1.0

    v = v * x_scale
    # Jv = (x_scale ** .5).reshape((1, -1)).dot(Jv).dot((x_scale ** .5).reshape((-1, 1)))
    Jv = Jv * x_scale
    return v, Jv


def get_x0(x0, num_params, bounds, is_bounded):
    """Gets starting point if none supplied.

    Args:
        x0: Optional initial parameter vector.
        num_params: Number of parameters, required when ``x0`` is ``None``.
        bounds: Optional ``(num_params, 2)`` lower/upper bounds.
        is_bounded: Whether bounds should be enforced.

    Returns:
        Tuple ``(x0, num_params, is_bounded)`` with feasible starting values.
    """
    if bounds is not None:
        bounds = np.asarray(bounds).astype(float)

    if x0 is None:
        if num_params is None:
            raise Exception("Must specify one of `x0` or `num_params`!")
        x0 = np.zeros(num_params)
        if is_bounded:
            for i in range(num_params):
                if np.all(np.isfinite(bounds[i])):
                    if bounds[i, 0] <= 0.0 <= bounds[i, 1]:
                        x0[i] = 0.0
                    else:
                        x0[i] = np.mean(bounds[i])
                elif np.isfinite(bounds[i, 0]):
                    x0[i] = max(0., bounds[i, 0])
                    if x0[i] == 0:
                        x0[i] += .05
                    else:
                        x0[i] += .05 * abs(bounds[i, 0])
                elif np.isfinite(bounds[i, 1]):
                    x0[i] = min(bounds[i, 1], 0.)
                    if x0[i] == 0:
                        x0[i] -= .05
                    else:
                        x0[i] -= .05 * abs(bounds[i, 1])
    else:
        x0 = np.array(x0).astype(float).flatten()
        if is_bounded:
            if not np.all(x0 >= bounds[:, 0]) and np.all(x0 <= bounds[:, 1]):
                raise Exception("Starting params `x0` not feasible!")

    # step back from the bounds a bit
    if bounds is not None:
        if np.any(x0 - bounds[:, 0] < 1e-8):
            x0[x0 - bounds[:, 0] < 1e-8] += 1e-4
        if np.any(bounds[:, 1] - x0 < 1e-8):
            x0[bounds[:, 1] - x0 < 1e-8] -= 1e-4

    num_params = len(x0)
    if is_bounded:
        if tuple(bounds.shape) != (num_params, 2):
            raise Exception(f"Bounds have wrong shape, {bounds.shape} instead of {(num_params, 2)}")
        if np.all(bounds[:, 0] == -np.inf) and np.all(bounds[:, 1] == np.inf):
            is_bounded = False

    return x0, num_params, is_bounded


def hits_bounds(x, bounds, tol=1e-14):
    """Count how many parameters are at lower or upper bounds.

    Args:
        x: Parameter vector.
        bounds: ``(num_params, 2)`` lower/upper bounds.
        tol: Numerical tolerance for considering a parameter active.

    Returns:
        Integer count of active bound constraints.
    """
    return np.sum(x <= bounds[:, 0] + tol) + np.sum(x >= bounds[:, 1] - tol)


def get_next_step(is_bounded, x0, g_hat, B_hat, scale, Delta, xtol, bounds, num_reflections=1,
                  try_newton_step=DEFAULT_NLLS_TRY_NEWTON_STEP,
                  theta=DEFAULT_NLLS_REFLECTION_THETA):
    """Choose the next trust-region step for the quadratic NLLS approximation.

    Computes a Steihaug step, optionally compares it to a clipped Newton step,
    and if box constraints would be hit, also evaluates reflected and Cauchy
    candidate steps before choosing the one with the best predicted reduction.

    Args:
        is_bounded: Whether finite parameter bounds are active.
        x0: Current parameter vector.
        g_hat: Scaled gradient.
        B_hat: Scaled Hessian/quadratic approximation.
        scale: Parameter scaling vector that maps scaled steps to original units.
        Delta: Trust-region radius.
        xtol: Step tolerance passed to Steihaug.
        bounds: Optional lower/upper bound array.
        num_reflections: Maximum reflected-path segments to consider.
        try_newton_step: Whether to compare a Newton step candidate.
        theta: Fraction used to stay inside bounds when truncating steps.

    Returns:
        Tuple ``(p_scaled, p, step_case, norm_dx, predicted_decrease)``.
    """
    step_case = 'steihaug'
    p_steihaug_scaled, cond, stei_iter = steihaug(g_hat, B_hat, Delta, eps=xtol)

    if try_newton_step and (not is_bounded or (is_bounded and not hits_bounds(x0 + p_steihaug_scaled * scale, bounds))):
        to_try = {'steihaug': p_steihaug_scaled}

        try:
            to_try['newton'] = newton(B_hat, g_hat, Delta)
        except:
            pass

        # Cauchy shouldn't actually be any better than steihaug
        # p_cauchy_scaled = cauchy_point(B_hat, g_hat, Delta)
        # to_try['cauchy'] = p_cauchy_scaled

        fvals = {k: quadratic_form(v, g_hat, B_hat) for k, v in to_try.items()}
        # print()
        # print("ZZZ ", fvals)
        min_key = min(fvals, key=fvals.get)
        p_steihaug_scaled = to_try[min_key]
        step_case = min_key

    if is_bounded and hits_bounds(x0 + p_steihaug_scaled * scale, bounds):

        max_step = get_max_step_to_bound(x0, p_steihaug_scaled * scale, bounds[:, 0], bounds[:, 1])
        # if np.isinf(max_step):
        #     print(pd.DataFrame({
        #         'x0': x0,
        #         'p_steihaug_scaled': p_steihaug_scaled,
        #         'lb':  bounds[:, 0],
        #         'ub': bounds[:, 1],
        #     }))
        #     raise Exception

        max_step_stei = theta * max_step

        p_reflect_path, f_param = get_reflect_path(
            x0, p_steihaug_scaled * scale, bounds[:, 0], bounds[:, 1], scale, Delta, num_reflections=num_reflections)
        # Evaluate a small grid along the reflected path and compare it to the
        # truncated Steihaug/Cauchy alternatives under the quadratic model.
        l_space = np.linspace(1 - theta, theta, 20)
        q_eval = [quadratic_form(f_param(t), g_hat, B_hat) for t in l_space]
        p_reflect = f_param(l_space[np.argmin(q_eval)]) - x0
        max_step = get_max_step_to_bound(x0, p_reflect, bounds[:, 0], bounds[:, 1])
        if max_step == 0:
            p_reflect *= theta
        p_reflect_scaled = p_reflect / np.clip(scale, a_min=.000001, a_max=np.inf)

        #print("<<<< ", max_step, max_step_stei)
        p_steihaug_scaled *= max_step_stei

        p_cauchy_scaled = cauchy_point(B_hat, g_hat, Delta)
        max_step = get_max_step_to_bound(x0, p_cauchy_scaled * scale, bounds[:, 0], bounds[:, 1])
        if max_step > 1:
            max_step = 1.0
        else:
            max_step = theta * max_step
        p_cauchy_scaled *= max_step

        p_scaled = None
        best_reduction = np.inf
        step_case = None
        for k, _p in {'stei-bnd': p_steihaug_scaled, 'cauchy': p_cauchy_scaled, 'reflect': p_reflect_scaled,
                      }.items():
            reduction = quadratic_form(_p, g_hat, B_hat)
            # print("%-10s%15.6e%15.6e%15.6e%8d%10.2e" %
            #       (
            #           k, reduction,
            #           quad_approx(_p, g_hat, B_hat) / cost,
            #           np.linalg.norm(_p) / Delta,
            #           hits_bounds(x0 + root_v * _p, bounds),
            #           get_max_step_to_bound(x0, _p*root_v, bounds)
            #       )
            #       )
            if reduction < best_reduction:
                best_reduction = reduction
                step_case = k
                p_scaled = _p

    else:

        # Try Newton Step
        p_scaled = p_steihaug_scaled
        # try:
        #     p_newton_scaled = np.linalg.solve(B_hat, -g_hat)
        #     p_newton_scaled *= min(1, Delta / np.linalg.norm(p_newton_scaled))
        #
        #     if quadratic_form(x0 + p_newton_scaled * scale, g_hat, B_hat) \
        #             < quadratic_form(x0 + p_steihaug_scaled, g_hat, B_hat):
        #         step_case = 'newton'
        #         p_scaled = p_newton_scaled
        #
        # except:
        #     pass

    p = p_scaled * scale
    norm_dx = np.linalg.norm(p_scaled)
    predicted_decrease = quadratic_form(p_scaled, g_hat, B_hat)

    return p_scaled, p, step_case, norm_dx, predicted_decrease


def transform_quadratic_approx_terms(x0, g, B, is_bounded, bounds, x_scale=None):
    """Scale quadratic-model sparse_terms for trust-region and bound handling.

    Args:
        x0: Current parameter vector.
        g: Gradient vector in original parameter coordinates.
        B: Hessian/quadratic approximation in original coordinates.
        is_bounded: Whether to apply Coleman-Li bound scaling.
        bounds: Optional lower/upper bound array.
        x_scale: Optional user or Jacobian-based parameter scale.

    Returns:
        Tuple ``(g_hat, B_hat, C, root_v)`` used by the trust-region subproblem.
    """
    if x_scale is None:
        x_scale = np.ones(g.shape)

    if is_bounded:
        v, Jv = compute_v(g, x0, bounds, x_scale)
        root_v = np.sqrt(v)
        g_hat = x_scale * root_v * g
        d = x_scale * root_v
        C = np.diag(Jv * g)
        B_hat = d.reshape((1, -1)) * B * d.reshape((-1, 1)) + C

    else:

        g_hat = g * np.sqrt(x_scale)
        B_hat = np.sqrt(x_scale).reshape((1, -1)) * B * np.sqrt(x_scale).reshape((-1, 1))

        v = np.ones(g.shape) * x_scale
        root_v = np.sqrt(v)
        C = 0.0

    return g_hat, B_hat, C, root_v


def get_starting_delta(Delta, x0, g, is_bounded, bounds):
    """Choose an initial trust-region radius.

    Args:
        Delta: User-provided radius, or ``None`` to infer from ``x0``.
        x0: Starting parameter vector.
        g: Initial gradient vector.
        is_bounded: Whether bounds are active.
        bounds: Optional lower/upper bounds.

    Returns:
        Positive trust-region radius.
    """
    if Delta is None:
        if is_bounded:
            v, _ = compute_v(g, x0, bounds)
        else:
            v = 1.0
        Delta = np.linalg.norm(x0 / (np.clip(v, a_min=.0001, a_max=np.inf) ** .5))
    if Delta == 0:
        Delta = np.sqrt(len(x0))
    return Delta


def step_loop_over_delta(residual_func, weights, cost, x0, g_hat, B_hat, C, root_v, Delta, Delta_increase_factor,
                         rho_quad_model_accept, xtol, is_bounded, bounds, l2_penalties, regularize_to_values,
                         num_reflections=1, max_iter=10,
                         root_loss_function=None, f_scale=1.0, do_line_search=False,
                         try_newton_step=DEFAULT_NLLS_TRY_NEWTON_STEP, theta=DEFAULT_NLLS_REFLECTION_THETA):
    """Try candidate trust-region radii and return the best accepted step data.

    Args:
        residual_func: Callable residual function.
        weights: Optional observation weights.
        cost: Current penalised objective_function value.
        x0: Current parameter vector.
        g_hat: Scaled gradient.
        B_hat: Scaled Hessian/quadratic approximation.
        C: Diagonal correction term from bounded scaling.
        root_v: Scale vector mapping scaled steps to original coordinates.
        Delta: Current trust-region radius.
        Delta_increase_factor: Factor used to test larger radii after good steps.
        rho_quad_model_accept: Minimum model-ratio threshold for accepting radius expansion.
        xtol: Step tolerance.
        is_bounded: Whether finite bounds are active.
        bounds: Optional lower/upper bounds.
        l2_penalties: Optional L2 penalty weights.
        regularize_to_values: L2 penalty target values.
        num_reflections: Maximum reflected-path segments for bounded steps.
        max_iter: Maximum inner radius attempts.
        root_loss_function: Optional robust root-loss transform.
        f_scale: Robust-loss scale or ``'adaptive'``.
        do_line_search: Whether to extend accepted steps along their direction.
        try_newton_step: Whether to compare Newton steps.
        theta: Bound-interior truncation factor.

    Returns:
        Tuple containing step vectors, trust-region diagnostics, updated
        residuals/loss scale, and new objective_function components.
    """
    Delta_temp = Delta
    for cnt_inner in range(max_iter):
        p_scaled_temp, p_temp, step_case_temp, norm_dx_temp, predicted_decrease_temp \
            = get_next_step(is_bounded, x0, g_hat, B_hat, root_v, Delta_temp, xtol, bounds, num_reflections,
                            try_newton_step, theta)

        resid_temp, r1_temp, d_root_loss_temp, loss_scale_temp \
            = get_residuals(x0 + p_temp, residual_func, root_loss_function, f_scale)

        cost_new_temp = objective_func(r1_temp, x0+p_temp, l2_penalties, regularize_to_values, weights)
        actual_decrease_temp = cost_new_temp - cost
        rho_temp = (
            (actual_decrease_temp + 0.5 * np.dot(p_scaled_temp, C).dot(p_scaled_temp)) \
            / predicted_decrease_temp
            if predicted_decrease_temp < 0 else -np.inf)

        if rho_temp >= rho_quad_model_accept or cnt_inner == 0:
            norm_dx = norm_dx_temp
            p_scaled, p = p_scaled_temp, p_temp
            step_case = step_case_temp
            predicted_decrease = predicted_decrease_temp
            actual_decrease = actual_decrease_temp
            rho = rho_temp
            r1 = r1_temp
            resid = resid_temp
            d_root_loss = d_root_loss_temp
            loss_scale = loss_scale_temp
            cost_new = cost_new_temp
            if rho_temp >= rho_quad_model_accept and norm_dx_temp > Delta - 1e-8:
                Delta = Delta_temp
                Delta_temp *= Delta_increase_factor
            else:
                break
        else:
            break

    if do_line_search and actual_decrease < 0:
        _a = 1.1
        for _ in range(10):
            resid_temp, r1_temp, d_root_loss_temp, loss_scale_temp \
                = get_residuals(x0 + _a * p, residual_func, root_loss_function, f_scale)
            cost_new_temp = objective_func(r1_temp, x0 + _a * p, l2_penalties, regularize_to_values, weights=weights)
            actual_decrease = cost_new_temp - cost
            predicted_decrease = quadratic_form(p_scaled * _a, g_hat, B_hat)

            if cost_new_temp < cost_new and predicted_decrease < 0:

                p *= _a
                p_scaled *= _a

                rho = (
                    (actual_decrease + 0.5 * np.dot(p_scaled, C).dot(p_scaled)) \
                    / predicted_decrease
                    if predicted_decrease < 0 else -np.inf)
                norm_dx = np.linalg.norm(p_scaled)
                Delta = max(Delta, norm_dx)

                r1 = r1_temp
                resid = resid_temp
                d_root_loss = d_root_loss_temp
                loss_scale = loss_scale_temp
                # print("\n\t\t", _a, actual_decrease, cost_new_temp - cost_new, rho, predicted_decrease)

                cost_new = cost_new_temp
                _a *= 1.1

            else:
                # print("\n\t\t-----------------\n\t\t", _a, actual_decrease, cost_new_temp - cost_new)
                break

    return (p, p_scaled, norm_dx, rho, Delta, actual_decrease, predicted_decrease, r1, d_root_loss, cost_new,
            step_case, cnt_inner, loss_scale, resid)


def get_residuals(x, residual_func, root_loss_function, f_scale):
    """Evaluate raw and transformed residuals at a parameter vector.

    Args:
        x: Parameter vector.
        residual_func: Callable residual function.
        root_loss_function: Optional robust root-loss transform.
        f_scale: Robust-loss scale or ``'adaptive'`` for MAD scaling.

    Returns:
        Tuple ``(resid, r0, d_root_loss, loss_scale)``.
    """
    resid = residual_func(x)
    if root_loss_function is None:
        r0 = resid
        d_root_loss = None
        loss_scale = None
    else:
        if f_scale == 'adaptive':
            loss_scale = get_mad_scale(resid)
        else:
            loss_scale = f_scale
        r0, d_root_loss = root_loss_function(resid / loss_scale)
        r0, d_root_loss = r0 * loss_scale, d_root_loss * loss_scale

    return resid, r0, d_root_loss, loss_scale


def get_quadratic_approx_from_jacobian(Jac, resid, weights):
    """Build gradient and Hessian approximation from residuals and Jacobian.

    Args:
        Jac: Residual Jacobian, dense or sparse.
        resid: Current residual vector.
        weights: Optional observation weights.

    Returns:
        Tuple ``(g, B)`` where ``g = J'Wr`` and ``B = J'WJ``.
    """

    if isspmatrix(Jac):
        if weights is None:
            g = csr_matrix(resid).dot(Jac).toarray().flatten()
            B = Jac.transpose().dot(Jac).toarray()
        else:
            g = csr_matrix(weights * resid).dot(Jac).toarray().flatten()
            Jac_w = csc_matrix_by_column_array_broadcast(Jac, weights)
            B = Jac_w.transpose().dot(Jac).toarray()
            # B = Jac.transpose().dot(spdiags(weights)).dot(Jac).toarray()
    else:
        if weights is None:
            Jac_w = Jac
        else:
            Jac_w = Jac * weights.reshape((-1, 1))
        g = resid.dot(Jac_w).flatten()
        B = Jac.T.dot(Jac_w)

    return g, B


def adjust_quadratic_for_l2_regularization(g, B, x0, l2_penalties, regularize_to_values):
    """Adds ridge term to NLLS objective_function function.

    Args:
        g: Gradient vector to adjust in-place.
        B: Hessian/quadratic approximation to adjust in-place.
        x0: Current parameter vector.
        l2_penalties: Optional L2 penalty weights.
        regularize_to_values: L2 target values.

    Returns:
        Tuple ``(g, B)`` with L2 gradient/Hessian sparse_terms included.
    """
    if l2_penalties is not None:
        g += l2_penalties * (x0 - regularize_to_values)
        B += np.diag(l2_penalties)
    return g, B



def nlls_minimize_internal(residual_func, weights=None, x0=None, jacobian_func=None, num_params=None, bounds=None,
                           Delta=DEFAULT_NLLS_DELTA, Delta_floor=DEFAULT_NLLS_DELTA_FLOOR,
                           max_iter=DEFAULT_NLLS_MAX_ITER,
                           xtol=DEFAULT_NLLS_XTOL, ftol=DEFAULT_NLLS_FTOL, gtol=DEFAULT_NLLS_GTOL,
                           rho_quad_model_accept=DEFAULT_NLLS_RHO_QUAD_MODEL_ACCEPT,
                           rho_quad_model_reject=DEFAULT_NLLS_RHO_QUAD_MODEL_REJECT,
                           Delta_increase_factor=DEFAULT_NLLS_DELTA_INCREASE_FACTOR,
                           Delta_decrease_factor=DEFAULT_NLLS_DELTA_DECREASE_FACTOR,
                           rho_step_accept_floor=DEFAULT_NLLS_RHO_STEP_ACCEPT_FLOOR,
                           x_scale=DEFAULT_NLLS_X_SCALE, num_reflections=DEFAULT_NLLS_NUM_REFLECTIONS,
                           do_broyden_jac_update=DEFAULT_NLLS_DO_BROYDEN_JAC_UPDATE,
                           broyden_jac_update_cadence=DEFAULT_NLLS_BROYDEN_JAC_UPDATE_CADENCE,
                           do_line_search=DEFAULT_NLLS_DO_LINE_SEARCH,
                           debug=False, root_loss_function=None, f_scale=1.0,
                           prompt_user_for_more_iters=DEFAULT_NLLS_PROMPT_USER_FOR_MORE_ITERS,
                           keep_optimization_path=DEFAULT_NLLS_KEEP_OPTIMIZATION_PATH,
                           jac_method=DEFAULT_NLLS_JAC_METHOD,
                           wtd_total_sum_of_squares=None, theta=DEFAULT_NLLS_REFLECTION_THETA,
                           l2_penalties=None, regularize_to_values=None,
                           scale_l2_penalties=DEFAULT_NLLS_SCALE_L2_PENALTIES,
                           try_newton_step=DEFAULT_NLLS_TRY_NEWTON_STEP,
                           dense_threshold_mb=DEFAULT_DENSE_THRESHOLD_MB,
                           do_analytic_jac_jit=DEFAULT_NLLS_DO_ANALYTIC_JAC_JIT,
                           param_names=None, specification_name=None) -> NllsMinimizeInternalResult:
    """Fit nonlinear least squares with a trust-region reflective algorithm.

    Minimises a weighted residual sum of squares, optionally with robust
    root-loss residuals, box constraints, and L2 regularisation.  The solver
    builds a local quadratic model from the residual Jacobian, solves a
    trust-region subproblem, accepts/rejects steps using model agreement, and
    returns an internal result object with diagnostics.

    Args:
        residual_func: Callable returning residuals for a parameter vector.
        weights: Optional observation weights.
        x0: Optional starting parameter vector.
        jacobian_func: Optional callable returning the residual Jacobian.
        num_params: Number of parameters when ``x0`` is not provided.
        bounds: Optional ``(num_params, 2)`` lower/upper bound array.
        Delta: Initial trust-region radius; inferred when ``None``.
        Delta_floor: Lower bound for trust-region radius before stopping checks.
        max_iter: Maximum outer iterations.
        xtol: Parameter-step convergence tolerance.
        ftol: Objective-change convergence tolerance.
        gtol: Gradient optimality convergence tolerance.
        rho_quad_model_accept: Ratio threshold for model-quality acceptance.
        rho_quad_model_reject: Ratio threshold below which the radius shrinks.
        Delta_increase_factor: Multiplicative radius increase factor.
        Delta_decrease_factor: Multiplicative radius decrease factor.
        rho_step_accept_floor: Minimum ratio for accepting a parameter update.
        x_scale: Parameter scale, ``None``, or ``'jac'`` for Jacobian-based scaling.
        num_reflections: Number of reflected bound segments to consider.
        do_broyden_jac_update: Whether to update the quadratic approximation
            between fresh Jacobian evaluations.
        broyden_jac_update_cadence: Fresh-Jacobian cadence when Broyden is used.
        do_line_search: Whether to extend accepted steps along their direction.
        debug: Whether to print solver progress.
        root_loss_function: Optional robust root-loss name or object.
        f_scale: Robust-loss scale or ``'adaptive'`` for MAD scaling.
        prompt_user_for_more_iters: Whether/how to prompt after ``max_iter``.
        keep_optimization_path: Whether to store iterates and objective_function values.
        jac_method: ``'analytic'``, ``'mid'``, or ``'fwd'`` Jacobian method.
        wtd_total_sum_of_squares: Optional denominator for progress R-squared.
        theta: Bound-interior truncation/reflection parameter.
        l2_penalties: Optional scalar or vector of L2 penalties.
        regularize_to_values: Parameter values targeted by L2 regularisation.
        scale_l2_penalties: Whether to scale L2 penalties by sample size/weight sum.
        try_newton_step: Whether to compare Newton steps against Steihaug steps.
        dense_threshold_mb: Dense Jacobian/covariance threshold in MB.
        do_analytic_jac_jit: Whether to JIT analytic Jacobian code.
        param_names: Optional parameter names for the result object.
        specification_name: Optional label stored on the result object.

    Returns:
        ``NllsMinimizeInternalResult`` containing estimates and diagnostics.

    Examples
    --------
    Low-level trust-region NLLS solver — most users should call
    :func:`nlls` / :func:`NLLS` instead. Use this when you already have
    a residual callable and want the bare optimiser:

    >>> import numpy as np
    >>> from kanly.api import nlls_minimize_internal
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=200)
    >>> y = 1.0 + 3.0 * np.exp(-0.5 * x) + 0.4 * rng.normal(size=200)
    >>> def resid(beta):
    ...     return y - (beta[0] + beta[1] * np.exp(beta[2] * x))
    >>> result = nlls_minimize_internal(                       # doctest: +SKIP
    ...     resid, x0=np.array([0.0, 1.0, -0.1]), num_params=3,
    ...     max_iter=50)
    >>> result.x.round(2)                                      # doctest: +SKIP
    array([ 0.99,  3.04, -0.51])
    """

    _t = time.time()

    root_loss_function_orig = root_loss_function
    root_loss_function = get_root_loss_func(root_loss_function)

    if l2_penalties is None:
        pass

    else:
        if isinstance(l2_penalties, (float, int)):
            l2_penalties = [l2_penalties] * num_params
        l2_penalties = np.array(l2_penalties)# * num_params

        assert np.all(l2_penalties >= 0)  # TODO msg
        #assert root_loss_function_orig is None and f_scale is None  # TODO msg
        #print(f_scale)
        #assert f_scale is None  # TODO msg

    if regularize_to_values is None:
        if l2_penalties is not None:
            regularize_to_values = np.full(num_params, 0.0)
    else:
        if isinstance(regularize_to_values, (float, int)):
            regularize_to_values = [regularize_to_values] * num_params
        regularize_to_values = np.array(regularize_to_values)# * num_params

    assert f_scale == 'adaptive' or (isinstance(f_scale, (int, float)) and f_scale > 0)
    assert xtol >= 0 and ftol >= 0 and gtol >= 0 and max_iter > 0

    is_bounded = bounds is not None
    x0, num_params, is_bounded = get_x0(x0, num_params, bounds, is_bounded)
    if is_bounded:
        bounds = np.asarray(bounds)

    x0_start = x0.copy()
    num_params = len(x0)

    jac_method = jac_method.lower()
    assert jac_method in ('analytic', 'mid', 'fwd')
    if jacobian_func is None:
        if jac_method == 'analytic':
            if isinstance(residual_func, ResidualFunction):
                if debug:
                    _t_jac = time.time()
                    print("constructing jacobian analytically...", end='')
                jacobian_func, jac_result = residual_func.get_analytical_jacobian(
                    dense_threshold_mb=dense_threshold_mb, debug=debug, do_jit=do_analytic_jac_jit)
                if debug:
                    _t_jac = time.time()
                    print("%.2fs" % (time.time() - _t_jac))
                    print('Jacobian:\n')
                    print('\n\t\t'.join(jac_result['func_str_code'].split('\n')))
            else:
                raise Exception(
                    "`residual_func` callable must be of type `ResidualFunction` to use analytic jacobians.")
        else:
            jacobian_func = get_finite_diff_jacobian(
                residual_func, jac_method=jac_method, dense_threshold_mb=dense_threshold_mb)

    resid, r0, d_root_loss, loss_scale = get_residuals(x0, residual_func, root_loss_function, f_scale)

    nobs = r0.shape[0]

    # adjust regularization for nobs
    if l2_penalties is not None:
        if scale_l2_penalties:
            if weights is not None:
                l2_penalties = l2_penalties * sum(weights)
            else:
                l2_penalties = l2_penalties * nobs

    cost = objective_func(r0, x0, l2_penalties, regularize_to_values, weights)

    Jac = compute_jacobian(jacobian_func, x0, d_root_loss)

    g, B = get_quadratic_approx_from_jacobian(Jac, r0, weights)
    g, B = adjust_quadratic_for_l2_regularization(g, B, x0, l2_penalties, regularize_to_values)

    Delta = get_starting_delta(Delta, x0, g, is_bounded, bounds)

    if debug:
        dbl_bar = '=' * 36
        bar = '-' * 36
        print(
            dbl_bar + "\n" +
            "Nonlinear Least Squares\n" +
            bar + "\n" +
            f"Nobs:             {nobs}\n" +
            f"Num Params:       {num_params}\n" +
            f"Initial Cost:     {'%.2e' % cost}\n" +
            f"Cost / Nobs:      {'%.2e' % (cost / nobs)}\n" +
            f"Bounded:          {is_bounded}\n" +
            "\n" +
            f"maxiter:          {max_iter}\n" +
            f"gtol:             {'%.2e' % gtol}\n" +
            f"ftol:             {'%.2e' % ftol}\n" +
            f"xtol:             {'%.2e' % xtol}\n" +
            f"Delta:            {'%.3e' % Delta}\n" +
            f"x_scale:          {x_scale}\n" +
            "\n" +
            f"rho_reject:          {'%.2e' % rho_quad_model_reject}\n" +
            f"rho_accept:          {'%.2e' % rho_quad_model_accept}\n" +
            f"min_rho_step:        {'%.2e' % rho_step_accept_floor}\n" +
            f"Delta_scale_up:      {'%.2e' % Delta_increase_factor}\n" +
            f"Delta_scale_down:    {'%.2e' % Delta_decrease_factor}\n" +
            f"Delta_floor:         {'%.2e' % Delta_floor}\n" +
            "\n" +
            f"broyden:             {do_broyden_jac_update}\n" +
            f"broyden_cadence:     {broyden_jac_update_cadence}\n" +
            f"jac_method:          {jac_method}\n" +
            f"try_newton:          {try_newton_step}\n" +
            "\n" +
            f"max_reflections:     {num_reflections}\n" +
            f"theta:               {'%.6f' % theta}\n" +
            "\n" +
            f"root loss func:      {str(root_loss_function)}\n" +
            f"f_scale:             {f_scale}\n" +
            "\n" +
            f"do line search:      {str(do_line_search)}\n" +
            dbl_bar
        )

    itr = 0
    cnt_outer = 0
    message = 'Did not converge'
    converged = False
    status = 0
    fresh_jac = 0

    optimization_path = [{'x': x0, 'fun': cost}] if keep_optimization_path else None

    itrs_since_updated = 0

    Delta_prev_iter = np.inf

    while True:

        try:

            if np.any(~np.isfinite(g)):
                converged = False
                message = f'Breaking: non-finite gradient elements on iteration {itr}!'
                if debug:
                    print(message)
                break

            if np.any(~np.isfinite(x0)):
                converged = False
                message = f'Breaking: non-finite parameters on iteration {itr}!'
                if debug:
                    print(message)
                break

            itrs_since_updated += 1

            if x_scale == 'jac':
                if isspmatrix(Jac):
                    scale = 1.0 / np.clip(spnorm(Jac, axis=0), a_min=.00001, a_max=np.inf)
                else:
                    scale = 1.0 / np.clip(np.linalg.norm(Jac, axis=0), a_min=.00001, a_max=np.inf)
                scale *= num_params / sum(scale)
            elif x_scale is None:
                scale = 1.0
            else:
                scale = x_scale

            g_hat, B_hat, C, root_v = transform_quadratic_approx_terms(x0, g, B, is_bounded, bounds, scale)

            (p, p_scaled, norm_dx, rho, Delta_step, actual_decrease, predicted_decrease, r1, d_root_loss, cost_new,
             step_case, cnt_inner, loss_scale, resid
             ) = step_loop_over_delta(residual_func, weights, cost, x0, g_hat, B_hat, C, root_v, Delta,
                                      Delta_increase_factor,
                                      rho_quad_model_accept, xtol, is_bounded, bounds,
                                      l2_penalties, regularize_to_values,
                                      num_reflections=num_reflections, max_iter=10, do_line_search=do_line_search,
                                      root_loss_function=root_loss_function, f_scale=f_scale,
                                      try_newton_step=try_newton_step, theta=theta)

            old_Delta = Delta
            if (np.isnan(rho) or rho < rho_quad_model_reject or actual_decrease >= 0) and fresh_jac == 0:
                Delta = min(Delta, norm_dx) / Delta_decrease_factor
                # print("A ", (rho, actual_decrease, predicted_decrease, fresh_jac, norm_dx, Delta, Delta_step))
            elif rho > rho_quad_model_accept and norm_dx >= Delta - 1e-6:
                Delta = max(Delta, norm_dx) * Delta_increase_factor
                # print("B ", (rho, actual_decrease, predicted_decrease, fresh_jac, norm_dx, Delta, Delta_step))
            else:
                pass
                # print("C ", (rho, actual_decrease, predicted_decrease, fresh_jac, norm_dx, Delta, Delta_step))

            updated = False
            old_cost = cost
            fresh_jac_old = fresh_jac
            if actual_decrease < 0 and rho >= rho_step_accept_floor:
                updated = True
                fresh_jac += 1

                x0 = x0 + p
                if is_bounded:
                    x0 = np.clip(x0, a_min=bounds[:, 0], a_max=bounds[:, 1])
                dfun = r1 - r0
                r0 = r1
                cost = cost_new

                if do_broyden_jac_update:
                    #     J_update = (dfun - Jac.dot(csc_matrix(p).reshape((-1, 1))).toarray().flatten()) \
                    #                / np.linalg.norm(p) ** 2
                    #
                    #     for c in range(Jac.shape[1]):
                    #         idx = Jac.indices[Jac.indptr[c]:Jac.indptr[c + 1]]
                    #         Jac.data[Jac.indptr[c]:Jac.indptr[c + 1]] += p[c] * J_update[idx]
                    #
                    #     g, B = get_quadratic_approx_from_jacobian(Jac, r0, weights)

                    old_grad = g
                    g = np.zeros(num_params)
                    dxx = 1e-8
                    for _i in range(num_params):
                        x0_copy = x0.copy()
                        x0_copy[_i] += dxx
                        temp = objective_func(residual_func(x0_copy), x0_copy,
                                              l2_penalties, regularize_to_values, weights=weights)
                        g[_i] = (temp - cost) / dxx

                    grad_diff = g - old_grad
                    v = B.dot(p)
                    B += (
                            np.outer(grad_diff, grad_diff) / np.dot(grad_diff, p)
                            - np.outer(v, v) / np.dot(v, p)
                    )

                    # TODO ridge adjustment for Broyden?
                    #g, B = adjust_quadratic_for_l2_regularization(g, B, x0, l2_penalties, regularize_to_values)

            if (((np.isnan(rho) or actual_decrease > 0 or rho < rho_quad_model_accept) or not do_broyden_jac_update
                 or fresh_jac % broyden_jac_update_cadence == 0)
                    and fresh_jac):
                fresh_jac = 0
                Jac = compute_jacobian(jacobian_func, x0, d_root_loss)

                g, B = get_quadratic_approx_from_jacobian(Jac, r0, weights)
                g, B = adjust_quadratic_for_l2_regularization(g, B, x0, l2_penalties, regularize_to_values)

            active_mask = np.zeros(num_params)
            if is_bounded:
                active_mask[np.abs(bounds[:, 1] - x0) < xtol] = 1
                active_mask[np.abs(bounds[:, 0] - x0) < xtol] = -1

            optimality = np.linalg.norm(g * root_v ** 2, ord=np.inf)
            iter_info_dict = [
                                 {'name': 'iter', 'len': 6, 'format': '%6d', 'value': itr},
                                 {'name': 'F=cost', 'len': 13, 'format': '%13.4e', 'value': cost},
                                 {'name': 'dF/F', 'len': 11, 'format': '%11.2e', 'value': actual_decrease / old_cost},
                                 {'name': 'pred/F', 'len': 11, 'format': '%11.2e',
                                  'value': predicted_decrease / old_cost},
                                 {'name': 'F/n', 'len': 13, 'format': '%13.4e', 'value': cost / nobs},
                                 {'name': 'rho', 'len': 10, 'format': '%10.2e', 'value': rho},
                                 {'name': 'Delta', 'len': 10, 'format': '%10.2e', 'value': old_Delta},
                                 {'name': '|dx|', 'len': 10, 'format': '%10.2e', 'value': norm_dx},
                                 {'name': 'optimality', 'len': 12, 'format': '%12.2e', 'value': optimality},
                                 {'name': 'active bnds', 'len': 14, 'format': '%14d',
                                  'value': np.sum(active_mask != 0)},
                                 {'name': 'accepted', 'len': 10, 'format': '%10s', 'value': updated},
                                 {'name': 'time', 'len': 11, 'format': '%10.2fs', 'value': time.time() - _t},
                                 {'name': 'step type', 'len': 11, 'format': '%11s', 'value': step_case},
                                 {'name': 'fresh J', 'len': 11, 'format': '%11s', 'value': fresh_jac_old},
                                 {'name': 'cnt_inner', 'len': 11, 'format': '%11d', 'value': cnt_inner},
                                 {'name': 'loss scale', 'len': 13, 'format': '%13s',
                                  'value': 'None' if loss_scale is None else '%.3e' % loss_scale},
                             ] + ([{'name': 'rsquared', 'len': 12, 'format': '%12.6f',
                                    'value': 1 - cost / wtd_total_sum_of_squares

                                    }] if wtd_total_sum_of_squares is not None and root_loss_function_orig is None else [])

            if keep_optimization_path:
                optimization_path.append({'x': x0.copy(), 'fun': cost})

            if debug:
                if cnt_outer == 0:
                    print_iter_info(iter_info_dict, is_header=True)
                print_iter_info(iter_info_dict, is_header=False)

            if (updated and fresh_jac_old == 0) or Delta < Delta_floor:
                converged, message, status = check_convergenge(
                    optimality, actual_decrease, nobs, old_cost, x0 / np.clip(root_v, a_min=.000001, a_max=np.inf),
                    p_scaled, gtol, ftol, xtol)
                if converged:
                    break
                itrs_since_updated = 0

            itr += 1  # TODO why in the block above?

            if itr == max_iter:
                incremental_iters = user_prompt_for_more_iters_method(
                    f"\n\t[iteration = {'%d' % itr}, |dx| = {'%.2e' % norm_dx}, optimality = {'%.2e' % optimality}, "
                    f"dF/F = {'%.2e' % (actual_decrease / old_cost)}]",
                    prompt_user_for_more_iters
                )
                if incremental_iters == 0:
                    break
                else:
                    max_iter += incremental_iters

            # if do_line_search and not updated and old_Delta >= Delta_prev_iter:
            #     print("REDUCE!!!")
            #     Delta /= Delta_decrease_factor*3

            cnt_outer += 1
            Delta_prev_iter = Delta

        except KeyboardInterrupt:
            message = 'Did not converge: keyboard interrupt'
            print("\nProcess interrupted, breaking...\n")
            break

        except Exception as exc:
            print(traceback.format_exc())
            raise exc

    if debug:
        print_iter_info(iter_info_dict, is_footer=True)
        if message is not None:
            print('\n\t' + message + '\n')
        else:
            print('\n\tFailed to converge!')

    # print("Cost              ", cost)
    # print("penalty           ", penalty_func(x0, l2_penalties, regularize_to_values))
    # print("cost min penalty  ", cost-penalty_func(x0, l2_penalties, regularize_to_values))

    penalty_value = penalty_func(x0, l2_penalties, regularize_to_values)
    fit_elapsed =  time.time() - _t
    wresid = resid if weights is None else resid * np.sqrt(weights)
    ncp = csc_matrix(get_matrix_inverse_internal(B))

    if message is None:
        message = 'Failed to converge!'

    return NllsMinimizeInternalResult(
        params=x0, l2_penalties=l2_penalties, regularize_to_values=regularize_to_values,
        scale_l2_penalties=scale_l2_penalties, cost=cost, penalty=penalty_value,
        jac=Jac, grad=g, hessian=B, converged=converged, iterations=itr + 1,
        fit_elapsed=fit_elapsed, resid=resid, wresid=wresid, root_loss_resid=r0,
        optimality=optimality, norm_dx=norm_dx, message=message, status=status,
        is_bounded=is_bounded, bounds=np.array(bounds), active_mask=active_mask, start_params=x0,
        dF_over_F=actual_decrease / old_cost, v=root_v ** 2, loss_scale=loss_scale, loss=root_loss_function,
        root_loss_function_orig=root_loss_function_orig, normalized_cov_params=ncp,
        optimization_path=optimization_path, jac_method=jac_method, dense_threshold_mb=dense_threshold_mb,
        jacobian_function_callable=jacobian_func, param_names=param_names,
        specification_name=specification_name
    )

    # return {'params': x0,
    #         'l2_penalties': l2_penalties,
    #         'regularize_to_values': regularize_to_values,
    #         'scale_l2_penalties': scale_l2_penalties,
    #         'cost': cost,
    #         'penalty': penalty_func(x0, l2_penalties, regularize_to_values),
    #         'jac': Jac,
    #         'grad': g,
    #         'hessian': B,
    #         'converged': converged,
    #         'iterations': itr + 1,
    #         'fit_elapsed': time.time() - _t,
    #         'resid': resid,
    #         'wresid': resid if weights is None else resid * np.sqrt(weights),
    #         'root_loss_resid': r0,
    #         'optimality': optimality,
    #         'norm_dx': norm_dx,
    #         'message': message,
    #         'status': status,
    #         'is_bounded': is_bounded,
    #         'bounds': np.array(bounds),
    #         'active_mask': active_mask,
    #         'start_params': x0_start,
    #         'dF/F': actual_decrease / old_cost,
    #         'v': root_v ** 2,
    #         'loss_scale': loss_scale,
    #         'loss': root_loss_function,
    #         'root_loss_function_orig': root_loss_function_orig,
    #         'normalized_covariance_parameters': csc_matrix(get_matrix_inverse_internal(B)),
    #         'optimization_path': optimization_path,
    #         'jac_method': jac_method,
    #         'dense_threshold_mb': dense_threshold_mb,
    #         'jacobian_function_callable': jacobian_func,
    #         }


# if __name__ == '__main__':
#
#     from kanly.api import nlls, lm
#     from kanly.bayes.bayesian_model import BayesianNonlinearLeastSquaresModel
#     import pandas as pd
#     import numpy as np
#     from kanly.api import minimize
#
#
#     n = 1000
#     np.random.seed(0)
#     x = np.random.rand(n)
#     y = 1.2 * np.random.randn(n) + 3 * x - 2.5
#     df = pd.DataFrame({
#         'x': x, 'y': y
#     })
#     mean_x = -2.2
#     penalty = 62.44264838704694
#
#     fit_nlls = nlls('[y]~{Int}+{x}*[x]', df, l2_penalties={'x': penalty},
#                     regularize_to_values={'x': mean_x},
#                     scale_l2_penalties=False
#                     )
#     print(fit_nlls)
#
#
#     def objective_function(params):
#         return np.sum((df.y - params[0] - params[1] * df.x) ** 2) / 2 + penalty * ((params[1] - mean_x) ** 2) / 2
#
#
#     res = minimize(objective_function, [0, 1])
#     print(res)
#     print(np.sum((df.y - res.x[0] - res.x[1] * df.x) ** 2) / 2, penalty * ((res.x[1] - mean_x) ** 2) / 2)
#     print(np.sum((df.y - fit_nlls.params[0] - fit_nlls['x'] * df.x) ** 2) / 2,
#           penalty * ((fit_nlls['x'] - mean_x) ** 2) / 2)
#     print(penalty_func(fit_nlls.params, fit_nlls.l2_penalties, fit_nlls.regularize_to_values))
#     print(sum(fit_nlls.l2_penalties * (fit_nlls.params - fit_nlls.regularize_to_values) ** 2) / 2)
#     print(fit_nlls.l2_penalties, fit_nlls.regularize_to_values)

# if __name__ == '__main__':
#
#     from kanly.api import nlls, lm, nlls_en, elastic_net
#     from kanly.bayes.bayesian_model import BayesianNonlinearLeastSquaresModel
#     import pandas as pd
#     import numpy as np
#
#
#     n = 1000
#     np.random.seed(0)
#     x = np.random.rand(n)
#     y = 1.2 * np.random.randn(n) + 3 * x - 2.5
#     df = pd.DataFrame({
#         'x': x, 'y': y
#     })
#     mean_x = -4
#     penalty = 1.4
#
#     fit_ridge = lm('y ~ x', df, ridge_kwds={'alpha': {'x': penalty}, 'normalize': False})
#     print(fit_ridge)
#
#     fit_en = elastic_net('y~x', df, alpha={'x': penalty},
#                          l1_ratio=0,
#                          regularize_to_values={'x': mean_x},
#                          normalize=False
#                          )
#     print(fit_en)
#
#     fit_nlls = nlls('[y]~{Int}+{x}*[x]', df,
#                     l2_penalties={'x': penalty},
#                     regularize_to_values={'x': mean_x},
#                     scale_l2_penalties=True,
#                     jac_method='analytic',
#                     )
#     print(fit_nlls)
#
#     fit_nlls_en = nlls_en('[y]~{Int}+{x}*[x]', df,
#                         alpha={'x': penalty},
#                         l1_ratio=0,
#                         normalize=False,
#                         regularize_to_values={'x': mean_x},
#                         )
#     print(fit_nlls_en)
