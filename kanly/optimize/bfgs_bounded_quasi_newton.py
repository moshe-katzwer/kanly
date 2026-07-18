from __future__ import absolute_import, print_function

import time

import numpy as np
from numba import njit

from kanly.optimize.optimization_results import OptimizationResult
from kanly.optimize.utilities import update_bfgs_hessian_approx
from kanly.utils.user_prompt_for_more_iters import user_prompt_for_more_iters_method
from kanly.utils.util import print_iter_info

DEFAULT_BFGS_PQN_XTOL = 1e-8
DEFAULT_BFGS_PQN_FTOL = 1e-15
DEFAULT_BFGS_PQN_GTOL = 1e-4
DEFAULT_BFGS_PQN_MAXITER = 500
DEFAULT_BFGS_PQN_ONESIDED_FD = None
DEFAULT_BFGS_PQN_DX_DF = 1e-6
DEFAULT_BFGS_PQN_USER_PROMPT_FOR_MORE_ITERS = False
DEFAULT_BFGS_PQN_MOMENTUM = 0.0
DEFAULT_BFGS_PQN_SEED = 0
DEFAULT_BFGS_PQN_MAXIMIZE = False
DEFAULT_BFGS_PQN_C1_WOLFE = 1e-4
DEFAULT_BFGS_PQN_C2_WOLFE = 0.8
DEFAULT_BFGS_PQN_WOLFE_REDUCTION_SCALE = 3
DEFAULT_BFGS_PQN_WOLFE_INCREASE_SCALE = 1.5
DEFAULT_BFGS_PQN_PBAR_UPDATE_CADENCE = .33
DEFAULT_BFGS_DEBUG = False
DEFAULT_BFGS_SAVE_OPTIMIZATION_PATH = False


def grad_finite_diff(fun, x, onesided=True, dx=1e-6):
    """Estimate the full gradient of an objective_function by finite differences.

    Args:
        fun: Objective callable accepting a parameter vector.
        x: Point at which to estimate the gradient.
        onesided: If True, use forward differences. If False, use centered
            differences.
        dx: Relative finite-difference step size. Each coordinate step is
            scaled by ``max(1, abs(x[i]))``.

    Returns:
        One-dimensional array of finite-difference derivative estimates.
    """
    df_dx = np.full(len(x), 0.0, dtype=float)

    f0 = fun(x)
    for i in range(len(x)):

        dx_i = dx * max(1.0, abs(x[i]))
        x1 = x.copy()
        x1[i] += dx_i

        if onesided:
            df_dx[i] = (fun(x1) - f0) / dx_i

        else:
            x2 = x.copy()
            x2[i] -= dx_i

            df_dx[i] = (fun(x1) - fun(x2)) / (2 * dx_i)
    return df_dx


@njit
def project(x, p, alpha, lb, ub):
    """Project a line-search step back into the feasible box.

    Args:
        x: Current parameter vector.
        p: Search direction.
        alpha: Step length.
        lb: Lower bounds for the parameter vector.
        ub: Upper bounds for the parameter vector.

    Returns:
        ``x + alpha * p`` clipped coordinate-wise to ``[lb, ub]``.
    """
    return np.clip(x + alpha * p, lb, ub)

@njit
def is_valid(x, lb, ub):
    """Check that a point and bounds define a valid box-constrained problem.

    Args:
        x: Parameter vector to validate.
        lb: Lower bounds with the same length as ``x``.
        ub: Upper bounds with the same length as ``x``.

    Returns:
        True when ``x`` is inside the bounds and every lower bound is strictly
        less than its matching upper bound.
    """
    return np.all(x >= lb) and np.all(x <= ub) and np.all(lb < ub)


#@njit
def get_direction(g, B, free_set=None):
    """Compute a descent direction from a Hessian approximation.

    Args:
        g: Gradient vector at the current iterate.
        B: Hessian or inverse-curvature approximation to solve against.
        free_set: Optional indices that are allowed to move. If omitted, all
            coordinates are considered free.

    Returns:
        Tuple ``(p, B_out)`` where ``p`` is a descent direction and ``B_out``
        is either the provided matrix or an identity reset when the solve fails
        or produces a non-descent direction.
    """
    n = len(g)

    if free_set is None or len(free_set) == n:
        try:
            p = np.linalg.solve(B, -g)
            if p.dot(g) >= 0:
                raise Exception
            else:
                return p, B
        except:
            # Fall back to steepest descent when the Hessian approximation is
            # singular, ill-conditioned, or yields a non-descent step.
            return -g, np.eye(n)

    else:
        B_sub = B[np.ix_(free_set, free_set)]
        g_sub = g[free_set]
        try:
            p_sub = np.linalg.solve(B_sub, -g_sub)
            p = np.zeros(n)
            p[free_set] = p_sub
            if p.dot(g) >= 0:
                raise Exception
            else:
                return p, B

        except:
            # Reset only after a failed restricted solve; the returned step
            # still respects the free coordinates chosen by the caller.
            B = np.eye(n)
            p = np.zeros(n)
            p[free_set] = -g[free_set]
            return p, B



class BFGSPQNResults(OptimizationResult):
    """Result object returned by bounded projected quasi-Newton optimization."""

    def __init__(self, fun, x, num_params, grad, grad_projected, hess, converged, message, time_elapsed, fun_callable,
                 ferr, xerr, gnorm, iter, B0, ub_binding, lb_binding, bounds, options, maximize, optimization_path):
        """Create a projected quasi-Newton result record.

        Args:
            fun: Final objective_function value in the user's original optimization
                direction.
            x: Final parameter vector.
            num_params: Number of optimized parameters.
            grad: Gradient at the final iterate.
            grad_projected: Gradient with bound-blocked coordinates zeroed.
            hess: Final Hessian or BFGS Hessian approximation.
            converged: Whether a stopping criterion was met.
            message: Human-readable convergence or failure message.
            time_elapsed: Wall-clock runtime in seconds.
            fun_callable: Original objective_function callable supplied by the user.
            ferr: Final relative objective_function-change diagnostic.
            xerr: Final relative parameter-change diagnostic.
            gnorm: Final projected-gradient convergence diagnostic.
            iter: Number of quasi-Newton iterations completed.
            B0: User-supplied initial Hessian approximation, if any.
            ub_binding: Boolean mask for coordinates binding at upper bounds.
            lb_binding: Boolean mask for coordinates binding at lower bounds.
            bounds: Bounds used by the optimizer, or None for unbounded runs.
            options: Solver options and starting values.
            maximize: Whether the user requested maximization.
            optimization_path: Optional list of per-iteration state snapshots.
        """
        super().__init__(fun, x, num_params, grad, grad_projected, converged, message, time_elapsed, fun_callable,
                         ferr, xerr, iter, bounds, options, maximize, optimization_path)

        self.hess = hess
        self.gnorm = gnorm
        self.B0 = B0
        self.ub_binding = ub_binding
        self.lb_binding = lb_binding


def bfgs_pqn(
        fun, x0, bounds=None, B0=None, xtol=DEFAULT_BFGS_PQN_XTOL, ftol=DEFAULT_BFGS_PQN_FTOL,
        gtol=DEFAULT_BFGS_PQN_GTOL, maxiter=DEFAULT_BFGS_PQN_MAXITER,
        onesided_fd=DEFAULT_BFGS_PQN_ONESIDED_FD, dx_fd=DEFAULT_BFGS_PQN_DX_DF,
        user_prompt_for_more_iters=DEFAULT_BFGS_PQN_USER_PROMPT_FOR_MORE_ITERS, momentum=DEFAULT_BFGS_PQN_MOMENTUM,
        seed=DEFAULT_BFGS_PQN_SEED, maximize=DEFAULT_BFGS_PQN_MAXIMIZE,
        c1_wolfe=DEFAULT_BFGS_PQN_C1_WOLFE, c2_wolfe=DEFAULT_BFGS_PQN_C2_WOLFE,
        debug=DEFAULT_BFGS_DEBUG,
        wolfe_reduction_scale=DEFAULT_BFGS_PQN_WOLFE_REDUCTION_SCALE,
        line_search_increase_scale=DEFAULT_BFGS_PQN_WOLFE_INCREASE_SCALE,
        save_optimization_path=DEFAULT_BFGS_SAVE_OPTIMIZATION_PATH,
        gradient_callable=None, hessian_callable=None,
        prior_bfgs_num_call=1, max_total_bfgs_calls=10, prior_bfgs_B0=None,
    ) -> BFGSPQNResults:
    """
    Minimize or maximize an objective_function with projected quasi-Newton steps.

    The algorithm builds BFGS Hessian approximations, restricts search
    directions to coordinates that can move within the bounds, projects each
    candidate line-search step into the feasible box, and stops on parameter,
    objective_function, or projected-gradient tolerances.

    Args:
        fun: Objective callable accepting a one-dimensional parameter vector.
        x0: Initial feasible parameter vector.
        bounds: Optional 2 x p array-like object. The first row contains lower
            bounds and the second row contains upper bounds.
        B0: Optional initial Hessian approximation. A scalar is expanded to a
            scaled identity matrix; an array is copied.
        xtol: Convergence tolerance for relative parameter changes.
        ftol: Convergence tolerance for relative objective_function improvement.
        gtol: Convergence tolerance for the projected-gradient norm.
        maxiter: Maximum number of quasi-Newton iterations for one run.
        onesided_fd: Finite-difference mode for numerical gradients. True uses
            forward differences, False uses centered differences, and None
            follows the historical default path, which is treated as centered
            differences by ``grad_finite_diff``.
        dx_fd: Relative finite-difference step size.
        user_prompt_for_more_iters: Whether to prompt for extra iterations
            after ``maxiter`` is reached.
        momentum: Fraction of the previous accepted step to add when it remains
            a descent direction.
        seed: Stored in options for reproducibility metadata.
        maximize: If True, maximize ``fun`` by minimizing ``-fun``.
        c1_wolfe: Armijo sufficient-decrease constant for line search.
        c2_wolfe: Curvature Wolfe constant stored with options; currently not
            enforced by the line-search loop.
        debug: Whether to print iteration diagnostics.
        wolfe_reduction_scale: Factor by which failed line-search steps are
            reduced.
        line_search_increase_scale: Factor by which successful line-search
            steps are expanded while the objective_function keeps improving.
        save_optimization_path: Whether to store accepted iterates and
            objective_function values in the result.
        gradient_callable: Optional analytical gradient callable. When omitted,
            finite differences are used.
        hessian_callable: Optional analytical Hessian callable. When supplied,
            it replaces BFGS updates.
        prior_bfgs_num_call: Internal restart counter used when non-finite
            values are encountered.
        max_total_bfgs_calls: Maximum number of recursive restarts after
            non-finite values.
        prior_bfgs_B0: Internal record of the previous restart's Hessian scale.

    Returns:
        A ``BFGSPQNResults`` instance containing the final point, objective_function
        value, gradients, Hessian approximation, convergence diagnostics,
        binding-bound masks, options, and optional optimization path.

    Examples
    --------
    Bounded BFGS minimisation of the Rosenbrock function:

    >>> import numpy as np
    >>> from kanly.api import bfgs_pqn
    >>> def rosen(x):
    ...     return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    >>> bounds = np.array([[-2.0, -2.0],
    ...                    [ 2.0,  2.0]])
    >>> res = bfgs_pqn(rosen, x0=np.array([-1.2, 1.0]), bounds=bounds)
    >>> res.x.round(3)                                # doctest: +SKIP
    array([1., 1.])
    >>> res.converged                                  # doctest: +SKIP
    True

    The alias ``bfgs`` points at the same function.
    """

    with np.errstate(divide='ignore', invalid='ignore'):

        if debug:
            settings = dict(maxiter=maxiter, xtol=xtol, gtol=gtol, ftol=ftol, maximize=maximize,
                            c1_wolfe=c1_wolfe, c2_wolfe=c2_wolfe, wolfe_reduction_scale=wolfe_reduction_scale,
                            line_search_increase_scale=line_search_increase_scale,
                            momentum=momentum)

        time0 = time.time()

        onesided_fd_orig = onesided_fd

        fun_original = fun
        if maximize:
            fun = lambda z: -fun_original(z)

        if gradient_callable is None:
            grad_func = lambda x: grad_finite_diff(fun, x, onesided_fd_orig, dx_fd)
        else:
            if maximize:
                grad_func = lambda x: -gradient_callable(x)
            else:
                grad_func = gradient_callable

        has_hessian = hessian_callable is not None
        if has_hessian:
            if maximize:
                hessian_func = lambda x: -hessian_callable(x)
            else:
                hessian_func = hessian_callable

        num_params = len(x0)
        is_bounded = bounds is not None

        x0_start = x0.copy()
        x0 = np.array(x0, dtype=np.float64, copy=True)

        if is_bounded:
            bounds = np.array(bounds)
            if bounds.shape[1] != x0.shape[0]:
                raise Exception("`bounds` must be 2 x p, where p is dimension of `x`. "
                                "The first row is the lb, the second the ub.")
            if bounds.shape[0] != 2:
                raise Exception("`bounds` must have 2 rows!")
            lb = bounds[0]
            ub = bounds[1]
            # print(lb, x0, ub)
            assert is_valid(x0, lb, ub)
        else:
            lb = -np.inf * np.ones(num_params)
            ub = +np.inf * np.ones(num_params)

        if has_hessian:
            B0 = hessian_func(x0)
            B0_orig = None
        else:
            B0_orig = B0
            if B0 is None:
                # Start with a large diagonal Hessian approximation so the
                # first solve produces conservative steps in high dimensions.
                B0 = np.eye(num_params) * min(10 ** num_params, 1_000_000)
                # print(B0)
            else:
                if isinstance(B0, (float, int)):
                    B0 = np.eye(num_params) * B0
                else:
                    B0 = B0.copy()

        g0 = grad_func(x0)
        f0 = fun(x0)

        if ~np.isfinite(f0) or np.any(~np.isfinite(g0)):
            raise Exception("Starting point `x0` must generate finite function and gradient value!")

        #if debug:
        #    pbar = tqdm(range(maxiter), position=0, leave=False)

        converged = False
        message = 'Did not converge!'
        itr_ = 0

        p_last = 0

        cnt_fail = 0
        last_pbar_update_time = 0

        xerr, ferr, gerr = 1, 1, 1

        if save_optimization_path:
            optimization_path = [{'x': x0, 'fun': f0}]
        else:
            optimization_path = None

        last_alpha = 1.0
        itr_info = None
        while True:

            if ~np.isfinite(f0) or np.any(~np.isfinite(g0)):

                if prior_bfgs_num_call == max_total_bfgs_calls:
                    if debug:
                        print(f"Ran out of BFGS restarts ({max_total_bfgs_calls})!")
                    break

                if prior_bfgs_B0 is None:
                    B0_new = 10.
                else:
                    B0_new = 10 * prior_bfgs_B0

                # Restart from the original point with a larger diagonal
                # Hessian scale when the current run reaches non-finite values.
                if debug:
                    print("\nStarting point `x0` must generate finite function and gradient value! Restarting...\n")
                return bfgs_pqn(
                    fun_original, x0_start, bounds=bounds, B0=B0_new, xtol=xtol, ftol=ftol,
                    gtol=gtol, maxiter=maxiter,
                    onesided_fd=onesided_fd, dx_fd=dx_fd,
                    user_prompt_for_more_iters=user_prompt_for_more_iters,
                    momentum=momentum,
                    seed=seed, maximize=maximize,
                    c1_wolfe=c1_wolfe, c2_wolfe=c2_wolfe,
                    debug=debug,
                    wolfe_reduction_scale=wolfe_reduction_scale,
                    line_search_increase_scale=line_search_increase_scale,
                    save_optimization_path=save_optimization_path,
                    gradient_callable=gradient_callable, hessian_callable=hessian_callable,
                    prior_bfgs_num_call=prior_bfgs_num_call + 1,
                    max_total_bfgs_calls=max_total_bfgs_calls, prior_bfgs_B0=B0_new,
                )


            try:

                if onesided_fd_orig is None:
                    # Keep the historical adaptive flag calculation for
                    # diagnostics/restarts; the finite-difference callable was
                    # built from the original mode selected above.
                    onesided_fd = (xerr > 1e-6 and ferr > 1e-6 and gerr > 1e-1) and onesided_fd
                else:
                    onesided_fd = onesided_fd

                x0_begin_itr = x0
                f0_begin_itr = f0

                # -------------
                # Get direction

                # generate first free set
                if is_bounded:
                    # Coordinates are free only if the gradient points toward
                    # the feasible interior rather than into an active bound.
                    free_set1 = np.argwhere(
                        ((g0 < 0) & (ub > x0))
                        | ((g0 > 0) & (x0 > lb))
                    ).flatten()
                else:
                    free_set1 = np.arange(num_params)

                # compute a search direction
                p, B0 = get_direction(g0, B0, free_set1)

                # compute second free set
                if is_bounded:
                    # Intersect with coordinates where the proposed direction
                    # can actually move before being clipped by the bounds.
                    free_set2 = np.argwhere(
                        ((p > 0) & (ub > x0))
                        | ((p < 0) & (x0 > lb))
                    ).flatten()
                else:
                    free_set2 = np.arange(num_params)

                free_set = np.array(list(set(free_set1) & set(free_set2)))
                if len(free_set) == 0:
                    if not has_hessian:
                        B0 = np.eye(num_params)
                    p, B0 = get_direction(g0, B0, free_set1)
                else:
                    p, B0 = get_direction(g0, B0, free_set)

                if momentum > 0:
                    if np.dot(p + momentum * p_last, g0) < 0:
                        p = p + momentum * p_last

                # ---------------
                # Choose step size
                alpha = last_alpha
                wolfe_reductions = 0
                for wolfe_i in range(100):

                    # project() multiplies its second and third arguments, so
                    # this historical call order still computes x0 + alpha * p.
                    x1 = project(x0, alpha, p, lb, ub)
                    f1 = fun(x1)
                    wolfe1 = f1 < f0 + c1_wolfe * alpha * p.dot(g0)

                    if not wolfe1:
                        wolfe_reductions += 1
                        alpha /= wolfe_reduction_scale
                        continue
                    else:
                        break

                search_increase = 0
                if f1 < f0:
                    cnt_fail = 0
                    while True:
                        f2 = fun(project(x0, line_search_increase_scale * alpha, p, lb, ub))
                        if f2 < f1:
                            search_increase += 1
                            alpha *= line_search_increase_scale
                            f1 = f2
                        else:
                            break

                    x1 = project(x0, alpha, p, lb, ub)
                    g1 = grad_func(x1)

                    if has_hessian:
                        B0 = hessian_func(x1)
                    else:
                        try:
                            B0 = update_bfgs_hessian_approx(B0, g1, g0, x1, x0)
                        except:
                            pass

                    x0, f0, g0 = x1, f1, g1
                    accepted = True
                    last_alpha = alpha
                    if save_optimization_path:
                        optimization_path.append({'x': x0, 'fun': f0})

                else:

                    cnt_fail += 1
                    xerr, ferr, gerr = 1., 1., 1.
                    if not has_hessian:
                        B0 = np.eye(num_params) * min(10**num_params, 1_000_000)
                    accepted = False
                    last_alpha = 1.0
                    #
                    # print()
                    # print(itr_, cnt_fail, f0)
                    # if cnt_fail >= 2:
                    #     gtemp = g0.copy()
                    #     gtemp[~np.isfinite(gtemp)] = 0
                    #     max_grad_coord = np.argmax(np.abs(gtemp))
                    #     sign = -1.0 if g0[max_grad_coord] > 0 else 1.0
                    #     step = xtol
                    #     print(f'\t>max coord {max_grad_coord}, x0={x0[max_grad_coord]}, grad={g0[max_grad_coord]}, sign={sign}')
                    #
                    #     xnew, fnew = None, None
                    #     while True:
                    #         x1 = x0.copy()
                    #         x1[max_grad_coord] += sign * step
                    #         f1 = fun(x1)
                    #         if f1 < f0:
                    #             step *= 10
                    #             fnew, xnew = f1, x1
                    #         else:
                    #             break
                    #
                    #     print('\t\t', fnew, step)
                    #     if fnew is not None:
                    #         f0 = fnew
                    #         x0 = xnew
                    #         g0 = grad_func(xnew)

                            # if cnt_fail >= 2:
                            #     # Try coordinate descent
                            #
                            #     x1 = x0
                            #     f0_start_cd = f0
                            #     for k in range(num_params):
                            #         Ik = np.zeros(num_params)
                            #         Ik[k] = 1.0
                            #         gk = (fun(x1 + Ik*dx_fd)-f0)/dx_fd
                            #         if gk == 0:
                            #             continue
                            #         direction = -np.sign(gk)
                            #
                            #         max_step = np.inf
                            #         if is_bounded:
                            #             if direction > 0:
                            #                 max_step = ub[k] - x0[k]
                            #             if direction < 0:
                            #                 max_step = x0[k] - lb[k]
                            #
                            #         a = min(1e-10, max_step/10)
                            #         f1k = fun(x1 + a * direction * Ik)
                            #         scale_inc = 1.5
                            #         if f1k < f0:
                            #             while True:
                            #                 if a == max_step:
                            #                     break
                            #                 f2k = fun(x1 + min(max_step, scale_inc * a) * direction * Ik)
                            #                 if f2k < f1k:
                            #                     f0 = f1k
                            #                     a = min(max_step, scale_inc * a)
                            #                 else:
                            #                     break
                            #             x1 = x1 + direction * a * Ik
                            #
                            #     g1 = grad(fun, x1, onesided_fd, dx_fd)
                            #     y = g1 - g0
                            #     s = x1 - x0
                            #
                            #     try:
                            #         B0 = B0 + np.outer(y, y) / np.dot(y, s) - B0.dot(np.outer(s, s)).dot(B0.T) / (s.dot(B0).dot(s))
                            #     except:
                            #         pass
                            #
                            #     x0, g0 = x1, g1
                            #     print('cd', f0 - f0_begin_itr)

                p_last = x0 - x0_begin_itr
                xerr = np.max(np.abs(x0 - x0_begin_itr) / np.clip(1, np.abs(x0), np.inf))
                ferr = (f0_begin_itr - f0) / np.max([1, abs(f0), abs(f0_begin_itr)])
                if is_bounded:
                    # Projected-gradient norm: ignore gradient components that
                    # would only push farther outside an active bound.
                    gerr = np.max(np.abs(g0 * (((g0 > 0) & (x0 > lb)) | ((g0 < 0) & (ub > x0)))))
                else:
                    gerr = np.max(np.abs(g0))

                # itr_info = (
                #     'itr = %3d' % itr_, 'f0 = %.4e' % f0, 'alpha = %.2e' % alpha, 'ferr = %4.1e' % ferr,
                #     'xerr = %4.1e' % xerr, 'gerr = %4.1e' % gerr, 'acc = %5s' % accepted, f'fwd_fd = {onesided_fd}',
                #     '+step = %3d' % search_increase, '-step = %3d' % wolfe_reductions
                # )

                iter_info = [
                    {'name': 'iter', 'len': 6, 'format': '%6d', 'value': itr_},
                    {'name': 'accepted', 'len': 11, 'format': '%11s', 'value': accepted},
                    {'name': 'f0', 'len': 15, 'format': '%15.4e', 'value': f0},
                    {'name': 'alpha', 'len': 9, 'format': '%9.1e', 'value': alpha},
                    {'name': 'dF', 'len': 9, 'format': '%9.1e', 'value': f0 - f0_begin_itr},
                    {'name': 'ferr', 'len': 9, 'format': '%9.1e', 'value': ferr},
                    {'name': 'xerr', 'len': 9, 'format': '%9.1e', 'value': xerr},
                    {'name': 'gerr', 'len': 9, 'format': '%9.1e', 'value': gerr},
                    {'name': 'time', 'len': 10, 'format': '%9.2fs', 'value': time.time() - time0},
                ]

                if debug:

                    if itr_ == 0:
                        print_iter_info(iter_info, is_header=True)
                    print_iter_info(iter_info)

                    #
                    # if time.time() - last_pbar_update_time > pbar_update_cadence:
                    #     last_pbar_update_time = time.time()
                    #     #pbar.update()
                    #     #pbar.set_description(str(itr_info))

                itr_ += 1

                if accepted or (cnt_fail >= 4 and alpha < 1e-20):
                    if xerr < xtol:
                        message = f'Converged: xerr < xtol ({"%.1e" % xerr} < {"%.1e" % xtol})'
                        converged = True
                    if ferr < ftol:
                        message = f'Converged: ferr < ftol ({"%.1e" % ferr} < {"%.1e" % ftol})'
                        converged = True
                    if gerr < gtol:
                        message = f'Converged: gerr < gtol ({"%.1e" % gerr} < {"%.1e" % gtol})'
                        converged = True

                if converged:
                    break

                if itr_ == maxiter:
                    if user_prompt_for_more_iters:
                        newiter = user_prompt_for_more_iters_method(
                            f'Algorithm has not converged:'
                            f'\n\tCurrent gerr={"%.2e" % gerr}, xerr={"%.2e" % xerr}, ferr={"%.2e" % ferr}',
                            user_prompt_for_more_iters)
                        if newiter > 0:
                            # pbar.total += newiter
                            maxiter += newiter
                        else:
                            break
                    else:
                        break

            except KeyboardInterrupt:
                message = 'Did not converge: keyboard interrupt'
                print("\nProcess interrupted, breaking...\n")
                break

            except Exception as e:
                raise e

        if debug:
            #pbar.set_description(str(itr_info))
            print_iter_info(iter_info, is_footer=True)
            print(message)

        return BFGSPQNResults(**{
            'fun': -f0 if maximize else f0,
            'x': x0,
            'num_params': num_params,
            'grad': g0,
            'maximize': maximize,
            'grad_projected': g0 * (((g0 > 0) & (x0 > lb)) | ((g0 < 0) & (ub > x0))),
            'hess': B0,
            'converged': converged,
            'message': message,
            'time_elapsed': time.time() - time0,
            'fun_callable': fun_original,
            'ferr': ferr,
            'xerr': xerr,
            'gnorm': gerr,
            'iter': itr_,
            'B0': B0_orig,
            'ub_binding': (ub - x0 < 1e-12) if is_bounded else None,
            'lb_binding': (x0 - lb < 1e-12) if is_bounded else None,
            'bounds': np.array(bounds).copy() if bounds is not None else None,
            'options': dict(
                seed=seed, momentum=momentum, gtol=gtol, xtol=xtol, ftol=ftol,
                maxiter=maxiter, x0=x0_start, maximize=maximize,
                gradient_options=dict(onesided_fd=onesided_fd_orig, dx=dx_fd),
                c1_wolfe=c1_wolfe, c2_wolfe=c2_wolfe,
                wolfe_reduction_scale=wolfe_reduction_scale, line_search_increase_scale=line_search_increase_scale,
                save_optimization_path=save_optimization_path
            ),
            'optimization_path': optimization_path
        })



# if __name__ == '__main__':
#
#     import numpy as np
#     from scipy.optimize import minimize
#
#     np.random.seed(0)
#
#     n = 1_00
#     px = 50
#
#     sx = 1
#     X = np.random.randn(n, px).dot(sx * np.eye(px) + np.ones((px, px)) / px)
#     beta = np.exp(np.random.randn(px))
#
#     pz = 4
#     Z = np.random.randn(n, pz).dot(sx * np.eye(pz) + np.ones((pz, pz)) / pz)
#     gamma = np.exp(np.random.randn(pz))
#
#     y = (.5  + X.dot(beta)) * (1 + Z.dot(gamma)) + 1.5 + np.random.randn(n)
#
#
#     def objective_function(param):
#         return np.sum((y - (param[1] + X.dot(param[3:px + 3])) * (param[2] + Z.dot(param[px + 3:])) - param[0]) ** 2) / (2 * n)
#
#
#     result = bfgs_pqn(objective_function, np.ones(pz + px + 3), maxiter=1000, ftol=1e-14, xtol=1e-6, gtol=1e-6, debug=True,
#                       momentum=.25)
#     # print(result.to_string(['fun', 'iter', 'xerr', 'gerr', 'ferr', 'time_elapsed', 'x', 'grad']))
#     #
#     # print('\n' * 3)
#     # res_sp = minimize(objective_function, np.ones(pz + px + 3), tol=1e-12)
#     # print(res_sp)
#     #
#     # print(objective_function(result.x))
#     # print(objective_function(res_sp.x))
#     # print(objective_function(result.x) / objective_function(res_sp.x) - 1)
#
# if __name__ == '__main__':
#
#     def f(x):
#         return (1 - x[0]) ** 2 + 100. * (x[1] - x[0] ** 2) ** 2
#
#
#     res1 = bfgs_pqn(f, [-3040., -1486.], debug=True, momentum=0,
#                     xtol=1e-8, gtol=1e-8, ftol=1e-50, save_optimization_path=True,
#                     B0=100000)
#     res2 = bfgs_pqn(f, [-3040., -1486.], debug=True, momentum=.16,
#                     xtol=1e-8, gtol=1e-8, ftol=1e-50, save_optimization_path=True,
#                     B0=100000)
#
#     import matplotlib.pyplot as plt
#
#     f, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
#     for i, res in enumerate([res1, res2]):
#         ax[i].plot([r['x'][0] for r in res.optimization_path], [r['x'][1] for r in res.optimization_path], marker='.')
#     plt.show()


# if __name__ == '__main__':
#     def f(x):
#         return (x[0]) ** 2 + np.sqrt(-x[1])
#
#
#     print(bfgs_pqn(f, [0., -.001], debug=True, onesided_fd=True))
