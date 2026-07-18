from __future__ import absolute_import, print_function

from kanly.optimize.optimization_results import OptimizationResult
import time

import numpy as np
from tqdm import tqdm

DEFAULT_CDB_MAXITER = 1000
DEFAULT_CDB_XTOL = 1e-6
DEFAULT_CDB_FTOL = 1e-8
DEFAULT_CDB_GTOL = 1e-4
DEFAULT_CDB_PBAR_UPDATE_CADENCE = .25
DEFAULT_CDB_DEBUG = False


def is_valid(x, lb, ub):
    """Check whether a point satisfies coordinate-wise bounds.

    Args:
        x: Parameter vector to validate.
        lb: Lower bounds with the same length as ``x``.
        ub: Upper bounds with the same length as ``x``.

    Returns:
        True when every coordinate lies between its lower and upper bounds.
    """
    return np.all(x >= lb) and np.all(x <= ub)


def _get_gradient(func, x0, idx, bounds):
    """Estimate a single partial derivative using bounded finite differences.

    Args:
        func: Objective callable accepting a parameter vector.
        x0: Parameter vector at which to evaluate the derivative.
        idx: Coordinate index for the partial derivative.
        bounds: Two-row array-like object containing lower and upper bounds.

    Returns:
        Finite-difference estimate of the derivative with respect to
        ``x0[idx]``.
    """
    dx = 1e-8 * max(1.0, abs(x0[idx]))
    xi = x0[idx]
    # Clip the stencil to the feasible interval so gradients remain defined
    # even when the current point sits on a bound.
    xi_l = max(bounds[0][idx], xi - dx)
    xi_h = min(bounds[1][idx], xi + dx)

    x0_l = x0.copy();
    x0_l[idx] = xi_l
    x0_h = x0.copy();
    x0_h[idx] = xi_h

    f_l = func(x0_l)
    f_h = func(x0_h)

    g_i = (f_h - f_l) / (xi_h - xi_l)
    return g_i


class CDBResults(OptimizationResult):
    """Result object returned by bounded coordinate descent."""

    def __init__(self, fun, x, num_params, grad, grad_projected, converged, message, time_elapsed, fun_callable, ferr,
                 xerr, iter, options, bounds, maximize, optimization_path):
        """Create a coordinate-descent result record.

        Args:
            fun: Final objective_function value in the user's original optimization
                direction.
            x: Final parameter vector.
            num_params: Number of optimized parameters.
            grad: Finite-difference gradient estimate at the final iterate.
            grad_projected: Gradient with bound-blocked coordinates zeroed.
            converged: Whether a stopping criterion was met.
            message: Human-readable convergence or failure message.
            time_elapsed: Wall-clock runtime in seconds.
            fun_callable: Original objective_function callable supplied by the user.
            ferr: Final relative objective_function-change diagnostic.
            xerr: Final relative parameter-change diagnostic.
            iter: Number of coordinate-descent sweeps completed.
            options: Solver options and starting values.
            bounds: Bounds used by the optimizer.
            maximize: Whether the user requested maximization.
            optimization_path: Optional optimization path; currently unused.
        """
        super().__init__(fun, x, num_params, grad, grad_projected, converged, message, time_elapsed, fun_callable,
                         ferr, xerr, iter, bounds, options, maximize, optimization_path)


def cdb(func, x0, bounds=None, maximize=False, maxiter=DEFAULT_CDB_MAXITER, xtol=DEFAULT_CDB_XTOL,
        ftol=DEFAULT_CDB_FTOL,
        gtol=DEFAULT_CDB_GTOL, debug=DEFAULT_CDB_DEBUG, pbar_update_cadence=DEFAULT_CDB_PBAR_UPDATE_CADENCE)\
        -> CDBResults:
    """Minimize or maximize an objective_function with bounded coordinate descent.

    The solver sweeps over active coordinates, follows the downhill coordinate
    direction, and expands successful steps until the objective_function stops
    improving. Bounds are enforced by clipping each coordinate step.

    Args:
        func: Objective callable accepting a one-dimensional parameter vector.
        x0: Initial feasible parameter vector.
        bounds: Optional two-row array-like bounds. The first row contains
            lower bounds and the second row contains upper bounds.
        maximize: If True, maximize ``func`` by minimizing ``-func``.
        maxiter: Maximum number of coordinate sweeps.
        xtol: Convergence tolerance for relative parameter changes.
        ftol: Convergence tolerance for relative objective_function changes.
        gtol: Convergence tolerance for the projected gradient norm.
        debug: Whether to show progress-bar diagnostics and final messages.
        pbar_update_cadence: Minimum seconds between progress-bar updates when
            ``debug`` is enabled.

    Returns:
        A ``CDBResults`` instance containing the final point, objective_function value,
        gradients, convergence diagnostics, and solver options.

    Examples
    --------
    Minimise a bounded quadratic with a box constraint:

    >>> import numpy as np
    >>> from kanly.api import cdb
    >>> def f(x):
    ...     return (x[0] - 2.0) ** 2 + (x[1] + 1.0) ** 2
    >>> bounds = np.array([[-5.0, -5.0],
    ...                    [ 1.5,  5.0]])     # cap x[0] at 1.5
    >>> res = cdb(f, x0=np.array([0.0, 0.0]), bounds=bounds)
    >>> res.x.round(3)                                # doctest: +SKIP
    array([ 1.5, -1.0])
    >>> res.converged                                  # doctest: +SKIP
    True

    Maximisation: pass ``maximize=True`` to flip the objective_function sign internally.
    """

    __t = time.time()
    last_pbar_update_time = 0

    x0 = np.asarray(x0, dtype=float)

    func_orig = func
    if maximize:
        func = lambda z: -func_orig(z)

    f0 = func(x0)
    x0_user = x0.copy()

    num_params = len(x0)

    if bounds is None:
        bounds = np.vstack([[-np.inf] * num_params, [np.inf] * num_params])

    assert is_valid(x0, bounds[0], bounds[1])
    assert np.isfinite(func(x0))

    I = np.eye(len(x0))

    pbar = tqdm(range(maxiter), disable=not debug, leave=False, position=0)

    converged = False
    message = 'Did not converge'

    g = np.zeros(num_params)
    g_projected = np.zeros(num_params)

    active_set = np.arange(num_params)
    for itr in pbar:
        try:

            successes = 0
            x0_start = x0.copy()
            f0_start = f0
            for i in active_set:

                g_i = _get_gradient(func, x0, i, bounds)
                sgn_g_i = np.sign(g_i)
                g[i] = g_i
                g_projected[i] = g_i

                if g_i > 0 and x0[i] == bounds[0][i]:
                    g_projected[i] = 0.0
                    continue
                elif g_i < 0 and x0[i] == bounds[1][i]:
                    g_projected[i] = 0.0
                    continue
                elif abs(g_i) < 1e-6:
                    continue

                alpha = 1.0

                # Reduce until we take a step down
                while alpha > 1e-14:
                    x1 = np.clip(x0 - sgn_g_i * alpha * I[i], bounds[0], bounds[1])
                    f1 = func(x1)

                    if f1 < f0:
                        break
                    else:
                        alpha /= 2

                # Increase while still stepping down
                while True:
                    x1 = np.clip(x0 - sgn_g_i * alpha * I[i], bounds[0], bounds[1])
                    f1 = func(x1)

                    if f1 < f0:
                        successes += 1
                        f0 = f1
                        x0 = x1
                        alpha *= 2
                    else:
                        break

                if not successes:
                    break

            x_err = np.max(np.abs(x0 - x0_start) / np.clip(np.abs(x0), 1, np.inf))
            f_err = abs(f0 - f0_start) / max(1, max(abs(f0), abs(f0_start)))
            # Projected-gradient norm treats bound-blocked coordinates as
            # already stationary for the constrained problem.
            g_err = max(np.abs(g_projected))

            info = dict(itr="%6d" % itr, f0="%8.4e" % f0, successes='%4d' % successes, x_err="%.2e" % x_err,
                        f_err="%.2e" % f_err, num_active='%5d' % len(active_set))
            if debug:

                if time.time() - last_pbar_update_time > pbar_update_cadence:
                    last_pbar_update_time = time.time()
                    pbar.set_description(str(info))

            if len(active_set) == num_params:
                if x_err < xtol:
                    converged = True
                    message = f'Converged: max(|| x(k+1) - x(k) ||) = {"%.2e" % x_err} < {"%.2e" % xtol}'
                    break
                if f_err < ftol:
                    converged = True
                    message = f'Converged: f(k+1) - f(k) = {"%.2e" % f_err} < {"%.2e" % ftol}'
                    break
                if g_err < gtol:
                    converged = True
                    message = f'Converged: max(||grad(k+1) - grad(k)||) = {"%.2e" % g_err} < {"%.2e" % gtol}'
                    break

            # Revisit only coordinates that moved on the last sweep. If none
            # moved meaningfully, reset to a full sweep so convergence checks
            # are based on every coordinate again.
            active_set = np.arange(num_params)[np.abs(x0 - x0_start) > max(1e-4, xtol)]
            if len(active_set) == 0:
                active_set = np.arange(num_params)

        except KeyboardInterrupt:
            message = 'Did not converge: keyboard interrupt'
            print("\nProcess interrupted, breaking...\n")
            break

        except Exception as e:
            raise e

    if debug:
        print(message)
        pbar.set_description(str(info))

    return CDBResults(**{
        'fun': -f0 if maximize else f0,
        'fun_callable': func_orig,
        'x': x0,
        'num_params': num_params,
        'grad': g,
        'grad_projected': g_projected,
        'converged': converged,
        'message': message,
        'time_elapsed': time.time() - __t,
        'ferr': f_err,
        'xerr': x_err,
        'iter': itr,
        'options': dict(
            gtol=gtol, xtol=xtol, ftol=ftol,
            maxiter=maxiter, x0=x0_user, maximize=maximize,
            bounds=bounds.copy()
        ),
        'bounds': bounds.copy() if bounds is not None else None,
        'maximize': maximize,
        'optimization_path': None,
    })

# if __name__ == '__main__':
#
#     def func(x):
#         return (-(x[0] - 5) ** 2 / 10) + (-(x[1] + 2) ** 2 / 2)
#
#     print(cdb(func, [0. ,0], maximize=True))
