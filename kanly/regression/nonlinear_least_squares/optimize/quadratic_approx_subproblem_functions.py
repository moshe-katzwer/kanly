from __future__ import absolute_import, print_function

import numpy as np
from numpy.linalg import norm


def steps_to_bound(x0, p, lb, ub):
    """Compute per-parameter step lengths until a direction hits a bound.

    Args:
        x0: Current parameter vector.
        p: Proposed direction vector.
        lb: Lower bounds for each parameter.
        ub: Upper bounds for each parameter.

    Returns:
        Array of step multipliers; ``np.inf`` where the direction never hits a
        finite bound.
    """
    steps = np.ones(x0.shape) * np.inf
    for j in range(len(x0)):
        if p[j] > 0 and ub[j] < np.inf:
            steps[j] = (ub[j] - x0[j]) / p[j]
        elif p[j] < 0 and lb[j] > -np.inf:
            steps[j] = (lb[j] - x0[j]) / p[j]
    return steps


def get_max_step_to_bound(x0, p, lb, ub):
    """Return the first positive step length at which ``x0 + step * p`` hits a bound.

    Args:
        x0: Current parameter vector.
        p: Proposed direction vector.
        lb: Lower bounds.
        ub: Upper bounds.

    Returns:
        Smallest bound-hitting step multiplier.
    """
    return min(steps_to_bound(x0, p, lb, ub))


def cauchy_point(B, g, Delta):
    """Compute the Cauchy point for a trust-region quadratic model.

    Minimises the quadratic approximation along the steepest-descent direction
    ``-g`` subject to the trust-region radius ``Delta``.

    Args:
        B: Quadratic-model Hessian approximation.
        g: Quadratic-model gradient vector.
        Delta: Trust-region radius.

    Returns:
        Step vector along ``-g`` with norm no larger than ``Delta``.
    """
    a = 0.5 * np.dot(g, B).dot(g)
    b = -np.dot(g, g)
    c = 0.0
    lb = 0.0
    ub = Delta / norm(g)
    tau, _ = min_1d_quadratic_function(a, b, c, lb, ub)
    return -g * tau


def min_along_direction(d, B, g, Delta):
    """Minimise the quadratic model along a single direction through the origin.

    Args:
        d: Direction vector.
        B: Quadratic-model Hessian approximation.
        g: Quadratic-model gradient vector.
        Delta: Trust-region radius.

    Returns:
        Best trust-region-feasible step of the form ``tau * d``.
    """
    a = 0.5 * np.dot(d, B).dot(d)
    b = g.dot(d)
    c = 0

    lb, ub = -Delta / norm(d), Delta / norm(d)

    tau, _ = min_1d_quadratic_function(a, b, c, lb, ub)
    return tau * d


def min_along_direction_with_start(p, d, B, g, Delta):
    """Minimise the quadratic model along a direction starting from ``p``.

    The feasible interval for ``tau`` is determined by the trust-region
    boundary ``||p + tau*d|| <= Delta``.

    Args:
        p: Current partial step.
        d: Search direction.
        B: Quadratic-model Hessian approximation.
        g: Quadratic-model gradient vector.
        Delta: Trust-region radius.

    Returns:
        Step ``p + tau*d`` that minimises the quadratic over the feasible line
        segment.
    """
    a = 0.5 * np.dot(d, B).dot(d)
    b = (g + np.dot(B, p)).dot(d)
    c = np.dot(g, p) + 0.5 * np.dot(p, B).dot(p)

    a_tilde = norm(d) ** 2
    b_tilde = 2 * np.dot(p, d)
    c_tilde = norm(p) ** 2 - Delta ** 2

    lb = (-b_tilde - np.sqrt(b_tilde ** 2 - 4 * a_tilde * c_tilde)) / (2 * a_tilde)
    ub = (-b_tilde + np.sqrt(b_tilde ** 2 - 4 * a_tilde * c_tilde)) / (2 * a_tilde)

    tau, _ = min_1d_quadratic_function(a, b, c, lb, ub)
    return p + tau * d


def _scale(x, weights):
    """Undo square-root weighting for a vector when weights are supplied.

    Args:
        x: Vector to scale.
        weights: Optional observation weights.

    Returns:
        ``x`` unchanged when ``weights is None``; otherwise ``x / sqrt(weights)``.
    """
    if weights is None:
        return x
    else:
        return x / np.sqrt(weights)


def steihaug(g, B, Delta, eps=1e-6, maxiter=None):
    """Solve the trust-region subproblem by Steihaug conjugate gradients.

    Args:
        g: Quadratic-model gradient vector.
        B: Quadratic-model Hessian approximation.
        Delta: Trust-region radius.
        eps: Residual norm tolerance for terminating the CG iterations.
        maxiter: Optional maximum number of CG iterations; defaults to
            ``len(g)``.

    Returns:
        Tuple ``(p, cond, iterations)`` where ``p`` is the trust-region step,
        ``cond`` encodes the termination case, and ``iterations`` is the final
        CG iteration index.
    """
    if maxiter is None:
        maxiter = len(g)

    p = g * 0.0
    r0 = g.copy()
    d0 = -g

    if norm(r0) < eps:
        return g * 0.0, -1, -1

    for j in range(maxiter):

        disc = np.dot(d0, B).dot(d0)

        if disc <= 0:
            return min_along_direction_with_start(p, d0, B, g, Delta), 0, j  # pos=False?

        alpha = np.dot(r0, r0) / disc
        p_new = p + alpha * d0

        if norm(p_new) > Delta:
            return min_along_direction_with_start(p, d0, B, g, Delta), 1, j  # pos=True

        r1 = r0 + alpha * np.dot(B, d0)
        if norm(r1) < eps:
            return p_new, 2, j

        p = p_new
        beta = np.dot(r1, r1) / np.dot(r0, r0)
        d0 = -r1 + beta * d0
        r0 = r1

    return p, 3, j


def min_1d_quadratic_function(a, b, c=0.0, lb=np.inf, ub=np.inf):
    """
    ax^2 + bx + {c}, constant doesn't
    matter for x_star but does for fval

    Args:
        a: Quadratic coefficient.
        b: Linear coefficient.
        c: Constant coefficient.
        lb: Lower bound for the scalar minimiser.
        ub: Upper bound for the scalar minimiser.

    Returns:
        Tuple ``(x_star, fval)`` where ``x_star`` minimises the bounded
        quadratic and ``fval`` is the objective_function value at ``x_star``.
    """
    if a == 0:
        if b < 0:
            x_star = ub
        else:
            x_star = lb
    else:
        x_star = -b / (2.0 * a)
        if a > 0:
            if lb <= x_star <= ub:
                pass
            elif x_star < lb:
                x_star = lb
            else:
                x_star = ub
        else:
            bnds = np.array([lb, ub])
            x_star = bnds[np.argmin([a * x ** 2 + b * x for x in bnds])]

    fval = x_star * (a * x_star + b) + c
    return x_star, fval


def intersect_direction_with_boundary(p, d, sign, Delta):
    """Find where the ray ``p + tau*d`` intersects the trust-region boundary.

    Args:
        p: Starting point inside the trust region.
        d: Direction vector.
        sign: Root selector, typically ``-1`` or ``1``.
        Delta: Trust-region radius.

    Returns:
        Scalar ``tau`` satisfying ``||p + tau*d|| == Delta``.
    """
    a = np.dot(d, d)
    b = 2.0 * np.dot(d, p)
    c = np.dot(p, p) - Delta ** 2
    return (-b - sign * np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)


def quadratic_form(_x, _g, _B):
    """Evaluate ``g'x + 0.5*x'Bx`` for a quadratic approximation.

    Args:
        _x: Step vector.
        _g: Gradient vector.
        _B: Hessian approximation matrix.

    Returns:
        Scalar quadratic-model value.
    """
    return np.dot(_x, _g) + 0.5 * np.dot(_x, _B).dot(_x)


def newton(B, g, Delta):
    """Compute a Newton step clipped to the trust-region radius.

    Args:
        B: Hessian approximation matrix.
        g: Gradient vector.
        Delta: Trust-region radius.

    Returns:
        Newton step ``solve(B, -g)`` scaled down if its norm exceeds ``Delta``.
    """
    p_newton_scaled = np.linalg.solve(B, -g)
    p_newton_scaled *= min(1.0, Delta / norm(p_newton_scaled))
    return p_newton_scaled
