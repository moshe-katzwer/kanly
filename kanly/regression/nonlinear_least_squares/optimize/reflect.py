from __future__ import absolute_import, print_function

import numpy as np

from kanly.regression.nonlinear_least_squares.optimize.quadratic_approx_subproblem_functions import \
    intersect_direction_with_boundary, steps_to_bound


def get_reflect_path(x0, p0, lb, ub, root_v, Delta, num_reflections):
    """Construct a reflected path for bounded trust-region steps.

    Starting at ``x0`` and moving in direction ``p0``, the path reflects off
    active lower/upper bounds and stops when it reaches the scaled
    trust-region boundary or the maximum number of reflections.  The returned
    path is used to search for a feasible step when the unconstrained
    Steihaug/Newton direction would leave the box constraints.

    Args:
        x0: Current parameter vector.
        p0: Initial proposed step direction in original parameter units.
        lb: Lower bounds.
        ub: Upper bounds.
        root_v: Scaling vector for the bounded trust-region metric.
        Delta: Trust-region radius in scaled coordinates.
        num_reflections: Maximum number of bound reflections to trace.

    Returns:
        Tuple ``(b, f_param)`` where ``b`` is an array of path breakpoints and
        ``f_param(alpha)`` maps ``alpha in [0, 1]`` to a point on the broken
        reflected path.
    """
    root_v = np.clip(root_v, a_min=1e-4, a_max=np.inf)
    x0 = np.asarray(x0)
    p0 = np.asarray(p0)
    lb = np.asarray(lb)
    ub = np.asarray(ub)
    p = p0.copy()

    b = [np.asarray(x0).copy()]
    beta = 0
    for i in range(num_reflections):
        BR = steps_to_bound(b[-1], p, lb, ub)
        beta_new = np.min(BR[BR > 0])
        b_new = b[-1] + (beta_new - beta) * p
        b_new = np.clip(b_new, a_min=lb, a_max=ub)
        b.append(b_new)
        p_new = p.copy()
        hits = (b_new == lb) | (b_new == ub)
        p_new[hits] *= -1

        if np.linalg.norm((b[-1] - x0) / root_v) > Delta:
            tau = intersect_direction_with_boundary((b[-2] - x0) / root_v, p / root_v, -1, Delta)
            b[-1] = b[-2] + tau * p
            break

        p = p_new

    b = np.asarray(b)

    def f_param(alpha):
        """Evaluate the piecewise-linear reflected path at a normalised location.

        Args:
            alpha: Scalar in ``[0, 1]`` selecting a position along the broken
                path.

        Returns:
            Parameter vector on the reflected path.
        """
        k = min(int(alpha * (len(b) - 1)), len(b) - 2)
        # todo this min thing I added, not sure what this does, before was just
        # `k=int(alpha * (len(b) - 1))`, need to investigate
        return b[k] + np.remainder(alpha, 1.0 / (len(b) - 1)) * (
                b[k + 1] - b[k]) / (
                1.0 / (len(b) - 1))

    return b, f_param
