"""Numerical helpers used by Bayesian optimization/sampling routines."""

from __future__ import absolute_import, print_function

import numpy as np
from tqdm import tqdm


def get_neg_inverse_observed_information(lp, x, dx=1e-4, diag_only=True, bounds=None, debug=False):
    """Approximate the negative inverse observed information matrix at ``x``.

    Uses finite differences on the provided log-density callable ``lp``.

    Args:
        lp: Log-density callable ``lp(x) -> float``.
        x: Evaluation point.
        dx: Relative finite-difference step-size baseline.
        diag_only: If True, estimate only diagonal curvature sparse_terms.
        bounds: Optional 2-by-p array-like of lower/upper bounds.
        debug: If True, show progress bars for second-derivative loops.
    """
    p = len(x)
    lp0 = lp(x)
    assert np.isfinite(lp0) and not np.isnan(lp0)

    if bounds is not None:
        lb = bounds[0]
        ub = bounds[1]
        dist_2_lower = x - lb
        dist_2_upper = ub - x

    dx_k = np.zeros(p)
    for i in range(p):
        dx_k[i] = dx * max(abs(x[i]), 1.0)
        if bounds is not None:
            if dist_2_lower > dx_k[i] or dist_2_upper > dx_k[i]:
                dx_k[i] = min(dist_2_lower[i], dist_2_upper[i])
            if dist_2_lower[i] + dist_2_upper[i] < 2 * dx_k[i]:
                raise Exception
            else:
                xi_l = max(lb[i], x[i] - dx_k[i])
                xi_h = min(ub[i], x[i] + dx_k[i])
                x[i] = (xi_l + xi_h) / 2

    lp0 = lp(x)
    assert np.isfinite(lp0) and not np.isnan(lp0)

    f_plus = np.zeros(p)
    f_minus = np.zeros(p)
    for i in range(p):
        xi = x.copy();
        xi[i] += dx_k[i]
        f_plus[i] = lp(xi)

        xi = x.copy();
        xi[i] -= dx_k[i]
        f_minus[i] = lp(xi)

    d2_lp_dx2 = np.zeros((p, p))
    for ij in tqdm(range(p ** 2), disable=not debug, leave=False):
        i = ij // p
        j = ij % p
        if i == j:
            d2_lp_dx2[i, j] = (f_plus[i] - 2 * lp0 + f_minus[i]) / (dx_k[i] ** 2)
        else:
            if not diag_only:
                if i < j:
                    x1 = x.copy();
                    x1[i] += dx_k[i];
                    x1[j] += dx_k[j]
                    x4 = x.copy();
                    x4[i] -= dx_k[i];
                    x4[j] -= dx_k[j]
                    l_plus = lp(x1)
                    l_minus = lp(x4)

                    d2_lp_dx2[i, j] = (
                                              l_plus
                                              - f_plus[i]
                                              - f_plus[j]
                                              + 2 * lp0
                                              - f_minus[i]
                                              - f_minus[j]
                                              + l_minus
                                      ) / (2 * dx_k[i] * dx_k[j])
                else:
                    d2_lp_dx2[i, j] = d2_lp_dx2[j, i]

    return -np.linalg.pinv(d2_lp_dx2)


def step_back_from_bounds(x0, V0, bounds, step_back=.01):
    """Move an initial point slightly away from hard bounds using proposal scale.

    Args:
        x0: Candidate initial parameter vector.
        V0: Proposal covariance used to scale the step-back distance.
        bounds: 2-by-p lower/upper bound array-like.
        step_back: Multiplier on per-parameter proposal standard deviation.
    """
    x0 = np.array(x0).copy()
    for i in range(len(x0)):
        step_back_i = step_back * V0[i, i] ** .5
        if x0[i] - bounds[0][i] < step_back_i:
            x0[i] += step_back_i
            if x0[i] > bounds[1][i] - step_back_i:
                raise Exception(f"Proposal Distribution too wide for param {i} given bounds")
        elif bounds[1][i] - x0[i] < step_back_i:
            x0[i] -= step_back_i
            if x0[i] < bounds[0][i] + step_back_i:
                raise Exception(f"Proposal Distribution too wide for param {i} given bounds")
    return x0
