"""
Computes an interval [a,b] on a sample of data points x
that solves

    min {over a,b}      b - a
    s.t.                F(b) - F(a) = coverage

where F is the empirical cdf of x, and coverage in [0,1]
is a parameter.

Basically the same as highest posterior density: see
https://en.wikipedia.org/wiki/Credible_interval#Choosing_a_credible_interval
"""
from __future__ import absolute_import, print_function

import numpy as np
from scipy.interpolate import interp1d


def get_highest_density_interval(data, level, num_cdf_interp_points=1_001, num_lower_bound_search_grid_points=1_001,
                                 debug=False):
    """Compute the Highest Density Interval (HDI) for a sample at a given coverage level.

    Finds the shortest interval ``[lb, ub]`` over the empirical distribution of
    ``data`` such that ``F(ub) - F(lb) = level``, where ``F`` is the empirical
    CDF.  The search is performed on a grid of candidate lower bounds; for each
    candidate the corresponding upper bound is read off the CDF interpolant.

    Args:
        data: 1-D array-like of scalar sample values.
        level: Coverage probability in (0, 1), e.g. ``0.95`` for a 95 % HDI.
        num_cdf_interp_points: Number of quantile points used to build the CDF
            and PPF interpolants.  More points improve accuracy at the cost of
            memory.
        num_lower_bound_search_grid_points: Number of candidate lower-bound
            values searched over ``[min(data), ppf(1-level)]``.  Finer grids
            yield more accurate HDI boundaries.
        debug: Unused; reserved for future diagnostic output.

    Returns:
        Tuple ``(lb, ub)`` of floats representing the lower and upper bounds of
        the HDI.

    Examples
    --------
    95 % HDI on a sample from an asymmetric (gamma) distribution:

    >>> import numpy as np
    >>> from kanly.api import get_highest_density_interval
    >>> rng = np.random.default_rng(0)
    >>> draws = rng.gamma(shape=2.0, scale=1.0, size=10_000)
    >>> lb, ub = get_highest_density_interval(draws, level=0.95)
    >>> round(lb, 2), round(ub, 2)                          # doctest: +SKIP
    (0.04, 5.32)

    For a symmetric posterior the HDI coincides with the equal-tail
    credible interval; for skewed posteriors the HDI is generally tighter
    on the long-tail side.

    Useful for MCMC posterior summaries produced by :func:`amha` or
    :func:`mala`.
    """
    quantiles = np.linspace(.0001, .9999, num_cdf_interp_points)
    quant_vals = np.quantile(data, quantiles)

    quantiles = np.hstack([0, quantiles, 1])
    quant_vals = np.hstack([min(data) - 1e-8, quant_vals, max(data) + 1e-8])

    cdf = interp1d(quant_vals, quantiles)
    ppf = interp1d(quantiles, quant_vals)

    def get_ub(_lb, _level):
        """Return the upper bound corresponding to a given lower bound and coverage level.

        Evaluates ``ppf(cdf(lb) + level)``; returns ``np.inf`` when the
        required upper-tail probability exceeds 1 (i.e. the lower bound is too
        close to the right edge of the distribution).

        Args:
            _lb: Candidate lower bound value.
            _level: Coverage probability (same as the outer ``level``).

        Returns:
            Upper bound float, or ``np.inf`` if out of range.
        """
        right_cdf = cdf(_lb) + _level
        if right_cdf >= 1:
            return np.inf
        else:
            return ppf(right_cdf)

    lbs = np.linspace(quant_vals[0], ppf(1 - level), num_lower_bound_search_grid_points)
    ubs = np.array([get_ub(l, level) for l in lbs])
    best_idx = np.argmin(ubs - lbs)
    lb_best = lbs[best_idx]
    ub_best = ppf(min(cdf(lb_best) + level, 1))

    return lb_best, float(ub_best)
