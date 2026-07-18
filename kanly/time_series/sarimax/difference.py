from __future__ import absolute_import, print_function

from numba import njit


@njit(cache=True)
def difference(y, d=0, D=0, s=0):
    """
    d = difference
    D = seasonal difference
    s = seasonality periods

    Args:
        y: One- or two-dimensional time-series array to difference along the
            observation axis.
        d: Number of ordinary first differences to apply.
        D: Number of seasonal differences to apply.
        s: Seasonal period. Must be at least 2 when ``D`` is nonzero.

    Returns:
        Copy of ``y`` after applying ordinary differences first and seasonal
        differences second. The output is shorter by ``d + D * s`` rows.
    """
    if d == D == 0:
        return y.copy()
    for _ in range(d):
        y = y[1:] - y[:-1]
    if D:
        assert s >= 2
        for _ in range(D):
            y = y[s:] - y[:-s]
    return y
