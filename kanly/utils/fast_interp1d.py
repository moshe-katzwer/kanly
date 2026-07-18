from __future__ import absolute_import, print_function

from numba import njit


@njit(cache=True)
def fast_linear_interp1d(xval, x, y):
    """Assumes sorted `x`, and xval in the range"""
    n = len(x)
    l = 0
    r = n - 1
    while r - l > 1:
        m = (r + l) // 2
        if x[m] > xval:
            r = m
        else:
            l = m
    return y[l] + (y[r] - y[l]) / (x[r] - x[l]) * (xval - x[l])
