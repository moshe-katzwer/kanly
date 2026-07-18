"""A very fast version of np.histogram"""
from __future__ import absolute_import, print_function

import numpy as np
from numba import njit


def fast_histogram(data, lower, upper, nbins, density=False):
    """Compute a histogram over a fixed range using a Numba-JIT inner loop.

    A thin public wrapper around ``_fast_histogram`` that coerces ``data`` to a
    contiguous float64 array before dispatching to the compiled function.

    Args:
        data: Array-like of numeric values to bin.
        lower: Left edge of the histogram range (inclusive).
        upper: Right edge of the histogram range (inclusive).
        nbins: Number of equal-width bins.  Must be >= 1.
        density: When ``True``, normalise the counts so that the integral over
            the range equals 1 (i.e. divide by ``total_count / nbins_per_unit``).

    Returns:
        NumPy float64 array of length ``nbins`` containing bin counts (or
        density values when ``density=True``).
    """
    return _fast_histogram(np.asarray(data, dtype=np.float64), lower, upper, nbins, density=density)


@njit(cache=True)
def _fast_histogram(data, lower, upper, nbins, density=False):
    """Assumes all data between lower, upper"""
    assert nbins >= 1
    cnt = np.zeros(nbins)
    c = nbins / (upper - lower)
    for x in data:
        cnt[max(0, min(int(c * (x - lower)), nbins - 1))] += 1
    if density:
        cnt /= (cnt.sum() / c)
    return cnt
