"""Gaussian kernel smoothing for scattered (x, y) data.

Two computational backends are provided:

- **FFT-based** (:func:`gaussian_kernel_smooth_fft`): bins data onto a
  regular grid, convolves with a discretized Gaussian via FFT, and trims
  the padded boundaries.  Runs in O(N + G log G) time where G is the grid
  size (must be a power of two).
- **Direct** (:func:`gaussian_kernel_smooth_direct`): evaluates a weighted
  average at each grid point using the raw Gaussian weights.  Simpler but
  O(N × G).

The unified :func:`gaussian_kernel_smooth` entry point dispatches to
either backend and can return raw ``(grid, values)`` arrays or a fitted
:class:`~kanly.nonparametric.interpolate.Interpolator1d` spline object.
"""
from __future__ import absolute_import, print_function

import numpy as np
from scipy.fft import fft, ifft

from kanly.nonparametric.interpolate import interp, CUBIC


def gaussian_kernel_smooth_fft(x, y, bandwidth=None, weights=None, n_grid=128, pad_factor=.1, debug=False, adjust=1):
    """
    Smooth scattered (x, y) data using FFT-based Gaussian kernel regression.

    Parameters
    ----------
    x, y : 1D arrays
        Data points.
    bandwidth : float
        Gaussian bandwidth (in same units as x).
    n_grid : int
        Number of bins across the *original* data range.
    pad_factor : float
        How much of the range to pad the x-range with to pad on each side (default .1).
    weights : weights on the data points
    adjust : scalar on bandwidth

    Returns
    -------
    grid : ndarray
        Smoothed x grid (unpadded region).
    y_smooth : ndarray
        Smoothed curve values on grid.

    Examples
    --------
    FFT-based smoothing of a noisy quadratic relationship:

    >>> import numpy as np
    >>> from kanly.api import gaussian_kernel_smooth_fft
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=1000)
    >>> y = x**2 + rng.normal(size=1000)
    >>> xs, ys = gaussian_kernel_smooth_fft(x, y, n_grid=128)
    """

    # Default bandwidth: Silverman's rule-of-thumb
    if bandwidth is None:
        bandwidth = 1.069 * np.std(x) * len(x) ** (-.2)
    else:
        assert bandwidth > 0

    assert adjust > 0

    assert 0 <= pad_factor < .5

    # Scale bandwidth by the user-supplied adjustment factor
    bandwidth *= adjust

    # The FFT circular convolution requires the grid size to be a power of two
    # so that scipy.fft.fft can operate efficiently.
    if not (n_grid > 0 and (n_grid & (n_grid - 1)) == 0):
        raise Exception('`n_grid` must be power of two!')

    x = np.asarray(x)
    y = np.asarray(y)
    if weights is None:
        weights = np.ones_like(x)

    x_min, x_max = np.min(x), np.max(x)
    xrng = (x_max - x_min)
    dx = xrng / (n_grid - 1)

    # Compute how far to pad in x (in same units)
    # Padding prevents wrap-around FFT convolution artefacts at the edges.
    pad_width = pad_factor * xrng
    n_pad = int(np.ceil(pad_width / dx))
    if 2 * n_pad >= n_grid - 1:
        raise Exception('Bandwidth is too big for the amount of padding!  Decrease padding or bandwidth!')

    # Shrink the unpadded region and build a total padded grid of n_total bins
    n_grid = n_grid - 2 * n_pad
    n_total = n_grid + 2 * n_pad
    dx_new = (x_max - x_min) / (n_grid - 1)

    x_pad_min = x_min - n_pad * dx_new
    x_pad_max = x_max + n_pad * dx_new

    grid_full = np.linspace(x_pad_min, x_pad_max, n_total)

    if debug:
        print(f'{grid_full[n_pad-1]=}, {grid_full[-n_pad]=}')

    # Bin data into padded grid: weighted sum of y and plain count per bin
    y_sum, _ = np.histogram(x, bins=n_total, range=(x_pad_min, x_pad_max), weights=y * weights)
    counts, _ = np.histogram(x, bins=n_total, range=(x_pad_min, x_pad_max), weights=weights)

    # Build a discretized Gaussian kernel on the same grid spacing.
    # The kernel is centred at index 0 (via np.roll) so that the circular
    # FFT convolution corresponds to a standard linear convolution on the
    # interior of the grid.
    kernel_x = np.arange(-n_total // 2, n_total // 2) * dx
    kernel = np.exp(-0.5 * (kernel_x / bandwidth) ** 2)
    kernel /= kernel.sum()
    kernel = np.roll(kernel, -len(kernel) // 2)  # center at index 0

    #     # FFT convolution with linear padding
    #     L = 1 << int(np.ceil(np.log2(len(y_sum) + len(kernel) - 1)))

    #     def conv_fft(a, b):
    #         A = fft(a, L)
    #         B = fft(b, L)
    #         c = np.real(ifft(A * B))
    #         return c[:len(a)]

    #     num = conv_fft(y_sum, kernel)
    #     den = conv_fft(counts, kernel)

    # Convolve both the weighted y-sum and the count histogram with the kernel.
    # Dividing the convolved sums gives the Nadaraya-Watson estimate.
    fft_kernel = fft(kernel)
    num = np.real(ifft(fft(y_sum) * fft_kernel))
    den = np.real(ifft(fft(counts) * fft_kernel))

    # Safe division: bins with no data contribution stay at 0
    f_full = np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    # Trim back to unpadded region
    if debug:
        print(f'{n_pad=}')
    f_smooth = f_full[n_pad - 1:-(n_pad - 1)]
    grid = grid_full[n_pad - 1:-(n_pad - 1)]
    # f_smooth = f_full
    # grid = grid_full
    if debug:
        print(f'{len(f_smooth)=}, {len(grid)=}')

    return grid, f_smooth


def gaussian_kernel_smooth_direct(x, y, n_grid=128, weights=None, bandwidth=None, debug=False, adjust=1.0):
    """Smooth scattered (x, y) data using direct (non-FFT) Gaussian kernel regression.

    For each point on the output grid, computes a weighted average of ``y``
    where the weights are Gaussian kernel values centred at that grid point.
    This is the Nadaraya-Watson estimator evaluated naively in O(N × G) time.
    Prefer :func:`gaussian_kernel_smooth_fft` for large datasets.

    Args:
        x: 1-D array-like of observed x-coordinates.
        y: 1-D array-like of observed response values, same length as ``x``.
        n_grid: Number of evenly-spaced evaluation points across
            ``[min(x), max(x)]``.  Defaults to 128.
        weights: Optional 1-D array-like of observation weights, or a scalar
            weight applied to all observations.  Defaults to uniform weights.
        bandwidth: Gaussian kernel bandwidth in the same units as ``x``.
            If ``None``, Silverman's rule-of-thumb
            ``1.069 * std(x) * n^{-0.2}`` is used.
        debug: Currently unused; reserved for future diagnostic output.
        adjust: Multiplicative scalar applied to the resolved bandwidth before
            fitting.  Values greater than 1 produce more smoothing.

    Returns:
        Tuple ``(grid, fitted)`` where ``grid`` is the evaluation x-coordinates
        (length ``n_grid``) and ``fitted`` is the smoothed y-values.

    Examples
    --------
    Direct (non-FFT) Gaussian smoothing — useful for irregular grids:

    >>> import numpy as np
    >>> from kanly.api import gaussian_kernel_smooth_direct
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(0, 10, size=200)
    >>> y = np.sin(x) + 0.3 * rng.normal(size=200)
    >>> xs, ys = gaussian_kernel_smooth_direct(x, y, n_grid=64,
    ...                                        bandwidth=0.5)
    """
    if weights is None:
        weights = 1.0
    if bandwidth is None:
        bandwidth = 1.069 * len(x) ** -.2 * np.std(x)
    bandwidth *= adjust
    grid = np.linspace(min(x), max(x), n_grid)
    fitted = np.array([
        np.average(y, weights=weights * np.exp(-(x0 - x) ** 2 / (2 * bandwidth ** 2)))
        for x0 in grid
    ])
    return grid, fitted


def gaussian_kernel_smooth(x, y, do_fft=True, bandwidth=None, weights=None, n_grid=128, pad_factor=.1, debug=False,
                           return_arrays=True, kind=CUBIC, adjust=1.0):
    """Smooth scattered (x, y) data using Gaussian kernel regression.

    Unified entry point that dispatches to either the FFT-based or direct
    convolution backend and optionally wraps the result in an interpolating
    spline object.

    Args:
        x: 1-D array-like of observed x-coordinates.
        y: 1-D array-like of observed response values, same length as ``x``.
        do_fft: If ``True`` (default), use the FFT-accelerated backend
            :func:`gaussian_kernel_smooth_fft`; otherwise use the direct
            backend :func:`gaussian_kernel_smooth_direct`.
        bandwidth: Gaussian kernel bandwidth in the same units as ``x``.
            If ``None``, Silverman's rule-of-thumb is used by whichever
            backend is active.
        weights: Optional 1-D array-like of observation weights.  Defaults to
            uniform weights.
        n_grid: Number of evaluation grid points (must be a power of two when
            ``do_fft=True``).  Defaults to 128.
        pad_factor: Fraction of the data range added as padding on each side
            in the FFT backend.  Prevents circular wrap-around artefacts.
            Must satisfy ``0 <= pad_factor < 0.5``.  Ignored when
            ``do_fft=False``.
        debug: If ``True``, print diagnostic information during FFT
            computation.
        return_arrays: If ``True`` (default), return ``(grid, y_smooth)``
            NumPy arrays.  If ``False``, fit a spline interpolator to the
            smoothed output and return that callable instead.
        kind: Spline kind (e.g. ``'cubic'``) used to build the interpolator
            when ``return_arrays=False``.  Ignored when ``return_arrays=True``.
        adjust: Multiplicative scalar applied to the resolved bandwidth.
            Values greater than 1 produce more smoothing.

    Returns:
        When ``return_arrays=True``: tuple ``(grid, y_smooth)`` where ``grid``
        contains the evaluation x-coordinates and ``y_smooth`` the smoothed
        values.

        When ``return_arrays=False``: an
        :class:`~kanly.nonparametric.interpolate.Interpolator1d` spline fitted
        to ``(grid, y_smooth)``.

    Examples
    --------
    Smooth a noisy non-linear relationship and obtain arrays:

    >>> import numpy as np
    >>> from kanly.api import gaussian_kernel_smooth
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(0, 10, size=500)
    >>> y = np.sin(x) + 0.3 * rng.normal(size=500)
    >>> xs, ys = gaussian_kernel_smooth(x, y, n_grid=128)

    Obtain a callable spline approximator instead:

    >>> spline = gaussian_kernel_smooth(x, y, return_arrays=False)
    >>> spline(5.0).round(2)                          # doctest: +SKIP
    -0.96
    """

    if do_fft:
        xsmooth, ysmooth = gaussian_kernel_smooth_fft(x, y, bandwidth=bandwidth, weights=weights, n_grid=n_grid,
                                                      pad_factor=pad_factor, debug=debug, adjust=adjust)
    else:
        xsmooth, ysmooth = gaussian_kernel_smooth_direct(x, y, bandwidth=bandwidth, weights=weights, n_grid=n_grid,
                                                         debug=debug, adjust=adjust)
    if return_arrays:
        return xsmooth, ysmooth
    else:
        return interp(xsmooth, ysmooth, kind=kind, assume_sorted=True)


# if __name__ == '__main__':
#
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from kanly.nonparametric.lowess import LOWESS
#     n = 1000
#     x = np.random.randn(n)
#     y = x**2 + np.random.rand(n)
#     plt.scatter(x,y, alpha=.4)
#     plt.plot(*gaussian_kernel_smooth(x, y, adjust=.2), c='k')
#     plt.plot(*LOWESS(y, x, return_arrays=True), c='r')
#     plt.show()