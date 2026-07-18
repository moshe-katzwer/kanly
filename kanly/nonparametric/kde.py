"""Kernel Density Estimation (KDE) for one-dimensional data.

Provides :func:`kde`, the main entry point, which estimates a continuous
probability density from a sample using a variety of kernels and two
computational backends:

- **FFT path** (``fft=True``): bins data into a histogram, evaluates the
  Fourier-transformed kernel analytically (Gaussian) or numerically, and
  uses ``ifft`` for O(G log G) convolution.  Following
  Silverman (1982) *Appl. Statist.* **31**, 93–99.
- **Direct path** (``fft=False``): evaluates the kernel sum at each grid
  point in O(N × G) time.

Supported kernels (identified by short string names):

- ``'gau'`` – Gaussian
- ``'epa'`` – Epanechnikov
- ``'uni'`` – Uniform
- ``'tri'`` – Triangular
- ``'biw'`` – Biweight
- ``'triw'`` – Triweight
- ``'cos'`` – Cosine
- ``'tric'`` – Tricube

Results can be returned as ``(support, density)`` arrays or as a callable
:class:`KDEObject` that interpolates the density at arbitrary points.
Optional boundary-reflection clipping is available via :func:`clip_kde`.
"""
from __future__ import absolute_import, print_function

import random

import numpy as np
from numpy.fft import fft as fft_func, ifft, fftshift

from kanly.dill_object import DillObject
from kanly.nonparametric.interpolate import LINEAR, interp
from kanly.utils.fast_histogram import fast_histogram

TRIANGULAR = 'tri'
TRIWEIGHT = 'triw'
GAUSSIAN = 'gau'
UNIFORM = 'uni'
COSINE = 'cos'
BIWEIGHT = 'biw'
TRICUBE = 'tric'
EPANECHNIKOV = 'epa'

KERNEL_NAMES = {TRICUBE, TRIWEIGHT, TRIANGULAR, GAUSSIAN,
                COSINE, UNIFORM, BIWEIGHT, EPANECHNIKOV}

REFERENCE_CONSTANTS = {
    GAUSSIAN: 1.0592238410488122,
    EPANECHNIKOV: 2.344914356323711,
    UNIFORM: 1.8431099195302554,
    TRIANGULAR: 2.5760303892892913,
    BIWEIGHT: 2.777936682192678,
    TRIWEIGHT: 3.154480796737293,
    COSINE: 2.409709532853482,
    TRICUBE: 2.764345005992905
}

DEFAULT_KDE_KERNEL = GAUSSIAN
DEFAULT_KDE_FFT = True
DEFAULT_KDE_BW = 'normal_reference'
DEFAULT_KDE_GRIDSIZE = 256
DEFAULT_KDE_CUT = 4.
DEFAULY_KDE_ADJUST = 1.


def kernel_epanechnikov(Y, y, bw):
    """
    Evaluate the Epanechnikov kernel centered at ``y``.

    Parameters
    ----------
    Y : array-like
        Points at which to evaluate the kernel.
    y : float
        Kernel center.
    bw : float
        Bandwidth parameter.

    Returns
    -------
    numpy.ndarray
        Kernel values evaluated at ``Y``.
    """
    u = np.abs((Y - y) / bw)
    return np.where(u < 1, .75 / bw * (1 - u ** 2), 0.0)


def kernel_gaussian(Y, y, bw):
    """
    Evaluate the Gaussian kernel centered at ``y``.

    Parameters
    ----------
    Y : array-like
        Points at which to evaluate the kernel.
    y : float
        Kernel center.
    bw : float
        Bandwidth parameter.

    Returns
    -------
    numpy.ndarray
        Kernel values evaluated at ``Y``.
    """
    u = (Y - y) / bw
    return np.exp(-u ** 2 / 2) / (np.sqrt(2 * np.pi) * bw)


def kernel_uniform(Y, y, bw):
    """
    Evaluate the uniform kernel centered at ``y``.

    Parameters
    ----------
    Y : array-like
        Points at which to evaluate the kernel.
    y : float
        Kernel center.
    bw : float
        Bandwidth parameter.

    Returns
    -------
    numpy.ndarray
        Kernel values evaluated at ``Y``.
    """
    u = np.abs((Y - y) / bw)
    return np.where(u < 1, .5 / bw, 0.0)


def kernel_triweight(Y, y, bw):
    """
    Evaluate the triweight kernel centered at ``y``.

    Parameters
    ----------
    Y : array-like
        Points at which to evaluate the kernel.
    y : float
        Kernel center.
    bw : float
        Bandwidth parameter.

    Returns
    -------
    numpy.ndarray
        Kernel values evaluated at ``Y``.
    """
    u = np.abs((Y - y) / bw)
    return np.where(u < 1, 35. / (32. * bw) * (1 - u ** 2) ** 3, 0.0)


def kernel_biweight(Y, y, bw):
    """
    Evaluate the biweight kernel centered at ``y``.

    Parameters
    ----------
    Y : array-like
        Points at which to evaluate the kernel.
    y : float
        Kernel center.
    bw : float
        Bandwidth parameter.

    Returns
    -------
    numpy.ndarray
        Kernel values evaluated at ``Y``.
    """
    u = np.abs((Y - y) / bw)
    return np.where(u < 1, 15. / (16. * bw) * (1 - u ** 2) ** 2, 0.0)


def kernel_cosine(Y, y, bw):
    """
    Evaluate the cosine kernel centered at ``y``.

    Parameters
    ----------
    Y : array-like
        Points at which to evaluate the kernel.
    y : float
        Kernel center.
    bw : float
        Bandwidth parameter.

    Returns
    -------
    numpy.ndarray
        Kernel values evaluated at ``Y``.
    """
    u = np.abs((Y - y) / bw)
    return np.where(u < 1, np.pi / 4 * np.cos(np.pi / 2 * u), 0.0)


def kernel_triangular(Y, y, bw):
    """
    Evaluate the triangular kernel centered at ``y``.

    Parameters
    ----------
    Y : array-like
        Points at which to evaluate the kernel.
    y : float
        Kernel center.
    bw : float
        Bandwidth parameter.

    Returns
    -------
    numpy.ndarray
        Kernel values evaluated at ``Y``.
    """
    u = np.abs((Y - y) / bw)
    return np.where(u < 1, 1 - u, 0.0)


def kernel_tricube(Y, y, bw):
    """
    Evaluate the tricube kernel centered at ``y``.

    Parameters
    ----------
    Y : array-like
        Points at which to evaluate the kernel.
    y : float
        Kernel center.
    bw : float
        Bandwidth parameter.

    Returns
    -------
    numpy.ndarray
        Kernel values evaluated at ``Y``.
    """
    u = np.abs((Y - y) / bw)
    return np.where(u < 1, 70 / (81 * bw) * (1 - u ** 3) ** 3, 0.0)


KERNELS = {
    GAUSSIAN: kernel_gaussian,
    EPANECHNIKOV: kernel_epanechnikov,
    UNIFORM: kernel_uniform,
    TRIANGULAR: kernel_triangular,
    BIWEIGHT: kernel_biweight,
    TRIWEIGHT: kernel_triweight,
    COSINE: kernel_cosine,
    TRICUBE: kernel_tricube,
}


def get_bandwidth(bw, data, kernel):
    """
    Resolve a bandwidth specification into a numeric bandwidth.

    Parameters
    ----------
    bw : str or float
        Bandwidth specification. Supported string values are ``'scott'``,
        ``'silverman'``, and ``'normal_reference'``. A positive float may
        also be supplied directly.
    data : array-like
        Sample data used to estimate the bandwidth when ``bw`` is a string.
    kernel : str
        Kernel name used to select the normal-reference constant.

    Returns
    -------
    float
        Numeric bandwidth.

    Raises
    ------
    Exception
        Raised when an unsupported bandwidth rule is provided.
    AssertionError
        Raised when a numeric bandwidth is not a positive float.
    """
    if isinstance(bw, str):
        n = len(data)
        q = np.quantile(data, [.25, .75])
        A = min(np.std(data), (q[1] - q[0]) / 1.34)

        if bw == 'scott':
            bw = 1.059 * A * n ** (-.2)
        elif bw == 'silverman':
            bw = .9 * A * n ** (-.2)
        elif bw == 'normal_reference':
            bw = REFERENCE_CONSTANTS[kernel] * A * n ** (-.2)
        else:
            raise Exception
    else:
        assert isinstance(bw, float) and bw > 0
    return bw


def get_kernel_func(kernel):
    """
    Resolve a kernel specification into a callable kernel function.

    Parameters
    ----------
    kernel : str or callable
        Kernel name or custom kernel function.

    Returns
    -------
    callable
        Kernel function accepting ``Y``, ``y``, and ``bw``.

    Raises
    ------
    Exception
        Raised when ``kernel`` is neither a supported string nor callable.
    """
    if isinstance(kernel, str):
        return KERNELS[kernel]
    elif isinstance(kernel, callable):
        return kernel
    else:
        raise Exception(f"kernel {kernel}")


def kde(data, gridsize=DEFAULT_KDE_GRIDSIZE, kernel=DEFAULT_KDE_KERNEL, bw=DEFAULT_KDE_BW, adjust=DEFAULY_KDE_ADJUST,
        cut=DEFAULT_KDE_CUT, fft=DEFAULT_KDE_FFT, return_arrays=True, clip=None, sample=None, seed=0):
    """
    Estimate a one-dimensional kernel density from sample data.

    Parameters
    ----------
    data : array-like
        One-dimensional sample data.
    gridsize : int or None, optional
        Number of grid points used to evaluate the density. If ``None`` and
        ``fft`` is true, a default FFT grid is used. If ``None`` and ``fft`` is
        false, the density is evaluated at the sorted data points.
    kernel : str or callable, optional
        Kernel to use. Supported string values are defined in ``KERNEL_NAMES``.
        A custom callable may also be supplied.
    bw : str or float, optional
        Bandwidth rule or positive numeric bandwidth. Supported rules are
        ``'scott'``, ``'silverman'``, and ``'normal_reference'``.
    adjust : float, optional
        Multiplicative adjustment applied to the resolved bandwidth.
    cut : float, optional
        Number of bandwidths by which to extend the evaluation grid beyond the
        observed data range.
    fft : bool, optional
        Whether to use FFT-based convolution for density estimation.
    return_arrays : bool, optional
        If true, return ``(support, density)`` arrays. If false, return a
        ``KDEObject``.
    clip : None, bool, or tuple[float, float], optional
        If provided, clips or reflects density mass into the specified support.
        If true, clips to the observed data range.
    sample : int or None, optional
        If provided, randomly subsample this many observations before estimating
        the density.
    seed : int, optional
        Random seed used when ``sample`` is provided.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray] or KDEObject
        Either the support and density arrays, or a callable ``KDEObject`` when
        ``return_arrays`` is false.

    Examples
    --------
    Estimate the density of a bimodal sample:

    >>> import numpy as np
    >>> from kanly.api import kde
    >>> rng = np.random.default_rng(0)
    >>> data = np.concatenate([rng.normal(-2, 0.5, 500),
    ...                        rng.normal(2, 0.5, 500)])
    >>> x_grid, density = kde(data, gridsize=128, bw='silverman')

    Get a callable density object instead of arrays:

    >>> kde_obj = kde(data, return_arrays=False)
    >>> p = kde_obj([-2.0, 0.0, 2.0])                # doctest: +SKIP
    """
    data = np.asarray(data, dtype=np.float64)

    if isinstance(kernel, str):
        assert kernel in KERNEL_NAMES
    bw = get_bandwidth(bw, data, kernel)

    if sample is not None:
        assert isinstance(sample, int) and sample > 0
        r = random.Random(seed)
        return kde(
            data=data[r.sample(range(len(data)), k=sample)],
            gridsize=gridsize, kernel=kernel, bw=bw, cut=cut, adjust=adjust, clip=clip,
            sample=None, seed=0, return_arrays=return_arrays, fft=fft
        )

    kernel_func = get_kernel_func(kernel)

    if gridsize is None:
        if fft:
            gridsize = 2 ** 7
        else:
            data = np.sort(data)

    if gridsize is not None:
        assert gridsize >= 16

    if not fft and gridsize is None:
        return data, np.array([
            np.mean(kernel_func(data, t, bw * adjust))
            for t in data
        ])

    minx, maxx = data.min(), data.max()
    # If an explicit clip interval is supplied, discard data outside it before
    # computing the support range; the density will be zero-padded outside.
    if clip is not None and not isinstance(clip, bool):
        if clip[0] > minx:
            data = data[data >= clip[0]]
            minx = clip[0]
        if clip[1] < maxx:
            data = data[data <= clip[1]]
            maxx = clip[1]

    # Extend the evaluation grid by `cut` bandwidths on each side so that
    # kernel mass near the data boundaries is fully captured.
    lo, hi = minx - cut * bw * adjust, maxx + cut * bw * adjust
    bins = np.linspace(lo, hi, gridsize + 1)
    bin_midpoints = bins[1:] - (bins[1] - bins[0]) / 2

    # Bin data into a histogram (not normalised yet); each bin receives the
    # count of data points falling inside it.
    hist_vals = fast_histogram(
        data, nbins=gridsize, lower=bins[0], upper=bins[-1], density=False)

    if fft:
        # Silverman B W (1982) Algorithm AS 176. Kernel density estimation using the fast Fourier transform Appl. Statist. 31 93–99
        mid = (bin_midpoints[-1] + bin_midpoints[0]) / 2
        if kernel == 'gau':
            # For the Gaussian kernel the Fourier transform is also Gaussian,
            # so we can multiply in frequency domain analytically: the transform
            # of exp(-x^2/(2h^2)) evaluated at frequency s_l is exp(-h^2 s_l^2/2).
            h = bw * adjust
            s_l = 2.0 * np.pi * np.arange(0, gridsize) / (hi - lo)
            inside = np.exp(-.5 * h ** 2 * s_l ** 2)
            density = ifft(fft_func(hist_vals) * inside).real
        else:
            # For non-Gaussian kernels, numerically transform the kernel
            # evaluated on the grid and multiply in frequency domain.
            # fftshift corrects for the circular indexing of ifft.
            density = fftshift(
                ifft(fft_func(hist_vals) * fft_func(kernel_func(bins[:-1], mid, bw * adjust))).real
            )

    else:
        # Direct convolution: sum weighted kernel contributions at each grid
        # point.  Slow for large datasets but avoids FFT approximation.
        density = np.array([
            np.sum(kernel_func(bin_midpoints, t, bw * adjust) * hist_vals)
            for t in bin_midpoints
        ])

    normalize_density(density, bins[1] - bins[0])
    bin_midpoints, density = clip_kde(clip, minx, maxx, bin_midpoints, density)

    if return_arrays:
        return bin_midpoints, density
    else:
        return KDEObject(bin_midpoints, density, bw, adjust, clip, kernel, cut, fft)


def normalize_density(density, dx, inplace=True):
    """
    Normalize a discretized density so that it integrates to one.

    Parameters
    ----------
    density : numpy.ndarray
        Density values to normalize.
    dx : float
        Grid spacing between adjacent support points.
    inplace : bool, optional
        If true, mutate ``density`` in place. If false, normalize a copy.

    Returns
    -------
    numpy.ndarray
        Normalized density values.
    """
    if not inplace:
        density = density.copy()
    density -= density.min()
    density /= density.sum() * dx
    return density


class KDEObject(DillObject):
    """
    Callable wrapper around a discretized kernel density estimate.

    Parameters
    ----------
    support : numpy.ndarray
        Grid points at which the density was estimated.
    density : numpy.ndarray
        Density values corresponding to ``support``.
    bw : float
        Bandwidth used for the estimate.
    adjust : float
        Multiplicative bandwidth adjustment.
    clip : None, bool, or tuple[float, float]
        Clip setting used when constructing the estimate.
    kernel : str or callable
        Kernel used for the estimate.
    cut : float
        Grid extension in bandwidth units.
    fft : bool
        Whether FFT-based estimation was used.

    Attributes
    ----------
    support : numpy.ndarray
        Density support grid.
    density : numpy.ndarray
        Density values on the support grid.
    lb : float
        Lower bound of the support.
    ub : float
        Upper bound of the support.
    eval_func : callable
        Interpolation function used to evaluate the KDE at arbitrary points.
    """

    def __init__(self, support, density, bw, adjust, clip, kernel, cut, fft):
        """Construct a :class:`KDEObject` from pre-computed density arrays.

        Typically created by :func:`kde` with ``return_arrays=False`` rather
        than directly.

        Args:
            support: 1-D NumPy array of grid points where the density was
                evaluated.
            density: 1-D NumPy array of density values at ``support``.
            bw: Numeric bandwidth used for the estimate.
            adjust: Multiplicative bandwidth adjustment factor that was
                applied (stored for reference).
            clip: Clip specification passed to :func:`clip_kde`; stored for
                introspection.
            kernel: Kernel name or callable used for the estimate.
            cut: Grid extension measured in bandwidth units.
            fft: Whether FFT-based estimation was used.
        """
        self.support = support
        self.density = density
        self.lb = support[0]
        self.ub = support[-1]

        self.bw = bw
        self.adjust = adjust
        self.kernel = kernel
        self.cut = cut
        self.fft = fft
        self.clip = clip

        # Build a piecewise-linear interpolator for fast density evaluation
        # at arbitrary query points without re-running the KDE.
        self.eval_func = interp(support, density, kind=LINEAR)

    def __call__(self, x, *args, **kwargs):
        """
        Evaluate the estimated density at one or more points.

        Parameters
        ----------
        x : float or array-like
            Point or points at which to evaluate the density.
        *args
            Additional positional arguments. Currently unused.
        **kwargs
            Additional keyword arguments. Currently unused.

        Returns
        -------
        float or numpy.ndarray
            Interpolated density value or values.
        """
        return self.eval_func(x)


def clip_kde(clip, lo, hi, bin_midpoints, density):
    """
    Clip a KDE to a bounded interval by reflecting tail mass inward.

    Parameters
    ----------
    clip : None, bool, or tuple[float, float]
        Clip configuration. If ``None`` or false, no clipping is applied. If
        true, the interval ``(lo, hi)`` is used. Otherwise, a two-element
        interval is expected.
    lo : float
        Lower bound of the observed or filtered data range.
    hi : float
        Upper bound of the observed or filtered data range.
    bin_midpoints : numpy.ndarray
        KDE support grid.
    density : numpy.ndarray
        Density values corresponding to ``bin_midpoints``. This array is
        modified in place when clipping is applied.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The support grid and clipped density values.

    Raises
    ------
    AssertionError
        Raised when an explicit clip interval does not have shape ``(2,)`` or
        has a lower bound greater than or equal to its upper bound.
    """
    gridsize = len(bin_midpoints)

    if clip is None or isinstance(clip, bool) and not clip:
        return bin_midpoints, density

    # Resolve boolean True to use the observed data range as the clip bounds
    if isinstance(clip, bool) and clip:
        clip = (lo, hi)
    else:
        assert np.shape(clip) == (2,)
        assert clip[0] < clip[1]

    if clip[0] > -np.inf:
        # Left-boundary reflection: find the first bin that lies inside the
        # support, then fold the density mass to the left of clip[0] back onto
        # its mirror image inside the support.  This conserves total probability
        # mass while respecting the lower boundary constraint.
        for j, v in enumerate(bin_midpoints):
            if v >= clip[0]:
                density[j:j + j] += density[:j][::-1]
                density[:j] = 0.0
                break

    if clip[1] < np.inf:
        # Right-boundary reflection: analogous fold for the right tail.
        # Walk inward from the right end of the grid to find the last bin
        # inside the support, then reflect the density to the right of clip[1]
        # back onto its mirror image.
        for j in range(gridsize):
            v = bin_midpoints[gridsize - j - 1]
            if v <= clip[1]:
                density[gridsize - 2 * j:gridsize - j] += density[gridsize - j:][::-1]
                density[gridsize - j:] = 0.0
                break

    return bin_midpoints, density