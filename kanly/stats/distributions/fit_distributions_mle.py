"""
Generates (x,y) series for plotting distribution pdfs, logpdfs, and cdfs in
Cartesian plane. Also fits a small set of scipy.stats-style distributions by
closed-form estimates or numerical maximum likelihood.
"""
from __future__ import absolute_import, print_function

from scipy.stats import norm, expon, lognorm, t
import numpy as np
from kanly.optimize.bfgs_bounded_quasi_newton import bfgs_pqn
from kanly.stats.distributions.nopython_logpdf import logpdf_t, logpdf_lognorm

NORM = 'norm'
LOGNORM = 'lognorm'
EXPON = 'expon'
T = 't'

DISTRIBUTIONS = [NORM, LOGNORM, EXPON, T]

PDF = 'pdf'
CDF = 'cdf'


def get_normal_pdf_x_y(mean=0., scale=1., num_points=200, num_sigma=3.5):
    """Generate plotting coordinates for a normal PDF.

    Args:
        mean: Normal distribution mean, passed as scipy's ``loc``.
        scale: Normal distribution standard deviation.
        num_points: Number of x-values to generate.
        num_sigma: Number of standard deviations to include on each side of
            ``mean``.

    Returns:
        Tuple ``(x, y)`` where ``y`` is ``scipy.stats.norm.pdf(x)``.

    Examples
    --------
    Generate plotting grid for a standard-normal PDF:

    >>> import numpy as np
    >>> from kanly.api import get_normal_pdf_x_y
    >>> x, y = get_normal_pdf_x_y(mean=0.0, scale=1.0, num_points=101)
    >>> x.shape, y.shape
    ((101,), (101,))
    """
    x = np.linspace(mean - num_sigma * scale, mean + num_sigma * scale, num_points)
    y = norm.pdf(x, mean, scale)
    return x, y


def get_normal_cdf_x_y(mean=0., scale=1., num_points=200, num_sigma=3.5):
    """Generate plotting coordinates for a normal CDF.

    Args:
        mean: Normal distribution mean, passed as scipy's ``loc``.
        scale: Normal distribution standard deviation.
        num_points: Number of x-values to generate.
        num_sigma: Number of standard deviations to include on each side of
            ``mean``.

    Returns:
        Tuple ``(x, y)`` where ``y`` is ``scipy.stats.norm.cdf(x)``.

    Examples
    --------
    Generate plotting grid for a standard-normal CDF:

    >>> import numpy as np
    >>> from kanly.api import get_normal_cdf_x_y
    >>> x, y = get_normal_cdf_x_y(mean=0.0, scale=1.0, num_points=101)
    >>> y[0].round(3)                          # left tail near 0
    0.0
    """
    x = np.linspace(mean - num_sigma * scale, mean + num_sigma * scale, num_points)
    y = norm.cdf(x, mean, scale)
    return x, y


def get_normal_pdf_x_y_from_data(data, num_points=200, num_sigma=3.5):
    """Generate normal PDF plotting coordinates from sample moments.

    Args:
        data: Observations used to estimate mean and standard deviation.
        num_points: Number of x-values to generate.
        num_sigma: Number of standard deviations to include around the sample
            mean.

    Returns:
        Tuple ``(x, y)`` for the fitted normal PDF.

    Examples
    --------
    Plot a normal PDF fit to sample data:

    >>> import numpy as np
    >>> from kanly.api import get_normal_pdf_x_y_from_data
    >>> rng = np.random.default_rng(0)
    >>> data = rng.normal(2.0, 1.5, size=500)
    >>> x, y = get_normal_pdf_x_y_from_data(data)
    """
    return get_normal_pdf_x_y(np.mean(data), np.std(data), num_points=num_points, num_sigma=num_sigma)


def get_normal_cdf_x_y_from_data(data, num_points=200, num_sigma=3.5):
    """Generate normal CDF plotting coordinates from sample moments.

    Args:
        data: Observations used to estimate mean and standard deviation.
        num_points: Number of x-values to generate.
        num_sigma: Number of standard deviations to include around the sample
            mean.

    Returns:
        Tuple ``(x, y)`` for the fitted normal CDF.

    Examples
    --------
    Plot a normal CDF fit to sample data:

    >>> import numpy as np
    >>> from kanly.api import get_normal_cdf_x_y_from_data
    >>> rng = np.random.default_rng(0)
    >>> data = rng.normal(2.0, 1.5, size=500)
    >>> x, y = get_normal_cdf_x_y_from_data(data)
    """
    return get_normal_cdf_x_y(np.mean(data), np.std(data), num_points=num_points, num_sigma=num_sigma)


def _fit_distribution_params_by_mle(data, dist=NORM):
    """Fit supported distribution parameters by MLE-style estimates.

    Args:
        data: One-dimensional sample of observations.
        dist: Distribution name. Supported values are ``'norm'``,
            ``'lognorm'``, ``'expon'``, and ``'t'``.

    Returns:
        Dictionary of parameters suitable for constructing the matching
        scipy.stats frozen distribution.
    """
    dist = dist.lower()
    if dist == NORM:
        loc, scale = data.mean(), data.std()
        return {'loc': loc, 'scale': scale}
    elif dist == LOGNORM:
        # estimates mu, sigma from https://en.wikipedia.org/wiki/Log-normal_distribution
        # not params of scipy distribution (no closed form)
        # ln_y = np.log(data)
        # mu = np.mean(ln_y)
        # sigma = np.sqrt(np.mean((ln_y - mu) ** 2))
        # return {'loc': 0.0, 's': sigma, 'scale': np.exp(mu)}
        data = np.asarray(data)
        # scipy.stats.lognorm uses ``s`` for the shape parameter and ``scale``
        # for exp(mu), so optimize directly in scipy's parameterization.
        result = bfgs_pqn(
            lambda params: logpdf_lognorm(data, loc=params[0], s=params[1], scale=params[2]).sum(),
            x0=[0., 1., 1.],
            maximize=True
        )
        return dict(zip(['loc', 's', 'scale'], result.x))
    elif dist == EXPON:
        return {'scale': np.mean(data)}  # scale, f(x) = scale * exp(-scale * x)
    elif dist == T:
        data = np.asarray(data)
        # Student's t has no simple closed-form MLE for df, loc, and scale.
        result = bfgs_pqn(
            lambda params: logpdf_t(data, df=params[2], loc=params[0], scale=params[1]).sum(),
            x0=[0.0, 1.0, 4.0],
            maximize=True
        )
        return {'df': result.x[2], 'loc': result.x[0], 'scale': result.x[1]}
    else:
        raise Exception(f'dist must be in {DISTRIBUTIONS}')


def get_mle_distribution(data, dist=NORM):
    """Fit and return a scipy.stats frozen distribution object.

    Args:
        data: One-dimensional sample of observations.
        dist: Distribution name. Supported values are ``'norm'``,
            ``'lognorm'``, ``'expon'``, and ``'t'``.

    Returns:
        Tuple ``(dist_obj, params)`` where ``dist_obj`` is a frozen scipy
        distribution and ``params`` is the fitted parameter dictionary.

    Examples
    --------
    Fit a Student-t distribution to heavy-tailed data:

    >>> import numpy as np
    >>> from kanly.api import get_mle_distribution
    >>> rng = np.random.default_rng(0)
    >>> data = rng.standard_t(df=5, size=500)
    >>> dist_obj, params = get_mle_distribution(data, dist='t')    # doctest: +SKIP
    >>> dist_obj.pdf(0.0).round(3)                                  # doctest: +SKIP
    0.379
    """
    if dist.lower() not in DISTRIBUTIONS:
        raise Exception(f'dist must be in {DISTRIBUTIONS}')

    dist_scipy = {
        NORM: norm,
        LOGNORM: lognorm,
        T: t,
        EXPON: expon
    }[dist.lower()]

    params = _fit_distribution_params_by_mle(data, dist)
    print(dist, params)
    return dist_scipy(**params), params


def _coalesce(x, fill):
    """Return a fallback value when ``x`` is None.

    Args:
        x: Candidate value.
        fill: Fallback value to use when ``x`` is None.

    Returns:
        ``fill`` if ``x`` is None, otherwise ``x``.
    """
    if x is None:
        return fill
    else:
        return x


def get_mle_x_y(data, dist=NORM, curve=PDF, num_points=201, left_quantile=None, right_quantile=None,
                return_dist_obj=False):
    """
    Fit a distribution and generate plotting coordinates.

    Args:
        data: One-dimensional sample of observations.
        dist: Distribution name. Supported values are ``'norm'``,
            ``'lognorm'``, ``'expon'``, and ``'t'``.
        curve: Curve type to evaluate. Supported values are ``'pdf'``,
            ``'cdf'``, and ``'logpdf'``.
        num_points: Number of x-values to generate between fitted quantiles.
        left_quantile: Optional lower fitted-distribution quantile for the
            plotting range. Defaults depend on ``dist``.
        right_quantile: Optional upper fitted-distribution quantile for the
            plotting range. Defaults depend on ``dist``.
        return_dist_obj: Whether to include the frozen distribution and fitted
            parameter dictionary in the return value.

    Returns: x,y values for plotting

    Examples
    --------
    Fit and obtain plotting grid for a log-normal sample:

    >>> import numpy as np
    >>> from kanly.api import get_mle_x_y
    >>> rng = np.random.default_rng(0)
    >>> data = rng.lognormal(mean=0.0, sigma=0.5, size=500)
    >>> x, y = get_mle_x_y(data, dist='lognorm', curve='pdf')   # doctest: +SKIP

    Get the fitted distribution back as well:

    >>> x, y, dist_obj, params = get_mle_x_y(                   # doctest: +SKIP
    ...     data, dist='lognorm', curve='pdf', return_dist_obj=True)
    """
    curve = curve.lower()
    assert curve in ['cdf', 'pdf', 'logpdf']

    dist = dist.lower()
    if dist in [NORM, T]:
        left_quantile = _coalesce(left_quantile, .005)
        right_quantile = _coalesce(right_quantile, .995)
    elif dist in [EXPON, LOGNORM]:
        left_quantile = _coalesce(left_quantile, .00001)
        right_quantile = _coalesce(right_quantile, .99)
    else:
        raise Exception(f'dist must be in {DISTRIBUTIONS}')

    dist_obj, params = get_mle_distribution(data, dist=dist)
    l, h = dist_obj.ppf([left_quantile, right_quantile])
    x = np.linspace(l, h, num_points)
    f_x = getattr(dist_obj, curve)(x)

    if return_dist_obj:
        return x, f_x, dist_obj, params
    else:
        return x, f_x
