"""
Generates (x,y) series for plotting distribution pdfs, logpdfs, and cdfs in
Cartesian plane. Also fits scipy.stats-style continuous distributions by
maximum likelihood.
"""
from __future__ import absolute_import, print_function

from collections.abc import Mapping

from scipy.stats import (
    beta, cauchy, expon, gamma, laplace, logistic, lognorm, norm, pareto, t,
    uniform, weibull_min,
)
import numpy as np

NORM = 'norm'
LOGNORM = 'lognorm'
EXPON = 'expon'
T = 't'
GAMMA = 'gamma'
WEIBULL_MIN = 'weibull_min'
BETA = 'beta'
LOGISTIC = 'logistic'
LAPLACE = 'laplace'
CAUCHY = 'cauchy'
PARETO = 'pareto'
UNIFORM = 'uniform'

_SCIPY_DISTRIBUTIONS = {
    NORM: norm,
    LOGNORM: lognorm,
    EXPON: expon,
    T: t,
    GAMMA: gamma,
    WEIBULL_MIN: weibull_min,
    BETA: beta,
    LOGISTIC: logistic,
    LAPLACE: laplace,
    CAUCHY: cauchy,
    PARETO: pareto,
    UNIFORM: uniform,
}

DISTRIBUTIONS = list(_SCIPY_DISTRIBUTIONS)

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


def _as_valid_sample(data):
    """Return *data* as a finite, non-degenerate one-dimensional sample."""
    sample = np.asarray(data, dtype=float)
    if sample.ndim != 1:
        raise ValueError('data must be one-dimensional')
    if sample.size < 2:
        raise ValueError('data must contain at least two observations')
    if not np.all(np.isfinite(sample)):
        raise ValueError('data must contain only finite observations')
    if np.ptp(sample) == 0:
        raise ValueError('data must contain at least two distinct values')
    return sample


def _get_scipy_distribution(dist):
    """Validate a distribution name and return its scipy implementation."""
    if not isinstance(dist, str) or dist.lower() not in DISTRIBUTIONS:
        raise ValueError(f'dist must be one of {DISTRIBUTIONS}')
    return dist.lower(), _SCIPY_DISTRIBUTIONS[dist.lower()]


def _fit_distribution_params_by_mle(data, dist=NORM, fit_kwargs=None):
    """Fit supported distribution parameters by maximum likelihood.

    Args:
        data: One-dimensional sample of observations.
        dist: Name in :data:`DISTRIBUTIONS`. Supported values are ``'norm'``,
            ``'lognorm'``, ``'expon'``, ``'t'``, ``'gamma'``,
            ``'weibull_min'``, ``'beta'``, ``'logistic'``, ``'laplace'``,
            ``'cauchy'``, ``'pareto'``, and ``'uniform'``.
        fit_kwargs: Optional keyword arguments passed to the distribution's
            ``scipy.stats`` ``fit`` method. This is useful for fixing known
            parameters, for example ``{'floc': 0}`` for an unshifted gamma or
            ``{'floc': 0, 'fscale': 1}`` for proportions fit by a beta.

    Returns:
        Dictionary of parameters suitable for constructing the matching
        scipy.stats frozen distribution.
    """
    _, dist_scipy = _get_scipy_distribution(dist)
    sample = _as_valid_sample(data)
    if fit_kwargs is None:
        fit_kwargs = {}
    elif not isinstance(fit_kwargs, Mapping):
        raise TypeError('fit_kwargs must be a mapping or None')
    else:
        fit_kwargs = dict(fit_kwargs)
    fitted = dist_scipy.fit(sample, **fit_kwargs)

    shape_names = [] if dist_scipy.shapes is None else [
        name.strip() for name in dist_scipy.shapes.split(',')
    ]
    parameter_names = shape_names + ['loc', 'scale']
    params = dict(zip(parameter_names, fitted))
    if len(params) != len(fitted) or not np.all(np.isfinite(fitted)):
        raise RuntimeError(f'{dist_scipy.name}.fit returned invalid parameters: {fitted}')
    if params['scale'] <= 0:
        raise RuntimeError(f'{dist_scipy.name}.fit returned a non-positive scale')
    return params


def get_mle_distribution(data, dist=NORM, fit_kwargs=None):
    """Fit and return a scipy.stats frozen distribution object.

    Args:
        data: One-dimensional sample of observations.
        dist: Name in :data:`DISTRIBUTIONS`.
        fit_kwargs: Optional keyword arguments passed to ``scipy.stats.fit``.
            For example, pass ``{'floc': 0}`` to fit a positive distribution
            whose lower support boundary is known to be zero.

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
    _, dist_scipy = _get_scipy_distribution(dist)
    params = _fit_distribution_params_by_mle(data, dist, fit_kwargs=fit_kwargs)
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
                return_dist_obj=False, fit_kwargs=None):
    """
    Fit a distribution and generate plotting coordinates.

    Args:
        data: One-dimensional sample of observations.
        dist: Name in :data:`DISTRIBUTIONS`.
        curve: Curve type to evaluate. Supported values are ``'pdf'``,
            ``'cdf'``, and ``'logpdf'``.
        num_points: Number of x-values to generate between fitted quantiles.
        left_quantile: Optional lower fitted-distribution quantile for the
            plotting range. Defaults depend on ``dist``.
        right_quantile: Optional upper fitted-distribution quantile for the
            plotting range. Defaults depend on ``dist``.
        return_dist_obj: Whether to include the frozen distribution and fitted
            parameter dictionary in the return value.
        fit_kwargs: Optional keyword arguments passed to ``scipy.stats.fit``.

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
    if not isinstance(curve, str) or curve.lower() not in ['cdf', 'pdf', 'logpdf']:
        raise ValueError("curve must be one of ['cdf', 'pdf', 'logpdf']")
    curve = curve.lower()

    dist, _ = _get_scipy_distribution(dist)
    if (
        not isinstance(num_points, (int, np.integer))
        or isinstance(num_points, (bool, np.bool_))
        or num_points < 2
    ):
        raise ValueError('num_points must be an integer greater than or equal to 2')
    if dist in [NORM, T, LOGISTIC, LAPLACE, CAUCHY]:
        left_quantile = _coalesce(left_quantile, .005)
        right_quantile = _coalesce(right_quantile, .995)
    elif dist in [EXPON, LOGNORM, GAMMA, WEIBULL_MIN, PARETO]:
        left_quantile = _coalesce(left_quantile, .00001)
        right_quantile = _coalesce(right_quantile, .99)
    elif dist in [BETA, UNIFORM]:
        left_quantile = _coalesce(left_quantile, .001)
        right_quantile = _coalesce(right_quantile, .999)
    else:
        raise ValueError(f'dist must be one of {DISTRIBUTIONS}')

    try:
        valid_quantiles = (
            np.isscalar(left_quantile)
            and np.isscalar(right_quantile)
            and np.isfinite(left_quantile)
            and np.isfinite(right_quantile)
            and 0 < left_quantile < right_quantile < 1
        )
    except TypeError:
        valid_quantiles = False
    if not valid_quantiles:
        raise ValueError('quantiles must satisfy 0 < left_quantile < right_quantile < 1')

    dist_obj, params = get_mle_distribution(data, dist=dist, fit_kwargs=fit_kwargs)
    l, h = dist_obj.ppf([left_quantile, right_quantile])
    if not np.all(np.isfinite([l, h])) or l >= h:
        raise RuntimeError(f'fitted {dist} distribution produced an invalid plotting range')
    x = np.linspace(l, h, num_points)
    f_x = getattr(dist_obj, curve)(x)
    if not np.all(np.isfinite(f_x)):
        raise RuntimeError(f'fitted {dist} distribution produced non-finite {curve} values')

    if return_dist_obj:
        return x, f_x, dist_obj, params
    else:
        return x, f_x
