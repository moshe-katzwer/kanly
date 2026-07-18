"""
Logpdf functions for some common distributions.

Not "frozen", so these are the exact logpdfs, while those in
`nopython_frozen_logpdf` are correct up to a constant assuming
the parameters of the distribution are "frozen".

That is, use `nopython_logpdf` logpdfs for MCMC when solving for
parameters of that distribution, use `nopython_frozen_logpdf` when the parameters
of that distribution are fixed in a prior or likelihood.

`nopython_` prefix are compatible with numba, but are deprecated since
`@overload` numba support was added.

These distributions don't do any domain checks
"""
# TODO FIX OVERLOAD FOR MULTIVARIATE DISTRIBUTIONS
from __future__ import absolute_import, print_function

import numpy as np
from numba import njit
from numba.extending import overload

import scipy.special as scsp
# import numba_scipy  # do not delete

from kanly.stats.distributions.nopython_scipy_special import (
    beta, betaln, ndtr, erf, erfc, gamma, gammaln
)

PI = np.pi
_2_PI = 2.0 * np.pi
LOG_PI = np.log(np.pi)
LOG_2 = np.log(2)
LOG_2_PI = np.log(_2_PI)
RT_2_PI = np.sqrt(_2_PI)
LOG_RT_2_OVER_PI = 0.5 * np.log(2 / np.pi)
LOG_2_OVER_PI = np.log(2 / np.pi)
# Reassign using PI so downstream constants share the same symbol source.
_2_PI = 2.0 * PI
_2_OVER_PI = 2.0 / PI
RT_2_OVER_PI = np.sqrt(2.0 / PI)


def logpdf_beta(x, a, b, loc=0., scale=1.):
    """
    Evaluate the scipy.stats.beta-style beta logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        a: First shape or lower-bound parameter, matching scipy.stats naming for this distribution.
        b: Second shape or upper-bound parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        beta logpdf value without domain checks.

    Examples
    --------
    Evaluate the standard Beta(2, 5) logpdf on a small grid:

    >>> import numpy as np
    >>> from kanly.api import logpdf_beta
    >>> x = np.array([0.1, 0.3, 0.5, 0.7])
    >>> logpdf_beta(x, a=2.0, b=5.0).round(3)            # doctest: +SKIP
    array([ 0.679,  0.998,  0.0  , -1.609])
    """
    const = - np.log(scale) - betaln(a, b)
    z = (x - loc) / scale
    return (a - 1) * np.log(z) + (b - 1) * np.log(1 - z) + const


def pdf_beta(x, a, b, loc=0., scale=1.):
    """
    Evaluate the scipy.stats.beta-style beta pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        a: First shape or lower-bound parameter, matching scipy.stats naming for this distribution.
        b: Second shape or upper-bound parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        beta pdf value without domain checks.

    Examples
    --------
    Density of Beta(2, 5) at a vector of points:

    >>> import numpy as np
    >>> from kanly.api import pdf_beta
    >>> pdf_beta(np.array([0.1, 0.3, 0.5]),
    ...          a=2.0, b=5.0).round(3)                  # doctest: +SKIP
    array([1.968, 2.7  , 0.938])
    """
    z = (x - loc) / scale
    return z**(a-1) * (1-z)**(b-1) /(scale*beta(a, b))


def logpdf_cauchy(x, loc=0., scale=1.):
    """
    Evaluate the scipy.stats.cauchy-style cauchy logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        cauchy logpdf value without domain checks.

    Examples
    --------
    Cauchy logpdf at a small symmetric grid:

    >>> import numpy as np
    >>> from kanly.api import logpdf_cauchy
    >>> logpdf_cauchy(np.array([-2.0, 0.0, 2.0])).round(3)   # doctest: +SKIP
    array([-2.755, -1.145, -2.755])
    """
    const = -LOG_PI - np.log(scale)
    z = (x - loc) / scale
    return -np.log1p(z ** 2) + const


def pdf_cauchy(x, loc=0., scale=1.):
    """
    Evaluate the scipy.stats.cauchy-style cauchy pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        cauchy pdf value without domain checks.

    Examples
    --------
    Cauchy density at a few points (location 0, scale 1):

    >>> import numpy as np
    >>> from kanly.api import pdf_cauchy
    >>> pdf_cauchy(np.array([-1.0, 0.0, 1.0])).round(3)      # doctest: +SKIP
    array([0.159, 0.318, 0.159])
    """
    z = (x - loc) / scale
    return 1.0 / ((PI * scale) * (1 + z ** 2))


def logpdf_chi2(x, df, loc=0., scale=1.):
    """
    Evaluate the scipy.stats.chi2-style chi-square logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        df: Degrees-of-freedom parameter.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        chi-square logpdf value without domain checks.

    Examples
    --------
    Chi-square logpdf with 3 degrees of freedom:

    >>> import numpy as np
    >>> from kanly.api import logpdf_chi2
    >>> logpdf_chi2(np.array([0.5, 2.0, 5.0]), df=3).round(3)   # doctest: +SKIP
    array([-1.846, -1.418, -2.418])
    """
    const = -(df / 2) * LOG_2 - gammaln(df / 2) - np.log(scale)
    z = (x - loc) / scale
    return (df / 2 - 1) * np.log(z) - z / 2 + const


def pdf_chi2(x, df, loc=0., scale=1.):
    """
    Evaluate the scipy.stats.chi2-style chi-square pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        df: Degrees-of-freedom parameter.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        chi-square pdf value without domain checks.

    Examples
    --------
    Chi-square density with 3 degrees of freedom:

    >>> import numpy as np
    >>> from kanly.api import pdf_chi2
    >>> pdf_chi2(np.array([0.5, 2.0, 5.0]), df=3).round(3)      # doctest: +SKIP
    array([0.158, 0.242, 0.089])
    """
    z = (x - loc) / scale
    return 1.0 / (2.0 ** (df/2) * gamma(df/2) * scale) * z ** (df/2-1) * np.exp(-z/2)


def logpdf_expon(x, loc=0., scale=1.):
    """
    Evaluate the scipy.stats.expon-style exponential logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        exponential logpdf value without domain checks.

    Examples
    --------
    Exponential logpdf with scale 2 at a few points:

    >>> import numpy as np
    >>> from kanly.api import logpdf_expon
    >>> logpdf_expon(np.array([0.5, 1.0, 3.0]),
    ...              scale=2.0).round(3)                       # doctest: +SKIP
    array([-0.943, -1.193, -2.193])
    """
    const = -np.log(scale)
    z = (x - loc) / scale
    return -z + const


def pdf_expon(x, loc=0., scale=1.):
    """
    Evaluate the scipy.stats.expon-style exponential pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        exponential pdf value without domain checks.

    Examples
    --------
    Exponential density (rate 1):

    >>> import numpy as np
    >>> from kanly.api import pdf_expon
    >>> pdf_expon(np.array([0.0, 1.0, 3.0])).round(3)          # doctest: +SKIP
    array([1.   , 0.368, 0.05 ])
    """
    z = (x - loc) / scale
    return np.exp(-z) / scale


def logpdf_f(x, dfn, dfd, loc=0.0, scale=0.0):
    """
    Evaluate the scipy.stats.f-style F logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        dfn: Numerator degrees-of-freedom parameter for the F distribution.
        dfd: Denominator degrees-of-freedom parameter for the F distribution.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        F logpdf value without domain checks.

    Examples
    --------
    F logpdf with (5, 20) degrees of freedom; pass ``scale=1.0`` because the
    historical default is ``0.0``:

    >>> import numpy as np
    >>> from kanly.api import logpdf_f
    >>> logpdf_f(np.array([0.5, 1.0, 2.0]),
    ...          dfn=5, dfd=20, scale=1.0).round(3)            # doctest: +SKIP
    array([-0.628, -0.553, -1.488])
    """
    # Keep the historical default scale value; callers should provide a
    # positive scale for finite F-distribution densities.
    const = dfn / 2 * np.log(dfn) + dfd / 2 * np.log(dfd) - betaln(dfn / 2, dfd / 2) - np.log(scale)
    z = (x - loc) / scale
    return (dfn / 2 - 1) * np.log(z) - (dfn + dfd) / 2 * np.log(dfn * z + dfd) + const


def pdf_f(x, dfn, dfd, loc=0.0, scale=0.0):
    """
    Evaluate the scipy.stats.f-style F pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        dfn: Numerator degrees-of-freedom parameter for the F distribution.
        dfd: Denominator degrees-of-freedom parameter for the F distribution.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        F pdf value without domain checks.

    Examples
    --------
    F density with (5, 20) degrees of freedom (note ``scale=1.0``):

    >>> import numpy as np
    >>> from kanly.api import pdf_f
    >>> pdf_f(np.array([0.5, 1.0, 2.0]),
    ...       dfn=5, dfd=20, scale=1.0).round(3)               # doctest: +SKIP
    array([0.534, 0.575, 0.226])
    """
    # Keep the historical default scale value; callers should provide a
    # positive scale for finite F-distribution densities.
    z = (x - loc) / scale
    return (
        z ** (dfn/2-1) * dfd ** (dfd/2) * dfn ** (dfn/2)
    ) / (
        (dfd + dfn * z) ** ((dfn+dfd)/2) * beta(dfn/2, dfd/2) * scale
    )


def logpdf_gamma(x, a, loc=0., scale=1.):
    """
    Evaluate the scipy.stats.gamma-style gamma logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        a: First shape or lower-bound parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        gamma logpdf value without domain checks.

    Examples
    --------
    Gamma logpdf with shape 2 and scale 2:

    >>> import numpy as np
    >>> from kanly.api import logpdf_gamma
    >>> logpdf_gamma(np.array([0.5, 2.0, 5.0]),
    ...              a=2.0, scale=2.0).round(3)                # doctest: +SKIP
    array([-2.636, -1.386, -2.136])
    """
    const = -gammaln(a) - np.log(scale)
    z = (x - loc) / scale
    return (a - 1) * np.log(z) - z + const



def pdf_gamma(x, a, loc=0., scale=1.):
    """
    Evaluate the scipy.stats.gamma-style gamma pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        a: First shape or lower-bound parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        gamma pdf value without domain checks.

    Examples
    --------
    Gamma density with shape 2 and scale 2:

    >>> import numpy as np
    >>> from kanly.api import pdf_gamma
    >>> pdf_gamma(np.array([0.5, 2.0, 5.0]),
    ...           a=2.0, scale=2.0).round(3)                   # doctest: +SKIP
    array([0.072, 0.184, 0.103])
    """
    z = (x - loc) / scale
    return z**(a-1) * np.exp(-z) / (gamma(a) * scale)


def logpdf_genextreme(x, c, loc=0., scale=1.):
    """
    Evaluate the scipy.stats.genextreme-style generalized extreme value logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        c: Shape parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        generalized extreme value logpdf value without domain checks.

    Examples
    --------
    GEV logpdf with c=0.1 at a small grid:

    >>> import numpy as np
    >>> from kanly.api import logpdf_genextreme
    >>> logpdf_genextreme(np.array([-1.0, 0.0, 1.0]),
    ...                   c=0.1).round(3)                      # doctest: +SKIP
    array([-2.018, -1.   , -1.39 ])
    """
    z = (x - loc) / scale
    const = -np.log(scale)
    if c == 0:
        return -np.exp(-z) - z + const
    else:
        return -(1 - c * z) ** (1 / c) + (1 / c - 1) * np.log(1 - c * z) + const


def pdf_genextreme(x, c, loc=0., scale=1.):
    """
    Evaluate the scipy.stats.genextreme-style generalized extreme value pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        c: Shape parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        generalized extreme value pdf value without domain checks.

    Examples
    --------
    GEV density with c=0.1 at a small grid:

    >>> import numpy as np
    >>> from kanly.api import pdf_genextreme
    >>> pdf_genextreme(np.array([-1.0, 0.0, 1.0]),
    ...                c=0.1).round(3)                          # doctest: +SKIP
    array([0.133, 0.368, 0.249])
    """
    z = (x - loc) / scale
    const = -np.log(scale)
    if c == 0:
        exp_z = np.exp(-z)
        f = exp_z * np.exp(-exp_z)
    else:
        f = np.exp(-(1-c*z)**(1/c)) * (1-c*z)**(1/c-1)
    return f/scale

def logpdf_halfcauchy(x, loc=0., scale=1.):
    """
    Evaluate the scipy.stats.halfcauchy-style half-Cauchy logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        half-Cauchy logpdf value without domain checks.

    Examples
    --------
    Half-Cauchy logpdf at a few points:

    >>> import numpy as np
    >>> from kanly.api import logpdf_halfcauchy
    >>> logpdf_halfcauchy(np.array([0.0, 1.0, 3.0])).round(3)   # doctest: +SKIP
    array([-0.452, -1.145, -2.755])
    """
    z = (x - loc) / scale
    const = LOG_2_OVER_PI - np.log(scale)
    return -np.log(1 + z ** 2) + const

def pdf_halfcauchy(x, loc=0., scale=1.):
    """
    Evaluate the scipy.stats.halfcauchy-style half-Cauchy pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        half-Cauchy pdf value without domain checks.

    Examples
    --------
    Half-Cauchy density at a few points:

    >>> import numpy as np
    >>> from kanly.api import pdf_halfcauchy
    >>> pdf_halfcauchy(np.array([0.0, 1.0, 3.0])).round(3)      # doctest: +SKIP
    array([0.637, 0.318, 0.064])
    """
    z = (x - loc) / scale
    return _2_OVER_PI / (scale * (1 + z ** 2))

def logpdf_halfnorm(x, loc=0., scale=1.):
    """
    Evaluate the scipy.stats.halfnorm-style half-normal logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        half-normal logpdf value without domain checks.

    Examples
    --------
    Half-normal logpdf at a small grid:

    >>> import numpy as np
    >>> from kanly.api import logpdf_halfnorm
    >>> logpdf_halfnorm(np.array([0.0, 1.0, 2.5])).round(3)     # doctest: +SKIP
    array([-0.226, -0.726, -3.351])
    """
    const = LOG_RT_2_OVER_PI - np.log(scale)
    z = (x - loc) / scale
    return const - z ** 2 / 2


def pdf_halfnorm(x, loc=0., scale=1.):
    """
    Evaluate the scipy.stats.halfnorm-style half-normal pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        half-normal pdf value without domain checks.

    Examples
    --------
    Half-normal density at a small grid:

    >>> import numpy as np
    >>> from kanly.api import pdf_halfnorm
    >>> pdf_halfnorm(np.array([0.0, 1.0, 2.5])).round(3)        # doctest: +SKIP
    array([0.798, 0.484, 0.035])
    """
    z = (x - loc) / scale
    return RT_2_OVER_PI * np.exp(-z**2/2) / scale


def logpdf_invgamma(x, a, loc=0., scale=1.):
    """
    a=a, scale=b in wikipedia def'n
    https://en.wikipedia.org/wiki/Inverse-gamma_distribution

    Evaluate the scipy.stats.invgamma-style inverse-gamma logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        a: Shape parameter.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Inverse-gamma logpdf value without domain checks.

    Examples
    --------
    Inverse-gamma logpdf with shape 3:

    >>> import numpy as np
    >>> from kanly.api import logpdf_invgamma
    >>> logpdf_invgamma(np.array([0.5, 1.0, 3.0]),
    ...                 a=3.0).round(3)                         # doctest: +SKIP
    array([-1.082, -0.693, -3.602])
    """
    const = -gammaln(a) - np.log(scale)
    z = (x - loc) / scale
    return -(a + 1) * np.log(z) - 1 / z + const

def pdf_invgamma(x, a, loc=0., scale=1.):
    """
    a=a, scale=b in wikipedia def'n
    https://en.wikipedia.org/wiki/Inverse-gamma_distribution

    Evaluate the scipy.stats.invgamma-style inverse-gamma pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        a: Shape parameter.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Inverse-gamma pdf value without domain checks.

    Examples
    --------
    Inverse-gamma density with shape 3:

    >>> import numpy as np
    >>> from kanly.api import pdf_invgamma
    >>> pdf_invgamma(np.array([0.5, 1.0, 3.0]),
    ...              a=3.0).round(3)                            # doctest: +SKIP
    array([0.339, 0.5  , 0.027])
    """
    z = (x - loc) / scale
    return np.exp(-1/z) / (scale * z ** (a + 1) * gamma(a))


def logpdf_laplace(x, loc=0., scale=1.):
    """
    Evaluate the scipy.stats.laplace-style Laplace logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Laplace logpdf value without domain checks.

    Examples
    --------
    Laplace logpdf at a small symmetric grid:

    >>> import numpy as np
    >>> from kanly.api import logpdf_laplace
    >>> logpdf_laplace(np.array([-1.0, 0.0, 1.0])).round(3)     # doctest: +SKIP
    array([-1.693, -0.693, -1.693])
    """
    const = -LOG_2 - np.log(scale)
    z = (x - loc) / scale
    return -np.abs(z) + const


def pdf_laplace(x, loc=0., scale=1.):
    """
    Evaluate the scipy.stats.laplace-style Laplace pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Laplace pdf value without domain checks.

    Examples
    --------
    Laplace density at a small symmetric grid:

    >>> import numpy as np
    >>> from kanly.api import pdf_laplace
    >>> pdf_laplace(np.array([-1.0, 0.0, 1.0])).round(3)        # doctest: +SKIP
    array([0.184, 0.5  , 0.184])
    """
    const = -LOG_2 - np.log(scale)
    z = (x - loc) / scale
    return np.exp(-np.abs(z)) / (2*scale)


def logpdf_logistic(x, loc=0., scale=1.):
    """
    Evaluate the scipy.stats.logistic-style logistic logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        logistic logpdf value without domain checks.

    Examples
    --------
    Logistic logpdf at a few points:

    >>> import numpy as np
    >>> from kanly.api import logpdf_logistic
    >>> logpdf_logistic(np.array([-1.0, 0.0, 1.0])).round(3)    # doctest: +SKIP
    array([-1.626, -1.386, -1.626])
    """
    z = (x - loc) / scale
    return -z - 2. * np.log1p(np.exp(-z)) - np.log(scale)

def pdf_logistic(x, loc=0., scale=1.):
    """
    Evaluate the scipy.stats.logistic-style logistic pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        logistic pdf value without domain checks.

    Examples
    --------
    Logistic density at a few points:

    >>> import numpy as np
    >>> from kanly.api import pdf_logistic
    >>> pdf_logistic(np.array([-1.0, 0.0, 1.0])).round(3)       # doctest: +SKIP
    array([0.197, 0.25 , 0.197])
    """
    z = (x - loc) / scale
    exp_term = np.exp(-z)
    return exp_term / (1+exp_term)**2 / scale


def logpdf_lognorm(x, s, loc=0., scale=1.):
    """
    Evaluate the scipy.stats.lognorm-style log-normal logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        s: Shape parameter for the log-normal distribution.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        log-normal logpdf value without domain checks.

    Examples
    --------
    Log-normal logpdf with shape 0.5:

    >>> import numpy as np
    >>> from kanly.api import logpdf_lognorm
    >>> logpdf_lognorm(np.array([0.5, 1.0, 2.5]),
    ...                s=0.5).round(3)                          # doctest: +SKIP
    array([-0.685, -0.226, -1.838])
    """
    const = -0.5 * LOG_2_PI - np.log(s * scale)
    z = (x - loc) / scale
    return const - 0.5 * (np.log(z) / s) ** 2 - np.log(z)


def pdf_lognorm(x, s, loc=0., scale=1.):
    """
    Evaluate the scipy.stats.lognorm-style log-normal pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        s: Shape parameter for the log-normal distribution.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        log-normal pdf value without domain checks.

    Examples
    --------
    Log-normal density with shape 0.5:

    >>> import numpy as np
    >>> from kanly.api import pdf_lognorm
    >>> pdf_lognorm(np.array([0.5, 1.0, 2.5]),
    ...             s=0.5).round(3)                              # doctest: +SKIP
    array([0.504, 0.798, 0.159])
    """
    z = (x - loc) / scale
    return np.exp(- 0.5 * (np.log(z) / s) ** 2) / (z * RT_2_PI * s * scale)


def logpdf_norm(x, loc=0., scale=1.):
    """
    Evaluate the scipy.stats.norm-style normal logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        normal logpdf value without domain checks.

    Examples
    --------
    Standard-normal logpdf at a small symmetric grid:

    >>> import numpy as np
    >>> from kanly.api import logpdf_norm
    >>> logpdf_norm(np.array([-1.0, 0.0, 1.0])).round(3)        # doctest: +SKIP
    array([-1.419, -0.919, -1.419])

    Loc/scale parameterization (mean 2, std 3):

    >>> logpdf_norm(2.0, loc=2.0, scale=3.0).round(3)            # doctest: +SKIP
    -2.018
    """
    z = (x - loc) / scale
    return -0.5 * z ** 2 - np.log(scale) - 0.5 * LOG_2_PI


def pdf_norm(x, loc=0., scale=1.):
    """
    Evaluate the scipy.stats.norm-style normal pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        normal pdf value without domain checks.

    Examples
    --------
    Standard-normal density:

    >>> import numpy as np
    >>> from kanly.api import pdf_norm
    >>> pdf_norm(np.array([-1.0, 0.0, 1.0])).round(3)            # doctest: +SKIP
    array([0.242, 0.399, 0.242])
    """
    z = (x - loc) / scale
    return np.exp(-0.5 * z ** 2) / (scale * RT_2_PI)


def logpdf_pareto(x, b, loc=0.0, scale=1.):
    """
    Evaluate the scipy.stats.pareto-style Pareto logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        b: Second shape or upper-bound parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Pareto logpdf value without domain checks.

    Examples
    --------
    Pareto logpdf with shape b=2.5:

    >>> import numpy as np
    >>> from kanly.api import logpdf_pareto
    >>> logpdf_pareto(np.array([1.0, 2.0, 5.0]),
    ...               b=2.5).round(3)                            # doctest: +SKIP
    array([ 0.916, -1.510, -4.106])
    """
    z = (x - loc) / scale
    return np.log(b) - (b + 1) * np.log(z) - np.log(scale)


def pdf_pareto(x, b, loc=0.0, scale=1.):
    """
    Evaluate the scipy.stats.pareto-style Pareto pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        b: Second shape or upper-bound parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Pareto pdf value without domain checks.

    Examples
    --------
    Pareto density with shape b=2.5:

    >>> import numpy as np
    >>> from kanly.api import pdf_pareto
    >>> pdf_pareto(np.array([1.0, 2.0, 5.0]),
    ...            b=2.5).round(3)                              # doctest: +SKIP
    array([2.5  , 0.221, 0.016])
    """
    z = (x - loc) / scale
    return b / (scale * z ** (b+1))


def logpdf_t(x, df, loc=0., scale=1.):
    """
    Evaluate the scipy.stats.t-style Student's t logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        df: Degrees-of-freedom parameter.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Student's t logpdf value without domain checks.

    Examples
    --------
    Student-t logpdf with 5 degrees of freedom:

    >>> import numpy as np
    >>> from kanly.api import logpdf_t
    >>> logpdf_t(np.array([-1.0, 0.0, 1.0]),
    ...          df=5).round(3)                                  # doctest: +SKIP
    array([-1.654, -1.083, -1.654])
    """
    const = gammaln((df + 1) / 2) - gammaln(df / 2) - 0.5 * np.log(df * np.pi) - np.log(scale)
    z = (x - loc) / scale
    return -((df + 1) / 2) * np.log1p(z * z / df) + const


def pdf_t(x, df, loc=0., scale=1.):
    """
    Evaluate the scipy.stats.t-style Student's t pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        df: Degrees-of-freedom parameter.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Student's t pdf value without domain checks.

    Examples
    --------
    Student-t density with 5 degrees of freedom:

    >>> import numpy as np
    >>> from kanly.api import pdf_t
    >>> pdf_t(np.array([-1.0, 0.0, 1.0]), df=5).round(3)         # doctest: +SKIP
    array([0.219, 0.380, 0.219])
    """
    const = gammaln((df + 1) / 2) - gammaln(df / 2) - 0.5 * np.log(df * np.pi) - np.log(scale)
    z = (x - loc) / scale
    return (1 + z * z / df) ** (-(df + 1) / 2) * np.exp(const)


def logpdf_truncnorm(x, a, b, loc=0., scale=1.):
    """
    Evaluate the scipy.stats.truncnorm-style truncated normal logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        a: First shape or lower-bound parameter, matching scipy.stats naming for this distribution.
        b: Second shape or upper-bound parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        truncated normal logpdf value without domain checks.

    Examples
    --------
    Truncated normal logpdf on standardized truncation bounds [-1, 2]:

    >>> import numpy as np
    >>> from kanly.api import logpdf_truncnorm
    >>> logpdf_truncnorm(np.array([-0.5, 0.0, 1.0]),
    ...                  a=-1.0, b=2.0).round(3)                # doctest: +SKIP
    array([-1.220, -0.970, -1.470])
    """
    const = -0.5 * LOG_2_PI - np.log(scale) - np.log(ndtr(b) - ndtr(a))
    z = (x - loc) / scale
    return -0.5 * z ** 2 + const

def pdf_truncnorm(x, a, b, loc=0., scale=1.):
    """
    Evaluate the scipy.stats.truncnorm-style truncated normal pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        a: First shape or lower-bound parameter, matching scipy.stats naming for this distribution.
        b: Second shape or upper-bound parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        truncated normal pdf value without domain checks.

    Examples
    --------
    Truncated normal density on standardized truncation bounds [-1, 2]:

    >>> import numpy as np
    >>> from kanly.api import pdf_truncnorm
    >>> pdf_truncnorm(np.array([-0.5, 0.0, 1.0]),
    ...               a=-1.0, b=2.0).round(3)                   # doctest: +SKIP
    array([0.295, 0.379, 0.230])
    """
    z = (x - loc) / scale
    return np.exp(-0.5 * z ** 2) / (
        scale * RT_2_PI * (ndtr(b) - ndtr(a))
    )


def logpdf_weibull_min(x, c, loc=0., scale=1.):
    """
    Evaluate the scipy.stats.weibull_min-style Weibull minimum logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        c: Shape parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Weibull minimum logpdf value without domain checks.

    Examples
    --------
    Weibull-min logpdf with shape c=1.5:

    >>> import numpy as np
    >>> from kanly.api import logpdf_weibull_min
    >>> logpdf_weibull_min(np.array([0.5, 1.0, 2.0]),
    ...                    c=1.5).round(3)                       # doctest: +SKIP
    array([-0.701, -0.595, -3.181])
    """
    const = np.log(c / scale)
    z = (x - loc) / scale
    return const + (c - 1) * np.log(z) - z ** c


def pdf_weibull_min(x, c, loc=0., scale=1.):
    """
    Evaluate the scipy.stats.weibull_min-style Weibull minimum pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        c: Shape parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Weibull minimum pdf value without domain checks.

    Examples
    --------
    Weibull-min density with shape c=1.5:

    >>> import numpy as np
    >>> from kanly.api import pdf_weibull_min
    >>> pdf_weibull_min(np.array([0.5, 1.0, 2.0]),
    ...                 c=1.5).round(3)                          # doctest: +SKIP
    array([0.496, 0.552, 0.042])
    """
    z = (x - loc) / scale
    return c * z ** (c-1) * np.exp(-z**c) / scale


def logpdf_multivariate_normal(x, mean=None, cov=None, tau=None):

    """
    Evaluate the scipy.stats.multivariate_normal-style multivariate normal logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        mean: Mean vector.
        cov: Covariance matrix.
        tau: Precision matrix, the inverse covariance matrix.

    Returns:
        multivariate normal logpdf value without domain checks.

    Examples
    --------
    Multivariate normal logpdf with a covariance matrix:

    >>> import numpy as np
    >>> from kanly.api import logpdf_multivariate_normal
    >>> mean = np.array([1., 0., 0.])
    >>> cov  = 2.0 * np.eye(3)
    >>> logpdf_multivariate_normal(np.array([1., 2., 3.]),
    ...                            mean=mean, cov=cov).round(3)  # doctest: +SKIP
    -6.005

    The precision matrix can be supplied via ``tau`` instead of ``cov``.
    """
    k = len(x)
    x = np.asarray(x)
    if mean is None:
        mean = np.zeros(k)
    if tau is None:
        if cov is None:
            tau = np.eye(k)
        else:
            tau = np.linalg.pinv(cov)

    x_demeaned = x - mean
    quad = np.dot(np.dot(x_demeaned, tau), x_demeaned)

    return -0.5 * quad - (k / 2) * LOG_2_PI + 0.5 * np.linalg.slogdet(tau)[1]


def pdf_multivariate_normal(x, mean=None, cov=None, tau=None):

    """
    Evaluate the scipy.stats.multivariate_normal-style multivariate normal pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        mean: Mean vector.
        cov: Covariance matrix.
        tau: Precision matrix, the inverse covariance matrix.

    Returns:
        multivariate normal pdf value without domain checks.

    Examples
    --------
    Multivariate normal density at a point with isotropic covariance:

    >>> import numpy as np
    >>> from kanly.api import pdf_multivariate_normal
    >>> pdf_multivariate_normal(np.array([0., 0., 0.]),
    ...                         cov=np.eye(3)).round(4)          # doctest: +SKIP
    0.0635
    """
    k = len(x)
    x = np.asarray(x)
    if mean is None:
        mean = np.zeros(k)
    if tau is None:
        if cov is None:
            tau = np.eye(k)
        else:
            tau = np.linalg.pinv(cov)

    x_demeaned = x - mean
    quad = np.dot(np.dot(x_demeaned, tau), x_demeaned)

    return np.exp(-0.5 * quad) / np.sqrt((2 * PI)**k / np.linalg.det(tau))


def logpdf_multivariate_t(x, df, mean=None, shape=None, tau=None):
    """
    Evaluate the scipy.stats.multivariate_t-style multivariate Student's t logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        df: Degrees-of-freedom parameter.
        mean: Mean vector.
        shape: Shape matrix for the multivariate t distribution.
        tau: Precision matrix, the inverse covariance matrix.

    Returns:
        multivariate Student's t logpdf value without domain checks.

    Examples
    --------
    Multivariate-t logpdf with df=4 and a shape matrix:

    >>> import numpy as np
    >>> from kanly.api import logpdf_multivariate_t
    >>> shape = 2.0 * np.eye(4)
    >>> logpdf_multivariate_t(np.array([0., 1., 2., 3.]),
    ...                       df=4, shape=shape).round(3)        # doctest: +SKIP
    -7.846
    """
    p = len(x)
    x = np.asarray(x)
    if mean is None:
        mean = np.zeros(p)
    if tau is None:
        if shape is None:
            tau = np.eye(p)
        else:
            tau = np.linalg.pinv(shape)

    x_demeaned = x - mean
    quad = np.dot(np.dot(x_demeaned, tau), x_demeaned)

    return (
            -(df + p) / 2 * np.log(1 + quad / df)
            + scsp.gammaln((df + p) / 2)
            - scsp.gammaln(df / 2)
            - p / 2 * (np.log(df) + LOG_PI)
            + 0.5 * np.linalg.slogdet(tau)[1]
    )


def pdf_multivariate_t(x, df, mean=None, shape=None, tau=None):
    """
    Evaluate the scipy.stats.multivariate_t-style multivariate Student's t pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        df: Degrees-of-freedom parameter.
        mean: Mean vector.
        shape: Shape matrix for the multivariate t distribution.
        tau: Precision matrix, the inverse covariance matrix.

    Returns:
        multivariate Student's t pdf value without domain checks.

    Examples
    --------
    Multivariate-t density with df=4 and a shape matrix:

    >>> import numpy as np
    >>> from kanly.api import pdf_multivariate_t
    >>> shape = 2.0 * np.eye(4)
    >>> pdf_multivariate_t(np.array([0., 1., 2., 3.]),
    ...                    df=4, shape=shape).round(5)           # doctest: +SKIP
    0.00039
    """
    p = len(x)
    x = np.asarray(x)
    if mean is None:
        mean = np.zeros(p)
    if tau is None:
        if shape is None:
            tau = np.eye(p)
        else:
            tau = np.linalg.pinv(shape)

    x_demeaned = x - mean
    quad = np.dot(np.dot(x_demeaned, tau), x_demeaned)

    return (
        (1 + quad / df) ** (-(df + p)/2)
        * gamma((df+p)/2)
        * np.sqrt(np.linalg.det(tau))
        / (
            (df * PI) ** (p/2)
            * gamma(df/2)
        )
    )


# convenience methods for testing, not actually needed
# anymore due to overloads
nopython_logpdf_beta = njit(logpdf_beta, cache=True)
nopython_logpdf_cauchy = njit(logpdf_cauchy, cache=True)
nopython_logpdf_chi2 = njit(logpdf_chi2, cache=True)
nopython_logpdf_expon = njit(logpdf_expon, cache=True)
nopython_logpdf_f = njit(logpdf_f, cache=True)
nopython_logpdf_gamma = njit(logpdf_gamma, cache=True)
nopython_logpdf_genextreme = njit(logpdf_genextreme, cache=True)
nopython_logpdf_halfcauchy = njit(logpdf_halfcauchy, cache=True)
nopython_logpdf_halfnorm = njit(logpdf_halfnorm, cache=True)
nopython_logpdf_invgamma = njit(logpdf_invgamma, cache=True)
nopython_logpdf_laplace = njit(logpdf_laplace, cache=True)
nopython_logpdf_logistic = njit(logpdf_logistic, cache=True)
nopython_logpdf_lognorm = njit(logpdf_lognorm, cache=True)
nopython_logpdf_norm = njit(logpdf_norm, cache=True)
nopython_logpdf_multivariate_normal = njit(logpdf_multivariate_normal, cache=True)
nopython_logpdf_pareto = njit(logpdf_pareto, cache=True)
nopython_logpdf_t = njit(logpdf_t, cache=True)
nopython_logpdf_multivariate_t = njit(logpdf_multivariate_t, cache=True)
nopython_logpdf_truncnorm = njit(logpdf_truncnorm, cache=True)
nopython_logpdf_weibull_min = njit(logpdf_weibull_min, cache=True)

nopython_pdf_beta = njit(pdf_beta, cache=True)
nopython_pdf_cauchy = njit(pdf_cauchy, cache=True)
nopython_pdf_chi2 = njit(pdf_chi2, cache=True)
nopython_pdf_expon = njit(pdf_expon, cache=True)
nopython_pdf_f = njit(pdf_f, cache=True)
nopython_pdf_gamma = njit(pdf_gamma, cache=True)
nopython_pdf_genextreme = njit(pdf_genextreme, cache=True)
nopython_pdf_halfcauchy = njit(pdf_halfcauchy, cache=True)
nopython_pdf_halfnorm = njit(pdf_halfnorm, cache=True)
nopython_pdf_invgamma = njit(pdf_invgamma, cache=True)
nopython_pdf_laplace = njit(pdf_laplace, cache=True)
nopython_pdf_logistic = njit(pdf_logistic, cache=True)
nopython_pdf_lognorm = njit(pdf_lognorm, cache=True)
nopython_pdf_multivariate_normal = njit(pdf_multivariate_normal, cache=True)
nopython_pdf_multivariate_t = njit(pdf_multivariate_t, cache=True)
nopython_pdf_norm = njit(pdf_norm, cache=True)
nopython_pdf_pareto = njit(pdf_pareto, cache=True)
nopython_pdf_t = njit(pdf_t, cache=True)
nopython_pdf_truncnorm = njit(pdf_truncnorm, cache=True)
nopython_pdf_weibull_min = njit(pdf_weibull_min, cache=True)


@overload(logpdf_f)
def overload_logpdf_f(x, dfn, dfd, loc=0.0, scale=0.0):
    """
    Register a numba overload for the scipy.stats.f-style logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        dfn: Numerator degrees-of-freedom parameter for the F distribution.
        dfd: Denominator degrees-of-freedom parameter for the F distribution.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the F logpdf.
    """
    return logpdf_f


@overload(logpdf_norm)
def overload_logpdf_norm(x, loc=0., scale=1.):
    """
    Register a numba overload for the scipy.stats.norm-style logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the normal logpdf.
    """
    return logpdf_norm


@overload(logpdf_t)
def overload_logpdf_t(x, df, loc=0., scale=1.):
    """
    Register a numba overload for the scipy.stats.t-style logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        df: Degrees-of-freedom parameter.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the Student's t logpdf.
    """
    return logpdf_t


@overload(logpdf_lognorm)
def overload_logpdf_lognorm(x, s, loc=0., scale=1.):
    """
    Register a numba overload for the scipy.stats.lognorm-style logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        s: Shape parameter for the log-normal distribution.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the log-normal logpdf.
    """
    return logpdf_lognorm


@overload(logpdf_halfnorm)
def overload_logpdf_halfnorm(x, loc=0., scale=1.):
    """
    Register a numba overload for the scipy.stats.halfnorm-style logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the half-normal logpdf.
    """
    return logpdf_halfnorm


@overload(logpdf_beta)
def overload_logpdf_beta(x, a, b, loc=0., scale=1.):
    """
    Register a numba overload for the scipy.stats.beta-style logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        a: First shape or lower-bound parameter, matching scipy.stats naming for this distribution.
        b: Second shape or upper-bound parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the beta logpdf.
    """
    return logpdf_beta


@overload(logpdf_gamma)
def overload_logpdf_gamma(x, a, loc=0., scale=1.):
    """
    Register a numba overload for the scipy.stats.gamma-style logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        a: First shape or lower-bound parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the gamma logpdf.
    """
    return logpdf_gamma


@overload(logpdf_laplace)
def overload_logpdf_laplace(x, loc=0., scale=1.):
    """
    Register a numba overload for the scipy.stats.laplace-style logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the Laplace logpdf.
    """
    return logpdf_laplace


@overload(logpdf_cauchy)
def overload_logpdf_cauchy(x, loc=0., scale=1.):
    """
    Register a numba overload for the scipy.stats.cauchy-style logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the cauchy logpdf.
    """
    return logpdf_cauchy


@overload(logpdf_invgamma)
def overload_logpdf_invgamma(x, a, loc=0., scale=1.):
    """
    Register a numba overload for the scipy.stats.invgamma-style logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        a: First shape or lower-bound parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the inverse-gamma logpdf.
    """
    return logpdf_invgamma


@overload(logpdf_expon)
def overload_logpdf_expon(x, loc=0., scale=1.):
    """
    Register a numba overload for the scipy.stats.expon-style logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the exponential logpdf.
    """
    return logpdf_expon


@overload(logpdf_chi2)
def overload_logpdf_chi2(x, df, loc=0., scale=1.):
    """
    Register a numba overload for the scipy.stats.chi2-style logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        df: Degrees-of-freedom parameter.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the chi-square logpdf.
    """
    return logpdf_chi2


@overload(logpdf_truncnorm)
def overload_logpdf_truncnorm(x, a, b, loc=0., scale=1.):
    """
    Register a numba overload for the scipy.stats.truncnorm-style logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        a: First shape or lower-bound parameter, matching scipy.stats naming for this distribution.
        b: Second shape or upper-bound parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the truncated normal logpdf.
    """
    return logpdf_truncnorm


@overload(logpdf_genextreme)
def overload_logpdf_genextreme(x, c, loc=0., scale=1.):
    """
    Register a numba overload for the scipy.stats.genextreme-style logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        c: Shape parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the generalized extreme value logpdf.
    """
    return logpdf_genextreme


@overload(logpdf_halfcauchy)
def overload_logpdf_halfcauchy(x, loc=0., scale=1.):
    """
    Register a numba overload for the scipy.stats.halfcauchy-style logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the half-Cauchy logpdf.
    """
    return logpdf_halfcauchy


@overload(logpdf_logistic)
def overload_logpdf_logistic(x, loc=0., scale=1.):
    """
    Register a numba overload for the scipy.stats.logistic-style logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the logistic logpdf.
    """
    return logpdf_logistic


@overload(logpdf_pareto)
def overload_logpdf_pareto(x, b, loc=0., scale=1.):
    """
    Register a numba overload for the scipy.stats.pareto-style logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        b: Second shape or upper-bound parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the Pareto logpdf.
    """
    return logpdf_pareto


@overload(logpdf_weibull_min)
def overload_logpdf_weibull_min(x, c, loc=0., scale=1.):
    """
    Register a numba overload for the scipy.stats.weibull_min-style logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        c: Shape parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the Weibull minimum logpdf.
    """
    return logpdf_weibull_min


@overload(logpdf_multivariate_normal)
def overload_logpdf_multivariate_normal(x, mean=None, cov=None, tau=None):
    """
    Register a numba overload for the scipy.stats.multivariate_normal-style logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        mean: Mean vector.
        cov: Covariance matrix.
        tau: Precision matrix, the inverse covariance matrix.

    Returns:
        Callable implementation for the multivariate normal logpdf.
    """
    if tau is None and cov is not None:

        def temp(x, mean=None, cov=None, tau=None):
            """
            Numba-compatible specialization for the selected covariance parameterization.

            Args:
                x: Point or array of points at which to evaluate the function.
                mean: Mean vector.
                cov: Covariance matrix.
                tau: Precision matrix, the inverse covariance matrix.

            Returns:
                Log-density value for the selected multivariate distribution.
            """
            k = len(x)
            x = np.asarray(x)
            if mean is None:
                mean = np.zeros(k)
            tau = np.linalg.pinv(cov)
            x_demeaned = x - mean
            quad = np.dot(np.dot(x_demeaned, tau), x_demeaned)

            return -0.5 * quad - (k / 2) * LOG_2_PI + 0.5 * np.linalg.slogdet(tau)[1]

        return temp

    elif tau is not None and cov is None:

        def temp(x, mean=None, cov=None, tau=None):
            """
            Numba-compatible specialization for the selected covariance parameterization.

            Args:
                x: Point or array of points at which to evaluate the function.
                mean: Mean vector.
                cov: Covariance matrix.
                tau: Precision matrix, the inverse covariance matrix.

            Returns:
                Log-density value for the selected multivariate distribution.
            """
            k = len(x)
            x = np.asarray(x)
            if mean is None:
                mean = np.zeros(k)
            x_demeaned = x - mean
            quad = np.dot(np.dot(x_demeaned, tau), x_demeaned)

            return -0.5 * quad - (k / 2) * LOG_2_PI + 0.5 * np.linalg.slogdet(tau)[1]

        return temp

    elif tau is None and cov is None:

        def temp(x, mean=None, cov=None, tau=None):
            """
            Numba-compatible specialization for the selected covariance parameterization.

            Args:
                x: Point or array of points at which to evaluate the function.
                mean: Mean vector.
                cov: Covariance matrix.
                tau: Precision matrix, the inverse covariance matrix.

            Returns:
                Log-density value for the selected multivariate distribution.
            """
            k = len(x)
            x = np.asarray(x)
            if mean is None:
                mean = np.zeros(k)
            x_demeaned = x - mean

            tau = np.eye(k)
            quad = np.dot(x_demeaned, x_demeaned)

            return -0.5 * quad - (k / 2) * LOG_2_PI

        return temp

    elif tau is not None and cov is not None:
        raise Exception("cannot supply both `tau` and `cov`")



@overload(logpdf_multivariate_t)
def overload_logpdf_multivariate_t(x, df, mean=None, shape=None, tau=None):
    """
    Register a numba overload for the scipy.stats.multivariate_t-style logpdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        df: Degrees-of-freedom parameter.
        mean: Mean vector.
        shape: Shape matrix for the multivariate t distribution.
        tau: Precision matrix, the inverse covariance matrix.

    Returns:
        Callable implementation for the multivariate Student's t logpdf.
    """
    if tau is None and shape is not None:

        def temp(x, df, mean=None, shape=None, tau=None):
            """
            Numba-compatible specialization for the selected covariance parameterization.

            Args:
                x: Point or array of points at which to evaluate the function.
                df: Degrees-of-freedom parameter.
                mean: Mean vector.
                shape: Shape matrix for the multivariate t distribution.
                tau: Precision matrix, the inverse covariance matrix.

            Returns:
                Log-density value for the selected multivariate distribution.
            """
            p = len(x)
            x = np.asarray(x)
            if mean is None:
                mean = np.zeros(p)
            tau = np.linalg.pinv(shape)

            x_demeaned = x - mean
            quad = np.dot(np.dot(x_demeaned, tau), x_demeaned)

            return (
                    -(df + p) / 2 * np.log(1 + quad / df)
                    + scsp.gammaln((df + p) / 2)
                    - scsp.gammaln(df / 2)
                    - p / 2 * (np.log(df) + LOG_PI)
                    + 0.5 * np.linalg.slogdet(tau)[1]
            )

        return temp

    elif tau is not None and shape is None:

        def temp(x, df, mean=None, shape=None, tau=None):
            """
            Numba-compatible specialization for the selected covariance parameterization.

            Args:
                x: Point or array of points at which to evaluate the function.
                df: Degrees-of-freedom parameter.
                mean: Mean vector.
                shape: Shape matrix for the multivariate t distribution.
                tau: Precision matrix, the inverse covariance matrix.

            Returns:
                Log-density value for the selected multivariate distribution.
            """
            p = len(x)
            x = np.asarray(x)
            if mean is None:
                mean = np.zeros(p)

            x_demeaned = x - mean
            quad = np.dot(np.dot(x_demeaned, tau), x_demeaned)

            return (
                    -(df + p) / 2 * np.log(1 + quad / df)
                    + scsp.gammaln((df + p) / 2)
                    - scsp.gammaln(df / 2)
                    - p / 2 * (np.log(df) + LOG_PI)
                    + 0.5 * np.linalg.slogdet(tau)[1]
            )

        return temp

    elif tau is None and shape is None:

        def temp(x, df, mean=None, cov=None, tau=None):
            """
            Numba-compatible specialization for the selected covariance parameterization.

            Args:
                x: Point or array of points at which to evaluate the function.
                df: Degrees-of-freedom parameter.
                mean: Mean vector.
                cov: Covariance matrix.
                tau: Precision matrix, the inverse covariance matrix.

            Returns:
                Log-density value for the selected multivariate distribution.
            """
            p = len(x)
            x = np.asarray(x)
            if mean is None:
                mean = np.zeros(p)
            tau = np.eye(p)

            x_demeaned = x - mean
            quad = np.dot(x_demeaned, x_demeaned)

            return (
                    -(df + p) / 2 * np.log(1 + quad / df)
                    + scsp.gammaln((df + p) / 2)
                    - scsp.gammaln(df / 2)
                    - p / 2 * (np.log(df) + LOG_PI)
            )

        return temp

    elif tau is not None and shape is not None:
        raise Exception("cannot supply both `tau` and `cov`")


@overload(pdf_beta)
def overload_pdf_beta(x, a, b, loc=0.0, scale=1.0):
    """
    Register a numba overload for the scipy.stats.beta-style pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        a: First shape or lower-bound parameter, matching scipy.stats naming for this distribution.
        b: Second shape or upper-bound parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the beta pdf.
    """
    return pdf_beta


@overload(pdf_cauchy)
def overload_pdf_cauchy(x, loc=0.0, scale=1.0):
    """
    Register a numba overload for the scipy.stats.cauchy-style pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the cauchy pdf.
    """
    return pdf_cauchy


@overload(pdf_chi2)
def overload_pdf_chi2(x, df, loc=0.0, scale=1.0):
    """
    Register a numba overload for the scipy.stats.chi2-style pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        df: Degrees-of-freedom parameter.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the chi-square pdf.
    """
    return pdf_chi2


@overload(pdf_expon)
def overload_pdf_expon(x, loc=0.0, scale=1.0):
    """
    Register a numba overload for the scipy.stats.expon-style pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the exponential pdf.
    """
    return pdf_expon


@overload(pdf_f)
def overload_pdf_f(x, dfn, dfd, loc=0.0, scale=0.0):
    """
    Register a numba overload for the scipy.stats.f-style pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        dfn: Numerator degrees-of-freedom parameter for the F distribution.
        dfd: Denominator degrees-of-freedom parameter for the F distribution.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the F pdf.
    """
    return pdf_f


@overload(pdf_gamma)
def overload_pdf_gamma(x, a, loc=0.0, scale=1.0):
    """
    Register a numba overload for the scipy.stats.gamma-style pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        a: First shape or lower-bound parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the gamma pdf.
    """
    return pdf_gamma


@overload(pdf_genextreme)
def overload_pdf_genextreme(x, c, loc=0.0, scale=1.0):
    """
    Register a numba overload for the scipy.stats.genextreme-style pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        c: Shape parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the generalized extreme value pdf.
    """
    return pdf_genextreme


@overload(pdf_halfcauchy)
def overload_pdf_halfcauchy(x, loc=0.0, scale=1.0):
    """
    Register a numba overload for the scipy.stats.halfcauchy-style pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the half-Cauchy pdf.
    """
    return pdf_halfcauchy


@overload(pdf_halfnorm)
def overload_pdf_halfnorm(x, loc=0.0, scale=1.0):
    """
    Register a numba overload for the scipy.stats.halfnorm-style pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the half-normal pdf.
    """
    return pdf_halfnorm


@overload(pdf_invgamma)
def overload_pdf_invgamma(x, a, loc=0.0, scale=1.0):
    """
    Register a numba overload for the scipy.stats.invgamma-style pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        a: First shape or lower-bound parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the inverse-gamma pdf.
    """
    return pdf_invgamma


@overload(pdf_laplace)
def overload_pdf_laplace(x, loc=0.0, scale=1.0):
    """
    Register a numba overload for the scipy.stats.laplace-style pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the Laplace pdf.
    """
    return pdf_laplace


@overload(pdf_logistic)
def overload_pdf_logistic(x, loc=0.0, scale=1.0):
    """
    Register a numba overload for the scipy.stats.logistic-style pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the logistic pdf.
    """
    return pdf_logistic


@overload(pdf_lognorm)
def overload_pdf_lognorm(x, s, loc=0.0, scale=1.0):
    """
    Register a numba overload for the scipy.stats.lognorm-style pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        s: Shape parameter for the log-normal distribution.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the log-normal pdf.
    """
    return pdf_lognorm


@overload(pdf_multivariate_normal)
def overload_pdf_multivariate_normal(x, mean=None, cov=None, tau=None):
    """
    Register a numba overload for the scipy.stats.multivariate_normal-style pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        mean: Mean vector.
        cov: Covariance matrix.
        tau: Precision matrix, the inverse covariance matrix.

    Returns:
        Callable implementation for the multivariate normal pdf.
    """
    return pdf_multivariate_normal


@overload(pdf_multivariate_t)
def overload_pdf_multivariate_t(x, df, mean=None, shape=None, tau=None):
    """
    Register a numba overload for the scipy.stats.multivariate_t-style pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        df: Degrees-of-freedom parameter.
        mean: Mean vector.
        shape: Shape matrix for the multivariate t distribution.
        tau: Precision matrix, the inverse covariance matrix.

    Returns:
        Callable implementation for the multivariate Student's t pdf.
    """
    return pdf_multivariate_t


@overload(pdf_norm)
def overload_pdf_norm(x, loc=0.0, scale=1.0):
    """
    Register a numba overload for the scipy.stats.norm-style pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the normal pdf.
    """
    return pdf_norm


@overload(pdf_pareto)
def overload_pdf_pareto(x, b, loc=0.0, scale=1.0):
    """
    Register a numba overload for the scipy.stats.pareto-style pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        b: Second shape or upper-bound parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the Pareto pdf.
    """
    return pdf_pareto


@overload(pdf_t)
def overload_pdf_t(x, df, loc=0.0, scale=1.0):
    """
    Register a numba overload for the scipy.stats.t-style pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        df: Degrees-of-freedom parameter.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the Student's t pdf.
    """
    return pdf_t


@overload(pdf_truncnorm)
def overload_pdf_truncnorm(x, a, b, loc=0.0, scale=1.0):
    """
    Register a numba overload for the scipy.stats.truncnorm-style pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        a: First shape or lower-bound parameter, matching scipy.stats naming for this distribution.
        b: Second shape or upper-bound parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the truncated normal pdf.
    """
    return pdf_truncnorm


@overload(pdf_weibull_min)
def overload_pdf_weibull_min(x, c, loc=0.0, scale=1.0):
    """
    Register a numba overload for the scipy.stats.weibull_min-style pdf.

    Args:
        x: Point or array of points at which to evaluate the function.
        c: Shape parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.

    Returns:
        Callable implementation for the Weibull minimum pdf.
    """
    return pdf_weibull_min



local_keys = list(locals().keys())
NOPYTHON_LOGPDF_IMPORT_STRING = []
for k in local_keys:
    if k[:7] == 'logpdf_' or k[:16] == 'nopython_logpdf_' or k[:4] == 'pdf_' or k[:13] == 'nopython_pdf_':
        NOPYTHON_LOGPDF_IMPORT_STRING.append(f'from kanly.stats.distributions.nopython_logpdf import {k}')
NOPYTHON_LOGPDF_IMPORT_STRING = '\n'.join(sorted(NOPYTHON_LOGPDF_IMPORT_STRING))

# for k in local_keys:
#     if 'nopython_logpdf' in k:
#         print(f"'{k.replace('nopython_logpdf_', '')}': {k},")

NAME_2_NOPYTHON_LOGPDF = {
    'beta': nopython_logpdf_beta,
    'cauchy': nopython_logpdf_cauchy,
    'chi2': nopython_logpdf_chi2,
    'expon': nopython_logpdf_expon,
    'f': nopython_logpdf_f,
    'gamma': nopython_logpdf_gamma,
    'genextreme': nopython_logpdf_genextreme,
    'halfcauchy': nopython_logpdf_halfcauchy,
    'halfnorm': nopython_logpdf_halfnorm,
    'invgamma': nopython_logpdf_invgamma,
    'laplace': nopython_logpdf_laplace,
    'logistic': nopython_logpdf_logistic,
    'lognorm': nopython_logpdf_lognorm,
    'norm': nopython_logpdf_norm,
    'multivariate_normal': nopython_logpdf_multivariate_normal,
    'pareto': nopython_logpdf_pareto,
    't': nopython_logpdf_t,
    'multivariate_t': nopython_logpdf_multivariate_t,
    'truncnorm': nopython_logpdf_truncnorm,
    'weibull_min': nopython_logpdf_weibull_min,
}

# @njit
# def f():
#     return logpdf_multivariate_normal(np.array([1., 2, 3]), mean=np.array([1., 0, 0]), cov=2 * np.eye(3))
#
# print(f())
# @njit
# def f2():
#     return logpdf_multivariate_normal(np.array([1., 2, 3]), mean=np.array([1., 0, 0]), tau=np.linalg.inv(2 * np.eye(3)))
#
# print(f2())
#
# from scipy.stats import multivariate_normal
#
# print(multivariate_normal(mean=[1, 0, 0], cov=2 * np.eye(3)).logpdf([1., 2, 3]))
#
# print()
#
# @njit
# def f3():
#     return logpdf_multivariate_normal(np.array([1., 2, 3]))
#
# print(f3())
#
# from scipy.stats import multivariate_normal
#
# print(multivariate_normal(mean=[0, 0, 0]).logpdf([1., 2, 3]))

#
# from scipy.stats import multivariate_t
#
# print(multivariate_t(df=4, shape=2 * np.eye(4)).logpdf([0, 1., 2, 3]))
# print(logpdf_multivariate_t([0, 1., 2, 3], df=4, shape=2 * np.eye(4)))
#
# @njit
# def f5():
#     print(logpdf_multivariate_t([0, 1., 2, 3], df=4, shape=2 * np.eye(4)))
#
# f5()
#
#
# @njit
# def f6():
#     print(logpdf_multivariate_t([0, 1., 2, 3], df=4, tau=np.linalg.inv(2 * np.eye(4))))
#
#
# f6()

# import inspect
# D = list(locals().keys())
# D = sorted([z for z in D if 'logpdf_' in z and '_logpdf_' not in z])
# for z in D:
#     print(f'{z}{inspect.signature(locals()[z])}')
#     s = inspect.getdoc(locals()[z])
#     if s is not None:
#         print('\t'+'\n\t'.join(s.split('\n')))
