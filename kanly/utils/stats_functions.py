"""
Functions proportional to some stats functions,
used in `function_str_to_callable`
"""
from __future__ import absolute_import, print_function

import numpy as np
from numba import njit, vectorize

import scipy.special as sp


# @njit(cache=True)
# def erf(x):
#     """
#     The algorithm comes from Handbook of Mathematical Functions, formula 7.1.26.
#     "Handbook of Mathematical Functions: with Formulas, Graphs, and Mathematical Tables"
#     Abramowitz et al
#     """
#     # save the sign of x
#     sign = np.sign(x)
#     x = np.abs(x)
#
#     # constants
#     a1 = 0.254829592
#     a2 = -0.284496736
#     a3 = 1.421413741
#     a4 = -1.453152027
#     a5 = 1.061405429
#     p = 0.3275911
#
#     # A&S formula 7.1.26
#     t = 1.0 / (1.0 + p * x)
#     y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x ** 2)
#
#     return sign * y
#
#
# @njit(cache=True)
# def erfc(x):
#     return 1.0 - erf(x)


@vectorize(cache=True)
def std_normal_cdf(x):
    """CDF of the standard normal distribution N(0, 1), vectorised via Numba.

    Computes ``Φ(x) = (erf(x / √2) + 1) / 2`` using the ``scipy.special.erf``
    function (available inside Numba kernels via ``numba_scipy``).

    Args:
        x: Scalar or array of real values.

    Returns:
        CDF value(s) in ``[0, 1]``.
    """
    z = x / np.sqrt(2)
    return (sp.erf(z) + 1) / 2


@njit(cache=True)
def normal_cdf(x, loc=0, scale=1):
    """CDF of a normal distribution N(loc, scale²), JIT-compiled via Numba.

    Standardises ``x`` and delegates to ``std_normal_cdf``.

    Args:
        x: Scalar or array of real values.
        loc: Mean of the distribution.  Defaults to ``0``.
        scale: Standard deviation of the distribution.  Defaults to ``1``.

    Returns:
        CDF value(s) in ``[0, 1]``.
    """
    z = (x - loc) / scale
    return std_normal_cdf(z)


@njit(cache=True)
def normal_pdf(x, loc=0, scale=0):
    """PDF of a normal distribution N(loc, scale²), JIT-compiled via Numba.

    Args:
        x: Scalar or array of real values.
        loc: Mean of the distribution.  Defaults to ``0``.
        scale: Standard deviation of the distribution.  Defaults to ``0``
            (caller must supply a positive value).

    Returns:
        Probability density value(s); always non-negative.
    """
    return np.exp(-0.5 * ((x - loc) / scale) ** 2) / (np.sqrt(np.pi * 2) * scale)


@njit(cache=True)
def normal_logpdf(x, loc=0, scale=0):
    """Log-PDF of a normal distribution N(loc, scale²), JIT-compiled via Numba.

    Args:
        x: Scalar or array of real values.
        loc: Mean of the distribution.  Defaults to ``0``.
        scale: Standard deviation of the distribution.  Defaults to ``0``
            (caller must supply a positive value).

    Returns:
        Log probability density value(s); always ≤ 0.
    """
    return -0.5 * ((x - loc) / scale) ** 2 - 0.5 * np.log(np.pi * 2) - np.log(scale)


@njit(cache=True)
def log_normal_pdf(x, s=1, scale=1, loc=0):
    """PDF of a log-normal distribution, JIT-compiled via Numba.

    Parameterisation follows ``scipy.stats.lognorm``:
    ``s`` is the shape (σ of the underlying normal), ``scale = exp(μ)``, and
    ``loc`` is a shift applied to ``x`` before taking the log.

    The underlying normal has mean ``log(s)`` and standard deviation ``scale``.

    Args:
        x: Scalar or array of values > ``loc``.
        s: Shape parameter (``σ`` of the underlying normal).  Defaults to ``1``.
        scale: Scale parameter (``exp(μ)`` of the underlying normal).
            Defaults to ``1``.
        loc: Location (shift) parameter; ``x - loc`` is the un-shifted value.
            Defaults to ``0``.

    Returns:
        Probability density value(s); always non-negative.
    """
    mu = np.log(s)
    z = x - loc
    return 1 / (z * np.sqrt(2 * np.pi) * scale) * np.exp(-0.5 * ((np.log(z) - mu) / scale) ** 2)


@njit(cache=True)
def log_normal_logpdf(x, s=1, scale=1, loc=0):
    """Log-PDF of a log-normal distribution, JIT-compiled via Numba.

    Same parameterisation as ``log_normal_pdf``.

    Args:
        x: Scalar or array of values > ``loc``.
        s: Shape parameter (``σ`` of the underlying normal).  Defaults to ``1``.
        scale: Scale parameter (``exp(μ)``).  Defaults to ``1``.
        loc: Location (shift) parameter.  Defaults to ``0``.

    Returns:
        Log probability density value(s).
    """
    mu = np.log(s)
    z = x - loc
    return -np.log(z * np.sqrt(2 * np.pi) * scale) - 0.5 * ((np.log(z) - mu) / scale) ** 2


@njit(cache=True)
def log_normal_cdf(x, s=1., scale=1., loc=0.):
    """CDF of a log-normal distribution, JIT-compiled via Numba.

    Transforms the input to the underlying normal scale and delegates to
    ``normal_cdf``.

    Args:
        x: Scalar or array of values > ``loc``.
        s: Shape parameter (``σ`` of the underlying normal).  Defaults to ``1``.
        scale: Scale parameter (``exp(μ)``).  Defaults to ``1``.
        loc: Location (shift) parameter.  Defaults to ``0``.

    Returns:
        CDF value(s) in ``[0, 1]``.
    """
    mu = np.log(s)
    z = x - loc
    return normal_cdf(np.log(z), scale=scale, loc=mu)


# if __name__ == '__main__':
#     from scipy.stats import lognorm
#     import matplotlib.pyplot as plt
#
#     xrng = np.linspace(.01, 10, 100)
#     plt.plot(xrng, lognorm.cdf(xrng, loc=0, scale=2, s=2))
#     plt.plot(xrng, log_normal_cdf(xrng, loc=0, scale=2, s=2), ls='--')
#     plt.show()
