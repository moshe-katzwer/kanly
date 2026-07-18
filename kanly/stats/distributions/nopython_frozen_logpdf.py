"""
Factory for generating super-fast computation of log pdf use in priors.
Faster than scipy.stats.norm because there is no arg checks, use at your own risk.

These are frozen distributions, their parameters do not change.  Intended for use in priors
only really.

Compatible with numba when `nopython = True`

These distributions don't do any domain checks
"""
# TODO move to separate module outside `bayes`


from __future__ import absolute_import, print_function

from scipy.special import loggamma, betaln
from scipy.stats import norm
import numpy as np
from numba import njit
from scipy.stats._distn_infrastructure import rv_frozen


LOG_2_PI = np.log(2. * np.pi)
LOG_PI = np.log(np.pi)

NORM = 'norm'
LOGNORM = 'lognorm'
TRUNCNORM = 'truncnorm'
HALFNORM = 'halfnorm'
MULTIVARIATE_NORMAL = 'multivariate_normal'
T = 't'
CAUCHY = 'cauchy'
HALFCAUCHY = 'halfcauchy'
LAPLACE = 'laplace'
PARETO = 'pareto'
LOGISTIC = 'logistic'
GENNORM = 'gennorm'
INVGAMMA = 'invgamma'
GAMMA = 'gamma'
EXPON = 'expon'
BETA = 'beta'
CHI2 = 'chi2'
MULTIVARIATE_T = 'multivariate_t'
UNIFORM = 'uniform'
FLAT = 'flat'
LOGUNIFORM = 'loguniform'
GENEXTREME = 'genextreme'
F = 'f'
WEIBULL_MIN = 'weibull_min'
DIRICHLET = 'dirichlet'

DISTRIBUTIONS = (
    NORM, LOGNORM, TRUNCNORM, HALFNORM, MULTIVARIATE_NORMAL,
    CAUCHY, HALFCAUCHY, LAPLACE, PARETO, LOGISTIC,
    GENNORM, INVGAMMA, GAMMA, EXPON, BETA, CHI2, T, MULTIVARIATE_T,
    UNIFORM, FLAT, LOGUNIFORM, GENEXTREME, F, WEIBULL_MIN
)
MULTIVARIATE_DISTRIBUTIONS = (
    MULTIVARIATE_T, MULTIVARIATE_NORMAL, DIRICHLET
)


class flat(rv_frozen):

    """
    rv_frozen-like flat prior distribution with optional finite support.

    Finite bounds create a proper uniform-like prior; infinite bounds represent an improper flat prior used by Bayesian model helpers.
    """
    def __init__(self, a=-np.inf, b=np.inf):
        """
        Initialize a flat prior distribution.

        Args:
            a: Lower support bound.
            b: Upper support bound.
        """
        self.a = a
        self.b = b
        self.is_proper = np.isfinite(a) and np.isfinite(b)
        self.dist = type('test', (object,), {'name': FLAT})()
        if self.is_proper:
            self.size_support = b - a
            self.pdf_height = 1 / (b - a)
        self.args = (a, b)
        self.kwds = dict()

    def support(self):
        """
        Return the support interval of the flat prior.

        Returns:
            Requested summary or density value for the flat prior.
        """
        return self.a, self.b

    def interval(self, alpha):
        """
        Return a central interval for a proper flat prior.

        Args:
            alpha: Dirichlet concentration vector or interval mass, depending on the function.

        Returns:
            Requested summary or density value for the flat prior.
        """
        if self.is_proper:
            assert 0 <= alpha <= 1
            return (self.ppf((1 - alpha) / 2), self.ppf(1 - (1 - alpha) / 2))
        else:
            raise NotImplementedError("improper flat prior")

    def logpdf(self, x):
        """
        Evaluate the flat prior log-density.

        Args:
            x: Point or array of points at which to evaluate the function.

        Returns:
            Requested summary or density value for the flat prior.
        """
        val = np.log(self.pdf_height) if self.is_proper else 0
        return np.where((x > self.a) & (x < self.b), val, -np.inf)

    def pdf(self, x):
        """
        Evaluate the flat prior density-like value.

        Args:
            x: Point or array of points at which to evaluate the function.

        Returns:
            Requested summary or density value for the flat prior.
        """
        # Preserve existing behavior: this returns the log-height for proper
        # flat priors even though the method is named ``pdf``.
        val = np.log(self.pdf_height) if self.is_proper else 0
        return np.where((x > self.a) & (x < self.b), val, 0)

    def cdf(self, x):
        """
        Evaluate the flat prior CDF.

        Args:
            x: Point or array of points at which to evaluate the function.

        Returns:
            Requested summary or density value for the flat prior.
        """
        if self.is_proper:
            return (x - self.b) / (self.b - self.a)
        else:
            raise NotImplementedError("improper flat prior")

    def ppf(self, x):
        """
        Evaluate the flat prior percent-point function.

        Args:
            x: Point or array of points at which to evaluate the function.

        Returns:
            Requested summary or density value for the flat prior.
        """
        if self.is_proper:
            return x * (self.b - self.a) + self.b
        else:
            raise NotImplementedError("improper flat prior")

    def mean(self):
        """
        Return the mean of a proper flat prior.

        Returns:
            Requested summary or density value for the flat prior.
        """
        if self.is_proper:
            return (self.b + self.a) / 2
        else:
            raise NotImplementedError("improper flat prior")

    def median(self):
        """
        Return the median of a proper flat prior.

        Returns:
            Requested summary or density value for the flat prior.
        """
        if self.is_proper:
            return (self.b + self.a) / 2
        else:
            raise NotImplementedError("improper flat prior")

    def std(self):
        """
        Return the standard deviation of a proper flat prior.

        Returns:
            Requested summary or density value for the flat prior.
        """
        if self.is_proper:
            return 1 / 144 * (self.b - self.a)
        else:
            raise NotImplementedError("improper flat prior")

    def var(self):
        """
        Return the variance of a proper flat prior.

        Returns:
            Requested summary or density value for the flat prior.
        """
        if self.is_proper:
            return 1 / 12 * (self.b - self.a) ** 2
        else:
            raise NotImplementedError("improper flat prior")


def _get_nopython_version(name, nopython, kwarg_dict):
    """
    Build a Python or numba-jitted wrapper around a frozen logpdf kernel.

    Args:
        name: Distribution name key.
        nopython: Whether to return a numba-jitted callable.
        kwarg_dict: Frozen keyword arguments passed to the selected log-density kernel.

    Returns:
        Callable that evaluates the selected frozen logpdf kernel.
    """
    log_pdf_name = f'__{"nopython_" if nopython else ""}frozen_internal_logpdf_{name}'
    _log_pdf = globals()[log_pdf_name]
    args = tuple(kwarg_dict.values())

    def _logpdf_wrapper(x):
        """
        Evaluate the selected frozen logpdf kernel with captured parameters.

        Args:
            x: Point or array of points at which to evaluate the log-density.

        Returns:
            Frozen log-density value.
        """
        return _log_pdf(x, *args)

    func = njit(_logpdf_wrapper, cache=True) if nopython else _logpdf_wrapper
    func.__name__ = f'{log_pdf_name}({", ".join([f"{k}={v}" for k, v in kwarg_dict.items()])})'

    return func


def __frozen_internal_logpdf_genextreme(x, c, loc=0.0, scale=0.0, const=0.0):
    """
    Evaluate a frozen scipy.stats.genextreme-style generalized extreme value log-density kernel.

    Args:
        x: Point or array of points at which to evaluate the function.
        c: Shape parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.
        const: Precomputed additive log-density constant for frozen parameters.

    Returns:
        Log-density value using frozen parameters and a precomputed constant.
    """
    z = (x - loc) / scale
    if c == 0:
        return -np.exp(-z) - z + const
    else:
        return -(1 - c * z) ** (1 / c) + (1 / c - 1) * np.log(1 - c * z) + const


def __frozen_internal_logpdf_norm(x, loc=0.0, scale=1.0, const=0):
    """
    Evaluate a frozen scipy.stats.norm-style normal log-density kernel.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.
        const: Precomputed additive log-density constant for frozen parameters.

    Returns:
        Log-density value using frozen parameters and a precomputed constant.
    """
    val = -0.5 * ((x - loc) / scale) ** 2 + const
    return val


def __frozen_internal_logpdf_truncnorm(x, loc=0.0, scale=1.0, const=0):
    """
    Evaluate a frozen scipy.stats.truncnorm-style truncated normal log-density kernel.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.
        const: Precomputed additive log-density constant for frozen parameters.

    Returns:
        Log-density value using frozen parameters and a precomputed constant.
    """
    return -0.5 * ((x - loc) / scale) ** 2 + const


def __frozen_internal_logpdf_beta(x, a, b, loc=0.0, scale=1.0, const=0):
    """
    Evaluate a frozen scipy.stats.beta-style beta log-density kernel.

    Args:
        x: Point or array of points at which to evaluate the function.
        a: First shape or lower-bound parameter, matching scipy.stats naming for this distribution.
        b: Second shape or upper-bound parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.
        const: Precomputed additive log-density constant for frozen parameters.

    Returns:
        Log-density value using frozen parameters and a precomputed constant.
    """
    z = (x - loc) / scale
    return (a - 1) * np.log(z) + (b - 1) * np.log(1 - z) + const


def __frozen_internal_logpdf_cauchy(x, loc=0.0, scale=1.0, const=0):
    """
    Evaluate a frozen scipy.stats.cauchy-style cauchy log-density kernel.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.
        const: Precomputed additive log-density constant for frozen parameters.

    Returns:
        Log-density value using frozen parameters and a precomputed constant.
    """
    z = (x - loc) / scale
    return -np.log(1 + z * z) + const


def __frozen_internal_logpdf_laplace(x, loc=0.0, scale=1.0, const=0):
    """
    Evaluate a frozen scipy.stats.laplace-style Laplace log-density kernel.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.
        const: Precomputed additive log-density constant for frozen parameters.

    Returns:
        Log-density value using frozen parameters and a precomputed constant.
    """
    z = (x - loc) / scale
    return -np.abs(z) + const


def __frozen_internal_logpdf_expon(x, loc=0.0, scale=1.0, const=0):
    """
    Evaluate a frozen scipy.stats.expon-style exponential log-density kernel.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.
        const: Precomputed additive log-density constant for frozen parameters.

    Returns:
        Log-density value using frozen parameters and a precomputed constant.
    """
    v = -(x - loc) / scale + const
    return v


def __frozen_internal_logpdf_t(x, df, loc=0.0, scale=1.0, const=0.0):
    """
    Evaluate a frozen scipy.stats.t-style Student's t log-density kernel.

    Args:
        x: Point or array of points at which to evaluate the function.
        df: Degrees-of-freedom parameter.
        loc: Location parameter.
        scale: Scale parameter.
        const: Precomputed additive log-density constant for frozen parameters.

    Returns:
        Log-density value using frozen parameters and a precomputed constant.
    """
    return (
            - (df + 1.) / 2. * np.log1p((x - loc) ** 2. / (df * scale ** 2)) + const
    )


def __frozen_internal_logpdf_gamma(x, a, loc=0.0, scale=1.0, const=0.0):
    """
    Evaluate a frozen scipy.stats.gamma-style gamma log-density kernel.

    Args:
        x: Point or array of points at which to evaluate the function.
        a: First shape or lower-bound parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.
        const: Precomputed additive log-density constant for frozen parameters.

    Returns:
        Log-density value using frozen parameters and a precomputed constant.
    """
    z = (x - loc) / scale
    return (a - 1) * np.log(z) - z + const


def __frozen_internal_logpdf_lognorm(x, s=1.0, loc=0.0, scale=1.0, const=0):
    """
    Evaluate a frozen scipy.stats.lognorm-style log-normal log-density kernel.

    Args:
        x: Point or array of points at which to evaluate the function.
        s: Shape parameter for the log-normal distribution.
        loc: Location parameter.
        scale: Scale parameter.
        const: Precomputed additive log-density constant for frozen parameters.

    Returns:
        Log-density value using frozen parameters and a precomputed constant.
    """
    z = (x - loc) / scale
    return -0.5 * (np.log(z) / s) ** 2 - np.log(z) + const


def __frozen_internal_logpdf_invgamma(x, a, loc=0.0, scale=1.0, const=0.0):
    """
    Evaluate a frozen scipy.stats.invgamma-style inverse-gamma log-density kernel.

    Args:
        x: Point or array of points at which to evaluate the function.
        a: First shape or lower-bound parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.
        const: Precomputed additive log-density constant for frozen parameters.

    Returns:
        Log-density value using frozen parameters and a precomputed constant.
    """
    z = (x - loc) / scale
    return -(a + 1) * np.log(z) - 1 / z + const


def __frozen_internal_logpdf_logistic(x, loc=0.0, scale=1.0, const=0.0):
    """
    Evaluate a frozen scipy.stats.logistic-style logistic log-density kernel.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.
        const: Precomputed additive log-density constant for frozen parameters.

    Returns:
        Log-density value using frozen parameters and a precomputed constant.
    """
    z = (x - loc) / scale
    return -z - 2 * np.log(1 + np.exp(-z)) + const


def __frozen_internal_logpdf_chi2(x, df, loc=0.0, scale=1.0, const=0.0):
    """
    Evaluate a frozen scipy.stats.chi2-style chi-square log-density kernel.

    Args:
        x: Point or array of points at which to evaluate the function.
        df: Degrees-of-freedom parameter.
        loc: Location parameter.
        scale: Scale parameter.
        const: Precomputed additive log-density constant for frozen parameters.

    Returns:
        Log-density value using frozen parameters and a precomputed constant.
    """
    z = (x - loc) / scale
    return (df / 2 - 1) * np.log(z) - z / 2 + const


def __frozen_internal_logpdf_gennorm(x, beta, loc=0.0, scale=1.0, const=0.0):
    """
    Evaluate a frozen scipy.stats.gennorm-style generalized normal log-density kernel.

    Args:
        x: Point or array of points at which to evaluate the function.
        beta: Shape parameter for the generalized normal distribution.
        loc: Location parameter.
        scale: Scale parameter.
        const: Precomputed additive log-density constant for frozen parameters.

    Returns:
        Log-density value using frozen parameters and a precomputed constant.
    """
    z = (x - loc) / scale
    return - np.abs(z) ** beta + const


def __frozen_internal_logpdf_multivariate_normal(x, mean, inv_cov, const=0.0):
    """
    Evaluate a frozen scipy.stats.multivariate_normal-style multivariate normal log-density kernel.

    Args:
        x: Point or array of points at which to evaluate the function.
        mean: Mean vector.
        inv_cov: Inverse covariance matrix.
        const: Precomputed additive log-density constant for frozen parameters.

    Returns:
        Log-density value using frozen parameters and a precomputed constant.
    """
    x_min_mean = x - mean
    return const - 0.5 * np.dot(x_min_mean, inv_cov).dot(x_min_mean)


def __frozen_internal_logpdf_multivariate_t(x, loc, inv_shape, df=1.0, const=0.0):
    """
    Evaluate a frozen scipy.stats.multivariate_t-style multivariate Student's t log-density kernel.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        inv_shape: Inverse shape matrix for the multivariate t distribution.
        df: Degrees-of-freedom parameter.
        const: Precomputed additive log-density constant for frozen parameters.

    Returns:
        Log-density value using frozen parameters and a precomputed constant.
    """
    p = len(loc)
    x_min_mean = x - loc
    return -((df + p) / 2) * np.log(1 + (1.0 / df) * np.dot(x_min_mean, inv_shape).dot(x_min_mean)) + const


def __frozen_internal_logpdf_halfnorm(x, loc=0, scale=1.0, const=0):
    """
    Evaluate a frozen scipy.stats.halfnorm-style half-normal log-density kernel.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.
        const: Precomputed additive log-density constant for frozen parameters.

    Returns:
        Log-density value using frozen parameters and a precomputed constant.
    """
    z = (x - loc) / scale
    return -(z ** 2 / 2) + const


def __frozen_internal_logpdf_pareto(x, b, loc=0, scale=1.0, const=0):
    """
    Evaluate a frozen scipy.stats.pareto-style Pareto log-density kernel.

    Args:
        x: Point or array of points at which to evaluate the function.
        b: Second shape or upper-bound parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.
        const: Precomputed additive log-density constant for frozen parameters.

    Returns:
        Log-density value using frozen parameters and a precomputed constant.
    """
    z = (x - loc) / scale
    return const - (b + 1) * np.log(z)


def __frozen_internal_logpdf_halfcauchy(x, loc=0.0, scale=1.0, const=0.0):
    """
    Evaluate a frozen scipy.stats.halfcauchy-style half-Cauchy log-density kernel.

    Args:
        x: Point or array of points at which to evaluate the function.
        loc: Location parameter.
        scale: Scale parameter.
        const: Precomputed additive log-density constant for frozen parameters.

    Returns:
        Log-density value using frozen parameters and a precomputed constant.
    """
    z = (x - loc) / scale
    return const - np.log(1 + z ** 2)


def __frozen_internal_logpdf_loguniform(x, a, b, loc=0.0, scale=1.0, const=0.0):
    """
    Evaluate a frozen scipy.stats.loguniform-style log-uniform log-density kernel.

    Args:
        x: Point or array of points at which to evaluate the function.
        a: First shape or lower-bound parameter, matching scipy.stats naming for this distribution.
        b: Second shape or upper-bound parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.
        const: Precomputed additive log-density constant for frozen parameters.

    Returns:
        Log-density value using frozen parameters and a precomputed constant.
    """
    z = (x - loc) / scale
    return -np.log(z) + const


def __frozen_internal_logpdf_f(x, dfn, dfd, loc=0.0, scale=1.0, const=0.0):
    """
    Evaluate a frozen scipy.stats.f-style F log-density kernel.

    Args:
        x: Point or array of points at which to evaluate the function.
        dfn: Numerator degrees-of-freedom parameter for the F distribution.
        dfd: Denominator degrees-of-freedom parameter for the F distribution.
        loc: Location parameter.
        scale: Scale parameter.
        const: Precomputed additive log-density constant for frozen parameters.

    Returns:
        Log-density value using frozen parameters and a precomputed constant.
    """
    z = (x - loc) / scale
    return (dfn / 2 - 1) * np.log(z) - (dfn + dfd) / 2 * np.log(dfn * z + dfd) + const


def __frozen_internal_logpdf_weibull_min(x, c, loc=0.0, scale=1.0, const=0.0):
    """
    Evaluate a frozen scipy.stats.weibull_min-style Weibull minimum log-density kernel.

    Args:
        x: Point or array of points at which to evaluate the function.
        c: Shape parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.
        const: Precomputed additive log-density constant for frozen parameters.

    Returns:
        Log-density value using frozen parameters and a precomputed constant.
    """
    z = (x - loc) / scale
    return const + (c - 1) * np.log(z) - z ** c


def __frozen_internal_logpdf_dirichlet(x, alpha, const=0.0):
    """
    Evaluate a frozen scipy.stats.dirichlet-style Dirichlet log-density kernel.

    Args:
        x: Point or array of points at which to evaluate the function.
        alpha: Dirichlet concentration vector or interval mass, depending on the function.
        const: Precomputed additive log-density constant for frozen parameters.

    Returns:
        Log-density value using frozen parameters and a precomputed constant.
    """
    # Keep the historical simplex check used by this kernel.
    if abs(np.sum(x) - 1e-5) > 0:
        return -np.inf
    else:
        return const + np.dot(alpha - 1, np.log(x))


__nopython_frozen_internal_logpdf_norm = njit(__frozen_internal_logpdf_norm, cache=True)
__nopython_frozen_internal_logpdf_halfnorm = njit(__frozen_internal_logpdf_halfnorm, cache=True)
__nopython_frozen_internal_logpdf_beta = njit(__frozen_internal_logpdf_beta, cache=True)
__nopython_frozen_internal_logpdf_cauchy = njit(__frozen_internal_logpdf_cauchy, cache=True)
__nopython_frozen_internal_logpdf_laplace = njit(__frozen_internal_logpdf_laplace, cache=True)
__nopython_frozen_internal_logpdf_expon = njit(__frozen_internal_logpdf_expon, cache=True)
__nopython_frozen_internal_logpdf_t = njit(__frozen_internal_logpdf_t, cache=True)
__nopython_frozen_internal_logpdf_multivariate_t = njit(__frozen_internal_logpdf_multivariate_t, cache=True)
__nopython_frozen_internal_logpdf_gamma = njit(__frozen_internal_logpdf_gamma, cache=True)
__nopython_frozen_internal_logpdf_lognorm = njit(__frozen_internal_logpdf_lognorm, cache=True)
__nopython_frozen_internal_logpdf_invgamma = njit(__frozen_internal_logpdf_invgamma, cache=True)
__nopython_frozen_internal_logpdf_logistic = njit(__frozen_internal_logpdf_logistic, cache=True)
__nopython_frozen_internal_logpdf_chi2 = njit(__frozen_internal_logpdf_chi2, cache=True)
__nopython_frozen_internal_logpdf_gennorm = njit(__frozen_internal_logpdf_gennorm, cache=True)
__nopython_frozen_internal_logpdf_multivariate_normal = njit(__frozen_internal_logpdf_multivariate_normal, cache=True)
__nopython_frozen_internal_logpdf_truncnorm = njit(__frozen_internal_logpdf_truncnorm, cache=True)
__nopython_frozen_internal_logpdf_pareto = njit(__frozen_internal_logpdf_pareto, cache=True)
__nopython_frozen_internal_logpdf_halfcauchy = njit(__frozen_internal_logpdf_halfcauchy, cache=True)
__nopython_frozen_internal_logpdf_loguniform = njit(__frozen_internal_logpdf_loguniform, cache=True)
__nopython_frozen_internal_logpdf_genextreme = njit(__frozen_internal_logpdf_genextreme, cache=True)
__nopython_frozen_internal_logpdf_f = njit(__frozen_internal_logpdf_f, cache=True)
__nopython_frozen_internal_logpdf_weibull_min = njit(__frozen_internal_logpdf_weibull_min, cache=True)
__nopython_frozen_internal_logpdf_dirichlet = njit(__frozen_internal_logpdf_dirichlet, cache=True)


def get_frozen_logpdf_pareto(b, loc=0.0, scale=1.0, nopython=False):
    """
    Create a frozen scipy.stats.pareto-style Pareto logpdf callable.

    Args:
        b: Second shape or upper-bound parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.
        nopython: Whether to return a numba-jitted callable.

    Returns:
        Callable accepting ``x`` and returning the frozen log-density.
    """
    assert b > 0
    assert scale > 0
    const = np.log(b / scale)
    return _get_nopython_version('pareto', nopython, dict(b=b, loc=loc, scale=scale, const=const))


def get_frozen_logpdf_norm(loc=0.0, scale=1.0, nopython=False):
    """
    Create a frozen scipy.stats.norm-style normal logpdf callable.

    Args:
        loc: Location parameter.
        scale: Scale parameter.
        nopython: Whether to return a numba-jitted callable.

    Returns:
        Callable accepting ``x`` and returning the frozen log-density.
    """
    assert scale > 0
    const = -np.log(scale) - 0.5 * LOG_2_PI
    return _get_nopython_version(NORM, nopython, dict(loc=loc, scale=scale, const=const))


def get_frozen_logpdf_truncnorm(a, b, loc=0, scale=1, nopython=False):
    """
    Create a frozen scipy.stats.truncnorm-style truncated normal logpdf callable.

    Args:
        a: First shape or lower-bound parameter, matching scipy.stats naming for this distribution.
        b: Second shape or upper-bound parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.
        nopython: Whether to return a numba-jitted callable.

    Returns:
        Callable accepting ``x`` and returning the frozen log-density.
    """
    assert a < b
    assert scale > 0
    const = -np.log(scale) - 0.5 * LOG_2_PI \
            - np.log(norm.cdf(b) - norm.cdf(a))
    return _get_nopython_version(TRUNCNORM, nopython, dict(loc=loc, scale=scale, const=const))


def get_frozen_logpdf_halfnorm(loc=0.0, scale=1.0, nopython=False):
    """
    Create a frozen scipy.stats.halfnorm-style half-normal logpdf callable.

    Args:
        loc: Location parameter.
        scale: Scale parameter.
        nopython: Whether to return a numba-jitted callable.

    Returns:
        Callable accepting ``x`` and returning the frozen log-density.
    """
    assert scale > 0
    const = 0.5 * np.log(2 / np.pi) - np.log(scale)
    return _get_nopython_version(HALFNORM, nopython, dict(loc=loc, scale=scale, const=const))


def get_frozen_logpdf_beta(a, b, loc=0.0, scale=1.0, nopython=False):
    """
    Create a frozen scipy.stats.beta-style beta logpdf callable.

    Args:
        a: First shape or lower-bound parameter, matching scipy.stats naming for this distribution.
        b: Second shape or upper-bound parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.
        nopython: Whether to return a numba-jitted callable.

    Returns:
        Callable accepting ``x`` and returning the frozen log-density.
    """
    assert a > 0
    assert b > 0
    assert scale > 0
    const = -betaln(a, b) - np.log(scale)

    return _get_nopython_version(BETA, nopython, dict(a=a, b=b, loc=loc, scale=scale, const=const))


def get_frozen_logpdf_cauchy(loc=0.0, scale=1.0, nopython=False):
    """
    Create a frozen scipy.stats.cauchy-style cauchy logpdf callable.

    Args:
        loc: Location parameter.
        scale: Scale parameter.
        nopython: Whether to return a numba-jitted callable.

    Returns:
        Callable accepting ``x`` and returning the frozen log-density.
    """
    assert scale > 0
    const = -np.log(scale) - LOG_PI
    return _get_nopython_version(CAUCHY, nopython, dict(loc=loc, scale=scale, const=const))


def get_frozen_logpdf_laplace(loc=0.0, scale=1.0, nopython=False):
    """
    Create a frozen scipy.stats.laplace-style Laplace logpdf callable.

    Args:
        loc: Location parameter.
        scale: Scale parameter.
        nopython: Whether to return a numba-jitted callable.

    Returns:
        Callable accepting ``x`` and returning the frozen log-density.
    """
    assert scale > 0

    const = -np.log(2.0 * scale)
    return _get_nopython_version(LAPLACE, nopython, dict(loc=loc, scale=scale, const=const))


def get_frozen_logpdf_expon(loc=0.0, scale=1.0, nopython=False):
    """
    Create a frozen scipy.stats.expon-style exponential logpdf callable.

    Args:
        loc: Location parameter.
        scale: Scale parameter.
        nopython: Whether to return a numba-jitted callable.

    Returns:
        Callable accepting ``x`` and returning the frozen log-density.
    """
    assert scale > 0

    const = -np.log(scale)
    return _get_nopython_version(EXPON, nopython, dict(loc=loc, scale=scale, const=const))


def get_frozen_logpdf_t(df, loc=0.0, scale=1.0, nopython=False):
    """
    Create a frozen scipy.stats.t-style Student's t logpdf callable.

    Args:
        df: Degrees-of-freedom parameter.
        loc: Location parameter.
        scale: Scale parameter.
        nopython: Whether to return a numba-jitted callable.

    Returns:
        Callable accepting ``x`` and returning the frozen log-density.
    """
    assert df > 0
    assert scale > 0

    const = loggamma((df + 1) / 2) - loggamma(df / 2) - 0.5 * np.log(df * np.pi) - np.log(scale)
    return _get_nopython_version(T, nopython, dict(df=df, loc=loc, scale=scale, const=const))


def get_frozen_logpdf_gamma(a, loc=0.0, scale=1.0, nopython=False):
    """
    Create a frozen scipy.stats.gamma-style gamma logpdf callable.

    Args:
        a: First shape or lower-bound parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.
        nopython: Whether to return a numba-jitted callable.

    Returns:
        Callable accepting ``x`` and returning the frozen log-density.
    """
    assert a > 0
    assert scale > 0

    const = -loggamma(a) - np.log(scale)
    return _get_nopython_version(GAMMA, nopython, dict(a=a, loc=loc, scale=scale, const=const))


def get_frozen_logpdf_invgamma(a, loc=0.0, scale=1.0, nopython=False):
    """
    Create a frozen scipy.stats.invgamma-style inverse-gamma logpdf callable.

    Args:
        a: First shape or lower-bound parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.
        nopython: Whether to return a numba-jitted callable.

    Returns:
        Callable accepting ``x`` and returning the frozen log-density.
    """
    assert a > 0
    assert scale > 0
    const = -loggamma(a) - np.log(scale)
    return _get_nopython_version(INVGAMMA, nopython, dict(a=a, loc=loc, scale=scale, const=const))


def get_frozen_logpdf_lognorm(s, loc=0.0, scale=1.0, nopython=False):
    """
    Create a frozen scipy.stats.lognorm-style log-normal logpdf callable.

    Args:
        s: Shape parameter for the log-normal distribution.
        loc: Location parameter.
        scale: Scale parameter.
        nopython: Whether to return a numba-jitted callable.

    Returns:
        Callable accepting ``x`` and returning the frozen log-density.
    """
    assert s > 0
    assert scale > 0

    const = -np.log(scale) - np.log(s) - 0.5 * LOG_2_PI

    return _get_nopython_version(LOGNORM, nopython, dict(s=s, loc=loc, scale=scale, const=const))


def get_frozen_logpdf_logistic(loc=0.0, scale=1.0, nopython=False):
    """
    Create a frozen scipy.stats.logistic-style logistic logpdf callable.

    Args:
        loc: Location parameter.
        scale: Scale parameter.
        nopython: Whether to return a numba-jitted callable.

    Returns:
        Callable accepting ``x`` and returning the frozen log-density.
    """
    assert scale > 0
    const = -np.log(scale)
    return _get_nopython_version(LOGISTIC, nopython, dict(loc=loc, scale=scale, const=const))


def get_frozen_logpdf_gennorm(beta, loc=0.0, scale=1.0, nopython=False):
    """
    Create a frozen scipy.stats.gennorm-style generalized normal logpdf callable.

    Args:
        beta: Shape parameter for the generalized normal distribution.
        loc: Location parameter.
        scale: Scale parameter.
        nopython: Whether to return a numba-jitted callable.

    Returns:
        Callable accepting ``x`` and returning the frozen log-density.
    """
    assert beta > 0
    assert scale > 0

    const = np.log(0.5 * beta) - loggamma(1.0 / beta) - np.log(scale)
    return _get_nopython_version(GENNORM, nopython, dict(beta=beta, loc=loc, scale=scale, const=const))


def get_frozen_logpdf_chi2(df, loc=0.0, scale=1.0, nopython=False):
    """
    Create a frozen scipy.stats.chi2-style chi-square logpdf callable.

    Args:
        df: Degrees-of-freedom parameter.
        loc: Location parameter.
        scale: Scale parameter.
        nopython: Whether to return a numba-jitted callable.

    Returns:
        Callable accepting ``x`` and returning the frozen log-density.
    """
    assert scale > 0
    assert df >= 1
    const = -np.log(scale) - (df / 2.0) * np.log(2) - loggamma(df / 2.0)
    return _get_nopython_version(CHI2, nopython, dict(df=df, loc=loc, scale=scale, const=const))


def get_frozen_logpdf_multivariate_normal(mean, cov, nopython=False):
    """
    Create a frozen scipy.stats.multivariate_normal-style multivariate normal logpdf callable.

    Args:
        mean: Mean vector.
        cov: Covariance matrix.
        nopython: Whether to return a numba-jitted callable.

    Returns:
        Callable accepting ``x`` and returning the frozen log-density.
    """
    inv_cov = np.linalg.pinv(cov)
    _, logabsdet = np.linalg.slogdet(cov)
    k = len(mean)
    const = -k / 2 * LOG_2_PI - 0.5 * logabsdet
    mean = np.array(mean)

    return _get_nopython_version(MULTIVARIATE_NORMAL, nopython, dict(mean=mean, inv_cov=inv_cov, const=const))


def get_frozen_logpdf_halfcauchy(loc=0.0, scale=1.0, nopython=False):
    """
    Create a frozen scipy.stats.halfcauchy-style half-Cauchy logpdf callable.

    Args:
        loc: Location parameter.
        scale: Scale parameter.
        nopython: Whether to return a numba-jitted callable.

    Returns:
        Callable accepting ``x`` and returning the frozen log-density.
    """
    assert scale > 0
    const = np.log(2 / np.pi) - np.log(scale)
    return _get_nopython_version(HALFCAUCHY, nopython, dict(loc=loc, scale=scale, const=const))


def get_frozen_logpdf_multivariate_t(loc, shape, df=1.0, nopython=False):
    """
    Create a frozen scipy.stats.multivariate_t-style multivariate Student's t logpdf callable.

    Args:
        loc: Location parameter.
        shape: Shape matrix for the multivariate t distribution.
        df: Degrees-of-freedom parameter.
        nopython: Whether to return a numba-jitted callable.

    Returns:
        Callable accepting ``x`` and returning the frozen log-density.
    """
    assert df > 0
    p = len(loc)
    _, logabsdet = np.linalg.slogdet(shape)
    const = loggamma((df + p) / 2) - loggamma(df / 2) - (p / 2) * np.log(df * np.pi) - 0.5 * logabsdet
    inv_shape = np.linalg.pinv(shape)
    return _get_nopython_version(MULTIVARIATE_T, nopython, dict(mean=loc, inv_shape=inv_shape, df=df, const=const))


def get_frozen_logpdf_uniform(loc, scale, nopython=False):
    """
    Create a frozen scipy.stats.uniform-style uniform logpdf callable.

    Args:
        loc: Location parameter.
        scale: Scale parameter.
        nopython: Whether to return a numba-jitted callable.

    Returns:
        Callable accepting ``x`` and returning the frozen log-density.
    """
    assert scale > 0

    const = -np.log(scale)
    def logpdf(x):
        """Evaluate the frozen uniform log-density.

        Args:
            x: Point or array of points at which to evaluate the log-density.

        Returns:
            Constant uniform log-density inside the frozen support.
        """
        return const

    func = njit(logpdf, cache=True) if nopython else logpdf
    func.name = f'{"nopython_" if nopython else ""}logpdf_{UNIFORM}'

    # Keep the historical return value; ``func`` is named above for debugging
    # but callers receive the plain Python closure.
    return logpdf


def get_frozen_logpdf_flat(a, b, nopython=False):
    """
    Create a frozen scipy.stats.flat-style flat logpdf callable.

    Args:
        a: First shape or lower-bound parameter, matching scipy.stats naming for this distribution.
        b: Second shape or upper-bound parameter, matching scipy.stats naming for this distribution.
        nopython: Whether to return a numba-jitted callable.

    Returns:
        Callable accepting ``x`` and returning the frozen log-density.
    """
    assert b > a

    def logpdf(x):
        """Evaluate the frozen flat log-density.

        Args:
            x: Point or array of points at which to evaluate the log-density.

        Returns:
            Zero log-density for the frozen flat prior.
        """
        return 0.0

    func = njit(logpdf, cache=True) if nopython else logpdf
    func.name = f'{"nopython_" if nopython else ""}logpdf_{FLAT}'

    # Keep the historical return value; ``func`` is named above for debugging
    # but callers receive the plain Python closure.
    return logpdf


def get_frozen_logpdf_loguniform(a, b, loc=0.0, scale=1.0, nopython=False):
    """
    Create a frozen scipy.stats.loguniform-style log-uniform logpdf callable.

    Args:
        a: First shape or lower-bound parameter, matching scipy.stats naming for this distribution.
        b: Second shape or upper-bound parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.
        nopython: Whether to return a numba-jitted callable.

    Returns:
        Callable accepting ``x`` and returning the frozen log-density.
    """
    assert b > a
    assert scale > 0

    const = -np.log(np.log(b / a)) - np.log(scale)

    return _get_nopython_version(LOGUNIFORM, nopython, dict(a=a, b=b, loc=loc, scale=scale, const=const))


def get_frozen_logpdf_genextreme(c, loc=0.0, scale=1.0, nopython=False):
    """
    Create a frozen scipy.stats.genextreme-style generalized extreme value logpdf callable.

    Args:
        c: Shape parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.
        nopython: Whether to return a numba-jitted callable.

    Returns:
        Callable accepting ``x`` and returning the frozen log-density.
    """
    assert scale > 0
    const = -np.log(scale)
    return _get_nopython_version(GENEXTREME, nopython, dict(c=c, loc=loc, scale=scale, const=const))


def get_frozen_logpdf_f(dfn, dfd, loc=0.0, scale=1.0, nopython=False):
    """
    Create a frozen scipy.stats.f-style F logpdf callable.

    Args:
        dfn: Numerator degrees-of-freedom parameter for the F distribution.
        dfd: Denominator degrees-of-freedom parameter for the F distribution.
        loc: Location parameter.
        scale: Scale parameter.
        nopython: Whether to return a numba-jitted callable.

    Returns:
        Callable accepting ``x`` and returning the frozen log-density.
    """
    assert scale > 0
    assert dfn > 0
    assert dfd > 0
    const = dfn / 2 * np.log(dfn) + dfd / 2 * np.log(dfd) - betaln(dfn / 2, dfd / 2) - np.log(scale)
    return _get_nopython_version(F, nopython, dict(dfn=dfn, dfd=dfd, loc=loc, scale=scale, const=const))


def get_frozen_logpdf_weibull_min(c, loc=0.0, scale=1.0, nopython=False):
    """
    Create a frozen scipy.stats.weibull_min-style Weibull minimum logpdf callable.

    Args:
        c: Shape parameter, matching scipy.stats naming for this distribution.
        loc: Location parameter.
        scale: Scale parameter.
        nopython: Whether to return a numba-jitted callable.

    Returns:
        Callable accepting ``x`` and returning the frozen log-density.
    """
    assert scale > 0
    assert c > 0
    const = np.log(c / scale)
    return _get_nopython_version(WEIBULL_MIN, nopython, dict(c=c, loc=loc, scale=scale, const=const))


def get_frozen_logpdf_dirichlet(alpha, nopython=False):
    """
    Create a frozen scipy.stats.dirichlet-style Dirichlet logpdf callable.

    Args:
        alpha: Dirichlet concentration vector or interval mass, depending on the function.
        nopython: Whether to return a numba-jitted callable.

    Returns:
        Callable accepting ``x`` and returning the frozen log-density.
    """
    alpha = np.array(alpha)
    assert np.all(alpha > 0)
    const = -(np.sum([loggamma(a) for a in alpha]) - loggamma(np.sum(alpha)))

    return _get_nopython_version(DIRICHLET, nopython, dict(alpha=alpha, const=const))


RV_NAME_2_GETTER_DICT = {
    BETA: get_frozen_logpdf_beta,
    CAUCHY: get_frozen_logpdf_cauchy,
    CHI2: get_frozen_logpdf_chi2,
    EXPON: get_frozen_logpdf_expon,
    GAMMA: get_frozen_logpdf_gamma,
    GENNORM: get_frozen_logpdf_gennorm,
    HALFCAUCHY: get_frozen_logpdf_halfcauchy,
    HALFNORM: get_frozen_logpdf_halfnorm,
    INVGAMMA: get_frozen_logpdf_invgamma,
    LAPLACE: get_frozen_logpdf_laplace,
    LOGISTIC: get_frozen_logpdf_logistic,
    LOGNORM: get_frozen_logpdf_lognorm,
    NORM: get_frozen_logpdf_norm,
    PARETO: get_frozen_logpdf_pareto,
    T: get_frozen_logpdf_t,
    TRUNCNORM: get_frozen_logpdf_truncnorm,
    UNIFORM: get_frozen_logpdf_uniform,
    FLAT: get_frozen_logpdf_flat,
    LOGUNIFORM: get_frozen_logpdf_loguniform,
    GENEXTREME: get_frozen_logpdf_genextreme,
    F: get_frozen_logpdf_f,
    WEIBULL_MIN: get_frozen_logpdf_weibull_min,

    MULTIVARIATE_NORMAL: get_frozen_logpdf_multivariate_normal,
    MULTIVARIATE_T: get_frozen_logpdf_multivariate_t,
    DIRICHLET: get_frozen_logpdf_dirichlet,
}


def rv_name_2_getter(name):
    """
    Return the frozen logpdf factory for a distribution name.

    Args:
        name: Distribution name key.

    Returns:
        Factory function from ``RV_NAME_2_GETTER_DICT``.
    """
    assert name in RV_NAME_2_GETTER_DICT
    return RV_NAME_2_GETTER_DICT[name]


local_keys = list(locals().keys())
NOPYTHON_FROZEN_LOGPDF_IMPORT_STRING = ''
for k in local_keys:
    if (
            k[:25] == '__frozen_internal_logpdf_'
            or k[:34] == '__nopython_frozen_internal_logpdf_'
            or k[:18] == 'get_frozen_logpdf_'
    ):
        # print(f'{k}, ', end='')
        NOPYTHON_FROZEN_LOGPDF_IMPORT_STRING += f'\nfrom kanly.stats.distributions.nopython_frozen_logpdf import {k}'



# if __name__ == '__main__':
#
#     sigma = np.array([[12, 10], [10, 20]])
#     mu = np.array([0, 10])
#
#     from scipy.stats import multivariate_normal
#
#     mn = multivariate_normal(mu, sigma)
#     print(mn.logpdf(mu))
#
#     f = get_frozen_logpdf_multivariate_normal(mu, sigma)
#     print(f(mu))
#
# from scipy.stats import dirichlet
#
# f = get_frozen_logpdf_dirichlet([1.2, 6, 10])
# print(f([.2, .3, .5]))
# print(dirichlet([1.2, 6, 10]).logpdf([.2, .3, .5]))
