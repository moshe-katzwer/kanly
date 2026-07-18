"""Exponential-family definitions used by sparse GLM estimation.

Each family supplies the pieces required by the GLM likelihood

    log L_i = (y_i * theta_i - b(theta_i)) / phi_i + c(y_i, phi_i)

where ``theta`` is the canonical parameter, ``b`` is the cumulant function,
``phi`` is the scale/dispersion (possibly adjusted by variance weights), and
``mu = b'(theta)`` is the fitted response mean.  The optimizer combines these
family objects with link functions from ``links.py`` to perform IRLS or
coordinate-descent estimation.
"""
from __future__ import absolute_import, print_function

import re
import warnings
from abc import ABC, abstractmethod
from math import lgamma

import numpy as np
from numba import vectorize, float64

from kanly.regression.generalized_linear_models.links import (
    Link, Identity, Logit, NegativeInverse, Log,
    NegativeTwoInverseSquared, Sqrt, Probit, CLogLog, Inverse, InverseSquared, Cauchy,
    NegativeBinomialCanonicalLink,
    _get_link as _get_link_from_link_class)

BINOMIAL = 'BINOMIAL'
BERNOULLI = 'BERNOULLI'
POISSON = 'POISSON'
GAUSSIAN = 'GAUSSIAN'
GAMMA = 'GAMMA'
INVERSE_GAUSSIAN = 'INVERSE_GAUSSIAN'
NEGATIVE_BINOMIAL = 'NEGATIVE_BINOMIAL'


@vectorize([float64(float64)], cache=True)
def _log_gamma_vectorized(x):
    """Evaluate ``log(Gamma(x))`` elementwise for likelihood constants.

    Args:
        x: Scalar or array values passed through the Numba vectorized wrapper.

    Returns:
        Elementwise log-gamma values.
    """
    return lgamma(x)


def positive_clip(y, tol=1e-6):
    """Clip response-scale values to a strictly positive lower bound.

    Args:
        y: Scalar or array of response-scale values.
        tol: Minimum positive value to allow.

    Returns:
        Values clipped below at ``tol``.
    """
    return np.clip(y, tol, np.inf)


class Family(ABC):
    """Abstract base class for exponential-dispersion GLM families.

    Subclasses define variance functions, log-likelihood pieces, canonical-link
    relationships, support checks, deviance, and scale behaviour.  These methods
    are consumed by ``glm_internal`` for fitting and by the results/covariance
    layers for diagnostics.
    """

    @classmethod
    def __str__(cls):
        """Return the registered name of this class or object."""
        return cls.name()

    def get_starting_intercept(self, endog, var_weights=None, link=None):
        """Compute a link-scale intercept from the weighted average response.

        Args:
            endog: Response vector.
            var_weights: Optional variance weights used in the response average.
            link: Optional link object; defaults to this family's canonical link.

        Returns:
            Scalar intercept on the link scale.
        """
        if link is None:
            link = self.canonical_link
        return link.link(self.b_deriv(self.b_deriv_inv(np.average(endog, weights=var_weights))))

    @abstractmethod
    def variance(self, mu):
        """Evaluate the GLM variance function V(mu)."""
        raise NotImplementedError

    def d_variance(self, mu):
        """Evaluate the derivative of the variance function with respect to mu."""
        h = 1e-6
        return (self.variance(mu + h) - self.variance(mu - h)) / (2 * h)

    @abstractmethod
    def b(self, theta):
        """Evaluate the cumulant function b(theta) for the family."""
        raise NotImplementedError

    @abstractmethod
    def b_deriv(self, theta):
        """Evaluate the mean map b_prime(theta)."""
        raise NotImplementedError

    @abstractmethod
    def b_deriv_inv(self, mu):
        """Map a response mean mu back to canonical parameter theta."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def c(y, scale):
        """Evaluate the base-measure term c(y, scale) in the log-likelihood."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def check_valid_range(y):
        """Return a boolean mask indicating which responses are valid for this family."""
        raise NotImplementedError

    def log_likelihood(self, endog, theta, scale=1., var_weights=1.):
        """Return the summed GLM log-likelihood for observations.

        Args:
            endog: Response vector ``y``.
            theta: Canonical parameter vector.
            scale: Dispersion/scale parameter ``phi``.
            var_weights: Variance weights; scalar or array.

        Returns:
            Scalar summed log-likelihood.
        """
        return self.log_likelihood_obs(endog, theta, scale=scale, var_weights=var_weights).sum()

    def log_likelihood_obs(self, endog, theta, scale=1., var_weights=1.):
        """Return observation-level GLM log-likelihood contributions.

        Args:
            endog: Response vector ``y``.
            theta: Canonical parameter vector.
            scale: Dispersion/scale parameter ``phi``.
            var_weights: Variance weights.  Array weights with value 0 are
                omitted from the returned vector.

        Returns:
            Vector of log-likelihood contributions.
        """
        theta = np.asarray(theta)
        endog = np.asarray(endog)
        if isinstance(var_weights, np.ndarray):
            # Zero-weighted rows do not contribute to the likelihood and can
            # otherwise create divisions by zero in scale / var_weights.
            idx_pos_wt = var_weights > 0
            return (
                    (endog * theta - self.b(theta))[idx_pos_wt] / (scale / var_weights[idx_pos_wt])
                    + self.c(endog[idx_pos_wt], scale / var_weights[idx_pos_wt])
            )
        else:
            return (
                    (endog * theta - self.b(theta)) / (scale / var_weights) + self.c(endog, scale / var_weights)
            )

    @staticmethod
    def starting_mu(y):
        """Return an initial response-scale mean vector for IRLS."""
        return .5 * (y + y.mean())

    @staticmethod
    @abstractmethod
    def param_transformation(theta, scale):
        """
        converts the parameters of the exponential dispersion
        representation of the distribution to its more natural
        parametrization
        """
        raise NotImplementedError

    @abstractmethod
    def deviance(self, endog, endog_predicted, var_weights=1):
        """Compute the family deviance for observed and fitted means."""
        raise NotImplementedError

    def pearson_chi2(self, endog, endog_predicted, var_weights=1.):
        """Compute the Pearson chi-squared statistic for observed and fitted means."""
        return (var_weights * (endog - endog_predicted) ** 2 / self.variance(endog_predicted)).sum()

    def is_canonical(self, link):
        """Return whether ``link`` is this family's canonical link.

        Args:
            link: Link name string or ``Link`` instance/class.

        Returns:
            Boolean indicating whether ``link.name()`` matches the canonical
            link name.
        """
        if isinstance(link, str):
            return link == self.canonical_link().name
        elif issubclass(link.__class__, Link.__class__) or isinstance(link, Link):
            return link.name == self.canonical_link().name
        else:
            raise Exception("'link' must be a str or a Link class")

    @abstractmethod
    def canonical_link(self):
        """Return an instance of the canonical link for this family."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def safe_links(cls):
        """Return link classes considered numerically safe for this family."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def name(cls):
        """Return the canonical uppercase name for this link or family."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def is_positive_range(cls):
        """Return whether the family response support is strictly non-negative or positive."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def is_fixed_dispersion(cls):
        """Return whether the family dispersion parameter is fixed."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def clip(cls, y, tol=1e-6):
        """Clips the variable to the valid range of the family"""
        raise NotImplementedError

    def default_link(self):
        """Return the default link used when no link is provided."""
        return self.canonical_link()

class Binomial(Family):
    """Binomial/Bernoulli-style family for outcomes in ``[0, 1]``.

    Uses variance ``V(mu) = mu * (1 - mu)`` and logit as the canonical link.
    This implementation supports both binary outcomes and fractional
    proportions in the unit interval.
    """
    @classmethod
    def is_fixed_dispersion(cls):
        """Return whether the family dispersion parameter is fixed."""
        return True

    @classmethod
    def name(cls):
        """Return the canonical uppercase name for this link or family."""
        return BINOMIAL

    @classmethod
    def safe_links(cls):
        """Return link classes considered numerically safe for this family."""
        return [Probit, Logit, CLogLog, Identity, Cauchy]

    def canonical_link(self):
        """Return an instance of the canonical link for this family."""
        return Logit()

    def variance(self, mu):
        """Evaluate the GLM variance function V(mu)."""
        thr = 1e-12
        # Keep fitted probabilities away from exact 0/1 so IRLS weights and
        # log-likelihood sparse_terms remain finite.
        mu = np.clip(mu, a_min=thr, a_max=1 - thr)
        return mu * (1 - mu)

    @classmethod
    def d_variance(cls, mu):
        """Evaluate the derivative of the variance function with respect to mu."""
        return 1 - 2 * mu

    def b(self, theta):
        """Evaluate the cumulant function b(theta) for the family."""
        return np.log(1 + np.exp(theta))

    def b_deriv(self, theta):
        """Evaluate the mean map b_prime(theta)."""
        return 1.0 / (1.0 + np.exp(-theta))

    def b_deriv_inv(self, mu):
        """Map a response mean mu back to canonical parameter theta."""
        return -np.log(1.0 / mu - 1)

    @staticmethod
    def c(y, scale):
        """Evaluate the base-measure term c(y, scale) in the log-likelihood."""
        return 0.0

    @staticmethod
    def param_transformation(theta, scale):
        """Convert exponential-family parameters to the family natural output parameterization."""
        return {'p': 1.0 / (1 + np.exp(-theta))}

    @staticmethod
    def starting_mu(y):
        """Return an initial response-scale mean vector for IRLS."""
        return .5 * (y + .5)

    @staticmethod
    def check_valid_range(y):
        """Return a boolean mask indicating which responses are valid for this family."""
        return (y >= 0) & (y <= 1)

    def deviance(self, endog, endog_predicted, var_weights=1):
        """Compute the binomial deviance.

        Args:
            endog: Observed responses in ``[0, 1]``.
            endog_predicted: Fitted probabilities.
            var_weights: Optional variance weights.

        Returns:
            Scalar deviance with special handling for exact 0/1 responses.
        """
        idx0 = endog == 0
        idx1 = endog == 1
        idx2 = ~(idx0 | idx1)

        x = np.zeros(len(endog))
        x[idx1] = endog[idx1] * (np.log(endog[idx1]) - np.log(endog_predicted[idx1]))
        x[idx0] = (1 - endog[idx0]) * (np.log(1 - endog[idx0]) - np.log(1 - endog_predicted[idx0]))
        x[idx2] = endog[idx2] * (np.log(endog[idx2]) - np.log(endog_predicted[idx2])) \
                  + (1 - endog[idx2]) * (np.log(1 - endog[idx2]) - np.log(1 - endog_predicted[idx2]))

        return 2 * (var_weights * x).sum()

    @classmethod
    def is_positive_range(cls):
        """Return whether the family response support is strictly non-negative or positive."""
        return True

    @classmethod
    def clip(cls, y, tol=1e-6):
        """Clip response-scale means to the valid range of the family."""
        return np.clip(y, tol, 1 - tol)


class Bernoulli(Binomial):
    """Bernoulli family alias for binary outcomes.

    Inherits all likelihood, variance, and deviance behaviour from
    ``Binomial`` but reports a distinct family name.
    """

    @classmethod
    def name(cls):
        """Return the canonical uppercase name for this link or family."""
        return BERNOULLI


class NegativeBinomial(Family):
    """Negative-binomial family for overdispersed count outcomes.

    Args:
        alpha: Positive overdispersion parameter, with variance
            ``V(mu) = mu + alpha * mu**2``.
    """

    # TODO
    def __init__(self, alpha=1.0):
        """
        alpha: Overdispersion parameter, default=1

        Args:
            alpha: Positive overdispersion parameter.
        """
        assert alpha > 0.0
        self.alpha = alpha

    @classmethod
    def is_fixed_dispersion(cls):
        """Return whether the family dispersion parameter is fixed."""
        return False

    @classmethod
    def name(cls):
        """Return the canonical uppercase name for this link or family."""
        return NEGATIVE_BINOMIAL

    @classmethod
    def safe_links(cls):
        """Return link classes considered numerically safe for this family."""
        return [Log, Sqrt, Identity]

    def canonical_link(self):
        """Return an instance of the canonical link for this family."""
        return NegativeBinomialCanonicalLink(alpha=self.alpha)

    def variance(self, mu):
        """Evaluate the GLM variance function V(mu)."""
        return mu + self.alpha * mu ** 2

    def d_variance(self, mu):
        """Evaluate the derivative of the variance function with respect to mu."""
        return 1 + 2 * self.alpha * mu

    def b(self, theta):
        """Evaluate the cumulant function b(theta) for the family."""
        return -1.0 / self.alpha * np.log(1 - np.exp(theta))

    def b_deriv(self, theta):
        """Evaluate the mean map b_prime(theta)."""
        e_theta = np.exp(theta)
        return 1.0 / self.alpha * e_theta / (1 - e_theta)

    def b_deriv_inv(self, mu):
        """Map a response mean mu back to canonical parameter theta."""
        return np.log(self.alpha * mu / (1 + self.alpha * mu))

    def c(self, y, scale):
        """Evaluate the base-measure term c(y, scale) in the log-likelihood."""
        return (_log_gamma_vectorized(y + 1 / self.alpha) - _log_gamma_vectorized(
            1 / self.alpha) - _log_gamma_vectorized(y + 1))

    @staticmethod
    def param_transformation(theta, scale):
        """Convert exponential-family parameters to the family natural output parameterization."""
        # return {'lambda': np.exp(theta)}
        raise NotImplementedError

    @staticmethod
    def check_valid_range(y):
        """Return a boolean mask indicating which responses are valid for this family."""
        return np.asarray(y) >= 0

    def deviance(self, endog, endog_predicted, var_weights=1):
        """Compute the family deviance for observed and fitted means."""
        return 2 * (
                var_weights * (
                endog * np.log(endog / endog_predicted)
                - (endog + 1.0 / self.alpha) * np.log((1 + self.alpha * endog) / (1 + self.alpha * endog_predicted))
        )).sum()

    @classmethod
    def is_positive_range(cls):
        """Return whether the family response support is strictly non-negative or positive."""
        return True

    @classmethod
    def clip(cls, y, tol=1e-6):
        """Clip response-scale means to the valid range of the family."""
        return positive_clip(y, tol)

    def default_link(self):
        """Return the default link used when no link is provided."""
        return Log()


class Poisson(Family):
    """Poisson family for non-negative count outcomes.

    Uses variance ``V(mu) = mu`` and log as the canonical link.  The fixed
    dispersion is 1 under the standard Poisson likelihood.
    """

    @classmethod
    def is_fixed_dispersion(cls):
        """Return whether the family dispersion parameter is fixed."""
        return True

    @classmethod
    def name(cls):
        """Return the canonical uppercase name for this link or family."""
        return POISSON

    @classmethod
    def safe_links(cls):
        """Return link classes considered numerically safe for this family."""
        return [Log, Sqrt, Identity]

    def canonical_link(self):
        """Return an instance of the canonical link for this family."""
        return Log()

    def variance(self, mu):
        """Evaluate the GLM variance function V(mu)."""
        return mu

    @classmethod
    def d_variance(cls, mu):
        """Evaluate the derivative of the variance function with respect to mu."""
        return np.ones(len(mu))

    def b(self, theta):
        """Evaluate the cumulant function b(theta) for the family."""
        return np.exp(theta)

    def b_deriv(self, theta):
        """Evaluate the mean map b_prime(theta)."""
        return np.exp(theta)

    def b_deriv_inv(self, mu):
        """Map a response mean mu back to canonical parameter theta."""
        return np.log(mu)

    @staticmethod
    def c(y, scale):
        """Evaluate the base-measure term c(y, scale) in the log-likelihood."""
        return -_log_gamma_vectorized(1 + y) / scale

    @staticmethod
    def param_transformation(theta, scale):
        """Convert exponential-family parameters to the family natural output parameterization."""
        return {'lambda': np.exp(theta)}

    @staticmethod
    def check_valid_range(y):
        """Return a boolean mask indicating which responses are valid for this family."""
        return np.asarray(y) >= 0

    def deviance(self, endog, endog_predicted, var_weights=1):
        """Compute the family deviance for observed and fitted means."""
        dev = 2 * (var_weights * (- (endog - endog_predicted)))
        idx = endog > 0
        dev[idx] += 2 * var_weights[idx] * endog[idx] * np.log(endog[idx] / endog_predicted[idx])
        return dev.sum()

    @classmethod
    def is_positive_range(cls):
        """Return whether the family response support is strictly non-negative or positive."""
        return True

    @classmethod
    def clip(cls, y, tol=1e-6):
        """Clip response-scale means to the valid range of the family."""
        return positive_clip(y, tol)


class Gaussian(Family):
    """Gaussian family for real-valued continuous outcomes.

    Uses constant variance and identity as the canonical link.  The scale
    parameter is estimated from the residual variance.
    """

    @classmethod
    def is_fixed_dispersion(cls):
        """Return whether the family dispersion parameter is fixed."""
        return False

    @classmethod
    def name(cls):
        """Return the canonical uppercase name for this link or family."""
        return GAUSSIAN

    @classmethod
    def safe_links(cls):
        """Return link classes considered numerically safe for this family."""
        return [Log, Identity]

    def canonical_link(self):
        """Return an instance of the canonical link for this family."""
        return Identity()

    def variance(self, mu):
        """Evaluate the GLM variance function V(mu)."""
        return np.ones(len(mu))

    @classmethod
    def d_variance(cls, mu):
        """Evaluate the derivative of the variance function with respect to mu."""
        return np.zeros(len(mu))

    def b(self, theta):
        """Evaluate the cumulant function b(theta) for the family."""
        return .5 * np.asarray(theta) ** 2

    def b_deriv(self, theta):
        """Evaluate the mean map b_prime(theta)."""
        return np.asarray(theta)

    def b_deriv_inv(self, mu):
        """Map a response mean mu back to canonical parameter theta."""
        return np.asarray(mu)

    @staticmethod
    def c(y, scale):
        """Evaluate the base-measure term c(y, scale) in the log-likelihood."""
        return -.5 * ((np.asarray(y) ** 2 / scale) + np.log(2 * np.pi * scale))

    @staticmethod
    def param_transformation(theta, scale):
        """Convert exponential-family parameters to the family natural output parameterization."""
        return {'mean': .5 * np.asarray(theta) ** 2, 'variance': scale}

    @staticmethod
    def check_valid_range(y):
        """Return a boolean mask indicating which responses are valid for this family."""
        return np.ones(len(y)).astype(bool)

    def deviance(self, endog, endog_predicted, var_weights=1):
        """Compute the family deviance for observed and fitted means."""
        return (var_weights * (endog - endog_predicted) ** 2).sum()

    @classmethod
    def is_positive_range(cls):
        """Return whether the family response support is strictly non-negative or positive."""
        return False

    @classmethod
    def clip(cls, y):
        """Clip response-scale means to the valid range of the family."""
        return y


class Gamma(Family):
    """Gamma family for strictly positive continuous outcomes.

    Uses variance ``V(mu) = mu**2`` and negative inverse as the canonical link.
    The implementation clips fitted values away from zero for numerical safety.
    """
    @classmethod
    def is_fixed_dispersion(cls):
        """Return whether the family dispersion parameter is fixed."""
        return False

    @classmethod
    def name(cls):
        """Return the canonical uppercase name for this link or family."""
        return GAMMA

    @classmethod
    def safe_links(cls):
        """Return link classes considered numerically safe for this family."""
        return [Log, Identity, Inverse, NegativeInverse]

    def canonical_link(self):
        """Return an instance of the canonical link for this family."""
        return NegativeInverse()

    def variance(self, mu):
        """Evaluate the GLM variance function V(mu)."""
        return (np.asarray(mu) + 1e-10) ** 2

    def b(self, theta):
        """Evaluate the cumulant function b(theta) for the family."""
        return -np.log(-theta)

    def b_deriv(self, theta):
        """Evaluate the mean map b_prime(theta)."""
        return -1.0 / theta

    def b_deriv_inv(self, mu):
        """Map a response mean mu back to canonical parameter theta."""
        return -1.0 / mu

    @staticmethod
    def c(y, scale):
        """Evaluate the base-measure term c(y, scale) in the log-likelihood."""
        return 1.0 / scale * np.log(1.0 / scale) - _log_gamma_vectorized(1.0 / scale) + (1 / scale - 1) * np.log(y)

    @staticmethod
    def param_transformation(theta, scale):
        """Convert exponential-family parameters to the family natural output parameterization."""
        return {'alpha': -1.0 / scale, 'beta': -scale / theta}

    @staticmethod
    def check_valid_range(y):
        """Return a boolean mask indicating which responses are valid for this family."""
        return np.asarray(y) > 0

    @staticmethod
    def starting_mu(y):
        """Return an initial response-scale mean vector for IRLS."""
        return np.ones(len(y)) * y.mean()

    def deviance(self, endog, endog_predicted, var_weights=1):
        """Compute the family deviance for observed and fitted means."""
        idx = endog > 0
        dev = (endog - endog_predicted) / endog_predicted
        dev[idx] -= np.log(endog[idx] / endog_predicted[idx])
        return 2 * (var_weights * dev).sum()

    @classmethod
    def is_positive_range(cls):
        """Return whether the family response support is strictly non-negative or positive."""
        return True

    @classmethod
    def clip(cls, y, tol=1e-6):
        """Clip response-scale means to the valid range of the family."""
        return positive_clip(y, tol)


class InverseGaussian(Family):
    """Inverse Gaussian family for strictly positive continuous outcomes.

    Uses variance ``V(mu) = mu**3`` and the negative half inverse-squared link as
    the canonical link.
    """

    @classmethod
    def is_fixed_dispersion(cls):
        """Return whether the family dispersion parameter is fixed."""
        return False

    @classmethod
    def name(cls):
        """Return the canonical uppercase name for this link or family."""
        return INVERSE_GAUSSIAN

    @classmethod
    def safe_links(cls):
        """Return link classes considered numerically safe for this family."""
        return [InverseSquared, NegativeTwoInverseSquared, Identity, Log]

    def canonical_link(self):
        """Return an instance of the canonical link for this family."""
        return NegativeTwoInverseSquared()

    def variance(self, mu):
        """Evaluate the GLM variance function V(mu)."""
        return np.asarray(mu) ** 3

    @classmethod
    def d_variance(cls, mu):
        """Evaluate the derivative of the variance function with respect to mu."""
        return 3.0 * np.asarray(mu) ** 2

    def b(self, theta):
        """Evaluate the cumulant function b(theta) for the family."""
        return -np.sqrt(-2 * theta)

    def b_deriv(self, theta):
        """Evaluate the mean map b_prime(theta)."""
        return (-2 * theta) ** -.5

    def b_deriv_inv(self, mu):
        """Map a response mean mu back to canonical parameter theta."""
        return -.5 * mu ** -2

    @staticmethod
    def c(y, scale):
        """Evaluate the base-measure term c(y, scale) in the log-likelihood."""
        return -1.0 / (2 * scale * y) - .5 * (np.log(2 * scale * np.pi * y ** 3))

    @staticmethod
    def param_transformation(theta, scale):
        """Convert exponential-family parameters to the family natural output parameterization."""
        return {'mu': 1.0 / np.sqrt(-2 * theta), 'lambda': 1.0 / scale}

    @staticmethod
    def check_valid_range(y):
        """Return a boolean mask indicating which responses are valid for this family."""
        return np.asarray(y) > 0

    def deviance(self, endog, endog_predicted, var_weights=1):
        """Compute the family deviance for observed and fitted means."""
        return ((endog - endog_predicted) ** 2 / (endog_predicted ** 2 * endog) * var_weights).sum()

    @classmethod
    def is_positive_range(cls):
        """Return whether the family response support is strictly non-negative or positive."""
        return True

    @classmethod
    def clip(cls, y, tol=1e-6):
        """Clip response-scale means to the valid range of the family."""
        return positive_clip(y, tol)


FAMILIES = [Gamma, Gaussian, Binomial, InverseGaussian, Poisson, Bernoulli, NegativeBinomial]

FAMILY_NAME_2_CLS = {
    f.name(): f for f in FAMILIES
}

FAMILY_NO_UNDERSCORE_TO_UNDERSCORE = {f.replace('_', ''): f for f in FAMILY_NAME_2_CLS.keys()}


def _get_link(link, family):
    """Resolve and validate a link for a specific family.

    Args:
        link: Optional link name, link class, or link instance.  ``None`` uses
            ``family.default_link()``.
        family: Concrete ``Family`` instance whose safe-link list is enforced.

    Returns:
        Concrete ``Link`` instance compatible with ``family``.

    Raises:
        Exception: If the link cannot be resolved or is unsafe for the family.
    """
    if link is None:
        return family.default_link()
    else:
        link = _get_link_from_link_class(link)

    if link.name() not in [l.name() for l in family.safe_links()]:
        raise Exception("Unsafe link %s for family %s" % (link, family))

    return link


def _get_family(family):
    """Resolve user family input to a concrete ``Family`` instance.

    Args:
        family: Family name string, ``Family`` subclass, or already-created
            ``Family`` object.  Negative binomial strings may include an alpha,
            e.g. ``'negative_binomial(0.5)'``.

    Returns:
        Concrete ``Family`` instance.

    Raises:
        Exception: If ``family`` cannot be resolved.
    """
    if isinstance(family, str):

        # Special case for negative binomial, allowing users to encode the
        # overdispersion parameter in the family string.
        family_temp = family.upper().replace('_', '')
        match = re.match(r'^(.*?)\((.+)\)$', family_temp)
        if match:
            temp = (match.group(1), match.group(2))
            if temp[0] == "NEGATIVEBINOMIAL":
                alpha = float(temp[1])
            return NegativeBinomial(alpha)
        # ------------------

        try:
            return FAMILY_NAME_2_CLS[FAMILY_NO_UNDERSCORE_TO_UNDERSCORE[family.replace('_', '').upper()]]()
        except:
            raise Exception("Family '%s' not found!" % family)
    elif isinstance(family, Family):
        return family
    else:
        try:
            if issubclass(family, Family):
                return family()
        except:
            pass

    raise Exception("Family '%s' not found!" % str(family))


def _get_family_and_link(family, link):
    """Resolve and validate a family/link pair for GLM fitting.

    Args:
        family: Family name/class/instance.
        link: Optional link name/class/instance.

    Returns:
        Tuple ``(family, link)`` of concrete objects.
    """
    family = _get_family(family)
    link = _get_link(link, family)
    if link.name() not in [l.name() for l in family.safe_links()]:
        raise Exception("Link %s not supported for family %s" % (link.name, family.name))
    return family, link
