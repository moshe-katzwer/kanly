"""Link functions for generalized linear models.

In a GLM the conditional mean is related to the linear predictor by

    g(mu_i) = eta_i = x_i' beta

where ``g`` is the link function, ``mu_i = E[y_i | x_i]`` is the response-scale
mean, and ``eta_i`` is the linear predictor.  The classes in this module provide
``link`` (g), ``inverse_link`` (g^{-1}), and derivative methods used by IRLS and
robust covariance calculations.
"""
from __future__ import absolute_import, print_function

from abc import ABC, abstractmethod

import numpy as np

from scipy.stats import norm, cauchy

LOGIT = 'LOGIT'
PROBIT = 'PROBIT'
IDENTITY = 'IDENTITY'
LOG = 'LOG'
EXPONENTIAL = 'EXPONENTIAL'
NEGATIVE_INVERSE = 'NEGATIVE_INVERSE'
NEGATIVE_TWO_INVERSE_SQUARED = 'NEGATIVE_TWO_INVERSE_SQUARED'
INVERSE = 'INVERSE'
INVERSE_SQUARED = 'INVERSE_SQUARED'
CLOGLOG = 'CLOGLOG'
SQRT = 'SQRT'
CAUCHY = 'CAUCHY'
POWER = 'POWER'
NEGATIVE_BINOMIAL_CANONICAL_LINK = 'NEGATIVE_BINOMIAL_CANONICAL_LINK'


class Link(ABC):
    """Abstract base class for GLM link functions.

    Subclasses define the map from response means to the linear predictor and
    the inverse map back to fitted means.  Derivatives are used to build IRLS
    weights and non-canonical-link covariance corrections.
    """

    @classmethod
    def __str__(cls):
        """Return the registered name of this class or object."""
        return cls.name()

    @abstractmethod
    def link(self, mu):
        """Map response-scale means to linear predictors.

        Args:
            mu: Scalar or array of fitted means on the response scale.

        Returns:
            Scalar or array on the linear predictor scale.
        """
        pass

    @abstractmethod
    def inverse_link(self, eta):
        """Map linear predictors to response-scale means.

        Args:
            eta: Scalar or array of linear predictor values.

        Returns:
            Scalar or array of fitted means in the support of the GLM family.
        """
        pass

    def deriv_inverse_link(self, mu):
        """Fallback if not implemented in class directly"""
        h = 1e-6
        return (self.inverse_link(mu + h / 2) - self.inverse_link(mu - h / 2)) / h

    def deriv2_inverse_link(self, mu):
        """Fallback if not implemented in class directly"""
        h = 1.22e-4
        return (self.inverse_link(mu + h) - 2 * self.inverse_link(mu) + self.inverse_link(mu - h)) / h ** 2

    def deriv(self, mu):
        """Evaluate the first derivative ``g'(mu)``.

        Args:
            mu: Response-scale mean value(s).

        Returns:
            First derivative of the link with respect to ``mu``.
        """
        h = 1e-6
        return (self.link(mu + h / 2) - self.link(mu - h / 2)) / h

    def deriv2(self, mu):
        """Evaluate the second derivative ``g''(mu)``.

        Args:
            mu: Response-scale mean value(s).

        Returns:
            Second derivative of the link with respect to ``mu``.
        """
        h = 1e-6
        return (self.deriv(mu + h / 2) - self.deriv(mu - h / 2)) / h

    @classmethod
    @abstractmethod
    def function_str(cls):
        """Return a short mathematical description of the link function."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def mean_function_str(cls):
        """Return a short mathematical description of the inverse-link mean function."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def name(cls):
        """Return the canonical uppercase name for this link or family."""
        raise NotImplementedError


class Log(Link):
    """Log link, ``g(mu) = log(mu)``.

    This is the canonical link for Poisson GLMs and a common safe link for
    positive-mean families because ``inverse_link(eta) = exp(eta)`` guarantees
    strictly positive fitted means.
    """

    def link(self, mu):
        """Map a response-scale mean value to the linear predictor scale."""
        return np.log(mu)

    def inverse_link(self, eta):
        """Map a linear predictor value back to the response mean scale."""
        return np.exp(eta)

    def deriv_inverse_link(self, eta):
        return np.exp(eta)

    def deriv2_inverse_link(self, eta):
        return np.exp(eta)

    def deriv(self, mu):
        """Evaluate the first derivative of the link function."""
        return 1.0 / mu

    def deriv2(self, mu):
        """Evaluate the second derivative of the link function."""
        return -1.0 / mu ** 2

    @classmethod
    def function_str(cls):
        """Return a short mathematical description of the link function."""
        return "g(x) = log(x)"

    @classmethod
    def mean_function_str(cls):
        """Return a short mathematical description of the inverse-link mean function."""
        return "E[y|x;b] = exp(x' b)"

    @classmethod
    def name(cls):
        """Return the canonical uppercase name for this link or family."""
        return LOG


class Identity(Link):
    """Identity link, ``g(mu) = mu``.

    This is the canonical Gaussian link.  It leaves the mean unconstrained, so
    it is only safe for positive-support families when the fitted linear
    predictor remains inside the family's support.
    """

    def link(self, mu):
        """Map a response-scale mean value to the linear predictor scale."""
        return mu

    def inverse_link(self, eta):
        """Map a linear predictor value back to the response mean scale."""
        return eta

    def deriv_inverse_link(self, eta):
        if isinstance(eta, (int, float, np.integer, np.floating)):
            return 1.0
        return np.ones(eta.shape)

    def deriv2_inverse_link(self, eta):
        if isinstance(eta, (int, float, np.integer, np.floating)):
            return 0.0
        return np.zeros(eta.shape)

    def deriv(self, mu):
        """Evaluate the first derivative of the link function."""
        return 1.0

    def deriv2(self, mu):
        """Evaluate the second derivative of the link function."""
        return 0.0

    @classmethod
    def function_str(cls):
        """Return a short mathematical description of the link function."""
        return "g(x) = x"

    @classmethod
    def mean_function_str(cls):
        """Return a short mathematical description of the inverse-link mean function."""
        return "E[y|x;b] = x' b"

    @classmethod
    def name(cls):
        """Return the canonical uppercase name for this link or family."""
        return IDENTITY


class Logit(Link):
    """Logit link, ``g(mu) = log(mu / (1 - mu))``.

    This is the canonical binomial/Bernoulli link.  Its inverse is the logistic
    sigmoid, which keeps fitted probabilities in ``(0, 1)``.
    """

    def link(self, mu):
        """Map a response-scale mean value to the linear predictor scale."""
        return np.log(mu / (1.0 - mu))

    def inverse_link(self, eta):
        """Map a linear predictor value back to the response mean scale."""
        return 1.0 / (1 + np.exp(-eta))

    def deriv_inverse_link(self, eta):
        p = self.inverse_link(eta)
        return p * (1 - p)

    def deriv2_inverse_link(self, eta):
        p = self.inverse_link(eta)
        return p * (1 - p) * (1 - 2 * p)

    def deriv(self, mu):
        """Evaluate the first derivative of the link function."""
        return 1.0 / (mu * (1.0 - mu))

    def deriv2(self, mu):
        """Evaluate the second derivative of the link function."""
        return -1.0 / (mu * (1.0 - mu)) ** 2 * (1 - 2 * mu)

    @classmethod
    def function_str(cls):
        """Return a short mathematical description of the link function."""
        return "g(x) = log(x/(1-x))"

    @classmethod
    def mean_function_str(cls):
        """Return a short mathematical description of the inverse-link mean function."""
        return "E[y|x;b] = exp[x' b] / (1 + exp[x' b])"

    @classmethod
    def name(cls):
        """Return the canonical uppercase name for this link or family."""
        return LOGIT


class NegativeBinomialCanonicalLink(Link):
    """Canonical negative-binomial link parameterized by overdispersion ``alpha``.

    For variance ``V(mu) = mu + alpha * mu**2``, the canonical parameter is
    ``theta = log(mu / (mu + 1 / alpha))``.  This link is less commonly used
    than the log link but is included for exponential-family completeness.
    """

    def __init__(self, alpha):
        """Initialize the canonical negative-binomial link.

        Args:
            alpha: Positive overdispersion parameter.
        """
        self.alpha = alpha

    def link(self, mu):
        """Map a response-scale mean value to the linear predictor scale."""
        return np.log(mu / (mu + 1.0 / self.alpha))

    def inverse_link(self, eta):
        """Map a linear predictor value back to the response mean scale."""
        e_theta = np.exp(eta)
        return e_theta * (1.0 / self.alpha) / (1 - e_theta)

    def deriv_inverse_link(self, eta):
        """Map a linear predictor value back to the response mean scale."""
        e_theta = np.exp(eta)
        return e_theta / (self.alpha * (1 - e_theta) ** 2)

    def deriv2_inverse_link(self, eta):
        e_theta = np.exp(eta)
        return e_theta * (1 + e_theta) / (self.alpha * (1 - e_theta) ** 3)

    def deriv(self, mu):
        """Evaluate the first derivative of the link function."""
        return 1.0 / mu - 1.0 / (mu + 1.0 / self.alpha)

    def deriv2(self, mu):
        """Evaluate the second derivative of the link function."""
        return -1.0 / mu ** 2 + 1.0 / (mu + 1.0 / self.alpha) ** 2

    @classmethod
    def function_str(cls):
        """Return a short mathematical description of the link function."""
        return "g(x) = log(x / (x + 1/alpha))"

    @classmethod
    def mean_function_str(cls):
        """Return a short mathematical description of the inverse-link mean function."""
        return "E[y|x;b,alpha] = exp(x'b)/(1-exp(x'b)) * 1/alpha"

    @classmethod
    def name(cls):
        """Return the canonical uppercase name for this link or family."""
        return NEGATIVE_BINOMIAL_CANONICAL_LINK


class NegativeInverse(Link):
    """Negative inverse link, g(mu) = -1 / mu."""

    def link(self, mu):
        """Map a response-scale mean value to the linear predictor scale."""
        return -1.0 / mu

    def inverse_link(self, eta):
        """Map a linear predictor value back to the response mean scale."""
        return -1.0 / eta

    def deriv_inverse_link(self, eta):
        """Map a linear predictor value back to the response mean scale."""
        return 1.0 / eta ** 2

    def deriv2_inverse_link(self, eta):
        """Map a linear predictor value back to the response mean scale."""
        return -2.0 / eta ** 3

    def deriv(self, mu):
        """Evaluate the first derivative of the link function."""
        return 1.0 / mu ** 2

    def deriv2(self, mu):
        """Evaluate the second derivative of the link function."""
        return -2 * mu ** -3

    @classmethod
    def function_str(cls):
        """Return a short mathematical description of the link function."""
        return "g(x) = -1/x"

    @classmethod
    def mean_function_str(cls):
        """Return a short mathematical description of the inverse-link mean function."""
        return "E[y|x;b] = -1 / (x'b)"

    @classmethod
    def name(cls):
        """Return the canonical uppercase name for this link or family."""
        return NEGATIVE_INVERSE


class Inverse(Link):
    """Inverse link, g(mu) = 1 / mu."""

    def link(self, mu):
        """Map a response-scale mean value to the linear predictor scale."""
        return 1.0 / mu

    def inverse_link(self, eta):
        """Map a linear predictor value back to the response mean scale."""
        return 1.0 / eta

    def deriv_inverse_link(self, eta):
        """Map a linear predictor value back to the response mean scale."""
        return -1.0 / eta ** 2

    def deriv2_inverse_link(self, eta):
        """Map a linear predictor value back to the response mean scale."""
        return 2.0 / eta ** 3

    def deriv(self, mu):
        """Evaluate the first derivative of the link function."""
        return -1.0 / mu ** 2

    def deriv2(self, mu):
        """Evaluate the second derivative of the link function."""
        return 2 * mu ** -3

    @classmethod
    def function_str(cls):
        """Return a short mathematical description of the link function."""
        return "g(x) = 1/x"

    @classmethod
    def mean_function_str(cls):
        """Return a short mathematical description of the inverse-link mean function."""
        return "E[y|x;b] = 1 / (x'b)"

    @classmethod
    def name(cls):
        """Return the canonical uppercase name for this link or family."""
        return INVERSE


class Exponential(Link):
    """Exponential link, g(mu) = exp(mu)."""

    def link(self, mu):
        """Map a response-scale mean value to the linear predictor scale."""
        return np.exp(mu)

    def inverse_link(self, eta):
        """Map a linear predictor value back to the response mean scale."""
        return np.log(eta)

    def deriv_inverse_link(self, eta):
        """Map a linear predictor value back to the response mean scale."""
        return 1.0 / eta

    def deriv2_inverse_link(self, eta):
        """Map a linear predictor value back to the response mean scale."""
        return -1.0 / eta ** 2

    def deriv(self, mu):
        """Evaluate the first derivative of the link function."""
        return np.exp(mu)

    def deriv(self, mu):
        """Evaluate the first derivative of the link function."""
        return np.exp(mu)

    def deriv2(self, mu):
        """Evaluate the second derivative of the link function."""
        return np.exp(mu)

    @classmethod
    def function_str(cls):
        """Return a short mathematical description of the link function."""
        return "g(x) = exp(x)"

    @classmethod
    def mean_function_str(cls):
        """Return a short mathematical description of the inverse-link mean function."""
        return "E[y|x;b] = log(x' b)"

    @classmethod
    def name(cls):
        """Return the canonical uppercase name for this link or family."""
        return EXPONENTIAL


class NegativeTwoInverseSquared(Link):
    """Negative half inverse-squared link for inverse Gaussian models."""

    def link(self, mu):
        """Map a response-scale mean value to the linear predictor scale."""
        return -.5 * mu ** -2.0

    def inverse_link(self, eta):
        """Map a linear predictor value back to the response mean scale."""
        return 1.0 / np.sqrt(-2 * eta)

    def deriv_inverse_link(self, eta):
        """Map a linear predictor value back to the response mean scale."""
        return (-2 * eta) ** -1.5

    def deriv2_inverse_link(self, eta):
        """Map a linear predictor value back to the response mean scale."""
        return -1.5 * (-2 * eta) ** -2.5

    def deriv(self, mu):
        """Evaluate the first derivative of the link function."""
        return mu ** -3

    def deriv2(self, mu):
        """Evaluate the second derivative of the link function."""
        return -3.0 * mu ** -4.0

    @classmethod
    def function_str(cls):
        """Return a short mathematical description of the link function."""
        return "g(x) = -1/(2 x^2)"

    @classmethod
    def mean_function_str(cls):
        """Return a short mathematical description of the inverse-link mean function."""
        return "1.0 / sqrt[-2 * (x'b)]"

    @classmethod
    def name(cls):
        """Return the canonical uppercase name for this link or family."""
        return NEGATIVE_TWO_INVERSE_SQUARED


class InverseSquared(Link):
    """Inverse-squared link, g(mu) = 1 / mu**2."""

    def link(self, mu):
        """Map a response-scale mean value to the linear predictor scale."""
        return 1.0 / mu ** 2

    def inverse_link(self, eta):
        """Map a linear predictor value back to the response mean scale."""
        return 1.0 / np.sqrt(eta)

    def deriv_inverse_link(self, eta):
        """Map a linear predictor value back to the response mean scale."""
        return -0.5 / eta ** 1.5

    def deriv2_inverse_link(self, eta):
        """Map a linear predictor value back to the response mean scale."""
        return 0.75 / eta ** 2.5

    def deriv(self, mu):
        """Evaluate the first derivative of the link function."""
        return -2. * mu ** -3

    def deriv2(self, mu):
        """Evaluate the second derivative of the link function."""
        return 6. * mu ** -4

    @classmethod
    def function_str(cls):
        """Return a short mathematical description of the link function."""
        return "g(x) = 1/x^2"

    @classmethod
    def mean_function_str(cls):
        """Return a short mathematical description of the inverse-link mean function."""
        return "1/ sqrt[x' b]"

    @classmethod
    def name(cls):
        """Return the canonical uppercase name for this link or family."""
        return INVERSE_SQUARED


class Probit(Link):
    """Probit link using the standard normal quantile function.

    Maps probabilities through ``Phi^{-1}``; the inverse link is the standard
    normal CDF.  Useful for binary response models with normal latent-error
    interpretation.
    """

    def link(self, mu):
        """Map a response-scale mean value to the linear predictor scale."""
        return norm.ppf(mu)

    def inverse_link(self, eta):
        """Map a linear predictor value back to the response mean scale."""
        return norm.cdf(eta)

    def deriv_inverse_link(self, eta):
        """Map a linear predictor value back to the response mean scale."""
        return norm.pdf(eta)

    def deriv(self, mu):
        """Evaluate the first derivative of the link function."""
        return 1.0 / norm.pdf(norm.ppf(mu))

    def deriv2(self, mu):
        """Evaluate the second derivative of the link function."""
        ppf_ = norm.ppf(mu)
        return ppf_ / norm.pdf(ppf_) ** 2

    @classmethod
    def function_str(cls):
        """Return a short mathematical description of the link function."""
        return "g(x) = Phi^{-1}(x)"

    @classmethod
    def mean_function_str(cls): "Phi(x'b)"

    @classmethod
    def name(cls):
        """Return the canonical uppercase name for this link or family."""
        return PROBIT


class Cauchy(Link):
    """Cauchy CDF link for binary-response GLMs.

    Uses the Cauchy quantile/CDF pair.  Compared with logit/probit, the heavier
    Cauchy tails can make fitted probabilities approach 0 or 1 more slowly.
    """

    def link(self, mu):
        """Map a response-scale mean value to the linear predictor scale."""
        return cauchy.ppf(mu)

    def inverse_link(self, eta):
        """Map a linear predictor value back to the response mean scale."""
        return cauchy.cdf(eta)

    def deriv_inverse_link(self, eta):
        """Map a linear predictor value back to the response mean scale."""
        return cauchy.pdf(eta)

    def deriv(self, mu):
        """Evaluate the first derivative of the link function."""
        return 1.0 / cauchy.pdf(cauchy.ppf(mu))

    # @classmethod
    # def deriv2(cls, mu):
    #     TODO
    #     complicated

    @classmethod
    def function_str(cls):
        """Return a short mathematical description of the link function."""
        return "g(x) = CauchyInverseCdf(x)"

    @classmethod
    def mean_function_str(cls):
        """Return a short mathematical description of the inverse-link mean function."""
        return "E[y|x;b] = CauchyCdf(x'b)"

    @classmethod
    def name(cls):
        """Return the canonical uppercase name for this link or family."""
        return CAUCHY


class CLogLog(Link):
    """Complementary log-log link, ``g(mu) = log(-log(1 - mu))``.

    This asymmetric binary-response link is often used when the probability of
    an event is generated by an underlying extreme-value process.
    """

    def link(self, mu):
        """Map a response-scale mean value to the linear predictor scale."""
        return np.log(-np.log(1 - mu))

    def inverse_link(self, eta):
        """Map a linear predictor value back to the response mean scale."""
        return 1 - np.exp(-np.exp(eta))

    def deriv_inverse_link(self, eta):
        e_eta = np.exp(eta)
        return e_eta * np.exp(-e_eta)

    def deriv2_inverse_link(self, eta):
        e_eta = np.exp(eta)
        return e_eta * np.exp(-e_eta) * (1 - e_eta)

    def deriv(self, mu):
        """Evaluate the first derivative of the link function."""
        return -1.0 / (np.log(1 - mu) * (1 - mu))

    def deriv2(self, mu):
        """Evaluate the second derivative of the link function."""
        log_term = np.log(1 - mu)
        return -(log_term + 1) / ((1 - mu) * log_term) ** 2

    @classmethod
    def function_str(cls):
        """Return a short mathematical description of the link function."""
        return "g(x) = log(-log(1-x))"

    @classmethod
    def mean_function_str(cls):
        """Return a short mathematical description of the inverse-link mean function."""
        return "1 - exp(-exp(x' b))"

    @classmethod
    def name(cls):
        """Return the canonical uppercase name for this link or family."""
        return CLOGLOG


class Power(Link):
    """Power link, ``g(mu) = mu ** power``.

    Args:
        power: Exponent applied by the link.  The inverse link raises ``eta`` to
            ``1 / power``.
    """

    def __init__(self, power=1):
        """Initialize a power link.

        Args:
            power: Exponent used in ``mu ** power``.
        """
        self.power = power

    def link(self, mu):
        """Map a response-scale mean value to the linear predictor scale."""
        return mu ** self.power

    def inverse_link(self, eta):
        """Map a linear predictor value back to the response mean scale."""
        return eta ** (1 / self.power)

    def deriv_inverse_link(self, eta):
        """Map a linear predictor value back to the response mean scale."""
        return (1 / self.power) * eta ** (1 / self.power - 1)

    def deriv2_inverse_link(self, eta):
        return ((1 / self.power) * (1 / self.power - 1)) * eta ** (1 / self.power - 2)

    def deriv(self, mu):
        """Evaluate the first derivative of the link function."""
        return self.power * mu ** (self.power - 1)

    def deriv2(self, mu):
        """Evaluate the second derivative of the link function."""
        return -2 * mu ** -3

    @classmethod
    def mean_function_str(cls):
        """Return a short mathematical description of the inverse-link mean function."""
        return "(x'beta) ** (1/power)"

    @classmethod
    def name(cls):
        """Return the canonical uppercase name for this link or family."""
        return POWER


class Sqrt(Link):
    """Square-root link for non-negative means."""

    def link(self, mu):
        """Map a response-scale mean value to the linear predictor scale."""
        return np.sqrt(mu)

    def inverse_link(self, eta):
        """Map a linear predictor value back to the response mean scale."""
        return eta ** 2

    def deriv_inverse_link(self, eta):
        """Map a linear predictor value back to the response mean scale."""
        return 2 * eta

    def deriv2_inverse_link(self, eta):
        """Map a linear predictor value back to the response mean scale."""
        if isinstance(eta, (int, float, np.integer, np.floating)):
            return 2.0
        else:
            return np.full(eta.shape, 2.0)

    def deriv(self, mu):
        """Evaluate the first derivative of the link function."""
        return .5 / np.sqrt(mu)

    def deriv2(self, mu):
        """Evaluate the second derivative of the link function."""
        return -.25 * mu ** (-1.5)

    @classmethod
    def function_str(cls):
        """Return a short mathematical description of the link function."""
        return "g(x) = sqrt(x)"

    @classmethod
    def mean_function_str(cls):
        """Return a short mathematical description of the inverse-link mean function."""
        return "(x'b) ** 2"

    @classmethod
    def name(cls):
        """Return the canonical uppercase name for this link or family."""
        return SQRT


LINK_NAME_2_CLS = {
    l.name(): l for l in [Logit, Probit, Identity, Log, Exponential, NegativeInverse, NegativeTwoInverseSquared,
                          Inverse, InverseSquared, CLogLog, Sqrt, Cauchy, NegativeBinomialCanonicalLink]
}

LINK_NO_UNDERSCORE_TO_UNDERSCORE = {f.replace('_', ''): f for f in LINK_NAME_2_CLS.keys()}


def _get_link(link):
    """Resolve user link input to a concrete ``Link`` instance.

    Args:
        link: Link name string (case-insensitive, underscores optional), a
            ``Link`` subclass, or an already-instantiated ``Link`` object.

    Returns:
        Concrete ``Link`` object.

    Raises:
        Exception: If the link cannot be resolved.
    """
    if isinstance(link, str):
        try:
            # Normalize names so ``logit``, ``LOGIT``, and spellings with or
            # without underscores hit the same registry entry.
            link = LINK_NAME_2_CLS[LINK_NO_UNDERSCORE_TO_UNDERSCORE[link.replace('_', '').upper()]]()
        except:
            raise Exception("Link '%s' not found!" % link)
    elif isinstance(link, Link):
        pass
    elif issubclass(link, Link):
        link = link()
    else:
        raise Exception("Link '%s' not found!" % str(link))
    return link


def is_overridden(cls, method_name):
    # Retrieve the class that actually implements the method using MRO
    # We unwrap the method descriptor using .__func__ to compare the raw underlying functions
    implementing_class = cls.__mro__

    # We need to find the first class in the MRO chain that provides this method
    for base_cls in cls.__mro__:
        if method_name in base_cls.__dict__:
            return base_cls is cls

    return False

# for c, v in LINK_NAME_2_CLS.items():
#     t = is_overridden(v, 'deriv_inverse_link')
#     if not t:
#         print(c, t)
#
# for c, v in LINK_NAME_2_CLS.items():
#     t = is_overridden(v, 'deriv2_inverse_link')
#     if not t:
#         print(c, t)
