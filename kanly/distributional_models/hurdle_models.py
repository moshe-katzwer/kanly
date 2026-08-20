"""Two-part hurdle models estimated through separable components.

The zero hurdle can use either a Bernoulli/logit model for
``P(Y = 0 | exog_infl)`` or a statsmodels-compatible right-censored Poisson
model. The latter is fitted through its equivalent Bernoulli complementary-
log-log representation for ``P(Y > 0 | exog_infl)``. Conditional on crossing
the hurdle, the response is modeled over the strictly positive observations.
Gaussian and lognormal components use OLS; Poisson, Gamma, and Inverse
Gaussian use GLMs; negative-binomial-P uses an exact zero-truncated likelihood
so its dispersion can be estimated. Because the likelihood separates, the two
components are fitted independently and their coefficient vectors and
covariance matrices are combined afterwards.

Parameter order follows the count-model convention used by the zero-inflated
classes: all positive-response parameters first, followed by zero-process
coefficients prefixed with ``hurdle_``.
"""

from __future__ import absolute_import, print_function

from abc import abstractmethod
from collections import Counter
import time
import warnings

import numpy as np
from scipy.sparse import csc_matrix, isspmatrix
from scipy.special import digamma, expit, gammaln

from kanly.bootstrap.bootstrap import (
    DEFAULT_BB_ALPHA,
    DEFAULT_BB_SEED,
    DEFAULT_BOOTSTRAP_N_SAMPLES,
    get_bayesian_bootstrap_weights,
    get_bootstrap_weights2,
)
from kanly.distributional_models.results import DistributionalModelResults
from kanly.distributional_models.base import (
    _NonnegativeDistributionalModel,
    _as_design_matrix,
    _build_score_obs,
    _is_constant_column,
    _score_meat,
    _sum_score_obs,
    _weight_score_obs,
)
from kanly.distributional_models.two_part import TwoPartModel
from kanly.regression.generalized_linear_models.families import (
    Bernoulli,
    Gamma as GammaFamily,
    InverseGaussian as InverseGaussianFamily,
    ZeroTruncatedPoisson,
    _get_family_and_link,
)
from kanly.regression.generalized_linear_models.links import (
    CLogLog,
    Log,
    Logit,
)
from kanly.regression.generalized_linear_models.model import (
    SparseGeneralizedLinearModel,
)
from kanly.regression.linear_models.model import SparseLinearModel


# Hurdle fits use the shared distributional-model result implementation.
HurdleModelResults = DistributionalModelResults


_POISSON_LIMIT_LOG_ALPHA = float(np.log(1e-8))
_ZERO_MODEL_ALIASES = {
    'logit': 'logit',
    'logistic': 'logit',
    'poisson': 'poisson',
    'censoredpoisson': 'poisson',
    'rightcensoredpoisson': 'poisson',
}


def _normalize_zero_model(zero_model):
    """Return the canonical hurdle zero-model name."""
    if not isinstance(zero_model, str):
        raise TypeError('zero_model must be a string')
    normalized = ''.join(
        character for character in zero_model.lower()
        if character.isalnum()
    )
    canonical = _ZERO_MODEL_ALIASES.get(normalized)
    if canonical is None:
        raise ValueError("zero_model must be 'logit' or 'poisson'")
    return canonical


class _ZeroTruncatedNegativeBinomialP(_NonnegativeDistributionalModel):
    """Exact NB-P regression conditional on observing a positive count.

    This private component model exists to support
    :class:`NegativeBinomialPHurdle`. Its regression predictor defines the
    mean ``mu = exp(X beta)`` of the *untruncated* NB-P distribution and its
    variance is ``mu + alpha * mu**p``. The fitted positive-response mean is
    therefore ``mu / (1 - P(Y=0))``. Parameter order is ``beta`` followed by
    ``log_alpha``.
    """

    def __init__(self, endog, exog, *args, p=2, **kwargs):
        """Validate ``p`` and initialize a positive-count component model."""
        if isinstance(p, bool) or p not in (1, 2):
            raise ValueError('p must be either 1 (NB1) or 2 (NB2)')
        self.p = int(p)
        self.negative_binomial_p = self.p
        values = np.asarray(endog, dtype=float)
        if (np.any(~np.isfinite(values)) or np.any(values <= 0.0)
                or np.any(values != np.floor(values))):
            raise ValueError(
                'Zero-truncated negative-binomial outcomes must be finite '
                'positive integers'
            )
        super().__init__(endog, exog, *args, **kwargs)

    def get_param_names(self):
        """Return regression coefficient names followed by ``log_alpha``."""
        return self._get_regression_param_names() + ['log_alpha']

    def get_start_params(self):
        """Return stable log-mean and conservative dispersion starts."""
        mean, variance = self._response_moments()
        safe_mean = max(mean, 1.0 + 1e-6)
        if self.p == 1:
            alpha = (variance - safe_mean) / safe_mean
        else:
            alpha = (variance - safe_mean) / safe_mean ** 2
        if not np.isfinite(alpha) or alpha <= 0.0:
            alpha = 0.25
        return np.append(
            self._mean_regression_start(safe_mean),
            self._log_dispersion_start(alpha),
        )

    @staticmethod
    def _poisson_limit_terms(endog, eta):
        """Return zero-truncated Poisson likelihood and score limits."""
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            mu = np.exp(eta)
            log_survival = np.log(-np.expm1(-mu))
            loglike = (
                endog * eta - mu - gammaln(endog + 1.0)
                - log_survival
            )
            zero_odds = 1.0 / np.expm1(mu)
            d_eta = endog - mu - mu * zero_odds
        valid = (
            np.isfinite(eta) & np.isfinite(mu)
            & np.isfinite(loglike) & np.isfinite(d_eta)
        )
        return (
            np.where(valid, loglike, -np.inf),
            np.where(valid, d_eta, np.nan),
        )

    def _distribution_terms(self, params):
        """Return ordinary NB-P and zero-mass likelihood derivatives."""
        params = np.asarray(params, dtype=float).reshape(-1)
        if len(params) != len(self.param_names):
            raise ValueError(
                f'Expected {len(self.param_names)} parameters but received '
                f'{len(params)}'
            )
        beta = params[:-1]
        log_alpha = float(params[-1])
        eta = self.exog @ beta

        if log_alpha <= _POISSON_LIMIT_LOG_ALPHA:
            loglike, d_eta = self._poisson_limit_terms(self.endog, eta)
            zeros = np.zeros(self.nobs, dtype=float)
            return loglike, d_eta, zeros

        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            mu = np.exp(eta)
            alpha = np.exp(log_alpha)
            if self.p == 1:
                size = np.exp(eta - log_alpha)
                log_denom = np.logaddexp(0.0, log_alpha)
                alpha_ratio = expit(log_alpha)
                ordinary_loglike = (
                    gammaln(self.endog + size)
                    - gammaln(size)
                    - gammaln(self.endog + 1.0)
                    - size * log_denom
                    + self.endog * (log_alpha - log_denom)
                )
                ordinary_d_eta = size * (
                    digamma(self.endog + size)
                    - digamma(size) - log_denom
                )
                ordinary_d_log_alpha = (
                    -ordinary_d_eta
                    + self.endog * (1.0 - alpha_ratio)
                    - size * alpha_ratio
                )
                log_pzero = -size * log_denom
                zero_d_eta = log_pzero
                zero_d_log_alpha = size * (
                    log_denom - alpha_ratio
                )
            else:
                size = np.exp(-log_alpha)
                z = eta + log_alpha
                log_denom = np.logaddexp(0.0, z)
                mean_ratio = expit(z)
                ordinary_loglike = (
                    gammaln(self.endog + size)
                    - gammaln(size)
                    - gammaln(self.endog + 1.0)
                    - size * log_denom
                    + self.endog * (z - log_denom)
                )
                ordinary_d_eta = (
                    self.endog - (self.endog + size) * mean_ratio
                )
                ordinary_d_log_alpha = (
                    ordinary_d_eta
                    + size * (
                        digamma(size) - digamma(self.endog + size)
                        + log_denom
                    )
                )
                log_pzero = -size * log_denom
                zero_d_eta = -size * mean_ratio
                zero_d_log_alpha = size * (
                    log_denom - mean_ratio
                )

            log_survival = np.log(-np.expm1(log_pzero))
            zero_odds = 1.0 / np.expm1(-log_pzero)
            loglike = ordinary_loglike - log_survival
            d_eta = ordinary_d_eta + zero_odds * zero_d_eta
            d_log_alpha = (
                ordinary_d_log_alpha
                + zero_odds * zero_d_log_alpha
            )

        valid = (
            np.isfinite(mu) & (mu > 0.0)
            & np.isfinite(alpha) & (alpha > 0.0)
            & np.isfinite(size) & (size > 0.0)
            & np.isfinite(loglike)
            & np.isfinite(d_eta) & np.isfinite(d_log_alpha)
        )
        return (
            np.where(valid, loglike, -np.inf),
            np.where(valid, d_eta, np.nan),
            np.where(valid, d_log_alpha, np.nan),
        )

    def loglike_obs(self, params, *args, **kwargs):
        """Return unweighted zero-truncated NB-P log likelihoods."""
        del args, kwargs
        return self._distribution_terms(params)[0]

    def score_obs(self, params, *args, **kwargs):
        """Return unweighted scores for ``beta`` and ``log_alpha``."""
        del args, kwargs
        return _build_score_obs(self._score_blocks(params))

    def _score_blocks(self, params):
        """Return factorized blocks used to build score columns directly."""
        _, d_eta, d_log_alpha = self._distribution_terms(params)
        return (
            (self.exog, d_eta),
            (d_log_alpha, None),
        )

    def score(self, params, *args, **kwargs):
        """Return the importance-weighted analytical aggregate score."""
        del args, kwargs
        values = self.score_obs(params)
        return _sum_score_obs(values, weights=self.weights)

    def log_zero_probability(self, params, exog=None):
        """Return the underlying NB-P log probability of a zero count."""
        params = np.asarray(params, dtype=float).reshape(-1)
        exog = self.exog if exog is None else _as_design_matrix(exog)
        eta = exog @ params[:-1]
        log_alpha = float(params[-1])
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            mu = np.exp(eta)
            if log_alpha <= _POISSON_LIMIT_LOG_ALPHA:
                log_pzero = -mu
            elif self.p == 1:
                size = np.exp(eta - log_alpha)
                log_pzero = -size * np.logaddexp(0.0, log_alpha)
            else:
                size = np.exp(-log_alpha)
                log_pzero = -size * np.logaddexp(
                    0.0, eta + log_alpha
                )
        return log_pzero

    def zero_probability(self, params, exog=None):
        """Return the underlying NB-P probability of a zero count."""
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            return np.exp(self.log_zero_probability(params, exog=exog))


class HurdleModel(TwoPartModel):
    """Base class for separable zero-hurdle and positive-response models.

    ``exog`` controls the positive conditional response.  ``exog_infl``
    controls the zero hurdle and defaults to a single constant. By default it
    models ``P(Y=0)`` through a Bernoulli/logit GLM. With
    ``zero_model='poisson'``, it instead models a latent Poisson rate whose
    zero probability is ``exp(-exp(exog_infl @ gamma))``. Observation
    ``weights`` multiply the component estimating equations as importance
    weights; the positive component receives the subset associated with
    positive responses.

    Subclasses specify the positive component. The inherited formula builder
    accepts ``exog_infl`` as a Patsy-style right-hand-side formula and rejects
    instruments through the distributional-model parser.
    """

    _requires_both_outcome_parts = True
    is_quasi_likelihood = False

    def __init__(self, *args, zero_model='logit', **kwargs):
        """Initialize aligned hurdle data and select the zero process."""
        self.zero_model = _normalize_zero_model(zero_model)
        self._hurdle_link_instance = (
            Logit() if self.zero_model == 'logit' else CLogLog()
        )
        super().__init__(*args, **kwargs)
        self.is_zero = self.endog == 0.0
        self.is_positive = ~self.is_zero
        self.nobs_zero = int(np.count_nonzero(self.is_zero))
        self.nobs_positive = int(np.count_nonzero(self.is_positive))

        if self.nobs_zero == 0 or self.nobs_positive == 0:
            raise ValueError(
                'Hurdle models require at least one zero and one strictly '
                'positive outcome'
            )
        if self.is_weighted:
            if np.dot(self.weights, self.is_zero) <= 0.0:
                raise ValueError(
                    'Zero outcomes must have positive total weight'
                )
            if np.dot(self.weights, self.is_positive) <= 0.0:
                raise ValueError(
                    'Positive outcomes must have positive total weight'
                )

        # Positive-row arrays are sliced lazily and cached. Component models
        # receive these exact objects, so repeated likelihood evaluations,
        # bootstrap refits, GLM fits, and LM fits do not reslice the parent
        # design. GLM responses remain dense as required by the native GLM
        # internals; the OLS subclass separately caches its sparse response.
        self._positive_row_indices = np.flatnonzero(self.is_positive)
        self.__positive_endog = None
        self.__positive_exog = None
        self.__positive_weights = None
        self._zero_indicator = self.is_zero.astype(float)
        self._positive_indicator = self.is_positive.astype(float)
        self._positive_scale = None
        self.positive_model_fit = None
        self.hurdle_model_fit = None

    @property
    def _positive_endog(self):
        """Return one cached dense positive-response subset."""
        if self.__positive_endog is None:
            self.__positive_endog = self.endog[self._positive_row_indices]
        return self.__positive_endog

    @property
    def _positive_exog(self):
        """Return one cached positive-row design subset."""
        if self.__positive_exog is None:
            self.__positive_exog = self.exog[self._positive_row_indices]
        return self.__positive_exog

    @property
    def _positive_weights(self):
        """Return one cached positive-row weight subset."""
        if self.weights is None:
            return None
        if self.__positive_weights is None:
            self.__positive_weights = self.weights[
                self._positive_row_indices
            ]
        return self.__positive_weights

    @property
    def _positive_component_endog(self):
        """Return the cached dense response expected by GLM internals."""
        return self._positive_endog

    @property
    def _hurdle_component_endog(self):
        """Return the binary response used by the selected hurdle GLM."""
        if self.zero_model == 'logit':
            return self._zero_indicator
        return self._positive_indicator

    @property
    def hurdle_link(self):
        """Return the GLM link equivalent to the selected zero process."""
        return self._hurdle_link_instance

    def _hurdle_component_description(self):
        """Return a concise description of the selected zero process."""
        if self.zero_model == 'logit':
            return 'BERNOULLI / LOGIT FOR P(Y=0)'
        return (
            'RIGHT-CENSORED POISSON '
            '(BERNOULLI / CLOGLOG FOR P(Y>0))'
        )

    def _hurdle_component_endog_name(self):
        """Return a response name matching the fitted binary orientation."""
        if self.endog_name is None:
            return None
        suffix = 'is_zero' if self.zero_model == 'logit' else 'is_positive'
        return f'{self.endog_name}_{suffix}'

    def _hurdle_probabilities_from_eta(self, hurdle_eta):
        """Return zero and positive probabilities from a hurdle predictor."""
        hurdle_eta = np.asarray(hurdle_eta, dtype=float)
        if self.zero_model == 'logit':
            zero_probability = expit(hurdle_eta)
            positive_probability = expit(-hurdle_eta)
        else:
            with np.errstate(over='ignore', under='ignore', invalid='ignore'):
                rate = np.exp(hurdle_eta)
                zero_probability = np.exp(-rate)
                positive_probability = -np.expm1(-rate)
        return zero_probability, positive_probability

    def _hurdle_loglike_from_eta(self, hurdle_eta):
        """Return unweighted binary hurdle log-likelihood contributions."""
        hurdle_eta = np.asarray(hurdle_eta, dtype=float)
        if self.zero_model == 'logit':
            return np.where(
                self.is_zero,
                -np.logaddexp(0.0, -hurdle_eta),
                -np.logaddexp(0.0, hurdle_eta),
            )

        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            rate = np.exp(hurdle_eta)
            log_positive_probability = np.log(-np.expm1(-rate))
        return np.where(
            self.is_zero, -rate, log_positive_probability
        )

    def _hurdle_score_factor(self, hurdle_eta):
        """Return the hurdle log-likelihood derivative with respect to eta."""
        hurdle_eta = np.asarray(hurdle_eta, dtype=float)
        if self.zero_model == 'logit':
            zero_probability, _ = self._hurdle_probabilities_from_eta(
                hurdle_eta
            )
            return self._zero_indicator - zero_probability

        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            rate = np.exp(hurdle_eta)
            positive_factor = rate / np.expm1(rate)
        positive_factor = np.where(rate == 0.0, 1.0, positive_factor)
        positive_factor = np.where(
            np.isposinf(rate), 0.0, positive_factor
        )
        return np.where(self.is_zero, -rate, positive_factor)

    @property
    @abstractmethod
    def positive_family(self):
        """Return the GLM family instance for positive responses."""
        raise NotImplementedError

    @property
    @abstractmethod
    def positive_link(self):
        """Return the GLM link instance for positive responses."""
        raise NotImplementedError

    def _get_positive_param_names(self):
        """Return parameter names belonging to the positive component."""
        return self._get_regression_param_names()

    def _positive_component_description(self):
        """Return a concise family/link description for diagnostics."""
        return (
            f'{self.positive_family.name()} / '
            f'{self.positive_link.name()} GLM'
        )

    def _get_positive_start_params(self):
        """Return family-aware starts for the positive component."""
        positive_endog = self._positive_endog
        positive_weights = self._positive_weights
        positive_intercept = self.positive_family.get_starting_intercept(
            positive_endog,
            var_weights=positive_weights,
            link=self.positive_link,
        )
        return self._constant_predictor_start(
            self._positive_exog, positive_intercept
        )

    def _positive_loglike_obs(self, params, positive_scale):
        """Return positive-component likelihood contributions."""
        positive_eta = self._positive_exog @ params
        positive_mu = self.positive_link.inverse_link(positive_eta)
        positive_theta = self.positive_family.b_deriv_inv(positive_mu)
        return self.positive_family.log_likelihood_obs(
            self._positive_endog,
            positive_theta,
            scale=positive_scale,
            var_weights=1.0,
        )

    def _positive_score_blocks(self, params, positive_scale):
        """Return factorized positive-component score columns."""
        positive_eta = self._positive_exog @ params
        positive_mu = self.positive_link.inverse_link(positive_eta)
        variance = self.positive_family.variance(positive_mu)
        link_derivative = self.positive_link.deriv(positive_mu)
        positive_factor = (
            self._positive_endog - positive_mu
        ) / (positive_scale * variance * link_derivative)
        return ((self._positive_exog, positive_factor),)

    def _positive_score_obs(self, params, positive_scale):
        """Return positive-component observation scores."""
        return _build_score_obs(
            self._positive_score_blocks(params, positive_scale)
        )

    def _positive_conditional_mean(self, params, exog):
        """Return ``E[Y | Y>0, X]`` for the positive component."""
        return self.positive_link.inverse_link(exog @ params)

    def _positive_underlying_mean(self, params, exog):
        """Return the positive component's untruncated/reference mean."""
        return self._positive_conditional_mean(params, exog)

    def _fit_positive_component(
            self, weights, start_params, fit_kwargs,
            first_column_constant):
        """Fit the default positive component as a GLM."""
        return SparseGeneralizedLinearModel.GLM(
            self._positive_component_endog,
            self._positive_exog,
            family=self.positive_family,
            link=self.positive_link,
            var_weights=weights,
            exog_names=self._get_positive_param_names(),
            endog_name=self.endog_name,
            var_weights_name=self.weights_name,
            start_params=start_params,
            first_column_constant=first_column_constant,
            **fit_kwargs,
        )

    def _positive_fit_params(self, fit):
        """Return positive-component parameters to merge into the hurdle fit."""
        return np.asarray(fit.params, dtype=float).reshape(-1)

    def _positive_fit_covariance(self, fit):
        """Return the covariance for the merged positive parameters."""
        return self._component_covariance(fit)

    @staticmethod
    def _positive_fit_converged(fit):
        """Return component convergence, treating closed-form fits as solved."""
        return bool(getattr(fit, 'converged', True))

    @staticmethod
    def _positive_fit_scale(fit):
        """Return the positive-component scale used by hurdle likelihoods."""
        return float(fit.scale)

    def _quasi_likelihood_footer(self):
        """Return the default quasi-likelihood summary explanation."""
        return (
            'The positive component uses GLM estimating equations with '
            'Pearson-estimated scale. This is a quasi-likelihood fit; '
            'likelihood-based AIC and BIC are not reported.'
        )

    def _get_inflation_param_names(self):
        """Return hurdle-prefixed names for zero-process coefficients."""
        if self.exog_infl_names is not None:
            return [f'hurdle_{name}' for name in self.exog_infl_names]
        if (self.k_inflate == 1
                and _is_constant_column(self.exog_infl, value=1.0)):
            return ['hurdle_const']
        return [f'hurdle_x{i}' for i in range(self.k_inflate)]

    def get_param_names(self):
        """Return positive coefficients followed by hurdle coefficients."""
        return (
            self._get_positive_param_names()
            + self._get_inflation_param_names()
        )

    def get_start_params(self):
        """Return family-aware positive and empirical hurdle starts."""
        positive_start = self._get_positive_start_params()

        if self.weights is None:
            zero_probability = float(np.mean(self.is_zero))
        else:
            zero_probability = float(
                np.dot(self.weights, self._zero_indicator)
                / np.sum(self.weights)
            )
        zero_probability = float(np.clip(zero_probability, 1e-4, 1 - 1e-4))
        if self.zero_model == 'logit':
            hurdle_intercept = (
                np.log(zero_probability) - np.log1p(-zero_probability)
            )
        else:
            hurdle_intercept = np.log(-np.log(zero_probability))
        hurdle_start = self._constant_predictor_start(
            self.exog_infl, hurdle_intercept
        )
        return np.concatenate((positive_start, hurdle_start))

    @property
    def k_positive(self):
        """Return the number of positive-response parameters."""
        return len(self._get_positive_param_names())

    @property
    def k_hurdle(self):
        """Return the number of zero-hurdle coefficients."""
        return self.exog_infl.shape[1]

    def _split_params(self, params):
        """Validate and split a combined parameter vector."""
        params = np.asarray(params, dtype=float).reshape(-1)
        expected = self.k_positive + self.k_hurdle
        if len(params) != expected:
            raise ValueError(
                f'Expected {expected} parameters but received {len(params)}'
            )
        return params[:self.k_positive], params[self.k_positive:]

    def _resolve_positive_scale(self, positive_scale):
        """Resolve a supplied, fitted, or unit positive-component scale."""
        if positive_scale is None:
            positive_scale = (
                1.0 if self._positive_scale is None
                else self._positive_scale
            )
        if (not np.isscalar(positive_scale)
                or not np.isfinite(positive_scale)
                or positive_scale <= 0.0):
            raise ValueError('positive_scale must be finite and positive')
        return float(positive_scale)

    def loglike_obs(self, params, positive_scale=None, *args, **kwargs):
        """Return unweighted combined log-likelihood contributions.

        For a zero response this includes the selected zero-process log
        probability. For a positive response it includes the log probability
        of crossing the hurdle plus the positive-family log likelihood. As
        with other distributional models, estimation weights are applied only
        by :meth:`loglike`, not to this observation-level return value.
        """
        del args, kwargs
        positive_params, hurdle_params = self._split_params(params)
        positive_scale = self._resolve_positive_scale(positive_scale)

        hurdle_eta = self.exog_infl @ hurdle_params
        hurdle_loglike = self._hurdle_loglike_from_eta(hurdle_eta)

        positive_loglike = self._positive_loglike_obs(
            positive_params, positive_scale
        )
        hurdle_loglike[self.is_positive] += positive_loglike
        return hurdle_loglike

    def loglike(self, params, positive_scale=None, *args, **kwargs):
        """Return the estimation-weighted combined log likelihood."""
        return self._weighted_sum(
            self.loglike_obs(
                params, positive_scale=positive_scale, *args, **kwargs
            )
        )

    def score_obs(self, params, positive_scale=None, *args, **kwargs):
        """Return unweighted scores from both separable components."""
        del args, kwargs
        positive_params, hurdle_params = self._split_params(params)
        positive_scale = self._resolve_positive_scale(positive_scale)

        hurdle_eta = self.exog_infl @ hurdle_params
        hurdle_factor = self._hurdle_score_factor(hurdle_eta)

        positive_blocks = tuple(
            (values, factors, self._positive_row_indices)
            for values, factors in self._positive_score_blocks(
                positive_params, positive_scale
            )
        )
        return _build_score_obs(
            positive_blocks + ((self.exog_infl, hurdle_factor),),
            nobs=self.nobs,
            force_sparse=self.is_sparse_model,
        )

    def score(self, params, positive_scale=None, *args, **kwargs):
        """Return the likelihood-weighted combined score vector."""
        score_obs = self.score_obs(
            params, positive_scale=positive_scale, *args, **kwargs
        )
        return _sum_score_obs(score_obs, weights=self.weights)

    @staticmethod
    def _first_column_is_constant(exog):
        """Return whether a design matrix starts with a constant column."""
        return _is_constant_column(exog)

    @staticmethod
    def _normalize_cov_type(cov_type):
        """Map distributional covariance terminology to component fits."""
        cov_type = str(cov_type).upper()
        if cov_type in {'SANDWICH', 'HC1'}:
            return cov_type, 'NONROBUST'
        if cov_type == 'COMPONENT_HC1':
            return cov_type, 'HC1'
        if cov_type == 'BOOTSTRAP':
            return cov_type, 'NONROBUST'
        if cov_type == 'NONROBUST':
            return cov_type, cov_type
        raise ValueError(
            "cov_type must be 'SANDWICH', 'HC1', 'COMPONENT_HC1', "
            "'NONROBUST', or 'BOOTSTRAP'"
        )

    @staticmethod
    def _component_covariance(fit):
        """Return a component covariance as an array, or ``None``."""
        if not fit.did_compute_var_covar():
            return None
        return np.asarray(fit.cov_params(), dtype=float)

    @staticmethod
    def _block_diagonal(first, second):
        """Combine two dense covariance matrices without cross terms."""
        if first is None or second is None:
            return None
        result = np.zeros(
            (first.shape[0] + second.shape[0],
             first.shape[1] + second.shape[1]),
            dtype=float,
        )
        result[:first.shape[0], :first.shape[1]] = first
        result[first.shape[0]:, first.shape[1]:] = second
        return result

    def fit(
            self, start_params=None, debug: bool = False,
            cov_type='SANDWICH',
            cov_kwds=None, positive_fit_kwargs=None,
            hurdle_fit_kwargs=None,
            **glm_fit_kwargs) -> DistributionalModelResults:
        """Fit the positive and zero-hurdle components and merge them.

        Args:
            start_params: Optional combined starting vector in positive-then-
                hurdle order.
            debug: Whether to show component fitting diagnostics.
            cov_type: ``'NONROBUST'`` for block-diagonal model information,
                ``'SANDWICH'``/``'HC1'`` for a full observation-score robust
                covariance, ``'COMPONENT_HC1'`` for the historical block-
                diagonal component estimator, or ``'BOOTSTRAP'``.
            cov_kwds: Hurdle-level covariance and bootstrap options.
            positive_fit_kwargs: Overrides passed only to the positive
                component estimator.
            hurdle_fit_kwargs: Overrides passed only to the binary-equivalent
                hurdle GLM.
            **glm_fit_kwargs: Additional options shared by both component
                fits when supported. Both zero-model choices use a GLM.

        Returns:
            :class:`DistributionalModelResults` containing the two component
            fits and their combined parameters and covariance.

        With ``debug=True``, component designs and starts are printed, verbose
        diagnostics are passed to both point-estimate components, and coherent
        bootstrap refits are shown with a progress bar.
        """
        display_cov_type, component_cov_type = self._normalize_cov_type(
            cov_type
        )
        cov_kwds = {} if cov_kwds is None else dict(cov_kwds)
        positive_fit_kwargs = (
            {} if positive_fit_kwargs is None
            else dict(positive_fit_kwargs)
        )
        hurdle_fit_kwargs = (
            {} if hurdle_fit_kwargs is None else dict(hurdle_fit_kwargs)
        )

        reserved = {
            'endog', 'exog', 'family', 'link', 'var_weights',
            'exog_names', 'endog_name', 'var_weights_name',
            'fit_intercept', 'first_column_constant', 'cov_type',
            'cov_kwds', 'debug',
        }
        for label, fit_kwargs in (
                ('common', glm_fit_kwargs),
                ('positive', positive_fit_kwargs),
                ('hurdle', hurdle_fit_kwargs)):
            duplicate = reserved.intersection(fit_kwargs)
            if duplicate:
                names = ', '.join(sorted(duplicate))
                raise TypeError(
                    f'{label} component fit keywords cannot override: {names}'
                )

        positive_start = positive_fit_kwargs.pop('start_params', None)
        hurdle_start = hurdle_fit_kwargs.pop('start_params', None)
        used_default_start = (
            start_params is None
            and positive_start is None
            and hurdle_start is None
        )
        used_component_starts = (
            start_params is None and not used_default_start
        )
        if (start_params is None and positive_start is None
                and hurdle_start is None):
            start_params = self.get_start_params()
        if start_params is not None:
            if positive_start is not None or hurdle_start is not None:
                raise TypeError(
                    'Use either combined start_params or component-specific '
                    'start_params, not both'
                )
            positive_start, hurdle_start = self._split_params(start_params)

        if debug:
            if used_default_start:
                start_description = 'automatic combined'
            elif used_component_starts:
                start_description = 'component-specific'
            else:
                start_description = 'user combined'
            print('\n' + '=' * 50)
            print('HURDLE MODEL FIT')
            print('-' * 50)
            print(f'* Model: {self.__class__.__name__}')
            print(f'* Observations: {self.nobs}')
            print(
                f'* Outcome split: {self.nobs_zero} zero / '
                f'{self.nobs_positive} positive'
            )
            print(f'* Positive exog shape: {self.exog.shape}')
            print(f'* Zero-hurdle exog shape: {self.exog_infl.shape}')
            print(
                '* Positive component: '
                f'{self._positive_component_description()}'
            )
            print(
                '* Zero component: '
                f'{self._hurdle_component_description()}'
            )
            print(f'* Covariance type: {display_cov_type}')
            print(f'* Component covariance type: {component_cov_type}')
            print(
                '* Importance weights: '
                f'{"provided" if self.is_weighted else "none"}'
            )
            print(f'* Starting-value source: {start_description}')
            positive_start_string = (
                None if positive_start is None
                else np.array2string(
                    np.asarray(positive_start), precision=6
                )
            )
            hurdle_start_string = (
                None if hurdle_start is None
                else np.array2string(
                    np.asarray(hurdle_start), precision=6
                )
            )
            print(
                '* Positive starts: '
                f'{positive_start_string}'
            )
            print(
                '* Hurdle starts: '
                f'{hurdle_start_string}'
            )
            print('* Passing debug=True to both component estimators.')

        common_kwargs = dict(glm_fit_kwargs)
        common_kwargs.update({
            'debug': debug,
            'cov_type': component_cov_type,
            # Hurdle-level covariance options are consumed below. Component
            # Components receive only options valid for their native estimator.
            'cov_kwds': {},
        })
        positive_kwargs = common_kwargs.copy()
        positive_kwargs.update(positive_fit_kwargs)
        positive_kwargs['cov_kwds'] = {}
        positive_first_column_constant = self._first_column_is_constant(
            self._positive_exog
        )
        positive_kwargs['fit_intercept'] = positive_first_column_constant
        hurdle_kwargs = common_kwargs.copy()
        hurdle_kwargs.update(hurdle_fit_kwargs)
        hurdle_kwargs['cov_kwds'] = {}
        hurdle_first_column_constant = self._first_column_is_constant(
            self.exog_infl
        )
        hurdle_kwargs['fit_intercept'] = hurdle_first_column_constant

        def fit_components(
                component_weights, positive_x0, hurdle_x0,
                component_debug=False):
            positive_call_kwargs = positive_kwargs.copy()
            positive_call_kwargs['debug'] = component_debug
            hurdle_call_kwargs = hurdle_kwargs.copy()
            hurdle_call_kwargs['debug'] = component_debug
            if component_weights is None:
                positive_weights = None
            elif component_weights is self.weights:
                positive_weights = self._positive_weights
            else:
                positive_weights = component_weights[
                    self._positive_row_indices
                ]
            positive_result = self._fit_positive_component(
                positive_weights,
                positive_x0,
                positive_call_kwargs,
                positive_first_column_constant,
            )
            hurdle_result = SparseGeneralizedLinearModel.GLM(
                self._hurdle_component_endog,
                self.exog_infl,
                family=Bernoulli(),
                link=self.hurdle_link,
                var_weights=component_weights,
                exog_names=self._get_inflation_param_names(),
                endog_name=self._hurdle_component_endog_name(),
                var_weights_name=self.weights_name,
                start_params=hurdle_x0,
                first_column_constant=hurdle_first_column_constant,
                **hurdle_call_kwargs,
            )
            return positive_result, hurdle_result

        positive_fit, hurdle_fit = fit_components(
            self.weights, positive_start, hurdle_start,
            component_debug=debug,
        )

        if debug:
            positive_converged = self._positive_fit_converged(positive_fit)
            print('\nComponent-fit diagnostics:')
            print(
                '* Positive component converged: '
                f'{positive_converged}'
            )
            print(f'* Zero-hurdle GLM converged: {hurdle_fit.converged}')
            print(
                '* Positive scale: '
                f'{self._positive_fit_scale(positive_fit):.6g}'
            )

        self._positive_scale = self._positive_fit_scale(positive_fit)
        self.positive_model_fit = positive_fit
        self.hurdle_model_fit = hurdle_fit
        component_covariance = self._block_diagonal(
            self._positive_fit_covariance(positive_fit),
            self._component_covariance(hurdle_fit),
        )
        params = np.concatenate((
            self._positive_fit_params(positive_fit),
            np.asarray(hurdle_fit.params, dtype=float),
        ))
        positive_scale = self._positive_fit_scale(positive_fit)
        score_at_params = self.score(
            params, positive_scale=positive_scale
        )
        weight_total = (
            float(self.nobs)
            if self.weights is None else float(np.sum(self.weights))
        )
        normalized_score = (
            float(np.max(np.abs(score_at_params))) / max(weight_total, 1.0)
        )
        optimizer_converged = bool(
            self._positive_fit_converged(positive_fit)
            and hurdle_fit.converged
        )
        first_order_valid = bool(
            np.all(np.isfinite(score_at_params))
            and normalized_score <= 1e-5
        )
        converged = optimizer_converged and first_order_valid
        inference_issues = []
        if not optimizer_converged:
            inference_issues.append(
                'At least one component estimator did not converge.'
            )
        if not first_order_valid:
            inference_issues.append(
                'The combined hurdle score did not satisfy the scaled '
                'first-order condition.'
            )

        if debug:
            print(f'* Combined first-order validation: {first_order_valid}')
            print(f'* Combined scaled maximum score: {normalized_score:.3e}')

        bread = None
        meat = None
        information = None
        bootstrapped_params = None
        bootstrap_string = None
        bootstrap_cov_elapsed = 0.0
        covariance = component_covariance
        if debug:
            print('\nCovariance phase:')
            print(f'* Requested estimator: {display_cov_type}')
            print(f'* Component estimator: {component_cov_type}')
        if display_cov_type in {'SANDWICH', 'HC1'}:
            bread = component_covariance
            if bread is not None:
                score_obs = self.score_obs(
                    params, positive_scale=positive_scale
                )
                score_obs = _weight_score_obs(score_obs, self.weights)
                meat = _score_meat(score_obs)
                covariance = bread @ meat @ bread.T
                if display_cov_type == 'HC1':
                    covariance *= self.nobs / max(
                        self.nobs - len(params), 1
                    )
                covariance = (covariance + covariance.T) / 2.0
        elif display_cov_type == 'NONROBUST':
            bread = component_covariance
        elif display_cov_type == 'BOOTSTRAP' and converged:
            bootstrap_start = time.perf_counter()
            n_samples = cov_kwds.get(
                'n_samples', DEFAULT_BOOTSTRAP_N_SAMPLES
            )
            if (isinstance(n_samples, bool)
                    or not isinstance(n_samples, (int, np.integer))
                    or n_samples < 2):
                raise ValueError(
                    'n_samples must be an integer of at least 2'
                )
            seed = cov_kwds.get('seed', DEFAULT_BB_SEED)
            alpha = cov_kwds.get('alpha', DEFAULT_BB_ALPHA)
            if (not np.isscalar(alpha) or not np.isfinite(alpha)
                    or alpha <= 0.0):
                raise ValueError('alpha must be a finite positive scalar')
            method = str(cov_kwds.get('method', 'BAYESIAN')).upper()
            if method != 'BAYESIAN':
                raise ValueError(
                    "Hurdle bootstrap covariance supports only "
                    "method='BAYESIAN'"
                )
            min_success_rate = cov_kwds.get('min_success_rate', 0.8)
            if (not np.isscalar(min_success_rate)
                    or not np.isfinite(min_success_rate)
                    or not 0.0 <= min_success_rate <= 1.0):
                raise ValueError(
                    'min_success_rate must be between 0 and 1'
                )

            bootstrap_weights, _ = get_bayesian_bootstrap_weights(
                self.nobs,
                n_samples=n_samples,
                seed=seed,
                alpha=alpha,
            )
            if debug:
                from tqdm import tqdm

                print('\n' + '=' * 50)
                print('COHERENT HURDLE BAYESIAN BOOTSTRAP')
                print('-' * 50)
                print(f'* Requested repetitions: {n_samples}')
                print(f'* Seed: {seed}')
                print(f'* Dirichlet alpha: {alpha:.6g}')
                print(f'* Minimum success rate: {min_success_rate:.3f}')
                print(
                    '* Original importance weights: '
                    f'{"multiplied into each draw" if self.is_weighted else "none"}'
                )
                print(
                    '* The same observation-weight draw is used for both '
                    'component estimators.'
                )
                print(
                    '* Per-draw component output is suppressed; progress is '
                    'shown.'
                )
                bootstrap_weights = tqdm(
                    bootstrap_weights,
                    total=n_samples,
                    desc=f'{self.__class__.__name__} bootstrap',
                )
            draws = []
            failure_reasons = Counter()
            point_positive = params[:self.k_positive]
            point_hurdle = params[self.k_positive:]
            for draw_weights in bootstrap_weights:
                combined_weights = get_bootstrap_weights2(
                    draw_weights, self.weights
                )
                try:
                    draw_positive, draw_hurdle = fit_components(
                        combined_weights, point_positive, point_hurdle,
                        component_debug=False,
                    )
                except Exception as error:
                    failure_reasons[
                        f'{type(error).__name__}: {error}'
                    ] += 1
                    continue
                draw = np.concatenate((
                    self._positive_fit_params(draw_positive),
                    np.asarray(draw_hurdle.params, dtype=float),
                ))
                draw_valid = bool(
                    self._positive_fit_converged(draw_positive)
                    and draw_hurdle.converged
                    and np.all(np.isfinite(draw))
                )
                if draw_valid:
                    draw_score_obs = self.score_obs(
                        draw,
                        positive_scale=self._positive_fit_scale(
                            draw_positive
                        ),
                    )
                    draw_score = _sum_score_obs(
                        draw_score_obs, weights=combined_weights
                    )
                    draw_normalized_score = (
                        float(np.max(np.abs(draw_score)))
                        / max(float(np.sum(combined_weights)), 1.0)
                    )
                    draw_valid = bool(
                        np.all(np.isfinite(draw_score))
                        and draw_normalized_score <= 1e-5
                    )
                if draw_valid:
                    draws.append(draw)
                elif (self._positive_fit_converged(draw_positive)
                      and draw_hurdle.converged
                      and np.all(np.isfinite(draw))):
                    failure_reasons[
                        'Combined bootstrap first-order validation failed '
                        f'(scaled score={draw_normalized_score:.3e})'
                    ] += 1
                else:
                    failure_reasons['Component non-convergence'] += 1

            bootstrapped_params = np.asarray(draws, dtype=float)
            minimum_successes = max(
                2, int(np.ceil(float(min_success_rate) * n_samples))
            )
            if len(bootstrapped_params) < minimum_successes:
                failures = '; '.join(
                    f'{count} x {reason}'
                    for reason, count in failure_reasons.most_common(3)
                )
                raise RuntimeError(
                    f'Only {len(bootstrapped_params)} of {n_samples} '
                    'coherent hurdle bootstrap repetitions converged; '
                    f'at least {minimum_successes} were required. {failures}'
                )

            use_correction = bool(
                cov_kwds.get('use_correction', True)
            )
            covariance = np.atleast_2d(np.cov(
                bootstrapped_params, rowvar=False, ddof=0
            ))
            if use_correction:
                covariance *= (
                    len(bootstrapped_params)
                    / (len(bootstrapped_params) - 1.0)
                )
            covariance = (covariance + covariance.T) / 2.0
            n_failed = n_samples - len(bootstrapped_params)
            if n_failed:
                warnings.warn(
                    'Hurdle bootstrap retained '
                    f'{len(bootstrapped_params)} of {n_samples} coherent '
                    'repetitions.',
                    RuntimeWarning,
                    stacklevel=2,
                )
            cov_kwds = {
                'method': method,
                'n_samples': int(n_samples),
                'n_successful': len(bootstrapped_params),
                'n_failed': n_failed,
                'seed': seed,
                'alpha': alpha,
                'use_correction': use_correction,
                'min_success_rate': float(min_success_rate),
                'failure_reasons': dict(failure_reasons),
                'joint_draws': True,
            }
            bootstrap_string = (
                f'Did {len(bootstrapped_params)} coherent Bayesian hurdle '
                f'bootstrap repetitions, alpha={alpha:.3f}.'
            )
            bootstrap_cov_elapsed = time.perf_counter() - bootstrap_start
            if debug:
                print('\nBootstrap diagnostics:')
                print(
                    '* Successful repetitions: '
                    f'{len(bootstrapped_params)}'
                )
                print(f'* Failed repetitions: {n_failed}')
                print(f'* Covariance correction: {use_correction}')
                print('...Coherent hurdle bootstrap complete!\n')

        information_rank = None
        information_condition = None
        information_min_eigenvalue = None
        if bread is not None and np.all(np.isfinite(bread)):
            bread = (bread + bread.T) / 2.0
            bread_rank = int(np.linalg.matrix_rank(bread))
            bread_condition = float(np.linalg.cond(bread))
            bread_eigenvalues = np.linalg.eigvalsh(bread)
            bread_eigenvalue_tolerance = (
                np.finfo(float).eps * max(len(params), 1)
                * max(float(np.max(np.abs(bread_eigenvalues))), 1.0)
            )
            if (bread_rank == len(params)
                    and np.isfinite(bread_condition)
                    and bread_condition <= 1e12
                    and np.min(bread_eigenvalues)
                    > bread_eigenvalue_tolerance):
                information = np.linalg.inv(bread)
                information = (information + information.T) / 2.0
                information_rank = int(np.linalg.matrix_rank(information))
                information_condition = float(np.linalg.cond(information))
                information_min_eigenvalue = float(np.min(
                    np.linalg.eigvalsh(information)
                ))
            else:
                inference_issues.append(
                    'The block-diagonal hurdle bread is rank deficient or '
                    'ill-conditioned, or not positive definite.'
                )
                covariance = None

        inference_valid = bool(
            converged
            and covariance is not None
            and np.all(np.isfinite(covariance))
        )
        if inference_valid:
            covariance_eigenvalues = np.linalg.eigvalsh(covariance)
            covariance_tolerance = (
                np.finfo(float).eps * max(len(params), 1)
                * max(float(np.max(np.abs(covariance_eigenvalues))), 1.0)
            )
            inference_valid = bool(
                np.min(covariance_eigenvalues) >= -covariance_tolerance
            )
        if not inference_valid:
            inference_issues.append(
                'The combined hurdle covariance is unavailable, non-finite, '
                'or not positive semidefinite.'
            )
            covariance = None

        message = (
            'Both hurdle components converged.'
            if converged
            else 'Hurdle fit diagnostics failed.'
        )
        if debug:
            print('\nFinal hurdle-fit diagnostics:')
            print(f'* Public convergence: {converged}')
            print(f'* Inference valid: {inference_valid}')
            print(
                '* Covariance available: '
                f'{covariance is not None}'
            )
            if information_rank is not None:
                print(
                    f'* Information rank: {information_rank}/'
                    f'{len(params)}'
                )
            if information_condition is not None:
                print(
                    '* Information condition number: '
                    f'{information_condition:.3e}'
                )
            print('...Hurdle model fit complete!\n')
        if inference_issues:
            warnings.warn(
                ' '.join(inference_issues), RuntimeWarning, stacklevel=2
            )
        return DistributionalModelResults(
            model=self,
            params=params,
            llf=self.loglike(params, positive_scale=positive_scale),
            converged=converged,
            optimizer_converged=optimizer_converged,
            first_order_valid=first_order_valid,
            inference_valid=inference_valid,
            inference_issues=inference_issues,
            normalized_score=normalized_score,
            covariance_status=(
                'valid full observation-score covariance'
                if inference_valid and display_cov_type in {'SANDWICH', 'HC1'}
                else (
                    'valid coherent bootstrap covariance'
                    if inference_valid and display_cov_type == 'BOOTSTRAP'
                    else 'valid block-diagonal component covariance'
                    if inference_valid else 'unavailable after diagnostics'
                )
            ),
            message=message,
            method='SEPARATE COMPONENT FITS',
            positive_fit=positive_fit,
            hurdle_fit=hurdle_fit,
            cov_params=covariance,
            information=information,
            information_rank=information_rank,
            information_condition=information_condition,
            information_min_eigenvalue=information_min_eigenvalue,
            bread=bread,
            meat=meat,
            bootstrapped_params=bootstrapped_params,
            bootstrap_string=bootstrap_string,
            cov_type=display_cov_type,
            component_cov_type=component_cov_type,
            cov_kwds=cov_kwds,
            fittedvalues=self.predict(params, which='mean'),
            fit_elapsed=(
                float(positive_fit.fit_elapsed)
                + float(hurdle_fit.fit_elapsed)
            ),
            cov_elapsed=(
                float(positive_fit.cov_elapsed)
                + float(hurdle_fit.cov_elapsed)
                + bootstrap_cov_elapsed
            ),
            iterations=(
                getattr(
                    positive_fit, 'num_iter',
                    getattr(positive_fit, 'iterations', None),
                ),
                getattr(
                    hurdle_fit, 'num_iter',
                    getattr(hurdle_fit, 'iterations', None),
                ),
            ),
            score_at_params=score_at_params,
            scale=positive_scale,
            loglike_kwargs={'positive_scale': positive_scale},
            is_quasi_likelihood=self.is_quasi_likelihood,
            report_information_criteria=(
                not self.is_quasi_likelihood and not self.is_weighted
            ),
        )

    def predict(
            self, params, exog=None, exog_infl=None, which='mean',
            data=None, index=None, debug=False):
        """Predict hurdle probabilities and component or overall means.

        Args:
            params: Combined parameter vector.
            exog: Optional positive-response design matrix.
            exog_infl: Optional zero-hurdle design matrix.
            data: Optional new data evaluated through both stored formulas.
            index: Optional positional row selector applied to ``data``.
            debug: Whether formula construction prints diagnostics.
            which: One of ``'mean'``, ``'positive_mean'``,
                ``'underlying_mean'``, ``'linear_predictor'``,
                ``'zero_probability'``, or ``'positive_probability'``.
                ``positive_mean`` is conditional on ``Y>0``. For truncated
                count components, ``underlying_mean`` is the mean before
                conditioning away zeros.
        """
        if data is not None:
            if exog is not None or exog_infl is not None:
                raise ValueError(
                    'Supply either data or numeric component designs, not both'
                )
            exog, exog_infl = self._get_formula_prediction_designs(
                data, index=index, debug=debug
            )
        positive_params, hurdle_params = self._split_params(params)
        exog = self.exog if exog is None else _as_design_matrix(exog)
        exog_infl = (
            self.exog_infl
            if exog_infl is None
            else _as_design_matrix(exog_infl, name='exog_infl')
        )
        if exog.shape[0] != exog_infl.shape[0]:
            raise ValueError(
                'exog and exog_infl must contain the same number of rows'
            )
        if exog.shape[1] != self.exog.shape[1]:
            raise ValueError('exog has the wrong number of columns')
        if exog_infl.shape[1] != self.k_hurdle:
            raise ValueError('exog_infl has the wrong number of columns')

        positive_mean = self._positive_conditional_mean(
            positive_params, exog
        )
        underlying_mean = self._positive_underlying_mean(
            positive_params, exog
        )
        linear_predictor = exog @ positive_params[:self.exog.shape[1]]
        zero_probability, positive_probability = (
            self._hurdle_probabilities_from_eta(
                exog_infl @ hurdle_params
            )
        )
        predictions = {
            'mean': positive_probability * positive_mean,
            'positive_mean': positive_mean,
            'underlying_mean': underlying_mean,
            'linear_predictor': linear_predictor,
            'zero_probability': zero_probability,
            'positive_probability': positive_probability,
        }
        which = str(which).lower()
        if which not in predictions:
            choices = ', '.join(sorted(predictions))
            raise ValueError(f'which must be one of: {choices}')
        return predictions[which]


class _OLSPositiveHurdle(HurdleModel):
    """Shared closed-form OLS machinery for continuous positive components."""

    _positive_distribution_name = 'Gaussian'
    _ignored_glm_fit_options = {
        'alpha', 'L2_penalty_matrix', 'l1_ratio', 'line_search_fallback',
        'max_iter', 'normalize', 'opt_method', 'penalize_scale',
        'pick_default_start', 'prompt_user_for_more_iters',
        'regularize_to_values', 'store_convergence_path', 'tol',
    }
    _linear_fit_options = {
        'compute_eigenvalues', 'dense_threshold_mb', 'inverse_method',
        'scale_design_matrix', 'test_level', 'use_t',
    }

    def __init__(self, *args, **kwargs):
        """Initialize lazy transformed-response caches before model setup."""
        self.__positive_working_endog = None
        self.__positive_linear_component_endog = None
        super().__init__(*args, **kwargs)

    @property
    def _positive_working_endog(self):
        """Return one cached transformed positive response."""
        if self.__positive_working_endog is None:
            self.__positive_working_endog = self._transform_positive_endog(
                self._positive_endog
            )
        return self.__positive_working_endog

    @property
    def _positive_linear_component_endog(self):
        """Return the transformed response in the OLS design's format."""
        if not isspmatrix(self._positive_exog):
            return self._positive_working_endog
        if self.__positive_linear_component_endog is None:
            self.__positive_linear_component_endog = csc_matrix(
                self._positive_working_endog.reshape(-1, 1)
            )
        return self.__positive_linear_component_endog

    @property
    def positive_family(self):
        """Return ``None`` because the positive component is fitted by OLS."""
        return None

    @property
    def positive_link(self):
        """Return ``None`` because OLS uses an identity linear predictor."""
        return None

    def _get_positive_param_names(self):
        """Return OLS coefficients followed by the log residual variance."""
        return self._get_regression_param_names() + ['log_scale']

    def _positive_component_description(self):
        """Return a concise description of the transformed OLS component."""
        return f'{self._positive_distribution_name} via weighted OLS'

    def _transform_positive_endog(self, endog):
        """Return the response used by OLS; subclasses may transform it."""
        return np.asarray(endog, dtype=float)

    def _positive_log_jacobian(self):
        """Return the response-transformation log Jacobian."""
        return np.zeros(self.nobs_positive, dtype=float)

    def _split_positive_params(self, params):
        """Split positive regression coefficients and log variance."""
        params = np.asarray(params, dtype=float).reshape(-1)
        expected = self.exog.shape[1] + 1
        if len(params) != expected:
            raise ValueError(
                f'Expected {expected} positive parameters but received '
                f'{len(params)}'
            )
        beta = params[:-1]
        log_scale = float(params[-1])
        with np.errstate(over='ignore', invalid='ignore'):
            scale = float(np.exp(log_scale))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError('log_scale must imply a finite positive scale')
        return beta, log_scale, scale

    def _get_positive_start_params(self):
        """Return weighted constant-mean and residual-variance starts."""
        transformed = self._positive_working_endog
        weights = self._positive_weights
        if weights is None:
            weight_total = float(self.nobs_positive)
            mean = float(np.mean(transformed))
        else:
            weight_total = float(np.sum(weights))
            mean = float(np.dot(weights, transformed) / weight_total)
        beta = self._constant_predictor_start(
            self._positive_exog, mean
        )
        residual = transformed - self._positive_exog @ beta
        scale = (
            float(np.mean(residual ** 2))
            if weights is None
            else float(np.dot(weights, residual ** 2) / weight_total)
        )
        scale = max(scale, np.finfo(float).eps)
        return np.concatenate((beta, [np.log(scale)]))

    def _positive_loglike_obs(self, params, positive_scale):
        """Return Gaussian working-density contributions on the fit scale."""
        beta, log_scale, scale = self._split_positive_params(params)
        del positive_scale
        transformed = self._positive_working_endog
        residual = transformed - self._positive_exog @ beta
        return (
            -0.5 * (
                np.log(2.0 * np.pi) + log_scale + residual ** 2 / scale
            )
            + self._positive_log_jacobian()
        )

    def _positive_score_blocks(self, params, positive_scale):
        """Return factorized beta and log-variance score columns."""
        beta, _, scale = self._split_positive_params(params)
        del positive_scale
        transformed = self._positive_working_endog
        residual = transformed - self._positive_exog @ beta
        scale_score = 0.5 * (residual ** 2 / scale - 1.0)
        return (
            (self._positive_exog, residual / scale),
            (scale_score, None),
        )

    def _positive_conditional_mean(self, params, exog):
        """Return the Gaussian positive-component working mean."""
        beta, _, _ = self._split_positive_params(params)
        return exog @ beta

    def _positive_underlying_mean(self, params, exog):
        """Return the same mean because this component is not truncated."""
        return self._positive_conditional_mean(params, exog)

    def _fit_positive_component(
            self, weights, start_params, fit_kwargs,
            first_column_constant):
        """Fit coefficients by kanly OLS and estimate Gaussian scale."""
        del start_params
        fit_kwargs = dict(fit_kwargs)
        cov_type = str(fit_kwargs.pop('cov_type', 'NONROBUST')).upper()
        cov_kwds = fit_kwargs.pop('cov_kwds', {})
        debug = bool(fit_kwargs.pop('debug', False))
        compute_cov = bool(fit_kwargs.pop('compute_cov', True))
        fit_kwargs.pop('fit_intercept', None)

        linear_options = {
            name: fit_kwargs.pop(name)
            for name in tuple(fit_kwargs)
            if name in self._linear_fit_options
        }
        for name in self._ignored_glm_fit_options:
            fit_kwargs.pop(name, None)
        if fit_kwargs:
            unsupported = ', '.join(sorted(fit_kwargs))
            raise TypeError(
                'The OLS positive component does not accept these fit '
                f'options: {unsupported}'
            )
        if cov_type not in {'NONROBUST', 'HC1'}:
            raise ValueError(
                "The OLS positive component covariance must be 'NONROBUST' "
                "or 'HC1'"
            )

        positive_endog = self._positive_working_endog
        positive_exog = self._positive_exog
        positive_weights = (
            None if weights is None else np.asarray(weights, dtype=float)
        )
        positive_name = self.endog_name
        if (positive_name is not None
                and self._positive_distribution_name == 'Lognormal'):
            positive_name = f'log({positive_name})'

        # Covariance is assembled below using likelihood-importance weights,
        # so avoid duplicating kanly's native WLS covariance calculation.
        fit = SparseLinearModel.LM(
            self._positive_linear_component_endog,
            positive_exog,
            weights=positive_weights,
            has_constant=(
                first_column_constant or self.has_implicit_constant
            ),
            exog_names=self._get_regression_param_names(),
            endog_name=positive_name,
            weights_name=self.weights_name,
            debug=debug,
            cov_type='NONROBUST',
            cov_kwds={},
            compute_cov=False,
            keep_model=True,
            **linear_options,
        )
        beta = np.asarray(fit.params, dtype=float).reshape(-1)
        residual = positive_endog - positive_exog @ beta
        weight_total = (
            float(self.nobs_positive)
            if positive_weights is None
            else float(np.sum(positive_weights))
        )
        scale = (
            float(np.mean(residual ** 2))
            if positive_weights is None
            else float(
                np.dot(positive_weights, residual ** 2) / weight_total
            )
        )
        if not np.isfinite(scale) or scale <= np.finfo(float).tiny:
            raise ValueError(
                'The positive OLS residual variance must be finite and '
                'strictly positive'
            )

        normalized_covariance = fit.normalized_cov_params
        if hasattr(normalized_covariance, 'toarray'):
            normalized_covariance = normalized_covariance.toarray()
        normalized_covariance = np.asarray(
            normalized_covariance, dtype=float
        )
        k_beta = positive_exog.shape[1]
        bread = np.zeros((k_beta + 1, k_beta + 1), dtype=float)
        bread[:k_beta, :k_beta] = scale * normalized_covariance
        bread[-1, -1] = 2.0 / weight_total

        full_covariance = None
        if compute_cov:
            if cov_type == 'NONROBUST':
                full_covariance = bread
            else:
                scale_score = 0.5 * (residual ** 2 / scale - 1.0)
                score_obs = _build_score_obs((
                    (positive_exog, residual / scale),
                    (scale_score, None),
                ))
                weighted_score = _weight_score_obs(
                    score_obs, positive_weights
                )
                meat = _score_meat(weighted_score)
                full_covariance = bread @ meat @ bread.T
                full_covariance *= self.nobs_positive / max(
                    self.nobs_positive - (k_beta + 1), 1
                )
                full_covariance = (
                    full_covariance + full_covariance.T
                ) / 2.0
            fit.set_cov_params(
                full_covariance[:k_beta, :k_beta],
                cov_type=cov_type,
                cov_kwds=cov_kwds,
                df_t_dist=fit.df_resid,
            )

        fit.scale = scale
        fit.scale_mle = scale
        fit.converged = True
        fit._distributional_params = np.concatenate((
            beta, [np.log(scale)]
        ))
        fit._distributional_covariance = full_covariance
        return fit

    @staticmethod
    def _positive_fit_params(fit):
        """Return OLS coefficients and the closed-form log variance."""
        return np.asarray(fit._distributional_params, dtype=float)

    @staticmethod
    def _positive_fit_covariance(fit):
        """Return joint coefficient/log-variance covariance."""
        covariance = fit._distributional_covariance
        return None if covariance is None else np.asarray(covariance)


class GaussianHurdle(_OLSPositiveHurdle):
    """Configurable zero hurdle with a Gaussian OLS positive component.

    The positive observations are regressed directly on ``exog``. Since an
    untruncated Gaussian distribution is not confined to ``(0, infinity)``,
    this is reported as a quasi-likelihood two-part model rather than a fully
    normalized positive-response likelihood.
    """

    is_quasi_likelihood = True
    _positive_distribution_name = 'Gaussian'

    def _quasi_likelihood_footer(self):
        """Explain why positive-response Gaussian OLS is a working model."""
        return (
            'The positive component uses an untruncated Gaussian OLS working '
            'model on observations restricted to Y>0. It is therefore '
            'reported as quasi-likelihood; likelihood-based AIC and BIC are '
            'not reported.'
        )


class LognormalHurdle(_OLSPositiveHurdle):
    """Configurable zero hurdle with a lognormal positive response.

    OLS is fitted to ``log(Y)`` for positive observations. If its fitted
    residual variance is ``scale``, then the positive-response median is
    ``exp(X beta)`` and its conditional mean is
    ``exp(X beta + scale / 2)``.
    """

    _positive_distribution_name = 'Lognormal'

    def _transform_positive_endog(self, endog):
        """Return ``log(Y)`` for the positive observations."""
        return np.log(np.asarray(endog, dtype=float))

    def _positive_log_jacobian(self):
        """Return the lognormal transformation Jacobian ``-log(Y)``."""
        return -self._positive_working_endog

    def _positive_conditional_mean(self, params, exog):
        """Return the lognormal arithmetic mean on the outcome scale."""
        beta, _, scale = self._split_positive_params(params)
        with np.errstate(over='ignore', invalid='ignore'):
            return np.exp(exog @ beta + 0.5 * scale)


class PoissonHurdle(HurdleModel):
    """Configurable zero hurdle with positive Poisson counts.

    The positive component uses the existing GLM
    :class:`ZeroTruncatedPoisson` family, whose linear predictor models the
    log of the underlying untruncated Poisson rate.
    """

    def __init__(self, endog, exog, *args, **kwargs):
        """Validate non-negative integer outcomes and initialize the model."""
        values = np.asarray(endog, dtype=float)
        if (np.any(~np.isfinite(values)) or np.any(values < 0.0)
                or np.any(values != np.floor(values))):
            raise ValueError(
                'PoissonHurdle outcomes must be finite non-negative integers'
            )
        self._positive_family_instance = ZeroTruncatedPoisson()
        self._positive_link_instance = (
            self._positive_family_instance.default_link()
        )
        super().__init__(endog, exog, *args, **kwargs)

    @property
    def positive_family(self):
        """Return the zero-truncated Poisson GLM family."""
        return self._positive_family_instance

    @property
    def positive_link(self):
        """Return the zero-truncated Poisson canonical link."""
        return self._positive_link_instance

    def _positive_underlying_mean(self, params, exog):
        """Return the underlying, untruncated Poisson rate ``exp(X beta)``."""
        with np.errstate(over='ignore', invalid='ignore'):
            return np.exp(exog @ params)


class GammaHurdle(HurdleModel):
    """Configurable zero hurdle with positive Gamma responses.

    Gamma already has support over ``(0, infinity)``, so no additional
    truncation normalization is needed.  The positive conditional mean uses a
    log link by default.
    """

    is_quasi_likelihood = True

    def __init__(self, endog, exog, *args, positive_link=None, **kwargs):
        """Initialize the Gamma family and its configurable positive link."""
        (
            self._positive_family_instance,
            self._positive_link_instance,
        ) = _get_family_and_link(
            GammaFamily(), Log() if positive_link is None else positive_link
        )
        super().__init__(endog, exog, *args, **kwargs)

    @property
    def positive_family(self):
        """Return the Gamma GLM family."""
        return self._positive_family_instance

    @property
    def positive_link(self):
        """Return the configured Gamma positive-response link."""
        return self._positive_link_instance


class InverseGaussianHurdle(HurdleModel):
    """Configurable zero hurdle with an Inverse Gaussian positive response.

    Given a positive response, the conditional mean is ``exp(X beta)`` and
    the Inverse Gaussian variance is ``scale * mu**3``. The positive
    component is estimated by the existing GLM IRLS implementation, with
    scale estimated from the Pearson residuals.

    Because the scale is Pearson-estimated rather than jointly maximized, the
    combined result is reported as quasi-likelihood and omits AIC and BIC.
    """

    is_quasi_likelihood = True

    def __init__(self, endog, exog, *args, positive_link=None, **kwargs):
        """Initialize the Inverse Gaussian family with a log link by default."""
        (
            self._positive_family_instance,
            self._positive_link_instance,
        ) = _get_family_and_link(
            InverseGaussianFamily(),
            Log() if positive_link is None else positive_link,
        )
        super().__init__(endog, exog, *args, **kwargs)

    @property
    def positive_family(self):
        """Return the Inverse Gaussian GLM family."""
        return self._positive_family_instance

    @property
    def positive_link(self):
        """Return the configured positive-response link."""
        return self._positive_link_instance


class NegativeBinomialPHurdle(HurdleModel):
    """Configurable zero hurdle with an exact truncated NB-P component.

    Conditional on a positive response, the count follows an NB-P
    distribution truncated at zero. Its underlying, untruncated mean and
    variance are

    ``mu = exp(X beta)`` and ``Var(Y | X) = mu + alpha * mu**p``.

    ``p=1`` selects NB1 and ``p=2`` selects NB2. Dispersion is estimated as
    ``alpha = exp(log_alpha)``. Parameter order is ``beta``, ``log_alpha``,
    then the hurdle ``gamma`` coefficients. The positive conditional mean is
    ``mu / (1 - P_NBP(Y=0))``; it is not ``mu``.
    """

    def __init__(self, endog, exog, *args, p=2, **kwargs):
        """Validate count support and initialize an NB1 or NB2 hurdle."""
        if isinstance(p, bool) or p not in (1, 2):
            raise ValueError('p must be either 1 (NB1) or 2 (NB2)')
        self.p = int(p)
        self.negative_binomial_p = self.p
        values = np.asarray(endog, dtype=float)
        if (np.any(~np.isfinite(values)) or np.any(values < 0.0)
                or np.any(values != np.floor(values))):
            raise ValueError(
                'NegativeBinomialPHurdle outcomes must be finite '
                'non-negative integers'
            )
        super().__init__(endog, exog, *args, **kwargs)
        self._positive_component_model = self._make_positive_model(
            self._positive_weights
        )

    @property
    def positive_family(self):
        """Return ``None`` because the positive model is exact MLE, not GLM."""
        return None

    @property
    def positive_link(self):
        """Return ``None`` because ``X beta`` directly models ``log(mu)``."""
        return None

    def _get_positive_param_names(self):
        """Return positive regression names followed by ``log_alpha``."""
        return self._get_regression_param_names() + ['log_alpha']

    def _positive_component_description(self):
        """Describe the exact truncated NB-P likelihood used for positives."""
        return f'ZERO-TRUNCATED NB{self.p} / LOG-MEAN MLE'

    def _make_positive_model(self, positive_weights):
        """Construct an aligned exact-likelihood positive component model."""
        return _ZeroTruncatedNegativeBinomialP(
            self._positive_endog,
            self._positive_exog,
            weights=positive_weights,
            p=self.p,
            endog_name=self.endog_name,
            exog_names=self._get_regression_param_names(),
            weights_name=self.weights_name,
            has_intercept=self.has_intercept,
            has_implicit_constant=self.has_implicit_constant,
        )

    def _get_positive_start_params(self):
        """Return NB-P mean-regression and log-dispersion starts."""
        return self._positive_component_model.get_start_params()

    def _positive_loglike_obs(self, params, positive_scale):
        """Return exact zero-truncated NB-P positive log likelihoods."""
        del positive_scale
        return self._positive_component_model.loglike_obs(params)

    def _positive_score_blocks(self, params, positive_scale):
        """Return factorized exact NB-P positive score columns."""
        del positive_scale
        return self._positive_component_model._score_blocks(params)

    def _positive_underlying_mean(self, params, exog):
        """Return ``mu = exp(X beta)`` before truncating away zeros."""
        with np.errstate(over='ignore', invalid='ignore'):
            return np.exp(exog @ params[:-1])

    def _positive_conditional_mean(self, params, exog):
        """Return ``E[Y | Y>0, X]`` under the fitted NB-P component."""
        underlying_mean = self._positive_underlying_mean(params, exog)
        log_zero_probability = (
            self._positive_component_model.log_zero_probability(
                params, exog=exog
            )
        )
        with np.errstate(divide='ignore', invalid='ignore'):
            return underlying_mean / (-np.expm1(log_zero_probability))

    def _fit_positive_component(
            self, weights, start_params, fit_kwargs,
            first_column_constant):
        """Fit the exact truncated NB-P likelihood, including dispersion."""
        del first_column_constant
        fit_kwargs = dict(fit_kwargs)
        debug = bool(fit_kwargs.pop('debug', False))
        requested_cov_type = str(
            fit_kwargs.pop('cov_type', 'NONROBUST')
        ).upper()
        fit_kwargs.pop('cov_kwds', None)
        fit_kwargs.pop('fit_intercept', None)
        if fit_kwargs:
            unsupported = ', '.join(sorted(fit_kwargs))
            raise TypeError(
                'The exact zero-truncated NB-P component does not accept '
                f'these GLM fit options: {unsupported}'
            )

        component = (
            self._positive_component_model
            if weights is self._positive_weights
            else self._make_positive_model(weights)
        )
        component_cov_type = (
            'SANDWICH' if requested_cov_type == 'HC1'
            else requested_cov_type
        )
        result = component.fit(
            start_params=start_params,
            debug=debug,
            cov_type=component_cov_type,
            cov_kwds={},
        )
        if requested_cov_type == 'HC1' and result.did_compute_var_covar():
            correction = component.nobs / max(
                component.nobs - len(result.params), 1
            )
            result.set_cov_params(
                np.asarray(result.cov_params()) * correction,
                cov_type='HC1',
            )
        return result


__all__ = [
    'HurdleModel',
    'HurdleModelResults',
    'GaussianHurdle',
    'LognormalHurdle',
    'PoissonHurdle',
    'GammaHurdle',
    'InverseGaussianHurdle',
    'NegativeBinomialPHurdle',
]
