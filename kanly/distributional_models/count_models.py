"""Likelihood-based regression models for count response data.

The module provides the shared :class:`DistributionalModel` estimation
interface plus Poisson, generalized-Poisson, negative-binomial, and
zero-inflated count likelihoods. All models support optional observation-level
likelihood weights, formula-based construction, analytical observation scores
where available, classical or sandwich covariance estimation, and
Bayesian-bootstrap covariance estimation.
"""

from __future__ import absolute_import, print_function

import numpy as np

from scipy.special import digamma, expit, gammaln
from kanly.distributional_models.base import (
    DistributionalModel,
    _NonnegativeDistributionalModel,
    _as_design_matrix,
    _build_score_obs,
    _is_constant_column,
)
from kanly.distributional_models.two_part import TwoPartModel


_POISSON_LIMIT_LOG_ALPHA = float(np.log(1e-8))


def _poisson_loglike_obs(endog, eta):
    """Return stable Poisson-limit contributions for count likelihoods."""
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        mean = np.exp(eta)
        values = endog * eta - mean - gammaln(endog + 1.0)
    valid = np.isfinite(eta) & np.isfinite(mean) & np.isfinite(values)
    return np.where(valid, values, -np.inf)


def _poisson_score_factor(endog, eta):
    """Return the Poisson linear-predictor score with finite guards."""
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        mean = np.exp(eta)
        values = endog - mean
    valid = np.isfinite(eta) & np.isfinite(mean) & np.isfinite(values)
    return np.where(valid, values, np.nan)


class Poisson(_NonnegativeDistributionalModel):
    """Poisson log-link regression with conditional mean ``exp(X beta)``.

    The model has no separately estimated dispersion parameter.  It accepts
    non-negative integer or continuous responses for quasi-likelihood-style
    estimation, although the probability-mass interpretation is for counts.
    """

    def get_param_names(self):
        """Return one parameter name per regression coefficient."""
        return self._get_regression_param_names()

    def _loglike_obs(self, params):
        """Compute unweighted Poisson log-likelihood contributions."""
        eta = self.exog @ params
        return _poisson_loglike_obs(self.endog, eta)

    def loglike(self, params, *args, **kwargs):
        """Return the weighted or unweighted Poisson log-likelihood."""
        return self._weighted_sum(self._loglike_obs(params))

    def score(self, params, *args, **kwargs):
        """Return the analytical score of the aggregated likelihood."""
        residual = _poisson_score_factor(self.endog, self.exog @ params)
        self._apply_weights(residual)
        return self.exog.T @ residual

    def score_obs(self, params, *args, **kwargs):
        """Return unweighted Poisson scores with shape ``(nobs, n_params)``."""
        residual = _poisson_score_factor(self.endog, self.exog @ params)
        return _build_score_obs(((self.exog, residual),))

    def loglike_obs(self, params, *args, **kwargs):
        """Return one unweighted Poisson log-likelihood value per row."""
        return self._loglike_obs(params)


class _ZeroInflatedModel(TwoPartModel):
    """Mixture algebra shared by zero-inflated count models."""

    def _get_inflation_param_names(self):
        """Return prefixed parameter names for the inflation equation."""
        if self.exog_infl_names is not None:
            return [f'inflate_{name}' for name in self.exog_infl_names]
        if (self.k_inflate == 1
                and _is_constant_column(self.exog_infl, value=1.0)):
            return ['inflate_const']
        return [f'inflate_x{i}' for i in range(self.k_inflate)]

    def _mixture_start_components(self):
        """Return count, inflation, probability, and latent-mean starts."""
        observed_mean, _ = self._response_moments()
        is_zero = (self.endog == 0.0).astype(float)
        if self.weights is None:
            zero_fraction = float(np.mean(is_zero))
        else:
            total_weight = float(np.sum(self.weights))
            if total_weight <= 0.0:
                raise ValueError(
                    'Cannot initialize parameters without positive total weight'
                )
            zero_fraction = float(
                np.dot(self.weights, is_zero) / total_weight
            )

        baseline_mean = max(observed_mean, 1e-6)
        poisson_zero_probability = float(np.exp(-baseline_mean))
        denominator = 1.0 - poisson_zero_probability
        if denominator > 1e-8:
            inflation_probability = (
                zero_fraction - poisson_zero_probability
            ) / denominator
        else:
            inflation_probability = 0.05
        inflation_probability = float(np.clip(
            inflation_probability, 0.02, 0.80
        ))

        latent_mean = max(
            observed_mean / (1.0 - inflation_probability), 1e-6
        )
        count_start = self._mean_regression_start(latent_mean)
        inflation_logit = (
            np.log(inflation_probability)
            - np.log1p(-inflation_probability)
        )
        inflation_start = self._constant_predictor_start(
            self.exog_infl, inflation_logit
        )
        return (
            count_start,
            inflation_start,
            inflation_probability,
            latent_mean,
        )

    def get_start_params(self):
        """Return moment-based count and structural-zero starting values."""
        count_start, inflation_start, _, _ = (
            self._mixture_start_components()
        )
        return np.concatenate((count_start, inflation_start))

    def _count_zero_probability(self, params, count_mean):
        """Return the count component's probability of zero."""
        raise NotImplementedError

    def predict(
            self, params, exog=None, exog_infl=None, which='mean',
            data=None, index=None, debug=False):
        """Predict mixture means and structural or observed-zero probabilities.

        Args:
            params: Full count-then-inflation parameter vector.
            exog: Optional count-component numeric design matrix.
            exog_infl: Optional structural-zero numeric design matrix.
            data: Optional new data evaluated through both stored formulas.
            index: Optional positional row selector applied to ``data``.
            debug: Whether formula construction prints diagnostics.
            which: ``'mean'``, ``'count_mean'``,
                ``'inflation_probability'``, ``'zero_probability'``, or
                ``'positive_probability'``.
        """
        if data is not None:
            if exog is not None or exog_infl is not None:
                raise ValueError(
                    'Supply either data or numeric component designs, not both'
                )
            exog, exog_infl = self._get_formula_prediction_designs(
                data, index=index, debug=debug
            )
        params = np.asarray(params, dtype=float).reshape(-1)
        if len(params) != len(self.param_names):
            raise ValueError('params has the wrong length')
        exog = self.exog if exog is None else _as_design_matrix(exog)
        exog_infl = (
            self.exog_infl
            if exog_infl is None
            else _as_design_matrix(exog_infl, name='exog_infl')
        )
        if exog.ndim != 2 or exog.shape[1] != self.exog.shape[1]:
            raise ValueError('exog has the wrong number of columns')
        if (exog_infl.ndim != 2
                or exog_infl.shape[1] != self.exog_infl.shape[1]):
            raise ValueError('exog_infl has the wrong number of columns')
        if exog.shape[0] != exog_infl.shape[0]:
            raise ValueError(
                'exog and exog_infl must contain the same number of rows'
            )

        count_params = params[:self.exog.shape[1]]
        inflation_start = self.exog.shape[1]
        inflation_params = params[
            inflation_start:inflation_start + self.k_inflate
        ]
        with np.errstate(over='ignore', invalid='ignore'):
            count_mean = np.exp(exog @ count_params)
        inflation_probability = expit(exog_infl @ inflation_params)
        count_zero_probability = self._count_zero_probability(
            params, count_mean
        )
        zero_probability = (
            inflation_probability
            + (1.0 - inflation_probability) * count_zero_probability
        )
        predictions = {
            'mean': (1.0 - inflation_probability) * count_mean,
            'count_mean': count_mean,
            'inflation_probability': inflation_probability,
            'zero_probability': zero_probability,
            'positive_probability': 1.0 - zero_probability,
        }
        which = str(which).lower()
        if which not in predictions:
            choices = ', '.join(sorted(predictions))
            raise ValueError(f'which must be one of: {choices}')
        return predictions[which]

    def _mixture_terms(self, inflation_params, count_loglike,
                       count_loglike_zero):
        """Combine a count distribution with a structural-zero process.

        Args:
            inflation_params: Coefficients of the structural-zero logit.
            count_loglike: Count-component log likelihood at observed outcomes.
            count_loglike_zero: Count-component log probability at zero.

        Returns:
            Tuple containing mixture log-likelihood contributions, posterior
            count-component probabilities, and derivatives with respect to
            each inflation linear predictor.
        """
        inflation_eta = self.exog_infl @ inflation_params
        log_inflation_prob = -np.logaddexp(0.0, -inflation_eta)
        log_count_prob = -np.logaddexp(0.0, inflation_eta)
        zero_loglike = np.logaddexp(
            log_inflation_prob, log_count_prob + count_loglike_zero
        )
        is_zero = self.endog == 0.0
        loglike_obs = np.where(
            is_zero, zero_loglike, log_count_prob + count_loglike
        )

        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            posterior_count_zero = np.exp(
                log_count_prob + count_loglike_zero - zero_loglike
            )
            inflation_prob = np.exp(log_inflation_prob)
        posterior_count = np.where(is_zero, posterior_count_zero, 1.0)
        posterior_structural_zero = np.where(
            is_zero, 1.0 - posterior_count_zero, 0.0
        )
        d_inflation_eta = posterior_structural_zero - inflation_prob

        valid = np.isfinite(inflation_eta)
        return (
            np.where(valid, loglike_obs, -np.inf),
            np.where(valid, posterior_count, np.nan),
            np.where(valid, d_inflation_eta, np.nan),
        )


class ZeroInflatedPoisson(_ZeroInflatedModel):
    """Zero-inflated Poisson regression with separate mean and zero logits.

    The count mean is ``mu = exp(exog @ beta)`` and the structural-zero
    probability is ``expit(exog_infl @ gamma)``.  Parameter order is ``beta``
    followed by ``gamma``.  ``exog_infl`` defaults to an intercept-only model.
    """

    def get_param_names(self):
        """Return count coefficient names followed by inflation names."""
        return (
            self._get_regression_param_names()
            + self._get_inflation_param_names()
        )

    def _count_zero_probability(self, params, count_mean):
        """Return the Poisson component probability of a zero count."""
        del params
        return np.exp(-count_mean)

    def _model_terms(self, params):
        """Compute mixture likelihood and both linear-predictor scores."""
        k_count = self.exog.shape[1]
        count_params = params[:k_count]
        inflation_params = params[k_count:]
        eta = self.exog @ count_params
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            mu = np.exp(eta)
            count_loglike = (
                self.endog * eta - mu - gammaln(self.endog + 1.0)
            )
            count_loglike_zero = -mu
            d_count_eta = self.endog - mu

        valid = np.isfinite(eta) & np.isfinite(mu)
        count_loglike = np.where(valid, count_loglike, -np.inf)
        count_loglike_zero = np.where(valid, count_loglike_zero, -np.inf)
        loglike_obs, posterior_count, d_inflation_eta = (
            self._mixture_terms(
                inflation_params, count_loglike, count_loglike_zero
            )
        )
        safe_d_count_eta = np.where(valid, d_count_eta, 0.0)
        d_count_eta = np.where(
            valid, posterior_count * safe_d_count_eta, np.nan
        )
        return loglike_obs, d_count_eta, d_inflation_eta

    def _loglike_obs(self, params):
        """Compute unweighted zero-inflated Poisson contributions."""
        return self._model_terms(params)[0]

    def _score_factors(self, params):
        """Return count-mean and inflation-logit score factors."""
        _, d_count_eta, d_inflation_eta = self._model_terms(params)
        return d_count_eta, d_inflation_eta

    def loglike(self, params, *args, **kwargs):
        """Return the aggregated zero-inflated Poisson log-likelihood."""
        return self._weighted_sum(self._loglike_obs(params))

    def score(self, params, *args, **kwargs):
        """Return the analytical score of the aggregated mixture likelihood."""
        d_count_eta, d_inflation_eta = self._score_factors(params)
        self._apply_weights(d_count_eta)
        self._apply_weights(d_inflation_eta)
        return np.concatenate((
            self.exog.T @ d_count_eta,
            self.exog_infl.T @ d_inflation_eta,
        ))

    def score_obs(self, params, *args, **kwargs):
        """Return unweighted zero-inflated Poisson observation scores."""
        d_count_eta, d_inflation_eta = self._score_factors(params)
        return _build_score_obs((
            (self.exog, d_count_eta),
            (self.exog_infl, d_inflation_eta),
        ))

    def loglike_obs(self, params, *args, **kwargs):
        """Return one unweighted mixture log likelihood per observation."""
        return self._loglike_obs(params)


class ZeroInflatedNegativeBinomial(_ZeroInflatedModel):
    """Zero-inflated NB-2 regression with an estimated dispersion.

    The count component has mean ``mu = exp(exog @ beta)`` and variance
    ``mu + alpha * mu ** 2``, where ``alpha = exp(log_alpha)``.  Structural
    zeros have probability ``expit(exog_infl @ gamma)``.  Parameter order is
    ``beta``, ``gamma``, then ``log_alpha``.
    """

    def get_param_names(self):
        """Return count, inflation, and log-dispersion parameter names."""
        return (
            self._get_regression_param_names()
            + self._get_inflation_param_names()
            + ['log_alpha']
        )

    def get_start_params(self):
        """Return mixture starts plus an NB-2 moment dispersion."""
        count_start, inflation_start, inflation_probability, latent_mean = (
            self._mixture_start_components()
        )
        _, observed_variance = self._response_moments()
        count_probability = 1.0 - inflation_probability
        count_variance = (
            observed_variance
            - inflation_probability * count_probability * latent_mean ** 2
        ) / count_probability
        alpha = (
            (count_variance - latent_mean) / latent_mean ** 2
        )
        log_alpha = self._log_dispersion_start(alpha, upper=3.0)
        return np.concatenate((
            count_start, inflation_start, [log_alpha]
        ))

    def _count_zero_probability(self, params, count_mean):
        """Return the NB-2 component probability of a zero count."""
        log_alpha = np.asarray(params, dtype=float)[-1]
        if log_alpha <= _POISSON_LIMIT_LOG_ALPHA:
            return np.exp(-count_mean)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            alpha = np.exp(log_alpha)
            zero_probability = np.exp(
                -np.log1p(alpha * count_mean) / alpha
            )
        return np.where(
            np.isfinite(zero_probability), zero_probability, np.nan
        )

    def _distribution_terms(self, params):
        """Compute stable NB-2 terms and split the two coefficient vectors."""
        k_count = self.exog.shape[1]
        inflation_end = k_count + self.k_inflate
        count_params = params[:k_count]
        inflation_params = params[k_count:inflation_end]
        log_alpha = params[-1]
        eta = self.exog @ count_params
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            size = np.exp(-log_alpha)
            z = eta + log_alpha
            log_denom = np.logaddexp(0.0, z)
        valid = (
            np.isfinite(eta)
            & np.isfinite(size)
            & (size > 0.0)
            & np.isfinite(log_denom)
        )
        return (
            eta, size, log_alpha, log_denom, inflation_params, valid
        )

    def _count_loglike_terms(self, params):
        """Return observed and zero NB-2 log-likelihood contributions."""
        if params[-1] <= _POISSON_LIMIT_LOG_ALPHA:
            k_count = self.exog.shape[1]
            eta = self.exog @ params[:k_count]
            mean = np.exp(eta)
            return (
                _poisson_loglike_obs(self.endog, eta),
                -mean,
                params[k_count:k_count + self.k_inflate],
            )
        eta, size, log_alpha, log_denom, inflation_params, valid = (
            self._distribution_terms(params)
        )
        safe_size = size if np.isfinite(size) and size > 0.0 else 1.0
        safe_eta = np.where(valid, eta, 0.0)
        safe_log_denom = np.where(valid, log_denom, 0.0)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            count_loglike = (
                gammaln(self.endog + safe_size)
                - gammaln(safe_size)
                - gammaln(self.endog + 1.0)
                - safe_size * safe_log_denom
                + self.endog * (
                    safe_eta + log_alpha - safe_log_denom
                )
            )
            count_loglike_zero = -safe_size * safe_log_denom
        return (
            np.where(valid, count_loglike, -np.inf),
            np.where(valid, count_loglike_zero, -np.inf),
            inflation_params,
        )

    def _loglike_obs(self, params):
        """Compute unweighted zero-inflated NB-2 contributions."""
        count_loglike, count_loglike_zero, inflation_params = (
            self._count_loglike_terms(params)
        )
        return self._mixture_terms(
            inflation_params, count_loglike, count_loglike_zero
        )[0]

    def _score_factors(self, params):
        """Return count, inflation, and log-dispersion score factors."""
        if params[-1] <= _POISSON_LIMIT_LOG_ALPHA:
            k_count = self.exog.shape[1]
            eta = self.exog @ params[:k_count]
            count_loglike = _poisson_loglike_obs(self.endog, eta)
            with np.errstate(over='ignore', invalid='ignore'):
                mean = np.exp(eta)
            count_loglike_zero = -mean
            inflation_params = params[
                k_count:k_count + self.k_inflate
            ]
            _, posterior_count, d_inflation_eta = self._mixture_terms(
                inflation_params, count_loglike, count_loglike_zero
            )
            d_count_eta = posterior_count * _poisson_score_factor(
                self.endog, eta
            )
            return d_count_eta, d_inflation_eta, np.zeros(self.nobs)

        eta, size, log_alpha, log_denom, inflation_params, valid = (
            self._distribution_terms(params)
        )
        safe_size = size if np.isfinite(size) and size > 0.0 else 1.0
        safe_log_denom = np.where(valid, log_denom, 0.0)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            count_loglike = (
                gammaln(self.endog + safe_size)
                - gammaln(safe_size)
                - gammaln(self.endog + 1.0)
                - safe_size * safe_log_denom
                + self.endog * (
                    eta + log_alpha - safe_log_denom
                )
            )
            count_loglike_zero = -safe_size * safe_log_denom
            d_count_eta = self.endog - (
                self.endog + safe_size
            ) * expit(eta + log_alpha)
            d_log_alpha = (
                d_count_eta
                + safe_size * (
                    digamma(safe_size)
                    - digamma(self.endog + safe_size)
                    + safe_log_denom
                )
            )

        count_loglike = np.where(valid, count_loglike, -np.inf)
        count_loglike_zero = np.where(valid, count_loglike_zero, -np.inf)
        loglike_obs, posterior_count, d_inflation_eta = (
            self._mixture_terms(
                inflation_params, count_loglike, count_loglike_zero
            )
        )
        del loglike_obs
        safe_d_count_eta = np.where(valid, d_count_eta, 0.0)
        safe_d_log_alpha = np.where(valid, d_log_alpha, 0.0)
        return (
            np.where(valid, posterior_count * safe_d_count_eta, np.nan),
            d_inflation_eta,
            np.where(valid, posterior_count * safe_d_log_alpha, np.nan),
        )

    def loglike(self, params, *args, **kwargs):
        """Return the aggregated zero-inflated NB-2 log-likelihood."""
        return self._weighted_sum(self._loglike_obs(params))

    def score(self, params, *args, **kwargs):
        """Return the analytical score of the aggregated mixture likelihood."""
        d_count_eta, d_inflation_eta, d_log_alpha = (
            self._score_factors(params)
        )
        self._apply_weights(d_count_eta)
        self._apply_weights(d_inflation_eta)
        self._apply_weights(d_log_alpha)
        return np.concatenate((
            self.exog.T @ d_count_eta,
            self.exog_infl.T @ d_inflation_eta,
            [d_log_alpha.sum()],
        ))

    def score_obs(self, params, *args, **kwargs):
        """Return unweighted zero-inflated NB-2 observation scores."""
        d_count_eta, d_inflation_eta, d_log_alpha = (
            self._score_factors(params)
        )
        return _build_score_obs((
            (self.exog, d_count_eta),
            (self.exog_infl, d_inflation_eta),
            (d_log_alpha, None),
        ))

    def loglike_obs(self, params, *args, **kwargs):
        """Return one unweighted mixture log likelihood per observation."""
        return self._loglike_obs(params)


class GeneralizedPoisson(_NonnegativeDistributionalModel):
    """Generalized-Poisson regression with raw dispersion ``alpha``.

    ``p=1`` gives GP-1 with variance ``mu * (1 + alpha) ** 2``;
    ``p=2`` gives GP-2 with variance ``mu * (1 + alpha * mu) ** 2``.
    Positive alpha permits overdispersion, valid negative alpha permits
    underdispersion, and ``alpha=0`` recovers Poisson.
    """

    def __init__(
            self, endog, exog, weights=None, p=1, endog_name=None,
            exog_names=None, weights_name=None, **model_metadata):
        """Initialize the model and select its variance parameterization.

        Args:
            endog: Response observations.
            exog: Regression design matrix.
            weights: Optional observation likelihood weights.
            p: Positive parameterization index.  Common choices are ``1`` for
                GP-1 and ``2`` for GP-2.
            endog_name: Optional response name.
            exog_names: Optional regression coefficient names.
            weights_name: Optional likelihood-weight variable name.
            **model_metadata: Remaining metadata accepted by
                :class:`DistributionalModel`.
        """
        if not np.isscalar(p) or not np.isfinite(p) or p <= 0:
            raise ValueError("p must be a finite positive scalar")
        self.p = p
        self.parameterization = p - 1.0
        super().__init__(
            endog,
            exog,
            weights=weights,
            endog_name=endog_name,
            exog_names=exog_names,
            weights_name=weights_name,
            **model_metadata,
        )

    def get_param_names(self):
        """Return coefficient names followed by the raw ``alpha`` parameter."""
        return self._get_regression_param_names() + ['alpha']

    def get_start_params(self):
        """Return log-mean coefficients and a stable near-Poisson alpha."""
        return np.append(self._mean_regression_start(), 0.05)

    def _distribution_terms(self, params):
        """Compute reusable generalized-Poisson likelihood terms.

        Returns:
            Tuple ``(eta, mu, mu_p, alpha, a1, a2)`` used by the likelihood
            and score, with ``mu_p = mu ** (p - 1)``.
        """
        eta = self.exog @ params[:-1]
        alpha = params[-1]
        # Extreme trial parameters are ordinary during line searches.  Their
        # implied means can be outside floating-point range; downstream
        # support checks reject those points, so do not emit runtime warnings.
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            mu = np.exp(eta)
            mu_p = np.exp(self.parameterization * eta)
            a1 = 1.0 + alpha * mu_p
            a2 = mu + alpha * mu_p * self.endog
        return eta, mu, mu_p, alpha, a1, a2

    @staticmethod
    def _valid_distribution_terms(eta, mu, mu_p, a1, a2):
        """Return the observation mask satisfying finite-value and support rules."""
        return (
            np.isfinite(eta)
            & np.isfinite(mu)
            & np.isfinite(mu_p)
            & np.isfinite(a1)
            & np.isfinite(a2)
            & (mu > 0.0)
            & (mu_p > 0.0)
            & (a1 > 0.0)
            & (a2 > 0.0)
        )

    def _loglike_obs(self, params):
        """Compute generalized-Poisson log-likelihood contributions.

        Observations outside the parameter-dependent support receive
        ``-inf`` so line searches reject invalid trial parameters.
        """
        eta, mu, mu_p, _, a1, a2 = self._distribution_terms(params)
        valid = self._valid_distribution_terms(
            eta, mu, mu_p, a1, a2
        )
        safe_a1 = np.where(valid, a1, 1.0)
        safe_a2 = np.where(valid, a2, 1.0)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            llf_obs = (
                    eta
                    + (self.endog - 1.0) * np.log(safe_a2)
                    - self.endog * np.log(safe_a1)
                    - gammaln(self.endog + 1.0)
                    - safe_a2 / safe_a1
            )
        return np.where(valid, llf_obs, -np.inf)

    def _score_factors(self, params):
        """Compute derivatives with respect to ``eta`` and raw ``alpha``."""
        eta, mu, mu_p, alpha, a1, a2 = self._distribution_terms(params)
        valid = self._valid_distribution_terms(
            eta, mu, mu_p, a1, a2
        )
        safe_mu = np.where(valid, mu, 1.0)
        safe_mu_p = np.where(valid, mu_p, 1.0)
        safe_a1 = np.where(valid, a1, 1.0)
        safe_a2 = np.where(valid, a2, 1.0)

        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            a3 = (
                alpha * self.parameterization * safe_mu_p / safe_mu
            )
            a4 = a3 * self.endog
            d_eta = 1.0 + safe_mu * (
                    -a4 / safe_a1
                    + a3 * safe_a2 / safe_a1 ** 2
                    + (1.0 + a4) * (
                        (self.endog - 1.0) / safe_a2 - 1.0 / safe_a1
                    )
            )
            d_alpha = safe_mu_p * (
                    self.endog * (
                        (self.endog - 1.0) / safe_a2 - 2.0 / safe_a1
                    )
                    + safe_a2 / safe_a1 ** 2
            )
        return (
            np.where(valid, d_eta, np.nan),
            np.where(valid, d_alpha, np.nan),
        )

    def _inference_issues(self, params):
        """Flag invalid or near-boundary generalized-Poisson support."""
        eta, mu, mu_p, _, a1, a2 = self._distribution_terms(params)
        valid = self._valid_distribution_terms(
            eta, mu, mu_p, a1, a2
        )
        if not np.all(valid):
            return [
                'The fitted generalized-Poisson parameters violate the '
                'distribution support.'
            ]
        relative_a2 = a2 / np.maximum(mu, 1.0)
        support_margin = min(float(np.min(a1)), float(np.min(relative_a2)))
        if support_margin <= 1e-5:
            return [
                'The generalized-Poisson estimate is on or extremely near '
                'its parameter-dependent support boundary.'
            ]
        return []

    def loglike(self, params, *args, **kwargs):
        """Return the aggregated generalized-Poisson log-likelihood."""
        return self._weighted_sum(self._loglike_obs(params))

    def score(self, params, *args, **kwargs):
        """Return the analytical score of the aggregated likelihood."""
        d_eta, d_alpha = self._score_factors(params)
        self._apply_weights(d_eta)
        self._apply_weights(d_alpha)
        return np.append(self.exog.T @ d_eta, d_alpha.sum())

    def score_obs(self, params, *args, **kwargs):
        """Return unweighted generalized-Poisson observation scores."""
        d_eta, d_alpha = self._score_factors(params)
        return _build_score_obs((
            (self.exog, d_eta),
            (d_alpha, None),
        ))

    def loglike_obs(self, params, *args, **kwargs):
        """Return unweighted generalized-Poisson likelihood contributions."""
        return self._loglike_obs(params)


class NegativeBinomial1(_NonnegativeDistributionalModel):
    """NB-1 log-link regression with variance ``mu * (1 + alpha)``.

    The conditional mean is ``mu = exp(X beta)`` and
    ``alpha = exp(log_alpha)``.  Unlike NB-2, the NB-1 overdispersion term is
    linear in the mean.  Parameter order is ``beta`` followed by
    ``log_alpha``.
    """

    def get_param_names(self):
        """Return coefficient names followed by ``log_alpha``."""
        return self._get_regression_param_names() + ['log_alpha']

    def get_start_params(self):
        """Return log-mean coefficients and an NB-1 moment dispersion."""
        mean, variance = self._response_moments()
        safe_mean = max(mean, 1e-6)
        alpha = (variance - safe_mean) / safe_mean
        return np.append(
            self._mean_regression_start(safe_mean),
            self._log_dispersion_start(alpha),
        )

    def _loglike_obs(self, params):
        """Compute unweighted NB-1 log-likelihood contributions."""
        log_alpha = params[-1]
        eta = self.exog @ params[:-1]
        if log_alpha <= _POISSON_LIMIT_LOG_ALPHA:
            return _poisson_loglike_obs(self.endog, eta)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            size = np.exp(eta - log_alpha)
            log1p_alpha = np.logaddexp(0.0, log_alpha)

            values = (
                gammaln(self.endog + size)
                - gammaln(size)
                - gammaln(self.endog + 1.0)
                - size * log1p_alpha
                + self.endog * (log_alpha - log1p_alpha)
            )
        valid = np.isfinite(size) & (size > 0.0) & np.isfinite(values)
        return np.where(valid, values, -np.inf)

    def _score_factors(self, params):
        """Compute NB-1 derivatives with respect to ``eta`` and ``log_alpha``."""
        log_alpha = params[-1]
        eta = self.exog @ params[:-1]
        if log_alpha <= _POISSON_LIMIT_LOG_ALPHA:
            return (
                _poisson_score_factor(self.endog, eta),
                np.zeros(self.nobs),
            )
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            size = np.exp(eta - log_alpha)
            log1p_alpha = np.logaddexp(0.0, log_alpha)
        alpha_ratio = expit(log_alpha)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            d_eta = size * (
                    digamma(self.endog + size) - digamma(size) - log1p_alpha
            )
            d_log_alpha = (
                    -d_eta
                    + self.endog * (1.0 - alpha_ratio)
                    - size * alpha_ratio
            )
        valid = (
            np.isfinite(size) & (size > 0.0)
            & np.isfinite(d_eta) & np.isfinite(d_log_alpha)
        )
        return (
            np.where(valid, d_eta, np.nan),
            np.where(valid, d_log_alpha, np.nan),
        )

    def loglike(self, params, *args, **kwargs):
        """Return the weighted or unweighted NB-1 log-likelihood."""
        return self._weighted_sum(self._loglike_obs(params))

    def score(self, params, *args, **kwargs):
        """Return the analytical score of the aggregated NB-1 likelihood."""
        d_eta, d_log_alpha = self._score_factors(params)
        self._apply_weights(d_eta)
        self._apply_weights(d_log_alpha)
        return np.append(self.exog.T @ d_eta, d_log_alpha.sum())

    def score_obs(self, params, *args, **kwargs):
        """Return unweighted NB-1 scores for every observation."""
        d_eta, d_log_alpha = self._score_factors(params)
        return _build_score_obs((
            (self.exog, d_eta),
            (d_log_alpha, None),
        ))

    def loglike_obs(self, params, *args, **kwargs):
        """Return one unweighted NB-1 log-likelihood value per observation."""
        return self._loglike_obs(params)


class NegativeBinomial2(_NonnegativeDistributionalModel):
    """NB-2 log-link regression with variance ``mu + alpha * mu ** 2``.

    The conditional mean is ``mu = exp(X beta)`` and
    ``alpha = exp(log_alpha)``.  Parameter order is ``beta`` followed by
    ``log_alpha``; the corresponding negative-binomial size is ``1 / alpha``.
    """

    def get_param_names(self):
        """Return coefficient names followed by ``log_alpha``."""
        return self._get_regression_param_names() + ['log_alpha']

    def get_start_params(self):
        """Return log-mean coefficients and an NB-2 moment dispersion."""
        mean, variance = self._response_moments()
        safe_mean = max(mean, 1e-6)
        alpha = (variance - safe_mean) / safe_mean ** 2
        return np.append(
            self._mean_regression_start(safe_mean),
            self._log_dispersion_start(alpha),
        )

    def _loglike_obs(self, params):
        """Compute unweighted NB-2 log-likelihood contributions."""
        log_alpha = params[-1]
        eta = self.exog @ params[:-1]
        if log_alpha <= _POISSON_LIMIT_LOG_ALPHA:
            return _poisson_loglike_obs(self.endog, eta)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            size = np.exp(-log_alpha)
            log_denom = np.logaddexp(0.0, eta + log_alpha)

            values = (
                gammaln(self.endog + size)
                - gammaln(size)
                - gammaln(self.endog + 1.0)
                - size * log_denom
                + self.endog * (eta + log_alpha - log_denom)
            )
        valid = np.isfinite(size) & (size > 0.0) & np.isfinite(values)
        return np.where(valid, values, -np.inf)

    def _score_factors(self, params):
        """Compute NB-2 derivatives with respect to ``eta`` and ``log_alpha``."""
        log_alpha = params[-1]
        eta = self.exog @ params[:-1]
        if log_alpha <= _POISSON_LIMIT_LOG_ALPHA:
            return (
                _poisson_score_factor(self.endog, eta),
                np.zeros(self.nobs),
            )
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            size = np.exp(-log_alpha)
            z = eta + log_alpha
            log_denom = np.logaddexp(0.0, z)
            d_eta = self.endog - (self.endog + size) * expit(z)
            d_log_alpha = (
                    d_eta
                    + size * (digamma(size) - digamma(self.endog + size)
                              + log_denom)
            )
        valid = (
            np.isfinite(size) & (size > 0.0)
            & np.isfinite(d_eta) & np.isfinite(d_log_alpha)
        )
        return (
            np.where(valid, d_eta, np.nan),
            np.where(valid, d_log_alpha, np.nan),
        )

    def loglike(self, params, *args, **kwargs):
        """Return the weighted or unweighted NB-2 log-likelihood."""
        return self._weighted_sum(self._loglike_obs(params))

    def score(self, params, *args, **kwargs):
        """Return the analytical score of the aggregated NB-2 likelihood."""
        d_eta, d_log_alpha = self._score_factors(params)
        self._apply_weights(d_eta)
        self._apply_weights(d_log_alpha)
        return np.append(self.exog.T @ d_eta, d_log_alpha.sum())

    def score_obs(self, params, *args, **kwargs):
        """Return unweighted NB-2 scores for every observation."""
        d_eta, d_log_alpha = self._score_factors(params)
        return _build_score_obs((
            (self.exog, d_eta),
            (d_log_alpha, None),
        ))

    def loglike_obs(self, params, *args, **kwargs):
        """Return one unweighted NB-2 log-likelihood value per observation."""
        return self._loglike_obs(params)


__all__ = [
    'DistributionalModel',
    'Poisson',
    'GeneralizedPoisson',
    'NegativeBinomial1',
    'NegativeBinomial2',
    'ZeroInflatedPoisson',
    'ZeroInflatedNegativeBinomial',
]
