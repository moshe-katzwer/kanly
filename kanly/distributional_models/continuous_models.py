"""Direct-likelihood regression models for continuous distributions."""

from __future__ import absolute_import, print_function

import numpy as np
from scipy.special import digamma, gammaln

from kanly.distributional_models.base import (
    DistributionalModel,
    _build_score_obs,
)


class Gamma(DistributionalModel):
    """Gamma regression for strictly positive continuous outcomes.

    The conditional mean is ``mu = exp(exog @ beta)`` and the variance is
    ``alpha * mu ** 2``. The final estimated parameter is ``log_alpha``, so
    the implied dispersion and Gamma shape are always positive:
    ``alpha = exp(log_alpha)`` and ``shape = 1 / alpha``.
    """

    def __init__(
            self, endog, exog, weights=None, endog_name=None,
            exog_names=None, weights_name=None, **model_metadata):
        """Initialize Gamma data, names, metadata, and positive support."""
        endog = np.asarray(endog, dtype=float)
        if endog.ndim != 1:
            raise ValueError("Gamma outcomes must be one-dimensional")
        if np.any(~np.isfinite(endog)) or np.any(endog <= 0.0):
            raise ValueError(
                "Gamma outcomes must be finite and strictly positive"
            )
        super().__init__(
            endog,
            exog,
            weights=weights,
            endog_name=endog_name,
            exog_names=exog_names,
            weights_name=weights_name,
            **model_metadata,
        )
        self._log_endog = np.log(self.endog)

    def get_param_names(self):
        """Return coefficient names followed by ``log_alpha``."""
        return self._get_regression_param_names() + ['log_alpha']

    def get_start_params(self):
        """Return log-mean coefficients and a Gamma moment dispersion."""
        mean, variance = self._response_moments()
        safe_mean = max(mean, 1e-6)
        alpha = variance / safe_mean ** 2
        return np.append(
            self._mean_regression_start(safe_mean),
            self._log_dispersion_start(alpha),
        )

    def _distribution_terms(self, params):
        """Compute the linear predictor, shape, response ratio, and validity.

        Returns:
            Tuple ``(eta, shape, y_over_mu, valid)`` where ``valid`` is an
            observation-level support and finite-value mask.
        """
        eta = self.exog @ params[:-1]
        log_alpha = params[-1]
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            shape = np.exp(-log_alpha)
            ratio = np.exp(self._log_endog - eta)
        valid = (
            np.isfinite(eta)
            & np.isfinite(ratio)
            & np.isfinite(shape)
            & (shape > 0.0)
        )
        return eta, shape, ratio, valid

    def _loglike_obs(self, params):
        """Compute unweighted Gamma log-density contributions."""
        eta, shape, ratio, valid = self._distribution_terms(params)
        safe_shape = shape if np.isfinite(shape) and shape > 0.0 else 1.0
        safe_eta = np.where(valid, eta, 0.0)
        safe_ratio = np.where(valid, ratio, 1.0)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            llf_obs = (
                safe_shape * np.log(safe_shape)
                - gammaln(safe_shape)
                + (safe_shape - 1.0) * self._log_endog
                - safe_shape * safe_eta
                - safe_shape * safe_ratio
            )
        return np.where(valid, llf_obs, -np.inf)

    def _score_factors(self, params):
        """Compute derivatives with respect to ``eta`` and ``log_alpha``."""
        eta, shape, ratio, valid = self._distribution_terms(params)
        safe_shape = shape if np.isfinite(shape) and shape > 0.0 else 1.0
        safe_eta = np.where(valid, eta, 0.0)
        safe_ratio = np.where(valid, ratio, 1.0)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            d_eta = safe_shape * (safe_ratio - 1.0)
            d_log_alpha = safe_shape * (
                digamma(safe_shape)
                - np.log(safe_shape)
                - 1.0
                - self._log_endog
                + safe_eta
                + safe_ratio
            )
        return (
            np.where(valid, d_eta, np.nan),
            np.where(valid, d_log_alpha, np.nan),
        )

    def loglike(self, params, *args, **kwargs):
        """Return the weighted or unweighted Gamma log-likelihood."""
        return self._weighted_sum(self._loglike_obs(params))

    def score(self, params, *args, **kwargs):
        """Return the analytical score of the aggregated Gamma likelihood."""
        d_eta, d_log_alpha = self._score_factors(params)
        self._apply_weights(d_eta)
        self._apply_weights(d_log_alpha)
        return np.append(self.exog.T @ d_eta, d_log_alpha.sum())

    def score_obs(self, params, *args, **kwargs):
        """Return unweighted Gamma scores for every observation."""
        d_eta, d_log_alpha = self._score_factors(params)
        return _build_score_obs((
            (self.exog, d_eta),
            (d_log_alpha, None),
        ))

    def loglike_obs(self, params, *args, **kwargs):
        """Return one unweighted Gamma log-density value per observation."""
        return self._loglike_obs(params)


__all__ = ['Gamma']
