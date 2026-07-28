from __future__ import absolute_import, print_function

from abc import ABC, abstractmethod
import numpy as np

from scipy.special import digamma, expit, gammaln
from kanly.api import bfgs_pqn


class CountModel(ABC):
    def __init__(self, endog, exog, weights=None):
        self.endog = endog
        self.exog = exog
        self.weights = weights
        self.is_weighted = self.weights is not None
        self.nobs = len(self.endog)

    def _apply_weights(self, values):
        if self.is_weighted:
            np.multiply(values, self.weights, out=values)
        return values

    def _weighted_sum(self, values):
        if self.is_weighted:
            return np.dot(self.weights, values)
        return values.sum()

    @abstractmethod
    def loglike_obs(self, params, *args, **kwargs):
        raise NotImplementedError(
            "Must implement a loglikelihood function at "
            "observation level!")

    def loglike(self, params, *args, **kwargs):
        return self._weighted_sum(self.loglike_obs(params, *args, **kwargs))

    def score_obs(self, params, dx=1e-6, *args, **kwargs):
        f0 = self.loglike_obs(params, *args, **kwargs)
        k = len(params)
        n = len(f0)
        g = np.zeros((n, k))
        for i in range(k):
            paramsi = params.copy()
            paramsi[i] += dx
            fi = self.loglike_obs(paramsi, *args, **kwargs)
            g[:, i] = (fi - f0) / dx
        return g

    def score(self, params, dx=1e-6, *args, **kwargs):
        f0 = self.loglike(params, *args, **kwargs)
        k = len(params)
        g = np.zeros(k)
        for i in range(k):
            paramsi = params.copy()
            paramsi[i] += dx
            fi = self.loglike(paramsi, *args, **kwargs)
            g[i] = (fi - f0) / dx
        return g

    def fit(self, start_params, debug=False):
        return bfgs_pqn(self.loglike, start_params, maximize=True, debug=debug,
                        gradient_callable=self.score)


class Poisson(CountModel):

    def _loglike_obs(self, params):
        eta = self.exog @ params
        return self.endog * eta - np.exp(eta) - gammaln(self.endog + 1.0)

    def loglike(self, params, *args, **kwargs):
        return self._weighted_sum(self._loglike_obs(params))

    def score(self, params, *args, **kwargs):
        residual = self.endog - np.exp(self.exog @ params)
        self._apply_weights(residual)
        return self.exog.T @ residual

    def score_obs(self, params, *args, **kwargs):
        residual = self.endog - np.exp(self.exog @ params)
        return self.exog * residual[:, None]

    def loglike_obs(self, params, *args, **kwargs):
        return self._loglike_obs(params)


class NegativeBinomial1(CountModel):

    def _loglike_obs(self, params):
        log_alpha = params[-1]
        eta = self.exog @ params[:-1]
        size = np.exp(eta - log_alpha)
        log1p_alpha = np.logaddexp(0.0, log_alpha)

        return (
                gammaln(self.endog + size)
                - gammaln(size)
                - gammaln(self.endog + 1.0)
                - size * log1p_alpha
                + self.endog * (log_alpha - log1p_alpha)
        )

    def _score_factors(self, params):
        log_alpha = params[-1]
        eta = self.exog @ params[:-1]
        size = np.exp(eta - log_alpha)
        log1p_alpha = np.logaddexp(0.0, log_alpha)
        alpha_ratio = expit(log_alpha)
        d_eta = size * (
                digamma(self.endog + size) - digamma(size) - log1p_alpha
        )
        d_log_alpha = (
                -d_eta
                + self.endog * (1.0 - alpha_ratio)
                - size * alpha_ratio
        )
        return d_eta, d_log_alpha

    def loglike(self, params, *args, **kwargs):
        return self._weighted_sum(self._loglike_obs(params))

    def score(self, params, *args, **kwargs):
        d_eta, d_log_alpha = self._score_factors(params)
        self._apply_weights(d_eta)
        self._apply_weights(d_log_alpha)
        return np.append(self.exog.T @ d_eta, d_log_alpha.sum())

    def score_obs(self, params, *args, **kwargs):
        d_eta, d_log_alpha = self._score_factors(params)
        return np.column_stack((self.exog * d_eta[:, None], d_log_alpha))

    def loglike_obs(self, params, *args, **kwargs):
        return self._loglike_obs(params)


class NegativeBinomial2(CountModel):

    def _loglike_obs(self, params):
        log_alpha = params[-1]
        size = np.exp(-log_alpha)
        eta = self.exog @ params[:-1]
        log_denom = np.logaddexp(0.0, eta + log_alpha)

        return (
                gammaln(self.endog + size)
                - gammaln(size)
                - gammaln(self.endog + 1.0)
                - size * log_denom
                + self.endog * (eta + log_alpha - log_denom)
        )

    def _score_factors(self, params):
        log_alpha = params[-1]
        size = np.exp(-log_alpha)
        z = self.exog @ params[:-1] + log_alpha
        log_denom = np.logaddexp(0.0, z)
        d_eta = self.endog - (self.endog + size) * expit(z)
        d_log_alpha = (
                d_eta
                + size * (digamma(size) - digamma(self.endog + size)
                          + log_denom)
        )
        return d_eta, d_log_alpha

    def loglike(self, params, *args, **kwargs):
        return self._weighted_sum(self._loglike_obs(params))

    def score(self, params, *args, **kwargs):
        d_eta, d_log_alpha = self._score_factors(params)
        self._apply_weights(d_eta)
        self._apply_weights(d_log_alpha)
        return np.append(self.exog.T @ d_eta, d_log_alpha.sum())

    def score_obs(self, params, *args, **kwargs):
        d_eta, d_log_alpha = self._score_factors(params)
        return np.column_stack((self.exog * d_eta[:, None], d_log_alpha))

    def loglike_obs(self, params, *args, **kwargs):
        return self._loglike_obs(params)


if __name__ == '__main__':
    np.random.seed(0)
    n = 40
    from kanly.api import GLM
    x = np.exp(np.random.randn(n))
    y = (np.exp(.2) * x ** .8) * np.exp(.3 * np.random.randn(n))
    X = np.vstack([np.ones(n), np.log(x)]).T
    w = np.exp(np.random.randn(n))

    poisson = Poisson(y, X, weights=w)
    print(poisson.fit([.1] * X.shape[1]))

    print(GLM(y, X, var_weights=w, family='poisson'))
