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

    def score_obs(self, params, dx=None, *args, **kwargs):
        if dx is None:
            dx = np.cbrt(np.finfo(float).eps)

        f0 = self.loglike_obs(params, *args, **kwargs)
        k = len(params)
        n = len(f0)
        g = np.zeros((n, k))
        for i in range(k):
            step = dx * max(1.0, abs(params[i]))
            paramsi = params.copy()
            paramsi[i] += step
            fi = self.loglike_obs(paramsi, *args, **kwargs)
            g[:, i] = (fi - f0) / step
        return g

    def score(self, params, dx=None, *args, **kwargs):
        if dx is None:
            dx = np.cbrt(np.finfo(float).eps)

        f0 = self.loglike(params, *args, **kwargs)
        k = len(params)
        g = np.zeros(k)
        for i in range(k):
            step = dx * max(1.0, abs(params[i]))
            paramsi = params.copy()
            paramsi[i] += step
            fi = self.loglike(paramsi, *args, **kwargs)
            g[i] = (fi - f0) / step
        return g

    def hessian(self, params, dx=None, *args, **kwargs):
        params = np.asarray(params, dtype=float)
        if dx is None:
            dx = np.cbrt(np.finfo(float).eps)

        k = len(params)
        hess = np.empty((k, k))
        for i in range(k):
            step = dx * max(1.0, abs(params[i]))
            params_lo = params.copy()
            params_hi = params.copy()
            params_lo[i] -= step
            params_hi[i] += step
            hess[:, i] = (
                    self.score(params_hi, *args, **kwargs)
                    - self.score(params_lo, *args, **kwargs)
            ) / (2.0 * step)

        return (hess + hess.T) / 2.0

    def fit(self, start_params, debug=False, cov_type='SANDWICH'):
        cov_type = str(cov_type).upper()
        if cov_type not in {'SANDWICH', 'NONROBUST'}:
            raise ValueError(
                "cov_type must be either 'SANDWICH' or 'NONROBUST'"
            )

        result = bfgs_pqn(
            self.loglike, start_params, maximize=True, debug=debug,
            gradient_callable=self.score,
        )

        hess = self.hessian(result.x)
        information = -hess
        try:
            bread = np.linalg.inv(information)
        except np.linalg.LinAlgError:
            bread = np.linalg.pinv(information)
        bread = (bread + bread.T) / 2.0

        if cov_type == 'SANDWICH':
            score_obs = self.score_obs(result.x)
            if self.is_weighted:
                score_obs = score_obs * np.asarray(self.weights)[:, None]
            meat = score_obs.T @ score_obs
            cov_params = bread @ meat @ bread.T
            cov_params = (cov_params + cov_params.T) / 2.0
        else:
            meat = None
            cov_params = bread.copy()

        result.params = result.x.copy()
        result.information = information
        result.bread = bread
        result.meat = meat
        result.cov_params = cov_params
        result.bse = np.sqrt(np.clip(np.diag(cov_params), 0.0, np.inf))
        result.standard_errors = result.bse
        result.cov_type = cov_type
        return result


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
    import pandas as pd
    x = np.exp(np.random.randn(n))
    y = (np.exp(.2) * x ** .8) * np.exp(.3 * np.random.randn(n))
    X = np.vstack([np.ones(n), np.log(x)]).T
    w = np.exp(np.random.randn(n))

    poisson = Poisson(y, X, weights=w)
    fit = poisson.fit([.1] * X.shape[1], cov_type='nonrobust')
    print(pd.DataFrame({'coef': fit.params, 'bse': fit.bse}))

    print(GLM(y, X, var_weights=w, family='poisson', cov_type='nonrobust'))
