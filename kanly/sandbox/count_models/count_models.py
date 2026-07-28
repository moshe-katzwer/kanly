from abc import ABC, abstractmethod
import numpy as np

from scipy.special import gammaln
from kanly.api import bfgs_pqn


class CountModel(ABC):
    def __init__(self, endog, exog):
        self.endog = endog
        self.exog = exog
        self.nobs = len(self.endog)

    @abstractmethod
    def loglike_obs(self, params, *args, **kwargs):
        raise NotImplementedError()

    def loglike(self, params, *args, **kwargs):
        return self.loglike_obs(params, *args, **kwargs).sum()

    def score_obs(self, params, *args, **kwargs):
        f0 = self.loglike_obs(params, *args, **kwargs)
        k = len(params)
        n = len(f0)
        g = np.zeros((n, k))
        h = 1e-6
        for i in range(k):
            paramsi = params.copy()
            paramsi[i] += h
            fi = self.loglike_obs(params, *args, **kwargs)
            g[:, i] = (fi - f0) / h
        return g

    def score(self, params, *args, **kwargs):
        f0 = self.loglike(params, *args, **kwargs)
        k = len(params)
        g = np.zeros(k)
        h = 1e-6
        for i in range(k):
            paramsi = params.copy()
            paramsi[i] += h
            fi = self.loglike(params, *args, **kwargs)
            g[i] = (fi - f0) / h
        return g

    def fit(self, start_params):
        return bfgs_pqn(nb2.loglike, [1., 1, 1], maximize=True)


class NegativeBinomial2(CountModel):

    def loglike_obs(self, params, *args, **kwargs):
        beta = params[:-1]
        mu = np.exp(self.exog @ beta)
        alpha = np.exp(params[-1])
        size = 1.0 / alpha

        llf_obs = (
                gammaln(self.endog + size)
                - gammaln(size)
                - gammaln(self.endog + 1.0)
                + size * (np.log(size) - np.log(size + mu))
                + self.endog * (np.log(mu) - np.log(size + mu))
        )

        return llf_obs


if __name__ == '__main__':
    np.random.seed(0)
    n = 1000
    x = np.exp(np.random.randn(n))
    y = (np.exp(.2) * x ** .8) * np.exp(.3 * np.random.randn(n))
    X = np.vstack([np.ones(n), np.log(x)]).T

    nb2 = NegativeBinomial2(y, X)
    print(nb2.fit())
