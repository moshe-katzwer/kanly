from __future__ import absolute_import, print_function

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from scipy.special import digamma, expit, gammaln
from scipy.stats import norm
from kanly.api import bfgs_pqn
from kanly.formula.data_getter import SparseDataGetter
from kanly.formula.keys import (
    ENDOG_KEY, EXOG_KEY, FORMULA_DESIGN_INFO_KEY, HAS_IMPLICIT_CONSTANT_KEY,
    HAS_INTERCEPT_KEY, INDEX_KEY, NULL_ROWS_INFO_DICT_KEY, TIME_ELAPSED_KEY,
    VALID_OBS_ROWS_KEY, WEIGHTS_KEY,
)


class CountModel(ABC):
    def __init__(self, endog, exog, weights=None):
        self.endog = endog
        self.exog = exog
        self.weights = weights
        self.is_weighted = self.weights is not None
        self.nobs = len(self.endog)

        self.formula_design_info = None
        self.formula = None
        self.from_formula = False
        self.endog_name = None
        self.exog_names = None
        self.weights_name = None
        self.exog_term_names = None
        self.exog_term_to_indices = None
        self.has_intercept = False
        self.has_implicit_constant = False
        self.valid_obs_rows = np.arange(self.nobs)
        self.null_rows_info_dict = {}
        self.index = None
        self.model_elapsed = 0.0
        self.param_names = self.get_param_names()

    @classmethod
    def build_model_from_formula(
            cls, formula, data, index=None, debug=False,
            check_constant_cols=False, fail_on_missing=False,
            cache_intermediate=True, sum_to_n=False,
            test_formula_on_dummy=True, drop_1_for_FE=True, **model_kwargs):
        """Build an unfitted count model from Kanly formula syntax.

        The ``$`` formula extension supplies optional likelihood weights.
        Instrumental-variable and absorbed-effect formulas are not supported.
        Missing rows are aligned and removed by ``SparseDataGetter``.
        """
        result = SparseDataGetter.get_data(
            data=data, formula=formula, index=index, debug=debug,
            check_constant_cols=check_constant_cols,
            fail_on_missing=fail_on_missing,
            cache_intermediate=cache_intermediate, sum_to_n=sum_to_n,
            test_formula_on_dummy=test_formula_on_dummy,
            drop_1_for_FE=drop_1_for_FE, fail_on_iv=True,
            fail_on_absorb=True,
        )

        endog_obj = result[ENDOG_KEY]
        exog_obj = result[EXOG_KEY]
        weights_obj = result[WEIGHTS_KEY]

        endog = endog_obj.values
        exog = exog_obj.values
        if hasattr(endog, 'toarray'):
            endog = endog.toarray()
        if hasattr(exog, 'toarray'):
            exog = exog.toarray()
        endog = np.asarray(endog)
        exog = np.asarray(exog)
        if endog.ndim != 2 or endog.shape[1] != 1:
            raise ValueError("Count models require exactly one outcome column")
        endog = endog.reshape(-1)

        if weights_obj is None:
            weights = None
        else:
            weights = weights_obj.values
            if hasattr(weights, 'toarray'):
                weights = weights.toarray()
            weights = np.asarray(weights).reshape(-1)

        model = cls(endog, exog, weights=weights, **model_kwargs)
        model.formula_design_info = result[FORMULA_DESIGN_INFO_KEY]
        model.formula = model.formula_design_info.formula
        model.from_formula = True
        model.endog_name = endog_obj.column_names[0]
        model.exog_names = list(exog_obj.column_names)
        model.weights_name = (
            None if weights_obj is None else weights_obj.column_names[0]
        )
        model.exog_term_names = exog_obj.term_names
        model.exog_term_to_indices = exog_obj.var_2_col_indices
        model.has_intercept = result[HAS_INTERCEPT_KEY]
        model.has_implicit_constant = result[HAS_IMPLICIT_CONSTANT_KEY]
        model.valid_obs_rows = result[VALID_OBS_ROWS_KEY]
        model.null_rows_info_dict = result[NULL_ROWS_INFO_DICT_KEY]
        model.index = result[INDEX_KEY]
        model.model_elapsed = result[TIME_ELAPSED_KEY]

        model.param_names = model.get_param_names()

        return model

    def _apply_weights(self, values):
        if self.is_weighted:
            np.multiply(values, self.weights, out=values)
        return values

    def _weighted_sum(self, values):
        if self.is_weighted:
            return np.dot(self.weights, values)
        return values.sum()

    def _get_regression_param_names(self):
        if self.exog_names is None:
            return [f'x{i}' for i in range(self.exog.shape[1])]
        return [str(name) for name in self.exog_names]

    @abstractmethod
    def get_param_names(self):
        """Return names for every parameter in estimation order."""
        raise NotImplementedError()

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

        result.model = self
        result.params = result.x.copy()
        result.param_names = self.get_param_names()
        result.information = information
        result.bread = bread
        result.meat = meat
        result.cov_params = cov_params
        result.bse = np.sqrt(np.clip(np.diag(cov_params), 0.0, np.inf))
        result.standard_errors = result.bse
        result.cov_type = cov_type

        result.summary_df = pd.DataFrame(
            {
                'coef': result.params,
                'std err': result.bse,
                'z': result.params / result.bse,
                'p>|z|': 2*norm.sf(np.abs(result.params) / result.bse),
            }, 
            index=self.param_names
        )

        return result


class Poisson(CountModel):

    def get_param_names(self):
        return self._get_regression_param_names()

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


class GeneralizedPoisson(CountModel):
    """Generalized-Poisson regression with raw dispersion ``alpha``.

    ``p=1`` gives GP-1 with variance ``mu * (1 + alpha) ** 2``;
    ``p=2`` gives GP-2 with variance ``mu * (1 + alpha * mu) ** 2``.
    Positive alpha permits overdispersion, valid negative alpha permits
    underdispersion, and ``alpha=0`` recovers Poisson.
    """

    def __init__(self, endog, exog, weights=None, p=1):
        super().__init__(endog, exog, weights=weights)
        if not np.isscalar(p) or not np.isfinite(p) or p <= 0:
            raise ValueError("p must be a finite positive scalar")
        self.p = p
        self.parameterization = p - 1.0

    def get_param_names(self):
        return self._get_regression_param_names() + ['alpha']

    def _distribution_terms(self, params):
        eta = self.exog @ params[:-1]
        mu = np.exp(eta)
        mu_p = np.exp(self.parameterization * eta)
        alpha = params[-1]
        a1 = 1.0 + alpha * mu_p
        a2 = mu + alpha * mu_p * self.endog
        return eta, mu, mu_p, alpha, a1, a2

    def _loglike_obs(self, params):
        eta, _, _, _, a1, a2 = self._distribution_terms(params)
        valid = (a1 > 0.0) & (a2 > 0.0)
        safe_a1 = np.where(valid, a1, 1.0)
        safe_a2 = np.where(valid, a2, 1.0)
        llf_obs = (
                eta
                + (self.endog - 1.0) * np.log(safe_a2)
                - self.endog * np.log(safe_a1)
                - gammaln(self.endog + 1.0)
                - safe_a2 / safe_a1
        )
        return np.where(valid, llf_obs, -np.inf)

    def _score_factors(self, params):
        _, mu, mu_p, alpha, a1, a2 = self._distribution_terms(params)
        valid = (a1 > 0.0) & (a2 > 0.0)
        a1 = np.where(valid, a1, np.nan)
        a2 = np.where(valid, a2, np.nan)
        a3 = alpha * self.parameterization * mu_p / mu
        a4 = a3 * self.endog

        d_eta = 1.0 + mu * (
                -a4 / a1
                + a3 * a2 / a1 ** 2
                + (1.0 + a4) * ((self.endog - 1.0) / a2 - 1.0 / a1)
        )
        d_alpha = mu_p * (
                self.endog * ((self.endog - 1.0) / a2 - 2.0 / a1)
                + a2 / a1 ** 2
        )
        return d_eta, d_alpha

    def loglike(self, params, *args, **kwargs):
        return self._weighted_sum(self._loglike_obs(params))

    def score(self, params, *args, **kwargs):
        d_eta, d_alpha = self._score_factors(params)
        self._apply_weights(d_eta)
        self._apply_weights(d_alpha)
        return np.append(self.exog.T @ d_eta, d_alpha.sum())

    def score_obs(self, params, *args, **kwargs):
        d_eta, d_alpha = self._score_factors(params)
        return np.column_stack((self.exog * d_eta[:, None], d_alpha))

    def loglike_obs(self, params, *args, **kwargs):
        return self._loglike_obs(params)


class NegativeBinomial1(CountModel):

    def get_param_names(self):
        return self._get_regression_param_names() + ['log_alpha']

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

    def get_param_names(self):
        return self._get_regression_param_names() + ['log_alpha']

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
    n = 40500
    from kanly.api import GLM
    import pandas as pd
    x = np.exp(np.random.randn(n))
    y = (np.exp(.2) * x ** .8) * np.exp(.3 * np.random.randn(n))
    X = np.vstack([np.ones(n), np.log(x)]).T
    w = np.exp(np.random.randn(n))

    poisson = Poisson.build_model_from_formula('y ~ np.log(x) $ w', dict(x=x,y=y,w=w))
    fit = poisson.fit([.1] * X.shape[1], cov_type='nonrobust')
    print(fit.summary_df)

    poisson = GeneralizedPoisson.build_model_from_formula('y ~ np.log(x) $ w', dict(x=x,y=y,w=w))
    fit = poisson.fit([.1] * 3, cov_type='nonrobust')
    print(fit.summary_df)

    print(GLM(y, X, var_weights=w, family='poisson', cov_type='nonrobust'))
