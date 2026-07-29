"""Likelihood-based regression models for count and positive response data.

The module provides a shared :class:`CountModel` estimation interface plus
Poisson, generalized-Poisson, negative-binomial, and Gamma likelihoods.  All
models support optional observation-level likelihood weights, formula-based
construction, analytical observation scores where available, classical or
sandwich covariance estimation, and Bayesian-bootstrap covariance estimation.
"""

from __future__ import absolute_import, print_function

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from scipy.special import digamma, expit, gammaln
from scipy.stats import norm
from kanly.api import bfgs_pqn
from kanly.bootstrap.bootstrap import (
    DEFAULT_BB_ALPHA, DEFAULT_BB_SEED, DEFAULT_BOOTSTRAP_N_SAMPLES,
    get_bayesian_bootstrap_weights, get_bootstrap_weights2,
)
from kanly.formula.data_getter import SparseDataGetter
from kanly.formula.exceptions import MissingDataException
from kanly.formula.keys import (
    ENDOG_KEY, EXOG_KEY, FORMULA_DESIGN_INFO_KEY, HAS_IMPLICIT_CONSTANT_KEY,
    HAS_INTERCEPT_KEY, INDEX_KEY, NULL_ROWS_INFO_DICT_KEY, TIME_ELAPSED_KEY,
    VALID_OBS_ROWS_KEY, WEIGHTS_KEY,
)
from kanly.utils.util import dict_2_dataframe


class CountModel(ABC):
    """Abstract base class for likelihood-based response models.

    Subclasses provide observation log-likelihoods, parameter names, and,
    preferably, analytical scores.  ``weights`` are estimation weights: they
    multiply observation log-likelihoods and scores during aggregation but do
    not alter the values returned by :meth:`loglike_obs` or :meth:`score_obs`.

    Args:
        endog: One-dimensional response array of length ``nobs``.
        exog: Two-dimensional design matrix with ``nobs`` rows.
        weights: Optional non-negative likelihood weights of length ``nobs``.
    """

    def __init__(self, endog, exog, weights=None):
        """Initialize response data, design data, weights, and model metadata."""
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
        """Multiply an observation array by stored weights in place.

        Args:
            values: Array whose first dimension corresponds to observations.

        Returns:
            The input array after weighting, or unchanged when unweighted.
        """
        if self.is_weighted:
            np.multiply(values, self.weights, out=values)
        return values

    def _weighted_sum(self, values):
        """Aggregate observation values using optional likelihood weights."""
        if self.is_weighted:
            return np.dot(self.weights, values)
        return values.sum()

    def _get_regression_param_names(self):
        """Return formula column names or generated names for coefficients."""
        if self.exog_names is None:
            return [f'x{i}' for i in range(self.exog.shape[1])]
        return [str(name) for name in self.exog_names]

    @abstractmethod
    def get_param_names(self):
        """Return names for every parameter in estimation order."""
        raise NotImplementedError()

    @abstractmethod
    def loglike_obs(self, params, *args, **kwargs):
        """Evaluate unweighted log-likelihood contributions.

        Args:
            params: Model parameters in the order given by
                :meth:`get_param_names`.

        Returns:
            One-dimensional array with one contribution per observation.
        """
        raise NotImplementedError(
            "Must implement a loglikelihood function at "
            "observation level!")

    def loglike(self, params, *args, **kwargs):
        """Return the optionally weighted sum of observation log-likelihoods."""
        return self._weighted_sum(self.loglike_obs(params, *args, **kwargs))

    def score_obs(self, params, dx=None, *args, **kwargs):
        """Approximate unweighted observation scores by forward differences.

        Args:
            params: Parameter vector at which to evaluate the score.
            dx: Relative finite-difference step.  Defaults to the cube root of
                machine precision.

        Returns:
            Array of shape ``(nobs, n_params)``.
        """
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
        """Approximate the gradient of the aggregated log-likelihood."""
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
        """Approximate and symmetrize the aggregated likelihood Hessian.

        Central differences of :meth:`score` are used column by column.

        Returns:
            Square array of shape ``(n_params, n_params)``.
        """
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

    def _fit_internal(self, start_params, weights=None, debug=False):
        """Optimize with temporary likelihood weights and no inference."""
        original_weights = self.weights
        original_is_weighted = self.is_weighted

        if weights is not None:
            weights = np.asarray(weights, dtype=float).reshape(-1)
            if len(weights) != self.nobs:
                raise ValueError("weights must have one value per observation")
            if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
                raise ValueError("weights must be finite and non-negative")

        try:
            self.weights = weights
            self.is_weighted = weights is not None
            return bfgs_pqn(
                self.loglike, start_params, maximize=True, debug=debug,
                gradient_callable=self.score,
            )
        finally:
            # A failed bootstrap fit must not leak its temporary weights into
            # the model used for subsequent draws or post-estimation work.
            self.weights = original_weights
            self.is_weighted = original_is_weighted

    def _bayesian_bootstrap(self, params, cov_kwds, debug=False):
        """Refit Bayesian-bootstrap likelihoods and return their covariance."""
        n_samples = cov_kwds.get(
            'n_samples', DEFAULT_BOOTSTRAP_N_SAMPLES
        )
        if (isinstance(n_samples, bool)
                or not isinstance(n_samples, (int, np.integer))
                or n_samples < 2):
            raise ValueError("n_samples must be an integer of at least 2")

        seed = cov_kwds.get('seed', DEFAULT_BB_SEED)
        alpha = cov_kwds.get('alpha', DEFAULT_BB_ALPHA)
        if not np.isscalar(alpha) or not np.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("alpha must be a finite positive scalar")

        method = str(cov_kwds.get('method', 'BAYESIAN')).upper()
        if method != 'BAYESIAN':
            raise ValueError(
                "CountModel bootstrap covariance currently supports only "
                "method='BAYESIAN'"
            )

        sample_weights, _ = get_bayesian_bootstrap_weights(
            self.nobs, n_samples=n_samples, seed=seed, alpha=alpha
        )
        params = np.asarray(params, dtype=float)
        parameter_draws = []
        n_failed = 0

        for bootstrap_weights in sample_weights:
            # This multiplies the bootstrap draw by any original likelihood
            # weights; it only mutates the newly generated draw.
            combined_weights = get_bootstrap_weights2(
                bootstrap_weights, self.weights
            )
            try:
                fit = self._fit_internal(
                    params, weights=combined_weights, debug=False
                )
            except Exception:
                n_failed += 1
                continue

            draw = np.asarray(fit.x, dtype=float)
            if fit.converged and np.all(np.isfinite(draw)):
                parameter_draws.append(draw.copy())
            else:
                n_failed += 1

        parameter_draws = np.asarray(parameter_draws, dtype=float)
        if len(parameter_draws) < 2:
            raise RuntimeError(
                "Fewer than two Bayesian bootstrap repetitions converged; "
                "the bootstrap covariance cannot be estimated"
            )

        cov_params = np.atleast_2d(
            np.cov(parameter_draws, rowvar=False)
        )
        use_correction = bool(cov_kwds.get('use_correction', True))
        if use_correction:
            n_successful = len(parameter_draws)
            cov_params *= n_successful / (n_successful - 1.0)
        cov_params = (cov_params + cov_params.T) / 2.0

        if debug and n_failed:
            print(
                f'Bayesian bootstrap retained {len(parameter_draws)} of '
                f'{n_samples} repetitions.'
            )

        bootstrap_kwds = {
            'method': method,
            'n_samples': int(n_samples),
            'n_successful': len(parameter_draws),
            'n_failed': n_failed,
            'seed': seed,
            'alpha': alpha,
            'use_correction': use_correction,
        }
        return cov_params, parameter_draws, bootstrap_kwds

    def fit(self, start_params, debug=False, cov_type='SANDWICH',
            cov_kwds=None):
        """Estimate parameters and a likelihood-based covariance matrix.

        For ``cov_type='BOOTSTRAP'``, ``cov_kwds`` may contain ``n_samples``,
        ``seed``, ``alpha``, and ``use_correction``.  Every Bayesian-bootstrap
        draw is multiplied by, rather than substituted for, any original
        observation weights.
        """
        cov_type = str(cov_type).upper()
        if cov_type not in {'SANDWICH', 'NONROBUST', 'BOOTSTRAP'}:
            raise ValueError(
                "cov_type must be 'SANDWICH', 'NONROBUST', or 'BOOTSTRAP'"
            )
        cov_kwds = {} if cov_kwds is None else dict(cov_kwds)

        result = self._fit_internal(
            start_params, weights=self.weights, debug=debug
        )

        bootstrapped_params = None
        bootstrap_string = None
        if cov_type == 'BOOTSTRAP':
            information = None
            bread = None
            meat = None
            cov_params, bootstrapped_params, cov_kwds = (
                self._bayesian_bootstrap(result.x, cov_kwds, debug=debug)
            )
            bootstrap_string = (
                f"Did {len(bootstrapped_params)} Bayesian bootstrap "
                f"repetitions, alpha={cov_kwds['alpha']:.3f}."
            )
        else:
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
                    score_obs = (
                        score_obs * np.asarray(self.weights)[:, None]
                    )
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
        result.cov_kwds = cov_kwds
        result.bootstrapped_params = bootstrapped_params
        result.bootstrap_string = bootstrap_string

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
        return self.endog * eta - np.exp(eta) - gammaln(self.endog + 1.0)

    def loglike(self, params, *args, **kwargs):
        """Return the weighted or unweighted Poisson log-likelihood."""
        return self._weighted_sum(self._loglike_obs(params))

    def score(self, params, *args, **kwargs):
        """Return the analytical score of the aggregated likelihood."""
        residual = self.endog - np.exp(self.exog @ params)
        self._apply_weights(residual)
        return self.exog.T @ residual

    def score_obs(self, params, *args, **kwargs):
        """Return unweighted Poisson scores with shape ``(nobs, n_params)``."""
        residual = self.endog - np.exp(self.exog @ params)
        return self.exog * residual[:, None]

    def loglike_obs(self, params, *args, **kwargs):
        """Return one unweighted Poisson log-likelihood value per row."""
        return self._loglike_obs(params)


class _ZeroInflatedModel(CountModel):
    """Shared data handling and mixture algebra for zero-inflated models.

    ``exog_infl`` controls the probability that an observation is a structural
    zero.  Its coefficients use a logit link.  When it is omitted, a single
    intercept column is used.
    """

    def __init__(self, endog, exog, weights=None, exog_infl=None,
                 exog_infl_names=None):
        """Initialize and validate count and zero-inflation design data.

        Args:
            endog: Finite non-negative response observations.
            exog: Design matrix for the conditional count mean.
            weights: Optional observation likelihood weights.
            exog_infl: Optional design matrix for the structural-zero logit.
                Defaults to an intercept-only matrix.
            exog_infl_names: Optional names for columns of ``exog_infl``.
        """
        endog = np.asarray(endog, dtype=float)
        if endog.ndim != 1:
            raise ValueError(
                "Zero-inflated outcomes must be one-dimensional"
            )
        if np.any(~np.isfinite(endog)) or np.any(endog < 0.0):
            raise ValueError(
                "Zero-inflated outcomes must be finite and non-negative"
            )

        if exog_infl is None:
            exog_infl = np.ones((len(endog), 1), dtype=float)
        else:
            exog_infl = np.asarray(exog_infl, dtype=float)
            if exog_infl.ndim == 1:
                exog_infl = exog_infl[:, None]
        if exog_infl.ndim != 2 or exog_infl.shape[0] != len(endog):
            raise ValueError(
                "exog_infl must be two-dimensional with one row per outcome"
            )
        if np.any(~np.isfinite(exog_infl)):
            raise ValueError("exog_infl must contain only finite values")

        if exog_infl_names is not None:
            exog_infl_names = [str(name) for name in exog_infl_names]
            if len(exog_infl_names) != exog_infl.shape[1]:
                raise ValueError(
                    "exog_infl_names must match the columns of exog_infl"
                )

        self.exog_infl = exog_infl
        self.exog_infl_names = exog_infl_names
        self.k_inflate = exog_infl.shape[1]
        super().__init__(endog, exog, weights=weights)

    @classmethod
    def build_model_from_formula(
            cls, formula, data, index=None, exog_infl=None, debug=False,
            check_constant_cols=False, fail_on_missing=False,
            cache_intermediate=True, sum_to_n=False,
            test_formula_on_dummy=True, drop_1_for_FE=True, **model_kwargs):
        """Build count and inflation equations from Patsy-style formulas.

        Args:
            formula: Full count-model formula, including the outcome and
                optional ``$`` likelihood-weight expression.
            data: DataFrame or dict-like object containing formula variables.
            index: Optional boolean or integer row selector.
            exog_infl: Patsy-style right-hand-side formula for the
                structural-zero logit, such as ``'z1 + C(group)'``.  ``None``
                creates a single constant column.
            debug: Whether to print formula-construction diagnostics.
            check_constant_cols: Whether to remove redundant constant columns.
            fail_on_missing: Raise instead of dropping rows with missing data.
            cache_intermediate: Formula-term cache configuration.
            sum_to_n: Normalize likelihood weights to sum to retained rows.
            test_formula_on_dummy: Validate the count formula on dummy data.
            drop_1_for_FE: Apply the formula engine's drop-one categorical rule.
            **model_kwargs: Additional constructor arguments.

        Returns:
            An unfitted zero-inflated model of type ``cls`` with aligned count,
            inflation, response, and weight arrays.
        """
        if exog_infl is not None and not isinstance(exog_infl, str):
            raise TypeError(
                "exog_infl must be a Patsy-style string or None when using "
                "build_model_from_formula"
            )

        # Formula-derived names replace any matrix-API names.  Removing the
        # keyword before the initial intercept-only construction also avoids a
        # temporary name-count mismatch.
        if exog_infl is not None:
            model_kwargs.pop('exog_infl_names', None)

        model = super().build_model_from_formula(
            formula=formula,
            data=data,
            index=index,
            debug=debug,
            check_constant_cols=check_constant_cols,
            fail_on_missing=fail_on_missing,
            cache_intermediate=cache_intermediate,
            # Normalize only after inflation-specific missing rows are removed;
            # the final retained sample size is the appropriate target.
            sum_to_n=False,
            test_formula_on_dummy=test_formula_on_dummy,
            drop_1_for_FE=drop_1_for_FE,
            **model_kwargs,
        )
        model.exog_infl_formula = exog_infl
        model.exog_infl_term_names = ['Intercept']
        model.exog_infl_term_to_indices = {'Intercept': np.array([0])}

        if exog_infl is None:
            return model

        data_frame = dict_2_dataframe(data)
        inflation_obj = SparseDataGetter.sparse_dmatrix(
            exog_infl,
            data_frame,
            debug=debug,
            check_constant_cols=check_constant_cols,
            cache_intermediate=cache_intermediate,
            drop_1_for_FE=drop_1_for_FE,
            name='EXOG_INFL',
            index=index,
        )
        inflation_null_rows = inflation_obj.null_rows.copy()
        if fail_on_missing and inflation_null_rows:
            missing_rows = sorted(int(row) for row in inflation_null_rows)
            raise MissingDataException(
                "Inflation exog has missing data in rows "
                f"{missing_rows}!"
            )

        main_valid_rows = np.asarray(model.valid_obs_rows, dtype=int)
        keep_main_rows = ~np.isin(main_valid_rows, list(inflation_null_rows))
        final_valid_rows = main_valid_rows[keep_main_rows]
        if len(final_valid_rows) == 0:
            raise ValueError(
                "No valid observations remain after aligning exog_infl"
            )

        # The count formula has already removed its own invalid rows.  Apply
        # only the additional inflation exclusions to those aligned arrays.
        model.endog = np.asarray(model.endog)[keep_main_rows]
        model.exog = np.asarray(model.exog)[keep_main_rows]
        if model.weights is not None:
            model.weights = np.asarray(model.weights)[keep_main_rows]

        # Inflation values still refer to the original selected-row space, so
        # slice them using the union-aligned row positions.  The helper also
        # removes sparse columns that become empty after row alignment.
        inflation_obj.slice_null_rows(final_valid_rows)
        exog_infl_values = inflation_obj.values
        if hasattr(exog_infl_values, 'toarray'):
            exog_infl_values = exog_infl_values.toarray()
        model.exog_infl = np.asarray(exog_infl_values, dtype=float)
        if model.exog_infl.ndim == 1:
            model.exog_infl = model.exog_infl[:, None]

        model.exog_infl_names = list(inflation_obj.column_names)
        model.exog_infl_term_names = list(inflation_obj.term_names)
        model.exog_infl_term_to_indices = inflation_obj.var_2_col_indices
        model.k_inflate = model.exog_infl.shape[1]
        model.nobs = len(model.endog)
        model.valid_obs_rows = final_valid_rows
        model.null_rows_info_dict['EXOG_INFL'] = inflation_null_rows

        if sum_to_n and model.weights is not None:
            model.weights *= model.nobs / model.weights.sum()

        model.param_names = model.get_param_names()
        return model

    def _get_inflation_param_names(self):
        """Return prefixed parameter names for the inflation equation."""
        if self.exog_infl_names is not None:
            return [f'inflate_{name}' for name in self.exog_infl_names]
        if (self.k_inflate == 1
                and np.allclose(self.exog_infl[:, 0], 1.0)):
            return ['inflate_const']
        return [f'inflate_x{i}' for i in range(self.k_inflate)]

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
        return np.column_stack((
            self.exog * d_count_eta[:, None],
            self.exog_infl * d_inflation_eta[:, None],
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
        return np.column_stack((
            self.exog * d_count_eta[:, None],
            self.exog_infl * d_inflation_eta[:, None],
            d_log_alpha,
        ))

    def loglike_obs(self, params, *args, **kwargs):
        """Return one unweighted mixture log likelihood per observation."""
        return self._loglike_obs(params)


class Gamma(CountModel):
    """Gamma regression for strictly positive continuous outcomes.

    The conditional mean is ``mu = exp(exog @ beta)`` and the variance is
    ``alpha * mu ** 2``.  The final estimated parameter is ``log_alpha``, so
    the implied dispersion and Gamma shape are always positive:
    ``alpha = exp(log_alpha)`` and ``shape = 1 / alpha``.

    This model inherits the count-model estimation interface for convenience,
    although the Gamma response is continuous rather than a count.
    """

    def __init__(self, endog, exog, weights=None):
        """Initialize a Gamma regression and validate its positive support."""
        endog = np.asarray(endog, dtype=float)
        if endog.ndim != 1:
            raise ValueError("Gamma outcomes must be one-dimensional")
        if np.any(~np.isfinite(endog)) or np.any(endog <= 0.0):
            raise ValueError("Gamma outcomes must be finite and strictly positive")
        super().__init__(endog, exog, weights=weights)
        self._log_endog = np.log(self.endog)

    def get_param_names(self):
        """Return coefficient names followed by ``log_alpha``."""
        return self._get_regression_param_names() + ['log_alpha']

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
        return np.column_stack((self.exog * d_eta[:, None], d_log_alpha))

    def loglike_obs(self, params, *args, **kwargs):
        """Return one unweighted Gamma log-density value per observation."""
        return self._loglike_obs(params)


class GeneralizedPoisson(CountModel):
    """Generalized-Poisson regression with raw dispersion ``alpha``.

    ``p=1`` gives GP-1 with variance ``mu * (1 + alpha) ** 2``;
    ``p=2`` gives GP-2 with variance ``mu * (1 + alpha * mu) ** 2``.
    Positive alpha permits overdispersion, valid negative alpha permits
    underdispersion, and ``alpha=0`` recovers Poisson.
    """

    def __init__(self, endog, exog, weights=None, p=1):
        """Initialize the model and select its variance parameterization.

        Args:
            endog: Response observations.
            exog: Regression design matrix.
            weights: Optional observation likelihood weights.
            p: Positive parameterization index.  Common choices are ``1`` for
                GP-1 and ``2`` for GP-2.
        """
        super().__init__(endog, exog, weights=weights)
        if not np.isscalar(p) or not np.isfinite(p) or p <= 0:
            raise ValueError("p must be a finite positive scalar")
        self.p = p
        self.parameterization = p - 1.0

    def get_param_names(self):
        """Return coefficient names followed by the raw ``alpha`` parameter."""
        return self._get_regression_param_names() + ['alpha']

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
        return np.column_stack((self.exog * d_eta[:, None], d_alpha))

    def loglike_obs(self, params, *args, **kwargs):
        """Return unweighted generalized-Poisson likelihood contributions."""
        return self._loglike_obs(params)


class NegativeBinomial1(CountModel):
    """NB-1 log-link regression with variance ``mu * (1 + alpha)``.

    The conditional mean is ``mu = exp(X beta)`` and
    ``alpha = exp(log_alpha)``.  Unlike NB-2, the NB-1 overdispersion term is
    linear in the mean.  Parameter order is ``beta`` followed by
    ``log_alpha``.
    """

    def get_param_names(self):
        """Return coefficient names followed by ``log_alpha``."""
        return self._get_regression_param_names() + ['log_alpha']

    def _loglike_obs(self, params):
        """Compute unweighted NB-1 log-likelihood contributions."""
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
        """Compute NB-1 derivatives with respect to ``eta`` and ``log_alpha``."""
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
        return np.column_stack((self.exog * d_eta[:, None], d_log_alpha))

    def loglike_obs(self, params, *args, **kwargs):
        """Return one unweighted NB-1 log-likelihood value per observation."""
        return self._loglike_obs(params)


class NegativeBinomial2(CountModel):
    """NB-2 log-link regression with variance ``mu + alpha * mu ** 2``.

    The conditional mean is ``mu = exp(X beta)`` and
    ``alpha = exp(log_alpha)``.  Parameter order is ``beta`` followed by
    ``log_alpha``; the corresponding negative-binomial size is ``1 / alpha``.
    """

    def get_param_names(self):
        """Return coefficient names followed by ``log_alpha``."""
        return self._get_regression_param_names() + ['log_alpha']

    def _loglike_obs(self, params):
        """Compute unweighted NB-2 log-likelihood contributions."""
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
        """Compute NB-2 derivatives with respect to ``eta`` and ``log_alpha``."""
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
        return np.column_stack((self.exog * d_eta[:, None], d_log_alpha))

    def loglike_obs(self, params, *args, **kwargs):
        """Return one unweighted NB-2 log-likelihood value per observation."""
        return self._loglike_obs(params)


if __name__ == '__main__':
    np.random.seed(0)
    n = 405
    from kanly.api import GLM
    import pandas as pd
    x = np.exp(np.random.randn(n))
    v = .05
    y = (np.exp(.2) * x ** .8) * np.exp(v * np.random.randn(n)) * np.exp(-v**2/2)
    X = np.vstack([np.ones(n), np.log(x)]).T
    w = np.exp(np.random.randn(n))

    poisson = Poisson.build_model_from_formula('y ~ np.log(x) $ w', dict(x=x,y=y,w=w))
    fit = poisson.fit([.1] * X.shape[1], cov_type='sandwich')
    print(fit.summary_df)

    poisson = GeneralizedPoisson.build_model_from_formula('y ~ np.log(x) $ w', dict(x=x,y=y,w=w))
    fit = poisson.fit([.1] * 3, cov_type='bootstrap', cov_kwds={'n_samples': 250})
    print(fit.summary_df)

    #print(GLM(y, X, var_weights=w, family='poisson', cov_type='hc1'))
