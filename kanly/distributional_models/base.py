"""Shared base class for likelihood-based distributional models."""

from __future__ import absolute_import, print_function

from abc import ABC, abstractmethod
from collections import Counter
import time
import warnings

import numpy as np

from kanly.optimize.bfgs_bounded_quasi_newton import bfgs_pqn
from kanly.bootstrap.bootstrap import (
    DEFAULT_BB_ALPHA, DEFAULT_BB_SEED, DEFAULT_BOOTSTRAP_N_SAMPLES,
    get_bayesian_bootstrap_weights, get_bootstrap_weights2,
)
from kanly.formula.data_getter import SparseDataGetter
from kanly.formula.keys import (
    ENDOG_KEY, EXOG_KEY, FORMULA_DESIGN_INFO_KEY, HAS_IMPLICIT_CONSTANT_KEY,
    HAS_INTERCEPT_KEY, INDEX_KEY, NULL_ROWS_INFO_DICT_KEY, TIME_ELAPSED_KEY,
    VALID_OBS_ROWS_KEY, WEIGHTS_KEY,
)
from kanly.distributional_models.results import DistributionalModelResults
from kanly.utils.util import dict_2_dataframe


_MAX_INFORMATION_CONDITION = 1e12
_SCALED_SCORE_TOLERANCE = 1e-5


class DistributionalModel(ABC):
    """Abstract base class for likelihood-based response models.

    Subclasses provide observation log-likelihoods, parameter names, and,
    preferably, analytical scores. ``weights`` are importance weights on the
    objective: they multiply observation log-likelihoods and scores during
    aggregation but do not alter values returned by :meth:`loglike_obs` or
    :meth:`score_obs`.

    Args:
        endog: One-dimensional response array of length ``nobs``.
        exog: Two-dimensional design matrix with ``nobs`` rows.
        weights: Optional non-negative importance-likelihood weights of
            length ``nobs`` and positive total mass.
    """

    def __init__(
            self, endog, exog, weights=None, endog_name=None,
            exog_names=None, weights_name=None, formula_design_info=None,
            formula=None, from_formula=False, exog_term_names=None,
            exog_term_to_indices=None, has_intercept=False,
            has_implicit_constant=False, valid_obs_rows=None,
            null_rows_info_dict=None, index=None, model_elapsed=0.0):
        """Initialize model data, names, formula metadata, and parameters.

        Args:
            endog: One-dimensional response data.
            exog: Two-dimensional regression design matrix.
            weights: Optional non-negative likelihood weights.
            endog_name: Optional response name.
            exog_names: Optional coefficient names matching the columns of
                ``exog``. Generated names are used when omitted.
            weights_name: Optional name of the likelihood-weight variable.
            formula_design_info: Formula engine metadata, when applicable.
            formula: Original formula string, when applicable.
            from_formula: Whether the model was constructed from a formula.
            exog_term_names: Formula terms represented by ``exog``.
            exog_term_to_indices: Mapping from formula terms to exog columns.
            has_intercept: Whether ``exog`` has an explicit intercept.
            has_implicit_constant: Whether its span contains a constant.
            valid_obs_rows: Retained row positions after missing-data handling.
            null_rows_info_dict: Missing-row diagnostics by formula block.
            index: Original row selector supplied to formula construction.
            model_elapsed: Formula/model construction time in seconds.
        """
        endog = np.asarray(endog, dtype=float)
        if endog.ndim == 2 and endog.shape[1] == 1:
            endog = endog.reshape(-1)
        if endog.ndim != 1:
            raise ValueError("endog must be one-dimensional")
        self._validate_endog(endog)

        exog = np.asarray(exog, dtype=float)
        if exog.ndim == 1:
            exog = exog[:, None]
        if exog.ndim != 2:
            raise ValueError("exog must be two-dimensional")
        if exog.shape[0] != len(endog):
            raise ValueError("endog and exog must contain the same number of rows")
        if len(endog) == 0:
            raise ValueError("endog and exog must contain at least one row")
        if exog.shape[1] == 0:
            raise ValueError("exog must contain at least one column")
        if np.any(~np.isfinite(exog)):
            raise ValueError("exog must contain only finite values")

        if weights is not None:
            weights = np.asarray(weights, dtype=float).reshape(-1)
            if len(weights) != len(endog):
                raise ValueError("weights must have one value per observation")
            if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
                raise ValueError("weights must be finite and non-negative")
            if weights.sum() <= 0.0:
                raise ValueError("weights must have a positive total")

        if exog_names is not None:
            exog_names = [str(name) for name in exog_names]
            if len(exog_names) != exog.shape[1]:
                raise ValueError("exog_names must match the columns of exog")

        self.endog = endog
        self.exog = exog
        self.weights = weights
        self.is_weighted = self.weights is not None
        self.nobs = len(self.endog)

        self.formula_design_info = formula_design_info
        self.formula = formula
        self.from_formula = bool(from_formula)
        self.endog_name = None if endog_name is None else str(endog_name)
        self.exog_names = exog_names
        self.weights_name = (
            None if weights_name is None else str(weights_name)
        )
        self.exog_term_names = (
            None if exog_term_names is None else list(exog_term_names)
        )
        self.exog_term_to_indices = exog_term_to_indices
        self.has_intercept = bool(has_intercept)
        self.has_implicit_constant = bool(has_implicit_constant)
        self.valid_obs_rows = (
            np.arange(self.nobs)
            if valid_obs_rows is None
            else np.asarray(valid_obs_rows, dtype=int).copy()
        )
        self.null_rows_info_dict = (
            {} if null_rows_info_dict is None else null_rows_info_dict.copy()
        )
        self.index = index
        self.model_elapsed = float(model_elapsed)
        self.param_names = self.get_param_names()

    def _validate_endog(self, endog):
        """Validate response values common to every distribution.

        Distribution-specific subclasses extend this hook with their support
        restrictions.  It is called during construction, before any starting
        values, likelihood evaluations, or optimization can occur.
        """
        if np.any(~np.isfinite(endog)):
            raise ValueError("endog must contain only finite values")

    def _inference_issues(self, params):
        """Return model-specific reasons asymptotic inference is unreliable."""
        del params
        return []

    def _is_quasi_likelihood(self):
        """Return whether fitted objective lacks a normalized likelihood."""
        return False

    @classmethod
    def _get_formula_constructor_kwargs(
            cls, formula, data, index=None, debug: bool = False,
            check_constant_cols=False, fail_on_missing=False,
            cache_intermediate=True, sum_to_n=False,
            test_formula_on_dummy=True, drop_1_for_FE=True):
        """Build aligned formula arrays and return constructor keywords.

        The ``$`` formula extension supplies optional likelihood weights.
        Instrumental-variable and absorbed-effect formulas are not supported.
        Missing rows are aligned and removed by ``SparseDataGetter``.

        Returns:
            Dictionary containing model data, names, and formula metadata that
            can be passed directly to a distributional-model constructor.
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
            raise ValueError(
                "Distributional models require exactly one outcome column"
            )
        endog = endog.reshape(-1)

        if weights_obj is None:
            weights = None
        else:
            weights = weights_obj.values
            if hasattr(weights, 'toarray'):
                weights = weights.toarray()
            weights = np.asarray(weights).reshape(-1)

        formula_design_info = result[FORMULA_DESIGN_INFO_KEY]
        return {
            'endog': endog,
            'exog': exog,
            'weights': weights,
            'endog_name': endog_obj.column_names[0],
            'exog_names': list(exog_obj.column_names),
            'weights_name': (
                None if weights_obj is None else weights_obj.column_names[0]
            ),
            'formula_design_info': formula_design_info,
            'formula': formula_design_info.formula,
            'from_formula': True,
            'exog_term_names': exog_obj.term_names,
            'exog_term_to_indices': exog_obj.var_2_col_indices,
            'has_intercept': result[HAS_INTERCEPT_KEY],
            'has_implicit_constant': result[HAS_IMPLICIT_CONSTANT_KEY],
            'valid_obs_rows': result[VALID_OBS_ROWS_KEY],
            'null_rows_info_dict': result[NULL_ROWS_INFO_DICT_KEY],
            'index': result[INDEX_KEY],
            'model_elapsed': result[TIME_ELAPSED_KEY],
        }

    @classmethod
    def build_model_from_formula(
            cls, formula, data, index=None, debug: bool = False,
            check_constant_cols=False, fail_on_missing=False,
            cache_intermediate=True, sum_to_n=False,
            test_formula_on_dummy=True, drop_1_for_FE=True, **model_kwargs):
        """Build an unfitted model with constructor-owned formula metadata.

        Set ``debug=True`` to show formula-parser output followed by a compact
        summary of the retained response, regression design, names, weights,
        and construction time.
        """
        constructor_kwargs = cls._get_formula_constructor_kwargs(
            formula=formula,
            data=data,
            index=index,
            debug=debug,
            check_constant_cols=check_constant_cols,
            fail_on_missing=fail_on_missing,
            cache_intermediate=cache_intermediate,
            # Normalize after all formula rows have been aligned.
            sum_to_n=False,
            test_formula_on_dummy=test_formula_on_dummy,
            drop_1_for_FE=drop_1_for_FE,
        )
        if sum_to_n and constructor_kwargs['weights'] is not None:
            weights = constructor_kwargs['weights']
            weights_sum = weights.sum()
            if weights_sum <= 0.0:
                raise ValueError(
                    "weights must have a positive sum when sum_to_n=True"
                )
            constructor_kwargs['weights'] = (
                weights * len(constructor_kwargs['endog']) / weights_sum
            )
        duplicate_kwargs = set(constructor_kwargs).intersection(model_kwargs)
        if duplicate_kwargs:
            duplicates = ', '.join(sorted(duplicate_kwargs))
            raise TypeError(
                f'Formula construction supplies these arguments: {duplicates}'
            )

        model = cls(**constructor_kwargs, **model_kwargs)
        if debug:
            model._print_formula_debug_summary()
        return model

    @staticmethod
    def _format_debug_values(values, max_items=12):
        """Format a short vector or name sequence for debug output."""
        values = list(values)
        displayed = values[:max_items]
        suffix = (
            f' ... ({len(values)} total)'
            if len(values) > max_items else ''
        )
        return f'{displayed}{suffix}'

    def _print_formula_debug_summary(self):
        """Print aligned formula-model metadata after construction."""
        print('\n' + '=' * 50)
        print('DISTRIBUTIONAL FORMULA MODEL')
        print('-' * 50)
        print(f'* Model: {self.__class__.__name__}')
        print(f'* Formula: {self.formula}')
        print(f'* Outcome: {self.endog_name or "y"}')
        print(f'* Retained observations: {self.nobs}')
        dropped_rows = set()
        for rows in self.null_rows_info_dict.values():
            if rows is not None:
                dropped_rows.update(int(row) for row in rows)
        print(f'* Rows removed for missing data: {len(dropped_rows)}')
        print(f'* Main exog shape: {self.exog.shape}')
        print(
            '* Main exog names: '
            f'{self._format_debug_values(self._get_regression_param_names())}'
        )
        if hasattr(self, 'exog_infl'):
            print(f'* Zero-process exog shape: {self.exog_infl.shape}')
            inflation_names = (
                self.exog_infl_names
                if self.exog_infl_names is not None
                else (
                    self.exog_infl_term_names
                    if self.exog_infl_term_names is not None
                    else [f'x{i}' for i in range(self.exog_infl.shape[1])]
                )
            )
            print(
                '* Zero-process exog names: '
                f'{self._format_debug_values(inflation_names)}'
            )
            print(
                '* Zero-process formula: '
                f'{self.exog_infl_formula or "constant only"}'
            )
        if self.is_weighted:
            print(
                f'* Importance weights: {self.weights_name or "provided"}; '
                f'sum={np.sum(self.weights):.6g}'
            )
        else:
            print('* Importance weights: none')
        print(f'* Construction time: {self.model_elapsed:.3f} s')
        print('...Distributional formula model complete!\n')

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

    def _response_moments(self):
        """Return likelihood-weighted response mean and population variance."""
        values = np.asarray(self.endog, dtype=float)
        if self.weights is None:
            valid = np.isfinite(values)
            if not np.any(valid):
                raise ValueError(
                    'Cannot initialize parameters without a finite outcome'
                )
            values = values[valid]
            mean = float(np.mean(values))
            variance = float(np.mean((values - mean) ** 2))
            return mean, variance

        weights = np.asarray(self.weights, dtype=float)
        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
        if not np.any(valid):
            raise ValueError(
                'Cannot initialize parameters without positive total weight'
            )
        values = values[valid]
        weights = weights[valid]
        total_weight = float(weights.sum())
        mean = float(np.dot(weights, values) / total_weight)
        variance = float(
            np.dot(weights, (values - mean) ** 2) / total_weight
        )
        return mean, variance

    @staticmethod
    def _constant_predictor_start(exog, linear_predictor):
        """Represent a constant linear predictor when a constant exists."""
        exog = np.asarray(exog, dtype=float)
        params = np.zeros(exog.shape[1], dtype=float)
        if exog.shape[1] == 0:
            return params

        is_constant = np.all(
            np.isclose(exog, exog[0:1, :], rtol=1e-10, atol=1e-12),
            axis=0,
        )
        usable = np.flatnonzero(
            is_constant & (np.abs(exog[0, :]) > np.finfo(float).eps)
        )
        if len(usable):
            column = int(usable[0])
            params[column] = linear_predictor / exog[0, column]
        return params

    def _mean_regression_start(self, mean=None, exog=None):
        """Return regression starts representing a constant response mean."""
        if mean is None:
            mean = self._response_moments()[0]
        mean = float(np.clip(mean, 1e-6, 1e12))
        exog = self.exog if exog is None else exog
        return self._constant_predictor_start(exog, np.log(mean))

    @staticmethod
    def _log_dispersion_start(alpha, lower=0.05, upper=10.0):
        """Return a finite log-dispersion start inside conservative bounds."""
        alpha = float(np.clip(alpha, lower, upper))
        return float(np.log(alpha))

    def get_start_params(self):
        """Return sensible default parameters for this model instance.

        The base implementation initializes a detected constant coefficient
        to the log of the likelihood-weighted response mean and leaves all
        other regression coefficients at zero. Distribution-specific
        subclasses append their own dispersion or mixture starts.
        """
        return self._mean_regression_start()

    @property
    def default_start_params(self):
        """Return a fresh copy of the instance's automatic starting values."""
        return self.get_start_params().copy()

    def _get_formula_prediction_exog(
            self, data, index=None, debug=False):
        """Build an aligned prediction design from stored formula metadata."""
        if self.formula_design_info is None:
            raise ValueError(
                'Formula prediction requires a model built from a formula'
            )
        data_frame = dict_2_dataframe(data)
        if index is not None:
            data_frame = data_frame.iloc[index]
        design = self.formula_design_info.get_design_data_exog(
            data_frame, fail_on_missing=True, debug=debug
        )[EXOG_KEY]
        values = design.values
        if hasattr(values, 'toarray'):
            values = values.toarray()
        values = np.asarray(values, dtype=float)
        names = list(design.column_names)
        if self.exog_names is not None and names != list(self.exog_names):
            raise ValueError(
                'Prediction formula produced columns that do not match the '
                f'fitted design: expected {self.exog_names}, got {names}'
            )
        return values

    def predict(
            self, params, exog=None, exog_infl=None, which='mean',
            data=None, index=None, debug=False):
        """Predict log-link conditional means from a numeric design matrix.

        Args:
            params: Full model parameter vector.  Distribution parameters
                following the regression coefficients are ignored here.
            exog: Optional numeric prediction design; defaults to fitted exog.
            exog_infl: Unused for one-part models; accepted for a common
                results interface.
            which: ``'mean'`` or ``'linear_predictor'``.
            data: Optional new data evaluated through stored formula metadata.
            index: Optional positional row selector applied to ``data``.
            debug: Whether formula construction prints diagnostics.
        """
        del exog_infl
        if data is not None:
            if exog is not None:
                raise ValueError('Supply either data or exog, not both')
            exog = self._get_formula_prediction_exog(
                data, index=index, debug=debug
            )
        params = np.asarray(params, dtype=float).reshape(-1)
        exog = self.exog if exog is None else np.asarray(exog, dtype=float)
        if exog.ndim == 1:
            exog = exog[None, :]
        if exog.ndim != 2 or exog.shape[1] != self.exog.shape[1]:
            raise ValueError('exog has the wrong number of columns')
        if len(params) != len(self.param_names):
            raise ValueError('params has the wrong length')

        linear_predictor = exog @ params[:self.exog.shape[1]]
        with np.errstate(over='ignore', invalid='ignore'):
            mean = np.exp(linear_predictor)
        predictions = {
            'mean': mean,
            'linear_predictor': linear_predictor,
        }
        which = str(which).lower()
        if which not in predictions:
            raise ValueError("which must be 'mean' or 'linear_predictor'")
        return predictions[which]

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

    def _fit_internal(
            self, start_params, weights=None, debug: bool = False):
        """Optimize using explicit likelihood weights and no model mutation."""
        if weights is not None:
            weights = np.asarray(weights, dtype=float).reshape(-1)
            if len(weights) != self.nobs:
                raise ValueError("weights must have one value per observation")
            if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
                raise ValueError("weights must be finite and non-negative")
            if weights.sum() <= 0.0:
                raise ValueError("weights must have a positive total")

        def objective(params):
            values = self.loglike_obs(params)
            return values.sum() if weights is None else np.dot(weights, values)

        def gradient(params):
            values = self.score_obs(params)
            if weights is not None:
                values = values * weights[:, None]
            return values.sum(axis=0)

        return bfgs_pqn(
            objective, start_params, maximize=True, debug=debug,
            gradient_callable=gradient,
        )

    def _bayesian_bootstrap(
            self, params, cov_kwds, debug: bool = False):
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
                "DistributionalModel bootstrap covariance currently supports "
                "only "
                "method='BAYESIAN'"
            )

        min_success_rate = cov_kwds.get('min_success_rate', 0.8)
        if (not np.isscalar(min_success_rate)
                or not np.isfinite(min_success_rate)
                or not 0.0 <= min_success_rate <= 1.0):
            raise ValueError(
                "min_success_rate must be a finite value between 0 and 1"
            )

        sample_weights, _ = get_bayesian_bootstrap_weights(
            self.nobs, n_samples=n_samples, seed=seed, alpha=alpha
        )
        if debug:
            from tqdm import tqdm

            print('\n' + '=' * 50)
            print('BAYESIAN BOOTSTRAP')
            print('-' * 50)
            print(f'* Model: {self.__class__.__name__}')
            print(f'* Requested repetitions: {n_samples}')
            print(f'* Seed: {seed}')
            print(f'* Dirichlet alpha: {alpha:.6g}')
            print(f'* Minimum success rate: {min_success_rate:.3f}')
            print(
                '* Original importance weights: '
                f'{"multiplied into each draw" if self.is_weighted else "none"}'
            )
            print('* Per-draw BFGS output is suppressed; progress is shown.')
            sample_weights = tqdm(
                sample_weights,
                total=n_samples,
                desc=f'{self.__class__.__name__} bootstrap',
            )
        params = np.asarray(params, dtype=float)
        parameter_draws = []
        n_failed = 0
        failure_reasons = Counter()

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
            except Exception as error:
                n_failed += 1
                reason = f'{type(error).__name__}: {error}'
                failure_reasons[reason] += 1
                continue

            draw = np.asarray(fit.x, dtype=float)
            draw_valid = bool(fit.converged and np.all(np.isfinite(draw)))
            draw_failure = None
            if draw_valid:
                draw_loglike = float(np.dot(
                    combined_weights, self.loglike_obs(draw)
                ))
                draw_score_obs = self.score_obs(draw)
                draw_score = (
                    draw_score_obs * combined_weights[:, None]
                ).sum(axis=0)
                draw_normalized_score = (
                    float(np.max(np.abs(draw_score)))
                    / max(float(np.sum(combined_weights)), 1.0)
                )
                draw_valid = bool(
                    np.isfinite(draw_loglike)
                    and np.all(np.isfinite(draw_score))
                    and draw_normalized_score <= _SCALED_SCORE_TOLERANCE
                )
                if not draw_valid:
                    draw_failure = (
                        'Bootstrap first-order validation failed '
                        f'(scaled score={draw_normalized_score:.3e})'
                    )

            if draw_valid:
                parameter_draws.append(draw.copy())
            else:
                n_failed += 1
                if draw_failure is None:
                    draw_failure = (
                        'Non-convergence: '
                        f'{getattr(fit, "message", "unknown")}'
                    )
                failure_reasons[draw_failure] += 1

        parameter_draws = np.asarray(parameter_draws, dtype=float)
        minimum_successes = max(
            2, int(np.ceil(float(min_success_rate) * n_samples))
        )
        if len(parameter_draws) < minimum_successes:
            failures = '; '.join(
                f'{count} x {reason}'
                for reason, count in failure_reasons.most_common(3)
            )
            raise RuntimeError(
                f'Only {len(parameter_draws)} of {n_samples} Bayesian '
                f'bootstrap repetitions converged; at least '
                f'{minimum_successes} were required. {failures}'
            )

        use_correction = bool(cov_kwds.get('use_correction', True))
        cov_params = np.atleast_2d(
            np.cov(parameter_draws, rowvar=False, ddof=0)
        )
        if use_correction:
            n_successful = len(parameter_draws)
            cov_params *= n_successful / (n_successful - 1.0)
        cov_params = (cov_params + cov_params.T) / 2.0

        if n_failed:
            retained_message = (
                f'Bayesian bootstrap retained {len(parameter_draws)} of '
                f'{n_samples} repetitions.'
            )
            warnings.warn(retained_message, RuntimeWarning, stacklevel=3)
            if debug:
                print(retained_message)

        bootstrap_kwds = {
            'method': method,
            'n_samples': int(n_samples),
            'n_successful': len(parameter_draws),
            'n_failed': n_failed,
            'seed': seed,
            'alpha': alpha,
            'use_correction': use_correction,
            'min_success_rate': float(min_success_rate),
            'failure_reasons': dict(failure_reasons),
        }
        if debug:
            print('\nBootstrap diagnostics:')
            print(f'* Successful repetitions: {len(parameter_draws)}')
            print(f'* Failed repetitions: {n_failed}')
            print(f'* Covariance correction: {use_correction}')
            print('...Bayesian bootstrap complete!\n')
        return cov_params, parameter_draws, bootstrap_kwds

    def fit(self, start_params=None, debug: bool = False,
            cov_type='SANDWICH',
            cov_kwds=None) -> DistributionalModelResults:
        """Estimate parameters and a likelihood-based covariance matrix.

        When ``start_params`` is omitted, :meth:`get_start_params` supplies
        distribution-specific starting values. Explicit starting values remain
        supported and take precedence.

        For ``cov_type='BOOTSTRAP'``, ``cov_kwds`` may contain ``n_samples``,
        ``seed``, ``alpha``, and ``use_correction``.  Every Bayesian-bootstrap
        draw is multiplied by, rather than substituted for, any original
        observation weights.

        Set ``debug=True`` to print model dimensions and starts, pass verbose
        diagnostics to BFGS-PQN, describe covariance validation, and show
        Bayesian-bootstrap progress when requested.
        """
        cov_type = str(cov_type).upper()
        if cov_type not in {'SANDWICH', 'NONROBUST', 'BOOTSTRAP'}:
            raise ValueError(
                "cov_type must be 'SANDWICH', 'NONROBUST', or 'BOOTSTRAP'"
            )
        cov_kwds = {} if cov_kwds is None else dict(cov_kwds)

        used_default_start = start_params is None
        if used_default_start:
            start_params = self.get_start_params()
        start_params = np.asarray(start_params, dtype=float).reshape(-1)
        if len(start_params) != len(self.param_names):
            raise ValueError(
                f'Expected {len(self.param_names)} starting parameters but '
                f'received {len(start_params)}'
            )
        if np.any(~np.isfinite(start_params)):
            raise ValueError('start_params must contain only finite values')

        if debug:
            print('\n' + '=' * 50)
            print('DISTRIBUTIONAL MODEL FIT')
            print('-' * 50)
            print(f'* Model: {self.__class__.__name__}')
            print(f'* Observations: {self.nobs}')
            print(f'* Main exog shape: {self.exog.shape}')
            if hasattr(self, 'exog_infl'):
                print(f'* Zero-process exog shape: {self.exog_infl.shape}')
                print(
                    '* Outcome split: '
                    f'{np.count_nonzero(self.endog == 0.0)} zero / '
                    f'{np.count_nonzero(self.endog > 0.0)} positive'
                )
            print(f'* Parameters: {len(self.param_names)}')
            print(
                '* Parameter names: '
                f'{self._format_debug_values(self.param_names)}'
            )
            print(f'* Covariance type: {cov_type}')
            print(
                '* Importance weights: '
                f'{"provided" if self.is_weighted else "none"}'
            )
            print(
                '* Starting values '
                f'({"automatic" if used_default_start else "user"}): '
                f'{np.array2string(start_params, precision=6)}'
            )
            print('* Passing debug=True to BFGS-PQN.')

        fit_start = time.perf_counter()
        optimization_result = self._fit_internal(
            start_params, weights=self.weights, debug=debug
        )
        fit_elapsed = time.perf_counter() - fit_start

        params = np.asarray(optimization_result.x, dtype=float)
        llf = float(self.loglike(params))
        score_at_params = np.asarray(self.score(params), dtype=float)
        weight_total = (
            float(self.nobs)
            if self.weights is None
            else float(np.sum(self.weights))
        )
        max_abs_score = (
            float(np.max(np.abs(score_at_params)))
            if len(score_at_params) else 0.0
        )
        normalized_score = max_abs_score / max(weight_total, 1.0)
        optimizer_converged = bool(optimization_result.converged)
        first_order_issues = []
        if not optimizer_converged:
            first_order_issues.append(
                'The numerical optimizer did not report convergence.'
            )
        if not np.isfinite(llf):
            first_order_issues.append(
                'The fitted aggregate log likelihood is not finite.'
            )
        if np.any(~np.isfinite(params)):
            first_order_issues.append(
                'The fitted parameter vector contains non-finite values.'
            )
        if np.any(~np.isfinite(score_at_params)):
            first_order_issues.append(
                'The fitted score contains non-finite values.'
            )
        elif normalized_score > _SCALED_SCORE_TOLERANCE:
            first_order_issues.append(
                'The optimizer stopped without satisfying the scaled '
                f'first-order condition ({normalized_score:.3e} > '
                f'{_SCALED_SCORE_TOLERANCE:.1e}).'
            )

        first_order_valid = not first_order_issues
        converged = optimizer_converged and first_order_valid
        model_inference_issues = list(self._inference_issues(params))
        inference_issues = first_order_issues + model_inference_issues

        if debug:
            print('\nPoint-estimate diagnostics:')
            print(f'* Optimizer converged: {optimizer_converged}')
            print(f'* First-order validation: {first_order_valid}')
            print(f'* Log likelihood: {llf:.8g}')
            print(f'* Scaled maximum score: {normalized_score:.3e}')
            print(f'* Fit elapsed: {fit_elapsed:.3f} s')

        bootstrapped_params = None
        bootstrap_string = None
        information = None
        bread = None
        meat = None
        cov_params = None
        information_rank = None
        information_condition = None
        information_min_eigenvalue = None
        covariance_status = 'not computed'
        inference_valid = False
        covariance_start = time.perf_counter()
        can_compute_inference = converged and not model_inference_issues
        if debug:
            print('\nCovariance phase:')
            print(f'* Requested estimator: {cov_type}')
            print(f'* Fit eligible for inference: {can_compute_inference}')
        if cov_type == 'BOOTSTRAP' and can_compute_inference:
            cov_params, bootstrapped_params, cov_kwds = (
                self._bayesian_bootstrap(
                    params, cov_kwds, debug=debug
                )
            )
            bootstrap_string = (
                f"Did {len(bootstrapped_params)} Bayesian bootstrap "
                f"repetitions, alpha={cov_kwds['alpha']:.3f}."
            )
            if np.all(np.isfinite(cov_params)):
                covariance_eigenvalues = np.linalg.eigvalsh(cov_params)
                covariance_tolerance = (
                    np.finfo(float).eps
                    * max(len(params), 1)
                    * max(float(np.max(np.abs(covariance_eigenvalues))), 1.0)
                )
                if np.min(covariance_eigenvalues) >= -covariance_tolerance:
                    inference_valid = True
                    covariance_status = 'valid bootstrap covariance'
                else:
                    inference_issues.append(
                        'The bootstrap covariance is not positive '
                        'semidefinite.'
                    )
                    cov_params = None
            else:
                inference_issues.append(
                    'The bootstrap covariance contains non-finite values.'
                )
                cov_params = None
        elif cov_type != 'BOOTSTRAP' and can_compute_inference:
            try:
                hess = self.hessian(params)
                information = -hess
            except Exception as error:
                information = None
                inference_issues.append(
                    'Observed-information calculation failed: '
                    f'{type(error).__name__}: {error}'
                )

            if information is not None and np.all(np.isfinite(information)):
                information_rank = int(np.linalg.matrix_rank(information))
                information_condition = float(np.linalg.cond(information))
                information_eigenvalues = np.linalg.eigvalsh(information)
                information_min_eigenvalue = float(
                    np.min(information_eigenvalues)
                )
                eigenvalue_tolerance = (
                    np.finfo(float).eps
                    * max(len(params), 1)
                    * max(float(np.max(np.abs(information_eigenvalues))), 1.0)
                )
                information_valid = (
                    information_rank == len(params)
                    and np.isfinite(information_condition)
                    and information_condition <= _MAX_INFORMATION_CONDITION
                    and information_min_eigenvalue > eigenvalue_tolerance
                )
            else:
                information_valid = False

            if information_valid:
                bread = np.linalg.inv(information)
                bread = (bread + bread.T) / 2.0
            else:
                inference_issues.append(
                    'Observed information is non-finite, rank deficient, '
                    'ill-conditioned, or not positive definite.'
                )

            if bread is not None and cov_type == 'SANDWICH':
                score_obs = self.score_obs(params)
                if self.is_weighted:
                    score_obs = (
                        score_obs * np.asarray(self.weights)[:, None]
                    )
                meat = score_obs.T @ score_obs
                cov_params = bread @ meat @ bread.T
                cov_params = (cov_params + cov_params.T) / 2.0
            elif bread is not None:
                cov_params = bread.copy()

            if cov_params is not None and np.all(np.isfinite(cov_params)):
                covariance_eigenvalues = np.linalg.eigvalsh(cov_params)
                covariance_tolerance = (
                    np.finfo(float).eps
                    * max(len(params), 1)
                    * max(float(np.max(np.abs(covariance_eigenvalues))), 1.0)
                )
                if np.min(covariance_eigenvalues) >= -covariance_tolerance:
                    inference_valid = True
                    covariance_status = 'valid asymptotic covariance'
                else:
                    inference_issues.append(
                        'The estimated covariance is not positive '
                        'semidefinite.'
                    )
                    cov_params = None
            elif cov_params is not None:
                inference_issues.append(
                    'The estimated covariance contains non-finite values.'
                )
                cov_params = None

        if not can_compute_inference:
            covariance_status = 'suppressed because fit diagnostics failed'
        elif cov_params is None:
            covariance_status = 'unavailable because covariance diagnostics failed'
        cov_elapsed = time.perf_counter() - covariance_start

        if debug:
            print('\nFinal fit diagnostics:')
            print(f'* Public convergence: {converged}')
            print(f'* Inference valid: {inference_valid}')
            print(f'* Covariance status: {covariance_status}')
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
            print(f'* Covariance elapsed: {cov_elapsed:.3f} s')
            print('...Distributional model fit complete!\n')

        if inference_issues:
            warnings.warn(
                ' '.join(inference_issues), RuntimeWarning, stacklevel=2
            )
        message = str(optimization_result.message)
        if first_order_issues:
            message += ' Fit validation failed: ' + ' '.join(
                first_order_issues
            )

        return DistributionalModelResults(
            model=self,
            params=params,
            cov_params=cov_params,
            cov_type=cov_type,
            cov_kwds=cov_kwds,
            llf=llf,
            converged=converged,
            optimizer_converged=optimizer_converged,
            first_order_valid=first_order_valid,
            inference_valid=inference_valid,
            inference_issues=inference_issues,
            normalized_score=normalized_score,
            covariance_status=covariance_status,
            information_rank=information_rank,
            information_condition=information_condition,
            information_min_eigenvalue=information_min_eigenvalue,
            message=message,
            method='BFGS-PQN',
            optimization_result=optimization_result,
            information=information,
            bread=bread,
            meat=meat,
            bootstrapped_params=bootstrapped_params,
            bootstrap_string=bootstrap_string,
            fit_elapsed=fit_elapsed,
            cov_elapsed=cov_elapsed,
            iterations=optimization_result.iter,
            score_at_params=score_at_params,
            scale=1.0,
            is_quasi_likelihood=self._is_quasi_likelihood(),
            report_information_criteria=(
                not self._is_quasi_likelihood() and not self.is_weighted
            ),
        )


class _NonnegativeDistributionalModel(DistributionalModel):
    """Shared constructor support for finite non-negative responses."""

    def _validate_endog(self, endog):
        """Require response values in the extended count-model support."""
        if np.any(~np.isfinite(endog)) or np.any(endog < 0.0):
            raise ValueError(
                f'{self.__class__.__name__} outcomes must be finite and '
                'non-negative'
            )

    def _is_quasi_likelihood(self):
        """Treat fractional count outcomes as likelihood-like estimation."""
        return bool(np.any(self.endog != np.floor(self.endog)))


__all__ = ['DistributionalModel']
