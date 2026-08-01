"""Two-part hurdle models estimated through separate GLM components.

The zero hurdle is a Bernoulli GLM with a logit link for
``P(Y = 0 | exog_infl)``.  Conditional on crossing that hurdle, the response
is modeled by a second GLM over the strictly positive observations.  Because
the likelihood separates, the two GLMs are fitted independently and their
coefficient vectors and covariance matrices are combined afterwards.

Parameter order follows the count-model convention used by the zero-inflated
classes: positive-response coefficients first, followed by zero-process
coefficients prefixed with ``hurdle_``.
"""

from __future__ import absolute_import, print_function

from abc import abstractmethod
from collections import Counter
import time
import warnings

import numpy as np
from scipy.special import expit

from kanly.bootstrap.bootstrap import (
    DEFAULT_BB_ALPHA,
    DEFAULT_BB_SEED,
    DEFAULT_BOOTSTRAP_N_SAMPLES,
    get_bayesian_bootstrap_weights,
    get_bootstrap_weights2,
)
from kanly.distributional_models.results import DistributionalModelResults
from kanly.distributional_models.two_part import TwoPartModel
from kanly.regression.generalized_linear_models.families import (
    Bernoulli,
    Gamma as GammaFamily,
    ZeroTruncatedPoisson,
    _get_family_and_link,
)
from kanly.regression.generalized_linear_models.links import Log, Logit
from kanly.regression.generalized_linear_models.model import (
    SparseGeneralizedLinearModel,
)


# Hurdle fits use the shared distributional-model result implementation.
HurdleModelResults = DistributionalModelResults


class HurdleModel(TwoPartModel):
    """Base class for separable zero-hurdle and positive-response GLMs.

    ``exog`` controls the positive conditional response.  ``exog_infl``
    controls ``P(Y=0)`` through a Bernoulli/logit GLM and defaults to a single
    constant. Observation ``weights`` multiply the component estimating
    equations as importance weights; the positive component receives the
    subset associated with positive responses.

    Subclasses specify the positive GLM family and link.  The inherited
    formula builder accepts ``exog_infl`` as a Patsy-style right-hand-side
    formula and rejects instruments through the distributional-model parser.
    """

    _requires_both_outcome_parts = True
    is_quasi_likelihood = False

    def __init__(self, *args, **kwargs):
        """Initialize aligned hurdle data and validate both response parts."""
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
            if self.weights[self.is_zero].sum() <= 0.0:
                raise ValueError(
                    'Zero outcomes must have positive total weight'
                )
            if self.weights[self.is_positive].sum() <= 0.0:
                raise ValueError(
                    'Positive outcomes must have positive total weight'
                )

        self._positive_scale = None
        self.positive_model_fit = None
        self.hurdle_model_fit = None

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

    def _get_inflation_param_names(self):
        """Return hurdle-prefixed names for zero-logit coefficients."""
        if self.exog_infl_names is not None:
            return [f'hurdle_{name}' for name in self.exog_infl_names]
        if (self.k_inflate == 1
                and np.allclose(self.exog_infl[:, 0], 1.0)):
            return ['hurdle_const']
        return [f'hurdle_x{i}' for i in range(self.k_inflate)]

    def get_param_names(self):
        """Return positive coefficients followed by hurdle coefficients."""
        return (
            self._get_regression_param_names()
            + self._get_inflation_param_names()
        )

    def get_start_params(self):
        """Return family-aware positive and empirical hurdle starts."""
        positive_endog = self.endog[self.is_positive]
        positive_weights = (
            None if self.weights is None else self.weights[self.is_positive]
        )
        positive_intercept = self.positive_family.get_starting_intercept(
            positive_endog,
            var_weights=positive_weights,
            link=self.positive_link,
        )
        positive_start = self._constant_predictor_start(
            self.exog[self.is_positive], positive_intercept
        )

        if self.weights is None:
            zero_probability = float(np.mean(self.is_zero))
        else:
            zero_probability = float(
                np.dot(self.weights, self.is_zero.astype(float))
                / np.sum(self.weights)
            )
        zero_probability = float(np.clip(zero_probability, 1e-4, 1 - 1e-4))
        hurdle_intercept = (
            np.log(zero_probability) - np.log1p(-zero_probability)
        )
        hurdle_start = self._constant_predictor_start(
            self.exog_infl, hurdle_intercept
        )
        return np.concatenate((positive_start, hurdle_start))

    @property
    def k_positive(self):
        """Return the number of positive-response coefficients."""
        return self.exog.shape[1]

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

        For a zero response this is the Bernoulli log probability of zero.
        For a positive response it is the Bernoulli log probability of
        crossing the hurdle plus the positive-family log likelihood.  As with
        other distributional models, estimation weights are applied only by
        :meth:`loglike`, not to this observation-level return value.
        """
        del args, kwargs
        positive_params, hurdle_params = self._split_params(params)
        positive_scale = self._resolve_positive_scale(positive_scale)

        hurdle_eta = self.exog_infl @ hurdle_params
        hurdle_loglike = np.where(
            self.is_zero,
            -np.logaddexp(0.0, -hurdle_eta),
            -np.logaddexp(0.0, hurdle_eta),
        )

        positive_eta = self.exog[self.is_positive] @ positive_params
        positive_mu = self.positive_link.inverse_link(positive_eta)
        positive_theta = self.positive_family.b_deriv_inv(positive_mu)
        positive_endog = self.endog[self.is_positive]
        positive_loglike = self.positive_family.log_likelihood_obs(
            positive_endog,
            positive_theta,
            scale=positive_scale,
            var_weights=1.0,
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
        """Return unweighted scores from both separable GLM components."""
        del args, kwargs
        positive_params, hurdle_params = self._split_params(params)
        positive_scale = self._resolve_positive_scale(positive_scale)

        hurdle_eta = self.exog_infl @ hurdle_params
        zero_probability = expit(hurdle_eta)
        hurdle_factor = self.is_zero.astype(float) - zero_probability

        score_obs = np.zeros(
            (self.nobs, self.k_positive + self.k_hurdle), dtype=float
        )
        score_obs[:, self.k_positive:] = (
            self.exog_infl * hurdle_factor[:, None]
        )

        positive_eta = self.exog[self.is_positive] @ positive_params
        positive_mu = self.positive_link.inverse_link(positive_eta)
        variance = self.positive_family.variance(positive_mu)
        link_derivative = self.positive_link.deriv(positive_mu)
        positive_factor = (
            self.endog[self.is_positive] - positive_mu
        ) / (positive_scale * variance * link_derivative)
        score_obs[self.is_positive, :self.k_positive] = (
            self.exog[self.is_positive] * positive_factor[:, None]
        )
        return score_obs

    def score(self, params, positive_scale=None, *args, **kwargs):
        """Return the likelihood-weighted combined score vector."""
        score_obs = self.score_obs(
            params, positive_scale=positive_scale, *args, **kwargs
        )
        if self.is_weighted:
            score_obs *= self.weights[:, None]
        return score_obs.sum(axis=0)

    @staticmethod
    def _first_column_is_constant(exog):
        """Return whether a design matrix starts with a constant column."""
        exog = np.asarray(exog)
        return bool(
            exog.ndim == 2
            and exog.shape[1] > 0
            and np.allclose(exog[:, 0], exog[0, 0])
        )

    @staticmethod
    def _normalize_cov_type(cov_type):
        """Map distributional covariance terminology to GLM terminology."""
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
            self, start_params=None, debug=False, cov_type='SANDWICH',
            cov_kwds=None, positive_fit_kwargs=None,
            hurdle_fit_kwargs=None, **glm_fit_kwargs):
        """Fit the positive and zero-hurdle GLMs separately and merge them.

        Args:
            start_params: Optional combined starting vector in positive-then-
                hurdle order.
            debug: Whether to show GLM fitting diagnostics.
            cov_type: ``'NONROBUST'`` for block-diagonal model information,
                ``'SANDWICH'``/``'HC1'`` for a full observation-score robust
                covariance, ``'COMPONENT_HC1'`` for the historical block-
                diagonal component estimator, or ``'BOOTSTRAP'``.
            cov_kwds: Hurdle-level covariance and bootstrap options.
            positive_fit_kwargs: Overrides passed only to the positive GLM.
            hurdle_fit_kwargs: Overrides passed only to the Bernoulli GLM.
            **glm_fit_kwargs: Additional options shared by both GLM fits.

        Returns:
            :class:`DistributionalModelResults` containing the two component
            fits and their combined parameters and covariance.
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
                    f'{label} GLM fit keywords cannot override: {names}'
                )

        positive_start = positive_fit_kwargs.pop('start_params', None)
        hurdle_start = hurdle_fit_kwargs.pop('start_params', None)
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

        common_kwargs = dict(glm_fit_kwargs)
        common_kwargs.update({
            'debug': debug,
            'cov_type': component_cov_type,
            # Hurdle-level covariance options are consumed below. Component
            # GLMs receive only options valid for their native estimator.
            'cov_kwds': {},
        })
        positive_kwargs = common_kwargs.copy()
        positive_kwargs.update(positive_fit_kwargs)
        positive_kwargs['cov_kwds'] = {}
        positive_first_column_constant = self._first_column_is_constant(
            self.exog[self.is_positive]
        )
        positive_kwargs['fit_intercept'] = positive_first_column_constant
        hurdle_kwargs = common_kwargs.copy()
        hurdle_kwargs.update(hurdle_fit_kwargs)
        hurdle_kwargs['cov_kwds'] = {}
        hurdle_first_column_constant = self._first_column_is_constant(
            self.exog_infl
        )
        hurdle_kwargs['fit_intercept'] = hurdle_first_column_constant

        def fit_components(component_weights, positive_x0, hurdle_x0):
            positive_weights = (
                None if component_weights is None
                else component_weights[self.is_positive]
            )
            positive_result = SparseGeneralizedLinearModel.GLM(
                self.endog[self.is_positive],
                self.exog[self.is_positive],
                family=self.positive_family,
                link=self.positive_link,
                var_weights=positive_weights,
                exog_names=self._get_regression_param_names(),
                endog_name=self.endog_name,
                var_weights_name=self.weights_name,
                start_params=positive_x0,
                first_column_constant=positive_first_column_constant,
                **positive_kwargs,
            )
            hurdle_result = SparseGeneralizedLinearModel.GLM(
                self.is_zero.astype(float),
                self.exog_infl,
                family=Bernoulli(),
                link=Logit(),
                var_weights=component_weights,
                exog_names=self._get_inflation_param_names(),
                endog_name=(
                    None
                    if self.endog_name is None
                    else f'{self.endog_name}_is_zero'
                ),
                var_weights_name=self.weights_name,
                start_params=hurdle_x0,
                first_column_constant=hurdle_first_column_constant,
                **hurdle_kwargs,
            )
            return positive_result, hurdle_result

        positive_fit, hurdle_fit = fit_components(
            self.weights, positive_start, hurdle_start
        )

        self._positive_scale = float(positive_fit.scale)
        self.positive_model_fit = positive_fit
        self.hurdle_model_fit = hurdle_fit
        component_covariance = self._block_diagonal(
            self._component_covariance(positive_fit),
            self._component_covariance(hurdle_fit),
        )
        params = np.concatenate((
            np.asarray(positive_fit.params, dtype=float),
            np.asarray(hurdle_fit.params, dtype=float),
        ))
        positive_scale = float(positive_fit.scale)
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
            positive_fit.converged and hurdle_fit.converged
        )
        first_order_valid = bool(
            np.all(np.isfinite(score_at_params))
            and normalized_score <= 1e-5
        )
        converged = optimizer_converged and first_order_valid
        inference_issues = []
        if not optimizer_converged:
            inference_issues.append(
                'At least one component GLM did not converge.'
            )
        if not first_order_valid:
            inference_issues.append(
                'The combined hurdle score did not satisfy the scaled '
                'first-order condition.'
            )

        bread = None
        meat = None
        information = None
        bootstrapped_params = None
        bootstrap_string = None
        bootstrap_cov_elapsed = 0.0
        covariance = component_covariance
        if display_cov_type in {'SANDWICH', 'HC1'}:
            bread = component_covariance
            if bread is not None:
                score_obs = self.score_obs(
                    params, positive_scale=positive_scale
                )
                if self.weights is not None:
                    score_obs = score_obs * self.weights[:, None]
                meat = score_obs.T @ score_obs
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
                        combined_weights, point_positive, point_hurdle
                    )
                except Exception as error:
                    failure_reasons[
                        f'{type(error).__name__}: {error}'
                    ] += 1
                    continue
                draw = np.concatenate((
                    np.asarray(draw_positive.params, dtype=float),
                    np.asarray(draw_hurdle.params, dtype=float),
                ))
                draw_valid = bool(
                    draw_positive.converged
                    and draw_hurdle.converged
                    and np.all(np.isfinite(draw))
                )
                if draw_valid:
                    draw_score_obs = self.score_obs(
                        draw, positive_scale=float(draw_positive.scale)
                    )
                    draw_score = (
                        draw_score_obs * combined_weights[:, None]
                    ).sum(axis=0)
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
                elif (draw_positive.converged and draw_hurdle.converged
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
            'Both component GLMs converged.'
            if converged
            else 'Hurdle fit diagnostics failed.'
        )
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
            method='SEPARATE GLMS',
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
            iterations=(positive_fit.num_iter, hurdle_fit.num_iter),
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
        """Predict hurdle probabilities and conditional or overall means.

        Args:
            params: Combined parameter vector.
            exog: Optional positive-response design matrix.
            exog_infl: Optional zero-hurdle design matrix.
            data: Optional new data evaluated through both stored formulas.
            index: Optional positional row selector applied to ``data``.
            debug: Whether formula construction prints diagnostics.
            which: One of ``'mean'``, ``'positive_mean'``,
                ``'zero_probability'``, or ``'positive_probability'``.
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
        exog = self.exog if exog is None else np.asarray(exog, dtype=float)
        exog_infl = (
            self.exog_infl
            if exog_infl is None
            else np.asarray(exog_infl, dtype=float)
        )
        if exog.ndim == 1:
            exog = exog[None, :]
        if exog_infl.ndim == 1:
            exog_infl = exog_infl[None, :]
        if exog.shape[0] != exog_infl.shape[0]:
            raise ValueError(
                'exog and exog_infl must contain the same number of rows'
            )
        if exog.shape[1] != self.k_positive:
            raise ValueError('exog has the wrong number of columns')
        if exog_infl.shape[1] != self.k_hurdle:
            raise ValueError('exog_infl has the wrong number of columns')

        positive_mean = self.positive_link.inverse_link(
            exog @ positive_params
        )
        zero_probability = expit(exog_infl @ hurdle_params)
        predictions = {
            'mean': (1.0 - zero_probability) * positive_mean,
            'positive_mean': positive_mean,
            'zero_probability': zero_probability,
            'positive_probability': 1.0 - zero_probability,
        }
        which = str(which).lower()
        if which not in predictions:
            choices = ', '.join(sorted(predictions))
            raise ValueError(f'which must be one of: {choices}')
        return predictions[which]


class PoissonHurdle(HurdleModel):
    """Hurdle model with Bernoulli/logit zeros and positive Poisson counts.

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


class GammaHurdle(HurdleModel):
    """Hurdle model with Bernoulli/logit zeros and positive Gamma responses.

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


__all__ = [
    'HurdleModel',
    'HurdleModelResults',
    'PoissonHurdle',
    'GammaHurdle',
]
