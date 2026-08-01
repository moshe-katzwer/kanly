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

import numpy as np
from scipy.special import expit

from kanly.distributional_models.count_models import _ZeroInflatedModel
from kanly.distributional_models.results import DistributionalModelResults
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


class HurdleModel(_ZeroInflatedModel):
    """Base class for separable zero-hurdle and positive-response GLMs.

    ``exog`` controls the positive conditional response.  ``exog_infl``
    controls ``P(Y=0)`` through a Bernoulli/logit GLM and defaults to a single
    constant.  Observation ``weights`` are passed to both component GLMs as
    variance weights; the positive component receives the subset associated
    with positive responses.

    Subclasses specify the positive GLM family and link.  The inherited
    formula builder accepts ``exog_infl`` as a Patsy-style right-hand-side
    formula and rejects instruments through the distributional-model parser.
    """

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
        if cov_type == 'SANDWICH':
            return cov_type, 'HC1'
        if cov_type in {'HC1', 'NONROBUST', 'BOOTSTRAP'}:
            return cov_type, cov_type
        raise ValueError(
            "cov_type must be 'SANDWICH', 'HC1', 'NONROBUST', or "
            "'BOOTSTRAP'"
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
            self, start_params=None, debug=False, cov_type='NONROBUST',
            cov_kwds=None, positive_fit_kwargs=None,
            hurdle_fit_kwargs=None, **glm_fit_kwargs):
        """Fit the positive and zero-hurdle GLMs separately and merge them.

        Args:
            start_params: Optional combined starting vector in positive-then-
                hurdle order.
            debug: Whether to show GLM fitting diagnostics.
            cov_type: ``'NONROBUST'``, ``'SANDWICH'``/``'HC1'``, or
                ``'BOOTSTRAP'``.  Each component computes this covariance and
                the combined covariance is block diagonal.
            cov_kwds: Common covariance keywords for both component GLMs.
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
            'cov_kwds': cov_kwds,
        })
        positive_kwargs = common_kwargs.copy()
        positive_kwargs.update(positive_fit_kwargs)
        positive_kwargs['cov_kwds'] = cov_kwds.copy()
        positive_first_column_constant = self._first_column_is_constant(
            self.exog[self.is_positive]
        )
        positive_kwargs['fit_intercept'] = positive_first_column_constant
        hurdle_kwargs = common_kwargs.copy()
        hurdle_kwargs.update(hurdle_fit_kwargs)
        hurdle_kwargs['cov_kwds'] = cov_kwds.copy()
        hurdle_first_column_constant = self._first_column_is_constant(
            self.exog_infl
        )
        hurdle_kwargs['fit_intercept'] = hurdle_first_column_constant

        positive_weights = (
            None
            if self.weights is None
            else self.weights[self.is_positive]
        )
        positive_fit = SparseGeneralizedLinearModel.GLM(
            self.endog[self.is_positive],
            self.exog[self.is_positive],
            family=self.positive_family,
            link=self.positive_link,
            var_weights=positive_weights,
            exog_names=self._get_regression_param_names(),
            endog_name=self.endog_name,
            var_weights_name=self.weights_name,
            start_params=positive_start,
            first_column_constant=positive_first_column_constant,
            **positive_kwargs,
        )
        hurdle_fit = SparseGeneralizedLinearModel.GLM(
            self.is_zero.astype(float),
            self.exog_infl,
            family=Bernoulli(),
            link=Logit(),
            var_weights=self.weights,
            exog_names=self._get_inflation_param_names(),
            endog_name=(
                None
                if self.endog_name is None
                else f'{self.endog_name}_is_zero'
            ),
            var_weights_name=self.weights_name,
            start_params=hurdle_start,
            first_column_constant=hurdle_first_column_constant,
            **hurdle_kwargs,
        )

        self._positive_scale = float(positive_fit.scale)
        self.positive_model_fit = positive_fit
        self.hurdle_model_fit = hurdle_fit
        covariance = self._block_diagonal(
            self._component_covariance(positive_fit),
            self._component_covariance(hurdle_fit),
        )
        params = np.concatenate((
            np.asarray(positive_fit.params, dtype=float),
            np.asarray(hurdle_fit.params, dtype=float),
        ))
        converged = bool(positive_fit.converged and hurdle_fit.converged)
        message = (
            'Both component GLMs converged.'
            if converged
            else 'At least one component GLM did not converge.'
        )
        positive_scale = float(positive_fit.scale)
        return DistributionalModelResults(
            model=self,
            params=params,
            llf=self.loglike(params, positive_scale=positive_scale),
            converged=converged,
            message=message,
            method='SEPARATE GLMS',
            positive_fit=positive_fit,
            hurdle_fit=hurdle_fit,
            cov_params=covariance,
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
            ),
            iterations=(positive_fit.num_iter, hurdle_fit.num_iter),
            score_at_params=self.score(
                params, positive_scale=positive_scale
            ),
            scale=positive_scale,
            loglike_kwargs={'positive_scale': positive_scale},
        )

    def predict(
            self, params, exog=None, exog_infl=None, which='mean'):
        """Predict hurdle probabilities and conditional or overall means.

        Args:
            params: Combined parameter vector.
            exog: Optional positive-response design matrix.
            exog_infl: Optional zero-hurdle design matrix.
            which: One of ``'mean'``, ``'positive_mean'``,
                ``'zero_probability'``, or ``'positive_probability'``.
        """
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
