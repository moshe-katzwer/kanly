"""Response-scale marginal effects for fitted distributional models.

The public entry point is
:meth:`kanly.distributional_models.results.DistributionalModelResults.get_marginal_effects`.
One-part effects differentiate the selected prediction with respect to the
main design.  Two-part effects report separate rows for the main and zero
process designs, using the model's existing coefficient names (for example,
``x`` and ``hurdle_x``).

Average effects and evaluation points use the fitted model's normalized
importance weights.  Standard errors use the delta method with the full
parameter covariance, so covariance between the two equations is retained.
"""

from __future__ import absolute_import, print_function

import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.sparse import isspmatrix
from scipy.special import expit
from scipy.stats import norm

from kanly import __version__
from kanly.dill_object import DillObject


if TYPE_CHECKING:
    from kanly.distributional_models.results import DistributionalModelResults


_AT_OPTIONS = ('overall', 'mean', 'median', 'all')
_EFFECT_TYPES = ('dydx', 'eydx', 'eyex', 'dyex')
_DUMMY_METHODS = ('secant', 'tangent')
_LP_DERIVATIVE_STEP = np.finfo(float).eps ** (1.0 / 5.0)
_PARAMETER_DERIVATIVE_STEP = np.cbrt(np.finfo(float).eps)
_POISSON_LIMIT_LOG_ALPHA = float(np.log(1e-8))


class DistributionalMarginalEffects(DillObject):
    """Marginal-effect estimates, inference, and formatted summaries."""

    eff_label = {
        'dydx': 'dy/dx',
        'eydx': 'ey/dx',
        'eyex': 'ey/ex',
        'dyex': 'dy/ex',
    }

    def __init__(
            self, margeff, margeff_cov, effect_names, equations, at,
            which, effect_type, dummy, dummy_method, dummies, fit,
            test_level):
        self.margeff = np.asarray(margeff, dtype=float)
        self.margeff_cov = (
            None
            if margeff_cov is None
            else np.asarray(margeff_cov, dtype=float)
        )
        self.effect_names = list(effect_names)
        self.exog_names = self.effect_names
        self.equations = list(equations)
        self.at = at
        self.which = which
        self.effect_type = effect_type
        self.dummy = bool(dummy)
        self.dummy_method = dummy_method
        self.dummies = dict(dummies)
        self.fit = fit
        self.endog_name = fit.endog_name
        self.test_level = float(test_level)
        self.is_weighted = bool(fit.model.weights is not None)
        self.has_cov = self.margeff_cov is not None

        if self.has_cov:
            variances = np.diag(self.margeff_cov)
            tolerance = np.finfo(float).eps * max(
                1.0, float(np.max(np.abs(variances), initial=0.0))
            )
            variances = np.where(
                (variances < 0.0) & (variances >= -tolerance),
                0.0,
                variances,
            )
            with np.errstate(invalid='ignore'):
                self.margeff_se = np.sqrt(variances)
            with np.errstate(divide='ignore', invalid='ignore'):
                self.margeff_zvalues = self.margeff / self.margeff_se
            self.margeff_pvalues = 2.0 * norm.sf(
                np.abs(self.margeff_zvalues)
            )
        else:
            self.margeff_se = None
            self.margeff_zvalues = None
            self.margeff_pvalues = None

        self.date = datetime.datetime.today().strftime('%b %d, %Y')
        self.timestamp = datetime.datetime.today().strftime('%H:%M:%S')

    def covariance_df(self):
        """Return the named marginal-effect covariance matrix, if present."""
        if not self.has_cov:
            return None
        return pd.DataFrame(
            self.margeff_cov,
            index=self.effect_names,
            columns=self.effect_names,
        )

    def summary_df(self, test_level=None):
        """Return observation effects or a marginal-effect inference table."""
        if self.at == 'all':
            return pd.DataFrame(self.margeff, columns=self.effect_names)

        if test_level is None:
            test_level = self.test_level
        test_level = _validate_test_level(test_level)
        result = pd.DataFrame(
            {self.eff_label[self.effect_type]: self.margeff},
            index=self.effect_names,
        )
        result.index.name = 'variable'
        result.insert(0, 'equation', self.equations)
        if self.has_cov:
            result['std err'] = self.margeff_se
            result['z'] = self.margeff_zvalues
            result['p>|z|'] = self.margeff_pvalues
            critical_value = -norm.ppf(test_level / 2.0)
            result[f'[{test_level / 2.0:.3f}, '] = (
                self.margeff - critical_value * self.margeff_se
            )
            result[f'{1.0 - test_level / 2.0:.3f}]'] = (
                self.margeff + critical_value * self.margeff_se
            )
        return result

    def summary(self, test_level=None):
        """Return a formatted marginal-effects summary."""
        if test_level is None:
            test_level = self.test_level
        summary_df = self.summary_df(test_level=test_level)
        table = summary_df.to_string()
        width = max(
            len(line) for line in table.splitlines()
        ) if table else 1
        bar = '─' * width
        double_bar = '═' * width
        weighted = 'importance-weighted' if self.is_weighted else 'uniform'
        title = 'Distributional Model Marginal Effects'
        if self.fit.specification_name is not None:
            title += f'\n{self.fit.specification_name}'
        version = f'[kanly v={__version__}]'.rjust(width)
        return (
            f'{double_bar}\n{title}\n{bar}\n'
            f'Dep. Var.: {self.endog_name}\n'
            f'Prediction: {self.which}\n'
            f'Method: {self.effect_type}\n'
            f'At: {self.at}\n'
            f'Averaging: {weighted}\n'
            f'Date: {self.date}\nTime: {self.timestamp}\n'
            f'{double_bar}\n{table}\n{bar}\n{version}'
        )

    def __str__(self):
        return self.summary()

    def __repr__(self):
        return self.summary()


def _validate_test_level(test_level):
    """Return a valid two-sided significance level."""
    if (not np.isscalar(test_level) or not np.isfinite(test_level)
            or test_level <= 0.0 or test_level >= 1.0):
        raise ValueError('test_level must be a finite scalar between 0 and 1')
    return float(test_level)


def _column(values, column):
    """Return one dense design column without densifying other columns."""
    if isspmatrix(values):
        return np.asarray(values.getcol(column).toarray()).reshape(-1)
    return np.asarray(values[:, column], dtype=float).reshape(-1)


def _is_constant_column(values, column):
    """Return whether a dense or sparse design column is constant."""
    column_values = _column(values, column)
    if len(column_values) == 0:
        return True
    return bool(np.allclose(
        column_values, column_values[0], rtol=1e-10, atol=1e-12
    ))


def _is_dummy_column(values, column):
    """Return whether a nonconstant design column contains only zero and one."""
    if _is_constant_column(values, column):
        return False
    if isspmatrix(values):
        values = values.tocsc(copy=False)
        start, stop = values.indptr[column:column + 2]
        stored = values.data[start:stop]
        return bool(np.isin(stored, (0.0, 1.0)).all())
    return bool(np.isin(np.asarray(values[:, column]), (0.0, 1.0)).all())


def _normalized_weights(fit):
    """Return normalized fitted importance weights."""
    weights = fit.model.weights
    if weights is None:
        return np.full(fit.nobs, 1.0 / fit.nobs, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError('model importance weights must have a positive sum')
    return weights / total


def _weighted_column_mean(values, weights):
    """Return design-column means under normalized importance weights."""
    means = values.T @ weights
    return np.asarray(means, dtype=float).reshape(-1)


def _weighted_median(values, weights):
    """Return a midpoint weighted median of finite one-dimensional values."""
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    active = weights > 0.0
    values = values[active]
    weights = weights[active]
    order = np.argsort(values, kind='mergesort')
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    half = 0.5 * cumulative[-1]
    location = int(np.searchsorted(cumulative, half, side='left'))
    if (location + 1 < len(values)
            and np.isclose(cumulative[location], half)):
        return 0.5 * (values[location] + values[location + 1])
    return float(values[location])


def _weighted_column_median(values, weights):
    """Return weighted medians without densifying the full design matrix."""
    return np.asarray([
        _weighted_median(_column(values, column), weights)
        for column in range(values.shape[1])
    ])


def _prediction_options(fit):
    """Return prediction targets supported by the fitted model type."""
    if fit.is_hurdle:
        return (
            'mean', 'positive_mean', 'underlying_mean',
            'linear_predictor', 'zero_probability',
            'positive_probability',
        )
    if fit.is_zero_inflated:
        return (
            'mean', 'count_mean', 'inflation_probability',
            'zero_probability', 'positive_probability',
        )
    return ('mean', 'linear_predictor')


def _active_blocks(fit, which):
    """Return design equations that can affect the requested prediction."""
    if fit.is_hurdle:
        if which == 'mean':
            return ('main', 'zero')
        if which in (
                'positive_mean', 'underlying_mean', 'linear_predictor'):
            return ('main',)
        return ('zero',)
    if fit.is_zero_inflated:
        if which in ('mean', 'zero_probability', 'positive_probability'):
            return ('main', 'zero')
        if which == 'count_mean':
            return ('main',)
        return ('zero',)
    return ('main',)


def _parameter_layout(fit):
    """Return main and zero coefficient slices in the combined parameters."""
    model = fit.model
    k_main = model.exog.shape[1]
    if fit.is_hurdle:
        zero_start = model.k_positive
        return k_main, zero_start, model.k_hurdle
    if fit.is_zero_inflated:
        return k_main, k_main, model.k_inflate
    return k_main, None, 0


def _effect_specs(fit, which):
    """Build named, nonconstant effect columns for active equations."""
    model = fit.model
    k_main, zero_start, k_zero = _parameter_layout(fit)
    active = _active_blocks(fit, which)
    specs = []
    if 'main' in active:
        equation = (
            'positive' if fit.is_hurdle
            else ('count' if fit.is_zero_inflated else 'mean')
        )
        for column in range(k_main):
            if _is_constant_column(model.exog, column):
                continue
            specs.append({
                'block': 'main',
                'equation': equation,
                'column': column,
                'param_index': column,
                'name': fit.param_names[column],
                'dummy': _is_dummy_column(model.exog, column),
            })
    if 'zero' in active:
        equation = 'hurdle' if fit.is_hurdle else 'inflate'
        for column in range(k_zero):
            if _is_constant_column(model.exog_infl, column):
                continue
            parameter = zero_start + column
            specs.append({
                'block': 'zero',
                'equation': equation,
                'column': column,
                'param_index': parameter,
                'name': fit.param_names[parameter],
                'dummy': _is_dummy_column(model.exog_infl, column),
            })
    return specs


def _evaluation_context(fit, at, specs):
    """Build linear-predictor evaluation designs and aggregation weights."""
    model = fit.model
    weights = _normalized_weights(fit)
    if at in ('overall', 'all'):
        main = model.exog
        zero = getattr(model, 'exog_infl', None)
    elif at == 'mean':
        main = _weighted_column_mean(model.exog, weights)[None, :]
        zero = getattr(model, 'exog_infl', None)
        if zero is not None:
            zero = _weighted_column_mean(zero, weights)[None, :]
    else:
        main = _weighted_column_median(model.exog, weights)[None, :]
        zero = getattr(model, 'exog_infl', None)
        if zero is not None:
            zero = _weighted_column_median(zero, weights)[None, :]

    effect_values = []
    for spec in specs:
        design = main if spec['block'] == 'main' else zero
        effect_values.append(_column(design, spec['column']))
    return {
        'at': at,
        'main': main,
        'zero': zero,
        'weights': weights,
        'effect_values': effect_values,
    }


def _hurdle_positive_means(fit, params, eta):
    """Return conditional-positive and underlying means from ``eta``."""
    model = fit.model
    k_main = model.exog.shape[1]
    positive_params = params[:model.k_positive]
    model_name = model.__class__.__name__

    if model_name == 'GaussianHurdle':
        positive_mean = eta
        underlying_mean = eta
    elif model_name == 'LognormalHurdle':
        log_scale = positive_params[k_main]
        with np.errstate(over='ignore', invalid='ignore'):
            scale = np.exp(log_scale)
            positive_mean = np.exp(eta + 0.5 * scale)
        underlying_mean = positive_mean
    elif model_name == 'NegativeBinomialPHurdle':
        log_alpha = positive_params[k_main]
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            mean = np.exp(eta)
            if log_alpha <= _POISSON_LIMIT_LOG_ALPHA:
                log_zero_probability = -mean
            else:
                alpha = np.exp(log_alpha)
                if model.p == 1:
                    log_zero_probability = (
                        -mean * np.log1p(alpha) / alpha
                    )
                else:
                    log_zero_probability = (
                        -np.log1p(alpha * mean) / alpha
                    )
            positive_mean = mean / (-np.expm1(log_zero_probability))
        underlying_mean = mean
    else:
        link = model.positive_link
        if link is None:
            raise NotImplementedError(
                'Marginal effects do not know how to evaluate the positive '
                f'mean for {model_name}'
            )
        positive_mean = link.inverse_link(eta)
        if link.__class__.__name__ == 'ZeroTruncatedPoissonLink':
            with np.errstate(over='ignore', invalid='ignore'):
                underlying_mean = np.exp(eta)
        else:
            underlying_mean = positive_mean
    return positive_mean, underlying_mean


def _prediction_from_linear_predictors(
        fit, params, eta_main, eta_zero, which):
    """Evaluate a supported prediction from component linear predictors."""
    params = np.asarray(params, dtype=float).reshape(-1)
    if fit.is_hurdle:
        positive_mean, underlying_mean = _hurdle_positive_means(
            fit, params, eta_main
        )
        zero_probability, positive_probability = (
            fit.model._hurdle_probabilities_from_eta(eta_zero)
        )
        predictions = {
            'mean': positive_probability * positive_mean,
            'positive_mean': positive_mean,
            'underlying_mean': underlying_mean,
            'linear_predictor': eta_main,
            'zero_probability': zero_probability,
            'positive_probability': positive_probability,
        }
    elif fit.is_zero_inflated:
        with np.errstate(over='ignore', invalid='ignore'):
            count_mean = np.exp(eta_main)
        inflation_probability = expit(eta_zero)
        count_zero_probability = fit.model._count_zero_probability(
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
    else:
        with np.errstate(over='ignore', invalid='ignore'):
            mean = np.exp(eta_main)
        predictions = {
            'mean': mean,
            'linear_predictor': eta_main,
        }
    return np.asarray(predictions[which], dtype=float).reshape(-1)


def _linear_predictors(fit, params, context):
    """Return main and zero linear predictors at an evaluation design."""
    params = np.asarray(params, dtype=float).reshape(-1)
    k_main, zero_start, k_zero = _parameter_layout(fit)
    eta_main = np.asarray(
        context['main'] @ params[:k_main], dtype=float
    ).reshape(-1)
    eta_zero = None
    if k_zero:
        eta_zero = np.asarray(
            context['zero'] @ params[zero_start:zero_start + k_zero],
            dtype=float,
        ).reshape(-1)
    return eta_main, eta_zero


def _linear_predictor_derivative(
        fit, params, eta_main, eta_zero, which, block):
    """Differentiate a prediction with a stable five-point stencil."""
    eta = eta_main if block == 'main' else eta_zero
    step = _LP_DERIVATIVE_STEP * np.maximum(1.0, np.abs(eta))

    def evaluate(offset):
        if block == 'main':
            return _prediction_from_linear_predictors(
                fit, params, eta_main + offset, eta_zero, which
            )
        return _prediction_from_linear_predictors(
            fit, params, eta_main, eta_zero + offset, which
        )

    return (
        evaluate(-2.0 * step)
        - 8.0 * evaluate(-step)
        + 8.0 * evaluate(step)
        - evaluate(2.0 * step)
    ) / (12.0 * step)


def _discrete_effect(
        fit, params, eta_main, eta_zero, which, spec, values):
    """Return a rowwise zero-to-one change for one indicator column."""
    coefficient = params[spec['param_index']]
    if spec['block'] == 'main':
        eta0 = eta_main - values * coefficient
        eta1 = eta0 + coefficient
        prediction0 = _prediction_from_linear_predictors(
            fit, params, eta0, eta_zero, which
        )
        prediction1 = _prediction_from_linear_predictors(
            fit, params, eta1, eta_zero, which
        )
    else:
        eta0 = eta_zero - values * coefficient
        eta1 = eta0 + coefficient
        prediction0 = _prediction_from_linear_predictors(
            fit, params, eta_main, eta0, which
        )
        prediction1 = _prediction_from_linear_predictors(
            fit, params, eta_main, eta1, which
        )
    return prediction1 - prediction0, prediction0


def _transform_effect(raw_effect, prediction, values, effect_type):
    """Transform dy/dx into a requested elasticity or semi-elasticity."""
    with np.errstate(divide='ignore', invalid='ignore'):
        if effect_type == 'dydx':
            return raw_effect
        if effect_type == 'eydx':
            return raw_effect / prediction
        if effect_type == 'dyex':
            return raw_effect * values
        return raw_effect * values / prediction


def _aggregate_effects(values, context):
    """Aggregate row effects at the requested evaluation setting."""
    at = context['at']
    if at == 'all':
        return values
    if at in ('mean', 'median'):
        return values[0]
    weights = context['weights']
    active = weights > 0.0
    return weights[active] @ values[active]


def _compute_effects(
        fit, params, specs, context, which, effect_type,
        dummy, dummy_method):
    """Compute an effect vector or observation-by-effect matrix."""
    params = np.asarray(params, dtype=float).reshape(-1)
    eta_main, eta_zero = _linear_predictors(fit, params, context)
    prediction = _prediction_from_linear_predictors(
        fit, params, eta_main, eta_zero, which
    )
    n_evaluations = len(prediction)
    effects = np.empty((n_evaluations, len(specs)), dtype=float)

    active_blocks = {spec['block'] for spec in specs}
    slopes = {
        block: _linear_predictor_derivative(
            fit, params, eta_main, eta_zero, which, block
        )
        for block in active_blocks
    }
    use_secants = (
        dummy and dummy_method == 'secant'
        and effect_type in ('dydx', 'eydx')
    )
    for effect, (spec, values) in enumerate(zip(
            specs, context['effect_values'])):
        if use_secants and spec['dummy']:
            raw_effect, discrete_baseline = _discrete_effect(
                fit, params, eta_main, eta_zero, which, spec, values
            )
            denominator = (
                discrete_baseline if effect_type == 'eydx' else prediction
            )
        else:
            raw_effect = (
                slopes[spec['block']] * params[spec['param_index']]
            )
            denominator = prediction
        effects[:, effect] = _transform_effect(
            raw_effect, denominator, values, effect_type
        )
    return _aggregate_effects(effects, context)


def _parameter_jacobian(function, params, baseline):
    """Numerically differentiate a vector effect with respect to parameters."""
    params = np.asarray(params, dtype=float).reshape(-1)
    baseline = np.asarray(baseline, dtype=float).reshape(-1)
    jacobian = np.empty((len(baseline), len(params)), dtype=float)
    for parameter in range(len(params)):
        step = _PARAMETER_DERIVATIVE_STEP * max(
            1.0, abs(params[parameter])
        )
        upper_params = params.copy()
        lower_params = params.copy()
        upper_params[parameter] += step
        lower_params[parameter] -= step
        upper = np.asarray(function(upper_params), dtype=float).reshape(-1)
        lower = np.asarray(function(lower_params), dtype=float).reshape(-1)
        if np.all(np.isfinite(upper)) and np.all(np.isfinite(lower)):
            jacobian[:, parameter] = (upper - lower) / (2.0 * step)
        elif np.all(np.isfinite(upper)) and np.all(np.isfinite(baseline)):
            jacobian[:, parameter] = (upper - baseline) / step
        elif np.all(np.isfinite(lower)) and np.all(np.isfinite(baseline)):
            jacobian[:, parameter] = (baseline - lower) / step
        else:
            raise ValueError(
                'Could not compute a finite marginal-effect Jacobian for '
                f'parameter {parameter}'
            )
    return jacobian


def _get_marginal_effects(
        fit: "DistributionalModelResults", at='overall', which='mean',
        effect_type='dydx', dummy=True, dummy_method='secant',
        test_level=.05):
    """Compute weighted response-scale effects and delta-method inference.

    Parameters
    ----------
    fit : DistributionalModelResults
        Fitted one-part, zero-inflated, or hurdle model result.
    at : {'overall', 'mean', 'median', 'all'}, optional
        ``overall`` returns importance-weighted average effects. ``mean`` and
        ``median`` evaluate at importance-weighted design summaries. ``all``
        returns observation effects without inference.
    which : str, optional
        Prediction target. ``mean`` always means the unconditional response
        mean. Other accepted targets match the fitted model's ``predict``
        method.
    effect_type : {'dydx', 'eydx', 'eyex', 'dyex'}, optional
        Marginal effect, response semi-elasticity, elasticity, or regressor
        semi-elasticity, respectively.
    dummy : bool, optional
        Whether nonconstant 0/1 columns use discrete effects for ``dx``
        effect types.
    dummy_method : {'secant', 'tangent'}, optional
        Use a zero-to-one change or a continuous derivative for dummy ``dx``
        effects.
    test_level : float, optional
        Two-sided significance level for normal confidence intervals.

    Returns
    -------
    DistributionalMarginalEffects
        Named effects and, when available, delta-method inference.
    """
    at = str(at).lower()
    which = str(which).lower()
    effect_type = str(effect_type).lower()
    dummy_method = str(dummy_method).lower()
    test_level = _validate_test_level(test_level)
    if at not in _AT_OPTIONS:
        raise ValueError(f'at must be one of: {", ".join(_AT_OPTIONS)}')
    if which not in _prediction_options(fit):
        choices = ', '.join(_prediction_options(fit))
        raise ValueError(f'which must be one of: {choices}')
    if effect_type not in _EFFECT_TYPES:
        choices = ', '.join(_EFFECT_TYPES)
        raise ValueError(f'effect_type must be one of: {choices}')
    if dummy_method not in _DUMMY_METHODS:
        choices = ', '.join(_DUMMY_METHODS)
        raise ValueError(f'dummy_method must be one of: {choices}')
    if not isinstance(dummy, (bool, np.bool_)):
        raise TypeError('dummy must be boolean')

    specs = _effect_specs(fit, which)
    if at == 'median' and dummy and any(
            spec['dummy'] for spec in specs):
        raise ValueError("at='median' is not supported with dummy detection")
    context = _evaluation_context(fit, at, specs)
    params = np.asarray(fit.params, dtype=float).reshape(-1)

    def compute(evaluation_params):
        return _compute_effects(
            fit, evaluation_params, specs, context, which, effect_type,
            dummy, dummy_method,
        )

    effects = compute(params)
    covariance = None
    if at != 'all' and fit.did_compute_var_covar():
        jacobian = _parameter_jacobian(compute, params, effects)
        parameter_covariance = np.asarray(fit.cov_params(), dtype=float)
        covariance = jacobian @ parameter_covariance @ jacobian.T
        covariance = (covariance + covariance.T) / 2.0

    return DistributionalMarginalEffects(
        effects,
        covariance,
        effect_names=[spec['name'] for spec in specs],
        equations=[spec['equation'] for spec in specs],
        at=at,
        which=which,
        effect_type=effect_type,
        dummy=dummy,
        dummy_method=dummy_method,
        dummies={spec['name']: spec['dummy'] for spec in specs},
        fit=fit,
        test_level=test_level,
    )


__all__ = [
    'DistributionalMarginalEffects',
    '_get_marginal_effects',
]
