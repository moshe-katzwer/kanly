"""One-shot formula and array APIs for distributional models.

The lower-case :func:`distributional_model` entry point parses a formula,
constructs the selected model, and fits it. The upper-case
:func:`DISTRIBUTIONAL_MODEL` entry point performs the same build-and-fit flow
from arrays. Their broad signatures are intentionally uniform: options that
do not apply to the selected model are ignored.
"""

from __future__ import absolute_import, print_function

from dataclasses import dataclass
import difflib
import re

import numpy as np
from scipy.sparse import csc_matrix, isspmatrix

from kanly.distributional_models.base import _as_design_matrix
from kanly.distributional_models.continuous_models import Gamma
from kanly.distributional_models.count_models import (
    GeneralizedPoisson,
    NegativeBinomial1,
    NegativeBinomial2,
    Poisson,
    ZeroInflatedNegativeBinomial,
    ZeroInflatedPoisson,
)
from kanly.distributional_models.hurdle_models import (
    GaussianHurdle,
    GammaHurdle,
    HurdleModel,
    InverseGaussianHurdle,
    LognormalHurdle,
    NegativeBinomialPHurdle,
    PoissonHurdle,
)
from kanly.distributional_models.results import DistributionalModelResults
from kanly.distributional_models.two_part import TwoPartModel
from kanly.utils.linalg_utils import DEFAULT_DENSE_THRESHOLD_MB


@dataclass(frozen=True)
class _ModelAliasGroup:
    """Describe aliases resolving to one model and optional fixed ``p``."""

    canonical_name: str
    model_class: type
    aliases: tuple[str, ...]
    p: float | None = None


# This is the authoritative alias table. Normalization removes capitalization
# and every non-alphanumeric separator, so each listed phrase also covers its
# capitalization, space, underscore, and hyphen variants.
_MODEL_ALIAS_GROUPS = (
    _ModelAliasGroup(
        'Poisson', Poisson, ('poisson', 'pois')
    ),
    _ModelAliasGroup(
        'ZeroInflatedPoisson', ZeroInflatedPoisson,
        ('zero inflated poisson', 'zip'),
    ),
    _ModelAliasGroup(
        'ZeroInflatedNegativeBinomial', ZeroInflatedNegativeBinomial,
        (
            'zero inflated negative binomial', 'zero inflated nb',
            'zinb', 'zinb2',
        ),
    ),
    _ModelAliasGroup(
        'NegativeBinomial1', NegativeBinomial1,
        ('negative binomial 1', 'negative binomial one', 'nb1'),
    ),
    _ModelAliasGroup(
        'NegativeBinomial2', NegativeBinomial2,
        (
            'negative binomial 2', 'negative binomial two',
            'negative binomial', 'nb2', 'nb',
        ),
    ),
    _ModelAliasGroup(
        'GeneralizedPoisson', GeneralizedPoisson,
        ('generalized poisson', 'generalized pois', 'gen poisson', 'gp'),
    ),
    _ModelAliasGroup(
        'GeneralizedPoisson', GeneralizedPoisson,
        ('generalized poisson 1', 'gp1'), p=1,
    ),
    _ModelAliasGroup(
        'GeneralizedPoisson', GeneralizedPoisson,
        ('generalized poisson 2', 'gp2'), p=2,
    ),
    _ModelAliasGroup(
        'PoissonHurdle', PoissonHurdle,
        ('poisson hurdle', 'hurdle poisson'),
    ),
    _ModelAliasGroup(
        'NegativeBinomialPHurdle', NegativeBinomialPHurdle,
        (
            'negative binomial p hurdle', 'negative binomial hurdle',
            'nbp hurdle', 'nb hurdle', 'hurdle negative binomial',
        ),
    ),
    _ModelAliasGroup(
        'NegativeBinomialPHurdle', NegativeBinomialPHurdle,
        ('negative binomial 1 hurdle', 'nb1 hurdle', 'hurdle nb1'), p=1,
    ),
    _ModelAliasGroup(
        'NegativeBinomialPHurdle', NegativeBinomialPHurdle,
        ('negative binomial 2 hurdle', 'nb2 hurdle', 'hurdle nb2'), p=2,
    ),
    _ModelAliasGroup(
        'GammaHurdle', GammaHurdle, ('gamma hurdle', 'hurdle gamma')
    ),
    _ModelAliasGroup(
        'GaussianHurdle', GaussianHurdle,
        (
            'gaussian hurdle', 'normal hurdle', 'hurdle gaussian',
            'hurdle normal', 'gaussian', 'normal',
        ),
    ),
    _ModelAliasGroup(
        'LognormalHurdle', LognormalHurdle,
        (
            'lognormal hurdle', 'log normal hurdle', 'hurdle lognormal',
            'lognormal', 'log normal',
        ),
    ),
    _ModelAliasGroup(
        'InverseGaussianHurdle', InverseGaussianHurdle,
        ('inverse gaussian hurdle', 'inverse gaussian', 'ig hurdle', 'igh'),
    ),
    _ModelAliasGroup(
        'Gamma', Gamma, ('gamma',)
    ),
)


# Public, inspection-friendly lookup rows used by documentation and callers.
# Each tuple is ``(canonical_name, aliases, implied_p)``. ``implied_p=None``
# means the class default applies unless the caller supplies ``p`` explicitly.
DISTRIBUTIONAL_MODEL_ALIASES = tuple(
    (group.canonical_name, group.aliases, group.p)
    for group in _MODEL_ALIAS_GROUPS
)


def _normalize_model_name(model_name):
    """Return a case- and separator-insensitive alphanumeric model key."""
    if not isinstance(model_name, str):
        raise TypeError('model_name must be a string')
    normalized = re.sub(r'[^a-z0-9]+', '', model_name.lower())
    if not normalized:
        raise ValueError('model_name must contain letters or numbers')
    return normalized


def _build_model_lookup():
    """Build the normalized alias lookup and reject internal collisions."""
    lookup = {}
    for group in _MODEL_ALIAS_GROUPS:
        names = (group.canonical_name,) + group.aliases
        for name in names:
            key = _normalize_model_name(name)
            existing = lookup.get(key)
            if existing is not None and existing != group:
                # Repeating a canonical class across p-specific alias groups
                # is intentional; the bare canonical name keeps class defaults.
                if name == group.canonical_name:
                    continue
                raise RuntimeError(
                    f'Distributional model alias collision for {name!r}'
                )
            lookup[key] = group
    return lookup


_MODEL_LOOKUP = _build_model_lookup()


def _resolve_model(model_name, p):
    """Resolve a user name and reconcile any alias-implied ``p`` value."""
    normalized = _normalize_model_name(model_name)
    group = _MODEL_LOOKUP.get(normalized)
    if group is None:
        suggestions = difflib.get_close_matches(
            normalized, tuple(_MODEL_LOOKUP), n=3, cutoff=0.45
        )
        suggestion_names = []
        for key in suggestions:
            name = _MODEL_LOOKUP[key].canonical_name
            if name not in suggestion_names:
                suggestion_names.append(name)
        canonical_names = sorted({
            item.canonical_name for item in _MODEL_ALIAS_GROUPS
        })
        message = (
            f'Unknown distributional model {model_name!r}. Available models: '
            + ', '.join(canonical_names)
        )
        if suggestion_names:
            message += '. Did you mean: ' + ', '.join(suggestion_names)
        raise ValueError(message)

    if group.p is not None and p is not None and float(p) != group.p:
        raise ValueError(
            f'model_name={model_name!r} implies p={group.p:g}, but p={p!r} '
            'was supplied'
        )
    effective_p = group.p if p is None else p
    return group, effective_p


def _model_constructor_options(model_class, p, positive_link, zero_model):
    """Return only constructor options relevant to ``model_class``."""
    options = {}
    if model_class in {GeneralizedPoisson, NegativeBinomialPHurdle}:
        if p is not None:
            options['p'] = p
    if model_class in {GammaHurdle, InverseGaussianHurdle} and (
            positive_link is not None):
        options['positive_link'] = positive_link
    if issubclass(model_class, HurdleModel):
        options['zero_model'] = zero_model
    return options


def _common_hurdle_fit_options(
        tol, max_iter, alpha, l1_ratio, L2_penalty_matrix,
        regularize_to_values, normalize, penalize_scale, use_t,
        test_level, compute_cov, store_convergence_path,
        line_search_fallback, pick_default_start, opt_method,
        prompt_user_for_more_iters):
    """Collect explicitly supplied shared GLM controls for hurdle fits."""
    values = {
        'tol': tol,
        'max_iter': max_iter,
        'alpha': alpha,
        'l1_ratio': l1_ratio,
        'L2_penalty_matrix': L2_penalty_matrix,
        'regularize_to_values': regularize_to_values,
        'normalize': normalize,
        'penalize_scale': penalize_scale,
        'use_t': use_t,
        'test_level': test_level,
        'compute_cov': compute_cov,
        'store_convergence_path': store_convergence_path,
        'line_search_fallback': line_search_fallback,
        'pick_default_start': pick_default_start,
        'opt_method': opt_method,
        'prompt_user_for_more_iters': prompt_user_for_more_iters,
    }
    return {name: value for name, value in values.items() if value is not None}


def _fit_selected_model(
        model, start_params, debug, cov_type, cov_kwds,
        positive_fit_kwargs, hurdle_fit_kwargs, common_hurdle_fit_options):
    """Fit a constructed model while filtering irrelevant wrapper options."""
    if not isinstance(model, HurdleModel):
        return model.fit(
            start_params=start_params,
            debug=debug,
            cov_type=cov_type,
            cov_kwds=cov_kwds,
        )

    positive_fit_kwargs = (
        None if positive_fit_kwargs is None else dict(positive_fit_kwargs)
    )
    hurdle_fit_kwargs = (
        None if hurdle_fit_kwargs is None else dict(hurdle_fit_kwargs)
    )

    if isinstance(model, NegativeBinomialPHurdle):
        # The positive component is exact BFGS MLE rather than a GLM. Shared
        # GLM controls therefore apply only to its binary-equivalent hurdle
        # component.
        hurdle_fit_kwargs = {
            **common_hurdle_fit_options,
            **({} if hurdle_fit_kwargs is None else hurdle_fit_kwargs),
        }
        common_hurdle_fit_options = {}

    return model.fit(
        start_params=start_params,
        debug=debug,
        cov_type=cov_type,
        cov_kwds=cov_kwds,
        positive_fit_kwargs=positive_fit_kwargs,
        hurdle_fit_kwargs=hurdle_fit_kwargs,
        **common_hurdle_fit_options,
    )


def _print_dispatch_debug(entry_point, requested_name, group, effective_p,
                          uses_inflation, exog_infl):
    """Print concise one-shot API dispatch information."""
    print('\n' + '=' * 50)
    print('DISTRIBUTIONAL MODEL API DISPATCH')
    print('-' * 50)
    print(f'* Entry point: {entry_point}')
    print(f'* Requested model name: {requested_name!r}')
    print(f'* Normalized key: {_normalize_model_name(requested_name)}')
    print(f'* Selected class: {group.canonical_name}')
    if effective_p is not None and group.model_class in {
            GeneralizedPoisson, NegativeBinomialPHurdle}:
        print(f'* Parameterization p: {effective_p}')
    if exog_infl is not None and not uses_inflation:
        print('* exog_infl: ignored for this one-part model')
    else:
        print(
            '* exog_infl: '
            + ('used' if exog_infl is not None else 'default constant/none')
        )
    print('...API dispatch complete!\n')


def distributional_model(
        formula, data, model_name='Poisson', exog_infl=None, p=None,
        positive_link=None, start_params=None, debug=False,
        cov_type='SANDWICH', cov_kwds=None, positive_fit_kwargs=None,
        hurdle_fit_kwargs=None, index=None, check_constant_cols=False,
        fail_on_missing=False, cache_intermediate=True, sum_to_n=False,
        test_formula_on_dummy=True, drop_1_for_FE=True,
        dense_threshold_mb=DEFAULT_DENSE_THRESHOLD_MB, tol=None,
        max_iter=None, alpha=None, l1_ratio=None, L2_penalty_matrix=None,
        regularize_to_values=None, normalize=None, penalize_scale=None,
        use_t=None, test_level=None, compute_cov=None,
        store_convergence_path=None, line_search_fallback=None,
        pick_default_start=None, opt_method=None,
        prompt_user_for_more_iters=None,
        zero_model='logit') -> DistributionalModelResults:
    """Build and fit a selected distributional model from a formula.

    Args:
        formula: Patsy-style outcome and main-equation formula. The ``$``
            extension supplies optional importance-likelihood weights.
        data: DataFrame or dict-like formula data.
        model_name: Canonical model name or documented alias. Matching ignores
            capitalization and non-alphanumeric separators.
        exog_infl: Right-hand-side formula for a zero-inflation or hurdle
            equation. ``None`` gives a constant-only zero equation. Ignored by
            one-part models.
        p: Generalized-Poisson or negative-binomial-P parameterization. Ignored
            by other models. Class defaults apply when omitted.
        positive_link: Gamma- or Inverse-Gaussian-hurdle positive-response
            link. Ignored otherwise.
        zero_model: Hurdle zero-process model. ``'logit'`` preserves the
            default Bernoulli/logit model for ``P(Y=0)``; ``'poisson'`` uses
            a statsmodels-compatible right-censored Poisson model. Ignored by
            non-hurdle models.
        start_params: Optional full starting parameter vector.
        debug: Print API dispatch, formula construction, and fit diagnostics.
        cov_type: Distributional covariance estimator.
        cov_kwds: Covariance or Bayesian-bootstrap options.
        positive_fit_kwargs: Positive-component-only hurdle GLM options.
        hurdle_fit_kwargs: Zero-hurdle-component-only GLM options.
        index: Optional formula row selector.
        check_constant_cols: Formula parser constant-column check.
        fail_on_missing: Raise rather than omit formula rows with missing data.
        cache_intermediate: Formula intermediate-cache configuration.
        sum_to_n: Normalize formula likelihood weights to retained sample size.
        test_formula_on_dummy: Prevalidate the formula on dummy data.
        drop_1_for_FE: Drop one categorical level in formula encoding.
        dense_threshold_mb: Dense-memory threshold in MB. Formula designs are
            built sparse first and converted to dense when their dense
            footprint is at or below this threshold.
        tol, max_iter, alpha, l1_ratio, L2_penalty_matrix,
            regularize_to_values, normalize, penalize_scale, use_t,
            test_level, compute_cov, store_convergence_path,
            line_search_fallback, pick_default_start, opt_method,
            prompt_user_for_more_iters: Shared component-GLM controls used by
            Poisson, Gamma, and Inverse Gaussian hurdles. For NB-P hurdles
            they apply only to the zero component. Gaussian and lognormal
            positive components use OLS, so iterative controls apply only to
            their zero component. These controls are ignored by direct
            one-part models.

    Returns:
        A fitted :class:`DistributionalModelResults` object.

    Notes:
        The ``alpha`` argument in this wrapper is a hurdle-component GLM
        regularization control. Distributional dispersion is estimated by the
        selected likelihood and is not supplied through this argument.
    """
    group, effective_p = _resolve_model(model_name, p)
    model_class = group.model_class
    uses_inflation = issubclass(model_class, TwoPartModel)
    if debug:
        _print_dispatch_debug(
            'formula', model_name, group, effective_p,
            uses_inflation, exog_infl,
        )

    builder_options = {
        'index': index,
        'debug': debug,
        'check_constant_cols': check_constant_cols,
        'fail_on_missing': fail_on_missing,
        'cache_intermediate': cache_intermediate,
        'sum_to_n': sum_to_n,
        'test_formula_on_dummy': test_formula_on_dummy,
        'drop_1_for_FE': drop_1_for_FE,
        'dense_threshold_mb': dense_threshold_mb,
    }
    if uses_inflation:
        builder_options['exog_infl'] = exog_infl
    builder_options.update(_model_constructor_options(
        model_class, effective_p, positive_link, zero_model
    ))
    model = model_class.build_model_from_formula(
        formula, data, **builder_options
    )

    common_options = _common_hurdle_fit_options(
        tol, max_iter, alpha, l1_ratio, L2_penalty_matrix,
        regularize_to_values, normalize, penalize_scale, use_t,
        test_level, compute_cov, store_convergence_path,
        line_search_fallback, pick_default_start, opt_method,
        prompt_user_for_more_iters,
    )
    return _fit_selected_model(
        model, start_params, debug, cov_type, cov_kwds,
        positive_fit_kwargs, hurdle_fit_kwargs, common_options,
    )


def DISTRIBUTIONAL_MODEL(
        endog, exog, model_name='Poisson', exog_infl=None, weights=None,
        endog_name=None, exog_names=None, weights_name=None,
        exog_infl_names=None, add_constant=False, has_intercept=False,
        has_implicit_constant=False, p=None, positive_link=None,
        start_params=None, debug=False, cov_type='SANDWICH', cov_kwds=None,
        positive_fit_kwargs=None, hurdle_fit_kwargs=None, index=None,
        tol=None, max_iter=None, alpha=None, l1_ratio=None,
        L2_penalty_matrix=None, regularize_to_values=None, normalize=None,
        penalize_scale=None, use_t=None, test_level=None, compute_cov=None,
        store_convergence_path=None, line_search_fallback=None,
        pick_default_start=None, opt_method=None,
        prompt_user_for_more_iters=None,
        zero_model='logit') -> DistributionalModelResults:
    """Build and fit a selected distributional model from arrays.

    Args:
        endog: One-dimensional response array.
        exog: Main-equation design matrix.
        model_name: Canonical model name or documented alias. Matching ignores
            capitalization and non-alphanumeric separators.
        exog_infl: Numeric zero-equation design. ``None`` creates a constant
            for two-part models and is ignored by one-part models.
        weights: Optional non-negative importance-likelihood weights.
        endog_name: Optional response name.
        exog_names: Optional main-design column names.
        weights_name: Optional likelihood-weight name.
        exog_infl_names: Optional zero-design column names; ignored for
            one-part models.
        add_constant: Prepend a constant to ``exog`` and its names.
        has_intercept: Whether ``exog`` already represents an intercept.
        has_implicit_constant: Whether the span of ``exog`` contains a
            constant without an explicit constant column.
        p: Generalized-Poisson or NB-P-hurdle parameterization; ignored by
            other models.
        positive_link: Gamma- or Inverse-Gaussian-hurdle positive-response
            link; ignored otherwise.
        zero_model: Hurdle zero-process model. ``'logit'`` is the default;
            ``'poisson'`` uses a statsmodels-compatible right-censored
            Poisson model. Ignored by non-hurdle models.
        start_params: Optional full starting parameter vector.
        debug: Print API dispatch and fitting diagnostics.
        cov_type: Distributional covariance estimator.
        cov_kwds: Covariance or Bayesian-bootstrap options.
        positive_fit_kwargs: Positive-component-only hurdle GLM options.
        hurdle_fit_kwargs: Zero-hurdle-component-only GLM options.
        index: Optional row metadata retained on the model.
        tol, max_iter, alpha, l1_ratio, L2_penalty_matrix,
            regularize_to_values, normalize, penalize_scale, use_t,
            test_level, compute_cov, store_convergence_path,
            line_search_fallback, pick_default_start, opt_method,
            prompt_user_for_more_iters: Shared hurdle component controls, as
            described by :func:`distributional_model`; ignored by direct
            one-part models.

    Returns:
        A fitted :class:`DistributionalModelResults` object.
    """
    group, effective_p = _resolve_model(model_name, p)
    model_class = group.model_class
    uses_inflation = issubclass(model_class, TwoPartModel)
    if debug:
        _print_dispatch_debug(
            'array', model_name, group, effective_p,
            uses_inflation, exog_infl,
        )

    if add_constant:
        if not isspmatrix(exog):
            exog = np.asarray(exog, dtype=float)
            if exog.ndim == 1:
                exog = exog[:, None]
        if exog.ndim != 2:
            raise ValueError('exog must be one- or two-dimensional')
        if isspmatrix(exog):
            exog = _as_design_matrix(exog)
            nobs, nvars = exog.shape
            total_nnz = nobs + exog.nnz
            index_dtype = (
                np.int64
                if max(nobs, total_nnz) > np.iinfo(np.int32).max
                else np.int32
            )
            data = np.empty(total_nnz, dtype=float)
            indices = np.empty(total_nnz, dtype=index_dtype)
            indptr = np.empty(nvars + 2, dtype=index_dtype)
            data[:nobs] = 1.0
            data[nobs:] = exog.data
            indices[:nobs] = np.arange(nobs, dtype=index_dtype)
            indices[nobs:] = exog.indices
            indptr[0] = 0
            np.add(exog.indptr, nobs, out=indptr[1:])
            exog = csc_matrix(
                (data, indices, indptr),
                shape=(nobs, nvars + 1),
                copy=False,
            )
        else:
            with_constant = np.empty(
                (exog.shape[0], exog.shape[1] + 1), dtype=float
            )
            with_constant[:, 0] = 1.0
            with_constant[:, 1:] = exog
            exog = with_constant
        if exog_names is not None:
            exog_names = ['Intercept'] + list(exog_names)
        has_intercept = True

    constructor_options = {
        'weights': weights,
        'endog_name': endog_name,
        'exog_names': exog_names,
        'weights_name': weights_name,
        'has_intercept': has_intercept,
        'has_implicit_constant': has_implicit_constant,
        'index': index,
    }
    if uses_inflation:
        constructor_options.update({
            'exog_infl': exog_infl,
            'exog_infl_names': exog_infl_names,
        })
    constructor_options.update(_model_constructor_options(
        model_class, effective_p, positive_link, zero_model
    ))
    model = model_class(endog, exog, **constructor_options)

    common_options = _common_hurdle_fit_options(
        tol, max_iter, alpha, l1_ratio, L2_penalty_matrix,
        regularize_to_values, normalize, penalize_scale, use_t,
        test_level, compute_cov, store_convergence_path,
        line_search_fallback, pick_default_start, opt_method,
        prompt_user_for_more_iters,
    )
    return _fit_selected_model(
        model, start_params, debug, cov_type, cov_kwds,
        positive_fit_kwargs, hurdle_fit_kwargs, common_options,
    )


__all__ = [
    'distributional_model',
    'DISTRIBUTIONAL_MODEL',
    'DISTRIBUTIONAL_MODEL_ALIASES',
]
