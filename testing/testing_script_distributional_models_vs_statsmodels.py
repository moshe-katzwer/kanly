"""Compare count-model parameter estimates between Kanly and statsmodels.

The two libraries maximize equivalent likelihoods but do not always expose
parameters in the same order or on the same scale.  In particular:

* statsmodels puts zero-process parameters first; Kanly puts them last.
* statsmodels reports negative-binomial ``alpha`` directly; Kanly estimates
  ``log_alpha``.  Comparisons below use ``alpha`` on its natural scale.
* statsmodels' hurdle implementation uses a right-censored count model for
  the zero process.  The Kanly hurdle cases therefore explicitly select
  ``zero_model='poisson'`` instead of Kanly's default logit hurdle.

The script uses deterministic simulated samples, checks optimizer convergence,
compares transformed parameter vectors, and checks maximized log likelihoods.
It is intended as a standalone parity test rather than a pytest test module.

Examples
--------
Run all comparisons::

    python testing/testing_script_distributional_models_vs_statsmodels.py

Run only ZIP and Poisson-hurdle comparisons with parameter tables::

    python testing/testing_script_distributional_models_vs_statsmodels.py \
        --case zero_inflated_poisson --case poisson_hurdle --verbose
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.special import expit
from statsmodels.discrete.count_model import (
    GeneralizedPoisson as StatsmodelsGeneralizedPoisson,
    NegativeBinomialP as StatsmodelsNegativeBinomialP,
    ZeroInflatedNegativeBinomialP as StatsmodelsZINB,
    ZeroInflatedPoisson as StatsmodelsZIP,
)
from statsmodels.discrete.discrete_model import (
    Poisson as StatsmodelsPoisson,
)
from statsmodels.discrete.truncated_model import HurdleCountModel
from statsmodels.distributions.discrete import genpoisson_p

from kanly.distributional_models import (
    GeneralizedPoisson,
    NegativeBinomial1,
    NegativeBinomial2,
    NegativeBinomialPHurdle,
    Poisson,
    PoissonHurdle,
    ZeroInflatedNegativeBinomial,
    ZeroInflatedPoisson,
)


CASE_NAMES = (
    'poisson',
    'generalized_poisson_1',
    'generalized_poisson_2',
    'negative_binomial_1',
    'negative_binomial_2',
    'zero_inflated_poisson',
    'zero_inflated_negative_binomial_2',
    'poisson_hurdle',
    'negative_binomial_2_hurdle',
)


@dataclass
class ModelPair:
    """Equivalent fitted-model inputs and parameter transformations."""

    name: str
    statsmodels_model: object
    kanly_model: object
    statsmodels_start: np.ndarray | None
    kanly_start: np.ndarray
    parameter_names: list[str]
    transform_statsmodels: Callable[[np.ndarray], np.ndarray]
    transform_kanly: Callable[[np.ndarray], np.ndarray]
    statsmodels_component_starts: tuple[np.ndarray, np.ndarray] | None = None


@dataclass
class ComparisonResult:
    """Outcome and diagnostics for one model-pair comparison."""

    name: str
    parameter_names: list[str] = field(default_factory=list)
    statsmodels_params: np.ndarray | None = None
    kanly_params: np.ndarray | None = None
    statsmodels_llf: float | None = None
    kanly_llf: float | None = None
    statsmodels_converged: bool = False
    kanly_converged: bool = False
    warning_messages: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def passed(self) -> bool:
        """Return whether fitting and all numerical checks succeeded."""
        return self.error is None

    @property
    def max_abs_parameter_difference(self) -> float:
        """Return the largest comparable parameter discrepancy."""
        if self.statsmodels_params is None or self.kanly_params is None:
            return np.inf
        return float(np.max(np.abs(
            self.statsmodels_params - self.kanly_params
        )))


def _designs(rng: np.random.Generator, nobs: int):
    """Return distinct main and zero-process designs with intercepts."""
    x = rng.normal(size=nobs)
    z = rng.normal(size=nobs)
    exog = np.column_stack((np.ones(nobs), x))
    exog_zero = np.column_stack((np.ones(nobs), z))
    return exog, exog_zero


def _sample_negative_binomial_p(
        rng: np.random.Generator,
        mean: np.ndarray,
        alpha: float,
        p: int,
) -> np.ndarray:
    """Draw NB-P counts with variance ``mean + alpha * mean**p``."""
    size = mean ** (2 - p) / alpha
    probability = size / (size + mean)
    return rng.negative_binomial(size, probability)


def _sample_zero_truncated(
        sampler: Callable[[], np.ndarray],
) -> np.ndarray:
    """Repeatedly draw through ``sampler`` until every count is positive."""
    values = sampler()
    is_zero = values == 0
    while np.any(is_zero):
        replacement = sampler()
        values[is_zero] = replacement[is_zero]
        is_zero = values == 0
    return values


def _identity(params: np.ndarray) -> np.ndarray:
    """Return a float parameter vector without changing its order."""
    return np.asarray(params, dtype=float)


def _nb_kanly_to_common(params: np.ndarray) -> np.ndarray:
    """Convert Kanly's final ``log_alpha`` to statsmodels' raw ``alpha``."""
    params = np.asarray(params, dtype=float).copy()
    params[-1] = np.exp(params[-1])
    return params


def _zip_statsmodels_to_common(params: np.ndarray, k_zero: int) -> np.ndarray:
    """Reorder statsmodels ZIP parameters from zero-first to count-first."""
    params = np.asarray(params, dtype=float)
    return np.r_[params[k_zero:], params[:k_zero]]


def _zinb_statsmodels_to_common(
        params: np.ndarray,
        k_count: int,
        k_zero: int,
) -> np.ndarray:
    """Reorder statsmodels ZINB parameters to count, zero, then alpha."""
    params = np.asarray(params, dtype=float)
    zero = params[:k_zero]
    count = params[k_zero:k_zero + k_count]
    alpha = params[-1:]
    return np.r_[count, zero, alpha]


def _zinb_kanly_to_common(
        params: np.ndarray,
        k_count: int,
        k_zero: int,
) -> np.ndarray:
    """Convert Kanly ZINB's trailing ``log_alpha`` to raw ``alpha``."""
    params = np.asarray(params, dtype=float)
    return np.r_[
        params[:k_count],
        params[k_count:k_count + k_zero],
        np.exp(params[-1]),
    ]


def _hurdle_statsmodels_to_common(
        params: np.ndarray,
        k_hurdle: int,
        has_alpha: bool,
) -> np.ndarray:
    """Reorder statsmodels hurdle parameters to positive-first order."""
    params = np.asarray(params, dtype=float)
    hurdle = params[:k_hurdle]
    positive = params[k_hurdle:]
    if has_alpha:
        return np.r_[positive[:-1], positive[-1], hurdle]
    return np.r_[positive, hurdle]


def _hurdle_kanly_to_common(
        params: np.ndarray,
        k_positive: int,
        has_alpha: bool,
) -> np.ndarray:
    """Convert a Kanly hurdle vector to the shared comparison scale."""
    params = np.asarray(params, dtype=float)
    if not has_alpha:
        return params
    return np.r_[
        params[:k_positive],
        np.exp(params[k_positive]),
        params[k_positive + 1:],
    ]


def make_model_pair(name: str, nobs: int, seed: int) -> ModelPair:
    """Simulate one deterministic sample and construct equivalent models."""
    rng = np.random.default_rng(seed)
    exog, exog_zero = _designs(rng, nobs)
    beta = np.array([0.25, -0.35])
    gamma = np.array([-0.65, 0.45])
    mean = np.exp(exog @ beta)
    k_count = exog.shape[1]
    k_zero = exog_zero.shape[1]
    count_names = ['count_const', 'count_x']
    zero_names = ['zero_const', 'zero_z']

    if name == 'poisson':
        endog = rng.poisson(mean)
        return ModelPair(
            name=name,
            statsmodels_model=StatsmodelsPoisson(endog, exog),
            kanly_model=Poisson(endog, exog),
            statsmodels_start=beta,
            kanly_start=beta,
            parameter_names=count_names,
            transform_statsmodels=_identity,
            transform_kanly=_identity,
        )

    if name.startswith('generalized_poisson_'):
        p = int(name.rsplit('_', 1)[1])
        alpha = 0.25
        endog = genpoisson_p.rvs(
            mean, alpha, p, random_state=rng,
        ).astype(int)
        start = np.r_[beta, alpha]
        return ModelPair(
            name=name,
            statsmodels_model=StatsmodelsGeneralizedPoisson(
                endog, exog, p=p,
            ),
            kanly_model=GeneralizedPoisson(endog, exog, p=p),
            statsmodels_start=start,
            kanly_start=start,
            parameter_names=count_names + ['alpha'],
            transform_statsmodels=_identity,
            transform_kanly=_identity,
        )

    if name in {'negative_binomial_1', 'negative_binomial_2'}:
        p = int(name.rsplit('_', 1)[1])
        alpha = 0.55
        endog = _sample_negative_binomial_p(rng, mean, alpha, p)
        kanly_class = NegativeBinomial1 if p == 1 else NegativeBinomial2
        return ModelPair(
            name=name,
            statsmodels_model=StatsmodelsNegativeBinomialP(
                endog, exog, p=p,
            ),
            kanly_model=kanly_class(endog, exog),
            statsmodels_start=np.r_[beta, alpha],
            kanly_start=np.r_[beta, np.log(alpha)],
            parameter_names=count_names + ['alpha'],
            transform_statsmodels=_identity,
            transform_kanly=_nb_kanly_to_common,
        )

    if name == 'zero_inflated_poisson':
        inflation_probability = expit(exog_zero @ gamma)
        endog = rng.poisson(mean)
        endog[rng.random(nobs) < inflation_probability] = 0
        return ModelPair(
            name=name,
            statsmodels_model=StatsmodelsZIP(
                endog, exog, exog_infl=exog_zero,
            ),
            kanly_model=ZeroInflatedPoisson(
                endog, exog, exog_infl=exog_zero,
            ),
            statsmodels_start=np.r_[gamma, beta],
            kanly_start=np.r_[beta, gamma],
            parameter_names=count_names + zero_names,
            transform_statsmodels=lambda params: (
                _zip_statsmodels_to_common(params, k_zero)
            ),
            transform_kanly=_identity,
        )

    if name == 'zero_inflated_negative_binomial_2':
        alpha = 0.55
        inflation_probability = expit(exog_zero @ gamma)
        endog = _sample_negative_binomial_p(rng, mean, alpha, p=2)
        endog[rng.random(nobs) < inflation_probability] = 0
        return ModelPair(
            name=name,
            statsmodels_model=StatsmodelsZINB(
                endog, exog, exog_infl=exog_zero, p=2,
            ),
            kanly_model=ZeroInflatedNegativeBinomial(
                endog, exog, exog_infl=exog_zero,
            ),
            statsmodels_start=np.r_[gamma, beta, alpha],
            kanly_start=np.r_[beta, gamma, np.log(alpha)],
            parameter_names=count_names + zero_names + ['alpha'],
            transform_statsmodels=lambda params: (
                _zinb_statsmodels_to_common(params, k_count, k_zero)
            ),
            transform_kanly=lambda params: (
                _zinb_kanly_to_common(params, k_count, k_zero)
            ),
        )

    if name == 'poisson_hurdle':
        # statsmodels currently requires the same design for both components.
        hurdle_mean = np.exp(exog @ gamma)
        positive_probability = -np.expm1(-hurdle_mean)
        endog = np.zeros(nobs, dtype=int)
        positive = rng.random(nobs) < positive_probability
        positive_counts = _sample_zero_truncated(
            lambda: rng.poisson(mean)
        )
        endog[positive] = positive_counts[positive]
        return ModelPair(
            name=name,
            statsmodels_model=HurdleCountModel(
                endog, exog, dist='poisson', zerodist='poisson',
            ),
            kanly_model=PoissonHurdle(
                endog, exog, exog_infl=exog, zero_model='poisson',
            ),
            statsmodels_start=gamma,
            kanly_start=np.r_[beta, gamma],
            parameter_names=count_names + ['hurdle_const', 'hurdle_x'],
            transform_statsmodels=lambda params: (
                _hurdle_statsmodels_to_common(
                    params, k_hurdle=k_count, has_alpha=False,
                )
            ),
            transform_kanly=lambda params: (
                _hurdle_kanly_to_common(
                    params, k_positive=k_count, has_alpha=False,
                )
            ),
        )

    if name == 'negative_binomial_2_hurdle':
        p = int(name.split('_')[2])
        alpha = 0.55
        hurdle_mean = np.exp(exog @ gamma)
        positive_probability = -np.expm1(-hurdle_mean)
        endog = np.zeros(nobs, dtype=int)
        positive = rng.random(nobs) < positive_probability
        positive_counts = _sample_zero_truncated(
            lambda: _sample_negative_binomial_p(
                rng, mean, alpha, p=p,
            )
        )
        endog[positive] = positive_counts[positive]
        return ModelPair(
            name=name,
            statsmodels_model=HurdleCountModel(
                endog, exog, dist='negbin', zerodist='poisson', p=p,
            ),
            kanly_model=NegativeBinomialPHurdle(
                endog, exog, exog_infl=exog,
                zero_model='poisson', p=p,
            ),
            # The combined statsmodels fitter sends one start vector to both
            # unequal-length components. Fit them separately below so each
            # receives a correctly sized, equivalent start vector.
            statsmodels_start=None,
            kanly_start=np.r_[beta, np.log(alpha), gamma],
            parameter_names=(
                count_names + ['alpha', 'hurdle_const', 'hurdle_x']
            ),
            transform_statsmodels=lambda params: (
                _hurdle_statsmodels_to_common(
                    params, k_hurdle=k_count, has_alpha=True,
                )
            ),
            transform_kanly=lambda params: (
                _hurdle_kanly_to_common(
                    params, k_positive=k_count, has_alpha=True,
                )
            ),
            statsmodels_component_starts=(
                gamma,
                np.r_[beta, alpha],
            ),
        )

    raise ValueError(f'Unknown comparison case: {name}')


def _statsmodels_converged(fit) -> bool:
    """Normalize scalar and component-wise statsmodels convergence flags."""
    convergence = getattr(fit, 'mle_retvals', {}).get('converged', True)
    return bool(np.all(np.asarray(convergence, dtype=bool)))


def _fit_statsmodels(pair: ModelPair):
    """Fit a statsmodels model, handling unequal hurdle components."""
    fit_options = {
        'method': 'bfgs',
        'maxiter': 1000,
        'disp': False,
        'gtol': 1e-8,
    }
    if pair.statsmodels_component_starts is None:
        fit = pair.statsmodels_model.fit(
            start_params=pair.statsmodels_start,
            **fit_options,
        )
        return (
            np.asarray(fit.params, dtype=float),
            float(fit.llf),
            _statsmodels_converged(fit),
        )

    zero_start, positive_start = pair.statsmodels_component_starts
    zero_fit = pair.statsmodels_model.model1.fit(
        start_params=zero_start,
        **fit_options,
    )
    positive_fit = pair.statsmodels_model.model2.fit(
        start_params=positive_start,
        **fit_options,
    )
    return (
        np.r_[zero_fit.params, positive_fit.params],
        float(zero_fit.llf + positive_fit.llf),
        bool(
            _statsmodels_converged(zero_fit)
            and _statsmodels_converged(positive_fit)
        ),
    )


def run_comparison(
        pair: ModelPair,
        parameter_atol: float,
        llf_atol: float,
) -> ComparisonResult:
    """Fit and numerically compare one equivalent model pair."""
    result = ComparisonResult(
        name=pair.name,
        parameter_names=pair.parameter_names,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            sm_params, sm_llf, sm_converged = _fit_statsmodels(pair)
            kanly_fit = pair.kanly_model.fit(
                start_params=pair.kanly_start,
                cov_type='NONROBUST',
            )

            result.statsmodels_params = pair.transform_statsmodels(
                sm_params
            )
            result.kanly_params = pair.transform_kanly(
                np.asarray(kanly_fit.params, dtype=float)
            )
            result.statsmodels_llf = sm_llf
            result.kanly_llf = float(kanly_fit.llf)
            result.statsmodels_converged = sm_converged
            result.kanly_converged = bool(kanly_fit.converged)

            if not result.statsmodels_converged:
                raise AssertionError('statsmodels fit did not converge')
            if not result.kanly_converged:
                raise AssertionError('Kanly fit did not converge')
            if result.statsmodels_params.shape != result.kanly_params.shape:
                raise AssertionError(
                    'parameter shapes differ: '
                    f'{result.statsmodels_params.shape} != '
                    f'{result.kanly_params.shape}'
                )
            np.testing.assert_allclose(
                result.kanly_params,
                result.statsmodels_params,
                rtol=0.0,
                atol=parameter_atol,
            )
            np.testing.assert_allclose(
                result.kanly_llf,
                result.statsmodels_llf,
                rtol=0.0,
                atol=llf_atol,
            )
        except Exception as exc:  # Let the remaining cases provide diagnostics.
            result.error = f'{type(exc).__name__}: {exc}'
        result.warning_messages = [str(item.message) for item in caught]
    return result


def print_report(
        results: Sequence[ComparisonResult],
        verbose: bool,
        show_warnings: bool,
) -> None:
    """Print concise results plus detailed parameter tables when requested."""
    for result in results:
        status = 'PASS' if result.passed else 'FAIL'
        print(
            f'{status:4} {result.name:38} '
            f'max |parameter difference|='
            f'{result.max_abs_parameter_difference:.3e}'
        )
        if result.error is not None:
            print(f'     {result.error}')
        if verbose and result.statsmodels_params is not None:
            table = pd.DataFrame({
                'statsmodels': result.statsmodels_params,
                'kanly': result.kanly_params,
                'difference': (
                    result.kanly_params - result.statsmodels_params
                ),
            }, index=result.parameter_names)
            print(table)
            print(
                '     log likelihoods: '
                f'statsmodels={result.statsmodels_llf:.10f}, '
                f'kanly={result.kanly_llf:.10f}, '
                f'difference='
                f'{result.kanly_llf - result.statsmodels_llf:.3e}'
            )
        if show_warnings:
            for message in dict.fromkeys(result.warning_messages):
                print(f'     warning: {message}')

    failures = sum(not result.passed for result in results)
    print(
        f'\nRan {len(results)} comparisons: '
        f'{len(results) - failures} passed, {failures} failed'
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse test selection, tolerance, and reporting options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nobs', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=20260821)
    parser.add_argument(
        '--case', action='append', choices=CASE_NAMES,
        help='comparison to run; repeat for several (default: all)',
    )
    parser.add_argument(
        '--parameter-atol', type=float, default=5e-4,
        help='absolute parameter tolerance (default: %(default)s)',
    )
    parser.add_argument(
        '--llf-atol', type=float, default=5e-5,
        help='absolute maximized-log-likelihood tolerance (default: %(default)s)',
    )
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--show-warnings', action='store_true')
    parser.add_argument('--fail-fast', action='store_true')
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run selected comparisons and return a CI-friendly process status."""
    args = parse_args(argv)
    selected = args.case if args.case else CASE_NAMES
    results = []
    for name in selected:
        pair = make_model_pair(
            name,
            nobs=args.nobs,
            seed=args.seed + CASE_NAMES.index(name),
        )
        result = run_comparison(
            pair,
            parameter_atol=args.parameter_atol,
            llf_atol=args.llf_atol,
        )
        results.append(result)
        if args.fail_fast and not result.passed:
            break

    print_report(
        results,
        verbose=args.verbose,
        show_warnings=args.show_warnings,
    )
    return int(any(not result.passed for result in results))


if __name__ == '__main__':
    raise SystemExit(main())
