"""Structured GLM parity sweep for Kanly, statsmodels, and ZTP references.

This is a proposed replacement for ``testing_script_glm.py``.  It exercises
the same optimizer, IV, residual-inclusion, variance-weight, family, and safe-
link combinations, but makes each combination an explicit test case.  Ordinary
families are compared with statsmodels; zero-truncated Poisson is compared with
an independently optimized conditional likelihood because statsmodels does not
provide that family.

Examples
--------
Run the complete sweep with concise output::

    python testing/testing_script_glm_proposed.py

Run only zero-truncated Poisson cases and print fitted-model details::

    python testing/testing_script_glm_proposed.py \
        --family ZERO_TRUNCATED_POISSON --verbose
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from statsmodels.formula.api import glm as statsmodels_glm
from statsmodels.genmod import families as sm_families
from statsmodels.genmod.families.links import (
    Link as StatsmodelsLink,
    inverse_power,
    inverse_squared,
)

from kanly.api import glm, lm
from kanly.regression.generalized_linear_models.families import (
    Binomial,
    Gamma,
    Gaussian,
    InverseGaussian,
    Poisson,
    ZeroTruncatedPoisson,
)
from kanly.regression.generalized_linear_models.sparse_glm_internal import (
    METHOD_COORD_DESC,
    METHOD_IRLS,
)


FAMILY_CLASSES = (
    Poisson,
    Gaussian,
    Gamma,
    InverseGaussian,
    Binomial,
    ZeroTruncatedPoisson,
)
OPTIMIZATION_METHODS = (METHOD_IRLS, METHOD_COORD_DESC)
NEGATED_LINKS = {'NEGATIVE_INVERSE', 'NEGATIVE_TWO_INVERSE_SQUARED'}


def _normalized_class_name(value) -> str:
    """Return a punctuation-insensitive class name for registry matching."""
    return value.__name__.upper().replace('_', '')


SM_FAMILY_TYPES = {
    _normalized_class_name(family): family
    for family in (
        sm_families.Tweedie,
        sm_families.Gamma,
        sm_families.Gaussian,
        sm_families.InverseGaussian,
        sm_families.NegativeBinomial,
        sm_families.Binomial,
        sm_families.Poisson,
    )
}
SM_LINK_TYPES = {
    _normalized_class_name(link): link
    for family in SM_FAMILY_TYPES.values()
    for link in family.links
}
SM_LINK_TYPES.update({
    'INVERSE': inverse_power,
    'NEGATIVEINVERSE': inverse_power,
    'NEGATIVETWOINVERSESQUARED': inverse_squared,
})


@dataclass(frozen=True)
class ComparisonCase:
    """One combination in the GLM cross-implementation parity sweep."""

    opt_method: str
    do_iv: bool
    residual_inclusion: bool
    do_weighted: bool
    family_class: type
    link_class: type

    @property
    def family_name(self) -> str:
        """Return the registered Kanly family name."""
        return self.family_class.name()

    @property
    def link_name(self) -> str:
        """Return the registered Kanly link name."""
        return self.link_class.name()

    @property
    def identifier(self) -> str:
        """Return a stable, human-readable case identifier."""
        return (
            f'{self.family_name}/{self.link_name} '
            f'method={self.opt_method} iv={self.do_iv} '
            f'residual_inclusion={self.residual_inclusion} '
            f'weighted={self.do_weighted}'
        )


@dataclass
class ReferenceFit:
    """Small common interface for statsmodels and direct ZTP fits."""

    params: np.ndarray
    llf: float
    converged: bool
    details: object


@dataclass
class CaseResult:
    """Observed result, diagnostics, and warnings for one comparison case."""

    case: ComparisonCase
    passed: bool = False
    reference_params: np.ndarray | None = None
    kanly_params: np.ndarray | None = None
    reference_llf: float | None = None
    kanly_llf: float | None = None
    error: str | None = None
    warning_messages: list[str] = field(default_factory=list)

    @property
    def max_abs_parameter_difference(self) -> float:
        """Return the largest coefficient discrepancy, or infinity on error."""
        if self.reference_params is None or self.kanly_params is None:
            return np.inf
        return float(np.max(np.abs(self.reference_params - self.kanly_params)))


def iter_cases(
        family_names: set[str] | None = None,
        methods: set[str] | None = None,
) -> Iterable[ComparisonCase]:
    """Yield every case covered by the original GLM testing script."""
    for opt_method in OPTIMIZATION_METHODS:
        if methods is not None and opt_method not in methods:
            continue
        for do_iv in (True, False):
            for residual_inclusion in (True, False):
                for do_weighted in (True, False):
                    for family_class in FAMILY_CLASSES:
                        if (
                                family_names is not None
                                and family_class.name() not in family_names
                        ):
                            continue
                        for link_class in family_class.safe_links():
                            yield ComparisonCase(
                                opt_method=opt_method,
                                do_iv=do_iv,
                                residual_inclusion=residual_inclusion,
                                do_weighted=do_weighted,
                                family_class=family_class,
                                link_class=link_class,
                            )


def make_data(case: ComparisonCase, nobs: int, seed: int) -> pd.DataFrame:
    """Generate deterministic covariates and a response for one case."""
    rng = np.random.RandomState(seed)
    data = pd.DataFrame(index=np.arange(nobs))
    data['z'] = 1.5 + 0.6 * rng.randn(nobs)
    data['x'] = 3.0 + 0.1 * rng.randn(nobs) + 0.8 * data['z']
    data['e'] = 0.3 * rng.rand(nobs)
    data['wtsvar'] = np.exp(rng.rand(nobs))

    if case.do_iv:
        first_stage_formula = 'x ~ z'
        if case.do_weighted:
            first_stage_formula += ' $ wtsvar'
        first_stage = lm(first_stage_formula, data)
        data['x_pred'] = np.asarray(first_stage.fittedvalues)
        data['x_ri'] = data.x.to_numpy() - data.x_pred.to_numpy()

    link = case.link_class()
    if case.family_name == 'BINOMIAL':
        if case.link_name == 'IDENTITY':
            probability = 0.5 + 0.05 * data.x.to_numpy()
        else:
            odds = np.exp(-5.0 + 1.5 * data.x.to_numpy() + data.e.to_numpy())
            probability = odds / (1.0 + odds)
        data['y'] = (rng.rand(nobs) < probability).astype(float)
    elif case.family_name == 'ZERO_TRUNCATED_POISSON':
        eta = -0.5 + 0.35 * data.x.to_numpy() + data.e.to_numpy()
        data['y'] = sample_zero_truncated_poisson(rng, np.exp(eta))
    else:
        linear_predictor = 3.0 + 1.5 * data.x.to_numpy() + data.e.to_numpy()
        if case.link_name in NEGATED_LINKS:
            linear_predictor = -linear_predictor
        data['y'] = link.inverse_link(linear_predictor)

    return data


def sample_zero_truncated_poisson(
        rng: np.random.RandomState,
        rate: np.ndarray,
) -> np.ndarray:
    """Draw Poisson observations conditional on each count being positive."""
    counts = rng.poisson(rate)
    is_zero = counts == 0
    while np.any(is_zero):
        counts[is_zero] = rng.poisson(rate[is_zero])
        is_zero = counts == 0
    return counts


def zero_truncated_poisson_mean(rate: np.ndarray) -> np.ndarray:
    """Compute the conditional mean independently of Kanly's link code."""
    rate = np.asarray(rate, dtype=float)
    small = rate < 1e-5
    safe_rate = np.where(small, 1.0, rate)
    mean = safe_rate / (-np.expm1(-safe_rate))
    rate2 = rate * rate
    mean_small = 1.0 + rate / 2.0 + rate2 / 12.0 - rate2 ** 2 / 720.0
    return np.where(small, mean_small, mean)


def formulas_for_case(case: ComparisonCase) -> tuple[str, str, list[str]]:
    """Build equivalent Kanly/reference formulas and reference predictors."""
    kanly_formula = 'y ~ x'
    reference_predictors = ['x']
    if case.do_iv:
        kanly_formula += ' | z'
        reference_predictors = ['x_pred']
        if case.residual_inclusion:
            reference_predictors.append('x_ri')
    if case.do_weighted:
        kanly_formula += ' $ wtsvar'

    reference_formula = 'y ~ ' + ' + '.join(reference_predictors)
    return kanly_formula, reference_formula, reference_predictors


def statsmodels_family(case: ComparisonCase):
    """Construct the statsmodels family/link corresponding to a Kanly case."""
    family_type = SM_FAMILY_TYPES[_normalized_class_name(case.family_class)]
    link_or_type = SM_LINK_TYPES[_normalized_class_name(case.link_class)]
    link = (
        link_or_type
        if isinstance(link_or_type, StatsmodelsLink)
        else link_or_type()
    )
    return family_type(link=link)


def fit_zero_truncated_poisson_reference(
        endog: np.ndarray,
        exog: np.ndarray,
        var_weights: np.ndarray | None,
) -> ReferenceFit:
    """Fit the weighted conditional Poisson likelihood directly with SciPy."""
    endog = np.asarray(endog, dtype=float)
    exog = np.asarray(exog, dtype=float)
    weights = (
        np.ones_like(endog)
        if var_weights is None
        else np.asarray(var_weights, dtype=float)
    )

    def objective(params: np.ndarray) -> float:
        eta = exog @ params
        rate = np.exp(eta)
        loglike_obs = (
            endog * eta - rate - gammaln(endog + 1.0)
            - np.log(-np.expm1(-rate))
        )
        return float(-np.dot(weights, loglike_obs))

    def gradient(params: np.ndarray) -> np.ndarray:
        eta = exog @ params
        rate = np.exp(eta)
        score_eta = endog - zero_truncated_poisson_mean(rate)
        return -(exog.T @ (weights * score_eta))

    result = minimize(
        objective,
        np.zeros(exog.shape[1]),
        jac=gradient,
        method='BFGS',
        options={'gtol': 1e-9, 'maxiter': 2000},
    )
    # BFGS occasionally reports precision loss at an otherwise valid optimum;
    # use the analytical score to distinguish that status from non-convergence.
    gradient_is_small = np.max(np.abs(gradient(result.x))) < 1e-5
    return ReferenceFit(
        params=np.asarray(result.x),
        llf=-float(result.fun),
        converged=bool(result.success or gradient_is_small),
        details=result,
    )


def fit_reference(
        case: ComparisonCase,
        data: pd.DataFrame,
        reference_formula: str,
        reference_predictors: Sequence[str],
) -> ReferenceFit:
    """Fit either the statsmodels reference or the direct ZTP likelihood."""
    weights = data.wtsvar.to_numpy() if case.do_weighted else None
    if case.family_name == 'ZERO_TRUNCATED_POISSON':
        exog = np.column_stack(
            [np.ones(len(data))]
            + [data[name].to_numpy() for name in reference_predictors]
        )
        return fit_zero_truncated_poisson_reference(
            data.y.to_numpy(), exog, weights,
        )

    fit = statsmodels_glm(
        reference_formula,
        data,
        family=statsmodels_family(case),
        var_weights=weights,
    ).fit(tol=1e-12, max_iter=1000)
    return ReferenceFit(
        params=np.asarray(fit.params),
        llf=float(fit.llf),
        converged=bool(fit.converged),
        details=fit,
    )


def transformed_kanly_params(case: ComparisonCase, params) -> np.ndarray:
    """Apply the historical sign/scale conventions used by the parity script."""
    transformed = np.asarray(params, dtype=float)
    if case.link_name in NEGATED_LINKS:
        transformed = -transformed
    if case.link_name == 'NEGATIVE_TWO_INVERSE_SQUARED':
        transformed = 2.0 * transformed
    return transformed


def run_case(
        case: ComparisonCase,
        nobs: int,
        seed: int,
        atol: float,
        verbose: bool,
) -> CaseResult:
    """Run a case without allowing its errors or warnings to stop the sweep."""
    result = CaseResult(case=case)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            data = make_data(case, nobs=nobs, seed=seed)
            kanly_formula, reference_formula, predictors = formulas_for_case(case)
            reference = fit_reference(
                case, data, reference_formula, predictors,
            )
            kanly_fit = glm(
                kanly_formula,
                data,
                family=case.family_class,
                link=case.link_class(),
                cov_type='HC1',
                tol=1e-12,
                max_iter=5000,
                opt_method=case.opt_method,
                line_search_fallback=True,
                residual_inclusion=case.residual_inclusion,
                pick_default_start=True,
                debug=False,
            )

            result.reference_params = reference.params
            result.kanly_params = transformed_kanly_params(
                case, kanly_fit.params,
            )
            result.reference_llf = reference.llf
            result.kanly_llf = float(kanly_fit.llf)

            if result.reference_params.shape != result.kanly_params.shape:
                raise AssertionError(
                    'parameter shapes differ: '
                    f'{result.reference_params.shape} != '
                    f'{result.kanly_params.shape}'
                )
            np.testing.assert_allclose(
                result.kanly_params,
                result.reference_params,
                rtol=0.0,
                atol=atol,
            )
            if (
                    case.family_name == 'ZERO_TRUNCATED_POISSON'
                    and not reference.converged
            ):
                raise AssertionError('reference fit did not converge')
            if not kanly_fit.converged:
                raise AssertionError('Kanly fit did not converge')
            result.passed = True

            if verbose:
                print(f'\n{case.identifier}')
                print(pd.DataFrame({
                    'reference': result.reference_params,
                    'kanly': result.kanly_params,
                    'difference': (
                        result.kanly_params - result.reference_params
                    ),
                }))
                if hasattr(reference.details, 'summary'):
                    print(reference.details.summary())
                else:
                    print(reference.details)
                print(kanly_fit)
        except Exception as exc:  # Continue so one case cannot hide others.
            result.error = f'{type(exc).__name__}: {exc}'

        result.warning_messages = [str(item.message) for item in caught]
    return result


def print_report(
        results: Sequence[CaseResult],
        show_warnings: bool,
) -> None:
    """Print a concise suite summary and actionable failure diagnostics."""
    failures = [result for result in results if not result.passed]
    print(f'Ran {len(results)} cases: {len(results) - len(failures)} passed, '
          f'{len(failures)} failed')

    for result in failures:
        print(f'\nFAIL: {result.case.identifier}')
        print(f'  {result.error}')
        if result.reference_params is not None:
            print(pd.DataFrame({
                'reference': result.reference_params,
                'kanly': result.kanly_params,
            }))
            print(
                '  max absolute parameter difference: '
                f'{result.max_abs_parameter_difference:.8g}'
            )
            print(
                f'  log likelihoods: reference={result.reference_llf:.8g}, '
                f'kanly={result.kanly_llf:.8g}'
            )

    if show_warnings:
        warned = [result for result in results if result.warning_messages]
        print(f'\nWarnings were emitted by {len(warned)} cases')
        for result in warned:
            print(f'  {result.case.identifier}')
            for message in dict.fromkeys(result.warning_messages):
                print(f'    {message}')


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line filters and reporting options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nobs', type=int, default=150)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--atol', type=float, default=1e-5,
        help='absolute coefficient tolerance (default: %(default)s)',
    )
    parser.add_argument(
        '--family', action='append', choices=[f.name() for f in FAMILY_CLASSES],
        help='family to run; repeat for multiple families (default: all)',
    )
    parser.add_argument(
        '--method', action='append', choices=list(OPTIMIZATION_METHODS),
        help='optimization method to run; repeat for multiple (default: all)',
    )
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--show-warnings', action='store_true')
    parser.add_argument('--fail-fast', action='store_true')
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Execute selected cases and return a CI-friendly process status."""
    args = parse_args(argv)
    cases = iter_cases(
        family_names=set(args.family) if args.family else None,
        methods=set(args.method) if args.method else None,
    )
    results = []
    for case in cases:
        result = run_case(
            case,
            nobs=args.nobs,
            seed=args.seed,
            atol=args.atol,
            verbose=args.verbose,
        )
        results.append(result)
        if args.fail_fast and not result.passed:
            break

    print_report(results, show_warnings=args.show_warnings)
    return int(any(not result.passed for result in results))


if __name__ == '__main__':
    raise SystemExit(main())
