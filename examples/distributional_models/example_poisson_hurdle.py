"""Simulate and recover a Poisson hurdle model.

The hurdle equation models the probability of an observed zero with a logit.
Once the hurdle is crossed, counts follow a Poisson distribution conditional
on being strictly positive.  The two equations use different covariates so
their parameter recovery is easy to inspect.
"""

import numpy as np
import pandas as pd
from scipy.special import expit

from kanly.distributional_models import PoissonHurdle


def sample_zero_truncated_poisson(rng, rate):
    """Draw independent Poisson counts conditional on being positive."""
    counts = rng.poisson(rate)
    is_zero = counts == 0
    while np.any(is_zero):
        counts[is_zero] = rng.poisson(rate[is_zero])
        is_zero = counts == 0
    return counts


def main():
    """Generate hurdle counts, fit both GLMs, and verify recovery."""
    rng = np.random.default_rng(20260801)
    nobs = 15_000

    # Positive equation: log(lambda) = beta_0 + beta_x * x, where lambda is
    # the rate of the underlying Poisson before conditioning on Y > 0.
    beta = np.array([0.20, 0.40])
    x = rng.normal(size=nobs)
    rate = np.exp(beta[0] + beta[1] * x)

    # Zero hurdle: logit(P(Y = 0)) = gamma_0 + gamma_z * z.
    gamma = np.array([-0.70, 0.65])
    z = rng.normal(size=nobs)
    zero_probability = expit(gamma[0] + gamma[1] * z)
    is_zero = rng.random(nobs) < zero_probability

    y = sample_zero_truncated_poisson(rng, rate)
    y[is_zero] = 0
    data = pd.DataFrame({'y': y, 'x': x, 'z': z})

    model = PoissonHurdle.build_model_from_formula(
        'y ~ x',
        data,
        exog_infl='z',
    )
    fit = model.fit(
        start_params=np.zeros(len(model.param_names)),
        cov_type='NONROBUST',
    )

    # Parameter order is positive-response beta followed by zero-hurdle gamma.
    true_params = np.concatenate((beta, gamma))
    comparison = pd.DataFrame(
        {
            'true': true_params,
            'estimated': fit.params,
            'std err': fit.bse,
            'error': fit.params - true_params,
        },
        index=fit.param_names,
    )

    k_positive = len(beta)
    cross_covariance = np.asarray(fit.cov_params())[
        :k_positive, k_positive:
    ]
    max_cross_covariance = np.max(np.abs(cross_covariance))

    print(f'Observed zero fraction: {np.mean(y == 0):.3f}')
    print(f'Mean simulated zero probability: {zero_probability.mean():.3f}')
    print(f'Positive family: {fit.positive_fit.family.name()}')
    print(f'Positive link: {fit.positive_fit.link.name()}')
    print(f'Hurdle family/link: {fit.hurdle_fit.family.name()} / '
          f'{fit.hurdle_fit.link.name()}')
    print(f'Positive GLM scale: {fit.positive_scale:.1f}')
    print('\nCombined model summary:')
    print(fit.summary_df())
    print('\nParameter recovery:')
    print(comparison)
    print('\nMaximum absolute cross-block covariance: '
          f'{max_cross_covariance:.1e}')

    if not fit.converged:
        raise RuntimeError(f'Poisson hurdle fit failed: {fit.message}')

    max_absolute_error = np.max(np.abs(comparison['error']))
    print(f'Maximum absolute parameter error: {max_absolute_error:.4f}')
    assert max_absolute_error < 0.08
    assert fit.positive_scale == 1.0
    assert max_cross_covariance == 0.0
    assert np.all(y[y > 0] >= 1)


if __name__ == '__main__':
    main()
