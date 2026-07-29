"""Simulate and recover a zero-inflated Poisson regression.

The count and structural-zero equations use different covariates so the two
parts of the mixture are easy to see.  Running this file prints the fitted
summary and a direct comparison of the estimated parameters with their true
simulation values.
"""

import numpy as np
import pandas as pd
from scipy.special import expit

from kanly.sandbox.count_models.count_models import ZeroInflatedPoisson


def main():
    """Generate zero-inflated counts, fit the model, and verify recovery."""
    rng = np.random.default_rng(20260729)
    nobs = 15_000

    # Count equation: log(mu) = beta_0 + beta_x * x.
    beta = np.array([0.20, 0.35])
    x = rng.normal(size=nobs)
    mu = np.exp(beta[0] + beta[1] * x)

    # Inflation equation: logit(pi) = gamma_0 + gamma_z * z, where pi is the
    # probability of belonging to the always-zero component.
    gamma = np.array([-1.00, 0.70])
    z = rng.normal(size=nobs)
    inflation_probability = expit(gamma[0] + gamma[1] * z)

    poisson_counts = rng.poisson(mu)
    is_structural_zero = rng.random(nobs) < inflation_probability
    y = np.where(is_structural_zero, 0, poisson_counts)

    data = pd.DataFrame({'y': y, 'x': x, 'z': z})

    # The count formula and inflation formula are supplied separately.  Both
    # include an intercept unless ``-1`` is specified.
    model = ZeroInflatedPoisson.build_model_from_formula(
        'y ~ x',
        data,
        exog_infl='z',
    )

    # Parameter order is count coefficients followed by inflation
    # coefficients: beta_0, beta_x, gamma_0, gamma_z.
    fit = model.fit(
        start_params=np.array([0.0, 0.0, -0.5, 0.0]),
        cov_type='NONROBUST',
    )

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

    print(f'Observed zero fraction: {np.mean(y == 0):.3f}')
    print(f'Mean structural-zero probability: {inflation_probability.mean():.3f}')
    print('\nModel summary:')
    print(fit.summary_df)
    print('\nParameter recovery:')
    print(comparison)

    if not fit.converged:
        raise RuntimeError(f'Zero-inflated Poisson fit failed: {fit.message}')

    # With this deterministic sample, every coefficient is recovered well
    # within this deliberately loose simulation tolerance.
    max_absolute_error = np.max(np.abs(comparison['error']))
    print(f'\nMaximum absolute parameter error: {max_absolute_error:.4f}')
    assert max_absolute_error < 0.08


if __name__ == '__main__':
    main()
