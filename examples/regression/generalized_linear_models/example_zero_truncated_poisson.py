"""Simulate and recover a zero-truncated Poisson regression.

The regression is specified for the underlying Poisson rate,
``log(lambda) = X beta``.  Counts are then sampled conditional on being
strictly positive.  This differs from dropping zeros after fitting or
simulating an ordinary Poisson model because the truncation normalizer enters
the likelihood and changes the conditional response mean.
"""

import numpy as np
import pandas as pd

from kanly.api import glm


def sample_zero_truncated_poisson(rng, rate):
    """Draw independent Poisson counts conditional on being positive."""
    counts = rng.poisson(rate)
    is_zero = counts == 0
    while np.any(is_zero):
        counts[is_zero] = rng.poisson(rate[is_zero])
        is_zero = counts == 0
    return counts


def main():
    """Generate positive counts, fit the GLM, and display parameter recovery."""
    rng = np.random.default_rng(0)
    nobs = 10_000

    true_params = np.array([0.25, 0.45])
    x = rng.normal(size=nobs)
    rate = np.exp(true_params[0] + true_params[1] * x)
    y = sample_zero_truncated_poisson(rng, rate)
    data = pd.DataFrame({'y': y, 'x': x})

    fit = glm(
        'y ~ x',
        data,
        family='zero_truncated_poisson',
        cov_type='nonrobust',
    )

    comparison = pd.DataFrame(
        {
            'true': true_params,
            'estimated': np.asarray(fit.params),
            'std err': np.asarray(fit.bse),
            'error': np.asarray(fit.params) - true_params,
        },
        index=fit.param_names,
    )

    print(f'Number of zero outcomes: {np.count_nonzero(y == 0)}')
    print(f'Family: {fit.family.name()}')
    print(f'Default link: {fit.link.name()}')
    print(f'Fixed dispersion: {fit.family.is_fixed_dispersion()}')
    print(f'Estimated GLM scale: {fit.scale:.1f}')
    print('\nModel summary:')
    print(fit.summary())
    print('\nParameter recovery:')
    print(comparison)

    if not fit.converged:
        raise RuntimeError('Zero-truncated Poisson fit did not converge')
    assert np.all(y > 0)
    assert fit.scale == 1.0
    assert np.max(np.abs(comparison['error'])) < 0.08


if __name__ == '__main__':
    main()
