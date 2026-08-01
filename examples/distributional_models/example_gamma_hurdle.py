"""Simulate and recover a Gamma hurdle model.

The first component is a Bernoulli/logit model for the probability of zero.
Conditional on a positive response, the second component is a Gamma GLM with
a log link.  Gamma already has support over ``(0, infinity)``, so no further
truncation normalization is required.
"""

import numpy as np
import pandas as pd
from scipy.special import expit

from kanly.distributional_models import GammaHurdle


def main():
    """Generate zero-or-Gamma outcomes, fit both GLMs, and verify recovery."""
    rng = np.random.default_rng(20260802)
    nobs = 15_000

    # Positive equation: log(E[Y | Y > 0]) = beta_0 + beta_x * x.
    beta = np.array([0.30, -0.35])
    x = rng.normal(size=nobs)
    positive_mean = np.exp(beta[0] + beta[1] * x)
    shape = 4.0
    true_scale = 1.0 / shape

    # Zero hurdle: logit(P(Y = 0)) = gamma_0 + gamma_z * z.
    gamma = np.array([-0.50, 0.60])
    z = rng.normal(size=nobs)
    zero_probability = expit(gamma[0] + gamma[1] * z)
    is_zero = rng.random(nobs) < zero_probability

    y = rng.gamma(shape=shape, scale=positive_mean / shape)
    y[is_zero] = 0.0
    data = pd.DataFrame({'y': y, 'x': x, 'z': z})

    model = GammaHurdle.build_model_from_formula(
        'y ~ x',
        data,
        exog_infl='z',
    )
    fit = model.fit(
        start_params=np.zeros(len(model.param_names)),
        cov_type='SANDWICH',
    )

    # Gamma scale is estimated by the positive GLM and is not an additional
    # regression coefficient in the combined parameter vector.
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
    print(f'Positive family/link: {fit.positive_fit.family.name()} / '
          f'{fit.positive_fit.link.name()}')
    print(f'Hurdle family/link: {fit.hurdle_fit.family.name()} / '
          f'{fit.hurdle_fit.link.name()}')
    print(f'Gamma scale: true={true_scale:.4f}, '
          f'estimated={fit.positive_scale:.4f}')
    print(f'Covariance: {fit.cov_type} '
          f'(component GLM type: {fit.component_cov_type})')
    print('\nCombined model summary:')
    print(fit.summary_df())
    print('\nParameter recovery:')
    print(comparison)
    print('\nMaximum absolute cross-block covariance: '
          f'{max_cross_covariance:.1e}')

    if not fit.converged:
        raise RuntimeError(f'Gamma hurdle fit failed: {fit.message}')

    max_absolute_error = np.max(np.abs(comparison['error']))
    scale_error = abs(fit.positive_scale - true_scale)
    print(f'Maximum absolute parameter error: {max_absolute_error:.4f}')
    print(f'Absolute Gamma scale error: {scale_error:.4f}')
    assert max_absolute_error < 0.08
    assert scale_error < 0.04
    assert max_cross_covariance == 0.0
    assert np.all(y[y > 0] > 0.0)


if __name__ == '__main__':
    main()
