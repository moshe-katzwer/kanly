"""Simulate and recover a Gaussian hurdle model.

The hurdle equation is a Bernoulli/logit model for the probability of zero.
Conditional on a positive response, the second component is an OLS Gaussian
working model. Because an untruncated Gaussian is not confined to positive
values, the combined fit is reported as quasi-likelihood.

This example keeps both formula designs sparse with ``dense_threshold_mb=0``
and demonstrates that the fitted observation-score Jacobian is sparse too.
"""

import numpy as np
import pandas as pd
from scipy.sparse import isspmatrix
from scipy.special import expit

from kanly.distributional_models import GaussianHurdle


def main():
    """Generate zero-or-Gaussian outcomes and fit the two components."""
    rng = np.random.default_rng(20260803)
    nobs = 15_000

    # Positive equation: E[Y | Y > 0] = beta_0 + beta_x * x. The bounded x
    # and well-separated intercept make the simulated Gaussian draws positive.
    beta = np.array([3.0, 0.4])
    x = rng.uniform(-1.0, 1.0, size=nobs)
    positive_mean = beta[0] + beta[1] * x
    true_scale = 0.25
    positive_y = positive_mean + np.sqrt(true_scale) * rng.normal(size=nobs)
    if np.any(positive_y <= 0.0):
        raise RuntimeError('The seeded Gaussian sample unexpectedly crossed zero')

    # Zero hurdle: logit(P(Y = 0)) = gamma_0 + gamma_z * z.
    gamma = np.array([-0.6, 0.55])
    z = rng.normal(size=nobs)
    zero_probability = expit(gamma[0] + gamma[1] * z)
    is_zero = rng.random(nobs) < zero_probability

    y = positive_y
    y[is_zero] = 0.0
    data = pd.DataFrame({'y': y, 'x': x, 'z': z})

    model = GaussianHurdle.build_model_from_formula(
        'y ~ x',
        data,
        exog_infl='z',
        dense_threshold_mb=0,
    )
    fit = model.fit(cov_type='SANDWICH')

    # Parameter order is positive beta, log residual variance, then the
    # zero-hurdle gamma coefficients.
    true_params = np.concatenate((beta, [np.log(true_scale)], gamma))
    comparison = pd.DataFrame(
        {
            'true': true_params,
            'estimated': fit.params,
            'std err': fit.bse,
            'error': fit.params - true_params,
        },
        index=fit.param_names,
    )
    fitted_mean = fit.predict(which='mean')
    fitted_zero_probability = fit.predict(which='zero_probability')

    print(f'Observed zero fraction: {np.mean(y == 0):.3f}')
    print(f'Mean simulated zero probability: {zero_probability.mean():.3f}')
    print(f'Gaussian scale: true={true_scale:.4f}, '
          f'estimated={fit.positive_scale:.4f}')
    print(f'Sparse main design: {isspmatrix(model.exog)}')
    print(f'Sparse hurdle design: {isspmatrix(model.exog_infl)}')
    print(f'Sparse score_obs: {isspmatrix(fit.score_obs())}')
    print('\nCombined model summary:')
    print(fit.summary_df())
    print('\nParameter recovery:')
    print(comparison)
    print(f'\nMean fitted outcome: {np.mean(fitted_mean):.3f}')
    print('Mean fitted zero probability: '
          f'{np.mean(fitted_zero_probability):.3f}')

    if not fit.converged:
        raise RuntimeError(f'Gaussian hurdle fit failed: {fit.message}')

    max_absolute_error = np.max(np.abs(comparison['error']))
    print(f'Maximum absolute parameter error: {max_absolute_error:.4f}')
    assert max_absolute_error < 0.08
    assert fit.is_quasi_likelihood
    assert fit.aic is None and fit.bic is None
    assert fit.inference_valid
    assert isspmatrix(model.exog)
    assert isspmatrix(model.exog_infl)
    assert isspmatrix(fit.score_obs())


if __name__ == '__main__':
    main()
