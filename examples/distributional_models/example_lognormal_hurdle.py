"""Simulate and recover a lognormal hurdle model.

The hurdle equation models the probability of zero with a Bernoulli logit.
Conditional on a positive outcome, ``log(Y)`` follows a Gaussian linear model.
The positive arithmetic mean therefore includes the fitted log-variance
correction. Formula designs and ``score_obs`` are kept sparse in this example.
"""

import numpy as np
import pandas as pd
from scipy.sparse import isspmatrix
from scipy.special import expit

from kanly.distributional_models import LognormalHurdle


def main():
    """Generate zero-or-lognormal outcomes and fit both components."""
    rng = np.random.default_rng(20260804)
    nobs = 15_000

    # Positive equation: E[log(Y) | Y > 0] = beta_0 + beta_x * x.
    beta = np.array([0.35, -0.3])
    true_scale = 0.36
    x = rng.normal(size=nobs)
    log_y = beta[0] + beta[1] * x
    positive_y = np.exp(
        log_y + np.sqrt(true_scale) * rng.normal(size=nobs)
    )

    # Zero hurdle: logit(P(Y = 0)) = gamma_0 + gamma_z * z.
    gamma = np.array([-0.55, 0.6])
    z = rng.normal(size=nobs)
    zero_probability = expit(gamma[0] + gamma[1] * z)
    is_zero = rng.random(nobs) < zero_probability

    y = positive_y
    y[is_zero] = 0.0
    data = pd.DataFrame({'y': y, 'x': x, 'z': z})

    model = LognormalHurdle.build_model_from_formula(
        'y ~ x',
        data,
        exog_infl='z',
        dense_threshold_mb=0,
    )
    fit = model.fit(cov_type='SANDWICH')

    # Parameter order is log-scale beta, log residual variance, then the
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
    fitted = pd.DataFrame({
        'mean_given_positive': fit.predict(which='positive_mean'),
        'overall_mean': fit.predict(which='mean'),
        'zero_probability': fit.predict(which='zero_probability'),
    })

    print(f'Observed zero fraction: {np.mean(y == 0):.3f}')
    print(f'Mean simulated zero probability: {zero_probability.mean():.3f}')
    print(f'Lognormal scale: true={true_scale:.4f}, '
          f'estimated={fit.positive_scale:.4f}')
    print(f'Sparse main design: {isspmatrix(model.exog)}')
    print(f'Sparse hurdle design: {isspmatrix(model.exog_infl)}')
    print(f'Sparse score_obs: {isspmatrix(fit.score_obs())}')
    print('\nParameter recovery:')
    print(comparison)
    print('\nFirst five fitted quantities:')
    print(fitted.head())

    if not fit.converged:
        raise RuntimeError(f'Lognormal hurdle fit failed: {fit.message}')

    assert np.max(np.abs(comparison['error'])) < 0.08
    assert abs(fit.positive_scale - true_scale) < 0.04
    assert not fit.is_quasi_likelihood
    assert fit.aic is not None and fit.bic is not None
    assert fit.inference_valid
    assert isspmatrix(fit.score_obs())
    assert np.all(fitted['overall_mean'] <= fitted['mean_given_positive'])


if __name__ == '__main__':
    main()
