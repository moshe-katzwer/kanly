"""Simulate and recover an NB2 hurdle model with estimated dispersion.

The zero equation is a Bernoulli logit for the probability of an observed
zero. Positive counts follow an NB2 distribution conditional on being
positive. The NB2 regression mean is the mean of the underlying distribution
before its zeros are truncated, so it differs from ``E[Y | Y > 0]``.
"""

import numpy as np
import pandas as pd
from scipy.special import expit

from kanly.distributional_models import NegativeBinomialPHurdle


def sample_zero_truncated_nb_p(rng, mean, alpha, p):
    """Draw NB-P counts repeatedly until every observation is positive."""
    size = mean ** (2 - p) / alpha
    probability = size / (size + mean)
    counts = rng.negative_binomial(size, probability)
    is_zero = counts == 0
    while np.any(is_zero):
        counts[is_zero] = rng.negative_binomial(
            size[is_zero], probability[is_zero]
        )
        is_zero = counts == 0
    return counts


def main():
    """Generate hurdle counts, fit the separate components, and show recovery."""
    rng = np.random.default_rng(20260801)
    nobs = 15_000
    p = 2

    # Positive component:
    #   log(mu) = beta_0 + beta_x*x
    # Here mu is the mean of the untruncated NB2 distribution, whose variance
    # is mu + alpha*mu**2. It is not the positive conditional mean.
    beta = np.array([0.30, -0.35])
    alpha = 0.65
    x = rng.normal(size=nobs)
    underlying_mean = np.exp(beta[0] + beta[1] * x)
    y = sample_zero_truncated_nb_p(
        rng, underlying_mean, alpha=alpha, p=p
    )

    # Zero hurdle:
    #   logit(P(Y=0)) = gamma_0 + gamma_z*z.
    gamma = np.array([-0.60, 0.55])
    z = rng.normal(size=nobs)
    zero_probability = expit(gamma[0] + gamma[1] * z)
    y[rng.random(nobs) < zero_probability] = 0
    data = pd.DataFrame({'y': y, 'x': x, 'z': z})

    model = NegativeBinomialPHurdle.build_model_from_formula(
        'y ~ x',
        data,
        exog_infl='z',
        p=p,
    )
    fit = model.fit(cov_type='NONROBUST')

    # Exact order: beta, log(alpha), then the logit-hurdle gamma parameters.
    true_params = np.r_[beta, np.log(alpha), gamma]
    comparison = pd.DataFrame(
        {
            'true': true_params,
            'estimated': fit.params,
            'std err': fit.bse,
            'error': fit.params - true_params,
        },
        index=fit.param_names,
    )

    print(f'NB-P choice p: {fit.negative_binomial_p}')
    print(f'Estimated alpha=exp(log_alpha): {fit.dispersion:.4f}')
    print(f'Observed zero fraction: {np.mean(y == 0):.3f}')
    print('\nParameter recovery:')
    print(comparison)

    predictions = pd.DataFrame(
        {
            # mu from the hypothetical untruncated NB2 distribution.
            'underlying_mean': fit.predict(which='underlying_mean'),
            # mu divided by one minus the NB2 probability of zero.
            'mean_given_positive': fit.predict(which='positive_mean'),
            # (1 - hurdle zero probability) times mean_given_positive.
            'overall_mean': fit.predict(which='mean'),
            'hurdle_zero_probability': fit.predict(
                which='zero_probability'
            ),
        }
    )
    print('\nFirst five fitted quantities:')
    print(predictions.head())

    if not fit.converged:
        raise RuntimeError(f'NB-P hurdle fit failed: {fit.message}')
    assert np.max(np.abs(comparison['error'])) < 0.08
    assert abs(fit.dispersion - alpha) < 0.12
    assert np.all(
        predictions['mean_given_positive']
        > predictions['underlying_mean']
    )


if __name__ == '__main__':
    main()
