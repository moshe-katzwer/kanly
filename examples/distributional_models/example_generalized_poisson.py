"""Simulate and recover a generalized-Poisson regression.

This example uses the GP-2 parameterization, whose conditional variance is
``mu * (1 + alpha * mu)**2``. Positive ``alpha`` produces overdispersion. The
formula design and fitted observation-score Jacobian are kept sparse with
``dense_threshold_mb=0``.
"""

import numpy as np
import pandas as pd
from scipy.sparse import isspmatrix
from scipy.special import gammaln

from kanly.distributional_models import GeneralizedPoisson


def sample_generalized_poisson(rng, mean, alpha, p):
    """Draw GP-P counts by vectorized inversion of the model PMF."""
    mean = np.asarray(mean, dtype=float)
    mean_power = mean ** (p - 1.0)
    denominator = 1.0 + alpha * mean_power
    uniforms = rng.random(len(mean))

    counts = np.zeros(len(mean), dtype=int)
    cumulative = np.exp(-mean / denominator)
    unresolved = uniforms > cumulative
    count = 1
    while np.any(unresolved):
        a2 = mean + alpha * mean_power * count
        log_probability = (
            np.log(mean)
            + (count - 1.0) * np.log(a2)
            - count * np.log(denominator)
            - gammaln(count + 1.0)
            - a2 / denominator
        )
        cumulative += np.exp(log_probability)
        newly_resolved = unresolved & (uniforms <= cumulative)
        counts[newly_resolved] = count
        unresolved &= ~newly_resolved
        count += 1
        if count > 10_000:
            raise RuntimeError('Generalized-Poisson inversion did not finish')
    return counts


def main():
    """Generate GP-2 counts, fit the likelihood, and show recovery."""
    rng = np.random.default_rng(20260805)
    nobs = 20_000
    p = 2

    beta = np.array([0.25, 0.35])
    alpha = 0.18
    x = rng.normal(size=nobs)
    mean = np.exp(beta[0] + beta[1] * x)
    y = sample_generalized_poisson(rng, mean, alpha=alpha, p=p)
    data = pd.DataFrame({'y': y, 'x': x})

    model = GeneralizedPoisson.build_model_from_formula(
        'y ~ x',
        data,
        p=p,
        dense_threshold_mb=0,
    )
    fit = model.fit(cov_type='SANDWICH')

    true_params = np.concatenate((beta, [alpha]))
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

    print(f'Generalized-Poisson choice p: {model.p}')
    print(f'Observed count mean: {np.mean(y):.3f}')
    print(f'Mean fitted conditional mean: {np.mean(fitted_mean):.3f}')
    print(f'Sparse design: {isspmatrix(model.exog)}')
    print(f'Sparse score_obs: {isspmatrix(fit.score_obs())}')
    print('\nParameter recovery:')
    print(comparison)

    if not fit.converged:
        raise RuntimeError(f'Generalized-Poisson fit failed: {fit.message}')

    assert np.max(np.abs(comparison['error'])) < 0.06
    assert model.p == p
    assert fit.inference_valid
    assert isspmatrix(model.exog)
    assert isspmatrix(fit.score_obs())
    assert np.all(fitted_mean > 0.0)


if __name__ == '__main__':
    main()
