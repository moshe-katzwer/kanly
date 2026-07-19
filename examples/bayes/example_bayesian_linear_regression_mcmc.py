import numpy as np

from kanly.api import lm, bayes_lm_model

n = 140
np.random.seed(0)
x = np.random.randn(n)
g = np.random.randint(0, 12, size=140)
y = 3 + 10 * x + np.random.randn(n) * 4.5
data = {'x': x, 'y': y, 'g': g, 'wts': np.random.rand(n) * 20}

fit_lm = lm('y ~ x + C(g) $ wts', data, cov_type='nonrobust')

fit_mcmc = bayes_lm_model('y ~ x + C(g) $ wts', data).sample(
    [0.] * 13 + [1.],
    debug=False, n_burnin=1_000, n_samples=10_000,
    max_subchain_draws_sample=5_000,
    do_mala_cd_warmup=False, n_chains=4,
    thinning=2,
    do_diff_evolution_mc=True)

print(fit_lm)
print(fit_mcmc)
