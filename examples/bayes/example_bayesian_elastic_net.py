import pandas as pd

from kanly.api import elastic_net, bayes_lm_model
import numpy as np


np.random.seed(0)
n = 150
x = 1.56 * np.random.randn(n)
z = np.random.rand(n)
y = 3 + 10 * x - 2 * z + np.random.randn(n) * 3
wts = .01 + np.random.rand(n)
data = {'x': x, 'y': y, 'z': z, 'wts': wts}

alpha, l1_ratio, normalize = 1, .9, True

fit_en = elastic_net('y ~ x + z', data, alpha=alpha, l1_ratio=l1_ratio, normalize=normalize)
print(fit_en)

bmodel = bayes_lm_model('y ~ x + z', data)
bmodel.set_priors({'': bmodel.get_elastic_net_log_prior(alpha=alpha, l1_ratio=l1_ratio, normalize=normalize)})
fit_mcmc = bmodel.amha([0., 0, 0, 1], n_samples=150_000, n_chains=6, max_subchain_draws_sample=30_000, seed=2,
                       thinning=3)
print(fit_mcmc)

map_estimate = pd.Series(bmodel.maximize_posterior([0., 0, 0, 1])['params'], index=bmodel.param_names)

print(pd.DataFrame({
    'frequentist mode (mcmc)': fit_en.params,
    'bayesian mean (mcmc)': fit_mcmc.params,
    'bayesian max posterior (mcmc)': fit_mcmc.map_params,
    'bayesian max posterior (bfgs)': map_estimate,
}).round(4).to_string())

fit_mcmc.diagnostic_plot('x', show=True)
