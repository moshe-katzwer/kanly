import numpy as np
import pandas as pd

from kanly.api import blm, elastic_net, lm, compare_results, build_data_model

n = 50
np.random.seed(0)
x = np.random.randn(n)
y = 3 + 10 * x + np.random.randn(n) * 4.5
wts = np.random.rand(n) * 20
wts *= n / np.sum(wts)  # NOTE! Important that weights sum to 1 when comparing to Elastic Net!
data = {'x': x, 'y': y, 'wts': wts}

# Fit a model where our prior is that
#   sigma2 has pdf 1/sigma2
#   beta|sigma2 ~ N([0,5], sigma2 * diag([0,50])^-1)
fit_blm = blm('y ~ x $ wts', data, mu0=[0, 5],
              Lambda0=[0, 50], a0=0, b0=0, specification_name='example')

fit_lm = lm('y ~ x $ wts', data, cov_type='nonrobust')
fit_en = elastic_net('y ~ x $ wts', data, l1_ratio=0, alpha=50 / n, normalize=False,
                     regularize_to_values={'x': 5})

fit_mcmc = build_data_model(
    data_code_block='self.wts = `wts`; self.x = `x`; self.y = `y`;',
    model_code_block='return logpdf_norm($x$, 5, ($__sigma2$ / 50)**.5)'  # prior on beta
                     '+ -np.log($__sigma2$)'  # prior on sigma2
                     '+ logpdf_norm(y, $x$*x + $Intercept$, np.sqrt($__sigma2$ / wts)).sum()',  # likelihood
    data=data
).to_bayesian_model().amha(start_params={'__sigma2': 5}, n_samples=135_000, n_burnin=5_000, thinning=4)

print(compare_results([fit_lm, fit_blm, fit_en, fit_mcmc]))
