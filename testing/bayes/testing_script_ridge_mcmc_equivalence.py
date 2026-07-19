"""
Compares MCMC on a very simple model OLS with one regressor
and a normal prior, to ridge regression
"""

from kanly.bayes.bayesian_regression_model import BayesianNonlinearLeastSquaresModel
from kanly.api import nlls, blm, compare_results
import numpy as np
import pandas as pd

np.random.seed(0)
n = 1200
df = pd.DataFrame()
df['x'] = 1.5 + 2 * np.random.randn(n)
df['y'] = 1.6 * np.random.randn(n) + 1.3 * df['x']
df['w'] = np.exp(df.x / 2 + np.random.randn(n) - 2.)

mean_x, std_x = -5, .1

model = BayesianNonlinearLeastSquaresModel.build_model_from_formula(
    '[y] ~ {Intercept} + {x}*[x] $ [w]', df,
    priors={'x': f'norm({mean_x}, {std_x})',
            '__sigma2': lambda x: -np.log(x),
            },
)

fit_mcmc = model.amha([0., 0., 1], debug=False,
                      n_burnin=5_000,
                      n_samples=15_000)
print(fit_mcmc)

scale0 = fit_mcmc.mean_params['__sigma2']
scale1 = np.inf

fit_nlls = nlls(
    '[y] ~ {Intercept} + {x}*[x] $ [w]', df,
    regularize_to_values={'x': mean_x},
    l2_penalties={'x': scale0 / std_x ** 2},
    scale_l2_penalties=False,
    cov_type='nonrobust',
)
print(fit_nlls)

fit_blm = blm(
    'y ~ x $ w', df,
    mu0={'x': mean_x}, Lambda0={'x': scale0 / std_x ** 2},
    a0=0, b0=0,
)
print(fit_blm)

print(compare_results([fit_blm, fit_nlls, fit_mcmc]))