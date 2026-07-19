from __future__ import absolute_import, print_function

import numpy as np
import pandas as pd
from kanly.api import lm, gmm_iv_linear, gmm, gmm_iv_nonlinear
from scipy.sparse import diags

n = 5000
np.random.seed(0)

integer_index = np.random.choice(range(n), 3 * n // 5, replace=False)
e = np.random.randn(n)
w = np.exp(np.random.randn(n))
z1 = np.random.randn(n) + 4
z2 = np.random.randn(n)
z3 = 6 * z2 + np.random.randn(n)
wts = np.exp(2 * np.random.randn(n)) / 10
x = 1.6 + .8 * z1 - .4 * z2 ** 3 + .0001 * z3 + .7 * e + .15 * w
y = 10 + 3 * x - .225 * w + 1.6 * e
df = pd.DataFrame({'x': x, 'y': y, 'z1': z1, 'z2': z2, 'z3': z3, 'w': w, 'wts': wts})

n_samples = 10

exogs = ['x', 'w']

for instrs in [['z1', 'w'], ['z1', 'z2', 'z3', 'w']]:

    for cov_type in ['sandwich', 'bootstrap']:

        formula = f'y ~ {"+".join(exogs)} | {"+".join(instrs)} $ wts'
        fit_gmm_l = gmm_iv_linear(formula, df, index=integer_index,
                                  method='iterative',
                                  ftol=1e-15, gtol=1e-15, xtol=1e-16,
                                  cov_type=cov_type, cov_kwds={'n_samples': n_samples, 'seed': 1})
        # print(fit_gmm_l)

        resid = f'[y] ~ {{Intercept}} + {"+".join([f"{{{x}}}*[{x}]" for x in exogs])} $ [wts]'
        fit_gmm_nl = gmm_iv_nonlinear(resid, ' + '.join(instrs),
                                      df, index=integer_index, cov_type=cov_type,
                                      method='iterative',
                                      ftol=1e-15, gtol=1e-15, xtol=1e-16,
                                      cov_kwds={'n_samples': n_samples, 'seed': 1})
        # print(fit_gmm_nl)

        resid = f'([y]-({{Intercept}} + {"+".join([f"{{{x}}}*[{x}]" for x in exogs])})) * [wts]'

        fit_gmm = gmm(
            [resid] + [(resid, f'[{z}]') for z in instrs],
            df,
            ftol=1e-15, gtol=1e-15, xtol=1e-16,
            index=integer_index,
            method='iterative',
            cov_type=cov_type, cov_kwds={'n_samples': n_samples, 'seed': 1}
        )
        # print(fit_gmm)

        fit_lm = lm(formula, df, cov_type='hc1' if cov_type == 'sandwich' else 'bootstrap',
                    cov_kwds=dict() if cov_type == 'sandwich' else {'n_samples': n_samples, 'seed': 1},
                    index=integer_index)
        # print(fit_lm)

        print('\n\n** ', instrs, cov_type)

        params = pd.DataFrame({'lm': fit_lm.params, 'gmm': fit_gmm.params,
                               'gmm_iv_l': fit_gmm_l.params, 'gmm_iv_nl': fit_gmm_nl.params})

        print(params)
        bse = pd.DataFrame({'lm': fit_lm.bse, 'gmm': fit_gmm.bse,
                            'gmm_iv_l': fit_gmm_l.bse, 'gmm_iv_nl': fit_gmm_nl.bse})

        print(bse)

        if len(exogs) == len(instrs):
            assert np.all(params.std(axis=1) < 1e-4)
            assert np.all(bse.std(axis=1) < 1e-4)

        # Test that 2SLS is the same as 1-step GMM with weighting matrix W = inv(Z' W Z)
        # Z = fit_lm.model.instruments
        # W = np.linalg.inv(Z.transpose().dot(diags(fit_lm.model.weights)).dot(Z).toarray() / n)
        fit_gmm_2sls = gmm_iv_linear(formula, df,
                                     do_2sls=True,
                                     # W=W, method='one_step',
                                     ftol=1e-25, gtol=1e-25, xtol=1e-25,
                                     index=integer_index)

        assert np.all(np.abs(fit_gmm_2sls.params - fit_lm.params) < 1e-4)
