"""
Testing script for `NLLS` function
"""

from numpy.testing import assert_allclose
import numpy as np
import pandas as pd

from kanly.api import lm, en, nlls, compare_results, nlls_minimize_internal, nlls_en, NLLS_EN, NLLS

n = 375
np.random.seed(0)
df = pd.DataFrame({
    'x': np.random.randn(n),
    'w': np.random.randint(1, 4, n),
    'grp': np.random.randint(0, 12, n),
})
df['y'] = 1.2 - 0.3 * df['x'] + .6 * np.random.randn(n) + .15 * (df.grp == 2)


def pred_func(a):
    return np.asarray(a[0] + a[1] * df.x + a[2] * df.x ** 2)


num_failures = 0

# ### #
# OLS #
# ### #


for cov_type, (cov_kwds, cov_kwds_NLLS) in zip(
        [
            'OLS_SMALL', 'NONROBUST', 'HC1', 'CLUSTER',
            'BOOTSTRAP'
        ],
        [
            (None,None), (None,None), (None,None), ({'groups': 'grp'}, {'groups': df.grp}),
            ({'n_samples': 100, 'seed': 10}, {'n_samples': 100, 'seed': 10})
        ],
):
    for is_weighted in [True, False]:
        formula_lm = 'y ~ poly(x,2)' + (' $ w' if is_weighted else '')
        formula_nlls = '[y] ~ {Intercept} + {x}*[x] + {x_sq}*[x**2]' + (' $ [w]' if is_weighted else '')

        fit_lm = lm(formula_lm, df, cov_type=cov_type, cov_kwds=cov_kwds)
        fit_nlls = nlls(formula_nlls, df, cov_type=cov_type, cov_kwds=cov_kwds)

        fit_NLLS = NLLS(df.y, pred_func, param_names=['Intercept', 'x', 'x_sq'],
                        weights=df.w if is_weighted else None,
                        cov_type=cov_type,
                        cov_kwds=cov_kwds_NLLS)

        print('='*100)
        try:
            for f in [fit_lm, fit_nlls, fit_NLLS]:
                assert_allclose(f.params, fit_lm.params)
                assert_allclose(f.bse, fit_lm.bse)
                assert_allclose(f.rsquared, fit_lm.rsquared)
                assert_allclose(f.rsquared_adj, fit_lm.rsquared_adj)
                assert_allclose(f.llf, fit_lm.llf)
            print(f"wtd={is_weighted}, cov_type={cov_type} PASSED!")
        except:
            print(f"wtd={is_weighted}, cov_type={cov_type} FAILED!")
            print(compare_results([fit_lm, fit_nlls, fit_NLLS]))
            num_failures += 1


# ########### #
# ELASTIC NET #
# ########### #

for alpha in [0, .05, .8]:
    for is_weighted in [True, False]:
        formula_lm = 'y ~ poly(x,2)' + (' $ w' if is_weighted else '')
        formula_nlls = '[y] ~ {Intercept} + {x}*[x] + {x_sq}*[x**2]' + (' $ [w]' if is_weighted else '')

        fit_lm = en(formula_lm, df, alpha={'x': alpha}, l1_ratio=.8, normalize=False, regularize_to_values={'x': 3},
                    xtol=1e-8, gtol=1e-8, ftol=1e-8, max_iter=1000,

                    )
        fit_nlls = nlls_en(formula_nlls, df, alpha={'x': alpha}, l1_ratio=.8, normalize=False,
                           regularize_to_values={'x': 3}, xtol=1e-8, gtol=1e-8, ftol=1e-8,
                           jac_method='mid',
                           max_iter=1000)
        fit_NLLS = NLLS_EN(df.y, pred_func, param_names=['Intercept', 'x', 'x_sq'],
                           alpha=[0, alpha, 0], l1_ratio=.8, regularize_to_values=[0, 3, 0],
                           xtol=1e-10, gtol=1e-10, ftol=1e-10,
                           max_iter=1000,
                           jac_method='mid',
                           weights=df.w if is_weighted else None)

        try:
            for f in [fit_lm, fit_nlls, fit_NLLS]:
                assert_allclose(f.params, fit_lm.params, rtol=1e-3, atol=1e-4)
        except Exception as e:
            print(f"alpha={alpha}")
            print(compare_results([fit_lm, fit_nlls, fit_NLLS]))
            num_failures += 1

if num_failures:
    print(f'NUM FAILURES = {num_failures}!')
else:
    print("ALL PASSED!")