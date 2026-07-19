import itertools
import time

import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal, assert_almost_equal
from kanly.api import get_joint_bootstrapped_distribution

from kanly.api import lm, sure

np.random.seed(0)

nobs = 50_000

df = pd.DataFrame(columns=['z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7'], data=np.random.randn(nobs, 7))
df[['x1', 'x2', 'x3', 'x4', 'x5']] = df[['z1', 'z2', 'z3', 'z4', 'z5']] + .5 * np.random.randn(nobs, 5)
df['y1'] = df[['x1', 'x2', 'x3', 'x4', 'x5']].dot([1, 2, 3, 0, 0]) + np.random.randn(nobs) * np.abs(df.x1)
df['y2'] = df[['x1', 'x2', 'x3', 'x4', 'x5']].dot([3, 3, 0, 0, 2]) + np.random.randn(nobs) * np.abs(df.x1)
df['y3'] = df[['x1', 'x2', 'x3', 'x4', 'x5']].dot([3, .5, -1, 1, 0]) + .5 * np.random.randn(nobs) * np.exp(df.x2)

df['w'] = np.exp(np.random.rand(nobs))
df['g'] = np.random.randint(0, 14, nobs).astype(int)

outcomes = ['y1', 'y2', 'y3']

exog_str = 'x1 + x2 + x3 + x4 + x5'

time_multi = 0.
time_indiv = 0.

configs = list(itertools.product(
    [True, False], [True, False], [True, False],
    [
        ('hc1', dict()),
        ('bootstrap', {'n_samples': 10, 'seed': 5}),
        ('cluster', {'groups': 'g'}),
    ]
))

for is_wtd, is_iv, is_absorb, (cov_type, cov_kwds) in configs:

    print(f'wtd={is_wtd}, iv={is_iv}, absorb={is_absorb}, cov={cov_type}...', end='')

    try:



        wt_str = ' $ w' if is_wtd else ''
        iv_str = ' | z1+z2+z3+z4+z5+z6+z7 ' if is_iv else ''
        absorb = 'g' if is_absorb else None

        formula_multi = f"{'+'.join(outcomes)} ~ {exog_str}{iv_str}{wt_str}"
        _t = time.time()
        result_multi = lm(formula_multi, df, cov_type=cov_type,
                          absorb=absorb, cov_kwds=cov_kwds)
        time_multi += time.time() - _t

        _t = time.time()
        results_individuals = {
            y: lm(f'{y} ~ {exog_str}{iv_str}{wt_str}', df,
                  cov_type=cov_type, absorb=absorb, cov_kwds=cov_kwds)
            for y in outcomes
        }
        time_indiv += time.time() - _t

        for y in outcomes:
            # print(compare_fits([result_multi[y], results_individuals[y]]))
            assert_array_almost_equal(result_multi[y].params, results_individuals[y].params)
            assert_array_almost_equal(result_multi[y].bse, results_individuals[y].bse)
            assert_array_almost_equal(result_multi[y].resid, results_individuals[y].resid)
            assert_array_almost_equal(result_multi[y].fittedvalues, results_individuals[y].fittedvalues)

            assert_almost_equal(result_multi[y].rsquared, results_individuals[y].rsquared)
            assert_almost_equal(result_multi[y].rsquared_adj, results_individuals[y].rsquared_adj)
            assert_almost_equal(result_multi[y].nobs, results_individuals[y].nobs)
            assert_almost_equal(result_multi[y].df_resid, results_individuals[y].df_resid)
            assert_almost_equal(result_multi[y].df_model, results_individuals[y].df_model)
            assert_almost_equal(result_multi[y].df_t_dist, results_individuals[y].df_t_dist)

            if is_absorb:
                assert_almost_equal(result_multi[y].absorb_info.rsquared_within,
                                    results_individuals[y].absorb_info.rsquared_within)
                assert_almost_equal(result_multi[y].absorb_info.rsquared_between,
                                    results_individuals[y].absorb_info.rsquared_between)
                assert_array_almost_equal(result_multi[y].absorb_info.absorbed_y_baselines,
                                          results_individuals[y].absorb_info.absorbed_y_baselines)

        print('passed!')

    except Exception as e:
        raise e
        print('failed!')

print(time_multi, time_indiv)


# BOOTSTRAPPING
np.random.seed(50)

nobs = 500

df = pd.DataFrame(columns=['z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7'], data=np.random.randn(nobs, 7))
df[['x1', 'x2', 'x3', 'x4', 'x5']] = df[['z1', 'z2', 'z3', 'z4', 'z5']] + .5 * np.random.randn(nobs, 5)
df['y1'] = df[['x1', 'x2', 'x3', 'x4', 'x5']].dot([1, 2, 3, 0, 0]) + np.random.randn(nobs) * np.abs(df.x1)
df['y2'] = df[['x1', 'x2', 'x3', 'x4', 'x5']].dot([3, 3, 0, 0, 2]) + np.random.randn(nobs) * np.abs(df.x1)
df['y3'] = df[['x1', 'x2', 'x3', 'x4', 'x5']].dot([3, .5, -1, 1, 0]) + .5 * np.random.randn(nobs) * np.exp(df.x2)

df['w'] = np.exp(np.random.rand(nobs))
df['g'] = np.random.randint(0, 5, nobs).astype(int)

outcomes = ['y1', 'y2', 'y3']

exog_str = 'x1 + x2 + x3 + x4 + x5'

fits = lm('y1 + y2 + y3 ~ ' + exog_str, df, debug=True, cov_type='bootstrap',
          cov_kwds={'groups': 'g', 'n_samples': 15, 'seed': 10})
V = get_joint_bootstrapped_distribution(fits)
print(V)

for j, y in enumerate(outcomes):
    f = lm(y + ' ~ ' + exog_str, df, debug=True, cov_type='bootstrap',
           cov_kwds={'groups': 'g', 'n_samples': 15, 'seed': 10})

    v1 = V.iloc[6 * j:(6 * (j + 1)), 6 * j:(6 * (j + 1))].values
    v2 = f._cov_params
    assert_array_almost_equal(v1, v2)
    print(v1 / v2)


# SURE COMPARISON
print("\n\nSURE COMPARISON")

for cov_type, cov_kwds in zip(
        ['hc1', 'cluster'],
        [dict(), {'groups': 'g'}]
):
    print(f'sure {cov_type}...', end='')
    fit_sure = sure([{'formula': f'{y} ~ {exog_str}', 'data': df} for y in outcomes],
                    cov_type=cov_type, cov_kwds=cov_kwds)
    print(fit_sure)

    try:
        for j, y in enumerate(outcomes):
            f = lm(y + ' ~ ' + exog_str, df, cov_type=cov_type, cov_kwds=cov_kwds)
            assert_array_almost_equal(fit_sure.params[6 * j:(6 * (j + 1))], f.params)
            assert_array_almost_equal(fit_sure.bse[6 * j:(6 * (j + 1))], f.bse, decimal=4)
        print('passed!')
    except Exception as e:
        print(f'failed! {e}')
