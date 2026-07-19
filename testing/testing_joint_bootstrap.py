from kanly.api import lm, gmm_iv_nonlinear, glm, get_joint_bootstrapped_distribution
import pandas as pd
import numpy as np
from numpy.testing import assert_array_almost_equal

nobs = 50_000

df = pd.DataFrame(columns=['z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7'], data=np.random.randn(nobs, 7))
df[['x1', 'x2', 'x3', 'x4', 'x5']] = df[['z1', 'z2', 'z3', 'z4', 'z5']] + .5 * np.random.randn(nobs, 5)
df['y1'] = df[['x1', 'x2', 'x3', 'x4', 'x5']].dot([1, 2, 3, 0, 0]) + np.random.randn(nobs) * np.abs(df.x1)
df['y2'] = df[['x1', 'x2', 'x3', 'x4', 'x5']].dot([3, 3, 0, 0, 2]) + np.random.randn(nobs) * np.abs(df.x1)
df['y3'] = df[['x1', 'x2', 'x3', 'x4', 'x5']].dot([3, .5, -1, 1, 0]) + .5 * np.random.randn(nobs) * np.exp(df.x2)

df['w'] = np.exp(np.random.rand(nobs))
df['g'] = np.random.randint(0, 14, nobs).astype(int)

outcomes = ['y1', 'y2', 'y3']

exog_str = 'x1 + x2'
exogs = exog_str.split('+')
iv_str = 'z1 + z2'

cov_kwds = {'seed': 71_843, 'n_samples': 50}

## Bootstrap results the same for lm and gmm_iv
fit_lm = lm(f'y1 ~ {exog_str} | {iv_str}', df, cov_type='bootstrap', cov_kwds=cov_kwds)
fit_gmm = gmm_iv_nonlinear('[y1] ~ {a} + ' + " + ".join(['{b%d}*[%s]' % (i, x) for i, x in enumerate(exogs)]),
                           iv_str,
                           df,
                           cov_type='bootstrap', cov_kwds=cov_kwds)

V = get_joint_bootstrapped_distribution([fit_lm, fit_gmm])
for k1 in range(2):
    for k2 in range(2):
        assert_array_almost_equal(V.values[k1 * 3:(k1 + 1) * 3, k2 * 3:(k2 + 1) * 3],
                                  V.values[:3, :3])

## Bootstrap results the same for lm and glm gaussian
fit_lm = lm(f'y1 ~ {exog_str}', df, cov_type='bootstrap', cov_kwds=cov_kwds)
fit_glm = glm(f'y1 ~ {exog_str}', df, cov_type='bootstrap', cov_kwds=cov_kwds)

V = get_joint_bootstrapped_distribution([fit_lm, fit_glm])
for k1 in range(2):
    for k2 in range(2):
        assert_array_almost_equal(V.values[k1 * 3:(k1 + 1) * 3, k2 * 3:(k2 + 1) * 3],
                                  V.values[:3, :3])
