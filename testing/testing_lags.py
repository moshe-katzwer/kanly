import numpy as np
import pandas as pd

from kanly.api import lm, compare_results
from numpy.testing import assert_array_almost_equal

n = 415
np.random.seed(0)
df = pd.DataFrame({
    'x': np.random.randn(n),
    'grp': np.random.randint(0, 15, n),
})

df['Lx'] = df.x.shift(1)

# Shift 2 back by group
for g in df.grp.unique():
    idx_g = df.grp == g
    df.loc[idx_g, 'L2x'] = df['x'][idx_g].shift(2)

df['y1'] = (1.2 - 0.3 * df['x'] + .2 * np.random.randn(n))

df['y2'] = (1.2 - 0.3 * df['x'] - .2 * df['Lx'] + .2 * np.random.randn(n))

form1_hardcode, form1_patsy = 'y1 ~ x', 'y1 ~ x'
form2_hardcode, form2_patsy = 'y1 ~ x + Lx + L2x', 'y1 ~ x + L(x) + L(x,2,grp)'

sub_idx = np.random.choice(np.arange(n), 214, replace=False)
sub_idx_bool = np.array([False] * n)
sub_idx_bool[sub_idx] = True

for is_lag, form_hc, form_patsy in [(False, form1_hardcode, form1_patsy), (True, form2_hardcode, form2_patsy)]:

    print('-' * 50 + f"\nis_lag = {is_lag}" + '\n')

    for name, indices in {
        'Full': [None, df.index, [True] * n, list(df.index)],
        'Sub': [sub_idx, sub_idx_bool, df.index[sub_idx]]
    }.items():

        params = []
        bses = []

        print('\n' * 3, name, '\n')
        for idx in indices:
            fit = lm(form_hc, df, debug=False, index=idx)
            fit2 = lm(form_patsy, df, debug=False, index=idx)

            assert_array_almost_equal(fit.params, fit2.params)
            assert_array_almost_equal(fit.bse, fit2.bse)

            params += [fit.params.values, fit2.params.values]
            bses += [fit.bse.values, fit2.bse.values]
            print('\n', pd.DataFrame(fit.params).transpose(), "\n", pd.DataFrame(fit2.params).transpose())

        assert_array_almost_equal(pd.DataFrame(params).var(axis=0), [0] * len(fit.params))
        assert_array_almost_equal(pd.DataFrame(bses).var(axis=0), [0] * len(fit.params))

# test regardless of data ordering
M = 3
T = 15
np.random.seed(0)
y = np.random.randn(M * T)
e = np.random.randn(M * T)
for i in range(1, M * T):
    y[i] = .7 * y[i - 1] + e[i]
df1 = pd.DataFrame({
    'y': y,
    't': np.tile(range(T), M),
    'g': np.repeat(range(M), T),
})

df1.loc[10, 'y'] = np.nan

df1 = df1.sort_values(by=['g', 't'])
df2 = df1.sort_values(by=['t', 'g'])
df3 = df1.sort_values(by=['t', 'g']).reset_index(drop=True)

fit1 = lm('y ~ L(y, 1, g)', df1)
fit2 = lm('y ~ L(y, 1, g)', df2)
fit3 = lm('y ~ L(y, 1, g)', df3)

for f in [fit2, fit3]:
    assert_array_almost_equal(fit1.params, f.params)
    assert_array_almost_equal(fit1.bse, f.bse)

print(fit1)
print(fit2)
print(fit3)
