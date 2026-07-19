from __future__ import absolute_import, print_function

import pandas as pd
import numpy as np

import statsmodels.formula.api as smf
from numpy.testing import assert_array_almost_equal

from kanly.api import lm, compare_results

np.random.seed(0)
n = 26
df = pd.DataFrame({'x': np.random.randint(0, 4, n), 'z': np.arange(n),
                   'w': np.random.randn(n), 'grp1': np.random.randint(0, 40, n),
                   'grp2': np.random.randint(0, 2, n).astype(str),
                   'wtsvar': .5 + np.random.rand(n)},
                  )
e = [0]
for j in range(1, n):
    e.append(.992 * e[-1] + .2 * np.random.randn())

print(e)

df['y'] = 15.3 - 2.5 * df.x + 3 * (df.grp2 == 1) + e

fit1 = lm('y ~ x', df)
fit2 = lm('y ~ x', df, cov_type='HAC', cov_kwds={'max_lags': 3})
fit3 = lm('y ~ x', df, cov_type='HAC-PANEL', cov_kwds={'max_lags': 3, 'groups': 'grp2'})

print(compare_results([fit1, fit2, fit3]))