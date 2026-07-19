import itertools
import time

import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal, assert_almost_equal

from kanly.api import lm

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

fits = lm('y1 + y2 ~ ' + exog_str, df, debug=True, cov_type='bootstrap', cov_kwds={'groups': 'g'})
for f in fits.values():
    print(f)
    print(f.cov_kwds)
