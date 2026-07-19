import pandas as pd
import numpy as np

from kanly.api import *
import matplotlib.pyplot as plt

from numpy.testing import assert_array_almost_equal

np.random.seed(0)
n = 500
z = np.random.randn(n)
x = .3 * z + 2.5 * np.random.randn(n)
w = np.exp(np.random.randn(n) + .15 * x)
p = 1.0 / (1.0 + np.exp(-(.4 + 1.2 * x + .05 * np.log(w))))

df = pd.DataFrame({'z': z, 'x': x, 'y': (np.random.rand(n) < p).astype(float),
                   'w': w,
                   'grp1': np.random.randint(0, 10, n),
                   'grp2': np.random.randint(0, 5, n),
                   'wts': .1 + np.random.rand(n)})

fit1 = lm(
    'y ~ x + w $ wts',
    df,
    absorb=('grp1', 'grp2'),
    debug=False
)

print(fit1)

for v in ['y', 'x', 'w']:
    df[v + '_abs'] = lm(v + ' ~ C(grp1)*C(grp2) $ wts', df).resid

print(lm('y ~ C(grp1)*C(grp2) $ wts', df))

print(lm('y_abs ~ x_abs + w_abs $ wts', df))