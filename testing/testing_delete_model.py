import pandas as pd
import numpy as np

from kanly.api import *
import matplotlib.pyplot as plt

from numpy.testing import assert_array_almost_equal

np.random.seed(0)
n = 10_000
z = np.random.randn(n)
x = .3 * z + 2.5 * np.random.randn(n)
w = np.exp(np.random.randn(n) + .15 * x)
p = 1.0 / (1.0 + np.exp(-(.4 + 3.2 * x + .1 * np.log(w))))

df = pd.DataFrame({'z': z, 'x': x, 'y': (np.random.rand(n) < p).astype(float),
                   'w': w,
                   'grp1': np.random.randint(0, 3, n),
                   'grp2': np.random.randint(0, 5, n),
                   'wts': .1 + np.random.rand(n)})


fit1 = lm(
    'y ~ x + w | z + w',
    df,
    absorb=('grp1', 'grp2'),
    keep_model=False
)
print(fit1)
# print(fit1.summary_iv())

fit1 = lm(
    'y ~ x + w',
    df,
    absorb=('grp1', 'grp2'),
    keep_model=False
)
print(fit1)

fit1.predict()
fit1.predict(df)
fit1.predict(df, absorb=True)