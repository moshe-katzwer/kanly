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
    'y ~ x + w',
    df,
    absorb=('grp1', 'grp2')
)
fit2 = lm(
    'y ~ x + w + C(grp1)*C(grp2)',
    df
)

assert_array_almost_equal(fit1.fittedvalues, fit1.predict())


#
# assert_array_almost_equal(fit2.fittedvalues, fit2.predict(df))
# assert_array_almost_equal(fit1.predict(df, absorb=True), fit2.predict(df))
#
# print(fit1)
