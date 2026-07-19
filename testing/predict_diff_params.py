import pandas as pd
import numpy as np

from kanly.api import *
import matplotlib.pyplot as plt

from numpy.testing import assert_array_almost_equal

np.random.seed(0)
n = 10
x = 2.5 * np.random.randn(n)

df = pd.DataFrame({'x': x,
                   'grp1': np.random.randint(0, 3, n)})
df['y'] = -1 + .4 * df.x + .1*np.random.randn(n)

fit1 = lm(
    'y ~ x',
    df,
)
print(fit1)

print(fit1.predict())
params = fit1.params.copy()
params.loc['Intercept'] += 2
print(fit1.predict(exog=df, params=params))

assert_array_almost_equal(fit1.predict() + 2, fit1.predict(exog=df, params=params))