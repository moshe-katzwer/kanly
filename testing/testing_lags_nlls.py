import numpy as np
import pandas as pd

from kanly.api import nlls, lm, compare_results, glm
from numpy.testing import assert_array_almost_equal

n = 5000
np.random.seed(0)
df = pd.DataFrame({'x': np.random.randn(n)})
df.loc[[40, 50], 'x'] = np.nan

df['g'] = np.random.randint(0, 3, n)
df.loc[[100, 101, 102], 'g'] = None
df['x_2'] = df.x ** 2

df['L1_x_g'] = df.groupby('g').x_2.shift(1)
df['y'] = 1.5 + 2 * df.x ** 2 + 10 * df['L1_x_g'] + 4.3 * np.random.randn(n)

df.loc[[1, 2, 10], 'y'] = np.nan

fit1 = lm('y ~ I(x**2) + L1_x_g', df)
fit2 = lm('y ~ I(x**2) + L(x**2,1,g)', df, debug=False)
fit3 = nlls('[y] ~ {Intercept}+{I(x**2)}*[x**2]+{L1_x_g}*[L(x**2,1,g)]', df, debug=True)

print(compare_results([fit1, fit2, fit3]))

for f in [fit2, fit3]:
    assert_array_almost_equal(f.params, fit1.params)
    assert_array_almost_equal(f.bse, fit1.bse)