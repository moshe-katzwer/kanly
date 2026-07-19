import pandas as pd
import numpy as np

from kanly.api import lm, glm




n = 100
np.random.seed(0)

df = pd.DataFrame({
    'y': np.random.randn(n),
    'x': np.random.randn(n),
    'z': [2.0] * n,
    'A': ['c'] * n
})

df.loc[1, 'x'] = np.nan
df.loc[1, 'z'] = 10
df['z2'] = 0

for func, kwargs in [
    (lm, dict()),
    (glm, {'family': 'gaussian'})
]:

    fit = func('y ~ x + z + C(A)', df, check_constant_cols=True)
    print(fit)

    fit = func('y ~ x + z + C(A)-1', df, check_constant_cols=True)
    print(fit)

    fit = func('y ~ x + C(A)-1', df, check_constant_cols=True)
    print(fit)

    fit = func('y ~ x + z2 + C(A)-1', df, check_constant_cols=True)
    print(fit)

    fit = func('y ~ x + z2 + C(A)', df, check_constant_cols=True)
    print(fit)
