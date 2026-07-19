import numpy as np
import pandas as pd
from kanly.api import nlls, lm, glm

n = 10000
np.random.seed(0)

e = np.random.randn(n)
z1 = np.random.randn(n)
z2 = np.random.randn(n)
x = np.random.randn(n) + 0.3 * z1 - 0.2 * z2 + 0.5 * e
g = np.random.randint(0, 4, n)
y = 2 + 0.6 * z1 + 1.2 * x + e - 4.3 * (g == 1) + 0.6 * (g == 2) + 2.2 * (g == 3)

df = pd.DataFrame({
    'e': e,
    'z1': z1,
    'z2': z2,
    'x': x,
    'g': g,
    'y': y,
    'c': np.random.randint(0, 10, n),
    'w': np.random.rand(n) * np.exp(x)
})

for cov_type, cov_kwds in zip(
        ('cluster', 'bootstrap'),
        ({'groups': 'c'}, {'groups': 'c', 'n_samples': 25, 'seed': 10})
):

    print(nlls('[y] ~ {alpha} + {beta} *[x]', df, cov_type=cov_type, cov_kwds=cov_kwds))
    print(lm('y ~ x', df, cov_type=cov_type, cov_kwds=cov_kwds))