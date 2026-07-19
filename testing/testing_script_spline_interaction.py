from kanly.api import build_data_model, lm, nlls, glm

import numpy as np

np.random.seed(0)
n = 1000

z = 4 * np.random.randn(n)
x = 4 * np.random.randn(n)
g1 = np.random.randint(0, 2, n)
g2 = np.random.randint(0, 8, n)

y = .3 * np.random.randn(n) + 4 * (z > 4)

data = {'y': y, 'g1': g1, 'g2': g2, 'z': z, 'x': x}

from statsmodels.formula.api import ols

print(ols('y ~ C(g1) : bs(z,df=3)', data).fit().summary())
print(lm('y ~ C(g1) : bs(z,df=3)', data))
