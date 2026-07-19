import pandas as pd
import numpy as np
from kanly.api import lm
from statsmodels.formula.api import wls

n = 1500
np.random.seed(0)

z1 = np.random.randn(n)
z2 = np.random.randn(n)
e = np.random.randn(n)
x = 1.6 + .3 * z1 - 1.2 * z2 + 2.3 * e
y = -3 + 1 * x + 1.2 * e

df = pd.DataFrame({
    'z1': z1,
    'z2': z2,
    'e': e,
    'x': x,
    'y': (y),
    'g': np.random.randint(0, 3, n),
    'w': .01 + np.random.rand(n)
})

fit = lm('y ~ x | z1 $ w', df, absorb='g')
print(fit)


fit = lm('y ~ x + C(g) | z1 + C(g) $ w', df)
print(fit)
