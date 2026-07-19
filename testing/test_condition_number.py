import numpy as np
import pandas as pd

from kanly.api import lm
from statsmodels.formula.api import ols

n = 100
np.random.seed(0)
df = pd.DataFrame({
    'x': np.random.randn(n),
    'grp': np.random.randint(0, 12, n)
})
df['y'] = 1.2 - 0.3 * df['x'] + .2 * np.random.randn(n)

fit = lm('y ~ x + C(grp)', df, debug=False)
print(fit)

fit2 = ols('y ~ x + C(grp)', df).fit()
print(fit2.summary())

print(fit.condition_number)
print(fit.eigenvals)

print(fit2.condition_number)
print(fit2.eigenvals)

print(fit.get_eigenvals_and_condition_number())