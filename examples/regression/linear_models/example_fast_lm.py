from textwrap import wrap

import numpy as np
import pandas as pd

from kanly.api import lm, lm_fast

np.random.seed(0)
n = 2_000_000
x = np.random.randn(n)
df = pd.DataFrame({'x': x,
                   'z': np.random.randn(n),
                   'g': np.random.randint(0, 1_500, n),
                   'y': 10 + x * 1.2 + np.random.randn(n)
                   })

fit = lm('y ~ x*C(g) + I(x**2) + poly(z, 3)', df, debug=False)
fit_fast = lm_fast('y ~ x*C(g) + I(x**2) + poly(z, 3)', df, debug=False)

print(fit)
print(fit_fast)

print(fit.fit_elapsed, fit_fast.fit_elapsed)
