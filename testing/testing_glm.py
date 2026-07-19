import pandas as pd
import numpy as np

from kanly.api import *

np.random.seed(0)
n = 50_000
z = np.random.randn(n)
x = .3 * z + .5 * np.random.randn(n)
w = np.exp(np.random.randn(n) + .15 * x)
p = 1.0 / (1.0 + np.exp(-(.4 + .2 * x + .1 * np.log(w))))

df = pd.DataFrame({'z': z, 'x': x, 'endog': (np.random.rand(n) < p).astype(float),
                   'w': w,
                   'grp': np.random.randint(0, 20, n),
                   'wts': .1 + np.random.rand(n)})

fit = glm(
    # 'endog ~ x + w | z + w $ wts',
    'endog ~ x + C(grp)*w',
    df,
    family='binomial', link='logit',
    # opt_method='COORDINATE_DESCENT_1_ITER',
    # alpha=1.2, l1_ratio=.1,
    debug=True)
print(fit.summary())

print(np.corrcoef(fit.predict(df), fit.endog_predicted))

# from statsmodels.formula.api import ols
#
# ols('endog ~ 1', df).fit()
