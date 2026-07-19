"""
For testing prediction on a partial data set where some items might be missing
from column space
"""

import numpy as np
from kanly.api import lm, qr, glm, rlm, elastic_net, nlls
import pandas as pd
from numpy.testing import assert_allclose

np.random.seed(0)
n = 100
x = np.random.randn(n)
y = np.exp(2*x + np.random.randn(n))
g = np.random.randint(0,20,n)
df = pd.DataFrame(dict(x=x,y=y,g=g))

# Linear Models
for method, kw in [
    (lm, dict()),
    (qr, dict(tau=.5)),
    (glm, dict(family='Poisson')),
    (rlm, dict()),
    (elastic_net, dict(alpha=.00001)),
]:

    fit = method('y~x+C(g)', df, **kw)
    pred = fit.predict(data=df[['x','g']].iloc[:4], debug=False, ignore_column_mismatch=True)
    print(method, pred, pred - fit.fittedvalues[:4])
    assert_allclose(pred, fit.fittedvalues[:4], atol=1e-4, rtol=1e-4)

# # NLLS TODO
# fit = nlls('[y] ~ {a}+{b}*[np.exp(x)]', df)
# pred = fit.predict(data=df[['x', 'g']].iloc[:4], debug=False, ignore_column_mismatch=True)
# print(method, pred, pred - fit.fittedvalues[:4])
# assert_allclose(pred, fit.fittedvalues[:4], atol=1e-4, rtol=1e-4)

