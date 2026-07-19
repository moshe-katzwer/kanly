import pandas as pd
import numpy as np
from kanly.api import *
import statsmodels.formula.api as smf
import time
import patsy
from kanly.formula.data_getter import SparseDataGetter
from scipy.sparse import hstack, csc_matrix
from numpy.testing import assert_almost_equal, assert_array_almost_equal

np.random.seed(0)
n = 300
df = pd.DataFrame(
    {
        'x': np.random.randint(0, 4, n), 'z': np.arange(n),
        'w': np.random.randn(n), 'grp': np.random.randint(0, 4, n), 'city': np.random.randint(0, 3, n),
        'wtsvar': .5 + np.random.rand(n)
    },
    index=np.random.choice(np.arange(10 * n), n, replace=False)  # TODO
)

e = np.random.randn(n)
df['z'] = -3 + .15 * df.w + .4 * e + 2 * np.random.randn(n)
df['y'] = 3 + 1.2 * df['x'] + df.z + 3 * e + df.city
df['q'] = np.random.randn(n)
df['_one_'] = 1

for fit_intercept in [True, False]:
    print('='*100)
    fit = elastic_net('y ~ z * q + x + C(grp)*I(w**2) $ wtsvar', df, alpha=3.21, l1_ratio=.9,
                      fit_intercept=fit_intercept)
    print(fit)

    print(np.corrcoef(fit.predict(), fit.fittedvalues))
    print(np.corrcoef(fit.predict(df[:100]), fit.fittedvalues[:100]))
