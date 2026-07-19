import traceback as tb

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from scipy.sparse import hstack, csc_matrix
from statsmodels.sandbox.regression.gmm import IV2SLS

from kanly.api import lm, LM
from kanly.formula.data_getter import SparseDataGetter

np.random.seed(0)
n = 50_000
df = pd.DataFrame({'x': np.random.randint(0, 4, n),
                   'x_zero': 0.0,
                   'w': np.random.randn(n), 'grp1': np.random.randint(0, 4, n), 'city': np.random.randint(0, 3, n),
                   'grp2': np.random.randint(0, 2, n).astype(str),
                   'wtsvar': .5 + np.random.rand(n)},
                   index=np.random.choice(np.arange(10 * n), n, replace=False)  # TODO
                  )

e = np.random.randn(n)
df['z'] = -3 + .15 * df.w + .4 * e + 2 * np.random.randn(n)
df['y'] = 3 + 1.2 * df['x'] + df.z + 3 * e + df.city - 1.2 * df['grp2'].map(lambda x: int(x))

# ## TEST regression on constant
# fit = lm('x ~ 1', df)
# print(fit)
#
# ## TEST SINGLE REGRESSOR
# fit = lm('y ~ x-1', df)
# print(fit)
#
# ## TEST SINGLE REGRESSOR IV
# fit = lm('y ~ z-1 | w-1', df)
# print(fit)

## TEST INSTRUMENT PARAMS
fit = lm('y ~ z + x + wtsvar | w + x + wtsvar', df, scale_design_matrix=True)
print(fit)
print(pd.DataFrame(fit.instrument_info.instrument_params.toarray(),
                   index=fit.instrument_names, columns=fit.exog_names))
for col, idx in [('x', 2), ('wtsvar', 3)]:
    assert_array_almost_equal(df[col], fit.model.instruments.dot(fit.instrument_info.instrument_params).toarray()[:, idx])

