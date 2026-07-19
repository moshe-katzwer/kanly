import numpy as np
import pandas as pd

from kanly.api import sure, lm
from statsmodels.formula.api import wls as wls_sm

from numpy.testing import assert_array_almost_equal

n = 500
np.random.seed(0)
df = pd.DataFrame({
    'x': np.random.randn(n),
    'w': np.random.randn(n),
    'user_id': range(n),
    'e': np.random.randn(n),
})
df['wts1'] = np.exp(.2 * np.random.randn(n) + np.abs(df.x))
df['y1'] = 1.2 - 0.3 * df['x'] + .2 * np.random.randn(n) - .3 * df['e']
df['y2'] = 0.4 + .1 * df['x'] + .5 * np.random.randn(n) + .2 * df['e'] + .1 * df.w

fit = sure(
    [
        # {'formula': 'y1 ~ x', 'data': df, 'specification_name': 'y1', 'weights': 'wts1'},
        {'formula': 'y1 ~ x $ wts1', 'data': df, 'specification_name': 'y1'},
        {'formula': 'y2 ~ x + w', 'data': df, 'specification_name': 'y2'},
    ],
    cov_type='cluster', cov_kwds={'groups': 'user_id'}, debug=True
)

print(fit)
#
# print(wls('y1 ~ x $ wts1', df))
# print(wls('y1 ~ x', df))

assert_array_almost_equal(wls_sm('y1 ~ x', df, weights=df.wts1).fit().params, fit.params[:2])
assert_array_almost_equal(wls_sm('y2 ~ x + w', df).fit().params, fit.params[2:])

