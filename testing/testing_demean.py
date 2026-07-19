import numpy as np
import pandas as pd

from kanly.api import nlls, lm, compare_results, glm
from numpy.testing import assert_array_almost_equal

np.random.seed(0)

n = 20
x = 12 + np.random.randn(n)
df = pd.DataFrame({'x': x,
                   'e': np.random.randn(n),
                   'w': np.exp(-.25 + .35 * x),
                   })
#df.loc[[40, 50], 'x'] = np.nan
# print(df.w)
df['y'] = 1.5 + 2 * df.x + 2.3 * df.e

for is_wtd in [False, True]:

    if is_wtd:
        df['x_dm'] = df.x - np.average(df.x, weights=df.w)
        dm_var = 'center(x,w)'
        wt_str = ' $ w'
        wt_str_nlls = ' $ [w]'
    else:
        df['x_dm'] = df.x - df.x.mean()
        dm_var = 'center(x)'
        wt_str = ''
        wt_str_nlls = ''

    fit1 = lm('y ~ x_dm' + wt_str, df)
    fit2 = lm(f'y ~ {dm_var}' + wt_str, df)
    #fit4 = glm(f'y ~ {dm_var}' + wt_str, df)
    fit3 = nlls('[y] ~ {Intercept}+{x_dm}*[%s]  %s' % (dm_var, wt_str_nlls), df)

    print(compare_results([fit1, fit2, fit3,
                        # fit4
                        ]))

    for f in [fit2, fit3]:
        assert_array_almost_equal(f.params, fit1.params)
        assert_array_almost_equal(f.bse, fit1.bse)