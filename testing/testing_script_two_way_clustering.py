import numpy as np
import pandas as pd

from kanly.api import lm, compare_results, nlls
from numpy.testing import assert_array_almost_equal

n = 10_000
np.random.seed(0)
df = pd.DataFrame({
    'x': np.random.randn(n),
    'grp1': np.random.randint(0, 60, n),
    'grp2': np.random.randint(0, 40, n),
})
df['y'] = 1.2 - 0.3 * df['x'] + .2 * (np.exp(-df['x']) * np.random.randn(n) * (1 + df['grp1']))

for index in [None, np.random.choice(range(n), 6000, replace=False)]:
    for missing in [True, False]:
        df_copy = df.copy()
        if missing:
            if index is None:
                df_copy.loc[4,'y'] = np.nan
            else:
                df_copy.loc[index[0], 'y'] = np.nan

        fit_ols = lm('y ~ x', df_copy, cov_type='cluster',
                     cov_kwds={'groups': ('grp1', 'grp2')}, specification_name='2way-Cluster ols',
                     index=index)

        fit_nlls = nlls('[y] ~ {Intercept}+{x}*[x]', df_copy, cov_type='cluster',
                     cov_kwds={'groups': ('grp1', 'grp2')}, specification_name='2way-Cluster nlls',
                     index=index)

        print(compare_results([fit_ols, fit_nlls]))
        assert_array_almost_equal(fit_ols.bse,fit_nlls.bse)
