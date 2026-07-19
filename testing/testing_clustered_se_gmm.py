import numpy as np
import pandas as pd

from kanly.api import lm, gmm_iv_linear, nlls, compare_results, glm
from numpy.testing import assert_allclose

n = 300
np.random.seed(0)
df = pd.DataFrame({
    'x': np.random.randn(n),
    'grp': np.random.randint(0, 12, n)
})

df['y'] = 1.2 - 0.3 * df['x'] + .2 * np.random.randn(n) + (df.grp - 20)

fit_lm = lm('y ~ x', df, cov_type='cluster', cov_kwds={'groups': 'grp'})
print(fit_lm)

import statsmodels.formula.api as smf
print(smf.ols('y~x', df).fit(cov_type='cluster', cov_kwds={'groups': df.grp, 'use_t': True}).summary())

fit_gmm = gmm_iv_linear('y ~ x | x', df, cov_type='cluster', cov_kwds={'groups': 'grp'})
print(fit_gmm)

fit_nlls = nlls('[y] ~ {Intercept} + {x}*[x]', df, cov_type='cluster', cov_kwds={'groups': 'grp'})
print(fit_nlls)

#fit_glm = glm('y ~ x', df, family='gaussian', cov_type='cluster', cov_kwds={'groups': 'grp'})
#print(fit_glm)

print(compare_results([fit_nlls, fit_gmm, fit_lm], show_t=True, show_bse=True, fit_titles=['nlls', 'gmm', 'ols']))

fits = [fit_nlls, fit_gmm, fit_lm]

for f in fits:
    assert_allclose(f.params, fits[0].params)

for f in fits:
    assert_allclose(f.bse, fits[0].bse)