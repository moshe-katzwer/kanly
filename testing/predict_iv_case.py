import numpy as np
import pandas as pd

from kanly.api import lm, compare_results, glm, qr, elastic_net
from numpy.testing import assert_array_almost_equal

n = 250
np.random.seed(0)
df = pd.DataFrame({
    'z': np.random.randn(n),
    'e': np.random.randn(n),
    'grp': np.random.randint(0, 5, n),
    'obs': np.random.randint(1, 6, n),
})
df['x'] = .5 * df['z'] + .7 * df['e']
df['y'] = 1.2 - 0.3 * df['x'] + .2 * df['e'] / df['obs']

fit_iv = lm('y ~ x + C(grp) | z + C(grp) $ obs', df, specification_name='WLS-IV')
print(fit_iv)

fit_iv.predict()

yhat = fit_iv.predict(df[['x', 'grp']], override_iv_error=True)

print(pd.DataFrame({'y1': fit_iv.fittedvalues, 'y2': yhat}))

for method, kwargs in zip([lm, glm, qr, elastic_net], [dict(), dict(), {'tau': .3}, {'alpha': .00001}]):
    fit = method('y ~ x + C(grp) $ obs', df, **kwargs)
    print(fit)
    temp = pd.DataFrame({'y1': fit.fittedvalues[:40], 'y2': fit.predict(df.loc[:39, ['x', 'grp']])})
    assert_array_almost_equal(temp.y1, temp.y2)
