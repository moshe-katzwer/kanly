import numpy as np
import pandas as pd

from kanly.api import gmm, lm, gmm_iv_linear, compare_results, gmm_iv_nonlinear

n = 10_000
np.random.seed(10)

data = pd.DataFrame()
data['e'] = np.random.randn(n)
scale = 1
data['z'] = (60 + 2.2 * np.random.randn(n)) * scale
data['x'] = (60 + 2.2 * np.random.randn(n) + data['z']) * scale
data['y'] = (1200 + 32 * data.x + 10 * data.e) * scale
data['Intercept'] = 1.0

print(data.corr())

fit_gmm = gmm_iv_linear(
    # [
    #      '[y] - ({Intercept} + {x}*[x])',
    #     ('[y] - ({Intercept} + {x}*[x])', '[z]')
    # ],
    'y ~ x | z',
    data,
    specification_name='GMM OLS',
    debug=True,
    cov_type='SANDWICH'
)
#
print(fit_gmm)
# print(fit_gmm.cov_params())
#
fit_iv = lm('y ~ x | z', data, cov_type='NONROBUST')

print(compare_results([fit_gmm, fit_iv]))


# print(fit_lm.condition_number)
# print(fit_lm.eigenvals)
#
# from statsmodels.formula.api import ols
# print(ols('y ~ x', data).fit().summary()	)
# print(ols('y ~ x', data).fit().eigenvals)
