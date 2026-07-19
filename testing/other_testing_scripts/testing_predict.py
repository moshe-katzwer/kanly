import pandas as pd
import numpy as np

from kanly.api import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from numpy.testing import assert_array_almost_equal

np.random.seed(0)
n = 10_000
z = np.random.randn(n)
x = .3 * z + 2.5 * np.random.randn(n)
w = np.exp(np.random.randn(n) + .15 * x)
p = 1.0 / (1.0 + np.exp(-(.4 + 3.2 * x + .1 * np.log(w))))

df = pd.DataFrame({'z': z, 'x': x, 'endog': (np.random.rand(n) < p).astype(float),
                   'w': w,
                   'grp': np.random.randint(0, 3, n),
                   'wts': .1 + np.random.rand(n)})

print('='*50 + "\n\nGLM")
fit = glm(
    #'endog ~ x + w | z + w $ wts',
    'endog ~ x + C(grp) + w',
    df,
    family='binomial', link='logit',
    #alpha=1.2, l1_ratio=.1,
)

print(np.corrcoef(fit.predict(), fit.endog_predicted))
# plt.scatter(fit.predict(), fit.endog_predicted)
# plt.show()
print(np.corrcoef(fit.predict(df), fit.endog_predicted))
try:
    print(np.corrcoef(fit.predict(df[:15], fail_on_column_difference=True), fit.endog_predicted[:15]))
except:
    print("Good!")
print(np.corrcoef(fit.predict(df[:15]), fit.endog_predicted[:15]))
# plt.scatter(fit.predict(df[:15]), fit.endog_predicted[:15])
# plt.show()

print('='*50 + "\n\nEla Net")

for fit_intercept in [True, False]:
    fit = elastic_net(
        #'endog ~ x + w | z + w $ wts',
        'endog ~ x + C(grp)*w',
        df,
        alpha=.01, l1_ratio=.1,
        fit_intercept=True
    )

    assert_array_almost_equal(fit.predict(), fit.fittedvalues, decimal=4)
    assert_array_almost_equal(fit.predict(df), fit.fittedvalues, decimal=4)
    assert_array_almost_equal(fit.predict(df[:10]), fit.fittedvalues[:10], decimal=4)


print('='*50 + "\n\nLM")

fit = lm(
    #'endog ~ x + w | z + w $ wts',
    'endog ~ x + C(grp)*w',
    df,
)

print(np.corrcoef(fit.predict(), fit.fittedvalues))
print(np.corrcoef(fit.predict(df), fit.fittedvalues))
print(np.corrcoef(fit.predict(df[:10]), fit.fittedvalues[:10]))


# Messy

import pandas as pd
import numpy as np
from kanly.api import lm, LM
import statsmodels.formula.api as smf
import time
import patsy
from kanly.formula.data_getter import SparseDataGetter
from scipy.sparse import hstack, csc_matrix
from numpy.testing import assert_almost_equal, assert_array_almost_equal

np.random.seed(0)
n = 400
df = pd.DataFrame({'x': np.random.randint(0, 4, n), 'z': np.arange(n),
                   'w': np.random.randn(n), 'grp': np.random.randint(0, 30, n), 'city': np.random.randint(0, 3, n),
                   'wtsvar': .5 + np.random.rand(n)},
                  #  index=np.random.choice(np.arange(10 * n), n, replace=False)  # TODO
                  )
e = np.random.randn(n)
df['z'] = -3 + .15 * df.w + .4 * e + 2 * np.random.randn(n)
df['y'] = 3 + 1.2 * df['x'] + df.z + 3 * e + df.city
df['q'] = np.random.randn(n)

for intercept in [' -1', '']:
    fit = lm('y ~ x + C(grp)*w + C(city):x' + intercept, df)

    assert_array_almost_equal(fit.fittedvalues, fit.predict(df))
    assert_array_almost_equal(fit.fittedvalues, fit.predict(df, fail_on_column_difference=True))
    for i in tqdm(df.index):
        assert_array_almost_equal(fit.fittedvalues[i], fit.predict(df.loc[[i], :]))
