import pandas as pd
import numpy as np

from kanly.api import *
import matplotlib.pyplot as plt

from numpy.testing import assert_array_almost_equal

np.random.seed(0)
n = 10_000
z = np.random.randn(n)
x = .3 * z + 2.5 * np.random.randn(n)
w = np.exp(np.random.randn(n) + .15 * x)
p = 1.0 / (1.0 + np.exp(-(.4 + 3.2 * x + .1 * np.log(w))))

df = pd.DataFrame({'z': z, 'x': x, 'y': (np.random.rand(n) < p).astype(float),
                   'w': w,
                   'grp1': np.random.randint(0, 40, n),
                   'wts': .1 + np.random.rand(n)})


fit1 = lm(
    'y ~ x + w',
    df,
    #cov_type='cluster', cov_kwds= {'groups': 'grp1'}
)
print(fit1)
v = fit1.recompute_cov('cluster', {'groups': 'grp1'}, save_cov_params=True, debug=True)
print(fit1)

#
# fit1 = reg(
#     'y ~ x + w',
#     df,
#     cov_type='cluster', cov_kwds= {'groups': 'grp1'}
# )
# print(fit1)
# v = fit1.compute_cov('cluster', {'groups': 'grp1'}).toarray()
# print(np.diag(v)**.5)
#
#
# fit1 = reg(
#     'y ~ x + w',
#     df,
#     keep_model=False,
#     cov_type='cluster', cov_kwds={'groups': 'grp1'}
# )
# print(fit1)
#
#
# fit2 = reg(
#     'y ~ x + w',
#     df,
#     keep_model=False,
#     compute_cov=False,
#     #cov_type='cluster', cov_kwds={'groups': 'grp1'}
# )
# print(fit2)
#
#
