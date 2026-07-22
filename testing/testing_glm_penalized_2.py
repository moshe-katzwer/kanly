import pandas as pd
from kanly.api import glm, elastic_net, ELASTIC_NET, GLM
import numpy as np
from numpy.testing import assert_allclose

np.random.seed(0)

n = 100
X = np.random.randn(n,10)
y = -3 + X @ np.ones(10) + np.random.randn(n)

for normalize in [False, True]:
    for intercept in [True, False]:
        for alpha in [.26, 1]:
            for regval in [0, 4]:

                if normalize and not intercept:
                    # elastic net ignores normalize without an intercept
                    continue

                fit = GLM(
                    y, X, opt_method='IRLS', 
                    alpha=alpha, l1_ratio=0, 
                    **(dict(fit_intercept=True, add_constant=True, first_column_constant=True) 
                    if intercept else dict(fit_intercept=False)),
                    cov_type='nonrobust', normalize=normalize,
                    regularize_to_values=regval
                )
                # print(fit)

                fit2 = ELASTIC_NET(y, X, fit_intercept=intercept, 
                                alpha=alpha, 
                                l1_ratio=0.0, normalize=normalize,
                                regularize_to_values=regval)
                # print(fit2)

                assert_allclose(fit.params, fit2.params, atol=1e-4, rtol=1e-4)
                print(f'{alpha=}, {regval=}, {intercept=}, {normalize=}')

print('ALL PASSED!')