import numpy as np
import pandas as pd
from kanly.api import nlls_en
from scipy.optimize import minimize, Bounds

from numpy.testing import assert_allclose

n = 100
np.random.seed(0)

df = pd.DataFrame({
    'x': np.random.randn(n),
    'z': np.zeros(n)
})

b0 = 2.2
l1_ratio = .26
df['y'] = 6 + np.random.randn(n) + 1.2 * df.x

for alpha in [0.0, 1.4, 20]:

    for b0 in [0, 2.]:
        for bounds in [
            {'b': (-np.inf, np.inf)},
            {'b': (2.2, 10)},
            {'b': (0, 1.8)},
            {'b': (-np.inf, 1.8)},
        ]:
            print(alpha, b0, bounds)

            fit = nlls_en('[y] ~ {a}+{b}*[x]+{c}*[z]', df,
                          # start_params=[0, -.3, 0],
                          max_iter=10000,
                          debug=False, regularize_to_values={'b': b0}, alpha={'b': alpha},
                          l1_ratio=l1_ratio, ftol=1e-10, xtol=1e-12, bounds=bounds)


            def func(x):
                return np.sum(fit.model.residual_function_callable(x) ** 2) / 2 \
                    + n * alpha * (1 - l1_ratio) / 2 * (x[1] - b0) ** 2 \
                    + n * alpha * l1_ratio * abs(x[1] - b0)


            fit_scipy = minimize(func, [0, 0, 0], tol=1e-14,
                                 bounds=[(-np.inf, np.inf), bounds['b'], (-np.inf, np.inf)])

            try:
                assert_allclose(fit_scipy.x, fit.params, rtol=1e-5, atol=1e-3)

            except:
                print(fit_scipy.fun, fit.objective_function_)
                assert_allclose(fit_scipy.x, fit.params, rtol=1e-5, atol=1e-3)
