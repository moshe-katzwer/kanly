import pandas as pd
import numpy as np
from kanly.api import bfgs_pqn
from scipy.optimize import minimize
from kanly.regression.linear_models.penalized.sparse_elastic_net_internal \
    import sparse_elastic_net_coordinate_descent_quad_form_setup

from numpy.testing import assert_allclose

n = 2_000
p = 40
s = .01
np.random.seed(0)
X = np.random.randn(n, p).dot((s * np.eye(p) + (1 - s) * np.ones((p, p))))
z = np.ones(4)
beta = np.hstack([z, np.zeros(p - len(z))])
y = 3 + X.dot(beta) + np.random.randn(n)


xcols = [f'x{d}' for d in range(p)]
df = pd.DataFrame(X, columns=xcols)
df['y'] = y

for alpha_ in [.5, 1, 10]:

    l1_ratio = .4

    reg_2_vals = np.array([.7] * 6 + [0.] * (p - 6))

    for wts in [
        np.ones(n),
        np.random.rand(n),
    ]:

        wts_normalized = wts * n / sum(wts)

        def obj_func(x):
            a, b = x[0], x[1:]
            ssr = np.sum(wts_normalized * (y - a - X.dot(b)) ** 2)
            penalty = alpha_ * np.sum(
                l1_ratio * np.abs(b - reg_2_vals)
                + (1 - l1_ratio) / 2 * (b - reg_2_vals) ** 2
            )
            return ssr / (2 * n) + penalty


        print('\n' * 4)
        print('=' * 50)
        print({'alpha': alpha_, 'wts[:4]': wts[:4]})
        print()

        res = bfgs_pqn(obj_func, [0.] * (p + 1), maxiter=5_000, xtol=1e-10, ftol=1e-20, gtol=1e-8,
                       momentum=.1, onesided_fd=False, B0=1e6, dx_fd=1e-8,
                       )

        print(res.converged, res.xerr, res.fun)
        print(res.x[1:].round(4))
        print(res.message, res.time_elapsed)

        fit = sparse_elastic_net_coordinate_descent_quad_form_setup(
            X, y, weights=wts,
            alpha=alpha_, l1_ratio=l1_ratio,
            normalize=False,
            regularize_to_values=reg_2_vals,
            max_iter=25_000,
            xtol=1e-12, ftol=1e-12, gtol=1e-8,
        )
        print('--')
        print(fit['converged'], fit['x_error'], fit['objective_function_'])
        print(fit['coef_'].round(4))
        print(fit['message'], fit['fit_time'])

        print('>> ', fit['objective_function'](fit['intercept_'], fit['coef_']))
        print('>> ', obj_func(np.hstack([fit['intercept_'], fit['coef_']])))
        print('>> ', fit['objective_function'](res.x[0], res.x[1:]))
        print('>> ', obj_func(np.hstack([res.x[0], res.x[1:]])))

        assert_allclose(fit['coef_'], res.x[1:], atol=1e-3, rtol=1e-3)
        assert_allclose(fit['fittedvalues'], X.dot(fit['coef_']) + fit['intercept_'], atol=1e-3, rtol=1e-3)
        assert_allclose(fit['objective_function_'], obj_func(np.hstack([[fit['intercept_']], fit['coef_']])), atol=1e-3, rtol=1e-3)
        assert_allclose(fit['objective_function_'], fit['objective_function'](fit['intercept_'], fit['coef_']),
                        atol=1e-3, rtol=1e-3)


print("ALL PASSED!")