import numpy as np
import pandas as pd
from kanly.api import nlls
from kanly.automatic_differentiation.graph import AutoDiffGraphNode
from kanly.regression.nonlinear_least_squares.model import SparseNonlinearLeastSquaresRegressionResults, SparseNonlinearLeastSquaresModel
from numpy.testing import assert_allclose

n = 1_000
np.random.seed(0)
df = pd.DataFrame({'x': np.random.randn(n)})
df['p'] = 1 / (1 + np.exp(-(.4 + .9 * df.x)))
df['z'] = np.random.randn(n)
df['y'] = (np.random.rand(n) < df['p']).astype(float)
df['g'] = pd.Series(np.random.randint(0, 10, n)).astype(str)

func_str = ('[y] ~ -{phi}*[z] '
            '+ np.exp(-np.log(5+[x])** -2 *{gamma} - 1.0 / (1.0 + exp({alpha} + {beta} * [x])) * [x])'
            '+ {q1}/(5+{q2}*[x])*({q3}+[z]) + 5**({q4}) + [C(g,-1)]'
            )

fit = nlls(func_str,
           df,
           cov_type='hc1',
           debug=False,
           jac_method='mid',
           # subsample=500,
           max_iter=500,
           specification_name='logistic regression')

print(fit)
assert isinstance(fit, SparseNonlinearLeastSquaresRegressionResults)

model = fit.model
assert isinstance(model, SparseNonlinearLeastSquaresModel)


for i, p in enumerate(fit.params.index):
    print(i, p)

pred_func = model.prediction_function_callable
resid_func = model.residual_function_callable

print(pred_func.func_str)

print(AutoDiffGraphNode(pred_func.func_str, debug=True))

x0 = np.ones(model.num_params)

atol, rtol = 1e-8, 1e-5

for name, func in {'pred': pred_func, 'resid': resid_func}.items():
    for do_jit in [True, False]:
        for dense_mb in [10_000, 0]:
            jac_func, info = func.get_analytical_jacobian(do_jac_jit=do_jit, dense_threshold_mb=dense_mb)
            Jac = jac_func(x0)
            print('\n' + '-' * 100)
            print(type(func))
            print(info['func_str_code'])
            print('do_jit = ', do_jit)
            print('dense_mb = ', dense_mb)
            pds = func.get_analytical_partial_derivatives(do_jit=do_jit)

            print()
            for i in range(model.num_params):
                print(name, i, end='')

                x0_h = x0.copy()
                x0_h[i] += 1e-6
                x0_l = x0.copy()
                x0_l[i] -= 1e-6
                df_i = (func(x0_h) - func(x0_l)) / 2e-6

                jac_i = Jac[:, i] if isinstance(Jac, np.ndarray) else Jac.getcol(i).toarray().flatten()

                # check jacobian
                assert_allclose(df_i, jac_i, atol=atol, rtol=rtol)
                print(' passed jacobian', end='')

                # check partial derivative, individual
                pd = func.get_analytical_partial_derivative(i, do_jit=do_jit)
                assert_allclose(df_i, pd(x0), atol=atol, rtol=rtol)
                print(' passed partial individual', end='')

                # check partial derivative from group
                assert_allclose(df_i, pds[i](x0), atol=atol, rtol=rtol)
                print(f' passed partial from group (max_diff={np.max(np.abs(df_i - jac_i))})')


print('\n\n ALL DONE!')