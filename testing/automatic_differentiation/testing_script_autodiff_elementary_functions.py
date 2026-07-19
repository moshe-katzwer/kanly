import numpy as np
import pandas as pd
from kanly.api import nlls
from kanly.automatic_differentiation.graph import AutoDiffGraphNode
from numpy.testing import assert_allclose

n = 1_000
np.random.seed(0)
df = pd.DataFrame({'x': 0.1 + .3 * np.random.rand(n)})

atol, rtol = 1e-4, 1e-4

for func in AutoDiffGraphNode.DERIV_FUNC_NAME_DICT.keys():
    func_str = f'[x] ~ {{a}} + {func}([x]*{{b}})'
    fit = nlls(func_str, df, debug=False, start_params=[0, 1])

    pred_func = fit.model.prediction_function_callable
    resid_func = fit.model.residual_function_callable

    for name, func in {'pred': pred_func, 'resid': resid_func}.items():
        for do_jit in [True, False]:
            for dense_mb in [10_000, 0]:

                print("\n", func_str, name, do_jit, dense_mb)

                x0 = np.array([0., 1.])

                jac_func, info = func.get_analytical_jacobian(do_jac_jit=do_jit, dense_threshold_mb=dense_mb)
                pds = func.get_analytical_partial_derivatives(do_jit=do_jit)

                Jac = jac_func(x0)

                for i in range(fit.model.num_params):
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
