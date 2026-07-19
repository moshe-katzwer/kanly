import numpy as np
from numpy.testing import assert_almost_equal

from kanly.api import func_str_to_callable
from kanly.automatic_differentiation.graph import AutoDiffGraphNode

x = 1.3
params = [1, 2., 10]
fmt = lambda z: "%12.5e" % z

for func_str in [
    '(1.1-{a})**2 + 200*({b}-{a}**2)**2 + 0*x',  # rosenbrock
    '{a} + {b} + expit({c}*x)',
    '{a} + {b} * logit(.4+.01*{c}*x)',
    '(({a}-3)**2 * np.exp({b}*x) / {a}*({b}-2) * 2*{a}**3 + np.log({c}))',
    '(({a}-3)**2 * np.exp({b}*x) / {a}*({b}-2)) * (2*{a}**3 + np.log({c}))',
    '(({a}-3)**2 + np.exp({b}*x) + {a}*({b}-2) + 2*{a}**3 + np.log({c}))',
    '(({a}-3)**2 + np.exp({b}*x) + {a}*({b}-2) + 2*{a}**3 + np.log({c}))**2',

]:

    func = func_str_to_callable(func_str, other_args='x')
    # print(AutoDiffGraphNode(func.func_str))
    print('\n\n\n', '=' * 50)
    print(func_str)
    print('F = ', fmt(func(params, x=x)))

    pds, pds_info = func.get_analytical_partial_derivatives(return_info=True)

    hessian, hess_info = func.get_analytical_hessian(do_jit=False, return_info=True)
    hessian_value = hessian(params, x=x)
    print(hessian_value)

    dx = 1e-6
    for i, pd in enumerate(pds):
        dx_i = dx * max(1, abs(params[i]))
        paramsh = np.array(params).copy()
        paramsh[i] += dx_i
        paramsl = np.array(params).copy()
        paramsl[i] -= dx_i

        print('\n', '-' * 30)
        print(pds_info[i]['partial_derivative_expression'])
        print(f'df/dx_{i} : ', fmt(pd(params, x=x)),
              fmt((func(paramsh, x=x) - func(paramsl, x=x)) / (2 * dx_i)),
              )

        df_di = AutoDiffGraphNode(pds_info[i]['partial_derivative_expression'])
        second_derivatives = df_di.get_analytical_partial_derivatives(
            func.num_params, 1, other_args='x')

        for j, df_d2 in enumerate(second_derivatives):
            dx_j = dx * max(1, abs(params[j]))
            paramsh = np.array(params).copy()
            paramsh[j] += dx_j
            paramsl = np.array(params).copy()
            paramsl[j] -= dx_j

            finite_diff_val = (pd(paramsh, x=x) - pd(paramsl, x=x)) / (2 * dx_j)
            pd_val = df_d2(params, x=x)

            print(f'\td2 f/(dx_{i} dx{j}) : ',
                  fmt(finite_diff_val),
                  fmt(pd_val),
                  fmt(hessian_value[i][j])
                  )

            assert_almost_equal(finite_diff_val, pd_val, decimal=5)
            assert_almost_equal(finite_diff_val, hessian_value[i][j], decimal=5)
