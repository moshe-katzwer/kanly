from kanly.automatic_differentiation.function_callable import func_str_to_callable
import numpy as np


for x in (
    2.0,
    np.array([1, 2.]),
):

    print('-'*100)
    print('x = ', x)

    func_str = '-((1.4 - {a})**3 + (2.0 - {b})**2) + {b} * x'
    f = func_str_to_callable(func_str, other_args='x', nobs=len(x) if np.ndim(x) > 0 else 1)
    print(f)
    print(f([1.4, 2.0], x))

    grad, grad_info = f.get_analytical_gradient(return_info=True, agg_func='mean')
    print(grad_info['func_str_code'])
    print(grad([1.41, 1.2], x))

    df0, df_info = f.get_analytical_partial_derivative(0, return_info=True)
    print(df_info['func_str_code'])
    print(df0([1.41, 1.2], x))

    pds = f.get_analytical_partial_derivatives()
    for j in range(2):
        print(j, pds[j]([1.41, 1.2], x))
