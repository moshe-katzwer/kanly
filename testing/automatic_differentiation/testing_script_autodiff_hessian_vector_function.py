from kanly.api import func_str_to_callable
import numpy as np
from numpy.testing import assert_allclose

print("LINEAR:\n")
func_str = 'y - {a} - {b}*x'
func = func_str_to_callable(func_str, other_args='x,y')
# print(func)

n = 10
np.random.seed(0)
x = np.random.randn(n)
y = 1 + 4 * x + np.random.randn(n)

for name, hessian, hessians_by_hand in [

    ('mean_squared', func.get_analytical_hessian(nobs=n, agg_func='mean_squared'),
     [
         [lambda p, x, y: 2.0, lambda p, x, y: 2.0 * np.mean(x)],
         [lambda p, x, y: 2.0 * np.mean(x), lambda p, x, y: 2.0 * np.mean(x ** 2)],
     ]),

    ('mean', func.get_analytical_hessian(nobs=n, agg_func='mean'),
     [
         [lambda p, x, y: 0.0, lambda p, x, y: 0.0],
         [lambda p, x, y: 0.0, lambda p, x, y: 0.0]
     ]),
]:

    print("\t", name, end='...')

    p = np.ones(func.num_params)
    hess_value = hessian(p, x=x, y=y)
    for i in range(func.num_params):
        for j in range(func.num_params):
            assert_allclose(hess_value[i][j], hessians_by_hand[i][j](p, x, y))

    print('passed!')

print("\n\nNON-LINEAR:\n")
func_str = 'y - {a} - {b}*x - np.exp({c}*x)'
func = func_str_to_callable(func_str, other_args='x,y')

n = 10
np.random.seed(0)
x = np.random.randn(n)
y = 1 + 4 * x + np.random.randn(n)

for name, hessian, hessians_by_hand in [

    ('mean_squared', func.get_analytical_hessian(nobs=n, agg_func='mean_squared'),
     [
         [lambda p, x, y: 2.0, lambda p, x, y: 2.0 * np.mean(x),
          lambda p, x, y: 2.0 * np.mean(np.exp(p[-1] * x) * x)],
         [lambda p, x, y: 2.0 * np.mean(x), lambda p, x, y: 2.0 * np.mean(x ** 2),
          lambda p, x, y: 2.0 * np.mean(np.exp(p[-1] * x) * x ** 2)],
         [lambda p, x, y: 2.0 * np.mean(np.exp(p[-1] * x) * x),
          lambda p, x, y: 2.0 * np.mean(np.exp(p[-1] * x) * x ** 2),
          lambda p, x, y: 2.0 * np.mean((y - p[0] - p[1] * x - np.exp(p[-1] * x)) * (-np.exp(p[-1] * x) * x ** 2))
                          + 2.0 * np.mean((np.exp(p[-1] * x) * x) ** 2)
          ]
     ],
     ),

    ('mean', func.get_analytical_hessian(nobs=n, agg_func='mean'),
     [
         [lambda p, x, y: 0.0, lambda p, x, y: 0.0, lambda p, x, y: 0.0],
         [lambda p, x, y: 0.0, lambda p, x, y: 0.0, lambda p, x, y: 0.0],
         [lambda p, x, y: 0.0, lambda p, x, y: 0.0, lambda p, x, y: -np.mean(np.exp(p[-1] * x) * x ** 2)]
     ]),
]:

    print("\t", name, end='...')

    p = np.ones(func.num_params)
    hess_value = hessian(p, x=x, y=y)
    for i in range(func.num_params):
        for j in range(func.num_params):
            assert_allclose(hess_value[i][j], hessians_by_hand[i][j](p, x, y))

    print('passed!')
