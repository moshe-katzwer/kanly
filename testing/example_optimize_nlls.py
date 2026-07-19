from kanly.api import bfgs_pqn, func_str_to_callable, nlls, lm, glm, rlm, qr, elastic_net, gmm
import numpy as np
import pandas as pd

np.random.seed(0)

n = 300
x = np.random.randn(n)
y = 1.2 + 3.4 * x + .5 * np.random.randn(n)
func = func_str_to_callable('y - ({a} + {b} * x)', other_args='x,y')

objective = func.get_frozen_mean_squared_function(x=x, y=y)

print(bfgs_pqn(objective, [0., 0]))

print(nlls('[y] ~ {a} + {b} * [x]', {'x': x, 'y': y}))
print(lm('y~x', {'x': x, 'y': y}))
print(glm('y~x', {'x': x, 'y': y}))
print(rlm('y~x', {'x': x, 'y': y}, M='leastsquares'))
print(qr('y~x', {'x': x, 'y': y}, .2))
print(elastic_net('y~x', {'x': x, 'y': y}, .01, l1_ratio=1.0))
print(gmm(['[y] - ({a} + {b} * [x])',
           ('[y] - ({a} + {b} * [x])', '[x]')],
          {'x': x, 'y': y}))


n = 300
x = np.random.randn(n)
y = 1.2 + 3.4 * x + .5 * np.random.randn(n)
func = func_str_to_callable('y - ({a} + ({b}) ** 2 * x)', other_args='x,y')
objective = func.get_frozen_mean_squared_function(x=x, y=y)
grad = func.get_frozen_analytical_gradient(agg_func='mean_squared', x=x, y=y)
hess = func.get_frozen_analytical_hessian(agg_func='mean_squared', x=x, y=y)

print(nlls('[y] ~ {a} + {b} * [x]', {'x': x, 'y': y}))
print(nlls('[y] ~ {a} + {b}**2 * [x]', {'x': x, 'y': y}))

print(bfgs_pqn(objective, [0., -.01], ftol=1e-12, xtol=1e-12, gtol=1e-44, gradient_callable=grad, hessian_callable=hess))
