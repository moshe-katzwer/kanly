import numpy as np

from kanly.api import lm, blm, glm

np.random.seed(0)
n = 50
x = np.random.randn(n)
z = np.random.rand(n)
y = 3 + 10 * x + np.random.randn(n) * 3
wts = .01 + np.random.rand(n)
data = {'x': x, 'y': y, 'z': z, 'wts': wts}

l2_penalty_x = 3.3

f1 = blm('y ~ x', data)
f2 = lm('y ~ x', data)
f3 = glm('y ~ x', data)

for fit in [f1, f2, f3]:
    print('\n\n')
    print(fit)
    print(fit[1], fit['x'])
    print(fit.test_from_clt_simulation(lambda x: x[1] / x[0]))
    print(fit.test_from_clt_simulation('{x}/{Intercept}'))

    print(fit.test_delta_method(lambda x: x[1] / x[0]))
    print(fit.test_delta_method('{x}/{Intercept}'))

    print(fit.test_ratio_fieller({'x': 1}, {'Intercept': 1}))

    print(fit.test_linear_combination({'x': 1.0}))