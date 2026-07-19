from kanly.api import bfgs_pqn, func_str_to_callable
from numpy.testing import assert_allclose
import numpy as np

for name, (formula, x_star, x0) in {
    'mccormick': ('sin({x} + {y}) + ({x}-{y})**2 - 1.5*{x} + 2.5*{y} + 1',
                  [-0.54719, -1.54719], [0, 0]),
    'easom': (f'-cos({{x}}) * cos({{y}}) * exp(-( ({{x}}-{np.pi})**2 + ({{y}}-{np.pi})**2) ) ',
              [np.pi, np.pi], [0, 0]),
    'himmelblau': ('({x}**2 + {y} - 11)**2 + ({x} + {y}**2 - 7)**2',
                   [3, 2], [0, 0]),
    'bukin': ('100.0*sqrt(np.abs({y}-0.01*{x}**2)) + .01*np.abs({x}+10)',
              [1, -10.], [.5, -4])
}.items():

    print(f'{name}...', end='')

    func = func_str_to_callable(formula)

    result = bfgs_pqn(func, x0=x0, xtol=1e-20, ftol=1e-20)

    try:
        assert_allclose(x_star, result.x, rtol=1e-3, atol=1e-3)
        print('passed')
    except:
        print('\n\n', result)
