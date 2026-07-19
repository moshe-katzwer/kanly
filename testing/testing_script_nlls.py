import pandas as pd
import numpy as np
from kanly.api import nlls
from kanly.regression.nonlinear_least_squares.function_callables.loss_functions import QuantileSmooth

from scipy.optimize import least_squares

n = 2_500
np.random.seed(0)

x = 3 + np.random.randn(n)
z = np.exp(np.random.randn(n) / 3)
e = np.random.randn(n) * (1 + np.abs(x - 3))
g = np.random.randint(0, 3, n)

df = pd.DataFrame({
    'x': x,
    'z': z,
    'e': e,
    'g': g,
    'y': (6 + np.exp(-.5 * x) + 2 * x ** 2 + .6 * e + 2 * z * (g == 1) - 3 * (g == 0) + .1 * z ** 3),
    'w': (1.0 / np.abs(x))
})

failed = []

for jac_method in ['mid', 'analytic']:
    for weighted in [False, True]:
        for bounds in [None, np.array([(0., 3.)] * 7)]:
            for loss in [dict(), {'root_loss_function': 'QuantileSmooth', 'f_scale': 1}]:

                print("\n\n" + "#" * 100 + "\n")
                print(f'is_wtd={weighted}, bndd={bounds is not None}, lf={loss}, jac_method={jac_method}')
                print()
                fit = nlls('[y] ~ {Intercept} + np.exp({alpha} * [x]) + {beta}*[x**2]'
                           ' + [C(g)]*[z]'
                           ' + .1 *[z]**(1+{gamma})'
                           + (' $ [w]' if weighted else '')
                           , df, ftol=1e-14, gtol=1e-14, xtol=1e-14,
                           max_iter=1000,
                           jac_method=jac_method,
                           dense_threshold_mb=1024,
                           bounds=bounds, **loss)

                qs = QuantileSmooth()
                loss_func = lambda x: (
                    x
                    if loss.get('root_loss_function', None) is None else
                    qs(x)[0]
                )
                print(loss_func)
                print(qs([1,2]))
                print(loss_func([1,2]))

                fit_scipy = least_squares(lambda p: (fit.model.weights ** .5 if weighted else 1.0)
                                                    * loss_func(fit.model.residual_function_callable(p)),
                                          fit.params.values,
                                          f_scale=1.0,
                                          ftol=1e-14, gtol=1e-14, xtol=1e-14,
                                          **({'bounds': bounds.T} if bounds is not None else dict()))

                df_coef = pd.DataFrame({
                    'kanly': fit.params.values,
                    'scipy': fit_scipy.x,
                    'diff': np.where(np.abs(fit_scipy.x)>1,
                                     np.abs(fit.params.values / fit_scipy.x - 1),
                                     np.abs(fit.params.values - fit_scipy.x)
                                     )
                }, index=fit.params.index)

                fun_kanly= sum((fit.model.weights if weighted else 1.0) * fit.model.residual_function_callable(fit.params) ** 2) / 2
                fun_scipy = sum((fit.model.weights if weighted else 1.0) * fit.model.residual_function_callable(fit_scipy.x) ** 2) / 2

                print(fit)
                print(fit_scipy)

                print(df_coef['diff'])
                try:
                    assert np.max(df_coef['diff']) < 1e-3
                except:
                    try:
                        print(f"PARAM MATCH FAILED kanly={fun_kanly}, scipy={fun_scipy}")
                        assert fun_kanly <= fun_scipy
                    except:
                        failed.append((weighted, bounds, loss, jac_method))
                        print("FAILED")

    print("#" * 100)
    print('num failed = ', len(failed))
    print(failed)
