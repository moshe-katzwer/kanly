from kanly.api import func_str_to_callable, bfgs_pqn
import matplotlib.pyplot as plt
from numpy.testing import assert_allclose

a, b = 1.1, 1000.0

for maximize in [True, False]:

    func = func_str_to_callable(f'{"-" if maximize else ""}(({a}-{{x}})**2 + {b}*({{y}}-{{x}}**2)**2)')
    grad_ = func.get_analytical_gradient()
    hess_, hess_info = func.get_analytical_hessian(return_info=True)
    # print(pprint.pformat(hess_info))

    print('\n', maximize, '\n')
    for g, H in [(None, None), (grad_, None), (grad_, hess_)]:
        result = bfgs_pqn(func, [1.6, -2.2],
                          gradient_callable=g,
                          hessian_callable=H,
                          maxiter=1000, xtol=1e-15, gtol=1e-15, ftol=1e-15,
                          debug=False, maximize=maximize,
                          save_optimization_path=True
                          )
        print(result.iter, result.fun, result.x)
        #plt.scatter(result.optimization_path[:, 0], result.optimization_path[:, 1], ls='-', lw=2, marker='.')
        #plt.show()
        assert_allclose(result.x, [a, a**2])
