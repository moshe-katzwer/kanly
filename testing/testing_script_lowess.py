import numpy as np
from kanly.api import LOWESS
import matplotlib.pyplot as plt
for noise in [.3, 0]:

    np.random.seed(0)
    n = 1000
    x = np.random.randint(3, 20, n) + np.random.randn(n) * noise
    y = np.log(x) + .3 * np.random.randn(n)
    for j in np.random.randint(0, n, 6):
        y[j] += 4

    for deg in range(4):
        for frac in [.05, .1, .3]:
            plt.figure()
            plt.title(f'degree = {deg}, frac = {frac}')
            plt.scatter(x, y, c='cyan')
            for xvals in [None, 15]:
                for q in [True, False]:
                    for it in [0, 1, 2]:
                        print(f'{noise=}, {frac=}, {deg=}, {it=}, {q=}, {xvals=}', end='...')
                        try:
                            xhat, yhat = LOWESS(y, x, xvals=xvals, it=it, do_xval_quantiles=q, degree=deg, frac=frac,
                                                do_njit=True)
                            assert np.all(~np.isnan(xhat))
                            assert np.all(~np.isnan(yhat))
                            print('Succeeded')
                        except Exception as e:
                            print('Failed')
                            raise e
                        plt.plot(xhat, yhat, marker='o', label=f'{it=}, {xvals=}, {q=}', ls=':' if q else '-', alpha=.5)
            plt.legend(loc='best')