import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from statsmodels.tsa.arima.model import ARIMA as ARIMA_SM

from kanly.api import arima
from kanly.api import simulate_sarima

fail = 0
cnt = 0

n = 2065
np.random.seed(0)

y = simulate_sarima(n, [.64, .1], [.15], d=2, sigma2=.4, seed=0)

X = np.random.randn(n, 2)
y += X.dot(1 + np.arange(X.shape[1]))

df = pd.DataFrame(dict(y=y, x=X[:, 0], z=X[:, 1]))

for (p, d, q), trend, exog, from_formula in [
    ((0, 0, 0), 'n', None, True),
    ((0, 0, 0), 'n', X, True),
    ((2, 2, 0), 'n', None, True),
    ((2, 2, 0), 'n', X, True),
    ((2, 2, 1), 'n', None, True),
    ((2, 2, 1), 'n', X, True),
    ((0, 0, 0), [0, 0, 1], None, True),
    ((0, 0, 0), [0, 0, 1], X, True),
    ((2, 2, 0), [0, 0, 1], None, True),
    ((2, 2, 0), [0, 0, 1], X, True),
    ((2, 2, 1), [0, 0, 1], None, True),
    ((2, 2, 1), [0, 0, 1], X, True),
]:

    print('\n' * 1)
    print('-' * 100)
    print(((p, d, q), trend, exog is None, from_formula))

    if from_formula:
        fitka3 = arima('y ~ x + z', df, order=(p, d, q), trend=trend,
                       gtol=1e-5, xtol=1e-100, ftol=1e-100, nlags=200)

    fitsm3 = ARIMA_SM(
        y, exog=X, order=(p, d, q), trend=trend,
        #enforce_stationarity=False, enforce_invertibility=False
    ).fit(
        method='innovations_mle'
    )

    try:
        assert_allclose(fitka3.params, fitsm3.params, rtol=1e-3)
        print(fitka3.params.values, fitsm3.params, fitka3.params.values/fitsm3.params-1)
        # assert_allclose(fitka3.bse, fitsm3.bse, rtol=1e-3)

        # for a in ['nobs', 'llf', 'loglikelihood_burn', 'aic', 'bic', 'hqic', 'aicc']:
        #     assert_allclose(
        #         getattr(fitka3, a),
        #         getattr(fitsm3, a),
        #         rtol=1e-3,
        #     )

        # for a in ['resid', 'fittedvalues', 'llf_obs']:
        #     assert_allclose(
        #         getattr(fitka3, a)[d:],
        #         getattr(fitsm3, a)[d:],
        #         rtol=1e-3,
        #     )
    except:
        print('\tFAIL', fitka3.llf - fitsm3.llf)
        print(fitka3)
        print(fitsm3.summary())
        fail += 1
    cnt += 1

print('='*100)
print(f'\n{fail=}/{cnt=}')