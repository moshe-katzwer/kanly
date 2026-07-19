import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from kanly.api import LM
from numpy.testing import assert_allclose
from kanly.regression.partial_least_squares.pls import PLS1, pls1, PLS2
from sklearn.cross_decomposition import PLSRegression

np.random.seed(0)
n = 1000
k = 10
l = 10
T = np.exp(np.random.randn(n, l))
X = T.dot(np.random.randn(l, k)) + .3 * np.random.randn(n, k)
y = T.dot(np.random.randn(l)) + 4 * np.random.randn(n) + 120

Xsp = csc_matrix(X)

df = pd.DataFrame(X, columns=[f'xx{d}' for d in range(X.shape[1])])
df['yy'] = y
formula0 = 'yy ~ ' + ' + '.join([f'xx{d}' for d in range(X.shape[1])])

# Test OLS case

for center in [True, False]:
    fit1 = PLS1(y, X, l, center=center)
    fit2 = PLS1(y, Xsp, l, center=center)

    formula = formula0 + ("" if center else "-1")
    fit3 = pls1(formula, df, l)

    fitlm = LM(y, X, add_constant=center, cov_type='nonrobust')

    assert_allclose(fit1.params[1-int(center):], fitlm.params)
    assert_allclose(fit2.params[1-int(center):], fitlm.params)
    assert_allclose(fit3.params[1-int(center):], fitlm.params)

    for f in [fit1, fit2, fit3]:
        assert_allclose(f.cov_params().values[1-int(center):,1-int(center):],
                        fitlm.cov_params())

# Test PLS case
np.random.seed(0)
n = 1000
k = 500
l = 10
T = np.exp(np.random.randn(n, l))
X = T.dot(np.random.randn(l, k)) + .3 * np.random.randn(n, k)
y = T.dot(np.random.randn(l)) + 4 * np.random.randn(n) + 120

Xsp = csc_matrix(X)

fit_sk = PLSRegression(n_components=l, scale=False).fit(X, y)
fit1 = PLS1(y, X, l)
fit2 = PLS1(y, Xsp, l)

for f in [fit1, fit2]:
    assert_allclose(f.predict(X), fit_sk.predict(X))

# test PLS2

np.random.seed(0)
n = 1000
k = 12
l = 3
T = np.exp(np.random.randn(n, l))
X = T.dot(np.random.randn(l, k)) + .3 * np.random.randn(n, k)
y = T.dot(np.random.randn(l)) + 4 * np.random.randn(n) + 120


for center in [True, False]:
    fit1 = PLS1(y, X, 4, center=center)
    fit2 = PLS2(y.reshape((-1,1)), X, 4, center=center)
    assert_allclose(fit1.coef, fit2['coef'].flatten())
    assert_allclose(fit1.intercept, fit2['intercept'])
    assert_allclose(fit1.fittedvalues, fit2['fittedvalues'].ravel())

    fit3 = PLS2(np.vstack([y,y]).T, X, 4, center=center)
    for c in [0,1]:
        # print(center, c)
        assert_allclose(fit1.coef, fit3['coef'][:,c])
        assert_allclose(fit1.fittedvalues, fit3['fittedvalues'][:,c])
        assert_allclose(fit1.intercept, fit3['intercept'][c])
