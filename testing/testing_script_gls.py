import numpy as np
import pandas as pd
import scipy as sp
from numpy.linalg import eigvalsh
from statsmodels.regression.linear_model import RegressionResults
from numpy.testing import assert_allclose

from kanly.api import LM, LM_model
from kanly.regression.linear_models.lm_internal import lm_internal, fgls_internal
from statsmodels.api import GLS

from kanly.regression.linear_models.regression_results import SparseLinearRegressionResults
from kanly.regression.regression_results_base import RegressionResultsBase

n = 55
k = 4
np.random.seed(1)
X = np.random.randn(n, k)
beta = np.random.randn(k)
y = X.dot(beta) + .5 * np.random.rand(n)
W = 1.74 * (np.eye(n))
for offset in range(1,3):
    W += np.eye(n, k=offset, dtype=int) * .2
    W += np.eye(n, k=-offset, dtype=int) * .2
W *= 4

Winv = np.linalg.pinv(W)


print(f'{(X.T.dot(Winv).dot(y))[:5]=}')

ncp = np.linalg.inv(X.T.dot(Winv).dot(X))
by_hand = ncp.dot(X.T.dot(Winv).dot(y))

fit: SparseLinearRegressionResults = LM_model(y, X).fit(sigma_inv=Winv, cov_type='ols-small')
print(fit)

fit_sm: RegressionResults = GLS(y, X, sigma=W).fit()
print(fit_sm.summary())

print(pd.DataFrame({
    'by_hand': by_hand,
    'kanly': fit.params,
    'sm': fit_sm.params,
}))


print(fit.scale)
print(np.mean(fit.resid ** 2))
print((fit_sm.resid**2).mean())

print(fit.normalized_cov_params / fit_sm.normalized_cov_params)
print(fit.cov_params() / fit_sm.cov_params())

print(('wssr', fit.wssr, fit_sm.ssr))

print(('uncentered_tss', y.dot(Winv).dot(y), fit_sm.uncentered_tss))
print(np.dot(y - y.mean(), Winv).dot(y - y.mean()))  # TODO

s = fit.scale * (n - k) / n
s = fit.scale_mle
print('scale_MLE', s, fit.scale_mle)

print(('scale', fit.scale, fit_sm.scale))
print(('llf', fit.llf, fit_sm.llf))
print(-0.5 * np.dot(fit.resid, Winv/s).dot(fit.resid) + 0.5 * np.linalg.slogdet(Winv/s)[1] - (n / 2) * np.log(2 * np.pi))
print("***** ", np.dot(fit.resid, Winv/s).dot(fit.resid))

print(np.corrcoef(fit.resid, fit_sm.resid))
print(np.corrcoef(fit.wresid, fit_sm.wresid))
print(np.corrcoef(fit_sm.wresid, np.dot(sp.linalg.sqrtm(Winv), fit_sm.resid)))
print(np.corrcoef(fit.fittedvalues, fit_sm.fittedvalues))

# print(fit_sm.wresid)
# print(fit_sm.resid)

# print(pd.DataFrame(
#     sp.linalg.sqrtm(Winv) @ sp.linalg.sqrtm(Winv) - Winv
# ).round(6).max().max()
# )

num_failed = 0

for x in ['llf', 'scale', 'params', 'bse', 'aic', 'bic', 'pvalues',
          ('wssr', 'ssr'),
          ('wsst', 'centered_tss'),
          'uncentered_tss',
          'condition_number',
          'tvalues', 'fvalue', 'resid', 'fittedvalues',
          'rsquared', 'rsquared_adj']:
    if isinstance(x, tuple):
        a1, a2 = x
    else:
        a1 = a2 = x
    print(x,end='...')
    try:
        assert_allclose(getattr(fit, a1), getattr(fit_sm, a2))
        print('passed')
    except Exception as e:
        print('failed!!')
        print(e)
        print()
        num_failed += 1

ncp = np.linalg.inv(X.T.dot(X))
eigenvals = eigvalsh(ncp)

if True:
    eigenvals = 1.0 / eigenvals

eigenvals = sorted(eigenvals)[::-1]
if eigenvals[-1] <= 0:
    condition_number = np.inf
else:
    condition_number = np.sqrt(eigenvals[0] / eigenvals[-1])

print(condition_number)

