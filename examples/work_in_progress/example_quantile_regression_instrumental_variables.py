import pandas as pd

from kanly.api import qr, compare_results
import numpy as np

np.random.seed(0)

n = 300
Z = np.random.randn(n, 3)
e = np.random.randn(n)
X = np.array([[4, -3]]) + Z.dot(np.random.randn(3, 2)) \
    + e.reshape((n, 1)).dot(np.array([[7.5, .7]])) + 1.2 * np.random.randn(n, 2)
y = X.dot([1., 2.]) + 10.6 * e + 15 + 3 * (np.exp(np.random.randn(n)) - np.exp(.5)) * (1 + .3 * np.abs(np.sum(X, axis=1)))

df = pd.DataFrame()
for j in range(2):
    df[f'x{j}'] = X[:,j]
for j in range(3):
    df[f'z{j}'] = Z[:,j]
df['y'] = y

tau = .95

cov_info = {'cov_type': 'bootstrap', 'cov_kwds': {'n_samples': 250}}
cov_info = {}

fit_qr = qr('y ~ x0 + x1', df, tau=tau, **cov_info)

fit_qr_iv_no_ri = qr('y ~ x0 + x1 | z0 + z1 + z2', df, tau=tau, residual_inclusion=False, **cov_info)

fit_qr_iv = qr('y ~ x0 + x1 | z0 + z1 + z2', df, tau=tau,
               residual_inclusion=True, residual_inclusion_order=2, **cov_info)

print(fit_qr_iv)
print(fit_qr_iv_no_ri)
print(fit_qr)

print(compare_results([fit_qr, fit_qr_iv_no_ri, fit_qr_iv],
                   fit_titles=['QR', 'QR-IV', 'QR-IV-RI'], ref_param_values={'x0': 1, 'x1': 2}))