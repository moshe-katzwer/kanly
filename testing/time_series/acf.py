import numpy as np
from kanly.api import acf as acf_kanly, pacf as pacf_kanly
from statsmodels.tsa.stattools import acf as acf_sm, pacf as pacf_sm
from numpy.testing import assert_allclose

rand = np.random.RandomState(0)
T = 50_000
y = np.zeros(T)
phi = [.3, .1, .1, .1]
theta = [.5]
e = 3*rand.randn(T)
p = len(phi)
q = len(theta)
y[:p] = e[:p]
for j in range(p, T):
    y[j] = np.dot(y[j-p:j], phi[::-1]) + e[j] + np.dot(e[j-q:j], theta[::-1])

acf_vals1 = acf_kanly(y, nlags=15)
acf_vals2 = acf_sm(y, nlags=15, adjusted=True)

pacf_vals1 = pacf_kanly(y, nlags=15)
pacf_vals2 = pacf_sm(y, nlags=15, method='ywmle')

assert_allclose(acf_vals1, acf_vals2, atol=1e-4, rtol=1e-2)
assert_allclose(pacf_vals1, pacf_vals2, atol=1e-4, rtol=1e-2)
