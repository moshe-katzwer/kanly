from kanly.api import QR, RLM, GLM, LM, ELASTIC_NET, GMM
import numpy as np

n = 1000
Z = np.random.randn(n, 8)
X = Z.dot(np.random.randn(8, 4)) + .2 * np.random.randn(n, 4)
x = X[:, 0]
y = X.dot([1, 2, 3, 4]) + .3 * np.random.randn(n) + 15
X = np.hstack((np.ones((n, 1)), X))
Z = np.hstack((np.ones((n, 1)), Z))
y[0] += 1000
tau = .1

print(QR(y, X, tau, has_constant=True))

print(LM(y, X, has_constant=True))
print(LM(y, X, has_constant=True, instruments=Z))

print(RLM(y, X, has_constant=True))

print(ELASTIC_NET(y, X, alpha=.1, fit_intercept=False))

print(GLM(y, X, family='poisson', debug=True))
print(GLM(y, X, instruments=Z, residual_inclusion=False, family='poisson', debug=True))
print(GLM(y, X, instruments=Z, residual_inclusion=True, family='poisson', debug=True))

print('=' * 100)

for add_constant in (True, False):
    print(LM(y, x, has_constant=False, add_constant=add_constant))
    print(GLM(y, x, add_constant=add_constant))
    print(RLM(y, x, has_constant=False, add_constant=add_constant))
    print(QR(y, x, has_constant=False, add_constant=add_constant, tau=.7))


def moment_func(params):
    resid = y - (params[0] + X[:, 1]*params[1] + X[:,2]*params[2])
    return np.column_stack([
        resid,
        resid*X[:,0],
        resid*X[:,1]
    ])


print(GMM(moment_func, num_moments=3, num_params=3, nobs=len(y)))

print("ALL PASSED!")