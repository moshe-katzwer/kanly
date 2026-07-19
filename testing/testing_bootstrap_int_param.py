from kanly.api import GLM, clear_timers, timer, add_constant, LM, GLM, RLM, NLLS, QR, GMM, ELASTIC_NET
import numpy as np

clear_timers()

np.random.seed(0)
n = 10_000

k = 10
X = np.random.randn(n,k)
X = X @ np.random.randn(k, k)
y = -.5 + X.dot(np.ones(k)) + np.random.randn(n)
X = add_constant(X)

y0 = np.exp(y / y.max())

for func in [LM, GLM, RLM, QR]: #, ELASTIC_NET]:

    if func == QR:
        kwargs = {'tau': .5}
    elif func == ELASTIC_NET:
        kwargs = {'l1_ratio': 0}
    else:
        kwargs = dict()

    f = func(y0, X, cov_type='bootstrap(20)', **kwargs)
    print(f)
    assert f.bootstrapped_params.shape[0] == 20

f = NLLS(y, lambda params: params[0] + params[1] * X[:, 1], num_params=2, cov_type='bootstrap(20)')
print(f)
assert f.bootstrapped_params.shape[0] == 20

# TODO
f = GMM(
    lambda params: np.column_stack([
        (y - (params[0] + params[1] * X[:, 1])),
        ((y - (params[0] + params[1] * X[:, 1])) * X[:, 1]),
    ]),
    num_moments=2,
    num_params=2,
    nobs=len(y),
    cov_type='bootstrap(20)'
)
print(f)
assert f.bootstrapped_params.shape[0] == 20

print("ALL PASSED!")
