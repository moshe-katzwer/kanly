"""PLS2 regression with multiple responses and sparse matrices.

PLS2 extracts shared latent components for every response column. This example
fits dense and CSR versions of the same model, verifies numerical parity, and
uses the returned ``PlsRegressionResults`` object for held-out prediction.
"""

import numpy as np
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix

from kanly.api import PLS2



rng = np.random.default_rng(11)
n_samples, n_features, n_targets = 320, 10, 3

X = rng.normal(size=(n_samples, n_features))
X[rng.random(X.shape) < 0.7] = 0.0
true_coef = rng.normal(size=(n_features, n_targets))
true_coef[rng.random(true_coef.shape) < 0.55] = 0.0
Y = X @ true_coef + 0.1 * rng.normal(size=(n_samples, n_targets))
Y[np.abs(Y) < 0.15] = 0.0

train = np.arange(260)
test = np.arange(260, n_samples)
feature_names = [f"x{j}" for j in range(n_features)]
target_names = ["sales", "margin", "retention"]

dense_fit = PLS2(
    Y[train],
    X[train],
    l=4,
    exog_names=feature_names,
    endog_names=target_names,
    specification_name="Dense PLS2",
)
sparse_fit = PLS2(
    csr_matrix(Y[train]),
    csr_matrix(X[train]),
    l=4,
    exog_names=feature_names,
    endog_names=target_names,
    specification_name="Sparse PLS2",
)

assert_allclose(sparse_fit.coef, dense_fit.coef, rtol=1e-10, atol=1e-10)
assert_allclose(
    sparse_fit.fittedvalues,
    dense_fit.fittedvalues,
    rtol=1e-10,
    atol=1e-10,
)

predictions = sparse_fit.predict(csr_matrix(X[test]))
test_r_squared = 1.0 - np.sum((Y[test] - predictions) ** 2, axis=0) / np.sum(
    (Y[test] - Y[test].mean(axis=0)) ** 2,
    axis=0,
)

print(sparse_fit.summary())
print("Held-out R-squared by response:", test_r_squared.round(4))
print("Score matrix shape:", sparse_fit.T.shape)
print("Coefficient matrix shape:", sparse_fit.coef.shape)
print("NIPALS iterations by component:", sparse_fit.n_iter)
