"""PLS1 regression with a single response.

This example fits the same three-component model through the sparse array API
and the formula API, then predicts held-out observations with the fitted
``PlsRegressionResults`` object.
"""

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix

from kanly.api import PLS1, pls1


rng = np.random.default_rng(7)
n_samples, n_features, n_components = 300, 12, 3

latent_scores = rng.normal(size=(n_samples, n_components))
X = (
    latent_scores @ rng.normal(size=(n_components, n_features))
    + 0.2 * rng.normal(size=(n_samples, n_features))
)
y = (
    2.5
    + latent_scores @ np.array([2.0, -1.5, 0.75])
    + 0.3 * rng.normal(size=n_samples)
)

train = np.arange(240)
test = np.arange(240, n_samples)
feature_names = [f"x{j}" for j in range(n_features)]

# The array API accepts a dense or SciPy sparse predictor matrix.
fit = PLS1(
    y[train],
    csr_matrix(X[train]),
    l=n_components,
    exog_names=feature_names,
    endog_name="y",
    specification_name="Sparse array PLS1",
)
predictions = fit.predict(csr_matrix(X[test]))
test_r_squared = 1.0 - np.sum((y[test] - predictions) ** 2) / np.sum(
    (y[test] - y[test].mean()) ** 2
)

# PLS1 also has a formula entry point. It returns the same result type.
train_data = pd.DataFrame(X[train], columns=feature_names)
train_data["y"] = y[train]
formula_fit = pls1(
    "y ~ " + " + ".join(feature_names),
    train_data,
    l=n_components,
    compute_cov=False,
)
assert_allclose(formula_fit.fittedvalues, fit.fittedvalues, rtol=1e-10, atol=1e-10)

print(fit.summary())
print(f"Held-out R-squared: {test_r_squared:.4f}")
print("Score matrix shape:", fit.T.shape)
print("Coefficient shape:", fit.coef.shape)