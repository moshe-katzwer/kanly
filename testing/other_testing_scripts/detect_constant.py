from kanly.api import lm
import pandas as pd
import numpy as np
from scipy.sparse.linalg import inv
from scipy.sparse import csc_matrix
from kanly.utils.util import csc_matrix_by_column_array_broadcast

n = 100
np.random.seed(0)
df = pd.DataFrame({
    'y': np.random.randn(n),
    'c': np.random.randint(0, 5, n),
    'w': np.random.rand(n)+.1
})

fit = lm('y ~ C(c) - 1 $ w', df)

print(fit)


def detect_constant(X, weights=None, tol=1e-8):
    nobs = X.shape[0]
    if weights is None:
        _ones = csc_matrix(np.ones((n, 1)))
    else:
        X = csc_matrix_by_column_array_broadcast(X, weights)
        _ones = csc_matrix(weights).reshape((-1,1))
    ncv = inv(X.transpose().dot(X))
    beta = ncv.dot(X.transpose().dot(_ones))
    pred = X.dot(beta)
    print(beta.toarray().flatten())
    return sum(np.abs((pred - _ones).toarray()) > tol)


print(detect_constant(fit.model.exog))