import numpy as np
from scipy.sparse import csc_matrix, isspmatrix, hstack as sp_hstack


def add_constant(X):
    """Adds a constant to a np.array or scipy sparse matrix"""
    if isspmatrix(X):
        n = X.shape[0]
        return sp_hstack([csc_matrix(np.ones(n, 1)), X])
    else:
        if np.ndim(X) == 1:
            X = X.reshape((-1, 1))
        n = X.shape[0]
        return np.hstack([np.ones((n, 1)), X])
