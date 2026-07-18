import numpy as np
from scipy import sparse

from kanly.regression.partial_least_squares.pls import PLS2


def SPARSE_PLS2(Y, X, l, max_iter=100, tol=1e-6, convergence_error='silent'):
    """
    Sparse-preserving PLS2 using NIPALS algorithm.

    Parameters
    ----------
    Y : array-like or sparse matrix, shape (n_samples, n_targets)
        Target matrix (can be sparse or dense)
    X : array-like or sparse matrix, shape (n_samples, n_features)
        Predictor matrix (can be sparse or dense)
    l : int
        Number of PLS components to extract

    Returns
    -------
    dict with keys:
        'T': X scores, shape (n_samples, l)
        'U': Y scores, shape (n_samples, l)
        'P': X loadings, shape (n_features, l)
        'Q': Y loadings, shape (n_targets, l)
        'W': X weights, shape (n_features, l)
        'C': Y weights, shape (n_targets, l)
        'B_pls': PLS regression coefficients, shape (n_features, n_targets)
        'X_mean': mean of X columns
        'Y_mean': mean of Y columns
    """
    # Convert to appropriate format and get dimensions
    X_is_sparse = sparse.issparse(X)
    Y_is_sparse = sparse.issparse(Y)

    if X_is_sparse:
        X = X.tocsc()  # CSC format for efficient column operations
    else:
        X = np.asarray(X)

    if Y_is_sparse:
        Y = Y.tocsc()
    else:
        Y = np.asarray(Y)

    n_samples, n_features = X.shape
    _, n_targets = Y.shape

    # Compute means without densifying
    if X_is_sparse:
        X_mean = np.asarray(X.mean(axis=0)).ravel()
    else:
        X_mean = X.mean(axis=0)

    if Y_is_sparse:
        Y_mean = np.asarray(Y.mean(axis=0)).ravel()
    else:
        Y_mean = Y.mean(axis=0)

    # Initialize storage for components
    T = np.zeros((n_samples, l))  # X scores
    U = np.zeros((n_samples, l))  # Y scores
    P = np.zeros((n_features, l))  # X loadings
    Q = np.zeros((n_targets, l))  # Y loadings
    W = np.zeros((n_features, l))  # X weights
    C = np.zeros((n_targets, l))  # Y weights

    # Working copies (we'll deflate these)
    X_work = X.copy()
    Y_work = Y.copy()

    # NIPALS algorithm
    for comp in range(l):
        # Initialize u as first column of Y_work (after mean centering)
        if Y_is_sparse:
            u = Y_work[:, 0].toarray().ravel() - Y_mean[0]
        else:
            u = Y_work[:, 0] - Y_mean[0]

        for iteration in range(max_iter):
            # 1. Compute X weights w (without densifying)
            # w = X.T @ u, but with mean centering
            if X_is_sparse:
                w = X_work.T @ u - X_mean * u.sum()
            else:
                w = (X_work - X_mean).T @ u

            w = w / np.linalg.norm(w)

            # 2. Compute X scores t
            # t = X @ w, but with mean centering
            if X_is_sparse:
                t = X_work @ w - X_mean @ w
            else:
                t = (X_work - X_mean) @ w

            # 3. Compute Y weights c
            # c = Y.T @ t, but with mean centering
            if Y_is_sparse:
                c = Y_work.T @ t - Y_mean * t.sum()
            else:
                c = (Y_work - Y_mean).T @ t

            c = c / np.linalg.norm(c)

            # 4. Compute Y scores u_new
            # u_new = Y @ c, but with mean centering
            if Y_is_sparse:
                u_new = Y_work @ c - Y_mean @ c
            else:
                u_new = (Y_work - Y_mean) @ c

            # Check convergence
            if np.linalg.norm(u_new - u) < tol:
                u = u_new
                break

            u = u_new

        print(iteration)

        # Compute loadings
        # p = X.T @ t / (t.T @ t), with mean centering
        t_norm_sq = t @ t
        if X_is_sparse:
            p = (X_work.T @ t - X_mean * t.sum()) / t_norm_sq
        else:
            p = (X_work - X_mean).T @ t / t_norm_sq

        # q = Y.T @ u / (u.T @ u), with mean centering
        u_norm_sq = u @ u
        if Y_is_sparse:
            q = (Y_work.T @ u - Y_mean * u.sum()) / u_norm_sq
        else:
            q = (Y_work - Y_mean).T @ u / u_norm_sq

        # Store components
        T[:, comp] = t
        U[:, comp] = u
        P[:, comp] = p
        Q[:, comp] = q
        W[:, comp] = w
        C[:, comp] = c

        # Deflate X and Y (preserve sparsity)
        # X_work = X_work - t @ p.T (in centered space)
        deflation_X = np.outer(t, p)
        if X_is_sparse:
            X_work = X_work - sparse.csr_matrix(deflation_X)
        else:
            X_work = X_work - deflation_X

        # Y_work = Y_work - t @ q.T (in centered space)
        deflation_Y = np.outer(t, q)
        if Y_is_sparse:
            Y_work = Y_work - sparse.csr_matrix(deflation_Y)
        else:
            Y_work = Y_work - deflation_Y

    # Compute PLS regression coefficients
    # B_pls = W @ (P.T @ W)^{-1} @ Q.T
    W_ortho = W @ np.linalg.pinv(P.T @ W)
    B_pls = W_ortho @ Q.T

    return {
        'T': T,
        'U': U,
        'P': P,
        'Q': Q,
        'W': W,
        'C': C,
        'B_pls': B_pls,
        'X_mean': X_mean,
        'Y_mean': Y_mean
    }


def predict_pls2(model, X_new):
    """
    Make predictions using fitted PLS2 model.

    Parameters
    ----------
    model : dict
        Output from SPARSE_PLS2
    X_new : array-like or sparse matrix, shape (n_samples_new, n_features)
        New predictor matrix

    Returns
    -------
    Y_pred : array, shape (n_samples_new, n_targets)
        Predicted target values
    """
    X_is_sparse = sparse.issparse(X_new)

    if X_is_sparse:
        X_centered = X_new - model['X_mean']
        Y_pred = X_centered @ model['B_pls']
        if sparse.issparse(Y_pred):
            Y_pred = Y_pred.toarray()
    else:
        X_centered = X_new - model['X_mean']
        Y_pred = X_centered @ model['B_pls']

    Y_pred = Y_pred + model['Y_mean']

    return Y_pred


if __name__ == "__main__":
    # Example usage with sparse matrices
    np.random.seed(42)

    # Create sparse data
    n_samples = 100
    n_features = 10
    n_targets = 3
    density = 0.1

    # Generate sparse X
    X_dense = np.random.randn(n_samples, n_features)
    X_dense[np.random.rand(n_samples, n_features) > density] = 0
    X_sparse = sparse.csr_matrix(X_dense)

    # Generate Y with some relationship to X
    true_coef = np.random.randn(n_features, n_targets)
    true_coef[np.random.rand(n_features, n_targets) > 0.2] = 0
    Y_dense = X_dense @ true_coef + 0.1 * np.random.randn(n_samples, n_targets)
    Y_dense[np.random.rand(n_samples, n_targets) > 0.3] = 0
    Y_sparse = sparse.csr_matrix(Y_dense)

    print("X shape:", X_sparse.shape)
    print("Y shape:", Y_sparse.shape)
    print("X sparsity:", 1 - X_sparse.nnz / (X_sparse.shape[0] * X_sparse.shape[1]))
    print("Y sparsity:", 1 - Y_sparse.nnz / (Y_sparse.shape[0] * Y_sparse.shape[1]))

    # Fit PLS2 model
    print("\nFitting PLS2 with 5 components...")
    model = SPARSE_PLS2(Y_sparse, X_sparse, l=5)

    print("\nModel components:")
    print("T (X scores) shape:", model['T'].shape)
    print("U (Y scores) shape:", model['U'].shape)
    print("P (X loadings) shape:", model['P'].shape)
    print("Q (Y loadings) shape:", model['Q'].shape)
    print("W (X weights) shape:", model['W'].shape)
    print("C (Y weights) shape:", model['C'].shape)
    print("B_pls (regression coefficients) shape:", model['B_pls'].shape)

    # Make predictions
    Y_pred = predict_pls2(model, X_sparse)

    # Calculate R^2
    if sparse.issparse(Y_sparse):
        Y_true = np.asarray(Y_sparse.toarray())
    else:
        Y_true = np.asarray(Y_sparse)

    Y_pred = np.asarray(Y_pred)

    ss_res = np.sum((Y_true - Y_pred) ** 2)
    ss_tot = np.sum((Y_true - Y_true.mean(axis=0)) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"\nR^2 score: {r2:.4f}")

    # Test with dense matrices too
    print("\n" + "=" * 50)
    print("Testing with dense matrices...")
    L = 5
    model_dense = SPARSE_PLS2(Y_dense, X_dense, l=L, max_iter=1000)
    Y_pred_dense = predict_pls2(model_dense, X_dense)

    Y_dense_array = np.asarray(Y_dense)
    Y_pred_dense = np.asarray(Y_pred_dense)

    ss_res_dense = np.sum((Y_dense_array - Y_pred_dense) ** 2)
    ss_tot_dense = np.sum((Y_dense_array - Y_dense_array.mean(axis=0)) ** 2)
    r2_dense = 1 - ss_res_dense / ss_tot_dense

    print(f"R^2 score (dense): {r2_dense:.4f}")

    # from kanly.api import PLS2
    from matplotlib import pyplot as plt

    plt.scatter(PLS2(Y_dense_array, X_dense, l=L, center=True)['fittedvalues'], Y_pred_dense)
    plt.show()
    print(PLS2(Y_dense_array, X_dense, l=L, center=True)['fittedvalues'] - Y_pred_dense)
