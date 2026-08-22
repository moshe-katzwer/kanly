"""Sparse-aware PLS2 regression.

The implementation in this module keeps sparse input matrices sparse. Centering
and component deflation are represented as matrix-vector operations instead of
materialising centered or deflated matrices, both of which are generally dense.
"""

import warnings

import numpy as np
from scipy import sparse


def _as_float_matrix(value, name):
    """Return a two-dimensional dense array or CSR matrix of floats."""
    if sparse.issparse(value):
        value = value.astype(float, copy=False).tocsr()
        is_finite = np.all(np.isfinite(value.data))
    else:
        value = np.asarray(value, dtype=float)
        is_finite = np.all(np.isfinite(value))

    if value.ndim != 2:
        raise ValueError(f"{name} must be 2-dimensional, got shape {value.shape}")
    if not is_finite:
        raise ValueError(f"{name} must contain only finite values")
    return value


def _column_mean(value):
    return np.asarray(value.mean(axis=0)).ravel()


def _matvec(value, mean, vector, scores, loadings, component, center):
    """Compute a centered, implicitly deflated matrix-vector product."""
    result = np.asarray(value @ vector).ravel()
    if center:
        result -= mean @ vector
    if component:
        result -= scores[:, :component] @ (loadings[:, :component].T @ vector)
    return result


def _transpose_matvec(value, mean, vector, scores, loadings, component, center):
    """Compute a transposed centered/deflated matrix-vector product."""
    result = np.asarray(value.T @ vector).ravel()
    if center:
        result -= mean * np.sum(vector)
    if component:
        result -= loadings[:, :component] @ (scores[:, :component].T @ vector)
    return result


def _initial_y_score(Y, Y_mean, T, Q, component, center):
    """Use the first non-constant residual response column as a Y score."""
    for column in range(Y.shape[1]):
        if sparse.issparse(Y):
            score = Y[:, column].toarray().ravel()
        else:
            score = np.asarray(Y[:, column]).ravel().copy()
        if center:
            score -= Y_mean[column]
        if component:
            score -= T[:, :component] @ Q[column, :component]
        if np.linalg.norm(score) > np.finfo(float).eps:
            return score
    raise ValueError(
        f"Y residual is constant or zero at component {component}; "
        "cannot extract another PLS component."
    )


def SPARSE_PLS2(
        Y,
        X,
        l,
        max_iter=100,
        tol=1e-6,
        convergence_error="silent",
        center=True,
):
    """Fit a PLS2 model to dense and/or SciPy sparse matrices.

    Parameters
    ----------
    Y : array-like or sparse matrix, shape (n_samples, n_targets)
        Target matrix.
    X : array-like or sparse matrix, shape (n_samples, n_features)
        Predictor matrix.
    l : int
        Number of PLS components to extract.
    max_iter : int, default=100
        Maximum number of NIPALS power iterations per component.
    tol : float, default=1e-6
        Convergence tolerance for the squared difference between successive X
        weight vectors.
    convergence_error : {'silent', 'warn', 'raise'}, default='silent'
        Action to take when a component does not converge within ``max_iter``.
    center : bool, default=True
        Center X and Y implicitly. Sparse matrices are never explicitly
        centered, since doing so would make them dense.

    Returns
    -------
    dict
        The score, loading, and weight matrices ``T``, ``U``, ``P``, ``Q``,
        ``W``, and ``C``; coefficients under both ``coef`` and the legacy
        ``B_pls`` key; ``intercept``; column means; and per-component
        convergence information.

        Predictions are ``X @ coef + intercept``.

    Notes
    -----
    The residual matrices are represented as low-rank corrections::

        X_h = X_centered - T_h @ P_h.T
        Y_h = Y_centered - T_h @ Q_h.T

    Consequently, the input matrices retain their sparse representation and
    the only dense allocations scale with scores/loadings, not with the full
    centered or deflated input matrices.
    """
    X = _as_float_matrix(X, "X")
    Y = _as_float_matrix(Y, "Y")

    n_samples, n_features = X.shape
    if Y.shape[0] != n_samples:
        raise ValueError(
            "X and Y must have the same number of samples. "
            f"Got X: {n_samples}, Y: {Y.shape[0]}"
        )
    if n_samples == 0 or n_features == 0 or Y.shape[1] == 0:
        raise ValueError("X and Y must have non-zero dimensions")

    original_l = l
    try:
        l = int(original_l)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("l must be an integer") from exc
    if isinstance(original_l, (bool, np.bool_)) or l != original_l:
        raise ValueError("l must be an integer")
    max_components = min(n_samples, n_features)
    if not 1 <= l <= max_components:
        raise ValueError(
            f"l must be between 1 and min(n_samples, n_features) = {max_components}"
        )

    original_max_iter = max_iter
    try:
        max_iter = int(original_max_iter)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("max_iter must be a positive integer") from exc
    if (
            isinstance(original_max_iter, (bool, np.bool_))
            or max_iter != original_max_iter
            or max_iter < 1
    ):
        raise ValueError("max_iter must be a positive integer")
    try:
        tol = float(tol)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("tol must be a positive finite number") from exc
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be a positive finite number")
    if convergence_error not in {"silent", "warn", "raise"}:
        raise ValueError("convergence_error must be 'silent', 'warn', or 'raise'")

    center = bool(center)
    n_targets = Y.shape[1]
    X_mean = _column_mean(X) if center else np.zeros(n_features)
    Y_mean = _column_mean(Y) if center else np.zeros(n_targets)

    T = np.zeros((n_samples, l))
    U = np.zeros((n_samples, l))
    P = np.zeros((n_features, l))
    Q = np.zeros((n_targets, l))
    W = np.zeros((n_features, l))
    C = np.zeros((n_targets, l))
    n_iter = np.zeros(l, dtype=int)
    converged = np.zeros(l, dtype=bool)

    epsilon = np.finfo(float).eps

    for component in range(l):
        u = _initial_y_score(Y, Y_mean, T, Q, component, center)
        previous_w = None

        for iteration in range(max_iter):
            w = _transpose_matvec(X, X_mean, u, T, P, component, center)
            w_norm = np.linalg.norm(w)
            if w_norm <= epsilon:
                raise ValueError(
                    f"X-Y covariance became zero at component {component}; "
                    "cannot extract another PLS component."
                )
            w /= w_norm

            t = _matvec(X, X_mean, w, T, P, component, center)
            t_norm_sq = t @ t
            if t_norm_sq <= epsilon:
                raise ValueError(
                    f"X score became zero at component {component}; "
                    "cannot extract another PLS component."
                )

            c = _transpose_matvec(Y, Y_mean, t, T, Q, component, center) / t_norm_sq
            c_norm_sq = c @ c
            if c_norm_sq <= epsilon:
                raise ValueError(
                    f"Y weight became zero at component {component}; "
                    "cannot extract another PLS component."
                )
            u_new = _matvec(Y, Y_mean, c, T, Q, component, center) / c_norm_sq

            n_iter[component] = iteration + 1
            if n_targets == 1 or (
                    previous_w is not None
                    and (w - previous_w) @ (w - previous_w) <= tol
            ):
                converged[component] = True
                u = u_new
                break

            previous_w = w.copy()
            u = u_new

        if not converged[component]:
            message = (
                f"PLS2 component {component} did not converge within "
                f"{max_iter} iterations"
            )
            if convergence_error == "raise":
                raise RuntimeError(message)
            if convergence_error == "warn":
                warnings.warn(message, RuntimeWarning, stacklevel=2)

        # These loadings define the low-rank residual used by later components.
        p = _transpose_matvec(X, X_mean, t, T, P, component, center) / t_norm_sq
        q = _transpose_matvec(Y, Y_mean, t, T, Q, component, center) / t_norm_sq

        T[:, component] = t
        U[:, component] = u
        P[:, component] = p
        Q[:, component] = q
        W[:, component] = w
        C[:, component] = c

    # Solve instead of explicitly inverting P.T @ W.
    try:
        latent_coef = np.linalg.solve(P.T @ W, Q.T)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "The extracted PLS components are linearly dependent; "
            "try using fewer components."
        ) from exc
    coef = W @ latent_coef
    intercept = Y_mean - X_mean @ coef if center else np.zeros(n_targets)

    return {
        "T": T,
        "U": U,
        "P": P,
        "Q": Q,
        "W": W,
        "C": C,
        "coef": coef,
        "B_pls": coef,
        "intercept": intercept,
        "X_mean": X_mean,
        "Y_mean": Y_mean,
        "n_iter": n_iter,
        "converged": converged,
    }


def predict_pls2(model, X_new):
    """Predict responses for dense or sparse new predictor matrices."""
    X_new = _as_float_matrix(X_new, "X_new")
    coef = model["coef"] if "coef" in model else model["B_pls"]
    if X_new.shape[1] != coef.shape[0]:
        raise ValueError(
            "X_new has the wrong number of features. "
            f"Expected {coef.shape[0]}, got {X_new.shape[1]}"
        )

    if "intercept" in model:
        intercept = model["intercept"]
    else:
        intercept = model["Y_mean"] - model["X_mean"] @ coef
    return np.asarray(X_new @ coef) + intercept


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 10))
    X[rng.random(X.shape) > 0.2] = 0.0
    Y = X @ rng.normal(size=(10, 3)) + 0.1 * rng.normal(size=(100, 3))

    for X_input, Y_input in (
            (X, Y),
            (sparse.csr_matrix(X), Y),
            (X, sparse.csr_matrix(Y)),
            (sparse.csr_matrix(X), sparse.csr_matrix(Y)),
    ):
        fitted = SPARSE_PLS2(Y_input, X_input, l=5)
        prediction = predict_pls2(fitted, X_input)
        print(type(X_input).__name__, type(Y_input).__name__, prediction.shape)
