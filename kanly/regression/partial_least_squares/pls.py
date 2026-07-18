from __future__ import absolute_import, print_function

import time
from typing import Dict, Optional

import numpy as np
from scipy.sparse import issparse

from kanly.formula.data_getter import SparseDataGetter
from kanly.formula.keys import ENDOG_KEY, EXOG_KEY, INSTRUMENTS_KEY, WEIGHTS_KEY
from kanly.regression.partial_least_squares.pls_regression_results import PlsRegressionResults
from kanly.utils.linalg_utils import DenseThreshold


# Helper function to apply centering to X @ v without densifying
def _centered_matvec(X, X_mean, v, center):
    """Compute (X - ones @ X_mean.T) @ v efficiently"""
    is_sparse = issparse(X)
    if not center:
        return X @ v
    # (X - ones @ X_mean.T) @ v = X @ v - ones @ (X_mean.T @ v)
    # = X @ v - ones * (X_mean.T @ v)
    # = X @ v - (X_mean.T @ v) * ones
    result = X @ v
    result -= (X_mean @ v)  # Broadcasting: subtract scalar from each element
    return result


# Helper function to apply centering to X.T @ u without densifying
def _centered_matvec_T(X, X_mean, u, center):
    """Compute (X - ones @ X_mean.T).T @ u efficiently"""
    is_sparse = issparse(X)
    if not center:
        if is_sparse:
            return X.T @ u
        else:
            return X.T @ u
    # (X - ones @ X_mean.T).T @ u = X.T @ u - X_mean @ ones.T @ u
    # = X.T @ u - X_mean * (ones.T @ u)
    # = X.T @ u - X_mean * sum(u)
    if is_sparse:
        result = X.T @ u
    else:
        result = X.T @ u
    result -= X_mean * np.sum(u)  # Broadcasting
    return result


def _PLS1_INTERNAL_SPARSE(y, X, l, center=True):
    """
    Sparse-aware Partial Least Squares Regression (PLS1) with single response.

    Implements the NIPALS algorithm for PLS1 while preserving sparse structure of X.

    Model equations:
        X = T * P.T + E_X
        y = T * q + E_y

    where:
        T: scores (latent variables)
        P: X-loadings
        q: y-loadings
        E_X, E_y: residuals

    Parameters
    ----------
    y : array-like, shape (n_samples,) or (n_samples, 1)
        Response vector
    X : array-like or sparse matrix, shape (n_samples, n_features)
        Predictor matrix. Can be dense numpy array or scipy sparse matrix.
    l : int
        Number of latent components to extract
    center : bool, default=True
        Whether to center X and y before fitting. If True, an intercept term
        is computed and returned. Centering is done implicitly to preserve sparsity.

    Returns
    -------
    T : ndarray, shape (n_samples, l)
        Score matrix (latent variables)
    P : ndarray, shape (n_features, l)
        X-loadings matrix
    q : ndarray, shape (l,)
        y-loadings vector
    W : ndarray, shape (n_features, l)
        Weight matrix (used for projection)
    coef : ndarray, shape (n_features,)
        Regression coefficients for prediction (excluding intercept)
    intercept : float
        Intercept term. If center=False, this is 0.0

        Prediction formula:
            y_pred = X @ coef + intercept

    Notes
    -----
    The algorithm preserves sparsity of X by:
    1. Never explicitly deflating X (which would densify it)
    2. Computing deflation effects on-the-fly using projection formulas
    3. Only materializing small dense matrices (W, P, projections)
    4. Centering X implicitly without creating a dense centered matrix

    When center=True:
    - X is centered implicitly: X_centered = X - ones @ X_mean.T
    - All operations account for centering without densifying X
    - y is centered explicitly (it's already dense)
    - The intercept adjusts for centering: intercept = mean(y) - mean(X) @ coef

    The regression coefficients are computed as:
        coef = W @ inv(P.T @ W) @ q

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> X = csr_matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    >>> y = np.array([1, 2, 3])
    >>> T, P, q, W, coef, intercept = PLS1_SPARSE(y, X, l=2)
    >>> y_pred = X @ coef + intercept
    """

    # Input validation and conversion
    y = np.asarray(y).ravel()  # Ensure y is 1D
    n_samples = X.shape[0]
    n_features = X.shape[1]

    if y.shape[0] != n_samples:
        raise ValueError(f"X and y must have same number of samples. Got X: {n_samples}, y: {y.shape[0]}")

    if l < 1 or l > min(n_samples, n_features):
        raise ValueError(
            f"Number of components l must be between 1 and min(n_samples, n_features) = {min(n_samples, n_features)}")

    is_sparse = issparse(X)

    # Compute means for centering
    if center:
        if is_sparse:
            # For sparse matrices, compute mean efficiently
            X_mean = np.asarray(X.mean(axis=0)).ravel()  # shape (n_features,)
        else:
            X_mean = X.mean(axis=0)  # shape (n_features,)

        y_mean = y.mean()  # scalar

        # Center y (it's dense, so this is fine)
        y_centered = y - y_mean
    else:
        X_mean = np.zeros(n_features)
        y_mean = 0.0
        y_centered = y.copy()

    # Initialize output matrices
    T = np.zeros((n_samples, l))  # Scores
    P = np.zeros((n_features, l))  # X-loadings
    q = np.zeros(l)  # y-loadings
    W = np.zeros((n_features, l))  # Weights

    # Working copy for y deflation
    y_current = y_centered.copy()

    # NIPALS algorithm for PLS1
    for component in range(l):

        # Step 1: Compute weight vector w
        # w = X_current.T @ y_current (where X_current is centered and deflated)
        # For first component, X_current = X_centered
        # For later components, we compute deflation effect implicitly

        if component == 0:
            # First component: only centering, no deflation
            w = _centered_matvec_T(X, X_mean, y_current, center)
        else:
            # Subsequent components: apply implicit deflation to centered X
            # X_current = X_centered - X_centered @ W[:,:component] @ inv(P[:,:component].T @ W[:,:component]) @ P[:,:component].T

            w = _centered_matvec_T(X, X_mean, y_current, center)

            # Compute the Gram matrix inverse: (P.T @ W)^{-1}
            PtW = P[:, :component].T @ W[:, :component]  # shape (component, component)
            PtW_inv = np.linalg.inv(PtW)

            # Subtract the projection: W @ inv(P.T @ W) @ P.T @ (X_centered.T @ y_current)
            Xty = _centered_matvec_T(X, X_mean, y_current, center)
            projection = W[:, :component] @ PtW_inv @ (P[:, :component].T @ Xty)
            w = w - projection

        # Normalize weight vector
        w_norm = np.linalg.norm(w)
        if w_norm < 1e-10:
            raise ValueError(f"Weight vector became zero at component {component}. Cannot continue.")
        w = w / w_norm

        # Step 2: Compute score vector t
        # t = X_current @ w (where X_current is centered and deflated)
        if component == 0:
            # First component: only centering
            t = _centered_matvec(X, X_mean, w, center)

        else:
            # Apply implicit deflation to centered X
            t = _centered_matvec(X, X_mean, w, center)

            # Reuse PtW_inv from above
            # Subtract projection: X_centered @ W @ inv(P.T @ W) @ P.T @ w
            Xw = _centered_matvec(X, X_mean, W[:, :component], center)
            if W[:, :component].ndim == 1:
                # Single column case
                projection = Xw @ PtW_inv @ (P[:, :component].T @ w)
            else:
                # Multiple columns
                projection = Xw @ PtW_inv @ (P[:, :component].T @ w)
            t = t - projection

        # Step 3: Compute X-loading vector p
        # p = X_current.T @ t / (t.T @ t)
        t_norm_sq = t.T @ t
        p = _centered_matvec_T(X, X_mean, t, center) / t_norm_sq

        # Step 4: Compute y-loading scalar q_i
        # q_i = t.T @ y_current / (t.T @ t)
        q_i = t.T @ y_current / t_norm_sq

        # Step 5: Deflate y
        # y_current = y_current - t * q_i
        y_current -= t * q_i

        # Store results for this component
        T[:, component] = t
        P[:, component] = p
        q[component] = q_i
        W[:, component] = w

    # Compute regression coefficients for centered data
    # coef = W @ inv(P.T @ W) @ q
    PtW = P.T @ W  # shape (l, l)
    coef = W @ np.linalg.solve(PtW, q)  # More stable than explicit inverse

    # Compute intercept
    # For centered data: y_centered = X_centered @ coef
    # Original scale: (y - y_mean) = (X - X_mean) @ coef
    # Therefore: y = X @ coef + (y_mean - X_mean @ coef)
    if center:
        intercept = y_mean - X_mean @ coef
    else:
        intercept = 0.0

    return T, P, q, W, coef, intercept


def _PLS1_INTERNAL(y, X, l, center=True):
    """
    Partial Least Squares Regression (PLS1) with single response - Dense implementation.

    Implements the NIPALS algorithm for PLS1 optimized for dense matrices.

    Model equations:
        X = T * P.T + E_X
        y = T * q + E_y

    where:
        T: scores (latent variables)
        P: X-loadings
        q: y-loadings
        E_X, E_y: residuals

    Parameters
    ----------
    y : array-like, shape (n_samples,) or (n_samples, 1)
        Response vector
    X : array-like, shape (n_samples, n_features)
        Predictor matrix (dense)
    l : int
        Number of latent components to extract
    center : bool, default=True
        Whether to center X and y before fitting. If True, an intercept term
        is computed and returned.

    Returns
    -------
    T : ndarray, shape (n_samples, l)
        Score matrix (latent variables)
    P : ndarray, shape (n_features, l)
        X-loadings matrix
    q : ndarray, shape (l,)
        y-loadings vector
    W : ndarray, shape (n_features, l)
        Weight matrix (used for projection)
    coef : ndarray, shape (n_features,)
        Regression coefficients for prediction (excluding intercept)
    intercept : float
        Intercept term. If center=False, this is 0.0

        Prediction formula:
            y_pred = X @ coef + intercept

    Notes
    -----
    This is the standard NIPALS algorithm optimized for dense matrices.
    X and y are explicitly deflated at each iteration for efficiency.

    When center=True:
    - X and y are centered to have zero mean
    - The model is fit on centered data
    - Regression coefficients coef are computed for centered data
    - The intercept adjusts for the centering: intercept = mean(y) - mean(X) @ coef

    The regression coefficients are computed as:
        coef = W @ inv(P.T @ W) @ q

    Examples
    --------
    >>> X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> y = np.array([1, 2, 3])
    >>> T, P, q, W, coef, intercept = PLS1(y, X, l=2)
    >>> y_pred = X @ coef + intercept

    >>> # Without centering
    >>> T, P, q, W, coef, intercept = PLS1(y, X, l=2, center=False)
    >>> y_pred = X @ coef + intercept  # intercept will be 0.0
    """
    # Input validation and conversion

    y = np.asarray(y).ravel()  # Ensure y is 1D
    X = np.asarray(X)

    n_samples, n_features = X.shape

    if y.shape[0] != n_samples:
        raise ValueError(f"X and y must have same number of samples. Got X: {n_samples}, y: {y.shape[0]}")

    if l < 1 or l > min(n_samples, n_features):
        raise ValueError(
            f"Number of components l must be between 1 and min(n_samples, n_features) = {min(n_samples, n_features)}")

    # Center data if requested
    if center:
        X_mean = X.mean(axis=0)  # shape (n_features,)
        y_mean = y.mean()  # scalar
        X_centered = X - X_mean
        y_centered = y - y_mean
    else:
        X_mean = np.zeros(n_features)
        y_mean = 0.0
        X_centered = X
        y_centered = y

    # Initialize output matrices
    T = np.zeros((n_samples, l))  # Scores
    P = np.zeros((n_features, l))  # X-loadings
    q = np.zeros(l)  # y-loadings
    W = np.zeros((n_features, l))  # Weights

    # Working copies for deflation (explicit deflation for dense case)
    X_current = X_centered.copy()
    y_current = y_centered.copy()

    # NIPALS algorithm for PLS1
    for component in range(l):

        # Step 1: Compute weight vector w = X_current.T @ y_current
        w = X_current.T @ y_current

        # Normalize weight vector
        w_norm = np.linalg.norm(w)
        if w_norm < 1e-10:
            raise ValueError(f"Weight vector became zero at component {component}. Cannot continue.")
        w = w / w_norm

        # Step 2: Compute score vector t = X_current @ w
        t = X_current @ w

        # Step 3: Compute X-loading vector p = X_current.T @ t / (t.T @ t)
        t_norm_sq = t.T @ t
        p = X_current.T @ t / t_norm_sq

        # Step 4: Compute y-loading scalar q_i = t.T @ y_current / (t.T @ t)
        q_i = t.T @ y_current / t_norm_sq

        # Step 5: Deflate X
        # X_current = X_current - t @ p.T
        # Using outer product for efficiency
        X_current -= np.outer(t, p)

        # Step 6: Deflate y
        # y_current = y_current - t * q_i
        y_current -= t * q_i

        # Store results for this component
        T[:, component] = t
        P[:, component] = p
        q[component] = q_i
        W[:, component] = w

    # Compute regression coefficients for centered data
    # coef = W @ inv(P.T @ W) @ q
    PtW = P.T @ W  # shape (l, l)
    coef = W @ np.linalg.solve(PtW, q)  # More stable than explicit inverse

    # Compute intercept
    # For centered data: y_centered = X_centered @ coef
    # Original scale: (y - y_mean) = (X - X_mean) @ coef
    # Therefore: y = X @ coef + (y_mean - X_mean @ coef)
    if center:
        intercept = y_mean - X_mean @ coef
    else:
        intercept = 0.0

    return T, P, q, W, coef, intercept


def _compute_r_squared(y, resid, center, weights=None):
    resid_sq = resid ** 2

    if weights is None:
        weights = 1.0
        if center:
            y_w_mean = y.mean()
        else:
            y_w_mean = 0.0
        scale = np.mean(resid_sq)
    else:
        if center:
            y_w_mean = np.average(y, weights=weights)
        else:
            y_w_mean = 0.0
        scale = np.sum(weights * resid_sq) / np.sum(weights)

    wssr = sum(weights * resid_sq)

    wsst = sum(weights * (y - y_w_mean) ** 2)
    return wssr, wsst, 1.0 - wssr / wsst, scale


def _compute_pls_cov(y, X, W, sigma2_hat, beta, center):
    """
    Args:
        y: endog
        X: exog
        W: weights (output of PLS1 routine)
        sigma2_hat: the scale, mean(resid**2)
        beta: the coefficients on X
        center: boolean, whether an intercept was included and data was centered

    Returns:
        variance-covariance matrix
    """
    n_ = X.shape[0]
    p_ = X.shape[1]
    l = len(beta)
    mu_X = X.mean(axis=0) if center else np.zeros(p_)

    XpX = X.T @ X
    if issparse(XpX):
        XpX = XpX.toarray()
    Sigma = 1.0 / n_ * XpX - np.outer(mu_X, mu_X)

    H_K = W @ np.linalg.pinv(W.T @ Sigma @ W) @ W.T

    V_bb = sigma2_hat * H_K @ Sigma @ H_K
    V_ab = (-mu_X @ V_bb).ravel()

    if center:
        y_cent = y - (y.mean() if center else 0.0)
        beta_ols = 1 / n_ * np.linalg.pinv(Sigma) @ (X.T @ y_cent)
        V_aa = sigma2_hat + (beta_ols - beta).T @ Sigma @ (beta_ols - beta) + mu_X @ V_bb @ mu_X.T
    else:
        V_aa = np.nan
        V_ab *= np.nan

    cov = np.zeros((l + 1, l + 1))
    cov[0, 0] = V_aa
    cov[1:, 1:] = V_bb
    cov[0, 1:] = V_ab
    cov[1:, 0] = V_ab
    cov /= n_

    return cov


def PLS1(y, X, l, center=True, specification_name=None, test_level=.05, compute_cov=True,
         endog_name=None, exog_names=None, model_elapsed=0.0):
    """
    Computes partial least squares

    Args:
        y: endog matrix, n x 1
        X: exog matrix, n x k
        l: number of components
        center:
        specification_name:
        test_level:
        compute_cov:
        endog_name:
        exog_names:

    Returns:

    Examples
    --------
    Single-response PLS regression with three latent components, useful when
    ``X`` is wide and predictors are correlated:

    >>> import numpy as np
    >>> from kanly.api import PLS1
    >>> rng = np.random.default_rng(0)
    >>> n, p = 200, 10
    >>> X = rng.normal(size=(n, p))
    >>> beta = np.zeros(p); beta[:5] = [1.5, -1.0, 0.5, 0.3, -0.2]
    >>> y = X @ beta + 0.5 * rng.normal(size=n)
    >>> fit = PLS1(y, X, l=3,                              # doctest: +SKIP
    ...            exog_names=[f'x{j}' for j in range(p)])
    >>> fit.coef.round(2)                                   # doctest: +SKIP
    array([ 1.42, -0.97,  0.47,  0.31, -0.18, ...])

    Use :func:`pls1` for the formula-based entry point.

    See Also
    --------
    :func:`pls1` : formula API.
    :func:`PLS2` : multi-response PLS via NIPALS.
    """
    _time = time.time()

    if issparse(y):
        y = y.toarray().flatten()

    if issparse(X):
        func = _PLS1_INTERNAL_SPARSE
    else:
        func = _PLS1_INTERNAL

    l = int(l)
    assert 1 <= l <= X.shape[1]

    T, P, q, W, coef, intercept = func(y, X, l, center=center)

    fittedvalues = X @ coef + intercept
    resid = y.ravel() - fittedvalues
    wssr, wsst, rsquared, scale = _compute_r_squared(y, resid, center, weights=None)
    fit_time = time.time() - _time

    _time = time.time()
    if compute_cov:
        cov_params = _compute_pls_cov(y, X, W, scale, coef, center=center)
    else:
        cov_params = None
    cov_time = time.time() - _time

    return PlsRegressionResults(T, P, q, W, coef, intercept, X, y, None,
                                l, center, fittedvalues, resid, wssr, wsst, rsquared, scale,
                                cov_params,
                                exog_names=exog_names,
                                endog_name=endog_name,
                                test_level=test_level,
                                specification_name=specification_name,
                                model_elapsed=model_elapsed,
                                fit_elapsed=fit_time,
                                cov_elapsed=cov_time)


def _build_pls_model_from_formula(
        formula, data, debug=False, index=None, check_constant_cols=False,
        fail_on_missing=False, cache_intermediate=True, dense_threshold_mb=1024):
    _time = time.time()
    center = "".join(formula.split())[-2:] != "-1"
    if center:
        formula += " -1"

    build_data_result = SparseDataGetter.get_data(
        data, formula, check_constant_cols=check_constant_cols, absorb=None, debug=debug, _time=_time,
        fail_on_missing=fail_on_missing, cache_intermediate=cache_intermediate, sum_to_n=False,
        index=index, test_formula_on_dummy=False,
        drop_1_for_FE=False,
    )

    endog_obj = build_data_result[ENDOG_KEY]
    exog_obj = build_data_result[EXOG_KEY]
    if build_data_result[INSTRUMENTS_KEY] is not None:
        raise Exception("Instrumental variables not possible for PLS!")
    if build_data_result[WEIGHTS_KEY] is not None:
        raise NotImplementedError("Weighted regression not implemented yet!")

    y = endog_obj.values.toarray().flatten()
    endog_name = endog_obj.column_names[0]

    X = exog_obj.values
    if DenseThreshold.is_convertible_to_dense(X, dense_threshold_mb=dense_threshold_mb):
        X = X.toarray()
    exog_names = exog_obj.column_names
    model_elapsed = time.time() - _time

    return y, X, center, endog_name, exog_names, model_elapsed


def pls1(formula, data, l, debug=False, index=None, check_constant_cols=False, fail_on_missing=False,
         cache_intermediate=True, compute_cov=True, test_level=.05, dense_threshold_mb=1024):
    """Fit single-response partial least squares from a patsy-style formula.

    Wraps :func:`PLS1` after parsing a Patsy formula to build sparse design
    matrices (no intercept term — PLS1 fits its own centred intercept; if a
    formula ends in ``" -1"`` the centring step is skipped).

    Args:
        formula (str): Patsy-style formula like ``'y ~ x1 + x2 + x3'``.
        data (DataFrame or dict): Source data.
        l (int): Number of latent PLS components to extract.
        debug (bool): Print parsing diagnostics.
        index (array-like, optional): Optional row index/subset.
        check_constant_cols (bool): Whether to detect constant design columns.
        fail_on_missing (bool): Whether to raise on NaN rows.
        cache_intermediate (bool or dict): Patsy term cache.
        compute_cov (bool): Whether to compute approximate covariance.
        test_level (float): Significance level used in printed summary.
        dense_threshold_mb (float): Sparse/dense conversion threshold.

    Returns:
        ``PlsRegressionResults``.

    Examples
    --------
    Three-component PLS regression from a formula:

    >>> import numpy as np, pandas as pd
    >>> from kanly.api import pls1
    >>> rng = np.random.default_rng(0)
    >>> n, p = 200, 10
    >>> df = pd.DataFrame(rng.normal(size=(n, p)),
    ...                   columns=[f'x{j}' for j in range(p)])
    >>> beta = np.zeros(p); beta[:5] = [1.5, -1.0, 0.5, 0.3, -0.2]
    >>> df['y'] = df.values @ beta + 0.5 * rng.normal(size=n)
    >>> fit = pls1('y ~ ' + ' + '.join(df.columns[:-1]),    # doctest: +SKIP
    ...            df, l=3)

    See Also
    --------
    :func:`PLS1` : array entry point.
    :func:`PLS2` : multi-response PLS via NIPALS.
    """

    y, X, center, endog_name, exog_names, model_elapsed = _build_pls_model_from_formula(
        formula, data, debug=debug, index=index, check_constant_cols=check_constant_cols,
        fail_on_missing=fail_on_missing, cache_intermediate=cache_intermediate, dense_threshold_mb=dense_threshold_mb
    )

    result = PLS1(y, X, l, center=center, compute_cov=compute_cov, test_level=test_level,
                  exog_names=exog_names, endog_name=endog_name,
                  model_elapsed=model_elapsed)

    return result


def PLS2(
        Y: np.ndarray,
        X: np.ndarray,
        l: int,
        center: bool = True,
        max_iter: int = 100,
        tol: float = 1e-6,
) -> Dict[str, Optional[np.ndarray]]:
    """
    Partial Least Squares Regression (PLS2) with multiple response variables.

    Implements the NIPALS algorithm for PLS regression, which finds latent components
    that maximize covariance between predictors X and responses Y.

    Parameters
    ----------
    Y : np.ndarray
        Response matrix of shape (n_samples, n_responses). Must be 2-dimensional.
    X : np.ndarray
        Predictor matrix of shape (n_samples, n_features). Must be 2-dimensional.
    l : int
        Number of latent components to extract.
    center : bool, default=True
        If True, center X and Y and compute intercept. If False, assume data
        is already centered and set intercept to None.

    Returns
    -------
    dict
        Dictionary containing:
        - 'T' : np.ndarray
            Score matrix (latent components) of shape (n_samples, l).
            These are the projections of X onto the latent space.
        - 'P' : np.ndarray
            X-loadings matrix of shape (n_features, l).
            Defines the relationship: X = T @ P.T + error
        - 'Q' : np.ndarray
            Y-loadings matrix of shape (n_responses, l).
            Defines the relationship: Y = T @ Q.T + error
        - 'W' : np.ndarray
            X-weights matrix of shape (n_features, l).
            Used to compute scores: T = X @ W
        - 'coef' : np.ndarray
            Regression coefficients of shape (n_features, n_responses).
            Defines the relationship: Y = X @ coef + error
        - 'intercept' : np.ndarray or None
            Intercept vector of shape (n_responses,) if center=True, else None.

    Raises
    ------
    ValueError
        If Y or X are not 2-dimensional arrays.

    Notes
    -----
    The algorithm uses the NIPALS (Nonlinear Iterative Partial Least Squares)
    method to iteratively extract latent components that maximize the covariance
    between X and Y.

    References
    ----------
    Wold, S., Sjöström, M., & Eriksson, L. (2001). PLS-regression: a basic tool
    of chemometrics. Chemometrics and intelligent laboratory systems, 58(2), 109-130.

    Examples
    --------
    Two-response PLS extracting two latent components:

    >>> import numpy as np
    >>> from kanly.api import PLS2
    >>> rng = np.random.default_rng(0)
    >>> n, p, q = 200, 10, 2
    >>> X = rng.normal(size=(n, p))
    >>> B = np.zeros((p, q))
    >>> B[0, 0] = 2.0; B[1, 1] = -1.5; B[2, 0] = 0.5
    >>> Y = X @ B + 0.3 * rng.normal(size=(n, q))
    >>> out = PLS2(Y, X, l=2)                              # doctest: +SKIP
    >>> out['coef'].round(2)                                # doctest: +SKIP
    array([[ 1.93,  0.01],
           [ 0.02, -1.46],
           [ 0.46,  0.04],
           ...])

    See Also
    --------
    :func:`PLS1` : single-response PLS.
    """
    # Validate input dimensions
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2-dimensional, got shape {Y.shape}")
    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")

    n_samples, n_features = X.shape
    n_responses = Y.shape[1]

    # Convert to float arrays to avoid integer division issues
    X = X.astype(float)
    Y = Y.astype(float)

    # Store original means for computing intercept
    if center:
        X_mean = np.mean(X, axis=0)
        Y_mean = np.mean(Y, axis=0)
        X_centered = X - X_mean
        Y_centered = Y - Y_mean
    else:
        X_mean = None
        Y_mean = None
        X_centered = X.copy()
        Y_centered = Y.copy()

    # Initialize matrices to store results
    T = np.zeros((n_samples, l))  # Scores (latent components)
    P = np.zeros((n_features, l))  # X-loadings
    Q = np.zeros((n_responses, l))  # Y-loadings
    W = np.zeros((n_features, l))  # X-weights

    # Working copies that will be deflated
    X_work = X_centered.copy()
    Y_work = Y_centered.copy()

    # Extract l components using NIPALS algorithm
    for component in range(l):
        # Initialize weights as the first column of X'Y (dominant direction)
        # This captures the direction of maximum covariance
        w = X_work.T @ Y_work[:, 0]
        w = w / np.linalg.norm(w)  # Normalize to unit length

        # Iteratively refine weights until convergence
        for iteration in range(max_iter):  # Max iterations to prevent infinite loops
            w_old = w.copy()

            # Compute scores: project X onto weight vector
            t = X_work @ w

            # Normalize scores
            t_norm = np.linalg.norm(t)
            t = t / t_norm

            # Compute Y-loadings: regression of Y on scores
            q = Y_work.T @ t

            # Compute X-weights: use Y-loadings to find new direction
            # This maximizes covariance between X and Y
            w = X_work.T @ (Y_work @ q)
            w = w / np.linalg.norm(w)

            # Check convergence: if weights haven't changed much, stop
            if np.allclose(w, w_old, atol=tol):
                break

        # Final scores with proper normalization
        t = X_work @ w
        t_norm = np.linalg.norm(t)
        t = t / t_norm

        # Adjust weight vector for the normalization
        w = w * t_norm

        # Compute X-loadings: regression of X on scores
        p = X_work.T @ t

        # Compute Y-loadings: regression of Y on scores
        q = Y_work.T @ t

        # def fix(xx):
        #     if issparse(xx):
        #         xx = t.toarray()
        #     if np.ndim(t) > 1:
        #         xx = t.flatten()
        #     return xx
        #
        # t = fix(t)
        # p = fix(p)
        # q = fix(q)
        # w = fix(w)

        # Store results for this component
        T[:, component] = t
        P[:, component] = p
        Q[:, component] = q
        W[:, component] = w

        # Deflate X and Y: remove the variance explained by this component
        X_work = X_work - np.outer(t, p)
        Y_work = Y_work - np.outer(t, q)

    # Compute regression coefficients: coef = W @ (P.T @ W)^(-1) @ Q.T
    # This transforms from latent space back to original predictor space
    W_star = W @ np.linalg.inv(P.T @ W)
    coef = W_star @ Q.T

    # Compute intercept if centering was applied
    if center:
        intercept = Y_mean - X_mean @ coef
    else:
        intercept = np.zeros(Y.shape[1])

    fittedvalues = X @ coef + intercept
    resid = Y - fittedvalues

    def predict(exog=None):
        if exog is None:
            return fittedvalues.copy()
        else:
            return exog @ coef + intercept

    return {
        'T': T,
        'P': P,
        'Q': Q,
        'W': W,
        'coef': coef,
        'intercept': intercept,
        'fittedvalues': fittedvalues,
        'resid': resid,
        'predict': predict,
    }
