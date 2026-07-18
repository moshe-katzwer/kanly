"""
Sparse whitening matrices and iterative core for GLSAR (feasible GLS with AR(p) errors).

:func:`make_ar_full_information_W` builds the whitening operator ``W``.
:func:`fit_glsar_internal` runs the AR → whiten → GLS iteration on raw ``(endog, exog)``.

:meth:`~kanly.regression.linear_models.model.SparseLinearModel.fit_glsar` calls
``fit_glsar_internal`` for estimation, then wraps the result in
:class:`~kanly.regression.linear_models.regression_results.SparseLinearRegressionResults`
(inference, summaries, ``glsar_info``).
"""

from __future__ import absolute_import, print_function

import numpy as np
from scipy.sparse import csr_matrix, bmat, isspmatrix, diags

from kanly.regression.linear_models.lm_internal import LinearModelRegressionResultsRaw, lm_internal
from kanly.time_series.autoregression.estimate_ar import estimate_ar
from kanly.time_series.autoregression.stationary_ar_covariance import ar_initial_gamma


def make_ar_D(rho, n, format="csr"):
    """
    Build sparse AR(p) innovation (filter) matrix ``D``.

    For errors following AR(p), ``D @ u`` maps the series to uncorrelated
    innovations from index ``p`` onward: each row is one step of the
    innovation recursion.

    Model
    -----
    .. math::

        u_t = \\rho_1 u_{t-1} + \\cdots + \\rho_p u_{t-p} + \\varepsilon_t

    Row ``t - p`` (0-based) of ``D @ u`` equals
    ``u_t - rho_1 u_{t-1} - ... - rho_p u_{t-p}`` for ``t = p, ..., n-1``.

    Parameters
    ----------
    rho : array-like, shape (p,)
        AR coefficients ``[rho_1, ..., rho_p]``.
    n : int
        Length of the time series.
    format : str, default="csr"
        Sparse matrix format.

    Returns
    -------
    D : scipy.sparse matrix, shape (n - p, n)
        Innovation operator used as the lower block of the whitening matrix.
    """
    rho = np.asarray(rho, dtype=float)
    p = len(rho)

    if p < 1:
        raise ValueError("rho must contain at least one AR coefficient.")
    if n <= p:
        raise ValueError("Need n > p.")

    # Diagonal coefficients from left to right:
    # columns row+i through row+i+p
    vals = np.r_[-rho[::-1], 1.0]

    # For shape (n-p, n), offset k means column = row + k.
    offsets = np.arange(p + 1)

    D = diags(
        diagonals=[np.full(n - p, v) for v in vals],
        offsets=offsets,
        shape=(n - p, n),
        format=format,
    )

    return D


def make_ar_full_information_W(rho, n, scale=1.0, format="csr", full_information=True):
    """
    Build sparse whitening matrix ``W`` for AR(p) regression errors.

    GLSAR applies ``W`` to the outcome and regressors each iteration:
    ``Wy = W @ y``, ``WX = W @ X``, then OLS/GLS on the whitened system.
    Under the assumed AR structure, ``W @ u`` has approximately scalar
    covariance (innovations with variance ``scale``).

    The matrix is stacked as ``W = vstack(top, bottom)`` with shape ``(n, n)``:

    * **Bottom block** (always): ``D / sqrt(scale)`` where ``D = make_ar_D(...)``
      has shape ``(n - p, n)``. These rows implement the innovation filter on
      observations ``p, ..., n-1``.

    * **Top block** (depends on ``full_information``):

      - ``full_information=True`` (**Prais-Winsten**): ``[C, 0]`` with
        ``C = L^{-1}`` and ``L`` the Cholesky factor of the stationary
        ``Gamma_p`` from :func:`ar_initial_gamma`. The first ``p`` rows of
        ``W @ u`` are whitened using the correct initial covariance.

      - ``full_information=False`` (**Cochrane-Orcutt**): ``p × n`` zeros.
        The first ``p`` observations are not transformed; only the ``n - p``
        innovation rows in the bottom block carry information (analogous to
        statsmodels ``GLSAR.whiten``, which drops the first ``p`` rows).

    Parameters
    ----------
    rho : array-like, shape (p,)
        AR coefficients estimated from current residuals.
    n : int
        Number of observations (time-series length).
    scale : float, default=1.0
        Innovation variance scale (typically from :func:`~kanly.time_series.autoregression.estimate_ar`).
    format : str, default="csr"
        Sparse format for ``W``.
    full_information : bool, default=True
        If True, use Prais-Winsten whitening (stationary initial conditions).
        If False, use Cochrane-Orcutt whitening (no top block; effective sample
        size reduced by ``p`` in :meth:`~kanly.regression.linear_models.model.SparseLinearModel.fit_glsar` summaries).

    Returns
    -------
    W : scipy.sparse matrix, shape (n, n)
        Whitening operator; ``W @ u`` yields approximately white residuals.

    See Also
    --------
    make_ar_D : innovation filter used in the bottom block.
    ar_initial_gamma : stationary covariance for the Prais-Winsten top block.
    fit_glsar_internal : iterative AR / whiten / GLS estimation core.
    SparseLinearModel.fit_glsar : wraps internal + inference into results object.
    """
    rho = np.asarray(rho, dtype=float)
    p = len(rho)

    if p < 1:
        raise ValueError("rho must contain at least one AR coefficient.")
    if n <= p:
        raise ValueError("Need n > p.")

    # Innovation block: shape (n-p, n)
    D = make_ar_D(rho, n, format=format)

    if full_information:
        # Prais-Winsten: whiten first p obs via stationary initial covariance
        Gamma = ar_initial_gamma(rho, m=p, sigma2=scale)

        # Gamma = L L.T
        L = np.linalg.cholesky(Gamma)

        # C = L^{-1}; this satisfies C Gamma C.T = I
        C = np.linalg.solve(L, np.eye(p))

        # Sparse top block: [C, 0]
        C_sparse = csr_matrix(C)
        Z_top_right = csr_matrix((p, n - p))

        top = bmat([[C_sparse, Z_top_right]], format=format)

    else:
        # Cochrane-Orcutt: no transform on first p observations
        top = csr_matrix((p, n))

    # Lower block: scale-normalized innovation filter
    bottom = D / np.sqrt(scale)

    # Stack top and bottom
    W = bmat([[top], [bottom]], format=format)

    return W


def fit_glsar_internal(endog, exog, nlags, maxiter=10, tol=1e-6, ar_method='yw', full_information=True,
                       compute_eigenvalues=True, debug=False):
    """
    Iterative GLSAR estimation on array ``(endog, exog)``.

    This is the **computational core** pushed out of
    :meth:`~kanly.regression.linear_models.model.SparseLinearModel.fit_glsar`:
    it performs only the repeated AR estimation, whitening, and GLS refit loop.
    The model wrapper adds covariance, fit summaries, and :class:`~kanly.regression.linear_models.regression_results.GLSARInfo`.

    Algorithm
    ---------
    1. **Starting fit** — OLS via :func:`~kanly.regression.linear_models.lm_internal.lm_internal`
       on unwhitened ``(endog, exog)``; residuals ``e`` seed the AR loop.
    2. **For each iteration** (up to ``maxiter``):

       a. **Estimate AR(p)** on current ``e`` with
          :func:`~kanly.time_series.autoregression.estimate_ar`
          (``ar_method``, order ``nlags``).
       b. **Whitening** — ``W = make_ar_full_information_W(...)``; form
          ``Wy = W @ endog``, ``WX = W @ exog`` (Prais-Winsten if
          ``full_information=True``, else Cochrane-Orcutt).
       c. **GLS step** — OLS on ``(Wy, WX)`` → updated ``beta``.
       d. **Convergence** — from the second iteration onward, stop if
          ``max |ar_params - ar_params_previous| < tol``.
       e. **Recompute residuals** — ``e = endog - exog @ beta`` (on the original
          scale, not whitened) for the next AR fit.

    3. **Return** final ``beta``, whitened design ``(Wy, WX)``, last AR
       parameters, and the :class:`~kanly.regression.linear_models.lm_internal.LinearModelRegressionResultsRaw`
       object from the final whitened OLS (for ``normalized_cov_params``, etc.).

    Parameters
    ----------
    endog : array-like or sparse matrix, shape (n,) or (n, 1)
        Outcome vector.
    exog : array-like or sparse matrix, shape (n, k)
        Design matrix.
    nlags : int
        AR order ``p`` (must be >= 1; not validated here).
    maxiter : int, default=10
        Maximum AR/whiten/GLS iterations after the initial OLS.
    tol : float, default=1e-6
        Convergence tolerance on changes in AR coefficients.
    ar_method : str, default='yw'
        Passed to :func:`~kanly.time_series.autoregression.estimate_ar`.
    full_information : bool, default=True
        Prais-Winsten (``True``) vs Cochrane-Orcutt (``False``) whitening; see
        :func:`make_ar_full_information_W`.
    compute_eigenvalues : bool, default=True
        Forwarded to :func:`~kanly.regression.linear_models.lm_internal.lm_internal`.
    debug : bool, default=False
        If True, print AR coefficient changes each iteration.

    Returns
    -------
    beta : ndarray, shape (k,)
        Final coefficient vector from the last whitened OLS.
    ncp : matrix
        ``normalized_cov_params`` from the final whitened OLS (for covariance).
    result : LinearModelRegressionResultsRaw
        Full raw result object from the final ``lm_internal`` call on ``(Wy, WX)``.
    Wy, WX : array or sparse matrix
        Whitened outcome and design from the **last** iteration (used for R² / SSR).
    ar_params : ndarray, shape (p,)
        AR coefficients from the last :func:`~kanly.time_series.autoregression.estimate_ar` call.
    scale : float
        Innovation variance scale from the last AR fit.
    num_iter : int
        Iteration count bookkeeping (``itr + 2``, including the initial OLS pass).
    ar_error : float
        Max absolute change in AR coefficients at termination; only defined after
        at least two AR fits (otherwise may be unset if the loop exits on the
        first AR iteration).

    See Also
    --------
    make_ar_full_information_W : whitening matrix construction.
    SparseLinearModel.fit_glsar : wraps this function with inference and results.
    """
    is_sparse = isspmatrix(exog)
    n = exog.shape[0]

    # --- Initial OLS on unwhitened data (starting point for AR on residuals) ---
    fit0 = lm_internal(endog, exog)
    e = fit0.resid_raw
    ar_last = None

    for itr in range(maxiter):

        # Step 1: fit AR(p) to current residuals (original scale, not whitened)
        ar_estimate = estimate_ar(e, nlags, method=ar_method)
        ar_params, scale = ar_estimate['params'], ar_estimate['scale']

        # Step 2: build W and transform (y, X) -> (Wy, WX) for GLS under AR errors
        # full_information=True  -> Prais-Winsten (uses first p obs via Gamma_p)
        # full_information=False -> Cochrane-Orcutt (innovation rows only)
        W = make_ar_full_information_W(ar_params, n, scale=1.0, full_information=full_information)
        if not is_sparse:
            W = W.toarray()
        WX = W.dot(exog)
        Wy = W.dot(endog)

        # Step 3: OLS/GLS on whitened system (equivalent to GLS with Sigma^{-1/2} X, y)
        result: LinearModelRegressionResultsRaw = lm_internal(
            Wy, WX, compute_eigenvalues=compute_eigenvalues)

        beta = result.params
        if not is_sparse:
            beta = beta.toarray().flatten()
        beta = beta.reshape((-1, 1))

        # Step 4: check AR coefficient stability (skip on first AR iteration)
        if ar_last is not None:
            ar_error = np.max(np.abs(ar_params - ar_last))
            if debug:
                print(f'{itr=}, {ar_error=}, {ar_params=}, {scale=}')
            if ar_error < tol:
                break

        # Step 5: update residuals on original scale for next AR estimation
        e = endog - exog.dot(beta)
        if is_sparse:
            e = e.toarray().flatten()
        ar_last = ar_params

    if isspmatrix(beta):
        beta = beta.toarray().flatten()

    # Normalized (X'X)^{-1} from final whitened fit — used downstream for cov_params
    ncp = result.normalized_cov_params

    return beta, ncp, result, Wy, WX, ar_params, scale, itr + 2, ar_error
