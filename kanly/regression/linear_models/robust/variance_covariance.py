"""M-estimator sandwich covariance matrices for robust linear regression.

Implements four analytic covariance variants (H1, H2, H3, SANDWICH) following
Huber (1981) and a BOOTSTRAP placeholder that is handled upstream in
``model.py``.

References:
    Huber, P. J. (1981). *Robust Statistics*. Wiley.
"""
from __future__ import absolute_import, print_function

import numpy as np

from kanly.utils.linalg_utils import csc_matrix_by_column_array_broadcast
from kanly.utils.linalg_utils import get_matrix_inverse_internal

# --- Covariance type string constants ---
# H1/H2/H3 are the three variants from Huber (1981), differing in how they
# account for the expected Hessian of the loss.
H1 = 'H1'
H2 = 'H2'
H3 = 'H3'
# Classic A⁻¹BA⁻¹ heteroscedasticity-consistent sandwich estimator.
SANDWICH = 'SANDWICH'
# Bootstrap covariance: not computed here; handled in model.py.
BOOTSTRAP = 'BOOTSTRAP'
# Sentinel used when covariance computation is skipped (compute_cov=False).
NOT_COMPUTED = 'NOT COMPUTED'
# Tuple of all valid analytic cov_type strings (BOOTSTRAP is accepted but
# raises NotImplementedError here so that model.py can intercept it first).
RLM_COV_TYPES = (H1, H2, H3, SANDWICH, BOOTSTRAP)


def check_cov_type(cov_type):
    """Validate that ``cov_type`` is a recognised covariance type string.

    Args:
        cov_type (str): Covariance type identifier.  Must be one of
            ``'H1'``, ``'H2'``, ``'H3'``, ``'SANDWICH'``, or ``'BOOTSTRAP'``
            (case-insensitive).

    Returns:
        None

    Raises:
        Exception: If ``cov_type`` is not in ``RLM_COV_TYPES``.
    """
    if cov_type.upper() not in RLM_COV_TYPES:
        raise Exception(f'cov_type "{cov_type}" is not supported, choose one of {RLM_COV_TYPES}!')


def get_rlm_variance_covariance(df_model, df_resid, nobs, exog, resid, scale, M, cov_type):
    """Compute the covariance matrix of the M-estimator coefficients.

    Dispatches to one of four analytic sandwich variants (H1, H2, H3,
    SANDWICH) based on ``cov_type``.  The BOOTSTRAP branch raises
    ``NotImplementedError``; it is intercepted earlier by ``model.py``.

    The small-sample correction factor ``k`` (see below) adjusts for
    estimation error in the scale and norm parameters and is applied in all
    analytic variants.

    **H1** (default — ``'H1'``)::

        Cov(β̂) = k² · (Σψ²·σ²/df_resid) / (Σψ′/n)² · (XᵀX)⁻¹

    **H2** (``'H2'``)::

        Cov(β̂) = k · (Σψ²·σ²/df_resid) / (Σψ′/n) · (XᵀΨ′X)⁻¹

    **H3** (``'H3'``)::

        Cov(β̂) = (1/k) · (Σψ²·σ²/df_resid) · (XᵀΨ′X)⁻¹ · XᵀX · (XᵀΨ′X)⁻¹

    **SANDWICH** (``'SANDWICH'``)::

        A = XᵀΨ′X / σ²,  B = XᵀΨΨᵀX / σ²
        Cov(β̂) = A⁻¹ · B · A⁻¹ · n/df_resid

    where ψ = M.psi(r/σ̂), ψ′ = M.psi_deriv(r/σ̂), σ̂ = ``scale``.

    Args:
        df_model (int): Degrees of freedom of the model (number of regressors
            excluding intercept).
        df_resid (int): Residual degrees of freedom (n − p).
        nobs (int): Number of observations.
        exog (scipy.sparse.csc_matrix): Design matrix, shape (n, p).
        resid (ndarray): Final residuals r = y − Xβ̂, shape (n,).
        scale (float): MAD-based scale estimate σ̂.
        M (RobustNormFunction): Norm object providing ``psi`` and
            ``psi_deriv`` methods.
        cov_type (str): One of ``'H1'``, ``'H2'``, ``'H3'``, ``'SANDWICH'``,
            or ``'BOOTSTRAP'`` (case-insensitive).

    Returns:
        ndarray: Covariance matrix of shape (p, p).

    Raises:
        NotImplementedError: If ``cov_type`` is ``'BOOTSTRAP'`` (delegated to
            ``model.py``) or an unrecognised string.
    """
    scaled_resid = resid / scale
    psi = M.psi(scaled_resid)
    psi_deriv = M.psi_deriv(scaled_resid)

    m = psi_deriv.mean()
    var_psiprime = np.var(psi_deriv)

    # Small-sample correction factor k accounts for variability in ρ″ across
    # observations; k → 1 as n → ∞ so its effect vanishes in large samples.
    k = 1 + (df_model + 1) / nobs * var_psiprime / m ** 2

    if cov_type.upper() == H1:
        xpx_inv = get_matrix_inverse_internal(exog.transpose().dot(exog).toarray())

        ss_psi = np.sum(M.psi(scaled_resid) ** 2)
        s_psi_deriv = np.sum(psi_deriv)

        scalar = k ** 2 * (1 / df_resid * ss_psi * scale ** 2) / ((1 / nobs * s_psi_deriv) ** 2)
        return scalar * xpx_inv

    elif cov_type.upper() == H2:
        exog_w = csc_matrix_by_column_array_broadcast(exog, psi_deriv)
        W_inv = get_matrix_inverse_internal(exog_w.transpose().dot(exog_w).toarray())
        return k * (1 / df_resid) * sum(psi ** 2) * scale ** 2 / (
                    (1 / nobs) * sum(psi_deriv)) * W_inv

    elif cov_type.upper() == H3:
        xpx = exog.transpose().dot(exog).toarray()
        exog_w = csc_matrix_by_column_array_broadcast(exog, psi_deriv)
        W_inv = get_matrix_inverse_internal(exog_w.transpose().dot(exog_w).toarray())
        return 1 / k * (1 / df_resid * sum(psi ** 2)) * scale ** 2 * W_inv.dot(xpx).dot(W_inv)

    elif cov_type.upper() == SANDWICH:

        exog_w = csc_matrix_by_column_array_broadcast(exog, psi_deriv ** .5)
        A = exog_w.transpose().dot(exog_w).toarray() / scale**2
        A_inv = get_matrix_inverse_internal(A)

        exog_w = csc_matrix_by_column_array_broadcast(exog, psi)
        B = exog_w.transpose().dot(exog_w).toarray() / scale ** 2

        return A_inv.dot(B).dot(A_inv) * nobs / df_resid

    elif BOOTSTRAP in cov_type.upper():
        raise NotImplementedError

    else:
        raise NotImplementedError(f"cov_type {cov_type}")
