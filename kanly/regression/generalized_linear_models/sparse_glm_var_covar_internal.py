from __future__ import absolute_import, print_function

import time

import numpy as np
from scipy.sparse import diags, isspmatrix

from kanly.utils.linalg_utils import get_matrix_inverse_internal, csc_matrix_by_column_array_broadcast
from kanly.regression.generalized_linear_models.constants import HC1, NONROBUST


def _get_l2_diagonal_for_cov(nobs, l2s, fit_intercept, first_column_constant, var_weights):
    """Build the ridge/L2 diagonal adjustment for GLM covariance bread.

    Args:
        nobs: Number of observations.
        l2s: Per-parameter L2 penalties from the fitted model.
        fit_intercept: Whether an intercept is represented separately.
        first_column_constant: Whether the first exogenous column is already
            the intercept/constant.
        var_weights: Observation variance weights.

    Returns:
        Dense diagonal matrix added to the Hessian/bread when penalization is
        active.
    """
    l2_arr = list(2 * nobs * np.asarray(l2s))
    if fit_intercept:
        if first_column_constant:
            l2_arr[0] = 0
        else:
            l2_arr = [0] + l2_arr
    if isinstance(var_weights, (float, int)):
        var_weights_mean = var_weights
    else:
        var_weights_mean = var_weights.mean()
    return np.diag(l2_arr) * var_weights_mean


def _get_XpX(exog_sparse, weights, fit_intercept, first_column_constant):
    """Compute ``X' diag(weights) X`` with optional explicit intercept column.

    Args:
        exog_sparse: Sparse design matrix.
        weights: Observation weights used on the diagonal.
        fit_intercept: Whether an intercept is fit separately from ``exog``.
        first_column_constant: Whether ``exog_sparse`` already includes the
            intercept as its first column.

    Returns:
        Dense weighted cross-product matrix.
    """
    add_first_col = fit_intercept and not first_column_constant
    K = exog_sparse.shape[1] + add_first_col

    meat_weights = np.zeros((K, K))
    # Scale rows before the sparse cross-product to avoid materializing a full
    # diagonal weight matrix.
    temp1 = csc_matrix_by_column_array_broadcast(exog_sparse, weights) #exog_sparse.transpose().dot(diags(weights))
    temp = temp1.transpose().dot(exog_sparse)

    if isspmatrix(temp):
        temp = temp.toarray()

    if add_first_col:
        meat_weights[1:, 1:] = temp
        meat_weights[0, 0] = np.sum(weights)
        temp2 = temp1.sum(axis=1).flatten()
        meat_weights[0, 1:] = temp2
        meat_weights[1:, 0] = temp2

    else:
        meat_weights = temp

    return meat_weights


def get_robust_glm_covariance(
        endog, exog, endog_predicted, var_weights, irls_weights, family, link, scale, cov_type,
        fit_intercept, first_column_constant, alpha, l2s, L2_penalty_matrix, normalized_cov_params=None):
    """Compute GLM covariance estimates from final fitted means and weights.

    Supports the model-based nonrobust covariance and an HC1-style sandwich
    covariance. Penalized fits add the L2 diagonal and/or
    ``L2_penalty_matrix`` to the bread matrix before inversion.

    Args:
        endog: Observed response vector.
        exog: Sparse design matrix used for the final GLM fit.
        endog_predicted: Final fitted mean response values.
        var_weights: Optional variance weights.
        irls_weights: Final IRLS weights.
        family: GLM family instance.
        link: Link instance.
        scale: Estimated dispersion/scale parameter.
        cov_type: Covariance type, either ``NONROBUST`` or ``HC1``.
        fit_intercept: Whether an intercept is represented separately.
        first_column_constant: Whether the first design column is already a constant.
        alpha: Overall regularization strength.
        l2s: Per-parameter L2 penalties.
        L2_penalty_matrix: Optional general ridge or GAM roughness matrix used
            by IRLS; added to the covariance bread when present.
        normalized_cov_params: Optional precomputed normalized covariance
            matrix. Reserved for API compatibility.

    Returns:
        Tuple ``(var_covar, cov_time)`` containing the covariance matrix and
        elapsed computation time in seconds.
    """

    _time_cov_start = time.time()

    nobs = len(endog)
    cov_type = cov_type.upper()

    if var_weights is None:
        var_weights = 1.0

    final_irls_wts = var_weights * irls_weights
    bread_diag = final_irls_wts.copy()

    if cov_type == NONROBUST:

        bread = _get_XpX(exog, bread_diag, fit_intercept, first_column_constant)
        if np.any(alpha) > 0:
            bread += _get_l2_diagonal_for_cov(nobs, l2s, fit_intercept, first_column_constant, var_weights)
        if L2_penalty_matrix is not None:
            bread += (L2_penalty_matrix.toarray()
                      if isspmatrix(L2_penalty_matrix)
                      else np.asarray(L2_penalty_matrix))
        bread = get_matrix_inverse_internal(bread)

        if np.any(alpha) > 0:
            meat = _get_XpX(exog, bread_diag, fit_intercept, first_column_constant)
            var_covar = scale * bread.dot(meat).dot(bread)
        else:
            var_covar = scale * bread

    elif cov_type == HC1:

        var_ = family.variance(endog_predicted)

        if not family.is_canonical(link):
            # Non-canonical links add curvature sparse_terms to the expected Hessian;
            # canonical links collapse this expression to the usual IRLS bread.
            g_prime_ = link.deriv(endog_predicted)
            bread_diag += (
                    (endog - endog_predicted) * (
                    family.d_variance(endog_predicted) * g_prime_ + var_ * link.deriv2(endog_predicted))
                    / (g_prime_ * var_) * final_irls_wts
            )

        bread = _get_XpX(exog, bread_diag, fit_intercept, first_column_constant)
        if alpha > 0:
            bread += _get_l2_diagonal_for_cov(nobs, l2s, fit_intercept, first_column_constant, var_weights)
        if L2_penalty_matrix is not None:
            bread += (L2_penalty_matrix.toarray()
                      if isspmatrix(L2_penalty_matrix)
                      else np.asarray(L2_penalty_matrix))
        bread = get_matrix_inverse_internal(bread)

        meat_weights = ((endog_predicted - endog) * link.deriv(endog_predicted) * final_irls_wts) ** 2
        meat = _get_XpX(exog, meat_weights, fit_intercept, first_column_constant)
        var_covar = bread.dot(meat).dot(bread)

    else:
        raise NotImplementedError("`cov_type` %s not supported!" % cov_type)

    cov_time = time.time() - _time_cov_start

    return var_covar, cov_time
