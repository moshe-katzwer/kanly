"""Variance-covariance helpers for generalized method of moments estimates."""

from __future__ import absolute_import, print_function

import numpy as np

from kanly.utils.linalg_utils import get_eigenvals_and_condition_number_internal, get_matrix_inverse_internal
from kanly.regression.linear_models.variance_covariance2 import SparseVarianceCovariance2
from scipy.sparse import csr_matrix


def get_Omega(moment_func_obs, nobs, params, cluster_groups=None):
    """Estimate the covariance matrix of observation-level moments.

    Args:
        moment_func_obs: Callable returning an ``nobs x num_moments`` sparse
            matrix of moment values for ``params``.
        nobs: Number of observations.
        params: Fitted parameter vector.
        cluster_groups: Optional cluster labels. When supplied, moments are
            summed within clusters before forming the covariance estimate.

    Returns:
        Tuple ``(Omega, num_clusters)`` where ``Omega`` is the moment covariance
        matrix and ``num_clusters`` is one for non-clustered covariance.
    """
    # mmo is the n x k matrix of moments, where n is num of obs
    mmo = moment_func_obs(params)
    num_clusters = 1
    if cluster_groups is not None:
        # Convert observation-level moments into cluster-level summed moments.
        group_to_row, num_clusters = SparseVarianceCovariance2._get_cluster_group_info2(cluster_groups)
        E_2 = csr_matrix((np.ones(nobs), (group_to_row, np.arange(nobs))), shape=(nobs, nobs))
        mmo = E_2.dot(mmo)

    Omega = mmo.transpose().dot(mmo).toarray() / nobs
    return Omega, num_clusters


def get_gmm_var_covar(moment_func_mean_jacobian, moment_func_obs, nobs, theta, W, Omega=None, ss_correction=1.0,
                      cluster_groups=None):
    """Compute the GMM sandwich variance-covariance matrix.

    Args:
        moment_func_mean_jacobian: Callable returning ``G = d E[g(theta)] /
            d theta``.
        moment_func_obs: Callable returning observation-level moment values.
        nobs: Number of observations.
        theta: Fitted parameter vector.
        W: Final GMM weighting matrix.
        Omega: Optional precomputed moment covariance matrix.
        ss_correction: Small-sample correction multiplier.
        cluster_groups: Optional cluster labels for clustered moment covariance.

    Returns:
        Tuple ``(var_covar, Omega, condition_number, eigenvals)`` containing the
        parameter covariance matrix, the moment covariance matrix, and numerical
        diagnostics for ``G'WG``.
    """

    G = moment_func_mean_jacobian(theta)

    if Omega is None:
        Omega, _ = get_Omega(moment_func_obs, nobs, theta, cluster_groups=cluster_groups)

    # Bread matrix for the GMM estimator; its conditioning is useful for
    # identifying weak or nearly collinear moment information.
    GWG = G.transpose().dot(W).dot(G)
    eigenvals, condition_number = get_eigenvals_and_condition_number_internal(GWG)

    GWGinv = get_matrix_inverse_internal(GWG)

    # Sandwich meat: G' W Omega W G.
    meat = G.transpose().dot(W).dot(Omega).dot(W).dot(G)

    var_covar = GWGinv.dot(meat).dot(GWGinv) * (ss_correction / nobs)

    return var_covar, Omega, condition_number, eigenvals
