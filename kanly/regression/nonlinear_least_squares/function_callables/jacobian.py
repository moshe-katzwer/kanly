from __future__ import absolute_import, print_function

import numpy as np
from numba import njit
from scipy.sparse import isspmatrix, csc_matrix

from kanly.regression.nonlinear_least_squares.constants import DEFAULT_NLLS_JAC_METHOD
from kanly.utils.linalg_utils import DEFAULT_DENSE_THRESHOLD_MB, DenseThreshold


@njit(cache=True)
def _get_finite_difference_univariate(r1, r, step):
    """Compute one finite-difference Jacobian column and its non-zero rows.

    Args:
        r1: Function values after perturbing one parameter.
        r: Baseline or backward function values.
        step: Perturbation distance used in the denominator.

    Returns:
        Tuple ``(diff_k, nz_idx)`` where ``diff_k`` is the finite-difference
        derivative vector and ``nz_idx`` are indices with non-zero derivatives.
    """
    diff_k = (r1 - r) / step
    nz_idx = np.nonzero(diff_k)[0]
    return diff_k, nz_idx


def get_finite_diff_jacobian(func, dense_threshold_mb=DEFAULT_DENSE_THRESHOLD_MB, jac_method=DEFAULT_NLLS_JAC_METHOD):
    """Build a finite-difference Jacobian callable for a vector-valued function.

    The returned closure evaluates one parameter perturbation at a time and
    returns either a dense array or CSC/CSR sparse matrix depending on the
    expected Jacobian size and the requested ``return_type``.

    Args:
        func: Callable mapping a parameter vector to a vector of predictions or
            residuals.
        dense_threshold_mb: Maximum estimated dense matrix size, in MB, before
            sparse output is preferred.
        jac_method: Finite-difference method, either ``'fwd'`` for forward
            differences or ``'mid'`` for central differences.

    Returns:
        Callable ``jacobian_function_to_return(params, ...)`` that evaluates
        the finite-difference Jacobian at ``params``.
    """
    jac_method = jac_method.lower()

    def jacobian_function_to_return(params, f0=None, idx=None, step=1e-6, weights=None, return_type='csc'):
        """Evaluate the finite-difference Jacobian at ``params``.

        Args:
            params: Parameter vector at which to evaluate derivatives.
            f0: Optional pre-computed baseline function values.
            idx: Optional row indexer; when provided, only these observations
                are returned.
            step: Relative finite-difference step size.
            weights: Optional observation weights applied row-wise to the
                returned Jacobian.
            return_type: Sparse return format when the result is not dense:
                ``'csc'`` or ``'csr'``.

        Returns:
            Dense ``ndarray`` or sparse matrix with shape
            ``(n_observations, n_params)``.
        """

        params = np.asarray(params).astype(float)

        if f0 is None:
            f0 = func(params)
            if idx is not None:
                f0 = f0[idx]

        if isspmatrix(f0):
            f0 = f0.toarray().flatten()

        f0 = np.asarray(f0)

        nobs = len(f0)
        num_params = len(params)

        is_dense = DenseThreshold.is_below_threshold_dim(
            (nobs, num_params), dense_threshold_mb=dense_threshold_mb)
        if is_dense:
            J = np.zeros((nobs, num_params))
        else:
            _data = []
            _indptr = []
            _indices = []

        for j in range(num_params):

            h_j = step * max(1.0, abs(params[j]))

            params_copy = params.copy()
            params_copy[j] += h_j

            # y_hat_j = func(params_copy, float_arr, int_arr, idx=idx)
            y_hat_j_fwd = np.asarray(func(params_copy))
            if idx is not None:
                y_hat_j_fwd = y_hat_j_fwd[idx]

            if jac_method == 'fwd':
                y_hat_j_bwd = f0
                delta_x = h_j
            elif jac_method == 'mid':
                params_copy[j] -= 2 * h_j
                y_hat_j_bwd = np.asarray(func(params_copy))
                if idx is not None:
                    y_hat_j_bwd = y_hat_j_bwd[idx]
                delta_x = 2 * h_j
            else:
                raise Exception("jacobian method must be 'fwd' or 'mid'")

            fin_diff_j, nz_idx = _get_finite_difference_univariate(y_hat_j_fwd, y_hat_j_bwd, delta_x)

            if is_dense:
                J[:, j] = fin_diff_j

            else:
                _data.append(fin_diff_j[nz_idx])
                _indices.append(nz_idx)

                # CSC stores column pointers in ``indptr``; append the running
                # count of non-zero row derivatives after each parameter.
                if j == 0:
                    _indptr = [0, len(nz_idx)]
                else:
                    _indptr.append(_indptr[-1] + len(nz_idx))

        if is_dense:

            if weights is not None:
                if idx is not None:
                    weights = weights[idx]
                J *= weights.reshape((-1,1))

            return J

        else:
            _data = np.hstack(_data)
            _indices = np.hstack(_indices)

            if weights is not None:
                if idx is not None:
                    weights = weights[idx]
                _data *= np.asarray(weights)[_indices]

            J = csc_matrix((_data, _indices, _indptr), shape=(nobs, num_params), dtype=np.float64)

            if return_type == 'csc':
                return J
            if return_type == 'csr':
                return J.tocsr()
            else:
                raise Exception

    return jacobian_function_to_return
