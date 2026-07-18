from __future__ import absolute_import, print_function

import time
import warnings

import numpy as np
from numpy import ndarray
from numpy.linalg import pinv as np_pinv
from pandas import DataFrame
from scipy.linalg import eigvalsh, pinv as scp_pinv, sqrtm
from scipy.sparse import isspmatrix, csc_matrix, isspmatrix_csc
from scipy.sparse.linalg import eigsh

DEFAULT_DENSE_THRESHOLD_MB = 1024  # mb
DEFAULT_SUFFICIENTLY_SPARSE_THRESHOLD = .1


def _inv_built_in(A):
    """Compute the (pseudo-)inverse of a dense symmetric matrix.

    Tries NumPy's ``pinv`` with the ``hermitian=True`` hint first (which
    exploits symmetry for faster computation).  Falls back to SciPy's ``pinv``
    with rank reporting when NumPy raises, and emits a warning if the matrix
    is rank-deficient.

    Args:
        A: 2-D square NumPy array assumed to be symmetric (or Hermitian).

    Returns:
        Pseudo-inverse of ``A`` as a dense NumPy array.
    """
    # TODO use only pinv here?
    try:
        return np_pinv(A, hermitian=True)
    except:
        inv_A, rank = scp_pinv(A, return_rank=True)
        if rank < A.shape[0]:
            warnings.warn(f"Rank of matrix `A` is {rank}, but has dimension {A.shape[0]} x {A.shape[0]}!")
        return inv_A
    # try:
    #     return np_inv(A)
    # except:
    #     try:
    #         return np_pinv(A)
    #     except:
    #         return scp_pinv(A)


class DenseThreshold:
    """
    Before computing Graham matrix, for performance we convert
    to dense if resulting np.array is less than `dense threshold`
    mb in size
    """

    dense_threshold_mb = DEFAULT_DENSE_THRESHOLD_MB  # mb
    sufficiently_sparse_threshold = DEFAULT_SUFFICIENTLY_SPARSE_THRESHOLD

    @classmethod
    def is_convertible_to_dense(cls, X, dense_threshold_mb=None, sufficiently_sparse_threshold=None):
        """Return ``True`` when it is efficient to convert ``X`` to a dense array.

        The matrix is considered convertible when it is not "sufficiently sparse"
        (fill ratio above threshold) *and* would fit within the memory budget.

        Args:
            X: Sparse matrix to evaluate.
            dense_threshold_mb: Memory limit override (MB); uses class default when ``None``.
            sufficiently_sparse_threshold: Fill-ratio override; uses class default when ``None``.

        Returns:
            ``True`` if both conditions are met; ``False`` otherwise.
        """
        return (cls.is_not_sufficiently_sparse(X, sufficiently_sparse_threshold=sufficiently_sparse_threshold)
                and cls.is_below_threshold(X, dense_threshold_mb=dense_threshold_mb)
                )

    @classmethod
    def is_below_threshold(cls, X, dense_threshold_mb=None):
        """Return ``True`` when the dense version of ``X`` would be under the memory limit.

        Args:
            X: Sparse matrix (shape used to estimate dense memory footprint).
            dense_threshold_mb: Memory limit override (MB); uses class default when ``None``.

        Returns:
            ``True`` if ``prod(X.shape) * 8 bytes <= dense_threshold_mb * 1024^2``.
        """
        return cls.is_below_threshold_dim(X.shape, dense_threshold_mb=dense_threshold_mb)

    @classmethod
    def is_below_threshold_dim(cls, shape, dense_threshold_mb=None):
        """Return ``True`` when the given shape would produce a dense array under the memory limit.

        Args:
            shape: Tuple of matrix dimensions ``(nrows, ncols)``.
            dense_threshold_mb: Memory limit override (MB); uses class default when ``None``.

        Returns:
            Boolean.
        """
        dense_threshold_mb = cls.dense_threshold_mb if dense_threshold_mb is None else dense_threshold_mb
        return np.prod(shape) * 8 / 1024 ** 2 <= dense_threshold_mb

    @classmethod
    def is_not_sufficiently_sparse(cls, X, sufficiently_sparse_threshold=None):
        """Return ``True`` when the fill ratio of ``X`` meets or exceeds the sparsity threshold.

        A matrix is considered "not sufficiently sparse" — and therefore a
        candidate for dense conversion — when its fill ratio
        ``nnz / (nrows * ncols)`` is at or above ``sufficiently_sparse_threshold``.

        Args:
            X: Sparse matrix with an ``nnz`` attribute.
            sufficiently_sparse_threshold: Fill-ratio override; uses class default when ``None``.

        Returns:
            Boolean.
        """
        sufficiently_sparse_threshold = (cls.sufficiently_sparse_threshold if sufficiently_sparse_threshold is None
                                         else sufficiently_sparse_threshold)
        return X.nnz / np.prod(X.shape) >= sufficiently_sparse_threshold

# TODO REMOVE
# def check_weights_dim_dense(weights):
#     """Checks dimension of weights, whether vector or matrix
#     returns
#         weights flattened (if necessary), or the square matrix
#         True/False for whether vector (True) or matrix (False)
#     """
#     if np.ndim(weights) == 1:
#         return weights, True
#     else:
#         if np.ndim(weights) != 2:
#             raise Exception("`weights` must be matrix or vector!")
#         shp = np.shape(weights)
#         if shp[0] == 1:
#             return weights[0], True
#         elif shp[1] == 1:
#             return weights.ravel(), True
#         else:
#             if shp[1] != shp[0]:
#                 raise Exception("`weights` must be square matrix if not a vector")
#             return weights, False


def gram_matrix(X, weights=None, sigma=None, sigma_inv=None,
                dense_threshold_mb=DEFAULT_DENSE_THRESHOLD_MB, debug=False):
    """
    Return X.T W X
    where W = Identity if weights=None

    Converts to dense if below a certain sizedense_threshold_mb=DenseThreshold.dense_threshold_mb
    """

    _t = time.time()
    if debug:
        print("Computing gram matrix ... ", end='')
    is_weighted = weights is not None
    is_gls = sigma is not None or sigma_inv is not None
    is_dense = not isspmatrix(X)
    if is_dense:
        if is_weighted:
            Xw = X * np.sqrt(weights).reshape((-1, 1))
            gm = Xw.T.dot(Xw)
        elif is_gls:
            if sigma_inv is None:
                sigma_inv = get_matrix_inverse_internal(sigma, normalize=False, copy=False)
            gm = X.T.dot(sigma_inv).dot(X)
        else:
            gm = X.T.dot(X)
    else:
        if DenseThreshold.is_convertible_to_dense(X, dense_threshold_mb=dense_threshold_mb):
            if debug:
                print('converting to dense ... ', end='')
            gm = gram_matrix(X.toarray(), weights=weights)
        elif is_gls:
            if sigma_inv is None:
                raise NotImplementedError()
            else:
                gm = X.transpose().dot(sigma_inv.dot(X))
        else:
            if is_weighted:
                Xw = csc_matrix_by_column_array_broadcast(X, weights ** .5)
                gm = Xw.transpose().dot(Xw).toarray()
            else:
                gm = X.transpose().dot(X).toarray()

    if debug:
        print(f' {time.time() - _t} s')

    return gm


def scale_in_place(X):
    """Normalise each column of ``X`` by its maximum absolute value, in-place.

    Divides every element of column ``j`` by ``max(0.01, max|X[:, j]|)``.
    The floor of ``0.01`` prevents division by near-zero columns.  Works for
    both dense NumPy arrays and CSC sparse matrices.

    Args:
        X: 2-D dense NumPy array or CSC sparse matrix to be scaled in-place.

    Returns:
        1-D NumPy array of length ``X.shape[1]`` containing the scale factors
        applied to each column.  Divide by these to reverse the scaling.
    """
    x_scales = np.zeros(X.shape[1])
    if isspmatrix(X):
        for j in range(X.shape[1]):
            x_scales[j] = max(
                .01,
                np.abs(X.data[X.indptr[j]:X.indptr[j + 1]]).max()
            )
            X.data[X.indptr[j]:X.indptr[j + 1]] /= x_scales[j]
    else:
        for j in range(X.shape[1]):
            x_scales[j] = max(.01, np.abs(X[:, j]).max())
            X[:, j] /= x_scales[j]

    return x_scales


def unscale_in_place(X, x_scales):
    """Undo column-wise scaling applied by ``scale_in_place``, in-place.

    Multiplies every element of column ``j`` by ``x_scales[j]``.

    Args:
        X: 2-D dense NumPy array or CSC sparse matrix to be un-scaled in-place.
        x_scales: 1-D array of scale factors as returned by ``scale_in_place``.
    """
    if isspmatrix(X):
        for j in range(X.shape[1]):
            X.data[X.indptr[j]:X.indptr[j + 1]] *= x_scales[j]
    else:
        for j in range(X.shape[1]):
            X[:, j] *= x_scales[j]


def get_eigenvals_and_condition_number_internal(A, is_inverse=False):
    """Compute the eigenvalues and spectral condition number of a symmetric matrix.

    For 1×1 matrices the result is returned analytically.  For larger matrices
    the function dispatches to ``scipy.sparse.linalg.eigsh`` (sparse) or
    ``scipy.linalg.eigvalsh`` (dense).  The condition number is defined as
    ``sqrt(lambda_max / lambda_min)``; it is ``inf`` when the smallest
    eigenvalue is non-positive (rank-deficient matrix).

    Args:
        A: Square symmetric (or Hermitian) 2-D array or sparse matrix.
        is_inverse: When ``True``, return the eigenvalues of ``A^{-1}`` instead
            (i.e. the reciprocals).

    Returns:
        Tuple ``(eigenvals, condition_number)`` where ``eigenvals`` is a list
        of floats sorted in descending order and ``condition_number`` is a
        non-negative float (or ``inf``).
    """
    if np.prod(A.shape) == 1:
        if is_inverse:
            return [1.0 / A.item()], 1.0
        else:
            return [A.item()], 1.0

    if isspmatrix(A):
        eigenvals = eigsh(A, return_eigenvectors=False, k=A.shape[1] - 1, tol=0., which='BE')
    else:
        eigenvals = eigvalsh(A)

    if is_inverse:
        eigenvals = 1.0 / eigenvals

    eigenvals = sorted(eigenvals)[::-1]
    if eigenvals[-1] <= 0:
        condition_number = np.inf
    else:
        condition_number = np.sqrt(eigenvals[0] / eigenvals[-1])

    return eigenvals, condition_number


def get_normalized_cov_params(X, weights=None, sigma=None, sigma_inv=None, normalize=True,
                              return_eigenvals=False, debug=False, _time=None,
                              compute_eigenvalues=False, ridge_parameters=None,
                              dense_threshold_mb=DEFAULT_DENSE_THRESHOLD_MB, inverse_method=None):
    """Compute the normalised covariance parameter matrix ``(X'WX)^{-1}``.

    Forms the Gram matrix ``X'WX`` (WLS) or ``X'Σ^{-1}X`` (GLS), optionally
    adds a ridge penalty, and then inverts it via ``get_matrix_inverse_internal``.
    The normalisation pre-scales the matrix by its diagonal to improve numerical
    stability before inversion.

    Args:
        X: Design matrix (dense or sparse), shape ``(n, p)``.
        weights: Optional 1-D weight vector of length ``n`` for WLS.  Mutually
            exclusive with ``sigma``/``sigma_inv``.
        sigma: Optional ``n×n`` error covariance matrix for GLS.  Either
            ``sigma`` or ``sigma_inv`` may be provided, not both.
        sigma_inv: Pre-computed inverse of ``sigma``.
        normalize: When ``True`` pre-condition ``X'WX`` by its diagonal before
            inverting (improves numerical stability for ill-conditioned matrices).
        return_eigenvals: When ``True``, return a 3-tuple
            ``(ncp, eigenvals, condition_number)``; requires
            ``compute_eigenvalues=True`` to get actual values.
        debug: Print timing information to stdout when ``True``.
        _time: Optional start time for debug timing (``time.time()`` value).
        compute_eigenvalues: When ``True``, compute eigenvalues of ``X'WX``
            before inversion.
        ridge_parameters: Optional 1-D array of ridge penalties to add to the
            diagonal of ``X'WX``.
        dense_threshold_mb: Size limit (MB) below which a sparse ``X`` is
            converted to dense before forming the Gram matrix.
        inverse_method: Optional callable to use instead of ``_inv_built_in``
            for matrix inversion.

    Returns:
        When ``return_eigenvals=False``: the normalised covariance parameter
        matrix ``(X'WX)^{-1}`` (dense or CSC).
        When ``return_eigenvals=True``: tuple
        ``(ncp, eigenvals, condition_number)`` where eigenvals/condition_number
        are ``None`` if ``compute_eigenvalues=False``.
    """
    if _time is None:
        _time = time.time()

    check_weights_and_sigma(weights, sigma, sigma_inv)

    XpX = gram_matrix(X, sigma=sigma, sigma_inv=sigma_inv, weights=weights,
                      dense_threshold_mb=dense_threshold_mb, debug=debug)

    if ridge_parameters is not None:
        if debug:
            print('\n\tAdding ridge parameters...', end='')
        np.fill_diagonal(XpX, np.diag(XpX) + np.asarray(ridge_parameters).flatten())
        if debug:
            print("%.3fs" % (time.time() - _time))

    if compute_eigenvalues:
        # do it before ncp because ncp mutates XpX
        if debug:
            print("\tComputing eigenvalues and condition number..", end='')
        eigenvals, condition_number = get_eigenvals_and_condition_number_internal(XpX)
        if debug:
            print("%.3fs" % (time.time() - _time))

    if debug:
        print("\tInverting X'X to get normalized cov params..", end='')

    ncp = get_matrix_inverse_internal(XpX, copy=False, normalize=normalize, inverse_method=inverse_method)
    if isspmatrix(X):
        ncp = csc_matrix(ncp)

    if debug:
        print("%.3fs" % (time.time() - _time))

    if return_eigenvals:
        if compute_eigenvalues:
            return ncp, eigenvals, condition_number
        else:
            return ncp, None, None
    else:
        return ncp


def get_matrix_inverse_internal(X, copy=False, normalize=True, return_csc=False, inverse_method=None):
    """X should be dense"""

    if inverse_method is None:
        inverse_method = _inv_built_in

    if isspmatrix(X):
        X = X.toarray()

    if normalize:
        diag_sqrt = 1.0 / np.clip(np.sqrt(np.diag(X)), a_min=1, a_max=np.inf)
        if copy:
            X = X.copy()
        X *= diag_sqrt.reshape((1, -1))
        X *= diag_sqrt.reshape((-1, 1))

    X_inv = inverse_method(X)

    if normalize:
        X_inv *= diag_sqrt.reshape((1, -1))
        X_inv *= diag_sqrt.reshape((-1, 1))

    if return_csc:
        X_inv = csc_matrix(X_inv)

    return X_inv


def get_wtd_sum_squares(v, weights=None, demean=False):
    """Compute the (optionally weighted and demeaned) sum of squares of a vector.

    For 2-D inputs with more than one column, applies column-wise and returns
    an array.

    Args:
        v: 1-D (or 2-D) array-like of values, dense or sparse.
        weights: Optional 1-D weight vector.  When provided, computes
            ``sum(w * (v - mean_w)^2)`` where ``mean_w`` is the weighted mean
            (if ``demean=True``) or plain ``sum(w * v^2)`` otherwise.
        demean: When ``True``, subtract the (weighted) mean before squaring.

    Returns:
        Scalar sum of squares for 1-D input, or a 1-D array for 2-D input.
    """
    if np.ndim(v) > 1 and v.shape[1] > 1:
        return np.array([get_wtd_sum_squares(v[:, k], weights=weights, demean=demean)
                         for k in range(v.shape[1])])
    else:
        if isspmatrix(v):
            v = v.toarray().ravel()
        else:
            v = np.asarray(v).ravel()

    if weights is None:
        return ((v - (v.mean() if demean else 0.0)) ** 2).sum()
    else:
        if isspmatrix(weights):
            weights = weights.toarray().ravel()

        return (weights * (
                v - (np.average(v, weights=weights) if demean else 0.0)
        ) ** 2).sum()


def to_dense_helper(x, flatten=False, copy=True):
    """Convert a sparse matrix (or pass through a dense array) to a dense NumPy array.

    Returns ``None`` unchanged.  For sparse inputs, converts to a dense array
    with ``toarray()``.  Optionally flattens the result to 1-D.

    Args:
        x: Sparse matrix, dense NumPy array, or ``None``.
        flatten: When ``True``, return a 1-D flattened copy.
        copy: Unused in this version; a copy is always returned.

    Returns:
        Dense NumPy array (or ``None`` when ``x`` is ``None``).
    """
    if x is None:
        return None

    if isspmatrix(x):
        x = x.toarray().copy()

    if flatten:
        return x.flatten().copy()
    else:
        return x.copy()


def matrix_by_column_array_broadcast_to_csc(mat, wts):
    """Multiply each row of ``mat`` by the corresponding element of ``wts``, returning a CSC matrix.

    Dispatches to dense elementwise broadcast (``mat * wts.reshape(-1, 1)``) for
    ``ndarray``/``DataFrame`` inputs and to
    ``csc_matrix_by_column_array_broadcast`` for sparse inputs.

    Args:
        mat: 2-D dense array, pandas ``DataFrame``, or sparse matrix of shape
            ``(n, p)``.
        wts: 1-D array of length ``n``; each element scales the corresponding
            row of ``mat``.

    Returns:
        CSC sparse matrix of shape ``(n, p)`` equal to ``diag(wts) @ mat``.

    Raises:
        Exception: If ``mat`` is not an ``ndarray``, ``DataFrame``, or sparse
            matrix.
    """
    if isinstance(mat, ndarray) or isinstance(mat, DataFrame):
        return csc_matrix(mat * wts.reshape((-1, 1)))
    elif isspmatrix(mat):
        return csc_matrix_by_column_array_broadcast(mat, wts)
    else:
        raise Exception(type(mat))


def csc_matrix_by_column_array_broadcast(mat, col_arr):
    """Multiply each row of a sparse (or dense) matrix by a scalar from ``col_arr``.

    For dense inputs, falls back to standard NumPy broadcasting.  For sparse
    CSC matrices, scales the stored values directly without materialising a
    dense intermediate.

    Args:
        mat: 2-D matrix (dense or sparse) of shape ``(n, p)``.
        col_arr: 1-D array-like of length ``n``; element ``i`` scales row ``i``
            of ``mat``.

    Returns:
        CSC sparse matrix of shape ``(n, p)`` equal to ``diag(col_arr) @ mat``.

    Raises:
        Exception: If the row count of ``mat`` and the length of ``col_arr``
            do not match.
    """
    if not isspmatrix(mat):
        return mat * col_arr.reshape((-1, 1))

    if not isspmatrix_csc(mat):
        mat = mat.tocsc(copy=False)
    col_arr = np.asarray(col_arr).flatten()
    if col_arr.shape[0] != mat.shape[0]:
        raise Exception(f"`mat` ({mat.shape[0]}) and `col_arr` ({col_arr.shape[0]}) must have the same length!")
    return csc_matrix((mat.data * col_arr[mat.indices],
                       mat.indices,
                       mat.indptr), shape=mat.shape)


def sandwich_diagonal(X, W):
    """
    return diag(X W X') for symmetric positive definite W.
    X and W dense or sparse, but W will be converted to array.

    We decompose W into (E diag(V) E') where V are eigenvalues and
    E are eigenvectors.  Then W ** 0.5 = (E diag(V ** 0.5)) and

    diag(X W X') = ((X (W**0.05)) .** 2).sum(axis=1)
    """
    if isspmatrix(W):
        W = W.toarray()
    evalues, evectors = np.linalg.eigh(W)

    if np.any(evalues < 0.0):
        raise Exception('`W` has negative eigenvalues and is not positive semidefinite')

    rt_W = (evectors * (evalues ** .5)).dot(evectors.T)

    if isspmatrix(X):
        rt_W = csc_matrix(rt_W)
        return np.array((X.dot(rt_W).power(2)).sum(axis=1)).flatten()
    else:
        return ((X.dot(rt_W)) ** 2).sum(axis=1)


def flexible_mat_dot_vec(X, y, return_dense=True):
    """dot(X, y) or sparse or dense X, y"""

    if X.shape[1] != y.shape[0]:
        raise Exception(f'{X.shape=} not aligned with {y.shape=}\n'
                        f'{type(X)=}, {type(y)=}')

    if isspmatrix(X) and isspmatrix(y):
        v = X.dot(y)
    elif not isspmatrix(X) and not isspmatrix(y):
        v = np.dot(X, y)
    elif isspmatrix(X) and not isspmatrix(y):
        v = X.dot(csc_matrix(y).reshape((-1, 1)))
    elif not isspmatrix(X) and isspmatrix(y):
        v = X.dot(y.toarray().flatten())

    if return_dense and isspmatrix(v):
        v = v.toarray().flatten()

    return v

def none_convert_2_sparse(Z):
    """if not none, makes sparse"""
    if Z is not None:
        if not isspmatrix(Z):
            Z = csc_matrix(Z)
            if Z.shape[0] == 1:
                Z = Z.reshape((-1,1))
        elif not isspmatrix_csc(Z):
            Z = csc_matrix(Z)
    return Z


def check_weights_and_sigma(weights, sigma, sigma_inv):
    """Validate that only one of WLS weights or GLS covariance is provided.

    Raises an ``AssertionError`` if both WLS and GLS inputs are supplied, or if
    both ``sigma`` and ``sigma_inv`` are simultaneously provided.

    Args:
        weights: Optional 1-D WLS weight vector.
        sigma: Optional ``n×n`` GLS error covariance matrix.
        sigma_inv: Optional pre-computed inverse of ``sigma``.

    Returns:
        Tuple ``(is_weighted, is_gls)`` of booleans indicating which mode is
        active.

    Raises:
        AssertionError: If both WLS and GLS arguments are non-``None``, or if
            both ``sigma`` and ``sigma_inv`` are provided.
    """
    is_gls = sigma is not None or sigma_inv is not None
    is_weighted = weights is not None
    if is_gls:
        assert sigma is None or sigma_inv is None
    assert not (is_weighted and is_gls)
    return is_weighted, is_gls


# if __name__ == '__main__':
#     X = np.random.randn(1000,10)
#     X[:,1] = X[:,0]
#     X = X.T @ X
#     _inv_built_in(X)

