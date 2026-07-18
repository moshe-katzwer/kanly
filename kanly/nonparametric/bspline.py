from __future__ import absolute_import, print_function

import numpy as np
from numba import njit
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix, csr_matrix


def bspline_design_matrix_dense(x, n_bases=10, degree=3, knots=None, include_intercept=True, lower_bound=None,
                                upper_bound=None):
    """
    Construct a B-spline design matrix exploiting local support.

    Each B-spline basis function is nonzero over at most (degree+1) consecutive
    knot spans. For any x[s], only (degree+1) bases are active. The inner kernel
    works with a (degree+1) scratch buffer per point rather than the full
    (n, n_bases) matrix, then scatters results at the end.

    Combined with Numba JIT, this makes runtime essentially independent of
    n_bases — O(n * degree²) rather than O(n * n_bases).

    Parameters
    ----------
    x : array-like, shape (n_samples,)
        Evaluation points.
    n_bases : int
        Number of B-spline basis functions.
    degree : int
        Spline degree (3 = cubic, standard for GAMs).
    knots : array-like or None
        Full knot vector. If None, built from quantile-spaced interior knots.
        Pass back the returned knots to skip recomputation across calls.
    include_intercept : bool, default=True
        If False, drops the very first basis function to prevent perfect
        collinearity when a global intercept is present in the model.
        Internally computes (n_bases + 1) columns and slices off the first.

    Returns
    -------
    B : ndarray, shape (n_samples, n_bases)
        B-spline design matrix.
    knots : ndarray
        Knot vector used.
    """
    if knots is not None:
        actual_bases = len(knots) - 2 * (degree - 1)
    else:
        actual_bases = n_bases if include_intercept else n_bases + 1

    x_c, knots = _prepare_bspline_inputs(
        x=x, n_bases=actual_bases, degree=degree, knots=knots,
        include_intercept=include_intercept,
        lower_bound=lower_bound, upper_bound=upper_bound)
    B = _dense_kernel(x_c, knots, actual_bases, degree)

    if not include_intercept:
        return B[:, 1:], knots
    return B, knots


def bspline_design_matrix_sparse(x, n_bases=10, degree=3, knots=None, include_intercept=True, lower_bound=None,
                                 upper_bound=None):
    """
    Construct a sparse B-spline design matrix as a scipy CSC matrix.

    Sparse-first: the CSC structure (indptr, indices, data) is built directly
    from the local support of B-splines — no dense matrix is ever allocated.
    Each row has exactly (degree+1) nonzeros, so the full matrix has
    n * (degree+1) nonzero entries regardless of n_bases.

    Runtime is O(n * degree²), independent of n_bases.

    Parameters
    ----------
    x : array-like, shape (n_samples,)
        Evaluation points.
    n_bases : int
        Number of B-spline basis functions (columns).
    degree : int
        Spline degree (3 = cubic, standard for GAMs).
    knots : array-like or None
        Full knot vector. If None, built with quantile-spaced interior knots.
        Pass the returned knots back in across repeated calls to skip the
        quantile computation (~0.6 ms for n=10000).
    include_intercept : bool, default=True
        If False, drops the very first basis function to prevent perfect
        collinearity when a global intercept is present in the model.
        The matrix is efficiently sliced after CSC construction.

    Returns
    -------
    B : scipy.sparse.csc_matrix, shape (n_samples, n_bases)
        Sparse B-spline design matrix.
    knots : ndarray
        Knot vector used.

    Notes
    -----
    CSC layout is constructed in two Numba-compiled passes:

      Pass 1 (_spans_and_colcounts): binary-search each x[s] to find its
        knot span k[s], and accumulate a column-count array in the same loop.
        Column j is nonzero for row s whenever k[s] in [j, j+degree], so

            col_counts[j] = #{s : k[s]-degree <= j <= k[s]}

        indptr follows by cumulative sum.

      Pass 2 (_fill_csc): de Boor triangular scheme on a (degree+1) scratch
        buffer per sample, then scatter into pre-allocated indices/data arrays
        using a per-column cursor derived from indptr. Because s increases
        monotonically, rows within each column are written in sorted order —
        the CSC invariant is satisfied without any sort.
    """
    actual_bases = n_bases if include_intercept else n_bases + 1

    x_c, knots = _prepare_bspline_inputs(x=x, n_bases=actual_bases, degree=degree, knots=knots,
                                         include_intercept=include_intercept,
                                         lower_bound=lower_bound, upper_bound=upper_bound)

    # Pass 1: knot spans + column entry counts
    spans, col_counts = _spans_and_colcounts(x_c, knots, actual_bases, degree)

    # Build CSC column pointers
    indptr = np.empty(actual_bases + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(col_counts, out=indptr[1:])

    # Pass 2: fill indices and data directly into CSC layout
    indices, data = _fill_csc(x_c, knots, spans, indptr, actual_bases, degree)

    B = csc_matrix((data, indices, indptr), shape=(len(x_c), actual_bases))

    if not include_intercept:
        return B[:, 1:], knots
    return B, knots


def _prepare_bspline_inputs(x, n_bases, degree, knots, include_intercept: bool,
                            lower_bound=None, upper_bound=None):
    """
    Validate spline inputs and prepare evaluation points.

    Parameters
    ----------
    x : array-like
        Evaluation locations.
    n_bases : int
        Number of basis functions.
    degree : int
        Spline degree.
    knots : ndarray or None
        Knot vector.

    Returns
    -------
    x_c : ndarray
        Evaluation points clipped into the valid knot range.
    knots : ndarray
        Knot vector used.
    """
    x = np.asarray(x, dtype=np.float64)
    if knots is None:
        knots = _make_knots(x, n_bases, degree, lower_bound, upper_bound)

    knots = np.asarray(knots, dtype=np.float64)

    # expected = len(knots) - 2*(degree-1) + include_intercept
    # if expected != n_bases:
    #     raise ValueError(f"n_bases={n_bases}, but knot vector implies {expected} basis functions")

    x_c = np.clip(x, knots[degree], knots[-degree - 1])
    return x_c, knots


def _make_knots(x, n_bases, degree, lower_bound=None, upper_bound=None):
    """
    Construct an open clamped knot vector.

    Interior knots are placed at equally spaced empirical quantiles.
    Boundary knots are repeated (degree + 1) times.

    Parameters
    ----------
    x : ndarray
        Sample locations.
    n_bases : int
        Number of basis functions.
    degree : int
        Spline degree.

    Returns
    -------
    ndarray
        Full knot vector.
    """
    n_internal = n_bases - degree - 1
    interior = np.nanquantile(x, np.linspace(0, 1, n_internal + 2))[1:-1] if n_internal > 0 else np.array([])

    xmin = x.min() - 1e-6 if lower_bound is None else lower_bound
    xmax = x.max() + 1e-6 if upper_bound is None else upper_bound
    return np.concatenate([np.repeat(xmin, degree + 1), interior, np.repeat(xmax, degree + 1)])


@njit(cache=True, inline="always")
def _find_span(xs, knots, degree):
    """
    Find the knot span containing xs.

    Returns k such that

        knots[k] <= xs < knots[k + 1]

    using binary search.
    """
    lo = degree
    hi = len(knots) - degree - 2

    while lo < hi:
        mid = (lo + hi + 1) // 2
        if knots[mid] <= xs:
            lo = mid
        else:
            hi = mid - 1
    return lo


@njit(cache=True, inline="always")
def _evaluate_local_basis(xs, k, knots, degree, d, d_new):
    """
    Evaluate the locally-supported B-spline basis functions at a point.

    Uses the de Boor triangular recurrence on a (degree+1)-function
    local support window.

    For a point lying in knot span k,

        knots[k] <= xs < knots[k+1]

    only the basis functions

        B_{k-degree,p},
        ...,
        B_{k,p}

    can be nonzero.

    The recurrence is evaluated on a scratch buffer d[] where, at
    recursion level p,

        d[j] = B_{k-degree+j,p}(xs)

    for

        j = degree-p, ..., degree.

    A second buffer d_new[] is used to construct the next level of
    the de Boor triangle.

    On return,

        d[0:degree+1]

    contains the (degree+1) nonzero basis values associated with xs.
    """
    for j in range(degree + 2):
        d[j] = 0.0
    d[degree] = 1.0

    for p in range(1, degree + 1):
        for j in range(degree + 2):
            d_new[j] = 0.0

        for j in range(degree - p, degree + 1):
            i = k - degree + j
            ld = knots[i + p] - knots[i]
            rd = knots[i + p + 1] - knots[i + 1]

            left = ((xs - knots[i]) / ld) * d[j] if ld > 0 else 0.0
            right = ((knots[i + p + 1] - xs) / rd) * d[j + 1] if rd > 0 else 0.0
            d_new[j] = left + right

        for j in range(degree + 2):
            B_val = d_new[j]
            d[j] = B_val if B_val > 0.0 else 0.0


@njit(cache=True)
def _dense_kernel(x, knots, n_bases, degree):
    """
    Construct a dense B-spline design matrix using local support.

    For each sample x[s]:

      1. Binary search to find knot span k.
      2. Run the de Boor recurrence on a (degree+1) scratch buffer.
      3. Scatter the (degree+1) nonzero basis values into the output row.

    Total work per sample is O(degree²), independent of n_bases.
    """
    n = len(x)
    B = np.zeros((n, n_bases))

    d = np.zeros(degree + 2)
    d_new = np.zeros(degree + 2)

    for s in range(n):
        xs = x[s]
        k = _find_span(xs, knots, degree)
        _evaluate_local_basis(xs, k, knots, degree, d, d_new)

        for j in range(degree + 1):
            col = k - degree + j
            if 0 <= col < n_bases:
                B[s, col] = d[j]
    return B


@njit(cache=True)
def _spans_and_colcounts(x, knots, n_bases, degree):
    """
    Single pass over x:

      - Binary search for each sample's knot span k[s].
      - Accumulate col_counts[j] += 1 for each of the (degree+1) columns
        that x[s] contributes to.

    Specifically,

        col_counts[j] = #{s : k[s]-degree <= j <= k[s]}

    Returns
    -------
    spans : ndarray[int32]
        Knot span index for each observation.
    col_counts : ndarray[int32]
        Number of entries that will appear in each CSC column.
    """
    n = len(x)
    spans = np.empty(n, dtype=np.int32)
    col_counts = np.zeros(n_bases, dtype=np.int32)

    for s in range(n):
        k = _find_span(x[s], knots, degree)
        spans[s] = k
        for j in range(degree + 1):
            col = k - degree + j
            if 0 <= col < n_bases:
                col_counts[col] += 1
    return spans, col_counts


@njit(cache=True)
def _fill_csc(x, knots, spans, indptr, n_bases, degree):
    """
    Fill CSC indices and data without any intermediate dense matrix.

    Uses a cursor array (copy of indptr) to track the next free slot in
    each column's segment. Iterating s = 0..n-1 ensures rows are written
    in ascending order within each column — the CSC sorted-indices invariant
    is satisfied for free.

    The de Boor triangular scheme runs on a (degree+1) scratch buffer d[]:

        d[j] = B_{k-degree+j,p}(x[s])

    for

        j = degree-p .. degree

    at recursion level p.

    A +1 sentinel avoids an out-of-bounds check on d[j+1] at j=degree.
    """
    n = len(x)
    nnz = len(spans) * (degree + 1)

    indices = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.float64)
    cursor = indptr.copy()

    d = np.zeros(degree + 2)
    d_new = np.zeros(degree + 2)

    for s in range(n):
        xs = x[s]
        k = spans[s]
        _evaluate_local_basis(xs, k, knots, degree, d, d_new)

        for j in range(degree + 1):
            col = k - degree + j
            if 0 <= col < n_bases:
                pos = cursor[col]
                indices[pos] = s
                data[pos] = d[j]
                cursor[col] = pos + 1
    return indices, data


def bspline_design_matrix(x, n_bases=10, degree=3, knots=None, return_dense=True, include_intercept=True,
                          lower_bound=None, upper_bound=None):
    """
    Construct a B-spline design matrix exploiting local support.

    Each B-spline basis function is nonzero over at most (degree+1) consecutive
    knot spans. For any x[s], only (degree+1) bases are active. The inner kernel
    works with a (degree+1) scratch buffer per point rather than the full
    (n, n_bases) matrix, then scatters results at the end.

    Combined with Numba JIT, this makes runtime essentially independent of
    n_bases — O(n * degree²) rather than O(n * n_bases).

    Parameters
    ----------
    x : array-like
        Evaluation points.
    n_bases : int
        Number of basis functions.
    degree : int
        Spline degree.
    knots : ndarray or None
        Optional knot vector.
    return_dense : bool, default=True
        If True return a dense ndarray.
        Otherwise return a scipy CSC matrix.
    include_intercept : bool, default=True
        If False, the first basis function column is dropped to allow
        the inclusion of a global intercept in external models without
        inducing perfect multicollinearity.

    Returns
    -------
    B : ndarray or scipy.sparse.csc_matrix
        Design matrix.
    knots : ndarray
        Knot vector used.
    """
    if return_dense:
        return bspline_design_matrix_dense(x, n_bases, degree, knots, include_intercept, lower_bound, upper_bound)
    return bspline_design_matrix_sparse(x, n_bases, degree, knots, include_intercept, lower_bound, upper_bound)


def _derivative_operator(knots, degree):
    """
    Construct the exact B-spline derivative operator.

    Maps coefficients of degree-p B-splines to coefficients of
    degree-(p-1) B-splines.

    If

        f(x) = sum_i c_i B_{i,p}(x)

    then

        f'(x) = sum_j d_j B_{j,p-1}(x)

    with

        d = D @ c

    Parameters
    ----------
    knots : ndarray
        Full knot vector (including repeated boundary knots).
    degree : int
        Degree p of the source B-spline basis.

    Returns
    -------
    D : csr_matrix
        Sparse derivative operator.
    """
    knots = np.asarray(knots, dtype=float)

    p = degree

    n_p = len(knots) - p - 1
    n_pm1 = len(knots) - (p - 1) - 1

    D = lil_matrix((n_pm1, n_p))
    D = np.zeros((n_pm1, n_p))

    for i in range(n_p):

        left_den = knots[i + p] - knots[i]
        if left_den > 0:
            D[i, i] += p / left_den

        right_den = knots[i + p + 1] - knots[i + 1]
        if right_den > 0:
            D[i + 1, i] -= p / right_den

    return D


def _linear_bspline_gram(knots):
    """
    Exact Gram matrix of degree-1 B-splines.

    Computes

        G_ij = ∫ N_i(x) N_j(x) dx

    where N_i are degree-1 B-splines.

    For each knot span [t_k, t_{k+1}]:

        h/6 * [[2,1],
               [1,2]]

    is added to the two active hat functions.
    """
    knots = np.asarray(knots, dtype=float)

    n_lin = len(knots) - 2

    G = lil_matrix((n_lin, n_lin))

    for k in range(1, len(knots) - 2):

        h = knots[k + 1] - knots[k]

        if h <= 0:
            continue

        local = (h / 6.0) * np.array(
            [[2.0, 1.0],
             [1.0, 2.0]]
        )

        G[k - 1:k + 1, k - 1:k + 1] += local

    return G.tocsr()


def bspline_penalty(knots, degree=3, include_intercept=False):
    """
    Exact cubic B-spline roughness penalty

        S_ij = ∫ B_i''(x) B_j''(x) dx

    for an open-clamped knot vector.

    Parameters
    ----------
    knots : ndarray
        Full knot vector including repeated endpoint knots.
    degree : int, default=3
        Spline degree.

    Returns
    -------
    S : csr_matrix
        Roughness penalty matrix.

    Notes
    -----
    Basis dimension is

        n_basis = len(knots) - degree - 1

    matching SciPy BSpline and Patsy.
    """

    if degree != 3:
        raise NotImplementedError(
            "This implementation currently supports cubic splines only."
        )

    knots = np.asarray(knots, dtype=float)

    # Cubic -> quadratic derivative space
    D1 = _derivative_operator(knots, 3)

    # Quadratic -> linear derivative space
    D2a = _derivative_operator(knots, 2)

    # Second derivative operator
    D2 = D2a @ D1

    # Gram matrix of degree-1 splines
    G = _linear_bspline_gram(knots)

    # Roughness penalty
    S = D2.T @ G @ D2

    if include_intercept:
        return S
    else:
        return S[1 - include_intercept:, 1 - include_intercept:]

# if __name__ == '__main__':
#
#     import pandas as pd
#     from kanly.api import lm
#     from statsmodels.formula.api import ols
#
#     np.random.seed(0)
#     deg = 3
#     n_bases = 10
#     T = 30
#     n = 100_000
#     z = np.random.rand(n)
#     x = np.array(sorted(np.random.randn(n)))
#     y = .4 * z + x * np.sin(x * 2) + x + np.random.randn(n) * .2 - 20
#
#     df = pd.DataFrame(dict(x=x, y=y, z=z))
#     f = lm('y ~ bs(x,3,df=10,include_intercept=False) + z', df, debug=False)
#     print(f)
#
#     f2 = ols('y ~ bs(x,degree=3,df=10,include_intercept=False) + z', df).fit()
#     print(f2.summary())
#
#     import matplotlib.pyplot as plt
#     plt.scatter(x, y, alpha=.5)
#     plt.plot(x, f.fittedvalues, c='orange')
#     plt.plot(x, f2.fittedvalues, c='r')
#     plt.show()
