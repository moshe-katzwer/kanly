from __future__ import absolute_import, print_function

import math

import numpy as np
from scipy.linalg import qr as qr_scipy
from scipy.sparse import csc_matrix
from scipy.sparse import hstack as sphstack

from kanly.formula.sparse_formula_data_object import SparseFormulaDataObj


def generate_trend_matrix(trend_values, n, term_var_name=None, name=None, return_array=True, return_dense=False):
    trend_values = sorted(set(trend_values))
    p = len(trend_values)
    vals = np.zeros((n, p))
    t = np.arange(n)
    for i, e in enumerate(trend_values):
        vals[:, i] = t ** e
    column_names = [f'trend' if e == 1 else ('Intercept' if e == 0 else f'trend[^{e}]')
                    for e in trend_values]
    if not return_dense:
        vals = csc_matrix(vals)
    if return_array:
        return vals, column_names
    else:
        return SparseFormulaDataObj(
            vals,
            column_names=column_names, null_rows=set(),
            var_2_col_indices={term_var_name: range(p)}, name=name
        )


def generate_seasonal_matrix(seasonal_periods, n, term_var_name=None, name=None, return_array=True, return_dense=False,
                             drop1=False):
    if term_var_name is None:
        term_var_name = f'seasonal({seasonal_periods})'
    if name is None:
        name = f'seasonal({seasonal_periods})'

    time_index = np.arange(n)
    temps = []
    column_names = []

    seasonal_periods = sorted(set(seasonal_periods))
    # Phase 1: drop any period whose span is fully contained in a larger period's span
    seasonal_periods = [
        p for p in seasonal_periods
        if not any(q % p == 0 and q != p for q in seasonal_periods)
    ]

    for ip, p in enumerate(seasonal_periods):
        values = time_index % p
        idx = values >= (ip > 0 - drop1)
        temp = csc_matrix(
            (np.ones(n)[idx], (time_index[idx], values[idx] - (ip > 0 - drop1))),
            shape=(n, p - (ip > 0 - drop1)),
        )
        temps.append(temp)
        column_names += [f'period[{j + 1}/{p}]' for j in range(ip > 0 - drop1, p)]
    values = sphstack(temps).tocsc()

    # check ranks
    deficient_rank = False
    for p1 in seasonal_periods:
        for p2 in seasonal_periods:
            if p1 != p2 and math.gcd(p1, p2) > 1:
                deficient_rank = True
                break

    if deficient_rank:
        # 1. Compute the small dense normal matrix
        # Even if X is huge, A will be a small N x N dense matrix
        A_dense = (values.T @ values).toarray()

        # 2. Perform Rank-Revealing QR with pivoting
        # p is an array of indices tracking how columns were permuted
        Q, R, p = qr_scipy(A_dense, pivoting=True)

        # 3. Determine the numerical rank using a standard safe tolerance
        # We check the absolute values of the diagonal elements of R
        diag_R = np.abs(np.diag(R))

        # Standard numerical tolerance based on matrix scale and machine epsilon
        tol = np.max(A_dense.shape) * np.spacing(np.max(diag_R))
        rank = np.sum(diag_R > tol)

        # 4. Map the pivots back to extract your full-rank columns
        # The first 'rank' elements of p are your independent column indices
        independent_column_indices = p[:rank]

        # 5. Extract the full-rank sparse matrix from your original X
        values = values[:, independent_column_indices]
        column_names = list(np.array(column_names)[independent_column_indices])

    if return_dense:
        values = values.toarray()

    if return_array:
        return values, column_names

    else:
        return SparseFormulaDataObj(
            values,
            column_names=column_names, null_rows=set(),
            var_2_col_indices={term_var_name: range(values.shape[1])}, name=name
        )
