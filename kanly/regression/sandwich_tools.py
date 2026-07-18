"""Sandwich covariance 'meat' computations for clustered and HAC standard errors.

The sandwich estimator for the covariance of OLS/GLS estimates has the form
``(X'X)^{-1}  B  (X'X)^{-1}``, where ``B`` (the "meat") captures the
dependence structure of the residuals.  This module provides:

- **Cluster-robust meat** (Liang-Zeger 1986): sums outer products of
  within-cluster score vectors.
- **HAC meat** (Newey-West 1987): sums cross-lag score covariances weighted
  by a kernel (Bartlett or uniform).
- **Normalised covariance** ``(X'X)^{-1}``: optionally augmented with a
  ridge/L2 penalty for use in penalised models.
- Small helpers for kernel functions and sparse row-scaling operations.
"""

from __future__ import absolute_import, print_function

import time

import numpy as np
from scipy.sparse import csr_matrix, diags as spdiags, spmatrix, csc_matrix, vstack as sp_vstack, isspmatrix
from tqdm import tqdm

from kanly.utils.linalg_utils import get_matrix_inverse_internal, csc_matrix_by_column_array_broadcast


class SandwichTools(object):
    """Tools for computing the sandwich covariance meat.

    All methods are static.  The two primary methods are:

    - ``cluster_robust_meat``: clustered sandwich estimator (Liang-Zeger).
    - ``hac_meat``: Newey-West HAC sandwich estimator.

    The helper ``get_normalized_cov_params`` computes ``(X'WX)^{-1}`` and
    optionally applies a ridge penalty, which is the "bread" of the
    sandwich and is used as both the normalised covariance and the scaling
    factor that wraps the meat.
    """

    @staticmethod
    def bartlett(l, maxlags):
        """Bartlett (triangular) kernel weight for lag ``l``.

        Returns the weight ``1 - l / (maxlags + 1)``, which decreases linearly
        from 1 at lag 0 to 0 at lag ``maxlags + 1``.  This ensures positive
        semi-definiteness of the HAC estimator.

        Args:
            l: Lag order (positive integer).
            maxlags: Maximum lag included in the HAC estimator.

        Returns:
            float: Bartlett weight in (0, 1].
        """
        return 1.0 - float(l) / (maxlags + 1)

    @staticmethod
    def uniform(l, max_lags):
        """Uniform (rectangular) kernel weight – returns 1.0 for all lags.

        All lags up to ``max_lags`` receive equal weight.  This is the
        Newey-West variant with no taper; it does not guarantee positive
        semi-definiteness in finite samples.

        Args:
            l: Lag order (unused; present for interface consistency).
            max_lags: Maximum lag (unused; present for interface consistency).

        Returns:
            float: Always 1.0.
        """
        return 1.0

    @staticmethod
    def cluster_robust_meat(exog, resid, grp_row_indices, return_sparse=True, debug=False, _time=None):
        """Compute the meat of the clustered sandwich (Liang-Zeger) estimator.

        Calculates ``B = sum_g (X_g' e_g e_g' X_g)`` where ``X_g`` and
        ``e_g`` are the rows of ``exog`` and the residuals belonging to
        cluster ``g``.  The computation uses a compact sparse representation:
        a residual matrix ``E`` of shape (n_groups × n_obs) is built so that
        ``B = (E @ X)' @ (E @ X)`` in a single sparse–sparse product.

        Args:
            exog: Design matrix (n_obs × n_cols), dense or sparse; converted
                to CSC internally.
            resid: Residual vector or matrix of length/shape n_obs.
            grp_row_indices: 1-D integer array of length n_obs where entry
                ``i`` gives the cluster index of observation ``i``.
            return_sparse: If ``True`` (default) return the meat as a sparse
                matrix; otherwise convert to dense.
            debug: If ``True``, print elapsed-time diagnostics.
            _time: Optional start time for logging; defaults to
                ``time.time()``.

        Returns:
            Sparse or dense (n_cols × n_cols) matrix ``B``.
        """

        if _time is None:
            _time = time.time()

        if not isspmatrix(exog):
            exog = csc_matrix(exog)

        if isspmatrix(resid):
            resid = resid.toarray()

        resid = resid.ravel()

        nobs = exog.shape[0]
        n_g = len(grp_row_indices)

        if debug:
            print("\tBuilding residual matrix E {E is %d x %d}..." % (n_g, nobs), end='')

        # Build a sparse (n_groups × n_obs) CSR matrix where row g contains
        # only the residuals of observations belonging to cluster g.  This
        # avoids explicit Python loops over groups; the meat is then a
        # single matrix product.
        E_ = csr_matrix((resid, (grp_row_indices, range(nobs))), shape=(n_g, nobs))

        if debug:
            print("...complete! (%.3f s)" % (time.time() - _time))
            print("\tComputing (exog' * resid)...", end='')

        v = E_.dot(exog).transpose()

        if debug:
            print("...complete! (%.3f s)" % (time.time() - _time))
            print("\tComputing (exog' * resid) * (resid' * exog) {(exog'*resid) is %d x %d}..." % v.shape, end='')

        meat2 = v.dot(v.transpose())

        if debug:
            print("...complete! (%.3f s)" % (time.time() - _time))

        if return_sparse:
            return meat2
        else:
            return meat2.toarray()

    @staticmethod
    def hac_meat(X, e, maxlags, kernel, group_idx=None, return_sparse=True, denom=None, debug=False):
        """Compute the meat of the Newey-West HAC sandwich estimator.

        Accumulates lag-``l`` cross-score covariances weighted by a kernel
        function up to ``maxlags`` lags:

        ``B = Omega_0 + sum_{l=1}^{maxlags} k(l) * (Q_l + Q_l')``

        where ``Omega_0 = (X * e)'(X * e)`` is the outer product at lag 0 and
        ``Q_l = (e[l:] * X[l:])' (e[:-l] * X[:-l])`` captures lag-``l``
        cross-moment dependence.  Panel-corrected HAC is supported by passing
        group labels via ``group_idx``; covariances are only accumulated
        within groups.

        Args:
            X: Design matrix (n_obs × n_cols), dense or sparse.
            e: Residual array of length n_obs.
            maxlags: Maximum lag order to include.
            kernel: Kernel name; one of ``'bartlett'`` or ``'uniform'``.
            group_idx: Optional 1-D array of group labels (length n_obs) for
                panel HAC; ``None`` treats all observations as a single group.
            return_sparse: If ``True`` (default) return a sparse matrix.
            denom: Optional scaling denominator; when supplied residuals are
                divided by ``sqrt(denom)`` before the meat is formed.
            debug: If ``True``, display a ``tqdm`` progress bar over lags.

        Returns:
            Sparse or dense (n_cols × n_cols) HAC meat matrix ``B``.

        Raises:
            Exception: If ``kernel`` is not ``'bartlett'`` or ``'uniform'``,
                or if a group is shorter than the requested lag.
        """

        if group_idx is None:
            group_idx = {0: np.arange(X.shape[0])}
        else:
            group_idx = {g: group_idx == g for g in np.unique(group_idx)}

        if isinstance(X, np.ndarray):
            Xsp = csr_matrix(X)
        elif isinstance(X, spmatrix):
            Xsp = X.tocsr(copy=False)

        if kernel == 'bartlett':
            kernel_func = SandwichTools.bartlett
        elif kernel == 'uniform':
            kernel_func = SandwichTools.uniform
        else:
            raise Exception("kernel must be one of 'bartlett' or 'uniform'!")

        if isinstance(e, spmatrix):
            e = e.toarray().flatten()

        if denom is not None:
            e = e / np.sqrt(denom)

        Xsp_e = csc_matrix_by_column_array_broadcast(Xsp, e)
        Omega0 = Xsp_e.transpose().dot(Xsp_e)

        Omega1 = csc_matrix(([], ([], [])), shape=Omega0.shape)
        for l in tqdm(range(1, maxlags + 1), desc='HAC lags', disable=not debug):

            V_lo = []
            V_hi = []
            for g, ind in group_idx.items():

                if len(ind) < l:
                    raise Exception("Length of group '%g' less than lag %d" % (g, l))

                X_g = Xsp[ind, :]
                e_g = e[ind]

                V_lo.append(spdiags(e_g[l:]).dot(X_g[l:, :]))
                V_hi.append(spdiags(e_g[:-l]).dot(X_g[:-l, :]))

            V_lo = sp_vstack(V_lo)
            V_hi = sp_vstack(V_hi)

            Q = V_lo.transpose().dot(V_hi)

            # Add the symmetrised lag-l covariance: k(l) * (Q + Q'), which
            # ensures the accumulated meat stays symmetric.
            Omega1 += (Q.T + Q).multiply(kernel_func(l, maxlags))

        meat = Omega1 + Omega0

        if return_sparse:
            return meat
        else:
            return meat.toarray()

    @staticmethod
    def row_multiply_col_reduce(b, x):
        """Scale each row of sparse matrix ``b`` by the corresponding element of vector ``x``.

        Performs the operation ``b[i, :] *= x[i]`` for every row ``i`` in-place
        on the CSC data array without materialising a dense intermediate, then
        returns the scaled matrix.  Equivalent to ``diag(x) @ b`` but operates
        directly on the stored non-zero values.

        Args:
            b: Sparse matrix to scale; converted to CSC format if necessary.
            x: 1-D array of row scale factors (length == number of rows in
               ``b``).

        Returns:
            CSC sparse matrix with rows of ``b`` scaled by ``x``.
        """
        if not isinstance(b, csc_matrix):
            b = csc_matrix(b)
        if isspmatrix(x):
            x = x.toarray()
        x = np.array(x).flatten()

        data = b.data * np.take(x, b.indices)
        return csc_matrix((data, b.indices, b.indptr))

    @staticmethod
    def get_normalized_cov_params(wexog_instrumented, invert_XpX=True, _time=None, ridge_kwds=None, debug=False,
                                  var_2_col_indices_exog=None):
        """Compute the normalised covariance matrix ``(X'WX)^{-1}`` (the sandwich bread).

        Forms the ``p × p`` matrix ``XpX = wexog_instrumented.T @ wexog_instrumented``
        (where ``wexog_instrumented`` is already pre-multiplied by
        ``sqrt(weights)``), optionally augments it with a ridge penalty, and
        returns its inverse.

        Ridge penalty construction (when ``ridge_kwds`` is not ``None``):

        - If ``normalize=True`` and ``fit_intercept=True``: the penalty is
          proportional to the sample variance of each column
          (``alpha * n * Var(X_j)``), so that penalisation is scale-invariant.
        - Otherwise: a uniform isotropic penalty ``alpha * I`` is applied.
        - In both cases the intercept column (identified by near-zero variance)
          is never penalised.
        - Terms listed in ``ridge_kwds['unpenalized_terms']`` have their
          diagonal penalty entries zeroed out.

        Args:
            wexog_instrumented: Pre-weighted design (or IV) matrix of shape
                (n_obs × p); converted to CSC internally.
            invert_XpX: If ``True`` (default) invert ``XpX`` and return the
                result.  If ``False`` return ``None`` immediately (used when
                the caller does not need the inverse).
            _time: Optional start time for elapsed-time logging.
            ridge_kwds: Optional dict controlling ridge augmentation.  Keys:
                ``'alpha'`` (float, penalty strength),
                ``'fit_intercept'`` (bool),
                ``'normalize'`` (bool),
                ``'sum_weights'`` (float, total weight sum, used to derive
                per-column second moments for scale-invariant normalisation),
                ``'wexog_instrumented_means'`` (array of column means),
                ``'unpenalized_terms'`` (optional list of term names to exempt
                from the penalty).
            debug: If ``True``, print diagnostic messages.
            var_2_col_indices_exog: Dict mapping term name → list of column
                indices in ``wexog_instrumented``; required when
                ``ridge_kwds['unpenalized_terms']`` is non-empty.

        Returns:
            CSC sparse matrix ``(X'WX + penalty)^{-1}`` of shape (p × p),
            or ``None`` when ``invert_XpX=False``.
        """
        if _time is None:
            _time = time.time()

        if invert_XpX:
            if isinstance(wexog_instrumented, np.ndarray):
                wexog_instrumented = csc_matrix(wexog_instrumented)
            if debug:
                print("{exog is (%d * %d) with %d non-zero elements (%.2f%%}"
                      % (wexog_instrumented.shape[0], wexog_instrumented.shape[1], wexog_instrumented.nnz,
                         100. * wexog_instrumented.nnz / (wexog_instrumented.shape[0] * wexog_instrumented.shape[1])))
                print("Forming (exog'exog)...", end='')
            XpX = wexog_instrumented.transpose().dot(wexog_instrumented).toarray()

            if debug:
                print("%.3f s" % (time.time() - _time))

            if ridge_kwds is not None:

                if debug:
                    print("Computing ridge penalty term...")

                # The diagonal of XpX = (sqrt(w)*X)' (sqrt(w)*X) gives for
                # column j: sum_i w_i * x_{ij}^2, which is the weighted
                # second raw moment.  Dividing by sum_weights gives the MLE
                # estimate E[w * x^2].  Subtracting the squared weighted mean
                # yields the weighted sample variance, used to scale the ridge
                # penalty so it is invariant to the units of each regressor.

                # in either case, we need to find intercepts
                # we identify them from zero-variance columns
                if ridge_kwds['fit_intercept'] or ridge_kwds['normalize']:
                    if debug:
                        print("\tFinding intercept-like columns...")

                    diag_XpX = np.array(XpX.diagonal(0)).flatten()
                    diag_XpX /= ridge_kwds['sum_weights']
                    var_X = np.clip(diag_XpX - ridge_kwds['wexog_instrumented_means'] ** 2, a_min=0, a_max=np.inf)

                    intercept_like = var_X < 1e-8
                    if intercept_like.sum() == 0:
                        raise Exception("Ridge kwds specified `fit_intercept=True`"
                                        " but could not find an intercept-like column!")

                if ridge_kwds['normalize'] and ridge_kwds['fit_intercept']:
                    penalty = np.diag((ridge_kwds['alpha'] * wexog_instrumented.shape[0]) * var_X)

                else:
                    penalty = np.eye(XpX.shape[0]) * ridge_kwds['alpha']

                if ridge_kwds['normalize'] or ridge_kwds['fit_intercept']:
                    penalty[np.nonzero(intercept_like)[0][0]] = 0
                if ridge_kwds.get('unpenalized_terms', None) is not None:
                    if var_2_col_indices_exog is None:
                        raise Exception("Must supply `var_2_col_indices_exog`")
                    for t in ridge_kwds['unpenalized_terms']:
                        for i in var_2_col_indices_exog[t]:
                            penalty[i, i] = 0.0

                XpX += penalty
                if debug:
                    print("%.3f s" % (time.time() - _time))

            if debug:
                print("Inverting (exog'exog)...", end='')
            normalized_cov_params = csc_matrix(get_matrix_inverse_internal(XpX))
            if debug:
                print("%.3f s" % (time.time() - _time))

            return normalized_cov_params

        else:
            return None
