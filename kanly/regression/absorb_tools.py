"""Fixed-effects absorption utilities for within-group de-meaning.

When a linear model includes one-way (or higher-way) fixed effects that span
many levels, it is numerically cheaper to "absorb" them by de-meaning the
outcome and regressors within each group before estimating the remaining
coefficients.  This module implements that projection step:

1. ``AbsorbInfo`` – value object that packages all de-meaned arrays and
   summary statistics produced by the absorption procedure.
2. ``AbsorbTools2`` – static-method class that performs the three steps of FE
   absorption: group-to-row mapping, weighted FE design matrix construction,
   and the projection (de-meaning) itself.
"""

from __future__ import absolute_import, print_function

import time

import numpy as np
from scipy.sparse import diags as spdiags, isspmatrix

from kanly.utils.linalg_utils import csc_matrix_by_column_array_broadcast, get_wtd_sum_squares


class AbsorbInfo(object):
    """Value object holding all outputs of the fixed-effects absorption step.

    After calling ``AbsorbTools2.do_absorb``, all de-meaned matrices,
    group-mean matrices, and summary statistics are stored in one place so
    they can be passed down to the model-fitting stage as a unit.

    Attributes:
        group_to_row_lists: Dict mapping each FE group index to the row
            indices (or boolean mask) of observations belonging to that group.
        endog_absorbed: De-meaned response matrix (obs × outcomes).
        endog_absorb_means: Group-mean matrix for the response
            (groups × outcomes).
        exog_absorbed: De-meaned regressor matrix (obs × columns).
        exog_absorb_means: Group-mean matrix for the regressors
            (groups × columns).
        instruments_absorbed: De-meaned instrument matrix, or ``None`` when
            no instruments are present.
        instruments_absorb_means: Group-mean matrix for the instruments,
            or ``None``.
        num_absorbed: Number of FE group indicators actually absorbed
            (equals the number of non-empty levels after empty-level pruning).
        rsquared_between: R-squared of regressing the raw response on the FE
            indicators, measuring how much variation is explained by the
            group structure.
    """

    def __init__(self, group_to_row_lists, endog_absorbed, endog_absorb_means, exog_absorbed, exog_absorb_means,
                 instruments_absorbed, instruments_absorb_means, num_absorbed, rsquared_between):
        """Store all absorption results.

        Args:
            group_to_row_lists: Dict mapping FE group index → row indices.
            endog_absorbed: De-meaned response array.
            endog_absorb_means: Per-group means of the response.
            exog_absorbed: De-meaned regressor array.
            exog_absorb_means: Per-group means of the regressors.
            instruments_absorbed: De-meaned instruments, or ``None``.
            instruments_absorb_means: Per-group means of the instruments,
                or ``None``.
            num_absorbed: Count of non-empty FE levels absorbed.
            rsquared_between: Between-group R-squared for the response.
        """
        self.group_to_row_lists = group_to_row_lists
        self.endog_absorbed = endog_absorbed
        self.endog_absorb_means = endog_absorb_means
        self.exog_absorbed = exog_absorbed
        self.exog_absorb_means = exog_absorb_means
        self.instruments_absorbed = instruments_absorbed
        self.instruments_absorb_means = instruments_absorb_means
        self.num_absorbed = num_absorbed
        self.rsquared_between = rsquared_between

    def get_absorb_info_column_i(self, i):
        """Return an ``AbsorbInfo`` slice for a single outcome column.

        For multi-outcome models the response matrix has more than one column.
        This method extracts column ``i`` from the response and its group
        means, reusing the shared exog/instrument fields.

        Args:
            i: Zero-based outcome column index.  Must satisfy
               ``0 <= i <= endog_absorbed.shape[1] - 1``.

        Returns:
            A new ``AbsorbInfo`` instance referencing outcome column ``i``,
            or ``self`` when the response is already single-column and
            ``i == 0``.

        Raises:
            Exception: If ``i`` is out of range.
        """
        if i < 0 or i > self.endog_absorbed.shape[1] - 1:
            raise Exception(f'`i` must be in [0, {self.endog_absorbed.shape[1] - 1}')
        elif i == 0 and self.endog_absorbed.shape[1] == 1:
            return self
        return AbsorbInfo(self.group_to_row_lists, self.endog_absorbed.getcol(i), self.endog_absorb_means.getcol(i),
                          self.exog_absorbed, self.exog_absorb_means, self.instruments_absorbed,
                          self.instruments_absorb_means, self.num_absorbed, self.rsquared_between[i])


class AbsorbTools2:
    """Utilities for absorbing fixed effects via within-group de-meaning.

    All methods are static.  The typical call sequence is:

    1. ``get_absorb_mappings2`` – map each FE level to the row indices that
       belong to it.
    2. ``get_absorb_design_mat_and_ncv2`` – build the weighted FE design matrix
       and its diagonal normalised covariance (reciprocal column sums).
    3. ``absorb_fixed_effect2`` – project any array onto the within-group
       subspace by subtracting per-group means.

    These three steps are orchestrated by the convenience method ``do_absorb``.
    """

    @staticmethod
    def do_absorb(absorb_fe_mat, endog, exog, weights=None, instruments=None, debug=False, _time=None):
        """Absorb a one-way fixed effect from the response, regressors, and instruments.

        The fixed effect is represented by a sparse indicator matrix
        ``absorb_fe_mat`` (obs × groups).  The procedure de-means each of
        ``endog``, ``exog``, and ``instruments`` within every FE group,
        returning the residuals along with group-mean matrices and summary
        statistics.

        Args:
            absorb_fe_mat: Sparse indicator matrix of shape (n_obs, n_groups)
                where entry [i, g] == 1 iff observation i belongs to group g.
            endog: Response array of shape (n_obs, n_outcomes).
            exog: Regressor matrix of shape (n_obs, n_columns).
            weights: Optional 1-D weight array of length n_obs.  When
                supplied, group means are weighted averages.
            instruments: Optional instrument matrix of shape (n_obs, n_inst);
                may be ``None`` for non-IV models.
            debug: If ``True``, prints elapsed-time diagnostics at each step.
            _time: Optional start time for elapsed-time logging; a fresh
                ``time.time()`` is used when ``None``.

        Returns:
            AbsorbInfo: Container with de-meaned arrays, group-mean matrices,
            count of absorbed levels, and between-group R-squared.
        """
        if _time is None:
            _time = time.time()

        if debug:
            print("\tStoring mapping of absorb groups to rows... ", end='')
        group_to_row_lists = AbsorbTools2.get_absorb_mappings2(absorb_fe_mat)
        if debug:
            print("%.3f s" % (time.time() - _time))

        if debug:
            print("\tWeighting absorb matrix and building absorb normalized cov params... ", end='')
        absorb_fe_design_mat_wtd, absorb_fe_norm_cov_params_wtd, num_absorbed \
            = AbsorbTools2.get_absorb_design_mat_and_ncv2(absorb_fe_mat, weights=weights)
        if debug:
            print("%.3f s" % (time.time() - _time))

        if debug:
            print("\tAbsorbing FE for endog... ", end='')
        y_absorb, y_absorb_means = AbsorbTools2.absorb_fixed_effect2(
            endog, absorb_fe_design_mat_wtd, absorb_fe_norm_cov_params_wtd, absorb_fe_mat)

        sst = get_wtd_sum_squares(endog, weights, demean=True)
        ssr = get_wtd_sum_squares(y_absorb, weights, demean=False)
        rsquared_between = 1.0 - ssr / sst

        if debug:
            print("%.3f s" % (time.time() - _time))

        if debug:
            print("\tAbsorbing FE for exog... ", end='')
        X_absorb, X_absorb_means = AbsorbTools2.absorb_fixed_effect2(
            exog, absorb_fe_design_mat_wtd, absorb_fe_norm_cov_params_wtd, absorb_fe_mat)
        if debug:
            print("%.3f s" % (time.time() - _time))
            
        if debug and instruments is not None:
            print("\tAbsorbing FE for instruments... ", end='')
        Z_absorb, Z_absorb_means = AbsorbTools2.absorb_fixed_effect2(
            instruments, absorb_fe_design_mat_wtd, absorb_fe_norm_cov_params_wtd, absorb_fe_mat)
        if debug and instruments is not None:
            print("%.3f s" % (time.time() - _time))

        return AbsorbInfo(group_to_row_lists, y_absorb, y_absorb_means, X_absorb, X_absorb_means,
                          Z_absorb, Z_absorb_means, num_absorbed, rsquared_between)

    @staticmethod
    def get_absorb_design_mat_and_ncv2(absorb_fe_mat, weights=None):
        """Build the (optionally weighted) FE design matrix and its diagonal normaliser.

        Scales each column of the FE indicator matrix by observation weights
        (if supplied), prunes empty FE levels, and computes the diagonal
        sparse matrix whose j-th entry is 1 / (column sum of column j), i.e.
        the reciprocal of the effective group size.  This normaliser is used
        in the projection step to compute per-group weighted means.

        Args:
            absorb_fe_mat: Sparse FE indicator matrix (n_obs × n_groups).
            weights: Optional 1-D weight array of length n_obs.

        Returns:
            Tuple of:
              - absorb_fe_design_mat: Weighted (or plain) FE indicator matrix
                after empty-column removal.
              - absorb_fe_norm_cov_params: Diagonal sparse matrix of
                reciprocal column sums (shape: n_groups × n_groups).
              - int: Number of non-empty FE levels retained.
        """
        if weights is None:
            absorb_fe_design_mat = absorb_fe_mat
        else:
            absorb_fe_design_mat = csc_matrix_by_column_array_broadcast(absorb_fe_mat, weights)

        # Drop FE levels that have zero total weight (empty groups after
        # observation weighting), as they would cause a divide-by-zero.
        valid_cols = np.array(absorb_fe_design_mat.sum(axis=0)).flatten() > 0
        if not np.all(valid_cols):
            absorb_fe_design_mat = absorb_fe_design_mat[:, valid_cols]

        absorb_fe_norm_cov_params = spdiags(1.0 / np.asarray(absorb_fe_design_mat.sum(axis=0)).flatten())

        return absorb_fe_design_mat, absorb_fe_norm_cov_params, absorb_fe_design_mat.shape[1]

    @staticmethod
    def get_absorb_mappings2(absorb_mat):
        """Extract the mapping from each FE group to the rows it contains.

        For sparse matrices the CSC index arrays are used directly for O(1)
        column slice extraction; for dense matrices a boolean mask is built
        for each column.

        Args:
            absorb_mat: FE indicator matrix (n_obs × n_groups), either sparse
                (CSC/CSR) or a dense ndarray.

        Returns:
            Dict mapping group index (0-based) to either:
              - a numpy integer array of row indices (sparse path), or
              - a boolean ndarray mask (dense path).
        """
        num_fe = absorb_mat.shape[1]
        if isspmatrix(absorb_mat):
            # Exploit the CSC index structure: indptr[i]:indptr[i+1] gives
            # the positions of non-zeros in column i, which are exactly the
            # rows belonging to FE group i.
            group_to_row_lists_dict = {
                i: absorb_mat.indices[absorb_mat.indptr[i]:absorb_mat.indptr[i + 1]]
                for i in range(num_fe)
            }
        else:
            group_to_row_lists_dict = {
                i: absorb_mat[:, i] == 1 for i in range(num_fe)
            }
        return group_to_row_lists_dict

    @staticmethod
    def absorb_fixed_effect2(matrix, FE_matrix, FE_norm_cov_params, FE_matrix_unwtd):
        """Project ``matrix`` onto the within-group subspace by subtracting per-group means.

        Computes group means as ``D_inv @ (G_wtd.T @ matrix)`` where ``D_inv``
        (``FE_norm_cov_params``) is the diagonal matrix of reciprocal column
        sums and ``G_wtd`` (``FE_matrix``) is the weighted indicator matrix.
        The unweighted indicator ``FE_matrix_unwtd`` is then used to broadcast
        each group mean back to all observations in that group, and the mean
        is subtracted.

        Args:
            matrix: Array to de-mean (n_obs × n_cols), or ``None`` to
                propagate a no-op.
            FE_matrix: Weighted FE indicator matrix (n_obs × n_groups).
            FE_norm_cov_params: Diagonal sparse matrix of reciprocal column
                sums (n_groups × n_groups) from
                ``get_absorb_design_mat_and_ncv2``.
            FE_matrix_unwtd: Unweighted FE indicator matrix (n_obs × n_groups)
                used to broadcast means back to observations.

        Returns:
            Tuple of:
              - within-group-demeaned matrix (same shape as ``matrix``), or
                ``None`` if ``matrix`` is ``None``.
              - group-mean matrix (n_groups × n_cols), or ``None``.
        """
        if matrix is None:
            return None, None

        # Compute per-group means: (D_inv @ G_wtd.T) @ matrix
        # For single-column outcomes this path is identical to the multi-
        # column path; both branches are kept explicit for readability.
        if matrix.shape[1] == 1:
            means = FE_norm_cov_params.dot(FE_matrix.transpose().dot(matrix))
            to_return = matrix - FE_matrix_unwtd.dot(means)
        else:
            means = FE_norm_cov_params.dot(FE_matrix.transpose().dot(matrix))
            g_dot_means = FE_matrix_unwtd.dot(means)
            to_return = matrix - g_dot_means

        return to_return, means
