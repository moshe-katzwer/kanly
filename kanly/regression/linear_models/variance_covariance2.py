"""Covariance estimators for sparse linear models.

This module implements every covariance type supported by
``SparseLinearModel.fit``:

- **NONROBUST / OLS_SMALL**: Classical OLS covariance (X'X)⁻¹ σ̂², with an
  optional n/(n−k) finite-sample correction (``OLS_SMALL``).
- **HC0 – HC3**: Heteroscedasticity-consistent (White) sandwich estimators,
  with leverage corrections for HC2 and HC3.
- **HAC / HAC_PANEL**: Newey-West HAC standard errors with Bartlett kernel;
  ``HAC_PANEL`` groups within panels before accumulating lags.
- **CLUSTER**: One-way cluster-robust covariance (Liang–Zeger).
- **Multi-way cluster**: Cameron–Gelbach–Miller inclusion-exclusion over all
  combinations of cluster dimensions.

The central static method is :meth:`SparseVarianceCovariance2.compute_cov_params`,
which dispatches to the appropriate estimator based on ``cov_type`` and
calls :class:`~kanly.regression.sandwich_tools.SandwichTools` for the
"meat" computation.

Reference:
    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
    Robust inference with multiway clustering.
    *Journal of Business & Economic Statistics*, 29(2), 238–249.
"""

from __future__ import absolute_import, print_function

import time
import warnings
from itertools import combinations

import numpy as np
from pandas import Series
from scipy.sparse import isspmatrix, csc_matrix

from kanly.regression.cov_types import HC0, HC1, HC2, HC3, CLUSTER, NONROBUST, OLS_SMALL, HAC, HAC_PANEL, COV_TYPES
from kanly.regression.sandwich_tools import SandwichTools
from kanly.utils.linalg_utils import csc_matrix_by_column_array_broadcast, sandwich_diagonal


class SparseVarianceCovariance2(object):
    """Collection of static covariance estimators for linear models.

    All methods are static; the class is a namespace used to group the
    covariance computation logic and avoid polluting the module level.
    Callers typically invoke :meth:`compute_cov_params`, which routes to the
    correct estimator and applies any small-sample correction.
    """

    @staticmethod
    def compute_multiway_cov_params(
            cov_type, cov_kwds, use_t, df_resid, wssr, resid, normalized_cov_params,
            is_sure, exog_absorb_instrumented, debug=False, _time=None, groups=None, weights=None,
            param_name='params'):
        """Compute a multi-way cluster-robust covariance matrix.

        Applies the Cameron–Gelbach–Miller inclusion-exclusion identity over
        all combinations of the supplied cluster dimensions.  For two
        dimensions (A, B), this is:

            V_two_way = V(A) + V(B) − V(A×B)

        For k dimensions, the general form alternates signs at each level of
        the power-set expansion (even-order intersections are subtracted,
        odd-order are added).

        The ``df_t_dist`` for t-tests is set to ``min(df_t_dist)`` across all
        single-dimension cluster fits.

        Reference: https://cameron.econ.ucdavis.edu/research/CGM_twoway_ALL_13may2008.pdf

        Args:
            cov_type (str): Covariance type (e.g. ``'CLUSTER'``).
            cov_kwds (dict): Covariance keyword arguments.
            use_t (bool): Use t-distribution (``True``) vs normal.
            df_resid (int): Residual degrees of freedom.
            wssr (float): Weighted SSR.
            resid (ndarray): Residual vector.
            normalized_cov_params: ``(X'WX)^{-1}``.
            is_sure (bool): True if this is a SURE model.
            exog_absorb_instrumented: Final design matrix.
            debug (bool): Verbose output.
            _time (float, optional): Elapsed-time baseline.
            groups (list of array-like): Per-dimension cluster label arrays.
                Length must be ≥ 2 for multi-way clustering.
            weights (array-like, optional): Observation weights.
            param_name (str): Label used in debug output.

        Returns:
            tuple: ``(var_covar, num_groups, df_t_dist, small_samp_correct,
                cov_time)`` — same structure as :meth:`compute_cov_params`.
        """

        # TODO multiway
        _start_cov_time = time.time()

        groups = [np.array(g).astype(str) for g in groups]

        var_covar = None
        num_groups = []
        small_samp_correct = []
        df_t_dist = np.inf
        for subsize in range(1, len(groups) + 1):
            subset = list(combinations(groups, subsize))
            for grp in subset:
                col = ['_'.join(z) for z in zip(*grp)]
                var_covar_grp, num_groups_grp, df_t_dist_grp, small_samp_correct_grp, _ \
                    = SparseVarianceCovariance2.compute_cov_params(
                    cov_type, cov_kwds, use_t, df_resid, wssr, resid, normalized_cov_params,
                    is_sure, exog_absorb_instrumented, debug=debug, _time=_time, groups=col, weights=weights,
                    param_name=param_name)
                # Inclusion-exclusion: even-order intersections are subtracted (−1).
                if subsize % 2 == 0:
                    var_covar_grp.data *= -1.0
                if var_covar is None:
                    var_covar = var_covar_grp
                else:
                    var_covar += var_covar_grp
                num_groups.append(num_groups_grp)
                small_samp_correct.append(small_samp_correct_grp)
                df_t_dist = int(min(df_t_dist, df_t_dist_grp)) if use_t else None

        cov_time = time.time() - _start_cov_time
        return var_covar, num_groups, df_t_dist, small_samp_correct, cov_time


    # TODO wssr=None, compute on fly (wssr can always be derived from resid)
    @staticmethod
    def compute_cov_params(cov_type, cov_kwds, use_t, df_resid, wssr, resid, normalized_cov_params,
                           is_sure, exog_absorb_instrumented, debug=False, _time=None, groups=None, weights=None,
                           param_name='params', sigma=None, sigma_inv=None):
        """Compute the parameter variance-covariance matrix for a given estimator.

        Routes to the appropriate covariance estimator based on ``cov_type``,
        applies small-sample corrections where applicable, and determines the
        degrees of freedom for t-tests.

        Routing logic:
        - If ``np.ndim(groups) > 1``, delegates to :meth:`compute_multiway_cov_params`.
        - ``NONROBUST`` / ``OLS_SMALL``: ``(X'WX)^{-1} * WSSR/n``.
        - ``CLUSTER``: Liang-Zeger sandwich with optional ``(n-1)/(n-k) * G/(G-1)``
          small-sample correction.
        - ``HC0``–``HC3``: White heteroscedasticity-robust sandwich, with
          leverage denominators for HC2/HC3.
        - ``HAC`` / ``HAC_PANEL``: Newey-West sandwich with Bartlett kernel;
          ``HAC_PANEL`` groups time observations by panel ID before accumulating
          lags.

        GLS restriction: only ``NONROBUST`` or ``OLS_SMALL`` is accepted when
        ``sigma`` / ``sigma_inv`` are provided.

        Args:
            cov_type (str): One of the constants in
                :mod:`~kanly.regression.cov_types`.
            cov_kwds (dict): Estimator-specific keyword arguments
                (e.g. ``'maxlags'``, ``'kernel'``, ``'groups'``,
                ``'use_correction'``).
            use_t (bool): Use t-distribution for inference.
            df_resid (int): Residual degrees of freedom.
            wssr (float): Weighted SSR (used for NONROBUST/OLS_SMALL).
            resid (ndarray): Residual vector, length ``n``.
            normalized_cov_params: ``(X'WX)^{-1}`` (the "bread").
            is_sure (bool): True for SURE models.
            exog_absorb_instrumented: Final design matrix (the "meat" source).
            debug (bool): Verbose output.
            _time (float, optional): Elapsed-time baseline.
            groups (array-like or list, optional): Cluster labels (1-D for
                single clustering, 2-D for multi-way).
            weights (array-like, optional): Observation weights.
            param_name (str): Label for debug messages.
            sigma (ndarray, optional): GLS covariance (restricts cov_type).
            sigma_inv (ndarray, optional): Pre-computed GLS inverse.

        Returns:
            tuple of 5 elements:
                - **var_covar**: Variance-covariance matrix (sparse or dense).
                - **num_groups** (int or None): Number of clusters; ``None``
                  for non-cluster estimators.
                - **df_t_dist** (int or None): Degrees of freedom for t-tests.
                - **small_samp_correct** (tuple or None): Small-sample
                  correction numerators/denominators.
                - **cov_time** (float): Wall-clock seconds for this call.

        Raises:
            Exception: If ``cov_type`` is not recognised.
            Exception: If ``groups`` is not supplied for CLUSTER or HAC_PANEL.
            Exception: If GLS is combined with a robust covariance type.
        """

        _start_cov_time = time.time()
        cov_type = cov_type.upper()

        is_gls = sigma is not None or sigma_inv is not None
        if is_gls:
            assert cov_type in (NONROBUST, OLS_SMALL)
            assert sigma is None or sigma_inv is None
            assert weights is None

        if cov_type == 'HAC-PANEL':
            cov_type = HAC_PANEL

        if np.ndim(groups) > 1:
            return SparseVarianceCovariance2.compute_multiway_cov_params(
                cov_type, cov_kwds, use_t, df_resid, wssr, resid, normalized_cov_params,
                is_sure, exog_absorb_instrumented, debug=debug, _time=_time, groups=groups, weights=weights,
                param_name=param_name)

        # cov_type, cov_kwds, use_t, df_resid, wssr, resid, normalized_cov_params,
        # self.is_sure, exog_absorb_instrumented, debug = False, _time = None)

        cov_string = None

        if debug:
            print(f"Computing Variance-Covariance of {param_name}... ", end='')
            if _time is None:
                _time = time.time()

        num_regressors = normalized_cov_params.shape[0]
        nobs = len(resid)
        small_samp_correct = None

        # Fetch cluster group mapping for estimators that need it.
        if cov_type in [CLUSTER, HAC_PANEL]:

            if groups is None:
                raise Exception("Must supply groups for clustered or hac_panel standard errors.")

            group_to_row, num_groups = SparseVarianceCovariance2._get_cluster_group_info2(groups)

            if num_groups < num_regressors:
                warnings.warn(("Number of clusters %d < number of params %d! "
                               "Variance-Covariance will be singular") % (num_groups, num_regressors))

        else:
            group_to_row, num_groups = None, None

        if cov_type in [NONROBUST, OLS_SMALL]:
            if isspmatrix(normalized_cov_params):
                var_covar = normalized_cov_params.multiply(wssr / nobs)
            else:
                var_covar = normalized_cov_params * (wssr / nobs)
            if cov_type == OLS_SMALL:
                small_samp_correct = ((float(nobs),), (df_resid,))

        elif cov_type == CLUSTER:

            if cov_kwds.get('use_correction', True):
                small_samp_correct = ((nobs - 1, num_groups), (df_resid, num_groups - 1))

            if weights is None:
                wexog_instrumented = exog_absorb_instrumented
                wresid = resid
            else:
                if isspmatrix(exog_absorb_instrumented):
                    wexog_instrumented = csc_matrix_by_column_array_broadcast(
                        exog_absorb_instrumented, np.sqrt(weights))
                else:
                    wexog_instrumented = exog_absorb_instrumented * np.sqrt(weights).reshape((-1, 1))
                wresid = resid * np.sqrt(weights)

            meat = SandwichTools.cluster_robust_meat(
                wexog_instrumented, wresid, group_to_row, debug=debug, _time=_time)

            var_covar = SparseVarianceCovariance2._make_var_covar_sandwich(bread=normalized_cov_params, meat=meat)

        elif cov_type in [HC0, HC1, HC2, HC3, HAC, HAC_PANEL]:

            maxlags = 0
            kernel = 'bartlett'  # TODO delete?

            if cov_type in (HAC, HAC_PANEL):
                if debug:
                    warnings.warn(
                        "Note! 'HAC' and 'HAC_PANEL' standard errors assume input data is sorted properly!")
                maxlags = cov_kwds.get('maxlags', int(np.floor(4.0 * (nobs / 100) ** (2. / 9))))
                kernel = cov_kwds.get('kernel', 'bartlett')

            # for HC2 and HC3, 1/h
            # https://jslsoc.sitehost.iu.edu/files_research/testing_tests/hccm/00TAS.pdf
            if weights is None:
                wexog_instrumented = exog_absorb_instrumented
                wresid = resid

            else:
                if isspmatrix(exog_absorb_instrumented):
                    wexog_instrumented = csc_matrix_by_column_array_broadcast(
                        exog_absorb_instrumented, np.sqrt(weights))
                else:
                    wexog_instrumented = exog_absorb_instrumented * np.sqrt(weights).reshape((-1, 1))
                wresid = resid * np.sqrt(weights)


            denom = SparseVarianceCovariance2._get_h(cov_type, wexog_instrumented, normalized_cov_params)

            meat = SandwichTools.hac_meat(
                wexog_instrumented, wresid, denom=denom, maxlags=maxlags, group_idx=group_to_row,
                kernel=kernel, debug=debug)

            var_covar = SparseVarianceCovariance2._make_var_covar_sandwich(bread=normalized_cov_params, meat=meat)

            # HC1 and HAC/HAC_PANEL with use_correction apply the n/(n-k) adjustment.

        else:
            raise Exception(f"`cov_type` must be one of {str(COV_TYPES)}, you gave {cov_type}!")

        if cov_type in HC1 or (cov_type in (HAC, HAC_PANEL) and cov_kwds.get('use_correction', True)):
            small_samp_correct = ((float(nobs),), (df_resid,))

        if small_samp_correct is not None:
            ssc = (np.prod(small_samp_correct[0]) / np.prod(small_samp_correct[1])
                   if np.prod(small_samp_correct[1]) else np.nan)
            if isspmatrix(var_covar):
                var_covar = var_covar.multiply(ssc)
            else:
                var_covar *= ssc

        if debug:
            print("Variance-Covariance complete (%.3f s)" % (time.time() - _time))

        # -------------------------------------
        # Degrees of freedom for t distribution
        df_t_dist = SparseVarianceCovariance2._get_df_t_dist(use_t, cov_type, df_resid, num_groups, cov_kwds)
        cov_time = time.time() - _start_cov_time

        return var_covar, num_groups, df_t_dist, small_samp_correct, cov_time

    @staticmethod
    def _get_h(cov_type, wexog_instrumented, normalized_cov_params):
        """Compute the leverage denominator for HC2 and HC3 estimators.

        For HC2: divide each squared residual by ``(1 − hᵢ)`` where
        ``hᵢ = xᵢ' (X'X)⁻¹ xᵢ`` is the hat-matrix diagonal.
        For HC3: divide by ``(1 − hᵢ)²`` (jackknife approximation).

        Reference: https://jslsoc.sitehost.iu.edu/files_research/testing_tests/hccm/00TAS.pdf

        Args:
            cov_type (str): Must be ``'HC2'`` or ``'HC3'`` to return a
                non-trivial denominator.  All other values return ``None``.
            wexog_instrumented (array-like or sparse): Weighted design matrix.
            normalized_cov_params: ``(X'WX)^{-1}``; used to compute hat values.

        Returns:
            ndarray or None: Per-observation denominator ``(1 − hᵢ)^ν``
                where ν = 1 for HC2 and ν = 2 for HC3; ``None`` otherwise.
        """

        cov_type = cov_type.upper()
        if cov_type in (HC2, HC3):
            h = sandwich_diagonal(wexog_instrumented, normalized_cov_params)
            return (1 - h) ** (2 if cov_type == HC3 else 1)
        else:
            return None

    @staticmethod
    def _get_df_t_dist(use_t, cov_type, df_resid, num_groups, cov_kwds):
        """Determine the degrees of freedom for t-distribution inference.

        For cluster-robust standard errors the usual convention is to use
        ``G − 1`` (number of clusters minus one) rather than ``n − k`` for
        the t-distribution, which provides a conservative correction for
        finite-cluster asymptotics.

        For ``HAC_PANEL`` the behaviour can be overridden via
        ``cov_kwds['test_hac_panel_use_cluster_df']`` to match the
        statsmodels convention.

        Args:
            use_t (bool): Use t-distribution (``True``) vs standard normal.
            cov_type (str): Covariance type string.
            df_resid (int): Residual degrees of freedom.
            num_groups (int or None): Number of clusters; ``None`` for
                non-cluster estimators.
            cov_kwds (dict): Covariance keyword arguments.

        Returns:
            int or None: Degrees of freedom for t-tests; ``None`` if
                ``use_t=False``.
        """
        cov_type = cov_type.upper()
        if use_t:
            # 'test_hac_panel_use_cluster_df' because statsmodels uses cluster df for t-dist when using hac-panel
            if cov_type == CLUSTER or (cov_type == HAC_PANEL and cov_kwds.get('test_hac_panel_use_cluster_df', False)):
                df_t_dist = num_groups - 1
            else:
                df_t_dist = df_resid
        else:
            df_t_dist = None

        return df_t_dist

    @staticmethod
    def _make_var_covar_sandwich(bread, meat):
        """Compute the sandwich variance-covariance matrix B M B.

        Ensures ``bread`` and ``meat`` are in compatible formats (both sparse
        or both dense) before computing ``bread @ meat @ bread``.

        Args:
            bread: Normalised covariance matrix ``(X'WX)^{-1}``, shape
                ``(p, p)``.  May be sparse or dense.
            meat: Robust "meat" matrix, shape ``(p, p)``.  May be sparse or
                dense.

        Returns:
            Sandwich covariance ``B M B``, in the same format as ``bread``.
        """
        if isspmatrix(bread):
            if not isspmatrix(meat):
                meat = csc_matrix(meat)
        else:
            if isspmatrix(meat):
                meat = meat.toarray()
        return bread.dot(meat).dot(bread)

    @staticmethod
    def get_df_resid_model(nobs, num_regressors, num_absorbed, has_implicit_constant, has_intercept, has_const,
                           debug=False):
        """Compute residual and model degrees of freedom.

        Accounts for both explicit regressors and absorbed fixed-effect
        levels (which consume degrees of freedom without appearing as
        columns in the design matrix).

        Args:
            nobs (int): Number of observations.
            num_regressors (int): Number of columns in the (absorbed,
                instrumented) design matrix.
            num_absorbed (int): Number of absorbed fixed-effect levels.
            has_implicit_constant (bool): True if the model has an implicit
                constant (e.g. all-ones column via absorbed FE).
            has_intercept (bool): True if the model has an explicit intercept.
            has_const (bool): True if the model has any constant term
                (used for debug printing).
            debug (bool): Print debug output.

        Returns:
            tuple:
                - **df_resid** (int): Residual DF = n − (k + num_absorbed).
                - **df_model** (int): Model DF = (k + num_absorbed) − 1 if a
                  constant exists, else k + num_absorbed.
        """

        df_resid = nobs - (num_regressors + num_absorbed)
        if debug:
            print(
                "\tComputing df resid: n=%d, num_regressors=%d, num_absorbed=%d," % (
                nobs, num_regressors, num_absorbed))

        df_model = num_regressors + num_absorbed - int(has_implicit_constant or has_intercept)
        if debug:
            print(
                "\tComputing df model: num_regressors=%d, num_absorbed=%d, has_implicit_constant=%s" % (
                    num_regressors, num_absorbed, has_const))

        return df_resid, df_model

    @staticmethod
    def _get_cluster_group_info2(groups):
        """Map cluster labels to integer indices and count unique clusters.

        Converts an arbitrary array of cluster labels (strings, ints, etc.)
        to a compact integer index array suitable for scatter-accumulation in
        the sandwich meat computation.

        Args:
            groups (array-like, length n): Per-observation cluster labels.

        Returns:
            tuple:
                - **group_row_indices** (ndarray, int): Integer index for each
                  observation's cluster, range ``[0, G)``.
                - **num_groups** (int): Total number of distinct clusters G.

        Raises:
            Exception: If fewer than 2 unique clusters are found.
        """

        groups = Series(np.asarray(groups))
        unique = groups.unique()
        num_groups = len(unique)
        grp_map = dict(zip(unique, range(num_groups)))
        group_row_indices = groups.map(grp_map).values

        if num_groups < 2:
            raise Exception("Must have more than 1 group!")

        return group_row_indices, num_groups

    @staticmethod
    def get_cov_string(nobs, cov_type, cov_kwds, cov_groups_name=None):
        """Return a human-readable description of the covariance estimator.

        Used to populate the "Covariance Type" footer in regression result
        summaries.

        Args:
            nobs (int): Number of observations (used for HAC default lag
                formula ``4 * (n/100)^(2/9)``).
            cov_type (str): Covariance type string.
            cov_kwds (dict): Covariance keyword arguments (``'kernel'``,
                ``'maxlags'`` for HAC; ignored for others).
            cov_groups_name (str, optional): Name of the cluster variable;
                ``None`` is displayed as ``<unknown>``.

        Returns:
            str or None: A descriptive string for CLUSTER and HAC types;
                ``None`` for all other types (no extra description needed).
        """
        if cov_type.upper() == CLUSTER:
            return "Variance clustered on %s" % (
                "'%s'" % cov_groups_name if cov_groups_name is not None else "<unknown>")
        elif cov_type in [HAC_PANEL, HAC]:
            ret = (f"HAC std errs with kernel {cov_kwds.get('kernel', 'bartlett')} and maxlags "
                   f"{cov_kwds.get('maxlags', int(np.floor(4.0 * (nobs / 100) ** (2. / 9))))}")
            if cov_type == HAC_PANEL:
                ret += ", with groups %s" % (
                    "'%s'" % cov_groups_name if cov_groups_name is not None else "<unknown>")
            return ret
        return None
