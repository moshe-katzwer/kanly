"""Numerical core of the OLS / WLS / IV / FGLS / absorb estimation pipeline.

This module contains the low-level functions that implement the complete
estimation pipeline executed on every call to ``SparseLinearModel.fit``:

1. **Type normalisation** (``check_types``) — ensures all inputs are
   compatible sparse or dense arrays.
2. **Fixed-effects absorption** (via ``AbsorbTools2``) — demeans endog,
   exog, and instruments within each group.
3. **IV first stage** (via ``iv_first_stage2``) — replaces endogenous
   columns with their instrument projections.
4. **Column scaling / unscaling** — optionally scales the design matrix to
   unit L2 norm to improve numerical stability, then unscales after solving.
5. **Gram-matrix inversion and parameter estimation**
   (``fit_least_squares_model_internal``) — inverts X'WX and solves for β.
6. **Summary statistics** (``get_fit_summary_stats``) — computes fitted
   values, residuals, R², log-likelihood, and degrees of freedom.
7. **FGLS loop** (``fgls_internal``) — wraps the above in an iterative
   re-weighting loop, using a log-squared-residual regression to update
   heteroscedasticity weights at each step.

The central entry points are:

- :func:`lm_internal` — single-shot estimation.
- :func:`fgls_internal` — FGLS wrapper around ``lm_internal``.
- :func:`get_fit_summary_stats` — summary statistics post-estimation.
"""

from __future__ import absolute_import, print_function

import time

import numpy as np
import scipy as sp
from scipy.sparse import csc_matrix, isspmatrix, diags as sp_diags

from kanly.regression.absorb_tools import AbsorbTools2
from kanly.regression.linear_models.constants import \
    (FGLS_TOL_DEFAULT, FGLS_MAX_ITER_DEFAULT, DEFAULT_LM_FORCE_IV_PROJECTION, DEFAULT_LM_SCALE_DESIGN_MATRIX,
     DEFAULT_LM_COMPUTE_EIGENVALUES, DEFAULT_LM_COMPUTE_EIGENVALUES_INSTRUMENTS,
     DEFAULT_LM_COMPUTE_EIGENVALUES_UNDER_MAX_DIM, DEFAULT_LM_INVERSE_METHOD)
from kanly.regression.linear_models.sparse_iv_first_stage2 import iv_first_stage2
from kanly.regression.linear_models.variance_covariance2 import SparseVarianceCovariance2
from kanly.utils.linalg_utils import \
    get_normalized_cov_params, get_matrix_inverse_internal, scale_in_place, unscale_in_place, gram_matrix, \
    check_weights_and_sigma
from kanly.utils.linalg_utils import (get_wtd_sum_squares, DEFAULT_DENSE_THRESHOLD_MB,
                                      csc_matrix_by_column_array_broadcast, flexible_mat_dot_vec,
                                      none_convert_2_sparse)


def fgls_internal(nobs, endog, exog, do_fgls=False, fgls_kwds=dict(), absorb=None, is_endog_regressor=None,
                  weights=None, instruments=None, debug=False, force_iv_projection=DEFAULT_LM_FORCE_IV_PROJECTION,
                  sigma=None, sigma_inv=None,
                  scale_design_matrix=DEFAULT_LM_SCALE_DESIGN_MATRIX,
                  compute_eigenvalues_instruments=DEFAULT_LM_COMPUTE_EIGENVALUES_INSTRUMENTS,
                  dense_threshold_mb=DEFAULT_DENSE_THRESHOLD_MB, inverse_method=DEFAULT_LM_INVERSE_METHOD,
                  compute_eigenvalues=DEFAULT_LM_COMPUTE_EIGENVALUES, ridge_parameters=None):
    """Run the full estimation pipeline, optionally iterating for FGLS.

    When ``do_fgls=False`` this is a thin wrapper around :func:`lm_internal`
    that packages the result in the same dict structure as the FGLS path.

    When ``do_fgls=True`` the function iterates up to ``fgls_kwds['maxiter']``
    times.  At each step:

    1. Re-estimate β using the current FGLS weights.
    2. Regress log(û²) on X to estimate heteroscedasticity (the FGLS
       log-squared-residual trick).
    3. Update weights as ``1 / sqrt(exp(X β_u))``, multiplied by any
       pre-existing ``weights`` if provided.
    4. Terminate when ``max|Δβ| < fgls_kwds['tol']`` or the iteration limit
       is reached.

    Cannot be used together with GLS (``sigma`` / ``sigma_inv`` must be
    ``None`` when ``do_fgls=True``).

    Args:
        nobs (int): Number of observations.
        endog (array-like): Dependent variable vector, shape ``(n,)`` or
            ``(n, 1)``.
        exog (array-like or sparse): Design matrix, shape ``(n, p)``.
        do_fgls (bool): If ``True``, perform iterative FGLS re-weighting.
        fgls_kwds (dict): FGLS options.  Recognised keys:
            - ``'maxiter'`` (int): Max iterations (default ``FGLS_MAX_ITER_DEFAULT``).
            - ``'tol'`` (float): Convergence tolerance (default ``FGLS_TOL_DEFAULT``).
        absorb: Fixed-effects absorption specification forwarded to
            :func:`lm_internal`.
        is_endog_regressor (array-like of bool, optional): Forwarded to
            :func:`lm_internal`.
        weights (array-like, optional): Initial observation weights.
        instruments: Instrument matrix forwarded to :func:`lm_internal`.
        debug (bool): Verbose output including per-iteration convergence.
        force_iv_projection (bool): Forwarded to :func:`lm_internal`.
        sigma (ndarray, optional): GLS error covariance matrix (mutually
            exclusive with ``do_fgls=True``).
        sigma_inv (ndarray, optional): Pre-computed inverse of ``sigma``.
        scale_design_matrix (bool): Forwarded to :func:`lm_internal`.
        compute_eigenvalues_instruments (bool): Forwarded.
        dense_threshold_mb (float): Forwarded.
        inverse_method: Forwarded.
        compute_eigenvalues: Forwarded.
        ridge_parameters (ndarray, optional): Per-column ridge penalties.

    Returns:
        dict: With keys:
            - ``'result'`` (LinearModelRegressionResultsRaw): Final
              estimation result.
            - ``'fgls_weights'`` (ndarray): Final weight vector used.
            - ``'gls'`` (dict, non-FGLS path only): ``{'sigma': ..., 'sigma_inv': ...}``.
            - ``'fgls_info'`` (dict): FGLS convergence metadata
              (``maxiter``, ``tol``, ``err``, ``n_iter``).  Empty when
              ``do_fgls=False``.

    Raises:
        Exception: If both GLS (``sigma``/``sigma_inv``) and
            ``do_fgls=True`` are requested simultaneously.
    """

    if not do_fgls:
        # Non-FGLS path: delegate directly to lm_internal.
        return {
            'result': lm_internal(
                endog, exog, weights, instruments, absorb, is_endog_regressor, debug, force_iv_projection,
                sigma=sigma, sigma_inv=sigma_inv,
                scale_design_matrix=scale_design_matrix, compute_eigenvalues=compute_eigenvalues,
                dense_threshold_mb=dense_threshold_mb, inverse_method=inverse_method,
                ridge_parameters=ridge_parameters, compute_eigenvalues_instruments=compute_eigenvalues_instruments),
            'fgls_weights': weights,
            'gls': {'sigma': sigma, 'sigma_inv': sigma_inv},
            'fgls_info': dict()
        }

    if not (sigma is None and sigma_inv is None):
        raise Exception("Can't do GLS and FGLS!")

    is_weighted = weights is not None

    fgls_max_iter = fgls_kwds.get('maxiter', FGLS_MAX_ITER_DEFAULT)
    fgls_tol = fgls_kwds.get('tol', FGLS_TOL_DEFAULT)
    fgls_weights = weights if is_weighted else np.ones(nobs)
    X_fgls = exog
    W_fgls = sp_diags(fgls_weights)

    ncp_fgls = get_matrix_inverse_internal(gram_matrix(X_fgls, weights=fgls_weights,
                                                       dense_threshold_mb=dense_threshold_mb))
    ncp_fgls = csc_matrix(ncp_fgls)

    beta_last = np.inf
    err = np.inf

    to_dense = lambda x: x.toarray() if isspmatrix(x) else x

    for fgls_iter in range(fgls_max_iter):

        result = lm_internal(
            endog, exog, weights=fgls_weights, instruments=instruments, absorb=absorb,
            scale_design_matrix=scale_design_matrix, is_endog_regressor=is_endog_regressor, debug=debug,
            force_iv_projection=force_iv_projection, compute_eigenvalues=compute_eigenvalues,
            ridge_parameters=ridge_parameters, dense_threshold_mb=dense_threshold_mb,
            inverse_method=inverse_method)

        params = result.params
        if isspmatrix(params):
            params = params.toarray()
        err = np.max(np.abs(params - beta_last))

        if debug:
            print("FGLS iter = %3d, err = %.2e" % (fgls_iter, err))

        if err < fgls_tol or fgls_iter == fgls_max_iter - 1:
            break

        beta_last = to_dense(result.params)

        # FGLS re-weighting: regress log(û²) on X to model heteroscedasticity.
        # New weights are 1 / sqrt( exp(X β_u) ), proportional to 1/std(û).
        log_u_sq = np.log(to_dense(result.resid_raw) ** 2)
        if isspmatrix(X_fgls):
            log_u_sq = csc_matrix(log_u_sq).reshape((-1, 1))
        beta_u = ncp_fgls.dot(X_fgls.transpose().dot(W_fgls.dot(log_u_sq)))
        fgls_weights = np.sqrt(1.0 / np.exp(to_dense(X_fgls.dot(beta_u)))).ravel()
        if is_weighted:
            fgls_weights *= weights

    return {
        'result': result,
        'fgls_weights': fgls_weights,
        'fgls_info': {'maxiter': fgls_max_iter, 'tol': fgls_tol, 'err': err, 'n_iter': fgls_iter + 1}
    }


def get_fitted_values(exog, endog, params):
    """Compute fitted values and residuals from parameters.

    Args:
        exog (ndarray or sparse): Design matrix, shape ``(n, p)``.
        endog (ndarray or sparse): Dependent variable, shape ``(n,)`` or
            ``(n, 1)``.
        params (ndarray or sparse): Coefficient vector, shape ``(p,)`` or
            ``(p, k)`` for multi-outcome.

    Returns:
        tuple:
            - **fittedvalues**: ``exog @ params``.
            - **resid**: ``endog - fittedvalues``.
    """
    fittedvalues = exog.dot(params)
    resid = endog - fittedvalues
    return fittedvalues, resid


def get_sst_within(endog, weights, resid):
    """Compute within-group total sum of squares and within R².

    Used after fixed-effects absorption to measure how much of the
    within-group variation in ``endog`` is explained by the regressors.

    Args:
        endog (ndarray or sparse): Absorbed (demeaned) dependent variable.
        weights (array-like or None): Observation weights.
        resid (ndarray or sparse): Residuals from the absorbed regression.

    Returns:
        tuple:
            - **sst_within** (float): Weighted SST computed on the demeaned
              absorbed endog.
            - **rsquared_within** (float): 1 − SSR_within / SST_within.
    """
    sst_within = get_wtd_sum_squares(endog, weights, demean=True)
    rsquared_within = 1.0 - get_wtd_sum_squares(resid, weights, demean=False) / sst_within
    return sst_within, rsquared_within


def lm_internal(endog, exog, weights=None, instruments=None, absorb=None, is_endog_regressor=None, debug=False,
                force_iv_projection=DEFAULT_LM_FORCE_IV_PROJECTION, _time=None,
                scale_design_matrix=DEFAULT_LM_SCALE_DESIGN_MATRIX, compute_eigenvalues=DEFAULT_LM_COMPUTE_EIGENVALUES,
                ridge_parameters=None, compute_eigenvalues_instruments=DEFAULT_LM_COMPUTE_EIGENVALUES_INSTRUMENTS,
                dense_threshold_mb=DEFAULT_DENSE_THRESHOLD_MB, inverse_method=DEFAULT_LM_INVERSE_METHOD,
                sigma=None, sigma_inv=None):
    """Core estimation pipeline: absorb → IV first stage → solve → unscale.

    Executes the complete OLS / WLS / IV / absorb estimation in a single
    pass:

    1. Type-normalise all inputs (``check_types``).
    2. If ``absorb`` is supplied, center endog, exog, and instruments within
       groups (``AbsorbTools2.do_absorb``).
    3. If ``instruments`` is supplied, run the IV first stage to replace
       endogenous regressors with their instrument projections
       (``iv_first_stage2``).
    4. If ``scale_design_matrix``, divide each column of the (absorbed,
       instrumented) design matrix by its L2 norm; ridge penalties are
       rescaled accordingly.
    5. Compute ``(X'WX)^{-1}`` and solve for β
       (``fit_least_squares_model_internal``).
    6. Unscale params and normalised covariance matrix back to the original
       column units.
    7. Compute raw fitted values, residuals, and within R².

    Args:
        endog (array-like): Dependent variable, shape ``(n,)`` or ``(n, 1)``.
        exog (array-like or sparse): Design matrix, shape ``(n, p)``.
        weights (array-like, optional): Observation weights.
        instruments (array-like or sparse, optional): Instrument matrix,
            shape ``(n, q)``.
        absorb: Fixed-effects specification accepted by
            ``AbsorbTools2.do_absorb`` (string, list of strings, or array).
        is_endog_regressor (array-like of bool, optional): Endogeneity mask
            of length ``p``.  Defaults to all ``True`` when instruments are
            present.
        debug (bool): Verbose timing / progress output.
        force_iv_projection (bool): Project all columns through the IV step
            even when they are classified as exogenous.
        _time (float, optional): ``time.time()`` baseline for elapsed logging.
        scale_design_matrix (bool): Scale columns before solving.
        compute_eigenvalues: ``None`` (auto), ``True``, or ``False``.
        ridge_parameters (ndarray, optional): Per-column L2 penalties.
        compute_eigenvalues_instruments (bool): Compute eigenvalues of Z'WZ.
        dense_threshold_mb (float): Threshold for sparse/dense matrix paths.
        inverse_method: Forwarded to ``get_normalized_cov_params``.
        sigma (ndarray, optional): GLS error covariance.
        sigma_inv (ndarray, optional): Pre-computed GLS covariance inverse.

    Returns:
        LinearModelRegressionResultsRaw: Raw estimation results before
            summary statistics are computed.
    """
    if _time is None:
        _time = time.time()

    is_weighted, is_gls = check_weights_and_sigma(weights, sigma, sigma_inv)
    if is_gls:
        if sigma_inv is None:
            if isspmatrix(sigma):
                raise Exception("Sparse GLS not supported yet!")
            else:
                sigma_inv = np.linalg.inv(sigma)
            sigma = None

    is_iv = instruments is not None
    is_absorb = absorb is not None

    # TODO add sigma_inv
    endog, exog, weights, instruments, absorb, sigma, sigma_inv \
        = check_types(endog, exog, weights, instruments, absorb, sigma, sigma_inv)

    if is_absorb:
        absorb_info = AbsorbTools2.do_absorb(
            absorb, endog, exog, weights=weights, instruments=instruments, debug=debug, _time=_time)
        exog_absorb = absorb_info.exog_absorbed
        endog_absorb = absorb_info.endog_absorbed
        instruments_absorb = absorb_info.instruments_absorbed
    else:
        absorb_info = None
        exog_absorb, endog_absorb, instruments_absorb = exog, endog, instruments

    if is_iv:
        if is_endog_regressor is None:
            is_endog_regressor = np.array([True] * exog.shape[1])
        iv_info = iv_first_stage2(
            exog_absorb, instruments_absorb, is_endog_regressor, debug=debug, _time=_time, residual_inclusion=False,
            weights=weights, residual_inclusion_order=1, force_iv_projection=force_iv_projection,
            dense_threshold_mb=dense_threshold_mb, inverse_method=DEFAULT_LM_INVERSE_METHOD,
            scale_design_matrix=scale_design_matrix, compute_eigenvalues_instruments=compute_eigenvalues_instruments)
        exog_absorb_instrumented = iv_info.exog_instrumented
        # print("Q ", pd.DataFrame(iv_info.instrument_params.toarray()))
    else:
        iv_info = None
        exog_absorb_instrumented = exog_absorb

    if scale_design_matrix:
        if debug:
            print('Scaling exog....', end='')
        x_scales = scale_in_place(exog_absorb_instrumented)
        # Ridge penalties are in units of the original (unscaled) design matrix,
        # so we rescale them to match the scaled columns: λ_scaled = λ / ||x_k||².
        if ridge_parameters is not None:
            ridge_parameters = ridge_parameters / x_scales ** 2
        if debug:
            print("%.4fs" % (time.time() - _time))

    params, normalized_cov_params, eigenvals, condition_number \
        = fit_least_squares_model_internal(exog_absorb_instrumented, endog_absorb,
                                           weights=weights,
                                           sigma=sigma, sigma_inv=sigma_inv,
                                           debug=debug, _time=_time,
                                           return_params_only=False, normalize_XpX=not scale_design_matrix,
                                           compute_eigenvalues=compute_eigenvalues,
                                           ridge_parameters=ridge_parameters, inverse_method=inverse_method,
                                           dense_threshold_mb=dense_threshold_mb)

    if scale_design_matrix:
        if debug:
            print('Unscaling exog and params....', end='')

        unscale_in_place(exog_absorb_instrumented, x_scales)

        if isspmatrix(params):
            params.data /= x_scales[params.indices]
        else:
            if np.ndim(params) > 1:
                params = np.column_stack([params[:, j] / x_scales for j in range(params.shape[1])])
            else:
                params /= x_scales

        if isspmatrix(normalized_cov_params):
            S = sp_diags(1.0 / x_scales)
            normalized_cov_params = S.dot(normalized_cov_params.dot(S))
        else:
            S = 1.0 / x_scales
            normalized_cov_params = S.reshape((1, -1)) * normalized_cov_params * S.reshape((-1, 1))

        if debug:
            print("%.4fs" % (time.time() - _time))

    if debug:
        print("Computing (raw) fitted values and residuals...", end='')
    fittedvalues_raw, resid_raw = get_fitted_values(exog_absorb_instrumented, endog_absorb, params)
    sst_within_raw, rsquared_within_raw = get_sst_within(endog_absorb, weights, resid_raw)
    if debug:
        print("%.3f s" % (time.time() - _time))

    return LinearModelRegressionResultsRaw(
        params, normalized_cov_params, exog_absorb_instrumented, absorb_info,
        iv_info, fittedvalues_raw, resid_raw, rsquared_within_raw, condition_number, eigenvals,
    )


def fit_least_squares_model_internal(exog, endog, weights=None, debug=False, _time=None,
                                     normalize_XpX=True, compute_eigenvalues=DEFAULT_LM_COMPUTE_EIGENVALUES,
                                     return_params_only=False, ridge_parameters=None,
                                     inverse_method=DEFAULT_LM_INVERSE_METHOD,
                                     dense_threshold_mb=DEFAULT_DENSE_THRESHOLD_MB,
                                     sigma=None, sigma_inv=None):
    """Invert the Gram matrix and solve for the OLS / WLS / GLS parameters.

    Computes ``(X'WX + diag(λ))^{-1}`` (the normalised covariance matrix)
    and then solves ``β = (X'WX)^{-1} X'Wy``.  Optionally returns only
    ``β`` for callers that do not need the covariance.

    Args:
        exog (array-like or sparse): Design matrix, shape ``(n, p)``.
        endog (array-like or sparse): Dependent variable.
        weights (array-like, optional): Observation weights.
        debug (bool): Print timing messages.
        _time (float, optional): Elapsed-time baseline.
        normalize_XpX (bool): If ``True``, normalise X'WX before inversion
            (used when column scaling has **not** been applied upstream).
        compute_eigenvalues: ``None`` (auto), ``True``, or ``False``.x
        return_params_only (bool): If ``True``, return only the parameter
            vector without the covariance matrix or eigenvalues.
        ridge_parameters (ndarray, optional): Per-column L2 ridge penalties
            added to the diagonal of X'WX before inversion.
        inverse_method: Forwarded to ``get_normalized_cov_params``.
        dense_threshold_mb (float): Memory threshold for sparse/dense paths.
        sigma (ndarray, optional): GLS error covariance.
        sigma_inv (ndarray, optional): Pre-computed GLS inverse.

    Returns:
        ndarray or tuple:
            - If ``return_params_only=True``: parameter vector β.
            - Otherwise: ``(params, normalized_cov_params, eigenvals,
              condition_number)``.
    """
    if _time is None:
        _time = time.time()

    check_weights_and_sigma(weights, sigma, sigma_inv)

    if debug:
        print("Computing the normalized cov params...", end='')
    if compute_eigenvalues is None:
        compute_eigenvalues = exog.shape[1] < DEFAULT_LM_COMPUTE_EIGENVALUES_UNDER_MAX_DIM
    normalized_cov_params, eigenvals, condition_number \
        = get_normalized_cov_params(exog, weights=weights, sigma=sigma, sigma_inv=sigma_inv,
                                    normalize=normalize_XpX,
                                    return_eigenvals=True, debug=debug, _time=_time,
                                    compute_eigenvalues=compute_eigenvalues,
                                    ridge_parameters=ridge_parameters, inverse_method=inverse_method,
                                    dense_threshold_mb=dense_threshold_mb)

    XpWy = get_exog_dot_endog(exog, endog, weights=weights, sigma=sigma, sigma_inv=sigma_inv)

    if debug:
        print("%.3f s" % (time.time() - _time))

    if debug:
        print("Fitting parameters...", end='')

    params = normalized_cov_params.dot(XpWy)

    if debug:
        print("%.3f s" % (time.time() - _time))

    if return_params_only:
        return params
    else:
        return params, normalized_cov_params, eigenvals, condition_number


class LinearModelRegressionResultsRaw(object):
    """Lightweight container for the raw numerical outputs of :func:`lm_internal`.

    Stores only the information needed to compute summary statistics and the
    covariance matrix downstream.  No inference is performed here; all
    post-estimation computations happen in :func:`get_fit_summary_stats` and
    :class:`~kanly.regression.linear_models.variance_covariance2.SparseVarianceCovariance2`.

    Attributes:
        params: Estimated parameter vector (or matrix for multi-outcome),
            shape ``(p,)`` or ``(p, k)``.  Sparse CSC or dense ndarray.
        normalized_cov_params: ``(X'WX)^{-1}``, shape ``(p, p)``.  Used to
            compute the sandwich / robust covariance downstream.
        exog_absorb_instrumented: Final design matrix after absorption and IV
            projection, shape ``(n, p)``.
        nobs (int): Number of observations.
        num_params (int): Number of model parameters (columns of ``exog_absorb_instrumented``).
        absorb_info: ``AbsorbInfo`` object from ``AbsorbTools2.do_absorb``, or
            ``None`` if no fixed effects were absorbed.
        instrument_info: ``InstrumentInfoInternal`` from ``iv_first_stage2``,
            or ``None`` for non-IV models.
        final_design_matrix: Alias for ``exog_absorb_instrumented``.
        fittedvalues_raw: Fitted values from the absorbed/instrumented design
            matrix: ``exog_absorb_instrumented @ params``.
        resid_raw: Raw residuals ``endog_absorb − fittedvalues_raw``.
        rsquared_within_raw (float): Within R² computed on the absorbed data
            (before adding back group means).
        condition_number (float or None): Condition number of X'WX.
        eigenvals (ndarray or None): Eigenvalues of X'WX.
    """

    def __init__(self, params, normalized_cov_params, exog_absorb_instrumented, absorb_info, instrument_info,
                 fittedvalues_raw, resid_raw, rsquared_within_raw, condition_number, eigenvals):
        """Initialise the raw results container.

        Args:
            params: Coefficient vector or matrix.
            normalized_cov_params: ``(X'WX)^{-1}``.
            exog_absorb_instrumented: Final (absorbed + instrumented) design
                matrix.
            absorb_info: Fixed-effects absorption info, or ``None``.
            instrument_info: IV first-stage info, or ``None``.
            fittedvalues_raw: ``exog_absorb_instrumented @ params``.
            resid_raw: ``endog_absorb - fittedvalues_raw``.
            rsquared_within_raw (float): Within-group R² on absorbed data.
            condition_number (float or None): Condition number of X'WX.
            eigenvals (ndarray or None): Eigenvalues of X'WX.
        """
        self.params = params
        self.normalized_cov_params = normalized_cov_params
        self.exog_absorb_instrumented = exog_absorb_instrumented
        self.nobs, self.num_params = self.exog_absorb_instrumented.shape
        self.absorb_info = absorb_info
        self.instrument_info = instrument_info
        self.final_design_matrix = exog_absorb_instrumented
        self.fittedvalues_raw = fittedvalues_raw
        self.resid_raw = resid_raw
        self.rsquared_within_raw = rsquared_within_raw
        self.condition_number = condition_number
        self.eigenvals = eigenvals

    def __repr__(self):
        """Return a string representation of all stored attributes."""
        return str(self.__dict__)


def check_types(endog, exog, weights, instruments, absorb, sigma, sigma_inv):
    """Normalise input array types for a consistent sparse or dense pipeline.

    Ensures that weights are a flat dense array (not sparse), and that when
    any of endog / exog / instruments / absorb is sparse, all of them are
    converted to CSC sparse format (using ``none_convert_2_sparse`` which
    leaves ``None`` inputs as ``None``).

    Args:
        endog: Dependent variable array.
        exog: Design matrix.
        weights: Weight vector.  If sparse, converted to a 1-D ndarray.
        instruments: Instrument matrix, or ``None``.
        absorb: Absorption matrix / specification, or ``None``.
        sigma: GLS covariance matrix, or ``None``.
        sigma_inv: Pre-computed GLS inverse, or ``None``.

    Returns:
        tuple: ``(endog, exog, weights, instruments, absorb, sigma, sigma_inv)``
            with type normalisations applied.
    """
    # TODO sigma, sigma_inv type check
    # TODO 4444444 DENSE EXOG FIX
    # if not isspmatrix(endog):
    #     endog = csc_matrix(endog).reshape((-1, 1))
    # if not isspmatrix(exog):
    #     exog = csc_matrix(exog)
    if weights is not None and isspmatrix(weights):
        weights = weights.toarray().flatten()
    # if instruments is not None and not isspmatrix(instruments):
    #     instruments = csc_matrix(instruments)
    # if absorb is not None and not isspmatrix(absorb):
    #     absorb = csc_matrix(absorb)
    if np.any([isspmatrix(t) for t in (exog, instruments, absorb, endog)]):
        exog = none_convert_2_sparse(exog)
        endog = none_convert_2_sparse(endog)
        instruments = none_convert_2_sparse(instruments)
        absorb = none_convert_2_sparse(absorb)
    return endog, exog, weights, instruments, absorb, sigma, sigma_inv


def get_exog_dot_endog(X, y, weights=None, sigma=None, sigma_inv=None):
    """Compute the cross-product X'Wy (or X'Σ⁻¹y for GLS).

    Computes the numerator of the OLS / WLS / GLS normal equations:
    - Unweighted OLS: X'y
    - WLS: X'diag(w)y = sum_i w_i x_i y_i
    - GLS: X'Σ⁻¹y

    Exactly one of (weights, sigma/sigma_inv) may be supplied.

    Args:
        X (array-like or sparse): Design matrix, shape ``(n, p)``.
        y (array-like or sparse): Dependent variable, shape ``(n,)`` or
            ``(n, 1)``.
        weights (array-like, optional): Observation weight vector.
        sigma (ndarray, optional): GLS covariance matrix (used when
            ``sigma_inv`` is not yet available).
        sigma_inv (ndarray, optional): Pre-computed inverse of ``sigma``.

    Returns:
        ndarray or sparse: Cross-product vector X'Wy, shape ``(p,)`` or
            ``(p, k)`` for multi-outcome.

    Raises:
        Exception: If both ``weights`` and GLS arguments are supplied, or if
            both ``sigma`` and ``sigma_inv`` are supplied.
    """
    is_weighted = weights is not None
    is_gls = sigma is not None or sigma_inv is not None
    if is_weighted and is_gls:
        raise Exception
    if sigma is not None and sigma_inv is not None:
        raise Exception

    if is_weighted:
        if isspmatrix(X):
            Xpy = X.transpose().dot(csc_matrix_by_column_array_broadcast(y, weights))
        else:
            Xpy = X.T.dot(y * weights.reshape(np.shape(y)))

    elif is_gls:
        # TODO sparse
        if sigma_inv is None:
            sigma_inv = get_matrix_inverse_internal(sigma, normalize=True, copy=True)
        Xpy = X.T.dot(sigma_inv).dot(y)

    else:
        Xpy = X.transpose().dot(y)

    return Xpy


def lin_mod_get_rsquared(wssr, wsst, wsst_no_intercept, has_constant, nobs, df_resid, const_number=1):
    """Compute R² and adjusted R² for a linear model.

    When the model has a constant (intercept or implicit intercept), the
    centred TSS ``wsst`` is used.  Without a constant, the uncentred TSS
    ``wsst_no_intercept = Σ(wᵢ yᵢ²)`` is used so that R² remains in [0, 1].

    Args:
        wssr (float): Weighted sum of squared residuals.
        wsst (float): Weighted, centred total sum of squares.
        wsst_no_intercept (float): Uncentred weighted TSS (Σ wᵢ yᵢ²).
        has_constant (bool): Whether the model includes any constant term.
        nobs (int): Number of observations.
        df_resid (int): Residual degrees of freedom (n - k).
        const_number (int): Number of constant sparse_terms (1 for intercept).
            Used in the adjusted R² denominator.

    Returns:
        tuple:
            - **rsquared** (float): 1 − WSSR / WSST (centred or uncentred).
            - **rsquared_adj** (float): Adjusted R²;  0.0 if df_resid == 0.
    """
    # If no intercept, total sum of squares is sum(endog**2), not sum([endog-\bar endog]**2)
    rsquared = 1.0 - wssr / (wsst if has_constant else wsst_no_intercept)
    rsquared_adj = (1.0 - (nobs - const_number) / df_resid * (1.0 - rsquared)) if df_resid else 0.0
    return rsquared, rsquared_adj


def lin_mod_get_method(is_iv=False, is_weighted=False, ridge_kwds=None):
    """Return a human-readable method string for regression results headers.

    Combines the IV, weighting, and ridge flags into a single label such as
    ``'OLS'``, ``'WLS'``, ``'IV (2SLS)'``, ``'WLS-RIDGE'``, etc.

    Args:
        is_iv (bool): True if instrumental variables are used.
        is_weighted (bool): True if observation weights are provided.
        ridge_kwds (dict, optional): Ridge penalty keyword dict.  A non-zero
            ``'alpha'`` causes ``'-RIDGE'`` to be appended to the label.

    Returns:
        str: Method label string.
    """
    if is_weighted:
        if not is_iv:
            method = 'WLS'
        else:
            method = 'IV (W2SLS)'
    else:
        if not is_iv:
            method = 'OLS'
        else:
            method = 'IV (2SLS)'

    if ridge_kwds is not None and np.any(np.array(ridge_kwds['alpha'])) > 0:
        method += '-RIDGE'

    return method


def lin_mod_fit_predicted_values_and_residuals(params, endog, exog, exog_absorb_instrumented, weights, debug=False,
                                               _time=None):
    """Compute fitted values, residuals, and their sums of squares.

    Evaluates predictions using both the original ``exog`` (X β, for
    reporting) and the absorbed/instrumented ``exog_absorb_instrumented``
    (X̂ β, for IV residuals).  Both the raw and weighted sums of squared
    residuals are returned.

    Args:
        params (array-like): Coefficient vector, shape ``(p,)``.
        endog (array-like): Dependent variable, shape ``(n,)``.
        exog (array-like or sparse): Original design matrix, shape ``(n, p)``.
        exog_absorb_instrumented (array-like or sparse): Absorbed/instrumented
            design matrix, shape ``(n, p)``.
        weights (array-like or None): Observation weights.
        debug (bool): Print timing messages.
        _time (float, optional): Elapsed-time baseline.

    Returns:
        tuple of 6 elements:
            - **y_hat**: ``X @ params`` (fitted values using original X).
            - **y_hat_instrumented**: ``X̂ @ params`` (IV fitted values).
            - **resid**: ``y − y_hat``.
            - **resid_instrumented**: ``y − y_hat_instrumented``.
            - **ssr** (float): Sum of squared residuals ``sum(resid²)``.
            - **wssr** (float): Weighted SSR ``sum(w * resid²)``.
    """
    if _time is None:
        _time = time.time()

    if debug:
        print("\tComputing fitted values...", end='')

    y_hat = flexible_mat_dot_vec(exog, params)
    y_hat_instrumented = flexible_mat_dot_vec(exog_absorb_instrumented, params)

    # y_hatw = wexog_instrumented.dot(beta)
    # y_hatw_uninstrumented = wexog.dot(beta)

    if debug:
        print("%.3f s" % (time.time() - _time))

    if debug:
        print("\tComputing residual values and sum of squares residual...", end='')

    if isspmatrix(endog):
        y = endog.toarray()
    else:
        y = endog
    y = y.flatten()

    resid = y - y_hat
    resid_instrumented = y - y_hat_instrumented

    ssr = (resid ** 2).sum()
    wssr = get_wtd_sum_squares(resid, weights, demean=False)

    if debug:
        print("%.3f s" % (time.time() - _time))

    return (
        y_hat,  # X * beta
        y_hat_instrumented,  # (Z (Z' W Z) (Z' W X) * beta
        resid,  # y - X * beta
        resid_instrumented,  # y - X_hat * beta
        ssr,  # sum of squared `resid`
        wssr,  # sum of squared weighted `wresid`
    )


def lin_mod_add_absorbed_fe_to_y_hat(params, wendog_group_means, wexog_group_means,
                                     group_to_row_lists,
                                     y_hat, resid, debug=False, _time=None):
    """Add back absorbed group means to produce full-sample fitted values.

    After estimating on the demeaned (absorbed) data, fitted values only
    reflect within-group variation.  This function recovers the group-level
    intercepts and adds them back to produce predictions on the original scale:

        full_y_hat = y_hat_absorbed + (ȳ_g − x̄_g β)

    where ȳ_g and x̄_g are the group means of endog and exog in group g.

    Args:
        params (ndarray): Coefficient vector, shape ``(p,)``.
        wendog_group_means: Group-level means of the (weighted) endog,
            shape ``(G,)`` where G is the number of groups.
        wexog_group_means: Group-level means of the (weighted) exog,
            shape ``(G, p)``.
        group_to_row_lists (dict): Mapping from group index to the list of
            observation row indices belonging to that group.
        y_hat (ndarray): Current fitted values from the absorbed regression,
            shape ``(n,)``.
        resid (ndarray): Current residuals, shape ``(n,)``.
        debug (bool): Print timing messages.
        _time (float, optional): Elapsed-time baseline.

    Returns:
        tuple:
            - **y_hat** (ndarray): Updated fitted values with group means
              added back.
            - **resid** (ndarray): Updated residuals (group means subtracted).
            - **absorbed_y_baselines** (ndarray): Per-observation group
              mean offsets, shape ``(n,)``.
    """
    if _time is None:
        _time = time.time()

    if debug:
        print("Adding/Subtracting absorbed group means from fittedvalues/resid, respectively...", end='')

    if isspmatrix(wendog_group_means):
        absorb_pred_vals = wendog_group_means.toarray() - wexog_group_means.dot(params)
    else:
        absorb_pred_vals = wendog_group_means - wexog_group_means.dot(params)

    absorbed_y_baselines = np.zeros(y_hat.shape[0])
    for k, v in group_to_row_lists.items():
        absorbed_y_baselines[v] = absorb_pred_vals[k]

    y_hat = y_hat + absorbed_y_baselines
    resid = resid - absorbed_y_baselines

    if debug:
        print(" %.3fs" % (time.time() - _time))

    return y_hat, resid, absorbed_y_baselines


def lin_mod_get_sum_of_squares(endog, weights=None, debug=False, _time=None):
    """Compute SST, weighted SST, and uncentred weighted TSS for OLS/WLS.

    Computes three variants of the total sum of squares needed for R²
    calculations:

    - ``sst``                = Σᵢ (yᵢ − ȳ)²  (unweighted, centred)
    - ``wsst``               = Σᵢ wᵢ(yᵢ − ȳ_w)²  (weighted, centred)
    - ``wsst_no_intercept``  = Σᵢ wᵢ yᵢ²  (weighted, uncentred)

    where ȳ_w = Σ wᵢ yᵢ / Σ wᵢ is the weighted mean.

    Args:
        endog (array-like or sparse): Dependent variable, shape ``(n,)``.
        weights (array-like, optional): Observation weights.  When ``None``
            all observations receive equal weight.
        debug (bool): Print timing messages.
        _time (float, optional): Elapsed-time baseline.

    Returns:
        tuple of 3 floats:
            - **sst**: Unweighted centred TSS.
            - **wsst**: Weighted centred TSS.
            - **wsst_no_intercept**: Weighted uncentred TSS.
    """
    if _time is None:
        _time = time.time()

    if debug:
        print("\tComputing SST...", end='')

    is_weighted = weights is not None

    if isspmatrix(endog):
        endog = endog.toarray().flatten()
    nobs = len(endog)

    # weight_matrix_temp2 = spdiags(weights).tocsc()
    # weighted_y = weight_matrix_temp.dot(endog)
    weighted_y = endog * weights if is_weighted else endog
    w_sum = weights.sum() if is_weighted else nobs
    weighted_y_mean = weighted_y.sum() / w_sum
    wsst_no_intercept = (weights * endog ** 2).sum() if is_weighted else (endog ** 2).sum()
    wsst = wsst_no_intercept - w_sum * (weighted_y_mean ** 2)
    sst = (endog ** 2).mean() - (endog.mean()) ** 2
    if debug:
        print("%.3f s" % (time.time() - _time))

    return (sst,  # sum( (y - y_bar) ** 2 )
            wsst,  # sum ( (w*y - weighted_avg(y | w)) ** 2 )
            wsst_no_intercept  # sum ( (w*y) ** 2 )
            )


def get_gls_sum_squares(resid, sigma, sigma_inv, demean=False, debug=False):
    """Compute the GLS-weighted sum of squared residuals û'Σ⁻¹û.

    Args:
        resid (ndarray): Residual vector, shape ``(n,)``.
        sigma (ndarray or None): GLS error covariance matrix, shape
            ``(n, n)``.  Used to compute ``sigma_inv`` when not supplied.
        sigma_inv (ndarray or None): Pre-computed inverse of ``sigma``.
        demean (bool): Unused parameter reserved for future use.
        debug (bool): Unused parameter reserved for future use.

    Returns:
        float: Generalised SSR: ``resid' @ sigma_inv @ resid``.
    """
    if sigma_inv is None:
        sigma_inv = np.linalg.pinv(sigma)
    return resid.dot(sigma_inv).dot(resid)


def lin_mod_get_sum_of_squares_gls(endog, sigma, sigma_inv, _time=None, debug=False):
    """Compute SST, weighted SST, and uncentred TSS under GLS.

    The GLS-weighted mean ȳ_GLS minimises the GLS criterion, yielding:

        ȳ* = (Σᵢ yᵢ Σⱼ Ω⁻¹ᵢⱼ) / (Σᵢⱼ Ω⁻¹ᵢⱼ)

    and the GLS-centred TSS is (y − ȳ*)' Ω⁻¹ (y − ȳ*).

    Reference (LaTeX from original code):

        (y − 1β)' Ω⁻¹ (y − 1β)
          = y'Ω⁻¹y − 2β Σᵢ yᵢ Σⱼ Ω⁻¹ᵢⱼ + β² Σᵢⱼ Ω⁻¹ᵢⱼ

    Args:
        endog (ndarray): Dependent variable, shape ``(n,)``.
        sigma (ndarray or None): GLS covariance matrix.  Used to compute
            ``sigma_inv`` when not supplied.
        sigma_inv (ndarray or None): Pre-computed ``Ω⁻¹``.
        _time (float, optional): Unused; retained for API consistency.
        debug (bool): Unused; retained for API consistency.

    Returns:
        tuple of 3 floats:
            - **sst**: Unweighted centred SST Σᵢ (yᵢ − ȳ)².
            - **wsst**: GLS-centred TSS (y − ȳ*)' Ω⁻¹ (y − ȳ*).
            - **wsst_no_intercept**: y' Ω⁻¹ y (uncentred GLS TSS).
    """
    if sigma_inv is None:
        sigma_inv = np.linalg.pinv(sigma)
    sst = sum(endog ** 2) - np.mean(endog) ** 2 * len(endog)
    mean_gls = np.dot(endog, sigma_inv.sum(axis=0)) / sigma_inv.sum()
    wsst = (endog - mean_gls).dot(sigma_inv).dot(endog - mean_gls)
    wsst_no_intercept = endog.dot(sigma_inv).dot(endog)

    return (sst,  # sum( (y - y_bar) ** 2 )
            wsst,  # sum ( (w*y - weighted_avg(y | w)) ** 2 )
            wsst_no_intercept  # sum ( (w*y) ** 2 )
            )


def lin_mod_get_mean_exog_columns(is_weighted, weights, exog, debug=False, _time=None):
    """Compute (weighted) column means of the design matrix.

    These means are stored in ``SparseLinearModel.wexog_instrumented_means``
    and used by ``SparseLinearRegressionResults.test_lift`` to construct the
    denominator of treatment-lift ratio tests.

    Args:
        is_weighted (bool): True if observation weights are used.
        weights (array-like or None): Weight vector of length ``n``.
        exog (array-like or sparse): Design matrix, shape ``(n, p)``.
        debug (bool): Print timing message.
        _time (float, optional): Elapsed-time baseline.

    Returns:
        tuple:
            - **wexog_instrumented_means** (ndarray, shape (p,)):
              (Weighted) column means of ``exog``.
            - **sum_weights** (float): Sum of weights (or ``n`` when
              unweighted).
    """
    if _time is None:
        _time = time.time()

    if debug:
        print("Computing means of regressors...", end="")

    if is_weighted:
        sum_weights = np.sum(weights)
        if isspmatrix(exog):
            weights_csc = csc_matrix(weights)
            if weights_csc.shape[1] == 1:
                weights_csc = weights_csc.transpose()
            wexog_instrumented_means = np.dot(weights_csc, exog).toarray().flatten() / sum_weights
        else:
            wexog_instrumented_means = (np.dot(weights, exog)).flatten() / sum_weights
    else:
        wexog_instrumented_means = np.array(exog.mean(axis=0)).ravel()
        sum_weights = exog.shape[0]
    if debug:
        print("%.3f s" % (time.time() - _time))

    return wexog_instrumented_means, sum_weights


def get_fit_summary_stats(nobs, num_absorbed, params, endog, exog, exog_absorb_instrumented, rsquared_within_raw,
                          weights, do_fgls, is_weighted, has_implicit_constant, has_intercept, is_absorb, absorb_info,
                          sigma, sigma_inv,
                          _time=None, debug=False):
    """Compute all post-estimation summary statistics for one outcome.

    Orchestrates fitted values, group-mean restoration (if absorb), sum-of-
    squares, degrees of freedom, R², log-likelihood, and residuals.
    Called once per outcome in the multi-outcome loop of
    ``SparseLinearModel.fit`` and in ``SparseLinearModel.sure``.

    Args:
        nobs (int): Number of observations.
        num_absorbed (int): Number of absorbed fixed-effect levels.
        params (sparse column): Parameter vector for outcome ``i``,
            shape ``(p, 1)``.
        endog (array-like): Full (unabsorbed) dependent variable.
        exog (array-like or sparse): Full (unabsorbed) design matrix.
        exog_absorb_instrumented: Absorbed + instrumented design matrix.
        rsquared_within_raw (float): Within R² from the absorbed regression.
        weights (array-like or None): Final observation weights (may be
            FGLS-updated).
        do_fgls (bool): True if FGLS was performed.
        is_weighted (bool): True if original model used weights.
        has_implicit_constant (bool): True if model has an implicit constant.
        has_intercept (bool): True if model includes an explicit intercept.
        is_absorb (bool): True if fixed effects were absorbed.
        absorb_info: ``AbsorbInfo`` object for outcome ``i``, or ``None``.
        sigma (ndarray or None): GLS covariance matrix.
        sigma_inv (ndarray or None): Pre-computed GLS inverse.
        _time (float, optional): Elapsed-time baseline.
        debug (bool): Verbose output.

    Returns:
        tuple of 16 elements (in order):
            df_resid, df_model, rsquared, rsquared_adj, wssr, ssr, wsst,
            sst, uncentered_tss, resid, wresid, fittedvalues, llf,
            absorbed_y_baselines, rsquared_within, rsquared_between.
    """

    is_gls = sigma is not None or sigma_inv is not None
    assert sigma is None or sigma_inv is None

    # --------------------------------------
    # fit the predicted values and residuals
    (y_hat, y_hat_instrumented, resid, resid_instrumented, ssr, wssr) \
        = lin_mod_fit_predicted_values_and_residuals(
        params, endog, exog, exog_absorb_instrumented, weights, debug=debug,
        _time=_time)

    # -------------------------
    # get df_resid and df_model
    df_resid, df_model = SparseVarianceCovariance2.get_df_resid_model(
        nobs, exog.shape[1], num_absorbed, has_implicit_constant, has_intercept,
        has_intercept, debug=debug)

    # ----------------------------------------------
    # Need to add back absorbed values to prediction
    if is_absorb:

        y_hat, resid, absorbed_y_baselines \
            = lin_mod_add_absorbed_fe_to_y_hat(
            params.toarray(), absorb_info.endog_absorb_means, absorb_info.exog_absorb_means,
            absorb_info.group_to_row_lists, y_hat, resid, debug=debug, _time=_time)

        rsquared_between = absorb_info.rsquared_between
        rsquared_within = rsquared_within_raw * (1 - rsquared_between)

    else:
        absorbed_y_baselines, rsquared_within, rsquared_between = None, None, None

    # -------------------------------
    # Get sum of squares and variants
    if is_gls:
        wssr = get_gls_sum_squares(resid, sigma, sigma_inv, demean=False)
        sst, wsst, uncentered_tss = lin_mod_get_sum_of_squares_gls(endog, sigma, sigma_inv, debug=debug, _time=_time)
    else:
        wssr = get_wtd_sum_squares(resid, weights, demean=False)
        sst, wsst, uncentered_tss = lin_mod_get_sum_of_squares(endog, weights, debug=debug, _time=_time)

    # ----------------------
    # Compute log likelihood
    if is_gls:
        llf = loglike_internal_gls(wssr, nobs, sigma=sigma, sigma_inv=sigma_inv)#, sigma_inv, wssr/nobs)
    else:
        llf = loglike_internal(resid, nobs, weights=weights)

    # -----------
    # Compute R^2
    # If no intercept, total sum of squares is sum(endog**2), not sum([endog-\bar endog]**2)
    has_constant = (has_intercept or has_implicit_constant)
    rsquared, rsquared_adj = lin_mod_get_rsquared(
        wssr, wsst, uncentered_tss, has_constant, nobs, df_resid,
        const_number=has_constant
    )

    wresid = resid * np.sqrt(weights) if (do_fgls or is_weighted) else resid
    fittedvalues = y_hat

    return \
        df_resid, df_model, rsquared, rsquared_adj, wssr, ssr, wsst, sst, uncentered_tss, \
            resid, wresid, fittedvalues, llf, \
            absorbed_y_baselines, rsquared_within, rsquared_between


def loglike_internal(resid, nobs, weights=None):
    """Compute the Gaussian log-likelihood for OLS / WLS.

    Uses the MLE formula:

        ℓ = −n/2 [log(2π/n) + 1 + log(WSSR)] + ½ Σ log(wᵢ)

    where WSSR = Σ wᵢ ûᵢ².  The weight term adjusts for the Jacobian of
    the variance-stabilising transformation.

    Args:
        resid (ndarray): Residual vector, shape ``(n,)``.
        nobs (int): Number of observations.
        weights (array-like, optional): Observation weights.

    Returns:
        float: Gaussian log-likelihood evaluated at the MLE scale estimate.
    """
    wssr = (resid ** 2).sum() if weights is None else (resid ** 2 * weights).sum()
    llf = -nobs / 2 * (np.log(2 * np.pi / nobs) + 1 + np.log(wssr))
    if weights is not None:
        llf += .5 * np.log(weights).sum()
    return llf


def loglike_internal_gls(wssr, nobs, scale_mle=None, sigma=None, sigma_inv=None):
    """Compute the GLS Gaussian log-likelihood.

    Evaluates the concentrated log-likelihood for a GLS model:

        ℓ = −n/2 + ½ log|Σ⁻¹ / σ²_MLE| − (n/2) log(2π)

    where σ²_MLE = WSSR / n is the MLE error variance.  Either ``sigma`` or
    ``sigma_inv`` must be supplied to compute the log-determinant term.

    Args:
        wssr (float): Generalised SSR: û'Σ⁻¹û.
        nobs (int): Number of observations.
        scale_mle (float, optional): MLE scale σ²_MLE.  Defaults to
            ``wssr / nobs``.
        sigma (ndarray, optional): Error covariance matrix.  Used when
            ``sigma_inv`` is not available.
        sigma_inv (ndarray, optional): Pre-computed inverse of ``sigma``.

    Returns:
        float: GLS Gaussian log-likelihood.
    """

    if scale_mle is None:
        scale_mle = wssr / nobs

    if sigma_inv is None:
        term2 = -0.5 * (np.linalg.slogdet(sigma * scale_mle)[1])
    else:
        term2 = 0.5 * (np.linalg.slogdet(sigma_inv / scale_mle)[1])

    llf = (
            -0.5 * nobs
            + term2
            - (nobs / 2) * np.log(2 * np.pi)
    )
    return llf
