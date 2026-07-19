"""IV first stage: project endogenous regressors onto instruments.

This module implements the first stage of Two-Stage Least Squares (2SLS):

    π̂ = (Z'WZ)^{-1} Z'WX   (the first-stage coefficient matrix)

    X̂ = Z π̂               (the instrumented / fitted exogenous regressors)

For each column of ``X``:
- If the column is classified as **endogenous**, it is replaced by its
  projection onto ``Z`` (i.e. ``Z π̂_k``).
- If the column is **exogenous**, it is passed through unchanged.

Optionally, first-stage residuals ``X - X̂`` (raised to integer powers) can
be appended to the instrumented design matrix for the Residual Inclusion
(control-function) approach to IV estimation.

Both dense (ndarray) and sparse (CSC) matrices are supported throughout.

Reference:
    Hausman, J. A. (1978). Specification tests in econometrics.
    *Econometrica*, 46(6), 1251–1271.
"""

from __future__ import absolute_import, print_function

import time

import numpy as np
from scipy.sparse import csc_matrix, hstack as sphstack, isspmatrix
from scipy.sparse.linalg import norm as sp_norm

from kanly.regression.linear_models.constants import (
    DEFAULT_LM_INVERSE_METHOD, DEFAULT_LM_SCALE_DESIGN_MATRIX, DEFAULT_LM_COMPUTE_EIGENVALUES_INSTRUMENTS)


from kanly.utils.linalg_utils import (
    get_normalized_cov_params, csc_matrix_by_column_array_broadcast, DEFAULT_DENSE_THRESHOLD_MB)


class InstrumentInfoInternal(object):
    """Container for first-stage IV estimation results.

    Stores all information produced by the IV first-stage projection that
    is needed by ``lm_internal`` to complete second-stage estimation and
    by ``SparseLinearRegressionResults`` to report IV diagnostics.

    Attributes:
        instrument_normalized_cov_params: The normalised covariance matrix
            ``(Z'WZ)^{-1}`` of the instrument Gram matrix (a sparse or dense
            ``(q, q)`` array where ``q`` is the number of instruments).
        exog_instrumented: The instrumented design matrix ``X̂ = Z π̂``,
            with endogenous columns replaced by their first-stage fitted
            values and exogenous columns passed through unchanged.
            Shape ``(n, p)`` — same as ``exog``, or wider when residual
            inclusion is used.
        instrument_params: First-stage coefficient matrix ``π̂``, shape
            ``(q, p)``.  Column ``k`` holds the OLS coefficients of regressor
            ``k`` on the full instrument set.
        exog_col_map (list of tuple): Maps columns of ``exog_instrumented``
            back to original ``exog`` columns.  Each entry is a 1- or 2-tuple
            ``(col_index,)`` or ``(col_index, residual_power)`` for residual-
            inclusion augmented columns.
        condition_number (float or None): Condition number of the instrument
            Gram matrix ``Z'WZ``.  ``None`` when eigenvalue computation is
            disabled.
        eigenvals (ndarray or None): Eigenvalues of ``Z'WZ`` in descending
            order.  ``None`` when eigenvalue computation is disabled.
    """

    def __init__(self, instrument_normalized_cov_params, exog_instrumented, instrument_params, exog_col_map,
                 condition_number, eigenvals):
        """Initialise the internal instrument-info container.

        Args:
            instrument_normalized_cov_params: ``(Z'WZ)^{-1}``, shape ``(q, q)``.
            exog_instrumented: Instrumented design matrix, shape ``(n, p)`` or
                ``(n, p + r)`` with residual-inclusion columns.
            instrument_params: First-stage coefficient matrix, shape ``(q, p)``.
            exog_col_map (list of tuple): Column provenance map (see class docs).
            condition_number (float or None): Condition number of ``Z'WZ``.
            eigenvals (ndarray or None): Eigenvalues of ``Z'WZ``.
        """
        self.instrument_normalized_cov_params = instrument_normalized_cov_params
        self.exog_instrumented = exog_instrumented
        self.instrument_params = instrument_params
        self.exog_col_map = exog_col_map
        self.condition_number = condition_number
        self.eigenvals = eigenvals


def do_residual_inclusion2(exog, wexog_instrumented, is_endog_regressors, order=1):
    """Augment the instrumented design matrix with first-stage residual powers.

    Implements the control-function (residual inclusion) approach: the
    residuals ``X_endog − X̂_endog`` (and their integer powers up to
    ``order``) are appended as additional columns to ``wexog_instrumented``.
    Columns whose residual norm is below 1e-6 are treated as perfectly
    instrumented and excluded from augmentation.

    Args:
        exog (ndarray or csc_matrix, shape (n, p)): Original design matrix.
        wexog_instrumented (ndarray or csc_matrix, shape (n, p)): Instrumented
            design matrix from the first-stage projection.
        is_endog_regressors (array-like of bool, length p): Mask identifying
            endogenous regressor columns.
        order (int): Maximum power of the residuals to include.  With
            ``order=1`` only the residuals themselves are added; with
            ``order=2`` both residuals and their squares are included.

    Returns:
        tuple:
            - **wexog_instrumented** (ndarray or csc_matrix): Augmented design
              matrix of shape ``(n, p + r)`` where ``r`` depends on the number
              of non-trivial endogenous residual columns and ``order``.
            - **exog_col_map** (list of tuple): Updated column provenance map.
              Original columns are encoded as 1-tuples ``(k,)``; residual
              columns are encoded as 2-tuples ``(k, power)``.
    """
    endog_regressors_idx = np.arange(exog.shape[1])[is_endog_regressors]
    exog_resid = exog[:, is_endog_regressors] - wexog_instrumented[:, is_endog_regressors]

    if isspmatrix(exog):
        assert isspmatrix(wexog_instrumented)


        valid_cols = sp_norm(exog_resid, axis=0) > 1e-6
        endog_regressors_idx = endog_regressors_idx[valid_cols]

        exog_resid = exog_resid[:, valid_cols]

        exog_resid = sphstack([exog_resid.power(o) if o > 1 else exog_resid
                               for o in range(1, order + 1)])

        wexog_instrumented = sphstack((wexog_instrumented, exog_resid))

    else:
        assert not isspmatrix(wexog_instrumented)

        valid_cols = np.linalg.norm(exog_resid, axis=0) > 1e-6
        endog_regressors_idx = endog_regressors_idx[valid_cols]

        exog_resid = exog_resid[:, valid_cols]

        exog_resid = np.hstack([exog_resid ** o if o > 1 else exog_resid
                                for o in range(1, order + 1)])

        wexog_instrumented = np.hstack((wexog_instrumented, exog_resid))


    exog_col_map = [(i,) for i in range(exog.shape[1])] \
                   + [(i, o) for o in range(1, order + 1) for i in endog_regressors_idx]

    return wexog_instrumented, exog_col_map


def _get_endog_regressors(exog_col_names, instrument_col_names):
    """Identify endogenous regressor columns by exclusion from the instrument set.

    A regressor column is considered endogenous if its name does **not**
    appear among the instrument column names (i.e. it is excluded from the
    excluded-instrument set and therefore suspected of endogeneity).

    Args:
        exog_col_names (list of str): Column names of the design matrix.
        instrument_col_names (list of str): Column names of the instrument
            matrix (includes both included and excluded instruments).

    Returns:
        list of bool: Boolean mask of length ``len(exog_col_names)``; ``True``
            for columns not found in ``instrument_col_names``.
    """
    return [c not in instrument_col_names for c in exog_col_names]


def project_exog_col_k_onto_instruments(k, exog, instruments, winstruments, winstruments_normalized_cov_params,
                                        is_endog, force_iv_projection, debug=False):
    """Project a single design-matrix column onto the instrument space.

    For an endogenous column (or when ``force_iv_projection=True``), this
    computes π̂_k = (Z'WZ)^{-1} Z'W x_k and replaces the column with
    Z π̂_k (its projection onto the column space of Z).

    For an exogenous column, the original column is returned unchanged and
    π̂_k is set to the corresponding canonical basis vector (one-hot).

    Args:
        k (int): Zero-based index of the column in ``exog`` to process.
        exog (ndarray or csc_matrix, shape (n, p)): Full design matrix.
        instruments (ndarray or csc_matrix, shape (n, q)): Instrument matrix.
        winstruments (ndarray or csc_matrix, shape (n, q)): Instruments
            pre-multiplied by sqrt(W); used to form Z'W products.
        winstruments_normalized_cov_params: Pre-computed ``(Z'WZ)^{-1}``,
            shape ``(q, q)``.
        is_endog (bool): ``True`` if column ``k`` is endogenous.
        force_iv_projection (bool): If ``True``, project all columns
            regardless of ``is_endog``.
        debug (bool): If ``True``, prints a per-column progress message.

    Returns:
        tuple:
            - **pi_k** (ndarray, shape (q, 1) or (q,)): First-stage
              coefficient vector for column ``k``.
            - **exog_k_projected** (ndarray or sparse column): Projected
              (instrumented) column of length ``n``.
    """
    if debug:
        print(f"\tRegressor column '{k}' is ", end='')

    exog_sparse = isspmatrix(exog)
    assert (isspmatrix(exog) and isspmatrix(instruments)) or (not isspmatrix(exog) and not isspmatrix(instruments))
    is_sparse = isspmatrix(exog)


    # Construct \hat Pi = inv(Z'Z) Z' exog for endogenous exog columns
    # We only do solve for coefficients of endogeneous regressors
    # exogenous regressors we just copy later, 1:1
    if is_endog or force_iv_projection:
        if debug:
            print("endogenous...projecting on instruments...")

        if instruments.shape[1] == 1:

            if is_sparse:
                pi_k = exog.getcol(k).transpose().dot(winstruments).toarray().item()
                pi_k *= float(winstruments_normalized_cov_params[0, 0])
                exog_k_projected = instruments.dot(csc_matrix([[pi_k]]))
                pi_k = np.array([pi_k])
            else:
                pi_k = np.dot(exog[:,k], winstruments).item()
                pi_k *= float(winstruments_normalized_cov_params[0, 0])
                exog_k_projected = instruments * pi_k
                pi_k = np.array([pi_k])

        else:
            if is_sparse:
                pi_k = winstruments_normalized_cov_params.dot(
                    (exog.getcol(k).transpose().dot(winstruments)).transpose())
                exog_k_projected = instruments.dot(pi_k)
                pi_k = pi_k.toarray()
            else:
                pi_k = winstruments_normalized_cov_params.dot(np.dot(exog[:, k], winstruments))
                exog_k_projected = instruments.dot(pi_k)

    else:
        if debug:
            print("exogenous...copying over...")

        exog_k_projected = exog.getcol(k) if exog_sparse else exog[:, k]
        pi_k = np.zeros((instruments.shape[1], 1))
        pi_k[k] = 1.0

    return pi_k, exog_k_projected


# In linear_models, weighting happens before this step
# in GLM, weighting happens here for prediction
def iv_first_stage2(exog, instruments, is_endog_regressor=None, debug=False, _time=None, residual_inclusion=False,
                    weights=None, residual_inclusion_order=1, force_iv_projection=False,
                    scale_design_matrix=DEFAULT_LM_SCALE_DESIGN_MATRIX,
                    compute_eigenvalues_instruments=DEFAULT_LM_COMPUTE_EIGENVALUES_INSTRUMENTS,
                    dense_threshold_mb=DEFAULT_DENSE_THRESHOLD_MB,
                    inverse_method=DEFAULT_LM_INVERSE_METHOD
) -> InstrumentInfoInternal:
    """Run the IV first-stage projection for all regressor columns.

    Computes the first-stage coefficient matrix:

        π̂ = (Z'WZ)^{-1} Z'WX

    and replaces each endogenous column of ``exog`` with its projection onto
    the column space of ``instruments``.  Exogenous columns are passed
    through unchanged.  Optionally appends first-stage residual powers for
    the control-function approach.

    Originally, parameters named *exog_col_names* and *instrument_col_names*
    were part of the signature (hence the legacy docstring below); they have
    since been removed.

    Legacy parameter note:
        - ``exog``: pre-multiplied as ``X * diag(sqrt(weights))`` upstream in
          some call paths; the caller is responsible for weight application.
        - ``instruments``: similarly ``Z * diag(sqrt(weights))`` in legacy paths.

    Args:
        exog (ndarray or csc_matrix, shape (n, p)): Design matrix.
        instruments (ndarray or csc_matrix, shape (n, q)): Instrument matrix.
        is_endog_regressor (array-like of bool, length p, optional): Mask
            identifying endogenous columns.  Defaults to all ``True``.
        debug (bool): Verbose timing output.
        _time (float, optional): ``time.time()`` baseline for elapsed logging.
        residual_inclusion (bool): If ``True``, append first-stage residuals
            to the instrumented matrix (control-function approach).
        weights (array-like, optional): Observation weights.
        residual_inclusion_order (int): Maximum power of residuals to include
            when ``residual_inclusion=True``.
        force_iv_projection (bool): Project all columns, even exogenous ones.
        scale_design_matrix (bool): Scale instruments before inverting their
            Gram matrix (improves numerical stability).
        compute_eigenvalues_instruments (bool): Compute eigenvalues of Z'WZ.
        dense_threshold_mb (float): Memory threshold for sparse/dense decision.
        inverse_method: Inversion algorithm selector.

    Returns:
        InstrumentInfoInternal: Container with the normalised covariance
            matrix of the instruments, the instrumented design matrix,
            first-stage coefficient matrix, column map, and optionally
            eigenvalues and condition number.
    """
    if _time is None:
        _time = time.time()
    if debug:
        print("\nProjecting endogenous regressors with instruments...")

    if is_endog_regressor is None:
        is_endog_regressor = [True] * exog.shape[1]

    assert (
        np.all([isspmatrix(z) for z in (exog, instruments)])
        or np.all([not isspmatrix(z) for z in (exog, instruments)])
    )
    is_sparse = isspmatrix(exog)

    if debug:
        print('\tBuilding instrument normalized cov params...', end='')
    winstruments, winstruments_normalized_cov_params, eigenvals, condition_number \
        = get_instrument_ncp(instruments, weights, debug=debug,
                             compute_eigenvalues_instruments=compute_eigenvalues_instruments,
                             dense_threshold_mb=dense_threshold_mb, scale_design_matrix=scale_design_matrix,
                             inverse_method=inverse_method)
    if debug:
        print('done (%.1fs)' % (time.time() - _time))

    wexog_instrumented_list = []
    instrument_params_list = []

    for k, is_endog in enumerate(is_endog_regressor):

        pi_k, exog_k_projected = project_exog_col_k_onto_instruments(
            k, exog, instruments, winstruments, winstruments_normalized_cov_params, is_endog, force_iv_projection,
            debug=debug)
        wexog_instrumented_list.append(exog_k_projected)
        instrument_params_list.append(pi_k)

    if is_sparse:
        instrument_params = csc_matrix(np.hstack(instrument_params_list))
        wexog_instrumented = sphstack(wexog_instrumented_list).tocsc(copy=False)
    else:
        instrument_params = np.column_stack(instrument_params_list)
        wexog_instrumented = np.column_stack(wexog_instrumented_list)

    # if is_sparse:
    #     temp = wexog_instrumented.toarray()
    # else:
    #     temp = wexog_instrumented
    # print('X_hat_code  = ', temp[:10])

    if residual_inclusion:
        wexog_instrumented, exog_col_map = do_residual_inclusion2(
            exog, wexog_instrumented, is_endog_regressor, order=residual_inclusion_order)
    else:
        exog_col_map = [(i,) for i in range(exog.shape[1])]

    if debug:
        print("%.3f s" % (time.time() - _time))

    return InstrumentInfoInternal(winstruments_normalized_cov_params, wexog_instrumented, instrument_params, exog_col_map,
                                  condition_number, eigenvals)


def get_instrument_ncp(instruments, weights=None, debug=False,
                       compute_eigenvalues_instruments=DEFAULT_LM_COMPUTE_EIGENVALUES_INSTRUMENTS,
                       dense_threshold_mb=DEFAULT_DENSE_THRESHOLD_MB,
                       scale_design_matrix=DEFAULT_LM_SCALE_DESIGN_MATRIX,
                       inverse_method=DEFAULT_LM_INVERSE_METHOD):
    """Compute the normalised covariance matrix of the instrument Gram matrix.

    Optionally pre-multiplies the instruments by ``sqrt(weights)`` before
    computing ``(Z'Z)^{-1}`` (or ``(Z'WZ)^{-1}`` when weights are supplied).
    When ``weights`` are provided, the weighted instruments are scaled a
    second time by ``sqrt(weights)`` so that subsequent products ``Z'W X``
    can be formed as ``winstruments' * exog``.

    Args:
        instruments (ndarray or csc_matrix, shape (n, q)): Instrument matrix.
        weights (array-like, optional): Non-negative weight vector of length
            ``n``.  When ``None``, computes the unweighted Gram inverse.
        debug (bool): If ``True``, prints timing messages.
        compute_eigenvalues_instruments (bool): Whether to compute eigenvalues
            of ``Z'WZ`` (used for diagnostics and condition number).
        dense_threshold_mb (float): Memory threshold in MB; matrices whose
            dense representation exceeds this are kept sparse.
        scale_design_matrix (bool): If ``True``, scale ``Z`` column-wise
            before inversion for improved numerical stability.
        inverse_method: Inversion algorithm selector passed to
            ``get_normalized_cov_params``.  ``None`` selects automatically.

    Returns:
        tuple:
            - **winstruments** (ndarray or csc_matrix, shape (n, q)):
              Instruments pre-multiplied by ``weights`` (if supplied), ready
              for forming ``Z'W X`` products.
            - **winstrument_ncp**: ``(Z'WZ)^{-1}``, shape ``(q, q)``.
            - **eigenvals** (ndarray or None): Eigenvalues of ``Z'WZ``.
            - **condition_number** (float or None): Condition number of ``Z'WZ``.
    """
    if weights is None:
        winstruments = instruments
    else:
        winstruments = csc_matrix_by_column_array_broadcast(instruments, weights ** .5)
    winstrument_ncp, eigenvals, condition_number = get_normalized_cov_params(
        winstruments, weights=None, normalize=True, return_eigenvals=True, debug=debug, _time=None,
        compute_eigenvalues=compute_eigenvalues_instruments, ridge_parameters=None,
        dense_threshold_mb=dense_threshold_mb, inverse_method=inverse_method)

    if weights is not None:
        # scale again, since winstruments = instruments .* weights
        winstruments = csc_matrix_by_column_array_broadcast(winstruments, weights ** .5)

    return winstruments, winstrument_ncp, eigenvals, condition_number


def convert_exog_col_map_to_col_names2(exog_col_map, exog_names):
    """Convert the column provenance map to human-readable column names.

    Translates the numeric ``exog_col_map`` (produced by
    :func:`iv_first_stage2` or :func:`do_residual_inclusion2`) into
    string labels suitable for use as DataFrame column headers.

    For standard columns the original name is returned unchanged.  For
    residual-inclusion columns the suffix ``_r(power)`` is appended.

    Args:
        exog_col_map (list of tuple): Each entry is a 1-tuple ``(k,)`` for
            a standard column or a 2-tuple ``(k, power)`` for a residual-
            inclusion column.
        exog_names (list of str): Original column names of the design
            matrix, indexed by the first element of each tuple.

    Returns:
        list of str: Column names corresponding to each entry in
            ``exog_col_map``.
    """
    return [exog_names[m[0]] + ('' if len(m) == 1 else f'_r({m[1]})')
            for m in exog_col_map]


# if __name__ == '__main__':
#
#     import pandas as pd
#     from kanly.api import lm
#     from kanly.regression.linear_models.rewrite.model import SparseLinearModel2
#
#     n = 30
#     np.random.seed(0)
#     df = pd.DataFrame({
#         'e': (e := np.random.randn(n)),
#         'z1': (z1 := np.random.randn(n)),
#         'z2': (z2 := np.random.randn(n)),
#         'x': (x := np.random.randn(n) + 0.3 * z1 - 0.2 * z2 + 0.5 * e),
#         'y': (y := 0.6 * z1 + 1.2 * x + e),
#         'q': np.random.randint(0, 24, n),
#         'c': np.random.randint(0, 3, n),
#         'g': np.random.randint(0, 10, n),
#         'w': 100.0 * np.exp(np.random.rand(n)),
#     })
#
#     model = SparseLinearModel2.build_model_from_formula('y ~ x + q + z1 | z1 + z2 + c + C(g) $ w', df,
#                                                         #absorb=('g',),
#                                                         cov_groups='c')
#
#     iv_info = iv_first_stage2(model.exog, model.instruments, model.is_endog_regressor,
#                               weights=model.weights, residual_inclusion=True, residual_inclusion_order=2)
#     print(iv_info.__dict__)
#     print(convert_exog_col_map_to_col_names2(iv_info.exog_col_map, model.exog_names))
#     # dff = pd.DataFrame(columns=['I', 'x_h', 'z1_h', 'x_ri'], data=iv_info.exog_instrumented.toarray())
#     # dff['x_h(0)'] = (fit:=lm('x ~ z1 + z2 $ w', df)).fittedvalues
#     # dff['x_h(0)'] = (fit:=lm('x ~ z1 + z2 $ w', df)).fittedvalues
#     # dff['z1_ri(0)'] = (fit:=lm('x ~ z1 + z2 $ w', df)).resid
#     # print(dff)
#     #
#     print(model.fit())