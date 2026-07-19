"""Very fast / memory-efficient least squares via ``scipy.sparse.linalg.lsmr``.

This module provides a thin wrapper around the LSMR iterative sparse solver
for obtaining regression coefficients when inference (standard errors,
covariance matrices, eigenvalues) is not required.  It is intended for very
large design matrices where forming and inverting X'X would be prohibitive.

Key differences from the full ``lm_internal`` pipeline:

- No matrix inversion — parameters are estimated iteratively; no
  normalised covariance matrix or eigenvalues are returned.
- No IV / absorb support — instrumental variables and fixed-effects
  absorption are not implemented here.
- Use this path (via ``SparseLinearModel.lm_fast`` or
  ``SparseLinearModel.fit_lsmr``) when you need coefficients quickly for
  large sparse problems.  Use the standard ``lm`` / ``fit`` path whenever
  inference, F-tests, or R² are needed.

See Also:
    ``SparseLinearModel.LM_fast``, ``SparseLinearModel.lm_fast``,
    ``SparseLinearModel.fit_lsmr``
"""
from __future__ import absolute_import, print_function

import time

import numpy as np
from scipy.sparse import csc_matrix, isspmatrix
from scipy.sparse.linalg import lsmr

from kanly.utils.linalg_utils import csc_matrix_by_column_array_broadcast


def fit_lsmr_internal(endog, exog, instruments=None, weights=None, bootstrap_weights=None, is_endog_regressor=None,
                      debug=False, _time=None, **kwargs):
    """Solve a (possibly weighted) least-squares system using the LSMR algorithm.

    Solves ``min_{beta} || sqrt(w) * (X beta - y) ||_2`` iteratively without
    forming or inverting X'WX.  Weight pre-multiplication is done by scaling
    both ``y`` and ``X`` column-wise by ``sqrt(w)`` before calling ``lsmr``.

    .. note::
        Instrumental variables (``instruments`` argument) are **not yet
        supported**.  Passing a non-``None`` value raises an ``Exception``.

    Args:
        endog (array-like or csc_matrix): Dependent variable vector of
            shape ``(n,)`` or ``(n, 1)``.  Converted to CSC sparse if dense.
        exog (array-like or csc_matrix): Design matrix of shape ``(n, p)``.
            Converted to CSC sparse if dense.
        instruments (None): Reserved for future IV support.  Must be ``None``.
        weights (array-like, optional): Non-negative weight vector of
            length ``n``.  The effective system solved is
            ``sqrt(weights) * X @ beta ≈ sqrt(weights) * y``.
        bootstrap_weights (array-like, optional): Additional multiplicative
            weights (e.g. block-bootstrap counts).  Combined with ``weights``
            via element-wise multiplication of square roots.
        is_endog_regressor (array-like of bool, optional): Unused in the
            current implementation (reserved for IV support).
        debug (bool): If ``True``, prints a start/end message with elapsed
            time.
        _time (float, optional): ``time.time()`` snapshot for elapsed-time
            reporting.  Created internally when ``None``.
        **kwargs: Additional keyword arguments forwarded directly to
            ``scipy.sparse.linalg.lsmr`` (e.g. ``damp``, ``atol``, ``btol``,
            ``maxiter``).

    Returns:
        dict: A dictionary with the following keys mirroring the ``lsmr``
            output tuple:
            - ``'x'`` (ndarray): Estimated coefficient vector of length ``p``.
            - ``'istop'`` (int): LSMR termination condition code.
            - ``'itn'`` (int): Number of iterations performed.
            - ``'normr'`` (float): Norm of the residual.
            - ``'normar'`` (float): Norm of the adjoint residual.
            - ``'norma'`` (float): Frobenius norm of ``X``.
            - ``'conda'`` (float): Condition number estimate of ``X``.
            - ``'normx'`` (float): Norm of the solution vector.
            - ``'fit_elapsed'`` (float): Wall-clock seconds from ``_time``
              to completion.

    Raises:
        Exception: If ``instruments`` is not ``None`` (IV not yet supported).
    """

    if _time is None:
        _time = time.time()

    if not isspmatrix(endog):
        endog = csc_matrix(endog)
    if not isspmatrix(exog):
        exog = csc_matrix(exog)
    if instruments is not None:
        raise Exception("Instruments not yet supported by fast lm!")

        # WIP IV block — not yet implemented for lsmr path:
        # if not isspmatrix(instruments):
        #     instruments = csc_matrix(instruments)
        # if is_endog_regressor is None:
        #     is_endog_regressor = np.array([True] * exog.shape[1])

    if weights is not None or bootstrap_weights is not None:

        # Pre-multiply endog and exog by sqrt(weights) so that lsmr solves
        # the weighted normal equations without forming any matrix product.
        rt_wts = 1.0
        if weights is not None:
            rt_wts = np.sqrt(weights)
        if bootstrap_weights is not None:
            # Bootstrap weights are multiplicative on top of any standard weights.
            rt_wts *= np.asarray(np.sqrt(bootstrap_weights))

        yw = csc_matrix_by_column_array_broadcast(endog, rt_wts)
        Xw = csc_matrix_by_column_array_broadcast(exog, rt_wts)

        #if instruments is not None:
        #    Iw = csc_matrix_by_column_array_broadcast(instruments, rt_wts)

    else:
        yw, Xw, Iw = endog, exog, instruments

    if instruments is not None:
        raise Exception("Instruments not yet supported by fast lm!")
        # Remaining WIP IV projection block omitted — see commented code above.

        # WIP!!!!
        # from tqdm import tqdm
        #
        # print('Projecting exog onto instruments...')
        # n_Z = instruments.shape[1]
        # pis = []
        # for j in tqdm(range(exog.shape[1])):
        #     if is_endog_regressor[j]:
        #         pi = lsmr(Iw, Xw.getcol(j).toarray().flatten())[0]
        #     else:
        #         pi = np.zeros((n_Z, 1))
        #         pi[j] = 1.0
        #     pis.append(csc_matrix(pi))
        # Pi = hstack(pis)
        # Xw = Iw.dot(Pi)
        #
        # # v = spsolve(Iw.transpose().dot(Iw).tocsc(), Iw.transpose().dot(exog).tocsc())
        # # Xw = Iw.dot(v).tocsc()

    yw = yw.toarray().ravel()

    if debug:
        print('Solving for params...', end='')
    fit = lsmr(Xw, yw, **kwargs)
    if debug:
        print(f"Done! {'%.2f' % (time.time()-_time)}s")

    #fit = spsolve(Xw.transpose().dot(Xw).tocsc(), Xw.transpose().dot(yw).tocsc())

    result = dict(zip(
        ['x', 'istop', 'itn', 'normr', 'normar', 'norma', 'conda', 'normx'],
        fit
    ))
    result['fit_elapsed'] = time.time() - _time

    return result
