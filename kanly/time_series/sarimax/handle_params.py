"""Validation and repair of SARIMAX starting coefficients.

The Hannan--Rissanen estimator returns coefficients in the model's parameter
space.  These helpers therefore check those coefficients directly instead of
treating them as unconstrained optimizer coordinates.  Valid estimates are
left alone; invalid estimates are moved into a conservative, guaranteed-safe
region before optimization begins.
"""

import numpy as np


def _lag_polynomial(params, lags, *, moving_average):
    """Construct an AR or MA lag polynomial in ascending powers of the lag."""
    params = np.asarray(params, dtype=float)
    lags = np.asarray(lags)

    if params.ndim != 1 or lags.ndim != 1:
        raise ValueError("Starting parameters and lags must be one-dimensional")
    if params.size != lags.size:
        raise ValueError("Starting parameters and lags must have equal lengths")
    if not np.all(np.isfinite(params)):
        raise ValueError("Starting parameters must be finite")
    if params.size == 0:
        return np.ones(1)
    if not np.issubdtype(lags.dtype, np.integer):
        raise ValueError("Lag positions must be integers")
    if np.any(lags < 1) or np.unique(lags).size != lags.size:
        raise ValueError("Lag positions must be positive and unique")

    polynomial = np.zeros(int(lags.max()) + 1)
    polynomial[0] = 1.0
    polynomial[lags] = params if moving_average else -params
    return polynomial


def _is_admissible(params, lags, *, moving_average, tolerance):
    """Return whether all roots of a model lag polynomial exceed unit modulus."""
    polynomial = _lag_polynomial(params, lags, moving_average=moving_average)
    roots = np.roots(polynomial[::-1])
    return bool(np.all(np.abs(roots) > 1.0 + tolerance))


def _repair_start(params, lags, *, moving_average, margin, tolerance):
    """Keep a valid start or shrink an invalid one into a stable region."""
    params = np.asarray(params, dtype=float)

    if _is_admissible(
            params, lags, moving_average=moving_average, tolerance=tolerance):
        return params.copy()

    coefficient_size = np.sum(np.abs(params))
    if coefficient_size == 0.0:
        return params.copy()

    # If sum(abs(params)) < 1, then for every |z| <= 1 the nonconstant
    # portion of the lag polynomial has modulus below one.  It therefore
    # cannot cancel the constant term, so all roots lie outside the unit disk.
    return params * (margin / coefficient_size)


def stabilize_arma_starts(
        ar, ma, sar, sma, *,
        ar_lags, ma_lags, sar_lags, sma_lags,
        enforce_stationarity, enforce_invertibility,
        margin=0.98, tolerance=1e-8):
    """Repair inadmissible Hannan--Rissanen ARMA starting coefficients.

    Parameters are already model coefficients, not unconstrained optimizer
    variables.  Consequently, an admissible block is returned unchanged.
    When repair is requested and a block is inadmissible, all coefficients in
    that block are scaled proportionally until their absolute sum is
    ``margin``.  This is a conservative sufficient condition for stationarity
    or invertibility and works for nonconsecutive lag specifications.

    Seasonal lag positions are expressed in seasonal units: for period 12,
    seasonal lag 2 is passed as ``2`` rather than absolute lag ``24``.
    """
    if not 0.0 < margin < 1.0:
        raise ValueError("margin must be strictly between zero and one")
    if tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative")

    ar = np.asarray(ar, dtype=float)
    ma = np.asarray(ma, dtype=float)
    sar = np.asarray(sar, dtype=float)
    sma = np.asarray(sma, dtype=float)

    if enforce_stationarity:
        ar = _repair_start(
            ar, ar_lags, moving_average=False,
            margin=margin, tolerance=tolerance)
        sar = _repair_start(
            sar, sar_lags, moving_average=False,
            margin=margin, tolerance=tolerance)

    if enforce_invertibility:
        ma = _repair_start(
            ma, ma_lags, moving_average=True,
            margin=margin, tolerance=tolerance)
        sma = _repair_start(
            sma, sma_lags, moving_average=True,
            margin=margin, tolerance=tolerance)

    return ar, ma, sar, sma
