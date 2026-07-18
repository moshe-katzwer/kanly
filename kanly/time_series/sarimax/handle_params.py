from __future__ import absolute_import, print_function

import numpy as np


# TODO REWRITE!!!
def constrain_stationary_univariate(unconstrained):
    """
    Transform unconstrained parameters used by the optimizer to constrained
    parameters used in likelihood evaluation

    Parameters
    ----------
    unconstrained : ndarray
        Unconstrained parameters used by the optimizer, to be transformed to
        stationary coefficients of, e.g., an autoregressive or moving average
        component.

    Returns
    -------
    constrained : ndarray
        Constrained parameters of, e.g., an autoregressive or moving average
        component, to be transformed to arbitrary parameters used by the
        optimizer.

    References
    ----------
    .. [*] Monahan, John F. 1984.
       "A Note on Enforcing Stationarity in
       Autoregressive-moving Average Models."
       Biometrika 71 (2) (August 1): 403-404.
    """

    n = unconstrained.shape[0]
    y = np.zeros((n, n), dtype=unconstrained.dtype)
    r = unconstrained / ((1 + unconstrained ** 2) ** 0.5)
    for k in range(n):
        for i in range(k):
            y[k, i] = y[k - 1, i] + r[k] * y[k - 1, k - i - 1]
        y[k, k] = r[k]
    return -y[n - 1, :]


def constrain_stationary_univariate_temp(x):
    """Handle empty arrays before applying the stationarity transform.

    Args:
        x: Candidate unconstrained parameter vector.

    Returns:
        Stationarity-transformed parameter vector.
    """
    return constrain_stationary_univariate(np.array(x))


def handle_params(ar, ma, sar, sma, enforce_stationarity, enforce_invertibility):
    """
    Apply stationarity and invertibility transforms to AR and MA parameters.

    Args:
        ar: Autoregressive coefficient vector.
        ma: Moving-average coefficient vector.
        sar: Seasonal autoregressive coefficient vector.
        sma: Seasonal moving-average coefficient vector.
        enforce_stationarity: Whether to transform AR parameters into the stationary region.
        enforce_invertibility: Whether to transform MA parameters into the invertible region.

    Returns:
        4-tuple ``(ar, ma, sar, sma)`` of 1-D arrays.  When
        ``enforce_stationarity`` is True the AR and SAR blocks are transformed
        via the Monahan bijection so the roots of their lag polynomials all lie
        outside the unit circle.  When ``enforce_invertibility`` is True the MA
        and SMA blocks are similarly constrained (with an additional sign flip
        to match the likelihood convention used in this SARIMAX implementation).
        Blocks that are not constrained are returned unchanged.
    """
    if enforce_stationarity:
        # The same Monahan transform is applied to seasonal and nonseasonal AR
        # blocks independently.
        if len(ar):
            ar = constrain_stationary_univariate_temp(ar)
        if len(sar):
            sar = constrain_stationary_univariate_temp(sar)
    if enforce_invertibility:
        # MA signs follow the likelihood convention used elsewhere in this
        # SARIMAX implementation.
        if len(ma):
            ma = -constrain_stationary_univariate_temp(ma)
        if len(sma):
            sma = -constrain_stationary_univariate_temp(sma)
    return ar, ma, sar, sma
