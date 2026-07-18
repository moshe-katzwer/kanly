from __future__ import absolute_import, print_function

import numpy as np
from numba import njit
from numba.extending import overload


def _check_types_arma(ar, ma):
    """
    Normalize AR and MA coefficient inputs for innovation recursions.

    Args:
        ar: Autoregressive coefficient vector.
        ma: Moving-average coefficient vector.

    Returns:
        3-tuple ``(ar, ma, dtype)`` where ``ar`` and ``ma`` are 1-D NumPy
        arrays cast to the common promoted dtype, and ``dtype`` is that dtype.
        None inputs are replaced with ``np.zeros(1)`` before promotion.
    """
    if ar is None:
        ar = np.zeros(1)
    else:
        ar = np.asarray(ar)
    if ma is None:
        ma = np.zeros(1)
    else:
        ma = np.asarray(ma)

    dtype = max(ar.dtype, ma.dtype)
    ar = ar.astype(dtype)
    ma = ma.astype(dtype)

    return ar, ma, dtype


def get_acf(ar, ma, nlags=1000, tol=1e-12):
    """
    Autocorrelation Function

    Args:
        ar: Autoregressive coefficient vector.
        ma: Moving-average coefficient vector.
        nlags: Number of autocorrelation lags to return.
        tol: Truncation tolerance passed to the causal representation helper.

    Returns:
        Autocorrelation sequence from lag 0 through ``nlags``.
    """
    acf = get_autocovariance_function(ar, ma, 1.0, nlags, tol)
    acf /= acf[0]

    return acf


def get_autocovariance_function(ar, ma, scale=1.0, nlags=1_000, tol=1e-12):
    """
    Autocovariance Function

    Args:
        ar: Autoregressive coefficient vector.
        ma: Moving-average coefficient vector.
        scale: Innovation variance multiplier.
        nlags: Number of autocovariance lags to return.
        tol: Truncation tolerance passed to the causal representation helper.

    Returns:
        Autocovariance sequence from lag 0 through ``nlags``.
    """
    ar, ma, dtype = _check_types_arma(ar, ma)

    return _get_autocovariance_function_internal(
        ar, ma, dtype=dtype,
        scale=scale, nlags=nlags, tol=tol)


@njit(cache=True)
def _get_autocovariance_function_internal(ar, ma, dtype=np.float64, scale=1.0, nlags=1_000, tol=1e-12):
    """
    Autocovariance Function

    Args:
        ar: Autoregressive coefficient vector.
        ma: Moving-average coefficient vector.
        dtype: NumPy dtype used for allocated arrays.
        scale: Innovation variance multiplier.
        nlags: Number of autocovariance lags to return.
        tol: Truncation tolerance passed to the causal representation helper.

    Returns:
        Autocovariance sequence from lag 0 through ``nlags``.
    """
    if scale is None:
        scale = 1.0

    psi = _get_causal_representation_internal(ar, ma, dtype=dtype, nlags=nlags, tol=tol)

    acf = np.zeros(nlags + 1, dtype=psi.dtype)
    acf[0] = np.dot(psi, psi)
    for h in range(1, nlags + 1):
        acf[h] = np.dot(psi[:-h], psi[h:])
    acf *= scale

    return acf


def get_causal_representation(ar, ma, nlags=1_000, tol=1e-12):
    """
    MA representation of ARMA process

    Return MA representation and auto-correlation function of process, or just MA coefficients
    if `return_acf=False`.

    See Brockwell and Davis 2nd Edition, "Introduction to Time Series and Forecasting",
    section 3.1.

    Args:
        ar: Autoregressive coefficient vector.
        ma: Moving-average coefficient vector.
        nlags: Number of MA representation weights to compute.
        tol: Truncation tolerance for the causal representation.

    Returns:
        MA-infinity representation weights for the ARMA process.
    """
    ar, ma, dtype = _check_types_arma(ar, ma)
    return _get_causal_representation_internal(ar, ma, dtype=dtype, nlags=nlags, tol=tol)


def _get_causal_representation_internal(ar, ma, nlags=1_000, dtype=np.float64, tol=1e-12):
    """
    MA representation of ARMA process

    Args:
        ar: Autoregressive coefficient vector.
        ma: Moving-average coefficient vector.
        nlags: Number of MA representation weights to compute.
        dtype: NumPy dtype used for allocated arrays.
        tol: Truncation tolerance for the causal representation.

    Returns:
        MA-infinity representation weights for the ARMA process.
    """

    p = np.shape(ar)[0]
    q = np.shape(ma)[0]
    m = max(p, q)
    ar_ = np.zeros(m + 1, dtype=dtype)
    ma_ = np.zeros(m + 1, dtype=dtype)

    ar_[1:p + 1] = ar

    ma_[1:q + 1] = ma

    ma_[0] = 1.0

    ar, ma = ar_, ma_

    psi = np.zeros(nlags + 1, dtype=dtype)
    psi[0] = 1.0
    cutoff = nlags + 1
    for j in range(1, nlags + 1):
        psi[j] = ma[j] if j <= q else 0.0
        for k in range(1, len(ar)):
            psi[j] += ar[k] * (psi[j - k] if (j - k) >= 0 else 0.)
        # TODO when to truncate?
        # if j >= 2 * max(p, q) and abs(psi[j]) < tol:
        #     cutoff = j
        #     break

    return psi[:cutoff]


@overload(_get_causal_representation_internal)
def overload__get_causal_representation_internal(ar, ma, nlags=1_000, dtype=np.float64, tol=1e-12):
    """
    Register a numba overload for ``_get_causal_representation_internal``.

    This overload makes ``_get_causal_representation_internal`` callable from
    numba-compiled (``@njit``) code by returning the Python implementation
    function so numba can dispatch to it.

    Args:
        ar: Autoregressive coefficient vector.
        ma: Moving-average coefficient vector.
        nlags: Number of lags to compute or use.
        dtype: NumPy dtype used for allocated arrays.
        tol: Numerical tolerance for truncation or steady-state checks.

    Returns:
        The Python implementation function ``_get_causal_representation_internal``
        for numba dispatch.
    """
    return _get_causal_representation_internal


def get_innovation_coeffs_internal(autocovariances, max_lag, dtype=np.float64, initial_value=None):
    """
    Computes the innovations algorithm coefficients and prediction error variances.
    So `prediction_ar_coeffs[k]` if for predicting X[k] using X[k-1], ..., X[0]
    `prediction_mses[k]` is the variance of the error of that prediction.

    Durbin-Levinson algorithm.

    So `prediction_ar_coeffs[j,k]` is the AR-coefficient on Y(j-(k+1))

    Returns:
        - prediction_ar_coeffs is a list of lists containing the innovation coefficients.
          These are the AR coefficients on the AR best linear predictors using past outcomes.
        - prediction_mses is a list containing the innovation variances.

    Args:
        autocovariances: Autocovariance sequence used by the innovation recursion.
        max_lag: Maximum lag included in the recursion.
        dtype: NumPy dtype used for allocated arrays.
        initial_value: Optional initial innovation variance.
    """

    if initial_value is not None:
        assert initial_value > 0

    prediction_mses = np.zeros(max_lag + 1, dtype=dtype)  # Innovation variance, starting from zero
    prediction_mses[0] = autocovariances[0] if initial_value is None else initial_value
    prediction_ar_coeffs = np.zeros((max_lag + 1, max_lag), dtype=dtype)  # Innovation coefficients, starting from 1

    for n in range(1, max_lag + 1):
        phi_n = prediction_ar_coeffs[n - 1].copy()

        numer = autocovariances[n] - np.sum(phi_n[:n - 1] * np.flip(autocovariances[1:n]))
        denom = prediction_mses[n - 1]

        phi_n[n - 1] = numer / denom

        if n > 1:
            phi_n[:n - 1] -= phi_n[n - 1] * np.flip(prediction_ar_coeffs[n - 1][:n - 1])

        prediction_ar_coeffs[n] = phi_n
        prediction_mses[n] = prediction_mses[n - 1] * (1 - phi_n[n - 1] ** 2)

    return prediction_ar_coeffs, prediction_mses


@overload(get_innovation_coeffs_internal)
def overload_get_innovation_coeffs_internal(autocovariances, max_lag, dtype=np.float64, initial_value=None):
    """
    Register a numba overload for ``get_innovation_coeffs_internal``.

    Analogous to ``overload__get_causal_representation_internal``: makes the
    Durbin-Levinson recursion callable from ``@njit`` contexts by returning
    the Python implementation for numba dispatch.

    Args:
        autocovariances: Autocovariance sequence used by the innovation recursion.
        max_lag: Maximum lag included in the recursion.
        dtype: NumPy dtype used for allocated arrays.
        initial_value: Optional initial recursion value.

    Returns:
        The Python implementation function ``get_innovation_coeffs_internal``
        for numba dispatch.
    """
    return get_innovation_coeffs_internal


@njit(cache=True)
def get_innovation_coeffs_internal_njit_gap(autocovariances, max_lag, dtype=np.float64, initial_value=None):
    """
    Bridge numba dispatch for innovation-coefficient recursion.

    ``@njit`` entry-point that routes calls from compiled code to the overloaded
    ``get_innovation_coeffs_internal``.  The indirection is needed because
    ``get_innovation_coeffs_internal`` itself is not decorated with ``@njit``
    so it cannot be called directly from a compiled context.

    Args:
        autocovariances: Autocovariance sequence used by the innovation recursion.
        max_lag: Maximum lag included in the recursion.
        dtype: NumPy dtype used for allocated arrays.
        initial_value: Optional initial innovation variance.

    Returns:
        2-tuple ``(prediction_ar_coeffs, prediction_mses)`` from the
        Durbin-Levinson recursion, as produced by ``get_innovation_coeffs_internal``.
    """
    return get_innovation_coeffs_internal(autocovariances, max_lag, dtype=dtype, initial_value=initial_value)


def get_innovation_coeffs(autocovariances, max_lag, initial_value=None):
    """
    Compute innovation coefficients from autocovariances.

    Public entry-point for the Durbin-Levinson recursion.  Extracts the dtype
    from the autocovariance array and delegates to the numba bridge
    ``get_innovation_coeffs_internal_njit_gap``, enabling JIT-compiled
    downstream code (e.g. Hannan-Rissanen residual estimation) to reuse the
    same compiled path.

    Args:
        autocovariances: Autocovariance sequence used by the innovation recursion.
        max_lag: Maximum lag included in the recursion.
        initial_value: Optional initial innovation variance (uses ``autocovariances[0]`` when None).

    Returns:
        2-tuple ``(prediction_ar_coeffs, prediction_mses)`` where
        ``prediction_ar_coeffs[k]`` is the ``k``-th step best-linear-predictor
        coefficient vector and ``prediction_mses[k]`` is its mean-squared
        prediction error.
    """
    dtype = autocovariances.dtype
    val = get_innovation_coeffs_internal_njit_gap(autocovariances, max_lag, dtype=dtype, initial_value=initial_value)
    return val

# if __name__ == '__main__':
#     ar = np.array([.6])
#     ar1 = np.array([.6, .1+.000001j])
#     ma = np.array([0])
#
#     gcr = get_causal_representation(ar, ma)
#     get_causal_representation(ar1, ma)
#     print('gcr')
#
#     acov = get_autocovariance_function(ar, ma)
#     acov1 = get_autocovariance_function(ar1, ma)
#     print('gaf')
#
#     c,v = get_innovation_coeffs(acov, 100)
#     c1, v1 = get_innovation_coeffs(acov1, 100)
#     print('gic')
#
#     print(c.round(3))
#     print(c1.round(3))
#
#     print(v.round(3))
#     print(v1.round(3))
#
#     print('----')
#     print(c[-1][:10])
#     print(gcr[:10])
#
#     import matplotlib.pyplot as plt
#     plt.scatter(c[-1][:10], gcr[1:11]**.5)
#     plt.show()

# if __name__ == '__main__':
#
#     def durbin_koopman(arma_params, sigma2, T):
#         """
#         Compute innovation variances and coefficients using the Durbin-Koopman recursion.
#
#         Parameters:
#         arma_params : tuple (ar_coeffs, ma_coeffs)
#             AR and MA coefficients (excluding the leading 1).
#         sigma2 : float
#             White noise variance.
#         T : int
#             Number of time steps to compute.
#
#         Returns:
#         K : np.array
#             Innovation coefficients.
#         v : np.array
#             Innovation variances.
#         """
#         ar_coeffs, ma_coeffs = arma_params
#         p = len(ar_coeffs)
#         q = len(ma_coeffs)
#
#         # Initialize innovation variances and coefficients
#         v = np.ones(T) * sigma2  # Start with white noise variance
#         K = np.zeros(T)
#
#         for n in range(1, T):
#             phi_n = ar_coeffs[n - 1] if n <= p else 0  # AR coefficient at lag n
#             theta_n = ma_coeffs[n - 1] if n <= q else 0  # MA coefficient at lag n
#
#             # Compute innovation coefficient using past variance
#             if v[n - 1] > 0:
#                 K[n] = (phi_n + theta_n) * v[n - 1] / (v[n - 1] + sigma2)
#             else:
#                 K[n] = 0
#
#             # Update innovation variance using proper recursion
#             v[n] = v[n - 1] * (1 - K[n] ** 2)
#
#         return K, v
#
#
#     ar = [.4]
#     ma = [.1]
#
#     print(durbin_koopman((ar, ma), 1.2, T=10))
#     print(get_innovation_coeffs(get_autocovariance_function(ar, ma), 100)[0][-1][:10])
