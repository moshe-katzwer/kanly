from __future__ import absolute_import, print_function

import numpy as np
from scipy.linalg import solve_toeplitz, lstsq

from kanly.time_series.sarimax.constants import DEFAULT_SARIMAX_ENFORCE_STATIONARITY, \
    DEFAULT_SARIMAX_ENFORCE_INVERTIBILITY, DEFAULT_SARIMAX_CONCENTRATE_SCALE
from kanly.time_series.sarimax.difference import difference
from kanly.time_series.sarimax.handle_params import handle_params
from kanly.time_series.auto_correlation_function import auto_correlation_function


def none_max(x):
    """
    Return the maximum of an optional lag collection.

    Args:
        x: Iterable of integer lags. Empty iterables are allowed.

    Returns:
        Maximum lag in ``x`` or ``0.0`` when ``x`` is empty.
    """
    if len(x):
        return max(x)
    else:
        return 0.0


def get_lags_mult(lags, seasonal_lags, s):

    """
    Combine nonseasonal and seasonal lag lists into absolute lag indices.

    Hannan-Rissanen works with one regression matrix whose columns are indexed
    by absolute lags. This helper maps nonseasonal lags and seasonal lags into
    that common lag scale, including interaction lags when both components are
    present.

    Args:
        lags: Nonseasonal lag indices, such as ``(1, 2, 3)``.
        seasonal_lags: Seasonal lag indices before multiplying by ``s``.
        s: Seasonal period length.

    Returns:
        Sorted array-like collection of absolute lags used in the preliminary
        Hannan-Rissanen regression.
    """
    if len(seasonal_lags) == 0:
        return lags
    if len(lags) == 0:
        return [s*i for i in seasonal_lags]

    lags_mult = np.zeros(s * max(seasonal_lags) + max(lags) + 1)
    for i in [0] + list(lags):
        for j in [0] + list(seasonal_lags):
            if i + s * j > 0:
                lags_mult[i + s * j] += 1
    return np.argwhere(lags_mult).flatten()


def hannan_rissanen(endog, ar_lags=None, ma_lags=None, sar_lags=None, sma_lags=None, d=0, D=0, s=2,
                    exog=None, trend=None, ar_order_initial=None, solve_ar_ma=True,
                    debug=False, trend_offset=1, trend_scale=1,
                    enforce_stationarity=DEFAULT_SARIMAX_ENFORCE_STATIONARITY,
                    enforce_invertibility=DEFAULT_SARIMAX_ENFORCE_INVERTIBILITY,
                    concentrate_scale=DEFAULT_SARIMAX_CONCENTRATE_SCALE,
                    ):
    """
    Assumes differenced data
    Returns estimates
        ([trend_params, exog_params, ar_params, ma_params, seasonal_ar_params, seasonal_ma_params], sigma2)

    solve_ar_ma: whether to do Hannan Rissanen to find AR, MA coefficients, or just return coefs on exog

    Estimate SARIMAX starting values using the Hannan-Rissanen procedure.

    The goal of this routine is not to produce final estimates; it builds a
    reasonable starting vector for maximum-likelihood optimization. It first
    normalizes the requested nonseasonal and seasonal lags onto a single
    absolute-lag scale. If exogenous regressors or deterministic trend terms are
    present, it estimates those coefficients by least squares and runs the
    autoregressive/moving-average starter on the residualized series.

    When MA terms are present, the routine follows the classic Hannan-Rissanen
    two-stage idea:

    1. Fit a high-order AR approximation using autocorrelations from the
       residualized series.
    2. Use that high-order AR approximation to construct innovation residuals.
    3. Regress the series on selected lagged observations and lagged innovation
       residuals to obtain AR and MA starting values.

    When no MA terms are requested, the routine skips the preliminary innovation
    step and estimates only the requested AR/seasonal AR lags by least squares.
    Seasonal coefficients are separated back out of the absolute-lag vector
    before returning so the packed output matches the parameter order expected
    by ``split_params`` and ``sarimax_internal``.

    If stationarity or invertibility enforcement is requested, the initial AR
    and MA blocks are transformed through ``handle_params`` before being packed.
    If ``concentrate_scale`` is true, the innovation variance is omitted because
    the likelihood optimizer estimates it after optimizing the remaining
    parameters.

    Args:
        endog: Observed time-series values, before optional differencing by
            ``d`` and ``D``.
        ar_lags: Active nonseasonal AR lags.
        ma_lags: Active nonseasonal MA lags.
        sar_lags: Active seasonal AR lags.
        sma_lags: Active seasonal MA lags.
        d: Nonseasonal differencing order.
        D: Seasonal differencing order.
        s: Seasonal period.
        exog: Optional exogenous regressor matrix.
        trend: Trend indicator list from ``parse_trend``.
        ar_order_initial: Optional preliminary AR order for innovation
            residuals.
        solve_ar_ma: Whether to estimate AR/MA starting values or only
            trend/exogenous coefficients.
        debug: Whether to print diagnostic information.
        trend_offset: Starting offset for deterministic trend indices.
        trend_scale: Scale for deterministic trend indices.
        enforce_stationarity: Whether to transform AR starting values.
        enforce_invertibility: Whether to transform MA starting values.
        concentrate_scale: Whether to omit ``sigma2`` from the returned vector.

    Returns:
        Packed starting parameter vector in the same component order consumed
        by ``split_params``; includes ``sigma2`` unless concentrated out.
    """

    if ar_lags is None:
        ar_lags = []
    if ma_lags is None:
        ma_lags = []
    if sar_lags is None:
        sar_lags = []
    if sma_lags is None:
        sma_lags = []

    k_ar, k_ma, k_sar, k_sma = len(ar_lags), len(ma_lags), len(sar_lags), len(sma_lags)

    ar_lags_mult = get_lags_mult(ar_lags, sar_lags, s)
    ma_lags_mult = get_lags_mult(ma_lags, sma_lags, s)

    k_exog = exog.shape[1] if exog is not None else 0

    # remove exog, trend
    k_trend = np.count_nonzero(trend)
    if k_trend:
        trend_arr = np.zeros((len(endog), k_trend))
        tr = np.arange(trend_offset, trend_offset + len(endog)) / trend_scale
        i = 0
        for power, c in enumerate(trend):
            if c == 1:
                trend_arr[:, i] = tr ** power
                i += 1

    # difference the data if necessary
    if d or D:
        endog = difference(endog, d, D, s)
        if k_exog:
            exog = difference(exog, d, D, s)
        if k_trend:
            trend_arr = difference(trend_arr, d, D, s)

    # Estimate exog/trend coeffs
    if k_exog or k_trend:
        if k_trend and not k_exog:
            temp = trend_arr
        elif not k_trend and k_exog:
            temp = exog
        else:
            temp = np.hstack([trend_arr, exog])
        exog_coef = lstsq(temp, endog)[0]
        endog_min_covar = endog - np.dot(temp, exog_coef)
    else:
        endog_min_covar = endog
        exog_coef = []

    scale_estimate = np.mean((endog - endog_min_covar)**2)

    if k_ar == k_ma == k_sar == k_sma == 0:
        return np.hstack([exog_coef, [] if concentrate_scale else [scale_estimate]])

    if not solve_ar_ma:
        return np.hstack([exog_coef, [0.0] * (k_ar + k_ma + k_sar + k_sma),
                          [] if concentrate_scale else [scale_estimate]
                          ])

    has_ma_terms = len(ma_lags_mult)
    if has_ma_terms:

        T = len(endog)
        if ar_order_initial is None:
            ar_order_initial = int(max(2 * max(none_max(ar_lags_mult), none_max(ma_lags_mult)), np.log(T) ** 2))
        if T - ar_order_initial < ar_order_initial:
            raise Exception(f"Not enough (differenced) observation{T} for H-R lag order {ar_order_initial}!")

        acf = auto_correlation_function(endog_min_covar)
        ar_coefs_initial = solve_toeplitz(acf[:ar_order_initial], acf[1:ar_order_initial + 1])

        endog_min_covar_resid = endog_min_covar.copy()
        for j, c in enumerate(ar_coefs_initial):
            endog_min_covar_resid[j + 1:] -= c * endog_min_covar[:-(j + 1)]

        T = len(endog_min_covar)
        hr_matrix = np.zeros((T - ar_order_initial, len(ar_lags_mult) + len(ma_lags_mult)))
        num_ar_params = len(ar_lags_mult)
        for j, l in enumerate(ar_lags_mult):
            hr_matrix[:, j] = endog_min_covar[ar_order_initial - l:-l]
        for j, l in enumerate(ma_lags_mult):
            hr_matrix[:, num_ar_params + j] = endog_min_covar_resid[ar_order_initial - l:-l]

        lag_params, residue, *_ = lstsq(hr_matrix, endog_min_covar[ar_order_initial:])

        ar_params_full = lag_params[:num_ar_params]
        ma_params_full = lag_params[num_ar_params:]
        ar = []
        sar = []
        ma = []
        sma = []
        for j, (l, v) in enumerate(zip(ar_lags_mult, ar_params_full)):
            if l in ar_lags:
                ar.append(v)
            if l % s == 0 and l // s in sar_lags:
                sar.append(v)
        for j, (l, v) in enumerate(zip(ma_lags_mult, ma_params_full)):
            if l in ma_lags:
                ma.append(v)
            if l % s == 0 and l // s in sma_lags:
                sma.append(v)

        ar, ma, sar, sma = handle_params(ar, ma, sar, sma, enforce_stationarity, enforce_invertibility)
        return np.hstack([exog_coef, ar, ma, sar, sma,
                          [] if concentrate_scale else residue/(T-ar_order_initial)])

    else:

        T_eff = len(endog_min_covar)
        hr_matrix = np.zeros((T_eff, len(ar_lags_mult)))
        for i, l in enumerate(ar_lags_mult):
            hr_matrix[l:, i] = endog_min_covar[:-l]

        lag_params, residue, *_ = lstsq(hr_matrix[max(ar_lags_mult):], endog_min_covar[max(ar_lags_mult):])
        ar = []
        sar = []
        for j, (l, v) in enumerate(zip(ar_lags_mult, lag_params)):
            if l in ar_lags:
                ar.append(v)
            if l % s == 0 and l // s in sar_lags:
                sar.append(v)

        return np.hstack([exog_coef, ar, sar, [] if concentrate_scale else [residue / (T_eff - max(ar_lags_mult))]])
