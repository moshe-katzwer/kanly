from __future__ import absolute_import, print_function
import numpy as np

from kanly.time_series.sarimax.hannan_rissanen import hannan_rissanen
# from kanly.time_series.sarimax.polynomial import combine_lag_coefs


def format_trend_scale(trend_scale, nobs):
    """
    Normalize trend scaling options into a numeric denominator.

    Converts the shorthand values accepted by the ``trend_scale`` argument
    into a concrete float: ``None`` or ``True`` maps to ``nobs`` (so trend
    indices run roughly in ``[0, 1]``), and ``False`` maps to ``1.0`` (no
    scaling).  Any other value is returned unchanged.

    Args:
        trend_scale: Scale used to normalize deterministic trend time indices.
        nobs: Number of observations.

    Returns:
        Float denominator used to divide trend time indices before evaluating
        polynomial trend terms.
    """
    if trend_scale is None or trend_scale is True:
        trend_scale = nobs
    elif trend_scale is False:
        trend_scale = 1.0
    return trend_scale


def residualize(endog, exog, trend, exogparams, trendparams, trend_offset, trend_scale):
    """
    Remove deterministic trend and exogenous contributions from endog.

    Args:
        endog: Observed endogenous time-series values.
        exog: Optional exogenous regressor matrix aligned to ``endog``.
        trend: Trend specification, such as None, ``c``, ``t``, ``ct``, ``n``, or an indicator list.
        exogparams: Coefficient vector for exogenous regressors.
        trendparams: Coefficient vector for deterministic trend terms.
        trend_offset: Starting offset for deterministic trend time indices.
        trend_scale: Scale used to normalize deterministic trend time indices.

    Returns:
        3-tuple ``(y, exog_dot_beta, trend_dot_beta)`` where ``y`` is a copy
        of ``endog`` with exogenous and trend contributions subtracted,
        ``exog_dot_beta`` is the removed exogenous contribution
        (0.0 when ``exog`` is None), and ``trend_dot_beta`` is the removed
        trend contribution (0.0 when ``trend`` is None).
    """
    y = endog.copy()
    T = len(y)
    if exog is not None:
        # Keep the removed components so result objects can reconstruct fitted
        # values and expose trend/exog contributions.
        exog_dot_beta = np.dot(exog, exogparams)
        y -= exog_dot_beta
    else:
        exog_dot_beta = 0.0

    if trend is not None:
        tr = np.arange(trend_offset, T + trend_offset) / trend_scale
        trend_dot_beta = 0.0
        i = 0
        for power, c in enumerate(trend):
            if c:
                if power == 0:
                    trend_dot_beta += trendparams[i]
                else:
                    trend_dot_beta += tr ** power * trendparams[i]
                i += 1
        y -= trend_dot_beta
    else:
        trend_dot_beta = 0.0
    return y, exog_dot_beta, trend_dot_beta


def combine_seasonal_lag_params_into_one_vector(lags, lag_params, season_lags, season_lag_params, s):
    """
    Expand nonseasonal and seasonal lag parameters into one lag-indexed vector.

    Args:
        lags: Lag indices associated with ``lag_params``.
        lag_params: Parameter values for nonseasonal lags.
        season_lags: Seasonal lag indices associated with ``season_lag_params``.
        season_lag_params: Parameter values for seasonal lags.
        s: Seasonal period length.

    Returns:
        1-D NumPy array of combined AR (or MA) coefficients indexed from lag 1
        to ``max(lags[-1], s * season_lags[-1])``, with nonseasonal and
        seasonally-strided entries placed at their correct lag positions and
        zero elsewhere.
    """
    ar_params2 = fill_out_lag_parameters_full_vector(lag_params, lags)
    sar_params2 = fill_out_lag_parameters_full_vector(season_lag_params, season_lags, s)

    a = np.r_[1.0, -np.asarray(ar_params2)]
    b = np.r_[1.0, -np.asarray(sar_params2)]
    c = np.convolve(a, b)
    return -c[1:]

    # # DEPRECATED OLD CALL
    # return combine_lag_coefs((ar_params2, 1), (sar_params2, 1))


def split_params(params, k_trend, k_exog, p, q, P, Q):
    """
    Split a packed SARIMAX parameter vector by component.

    Args:
        params: Packed parameter vector in model parameter order.
        k_trend: Number of deterministic trend parameters.
        k_exog: Number of exogenous-regressor parameters.
        p: Nonseasonal autoregressive order or lag specification.
        q: Nonseasonal moving-average order or lag specification.
        P: Seasonal autoregressive order or lag specification.
        Q: Seasonal moving-average order or lag specification.

    Returns:
        7-tuple ``(trendparams, exogparams, arparams, maparams, sarparams,
        smaparams, sigma2)`` containing the sliced sub-vectors in the packed
        parameter order: trend coefficients, exogenous coefficients, nonseasonal
        AR coefficients, nonseasonal MA coefficients, seasonal AR coefficients,
        seasonal MA coefficients, and innovation variance (defaulting to 1.0
        when the vector does not include a variance term, i.e. concentrated-scale
        mode).
    """
    params = np.asarray(params)

    # Packed order matches get_arma_param_names and the Hannan-Rissanen start
    # vector: trend, exog, AR, MA, seasonal AR, seasonal MA, optional sigma2.
    trendparams = params[:k_trend]
    exogparams = params[k_trend:k_trend + k_exog]
    arparams = params[k_trend + k_exog:k_trend + k_exog + p]
    maparams = params[k_trend + k_exog + p:k_trend + k_exog + p + q]
    sarparams = params[k_trend + k_exog + p + q:k_trend + k_exog + p + q + P]
    smaparams = params[k_trend + k_exog + p + q + P:k_trend + k_exog + p + q + P + Q]

    if len(params) == k_trend + k_exog + p + q + P + Q:
        sigma2 = 1.0
    else:
        sigma2 = params[-1]

    return trendparams, exogparams, arparams, maparams, sarparams, smaparams, sigma2


def scale_params(params, k_trend, k_exog, p, q, P, Q, trend, scale, mean):
    """
    Map parameters estimated on standardized data back to the original data scale.

    Args:
        params: Packed parameter vector in model parameter order.
        k_trend: Number of deterministic trend parameters.
        k_exog: Number of exogenous-regressor parameters.
        p: Nonseasonal autoregressive order or lag specification.
        q: Nonseasonal moving-average order or lag specification.
        P: Seasonal autoregressive order or lag specification.
        Q: Seasonal moving-average order or lag specification.
        trend: Trend specification, such as None, ``c``, ``t``, ``ct``, ``n``, or an indicator list.
        scale: Innovation scale or data standardization scale, depending on context.
        mean: Mean used when mapping standardized parameters back to the original scale.

    Returns:
        Copy of ``params`` with trend and exogenous coefficients multiplied by
        ``scale``, the constant trend term shifted by ``mean``, and the
        innovation variance (last element when present) multiplied by
        ``scale ** 2``.  AR and MA coefficients are left unchanged because
        they are dimensionless ratios.
    """
    params = np.array(params)

    # AR and MA coefficients are scale-free; only deterministic regression
    # terms and sigma2 need to be mapped back to the original endog scale.
    params[:k_trend] *= scale
    params[k_trend:k_trend + k_exog] *= scale

    if len(params) == k_trend + k_exog + p + q + P + Q + 1:
        params[-1] *= scale ** 2

    if trend is not None and len(trend) and trend[0]:
        params[0] += mean

    return params


def fill_out_lag_parameters_full_vector(params, lags, s=1):
    """
    So if lags=(2,4) and params=(-.1,.3) and s=1 this returns
    (0.0, -.1, 0.0, -.3)

    If s=3, this returns
    (0,0,0,   0,0 -.1,   0,0,0,    0,0 -.3)
    """
    if len(lags):
        expanded_params = np.zeros(s * lags[-1])
        for i, l in enumerate(lags):
            expanded_params[l * s - 1] = params[i]
        return expanded_params
    else:
        return np.array([], dtype=np.float64)


def none_tuple(x):
    """
    Convert optional iterables to tuples while preserving None.

    Used to serialise order specifications and trend indicators into
    plain tuples for storage in the ``sarimax_options`` dictionary without
    holding references to the original mutable objects.

    Args:
        x: Input iterable, or None.

    Returns:
        ``tuple(x)`` if ``x`` is not None, otherwise None.
    """
    if x is None:
        return None
    else:
        return tuple(x)


def get_sarimax_start_params(start_params, ar_lags, ma_lags, sar_lags, sma_lags,
                             d, D, s, endog, exog, trend, trend_offset, trend_scale,
                             do_hannan_rissanen, enforce_stationarity, enforce_invertibility,
                             concentrate_scale,
                             debug=False):
    """
    Build optimizer starting parameters for SARIMAX MLE.

    Args:
        start_params: Optional initial parameter vector.
        ar_lags: Active nonseasonal autoregressive lags.
        ma_lags: Active nonseasonal moving-average lags.
        sar_lags: Active seasonal autoregressive lags.
        sma_lags: Active seasonal moving-average lags.
        d: Nonseasonal differencing order.
        D: Seasonal differencing order.
        s: Seasonal period length.
        endog: Observed endogenous time-series values.
        exog: Optional exogenous regressor matrix aligned to ``endog``.
        trend: Trend specification, such as None, ``c``, ``t``, ``ct``, ``n``, or an indicator list.
        trend_offset: Starting offset for deterministic trend time indices.
        trend_scale: Scale used to normalize deterministic trend time indices.
        do_hannan_rissanen: Whether to estimate starting ARMA parameters with Hannan-Rissanen.
        enforce_stationarity: Whether to transform AR parameters into the stationary region.
        enforce_invertibility: Whether to transform MA parameters into the invertible region.
        concentrate_scale: Whether to optimize with innovation variance concentrated out.
        debug: Whether to print fitting diagnostics.

    Returns:
        1-D NumPy array of initial parameter values in the packed SARIMAX order
        ``[trend..., exog..., AR..., MA..., SAR..., SMA..., (sigma2)]``.
        If ``start_params=0.0`` returns an all-zeros vector (with 1.0 appended
        for variance when not concentrating scale).  If ``start_params`` is
        provided explicitly it is validated for length and returned as-is.
        Otherwise the Hannan-Rissanen procedure is called to produce data-driven
        starting values.
    """
    if start_params is not None:
        num_params = (
                len(ar_lags) + len(ma_lags) + len(sar_lags) + len(sma_lags) + np.count_nonzero(trend)
                + (0 if exog is None else exog.shape[1])
        )
        if isinstance(start_params, (int, float)) and start_params == 0.0:
            return np.array([0.0] * num_params + ([] if concentrate_scale else [1.0]))
        else:
            assert len(start_params) == num_params + concentrate_scale
            return start_params

    return hannan_rissanen(
        endog, ar_lags, ma_lags, sar_lags, sma_lags, d, D, s,
        exog=exog, trend=trend, trend_offset=trend_offset, trend_scale=trend_scale,
        solve_ar_ma=do_hannan_rissanen,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
        concentrate_scale=concentrate_scale,
        debug=debug
    )
