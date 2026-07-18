from __future__ import absolute_import, print_function

import numpy as np
from numba import njit
from numpy.fft import ifft, fft, ifftshift


def auto_correlation_function(x, adjusted=True, nlags=None, bartlett_std_err=False):
    """See from statsmodels.tsa.stattools.acf with `adjusted=True`

    bartlett_std_err=True returns the standard error as well using the Bartlett formula:
        se[rho(0)] = 0.
        se[rho(1)] = 1. / sqrt(n)
        se[rho(k)] = sqrt( (1 + 2*sum_{l=1}^{k-1} rho[k]**2) / n ), k>1

    Args:
        x: One-dimensional time-series values.
        adjusted: Whether to divide autocovariances by lag-specific sample
            counts instead of the full sample size.
        nlags: Optional maximum lag to return. When omitted, returns lags up
            to half the FFT-padded sample length.
        bartlett_std_err: Whether to return Bartlett standard-error estimates
            along with autocorrelation values.

    Returns:
        Autocorrelation values, or ``(acf, std_err)`` when
        ``bartlett_std_err`` is True.

    Examples
    --------
    Estimate the autocorrelation of an AR(1) process:

    >>> import numpy as np
    >>> from kanly.api import acf
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> x = np.zeros(n)
    >>> for t in range(1, n):
    ...     x[t] = 0.7*x[t-1] + rng.normal()
    >>> vals = acf(x, nlags=5)
    >>> vals[:3].round(2)                                # doctest: +SKIP
    array([1.  , 0.7 , 0.49])

    With Bartlett standard errors for plotting confidence bands:

    >>> vals, se = acf(x, nlags=20, bartlett_std_err=True)
    >>> bands = 1.96 * se                                # 95% Bartlett band
    """
    std_x = np.std(x)
    if std_x < 1e-15:
        return np.hstack([[1.], [0.] * (min(len(x), nlags if nlags else 10_000_000_000) // 2 - 1)])
    xp = ifftshift((x - np.average(x)) / std_x)
    n, = xp.shape
    xp = np.r_[xp[: n // 2 + (n % 2)], np.zeros_like(xp), xp[n // 2 + (n % 2):]]
    f = fft(xp)
    p = np.absolute(f) ** 2
    pi = ifft(p)
    if adjusted:
        denom = 1 + np.arange(n // 2)[::-1] + n // 2 + (n % 2)
    else:
        denom = n
    vals = np.real(pi)[: n // 2] / denom

    if bartlett_std_err:
        acf_sq_cum_sum = np.cumsum(vals[:-1] ** 2)
        std_acf = np.hstack([0, np.sqrt((2 * acf_sq_cum_sum - 1) / n)])
        std_acf[0] = 0

    if nlags is not None:
        vals = vals[:nlags + 1]

    if bartlett_std_err:
        return vals, std_acf
    else:
        return vals


def autocovariance_function(x, adjusted=True, nlags=None, bartlett_std_err=False):
    """
    Estimate autocovariances from autocorrelations and sample variance.

    Args:
        x: One-dimensional time-series values.
        adjusted: Whether to use adjusted denominators for autocovariance estimates.
        nlags: Number of lags to compute or use.
        bartlett_std_err: Whether to include Bartlett standard errors.

    Returns:
        Autocovariance values, or ``(autocovariances, std_err)`` when
        ``bartlett_std_err`` is True.
    """
    res = auto_correlation_function(x, adjusted=adjusted, nlags=nlags, bartlett_std_err=bartlett_std_err)
    var0 = np.var(x)
    if bartlett_std_err:
        res = (res[0] * var0, res[1] * var0)
    else:
        res = res * var0
    return res


def levinson_durbin(x, n_lags=15):
    """
    See ``partial_autocorrelation_function``
    """
    return partial_autocorrelation_function(x, n_lags)


def partial_autocorrelation_function(x, nlags=15):
    """
    Estimate partial autocorrelations with the Durbin-Levinson recursion.

    Args:
        x: One-dimensional time-series values.
        nlags: Maximum partial-autocorrelation lag to return.

    Returns:
        Array of partial autocorrelation values from lag 0 through ``nlags``.

    Examples
    --------
    Estimate the PACF of an AR(2) process; the first two lags should dominate:

    >>> import numpy as np
    >>> from kanly.api import pacf
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> x = np.zeros(n)
    >>> for t in range(2, n):
    ...     x[t] = 0.6*x[t-1] - 0.3*x[t-2] + rng.normal()
    >>> vals = pacf(x, nlags=6)
    >>> vals[:3].round(2)                                # doctest: +SKIP
    array([1.  ,  0.6 , -0.3 ])
    """
    acf_vals = auto_correlation_function(x)
    return partial_autocorrelation_function_internal(acf_vals, nlags)


def partial_autocorrelation_function_internal(acf_vals, nlags=15):
    """
    Estimate PACF values from a precomputed ACF sequence.

    Args:
        acf_vals: Autocorrelation values used by the PACF recursion.
        nlags: Maximum partial-autocorrelation lag to return.

    Returns:
        Array of partial autocorrelation values from lag 0 through ``nlags``.
    """
    return partial_autocorrelation_function_internal_njit(
        np.asarray(acf_vals).astype(float), nlags)


@njit(cache=True)
def partial_autocorrelation_function_internal_njit(acf_vals, nlags=15):
    """
    See Brockwell and Davis "Introduction to Time Series and Forecasting",
    2nd edition, section 2.5.1 on 'The Durbin-Levinson Algorithm'.

    See from statsmodels.tsa.stattools.pacf

    Args:
        acf_vals: Autocorrelation sequence used by the Durbin-Levinson
            recursion.
        nlags: Maximum partial-autocorrelation lag to return.

    Returns:
        Array of partial autocorrelation values from lag 0 through ``nlags``.
    """

    v = 1.0

    pacf_vals = np.zeros(nlags + 1)
    pacf_vals[0] = 1.0

    phis = np.zeros(nlags + 1)

    for n in range(1, nlags + 1):

        phi_nn = acf_vals[n]
        for j in range(1, n):
            phi_nn -= phis[j] * acf_vals[n - j]

        phi_nn /= v
        v *= 1.0 - phi_nn ** 2

        pacf_vals[n] = phi_nn

        if n > 1:
            phi_sub = phis[1:n]
            phi_sub = phi_sub - phi_nn * phi_sub[::-1]
            phis[1:n] = phi_sub

        phis[n] = phi_nn

    return pacf_vals


