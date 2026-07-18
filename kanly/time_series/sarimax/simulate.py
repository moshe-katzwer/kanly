from __future__ import absolute_import, print_function

import numpy as np
from numba import njit

from kanly.time_series.sarimax.polynomial import combine_lag_coefs, get_combined_differencing_coefs, check_intersection


def simulate_sarima(n, ar=None, ma=None, d=0, s=2, D=0, sar=None, sma=None, burnin=1.0, sigma2=1.0, seed=0,
                    demean=True):
    """
    Simulate a univariate seasonal ARIMA process.

    Generates `n` observations from a SARIMA process with optional non-seasonal
    autoregressive and moving-average terms, seasonal autoregressive and
    moving-average terms, and non-seasonal/seasonal differencing. The simulation
    is performed by expanding the regular and seasonal lag polynomials, drawing
    Gaussian innovations with variance `sigma2`, discarding an initial burn-in
    period, and optionally demeaning the returned series.

    Parameters
    ----------
    n : int
        Number of observations to return after discarding the burn-in period.
    ar : sequence of float, optional
        Non-seasonal autoregressive coefficients. The coefficient at index `i`
        corresponds to lag `i + 1`.
    ma : sequence of float, optional
        Non-seasonal moving-average coefficients. The coefficient at index `i`
        corresponds to lag `i + 1`.
    d : int, default 0
        Order of non-seasonal differencing.
    s : int, default 2
        Seasonal period. Must be at least 2 when seasonal AR or MA terms are
        supplied.
    D : int, default 0
        Order of seasonal differencing.
    sar : sequence of float, optional
        Seasonal autoregressive coefficients. The coefficient at index `i`
        corresponds to seasonal lag `(i + 1) * s`.
    sma : sequence of float, optional
        Seasonal moving-average coefficients. The coefficient at index `i`
        corresponds to seasonal lag `(i + 1) * s`.
    burnin : float, default 1.0
        Burn-in length as a fraction of `n`. The function simulates
        `int(n * (1 + burnin))` observations and returns the final `n`.
    sigma2 : float, default 1.0
        Variance of the Gaussian innovations.
    seed : int, default 0
        Random seed used to generate innovations.
    demean : bool, default True
        If True, subtract the sample mean from the returned series.

    Returns
    -------
    numpy.ndarray
        Array of shape `(n,)` containing the simulated SARIMA series.

    Raises
    ------
    AssertionError
        If seasonal AR or MA terms are provided and `s < 2`.
    ValueError
        Propagated from `check_intersection` if regular and seasonal lag
        specifications overlap.

    Notes
    -----
    The AR polynomial used for simulation includes the effects of both regular
    and seasonal differencing. Regular and seasonal lag specifications are
    checked for overlap before the lag polynomials are expanded.


    TODO on shares lags between season and regular

    Examples
    --------
    Simulate an AR(2) and inspect the empirical autocorrelation:

    >>> from kanly.api import simulate_sarima, acf
    >>> y = simulate_sarima(n=500, ar=[0.5, 0.1], seed=0, burnin=1000)
    >>> acf(y, nlags=3).round(2)                              # doctest: +SKIP
    array([1.  , 0.56, 0.39, 0.21])

    Simulate a seasonal ARMA(1,0)(0,1)[12] process:

    >>> y_seasonal = simulate_sarima(n=400, ar=[0.4], sma=[0.6],
    ...                              s=12, seed=1, burnin=1.0)
    """

    if ar is None:
        ar = []
    if ma is None:
        ma = []
    if sar is None:
        sar = []
    if sma is None:
        sma = []

    check_intersection(ar, sar, s)
    check_intersection(ma, sma, s)

    if len(sma) or len(sar):
        assert s >= 2

    ma_params_expanded = combine_lag_coefs(
        (ma, 1),
        (sma, s)
    )
    ar_params_expanded = combine_lag_coefs(
        (ar, 1),
        (sar, s),
        (get_combined_differencing_coefs(d=d, D=D, s=s), 1)
    )

    return simulate_sarima_internal(n, seed, ar_params_expanded, ma_params_expanded, sigma2, burnin, demean)


@njit(cache=True)
def simulate_sarima_internal(n, seed, ar_params_expanded, ma_params_expanded, sigma2, burnin, demean):
    """
    Internal helper for SARIMAX estimation: ``simulate_sarima_internal``.

    Args:
        n: Number of simulated observations.
        seed: Random seed.
        ar_params_expanded: Expanded AR coefficients over all active lags.
        ma_params_expanded: Expanded MA coefficients over all active lags.
        sigma2: Innovation variance parameter.
        burnin: Number or fraction of initial simulated observations to discard.
        demean: Whether to subtract the simulated mean after burn-in.

    Returns:
        Simulated SARMA values after burn-in and optional demeaning.
    """
    sigma = sigma2 ** .5

    np.random.seed(seed)
    innovations = np.random.randn(int(n * (1 + burnin))) * sigma
    y = np.zeros(innovations.shape)

    q = len(ma_params_expanded)
    p = len(ar_params_expanded)
    r = max(q + 1, p + 1)

    for i in range(r, len(innovations)):
        y[i] = innovations[i]
        for l, c in enumerate(ma_params_expanded):
            if c:
                y[i] += c * innovations[i - (l + 1)]
        for l, c in enumerate(ar_params_expanded):
            if c:
                y[i] += c * y[i - (l + 1)]

    y = y[-n:]
    if demean:
        y -= y.mean()
    return y

#
# if __name__ == '__main__':
#
#     simulate_sarima(1000, ar=[0,0,.3], sar=[.1], s=4)

# if __name__ == '__main__':
#     from statsmodels.tsa.statespace.sarimax import SARIMAX as SARIMAX_SM
#     from statsmodels.tsa.arima.model import ARIMA as ARIMA_SM
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import pandas as pd
#     from kanly.time_series.sarimax.arma_innovation_functions import get_causal_representation
#
#     import time
#     from kanly.api import SARIMAX, simulate_sarima, lm
#
#     n = 1_000
#     np.random.seed(0)
#     # e[1:] += .4 * e[:-1]
#
#     d = 0
#     D = 1
#     s = 4
#
#     y = simulate_sarima(n, [], [.5], d=d, D=D, s=s, sigma2=.4, burnin=4, seed=51)
#     plt.plot(difference(y, d,D,s))
#     plt.show()
