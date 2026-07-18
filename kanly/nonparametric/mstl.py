from __future__ import absolute_import, print_function

import numpy as np
from kanly.nonparametric.lowess import LOWESS
import matplotlib.pyplot as plt


def mstl(endog, period, swindow=np.inf, twindow=None, robust=True,
         l_iterations=3, mstl_iterations=2):
    """Decompose a time series into trend, one or more seasonal components, and residual.

    Implements Multiple Seasonal-Trend decomposition using LOESS (MSTL). When
    ``period`` is a single int this reduces to classical STL with one seasonal
    component. When ``period`` is a list of ints, MSTL iteratively isolates
    each seasonal component by subtracting the others from the series before
    running the inner STL backfitting loop. This allows series with multiple
    overlapping periodicities (e.g. daily data with both weekly and yearly
    cycles) to be cleanly separated.

    Args:
        endog (array_like): 1-D array of observed time-series values. Must be
            regularly spaced with no missing (NaN) observations.
        period (int or list of int): Seasonal period(s) of the data. Pass a
            single int for classical STL (e.g. 12 for monthly data) or a list
            of ints for MSTL with multiple seasonalities (e.g. ``[7, 365]`` for
            daily data with weekly and yearly cycles). Periods are processed
            smallest-to-largest internally.
        swindow (int, float, or list, optional): Number of complete seasons on
            each side of an observation to include when smoothing each seasonal
            component. Pass a list to use different windows per seasonal period;
            a scalar is broadcast to all periods. Defaults to ``np.inf``, which
            applies a global median to all observations sharing the same phase.
        twindow (int, optional): Number of consecutive observations used in the
            LOWESS trend smoother. Defaults to ``int(1.5 * max(period))``
            (forced odd) so the trend window spans past the longest seasonal
            cycle. Capped at ``len(endog)``.
        robust (bool, optional): If ``True`` (default), performs an outer loop
            where observations are re-weighted by the magnitude of their
            residuals, reducing the impact of anomalies.
        l_iterations (int, optional): Inner backfitting iterations (trend ↔
            seasonality) per seasonal-component pass. Defaults to 3.
        mstl_iterations (int, optional): Outer passes over the list of seasonal
            periods. Only takes effect when more than one period is specified.
            Defaults to 2.

    Returns:
        tuple: ``(trend, seasonality, resid)`` where:
            - **trend** (ndarray): Long-term underlying direction.
            - **seasonality** (ndarray or list of ndarray): The seasonal
              component(s). A single array when ``period`` is an int; a list
              of arrays in the user's input order when ``period`` is a list.
            - **resid** (ndarray): ``endog - trend - sum(seasonality)``.

    Examples:
        >>> import numpy as np
        >>> # Single seasonality (behaves like STL)
        >>> y = np.sin(np.linspace(0, 10, 100)) + np.linspace(0, 5, 100)
        >>> trend, seasonal, resid = mstl(y, period=10)
        >>>
        >>> # Multiple seasonalities
        >>> t = np.arange(365 * 2)
        >>> y = np.sin(2*np.pi*t/7) + 0.5*np.sin(2*np.pi*t/365) + 0.01*t
        >>> trend, seasonals, resid = mstl(y, period=[7, 365])
        >>> weekly, yearly = seasonals

    Multi-seasonal decomposition with diagnostic plot:

        >>> import numpy as np
        >>> from kanly.api import mstl, plot_mstl
        >>> rng = np.random.default_rng(0)
        >>> n = 365 * 3
        >>> t = np.arange(n)
        >>> y = (2.0 * np.sin(2*np.pi*t/7)
        ...      + 5.0 * np.sin(2*np.pi*t/365)
        ...      + 0.005 * t + rng.normal(size=n))
        >>> trend, seasons, resid = mstl(y, period=[7, 365])
        >>> fig = plot_mstl(y, trend, seasons, resid,                 # doctest: +SKIP
        ...                 period_labels=['Weekly (7)', 'Yearly (365)'])
    """
    endog = np.asarray(endog, dtype=float)
    n = len(endog)

    # Normalize period: internally always a list; remember if the caller passed a scalar
    return_scalar_season = isinstance(period, (int, np.integer))
    if return_scalar_season:
        periods = [int(period)]
    else:
        periods = [int(p) for p in period]
    num_seasons = len(periods)

    # Sort smallest -> largest so longer cycles see shorter ones already subtracted out
    order = np.argsort(periods)
    periods_sorted = [periods[i] for i in order]

    # swindow: scalar broadcast, or one value per (sorted) period
    if isinstance(swindow, (list, tuple, np.ndarray)):
        assert len(swindow) == num_seasons, "swindow list must match number of periods"
        swindows_sorted = [swindow[i] for i in order]
    else:
        swindows_sorted = [swindow] * num_seasons

    # Trend window scales with the longest cycle by default
    if twindow is None:
        twindow = int(1.5 * max(periods_sorted))
        if twindow % 2 == 0:  # Prefer odd windows for symmetric smoothing center
            twindow += 1
    twindow = min(n, max(3, twindow))
    lowess_frac = twindow / n

    # Precompute phase-index arrays for each (sorted) period
    phase_indices = [np.arange(n) % p for p in periods_sorted]

    # Initialize components
    seasonalities = [np.zeros(n) for _ in periods_sorted]
    trend = np.zeros(n)
    w = None  # Robustness weights

    # Looping over the period list only helps when there's more than one
    n_mstl_iter = mstl_iterations if num_seasons > 1 else 1

    # Outer robust loops
    for outer in range(1 + robust):
        # MSTL outer loop over each seasonal component
        for _ in range(n_mstl_iter):
            for j in range(num_seasons):
                period_j = periods_sorted[j]
                x_freq = phase_indices[j]
                swindow_j = swindows_sorted[j]

                # Subtract every OTHER seasonal component, so the inner loop only
                # has to disentangle trend from this single seasonality.
                other_seasons = np.zeros(n)
                for k in range(num_seasons):
                    if k != j:
                        other_seasons += seasonalities[k]
                adjusted = endog - other_seasons

                # Inner backfitting loop: alternate trend and S_j estimation
                for _inner in range(l_iterations):

                    # Step 1: Deseasonalize and estimate Trend
                    deseasonalized = adjusted - seasonalities[j]
                    _, trend = LOWESS(deseasonalized, frac=lowess_frac,
                                      weights=w, it=0, delta=0)

                    # Step 2: Detrend and estimate Seasonality_j
                    y_detrend = adjusted - trend

                    new_seasonality = np.zeros(n)
                    # Loop through each distinct phase of the seasonal cycle
                    for p in range(period_j):
                        # Isolate indices belonging to the current phase
                        phase_mask = (x_freq == p)
                        sub_series = y_detrend[phase_mask]
                        sub_n = len(sub_series)

                        # Case 1: Global Seasonal Median
                        if swindow_j >= sub_n:
                            new_seasonality[phase_mask] = np.median(sub_series)

                        # Case 2: Evolving Seasonality (Moving Window)
                        else:
                            smoothed_sub = np.zeros(sub_n)
                            for idx in range(sub_n):
                                start = max(0, idx - swindow_j)
                                end = min(sub_n, idx + swindow_j + 1)
                                smoothed_sub[idx] = np.median(sub_series[start:end])
                            new_seasonality[phase_mask] = smoothed_sub

                    seasonalities[j] = new_seasonality

        # Residuals & robust weights for the next outer loop
        total_season = np.zeros(n)
        for s_arr in seasonalities:
            total_season += s_arr
        resid = endog - total_season - trend

        if robust and outer < robust:
            s = np.median(np.abs(resid))
            if s > 1e-6:
                z = np.abs(resid / (6 * s))
                w = np.where(z < 1, (1.0 - z ** 2) ** 2, 0.0)
            else:
                w = np.ones(n)

    # Restore the caller's original ordering of seasonalities
    inv_order = np.argsort(order)
    seasonalities_out = [seasonalities[i] for i in inv_order]

    if return_scalar_season:
        return trend, seasonalities_out[0], resid
    return trend, seasonalities_out, resid


def plot_mstl(y, trend, seasonality, resid, show=False, figsize=(6, 4), dpi=150,
              title='MSTL Decomposition', period_labels=None):
    """Plot an MSTL decomposition as a stacked matplotlib figure.

    Renders the observed series with the overlaid total fit, the trend, one
    panel per seasonal component, and a residual panel that annotates the
    lag-1 residual autocorrelation.

    Args:
        y: Observed series array used for the top panel and overlay.
        trend: Trend component from :func:`mstl`.
        seasonality: Either a single seasonal-component array (single-season
            MSTL) or a list of arrays returned by multi-period MSTL.
        resid: Residual component from :func:`mstl`.
        show: If ``True``, call ``plt.show()`` after building the figure.
        figsize: Matplotlib figure size in inches.
        dpi: Matplotlib figure resolution.
        title: Figure suptitle.
        period_labels: Optional list of labels for each seasonal-component
            panel. Defaults to ``Seasonality 1``, ``Seasonality 2``, ...

    Returns:
        The matplotlib ``Figure`` instance.

    Examples
    --------
    Plot a daily series with weekly and yearly cycles:

    >>> import numpy as np
    >>> from kanly.api import mstl, plot_mstl
    >>> rng = np.random.default_rng(0)
    >>> n = 365 * 3
    >>> t = np.arange(n)
    >>> y = (2.0 * np.sin(2*np.pi*t/7)
    ...      + 5.0 * np.sin(2*np.pi*t/365)
    ...      + 0.005 * t + rng.normal(size=n))
    >>> trend, seasons, resid = mstl(y, period=[7, 365])
    >>> fig = plot_mstl(y, trend, seasons, resid,
    ...                 period_labels=['Weekly', 'Yearly'])
    """
    # Normalize seasonality to a list so we can handle 1 or many uniformly
    if isinstance(seasonality, np.ndarray):
        seasonalities = [seasonality]
    else:
        seasonalities = list(seasonality)
    n_seasons = len(seasonalities)

    if period_labels is None:
        if n_seasons == 1:
            season_labels = ['Seasonality']
        else:
            season_labels = [f'Seasonality {i + 1}' for i in range(n_seasons)]
    else:
        season_labels = list(period_labels)

    n_panels = 2 + n_seasons + 1  # observed, trend, *seasonalities, residual
    fig, ax = plt.subplots(n_panels, 1, figsize=figsize, sharex=True, dpi=dpi)

    total_season = np.sum(seasonalities, axis=0)
    ax[0].plot(y, label='Observed', color='black', alpha=0.6)
    ax[0].plot(trend + total_season,
               label=f'Trend + Seasonality (R^2={np.corrcoef(y,trend+total_season)[0][1]**2:.3f})',
               color='blue', linestyle='--')
    ax[0].set_ylabel('Data')

    ax[1].plot(trend, label='Trend', color='red')
    ax[1].set_ylabel('Trend')

    cmap = plt.get_cmap('tab10')
    for i, (s_arr, lbl) in enumerate(zip(seasonalities, season_labels)):
        ax[2 + i].plot(s_arr, label=lbl, color=cmap(i % 10))
        ax[2 + i].set_ylabel(lbl)

    ax[-1].plot(resid,
                label=f'Residuals (AR(1)={np.corrcoef(resid[1:], resid[:-1])[0][1]:.2f})',
                color='purple')
    ax[-1].set_ylabel('Residual')

    for axis in ax:
        axis.legend(loc='upper left')
        axis.grid(True, linestyle=':', alpha=0.6)

    ax[-1].set_xlabel('Index')
    plt.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()

    if show:
        plt.show()
    return fig


# if __name__ == '__main__':
#     n = 365 * 3
#     t = np.arange(n)
#     weekly = 2.0 * np.sin(2 * np.pi * t / 7)
#     yearly = 5.0 * np.sin(2 * np.pi * t / 365)
#     trend_true = 0.005 * t
#     noise = np.random.randn(n)
#     y = weekly + yearly + trend_true + noise
#
#     trend, seasonalities, resid = mstl(y, period=[7, 365])
#     plot_mstl(y, trend, seasonalities, resid, show=True,
#               period_labels=['Weekly (7)', 'Yearly (365)'])
