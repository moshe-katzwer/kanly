from __future__ import absolute_import, print_function

import numpy as np
from kanly.nonparametric.lowess import LOWESS
import matplotlib.pyplot as plt


def stl(endog, period, swindow=np.inf, twindow=None, robust=True, l_iterations=3):
    """Decompose a time series into trend, seasonality, and residual components.

    Implements a Seasonal-Trend decomposition procedure using an alternating
    backfitting loop with LOESS/LOWESS smoothers. The algorithm iteratively
    estimates the trend component from the deseasonalized series and the 
    seasonal component from the detrended series. An optional outer loop 
    uses a robust bisquare weight function to down-weight outliers.

    Args:
        endog (array_like): 1-D array of observed time-series values. Must be
            regularly spaced with no missing (NaN) observations.
        period (int): The seasonal period of the data. For example:
            - 4 for quarterly data
            - 12 for monthly data
            - 52 for weekly data
            - 7 for daily data with weekly seasonality
        swindow (int or float, optional): Number of complete seasons on each side 
            of an observation to include when smoothing its seasonal component. 
            Defaults to ``np.inf``, which applies a global median to all 
            observations sharing the same seasonal phase.
        twindow (int, optional): Number of consecutive observations used in the 
            LOWESS trend smoother. Defaults to ``int(1.5 * period)`` (forced odd), 
            which ensures the window spans past a single seasonal cycle to counteract 
            LOWESS's local weight concentration. Capped at ``len(endog)``.
        robust (bool, optional): If ``True`` (default), performs an outer loop pass 
            where observations are re-weighted based on the magnitude of their 
            residuals, reducing the impact of anomalies on subsequent iterations.
        l_iterations (int, optional): Number of inner backfitting iterations to run 
            per outer loop. Higher values allow the trend and seasonal components 
            to cleanly separate. Defaults to 3.

    Returns:
        tuple of numpy.ndarray: A tuple containing three 1-D NumPy arrays of 
        length ``n = len(endog)``:
            - **trend**: The long-term underlying direction of the series.
            - **seasonality**: The repeating periodic patterns.
            - **resid**: The remainder/irregular component after removing trend 
              and seasonality (``endog - trend - seasonality``).

    Raises:
        AssertionError: If `twindow` or `swindow` resolve to less than or equal to 0.
        
    Examples:
        >>> import numpy as np
        >>> data = np.sin(np.linspace(0, 10, 100)) + np.linspace(0, 5, 100)
        >>> trend, seasonal, resid = stl(data, period=10)

    Robust monthly STL with explicit trend window:

        >>> import numpy as np
        >>> from kanly.api import stl, plot_stl
        >>> rng = np.random.default_rng(0)
        >>> n = 12 * 6
        >>> t = np.arange(n)
        >>> y = 4 * np.sin(2*np.pi*t/12) + 0.15*t + rng.normal(size=n)
        >>> trend, seasonal, resid = stl(y, period=12, twindow=18)
        >>> fig = plot_stl(y, trend, seasonal, resid, show=False)   # doctest: +SKIP
    """
    endog = np.asarray(endog, dtype=float)
    n = len(endog)

    # Rule of thumb: twindow should span multiple periods to fight tricube weighting
    if twindow is None:
        twindow = int(1.5 * period)
        if twindow % 2 == 0:  # Prefer odd windows for symmetric smoothing center
            twindow += 1

    twindow = min(n, max(3, twindow))
    lowess_frac = twindow / n
    x_freq = np.arange(n) % period

    # Initialize components
    seasonality = np.zeros(n)
    trend = np.zeros(n)
    w = None  # Robustness weights

    # Outer robust loops
    for outer in range(1 + robust):
        # Inner backfitting loops to let trend and seasonality separate
        for inner in range(l_iterations):

            # Step 1: Deseasonalize and estimate Trend
            deseasonalized = endog - seasonality
            _, trend = LOWESS(deseasonalized, frac=lowess_frac, weights=w, it=0, delta=0)

            # Step 2: Detrend and estimate Seasonality
            y_detrend = endog - trend

            new_seasonality = np.zeros(n)
            # Loop through each distinct phase of the seasonal cycle (e.g., each of the 12 months)
            for p in range(period):
                # Isolate indices belonging to the current phase (e.g., all Januaries)
                phase_mask = (x_freq == p)
                sub_series = y_detrend[phase_mask]
                sub_n = len(sub_series)

                # Case 1: Global Seasonal Mean/Median
                # If the smoothing window covers the entire sub-series, use a flat median across all years
                if swindow >= sub_n:
                    new_seasonality[phase_mask] = np.median(sub_series)

                # Case 2: Evolving Seasonality (Moving Window)
                # If swindow is finite, apply a localized moving median to allow seasonality to change over time
                else:
                    smoothed_sub = np.zeros(sub_n)
                    # Compute a localized median for every individual year's phase point
                    for idx in range(sub_n):
                        # Define boundaries for the moving window centered at the current year
                        start = max(0, idx - swindow)
                        end = min(sub_n, idx + swindow + 1)

                        # Take the median of the detrended values within this local time window
                        smoothed_sub[idx] = np.median(sub_series[start:end])

                    # Scatter the smoothed sub-series values back into the main seasonality array
                    new_seasonality[phase_mask] = smoothed_sub

            seasonality = new_seasonality

        # Step 3: Compute Residuals & Robust Weights for the next outer loop
        resid = endog - seasonality - trend
        if robust and outer < robust:
            s = np.median(np.abs(resid))
            if s > 1e-6:
                z = np.abs(resid / (6 * s))
                w = np.where(z < 1, (1.0 - z ** 2) ** 2, 0.0)
            else:
                w = np.ones(n)

    return trend, seasonality, resid


def plot_stl(y, trend, seasonality, resid, show=False, figsize=(6, 4), dpi=150, title='STL Decomposition'):
    """Plot an STL decomposition as a stacked four-panel matplotlib figure.

    Renders the observed series with overlaid trend + seasonality, followed by
    individual trend, seasonal, and residual panels with shared x-axes.

    Args:
        y: Observed series array used for the top panel and overlay.
        trend: Trend component from :func:`stl`.
        seasonality: Seasonal component from :func:`stl`.
        resid: Residual component from :func:`stl`.
        show: If ``True``, call ``plt.show()`` after building the figure.
        figsize: Matplotlib figure size in inches.
        dpi: Matplotlib figure resolution.
        title: Figure suptitle.

    Returns:
        The matplotlib ``Figure`` instance (so it can be saved or further
        styled).

    Examples
    --------
    Build a quick STL diagnostic plot:

    >>> import numpy as np
    >>> from kanly.api import stl, plot_stl
    >>> rng = np.random.default_rng(0)
    >>> n = 12 * 6
    >>> t = np.arange(n)
    >>> y = 4 * np.sin(2*np.pi*t/12) + 0.15*t + rng.normal(size=n)
    >>> trend, seasonal, resid = stl(y, period=12, twindow=18)
    >>> fig = plot_stl(y, trend, seasonal, resid, show=False)
    """
    fig, ax = plt.subplots(4, 1, figsize=figsize, sharex=True, dpi=dpi)

    ax[0].plot(y, label='Observed', color='black', alpha=0.6)
    ax[0].plot(trend + seasonality, label='Trend + Seasonality', color='blue', linestyle='--')
    ax[1].plot(trend, label='Trend', color='red')
    ax[2].plot(seasonality, label='Seasonality', color='green')
    ax[3].plot(resid, label='Residuals', color='purple')

    ylabels = ['Data', 'Trend', 'Seasonality', 'Residual']
    for i, axis in enumerate(ax):
        axis.legend(loc='upper left')
        axis.set_ylabel(ylabels[i])
        axis.grid(True, linestyle=':', alpha=0.6)

    ax[-1].set_xlabel('Index')
    plt.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()

    if show:
        plt.show()
    return fig

# if __name__ == '__main__':
#     n = 12*6
#     x = np.random.randn(n)
#     t = np.arange(n)
#     s = np.sin(2* np.pi * t / 12)
#     y = 4 * s + .15 * t + x
#
#     trend, seasonality, resid = stl(y, period=12, twindow=18)
#     plot_stl(y, trend, seasonality, resid, show=True)
