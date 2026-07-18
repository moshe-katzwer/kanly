"""Six-panel matplotlib diagnostic grid for regression residuals.

Provides ``plot_diagnostics``, a convenience function that generates a
standard set of visual checks for the residuals of a fitted regression model:
standardised residuals over observation order, a residual histogram with
standard-normal and KDE overlays, a normal Q-Q plot, an autocorrelation
correlogram, a residuals-vs-fitted scatter, and a scale-location plot.
"""

from __future__ import absolute_import, print_function

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from kanly.nonparametric.lowess import LOWESS

from kanly.time_series.auto_correlation_function import auto_correlation_function as acf
from kanly.nonparametric.kde import kde


def plot_diagnostics(resid, fittedvalues, scale, figsize=(8, 5), dpi=130, show=True, maxlags=15):
    """
    Plot standard regression residual diagnostics.

    Creates a 2-by-3 diagnostic figure containing standardized residuals over
    observation order, a residual histogram with standard normal and KDE
    overlays, a normal Q-Q plot, an autocorrelation correlogram, residuals
    versus fitted values, and a scale-location plot. These plots are useful for
    visually assessing normality, autocorrelation, heteroskedasticity, and
    systematic structure in residuals.

    Parameters
    ----------
    resid : array-like
        Model residuals.
    fittedvalues : array-like
        Fitted or predicted values corresponding to `resid`.
    scale : float
        Residual variance estimate used to standardize residuals. Standardized
        residuals are computed as `resid / sqrt(scale)`.
    figsize : tuple of float, default (8, 5)
        Figure size passed to `matplotlib.pyplot.subplots`.
    dpi : int, default 130
        Figure resolution in dots per inch.
    show : bool, default True
        If True, display the figure with `matplotlib.pyplot.show`.
    maxlags : int, default 15
        Maximum autocorrelation lag to display in the correlogram.

    Returns
    -------
    matplotlib.figure.Figure
        The created Matplotlib figure.

    Notes
    -----
    The correlogram uses Bartlett standard errors from
    `auto_correlation_function`. The residuals-vs-fitted and scale-location
    panels include LOWESS smooths. The scale-location plot uses
    `sqrt(abs(standardized residuals))` on the vertical axis.

    The inputs `resid` and `fittedvalues` should have the same length, and
    `scale` should be positive.
    """

    # Layout: 2 rows × 3 columns = 6 panels.
    # Row 0: standardised residual trace | histogram | residuals-vs-fitted
    # Row 1: normal Q-Q                  | correlogram | scale-location
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, nrows=2, ncols=3)
    resid_norm = resid / scale ** .5
    # Panel [0,0]: residual index plot – useful for detecting trends or
    # time-ordered heteroskedasticity.
    ax[0][0].plot(resid_norm)
    ax[0][0].axhline(0, color='k')
    ax[0][0].set_title('Standardized Residual')

    # Panel [0,1]: histogram overlaid with standard-normal PDF and KDE –
    # gives a quick visual normality check.
    xrng = np.linspace(-3, 3, 150)
    ax[0][1].plot(xrng, norm.pdf(xrng, 0, 1), label=f'N(0,1)', lw=2, color='red')
    ax[0][1].hist(resid_norm, density=True, alpha=.3, color='b')
    xrng, pdf_xrng = kde(resid_norm)
    ax[0][1].plot(xrng, pdf_xrng, lw=2, label='kde')
    ax[0][1].legend(loc='best')
    ax[0][1].set_title('Residual Histogram')

    # Panel [1,0]: normal Q-Q using the modified plotting positions
    # (i - 0.5)/n (Hazen formula) to avoid the extreme quantiles ±∞.
    nobs = len(resid)
    quants = norm.ppf(np.arange(1, nobs + 1) / nobs - .5 / nobs)
    minq, maxq = quants.min(), quants.max()
    ax[1][0].scatter(
        quants,
        sorted(resid / scale ** .5),
    )
    ax[1][0].plot([minq, maxq], [minq, maxq], lw=2, color='r')
    ax[1][0].set_xlabel('Theoretical Quantiles')
    ax[1][0].set_ylabel('Sample Quantiles')
    ax[1][0].set_title('Normal Q-Q')

    # Panel [1,1]: correlogram with Bartlett ±2σ confidence band.
    a, s = acf(resid, bartlett_std_err=True)
    ax[1][1].plot(a[:maxlags + 1], marker='.', lw=0, markersize=10)
    ax[1][1].fill_between(range(maxlags + 1),
                          -2 * s[:maxlags + 1], 2 * s[:maxlags + 1], alpha=.3)
    ax[1][1].axhline(0, color='k', lw=.5)
    ax[1][1].set_ylim([-1, 1])
    ax[1][1].set_title('Correlogram')

    # Panel [0,2]: residuals vs. fitted – checks for systematic non-linearity
    # and heteroskedasticity (trumpet shapes indicate non-constant variance).
    ax[0][2].set_title("Residuals vs. Fitted")
    ax[0][2].set_xlabel("Fitted Values")
    ax[0][2].set_ylabel("Residuals")
    ax[0][2].scatter(fittedvalues, resid, alpha=.5)
    delta = (max(fittedvalues) - min(fittedvalues)) / 15
    ax[0][2].plot(*LOWESS(resid, fittedvalues, delta=delta), color='r')
    ax[0][2].axhline(0, color='k')

    # Panel [1,2]: scale-location – sqrt(|standardised resid|) vs. fitted.
    # A horizontal LOWESS smooth indicates homoskedasticity.
    ax[1][2].set_title("Scale-Location")
    ax[1][2].set_xlabel("Fitted Values")
    ax[1][2].set_ylabel("sqrt(Standardized Resid)")
    ax[1][2].scatter(fittedvalues, np.sqrt(np.abs(resid_norm)), alpha=.5)
    ax[1][2].plot(*LOWESS(np.sqrt(np.abs(resid_norm)), fittedvalues, delta=delta), color='r')

    plt.tight_layout()

    if show:
        plt.show()

    return fig

# if __name__ == '__main__':
#     import numpy as np
#     import pandas as pd
#
#     from kanly.api import lm
#
#     n = 300
#     np.random.seed(0)
#     df = pd.DataFrame({
#         'x': np.random.randn(n),
#     })
#     df['y'] = 1.2 - 0.3 * df['x'] + .2 * np.random.randn(n) * (.5+np.abs(df.x))
#
#     fit = lm('y ~ x', df, use_t=True,
#              debug=False,
#              # cov_type='bootstrap',
#              # cov_kwds={'max_processes': 6, 'n_samples': 10_000}
#              )
#     print(fit.summary())
#     fit.plot_diagnostics()
#     plt.show()