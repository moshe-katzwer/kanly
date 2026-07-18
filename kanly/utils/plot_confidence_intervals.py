from __future__ import absolute_import, print_function

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def plot_normal_conf_intervals_from_fit(fit, params, labels=None, title=None, figsize=(10, 5),
                                        dpi=130, show=False, level=.95, plot_horizontal_line=False):
    """Plot normal-approximation confidence intervals extracted from a fit object.

    Convenience wrapper that reads ``fit.params`` and ``fit.bse`` for the
    requested parameters and delegates to ``plot_normal_conf_intervals``.

    Args:
        fit: A regression result object exposing ``.params`` and ``.bse``
            attributes indexed by parameter name.
        params: List of parameter name strings to include in the plot.
        labels: Optional display labels for the x-axis ticks; defaults to
            ``params``.
        title: Optional plot title string.
        figsize: Matplotlib figure size tuple ``(width, height)`` in inches.
        dpi: Figure resolution in dots per inch.
        show: When ``True``, call ``plt.show()`` before returning.
        level: Coverage level for the confidence interval, e.g. ``0.95``.
        plot_horizontal_line: ``False`` / ``True`` (draws line at 0) or a
            numeric value at which to draw a horizontal reference line.

    Returns:
        Matplotlib ``Figure`` object.
    """
    param_values = [fit.params[c] for c in params]
    bse = [fit.bse[c] for c in params]
    return plot_normal_conf_intervals(param_values, bse, labels=labels, title=title, figsize=figsize,
                                      dpi=dpi, show=show, level=level, plot_horizontal_line=plot_horizontal_line)


def plot_normal_conf_intervals(params, bse, labels=None, title=None, figsize=(10, 5),
                               dpi=130, show=False, level=.95, plot_horizontal_line=False):
    """Compute and plot symmetric normal-approximation confidence intervals.

    Constructs ``(estimate ± z * se)`` intervals where ``z`` is the critical
    value for the given ``level`` from a standard normal, then calls
    ``plot_confidence_intervals``.

    Args:
        params: Sequence of point estimates (one per parameter).
        bse: Sequence of standard errors corresponding to each element of
            ``params``.
        labels: Optional sequence of display labels for the x-axis ticks.
        title: Optional plot title string.
        figsize: Matplotlib figure size tuple ``(width, height)`` in inches.
        dpi: Figure resolution in dots per inch.
        show: When ``True``, call ``plt.show()`` before returning.
        level: Coverage level, e.g. ``0.95`` for 95 % confidence intervals.
        plot_horizontal_line: See ``plot_confidence_intervals``.

    Returns:
        Matplotlib ``Figure`` object.
    """
    cv = -norm.ppf((1 - level) / 2)
    point_estims = params
    conf_ints = [(b - cv * s, b + cv * s) for b, s in zip(params, bse)]

    return plot_confidence_intervals(conf_ints, point_estims, labels, title, figsize, dpi, show, level,
                                     plot_horizontal_line)


def plot_confidence_intervals(conf_ints, point_estims=None, labels=None, title=None, figsize=(10, 5),
                              dpi=130, show=False, level=None, plot_horizontal_line=False):
    """Plot a set of pre-computed confidence (or credible) intervals as vertical line segments.

    Each interval is rendered as a vertical three-point line: lower bound,
    point estimate, upper bound.  The x-axis is labelled with parameter names.

    Args:
        conf_ints: Sequence of ``(lo, hi)`` tuples defining each interval.
        point_estims: Optional sequence of point estimates (one per interval);
            uses ``np.nan`` for intervals without a point estimate.
        labels: Optional sequence of display labels for the x-axis ticks;
            defaults to ``['param0', 'param1', ...]``.
        title: Optional plot title string.
        figsize: Matplotlib figure size tuple ``(width, height)`` in inches.
        dpi: Figure resolution in dots per inch.
        show: When ``True``, call ``plt.show()`` before returning.
        level: Coverage level used for the y-axis label (e.g. ``0.95``).
            When ``None``, no y-axis label is added.
        plot_horizontal_line: ``False`` for no line, ``True`` to draw a line
            at ``y=0``, or a numeric value to draw the line at that ``y``
            position.

    Returns:
        Matplotlib ``Figure`` object.
    """
    f = plt.figure(figsize=figsize, dpi=dpi)
    if labels is None:
        labels = [f'param{i}' for i in range(len(conf_ints))]
    assert len(labels) == len(conf_ints)
    if point_estims is not None:
        assert len(point_estims) == len(conf_ints)
    else:
        point_estims = [np.nan] * len(conf_ints)
    xticklabels = []
    for i, (lab, b, (ci_lo, ci_hi)) in enumerate(zip(labels, point_estims, conf_ints)):
        plt.plot([i] * 3, [ci_lo, b, ci_hi], marker='.', lw=2)
        xticklabels.append(lab)
    plt.xticks(range(len(labels)), labels=xticklabels, rotation=90)
    if title:
        plt.title(title)

    if level is not None:
        plt.ylabel(f'{level*100:.1f}% confidence interval')

    if plot_horizontal_line is not None:
        if isinstance(plot_horizontal_line, (float, int)):
            plt.axhline(plot_horizontal_line, c='k', lw=.5)
        elif isinstance(plot_horizontal_line, bool) and plot_horizontal_line:
            plt.axhline(0.0, c='k', lw=.5)

    plt.tight_layout()

    if show:
        plt.show()

    return f


# def plot_confidence_intervals(conf_ints, point_estims=None, labels=None, title=None, figsize=(10, 5),
#                               dpi=130, show=False, level=None, plot_horizontal_line=False, sharey=True):
#     f, ax = plt.subplots(figsize=figsize, dpi=dpi, ncols=len(conf_ints), sharey=sharey)
#     if labels is None:
#         labels = [f'param{i}' for i in range(len(conf_ints))]
#     assert len(labels) == len(conf_ints)
#     if point_estims is not None:
#         assert len(point_estims) == len(conf_ints)
#     else:
#         point_estims = [np.nan] * len(conf_ints)
#     for i, (lab, b, (ci_lo, ci_hi)) in enumerate(zip(labels, point_estims, conf_ints)):
#         ax[i].plot([i] * 3, [ci_lo, b, ci_hi], marker='.', lw=2)
#         ax[i].set_xticks([])
#         if i == 0:
#             ax[i].set_ylabel(f'{level * 100:.1f}% confidence interval')
#     if title:
#         plt.title(title)
#
#
#     if plot_horizontal_line is not None:
#         if isinstance(plot_horizontal_line, (float, int)):
#             val = plot_horizontal_line
#         elif isinstance(plot_horizontal_line, bool) and plot_horizontal_line:
#             val = 0.0
#         else:
#             val = None
#         if val is not None:
#             for i in range(len(conf_ints)):
#                 ax[i].axhline(val, c='k', lw=.5)
#
#     plt.tight_layout()
#
#     if show:
#         plt.show()
#
#     return f


if __name__ == '__main__':
    import numpy as np
    import pandas as pd

    from kanly.api import lm

    n = 50
    np.random.seed(0)
    df = pd.DataFrame({
        'x': np.random.randn(n),
        'z': np.random.randn(n),
        'grp': np.random.randint(0, 12, n),
    })
    df['y'] = 1.2 - 0.3 * df['x'] + .6 * np.random.randn(n)
    df['y'] += .3 * (df.grp == 5)

    fit = lm('y ~ x  + z + C(grp)', df, use_t=True,
             debug=False,
             # cov_type='bootstrap',
             # cov_kwds={'max_processes': 6, 'n_samples': 10_000}
             )
    print(fit.summary())

    fit.plot_confidence_intervals(
        [#'x',
         '-.1 / (1 + {x})', lambda x: -.1 / (1+x[1])],
        plot_horizontal_line=0,
        level=.99,
        show=True,
    )

    v = np.random.randn(10000) * fit.bse['x'] + fit['x']
    plt.hist(-.1/(1+v))
    plt.show()