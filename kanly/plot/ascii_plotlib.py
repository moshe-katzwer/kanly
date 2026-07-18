"""ASCII line plots, scatter density, and histograms for terminal / log output.

Uses Unicode box-drawing (``┌┐└┘│─``), bullets, and block / marker characters to render
``numpy`` arrays as multi-line strings. The same entry points are re-exported from
``kanly.api`` as ``plot``, ``scatter``, and ``hist`` for notebooks, logs, and REPLs where
matplotlib is unnecessary or unavailable.
"""
from __future__ import absolute_import, print_function
import numpy as np


def _transform(data, lc, rc):
    """Return finite ``data`` trimmed by lower/upper tail quantiles; pair with min/max.

    Parameters
    ----------
    data : array-like or None
        Series to clean. If ``None``, returns ``(None, nan, nan)``.
    lc, rc : float
        Tail mass in ``[0, 0.5)`` for lower and upper trimming via ``np.quantile`` at
        ``lc`` and ``1 - rc``.

    Returns
    -------
    data : ndarray or None
        Filtered values, or ``None``.
    lo, hi : float
        Minimum and maximum of retained data (``nan`` when ``data`` is ``None``).
    """
    if data is None:
        return None, np.nan, np.nan
    else:
        data = np.asarray(data)
        data = data[np.isfinite(data)]
        if lc > 0 or rc > 0:
            ql, qh = np.quantile(data, [lc, 1 - rc])
            data = data[(data > ql) & (data < qh)]
        return data, min(data), max(data)


def plot(y, nrows=25, ncols=140, xlabel=None, ylabel=None, do_print=False, coverage=None):
    """ASCII line chart of a 1-D series versus its (implicit) index.

    ``y`` is truncated so its length is a multiple of ``ncols``, then reshaped into
    ``ncols`` contiguous buckets. Each column shows the **mean** of that bucket on the
    vertical axis (``nrows`` character rows). A ``●`` marks the mean row; optional
    ``coverage`` in ``(0, 1]`` draws inner-quantile band edges per bucket using ``┬`` /
    ``┴`` at the bucket's ``coverage`` central interval.

    Parameters
    ----------
    y : array-like
        One-dimensional values (non-finite entries dropped before bucketing).
    nrows, ncols : int
        Canvas height in text rows and width in monospace columns (``ncols`` capped by
        ``len(y)`` after cleaning).
    xlabel, ylabel : str, optional
        Short labels printed below the frame.
    do_print : bool
        If True, ``print`` the returned string in addition to returning it.
    coverage : float, optional
        If set, symmetric tail mass ``(1 - coverage) / 2`` is trimmed per bucket for the
        band display.

    Returns
    -------
    str
        Multi-line ASCII plot (newline-terminated block).

    Examples
    --------
    Draw an ASCII line plot of a simulated AR(1) series:

    >>> import numpy as np
    >>> from kanly.api import plot
    >>> rng = np.random.default_rng(0)
    >>> y = np.cumsum(rng.normal(size=1000))
    >>> s = plot(y, nrows=12, ncols=80, ylabel='y_t', xlabel='t')
    >>> print(s)                          # doctest: +SKIP

    Add a 90% coverage band to summarise dispersion per bucket:

    >>> s = plot(y, coverage=0.9)
    """
    ncols = min(ncols, len(y))
    y = np.asarray(y)
    y = y[np.isfinite(y)]
    # Buckets are equal-width along the index; drop a short tail so len is divisible by ncols.
    y = y[:len(y) - (len(y) % ncols)]
    y = y.reshape((ncols, -1))
    means = y.mean(axis=1)
    l, u = np.min(means), np.max(means)

    quants = None
    if coverage is not None:
        assert isinstance(coverage, float) and 0 < coverage <= 1
        tail = (1 - coverage) / 2
        quants = np.quantile(y, q=(tail, 1 - tail), axis=1)
        l, u = np.min(quants), np.max(quants)
        quants = np.round((quants - l) / (u - l) * nrows, 0)

    # adjust scale
    means = np.round((means - l) / (u - l) * nrows, 0)

    s = '\n'

    lo_str = '%10.3e' % l
    hi_str = '%10.3e' % u

    if lo_str == hi_str:
        hi_str = '+%9.2e' % (u - l)

    s += f'{hi_str}' + ' ' + '┌' + '─' * ncols + '┐' + '\n'
    for i in range(nrows):
        s += ' ' * 11 + '│'
        for j in range(ncols):
            if means[j] == nrows - i:
                s += '●'
            else:
                if quants is not None and (quants[0][j] == nrows - i or quants[1][j] == nrows - i):
                    if quants[0][j] == nrows - i:
                        s += '┬'
                    else:
                        s += '┴'
                else:
                    s += ' '
        s += '│\n'
    s += lo_str + ' ' + '└' + '─' * ncols + '┘'
    if xlabel is not None:
        s_temp = f" x = {xlabel}"
        fill = ' ' * (ncols - len(s_temp)) + '│'
        s += f'\n{" " * 11}│{s_temp}{fill}'
    if ylabel is not None:
        s_temp = f" y = {ylabel}"
        fill = ' ' * (ncols - len(s_temp)) + '│'
        s += f'\n{" " * 11}│{s_temp}{fill}'
    if coverage is not None:
        s_temp = f' (┤, ├) is {coverage:.3f} interval'
        fill = ' ' * (ncols - len(s_temp)) + '│'
        s += f'\n{" " * 11}│{s_temp}{fill}'
    if xlabel is not None or ylabel is not None or coverage is not None:
        s += '\n' + ' ' * 11 + '└' + '─' * ncols + '┘' '\n'

    if do_print:
        print(s)

    return s

def scatter(
        x, y, nrows=25, ncols=140, xlabel=None, ylabel=None, do_print=False,
        xscale=None, yscale=None, left_censor=0, right_censor=0,
        shade=True, marker=None,
):
    """ASCII scatter / density plot: bin ``(x, y)`` pairs into a character grid.

    Points are mapped linearly from data min–max to ``[0, ncols-1]`` × ``[0, nrows-1]``.
    Cell counts drive symbol choice: with ``shade=True``, darker blocks (``█▓▒░…``)
    indicate higher density; with ``shade=False``, a single ``marker`` character marks
    occupied cells. Optional ``left_censor`` / ``right_censor`` trim extreme quantiles
    on **both** axes before binning.

    Parameters
    ----------
    x, y : array-like
        Same-length coordinate pairs (non-finite pairs removed).
    nrows, ncols : int
        Raster size in characters.
    xlabel, ylabel : str, optional
        Axis annotations (``ylabel`` also sets left margin width when non-empty).
    do_print : bool
        If True, ``print`` the returned string.
    xscale, yscale : str, optional
        Must be ``None`` or ``'log'`` (only validated; binning uses linear min/max).
    left_censor, right_censor : float
        Quantile cutoffs in ``[0, 0.5)`` for trimming tails on **x** and **y** independently.
    shade : bool
        If True, use a multi-symbol density ramp; if False, use ``marker`` (default ``♦``).
    marker : str, optional
        Single character used when ``shade`` is False.

    Returns
    -------
    str
        Multi-line ASCII graphic.

    Examples
    --------
    ASCII scatter plot of two correlated samples:

    >>> import numpy as np
    >>> from kanly.api import scatter
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=1_000)
    >>> y = 0.5 * x + rng.normal(size=1_000)
    >>> print(scatter(x, y, nrows=20, ncols=80,
    ...               xlabel='x', ylabel='y'))   # doctest: +SKIP

    Single-marker style (no density shading):

    >>> print(scatter(x, y, shade=False, marker='*'))   # doctest: +SKIP
    """

    x, y = np.asarray(x), np.asarray(y)
    assert len(x) == len(y)

    idx = np.isfinite(x) & np.isfinite(y)
    x, y = x[idx], y[idx]

    if left_censor > 0 or right_censor > 0:
        qlx, qhx = np.quantile(x, [left_censor, 1 - right_censor])
        qly, qhy = np.quantile(y, [left_censor, 1 - right_censor])
        idx = (x > qlx) & (x < qhx) & (y > qly) & (y < qhy)
        x, y = x[idx], y[idx]

    assert xscale is None or xscale == 'log'
    assert yscale is None or yscale == 'log'

    arr = np.zeros((nrows, ncols))  # rows = y (high at top), cols = x
    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)

    min_y_str = '%10.3e' % min_y
    max_y_str = '%10.3e' % max_y
    if min_y_str == max_y_str:
        max_y_str = '+%9.3e' % (max_y - min_y)

    min_x_str = '%10.3e' % min_x
    max_x_str = '%10.3e' % max_x
    if min_x_str == max_x_str:
        max_x_str = '+%9.3e' % (max_x - min_x)

    x1 = (x - min_x) / (max_x - min_x)
    y1 = (y - min_y) / (max_y - min_y)

    for _x, _y in zip(x1, y1):
        arr[int(_y * (nrows - 1)), int(_x * (ncols - 1))] += 1

    if shade:
        fills = '█▓▒░○·'
    else:
        if marker is None:
            marker = '♦'
        fills = marker

    arr_nz = arr.ravel()
    arr_nz = arr_nz[arr_nz > 0]
    quants = np.quantile(arr_nz, np.linspace(0,1,len(fills)+1))[:-1][::-1]

    if ylabel is None:
        ylabel = ''
    shift = max(10, len(ylabel))

    s = f"%{shift}s" % max_y_str + ' ┌' + '─' * ncols + '┐'
    for i in range(nrows)[::-1]:
        s += '\n' + ((' ' * shift) if i != nrows // 2 else (f'%{shift}s' % ylabel)) + ' │'
        for j in range(ncols):
            if arr[i, j] > 0:
                for k, q in enumerate(quants):
                    if arr[i, j] >= q:
                        s += fills[k]
                        break
            else:
                s += ' '
        s += '│'

    s += '\n' + f"%{shift}s" % min_y_str + ' └' + '─' * ncols + '┘'
    xlabel = '' if xlabel is None else xlabel
    s += '\n' + ' ' * shift + min_x_str + ' ' * ((ncols - 22 - len(xlabel)) // 2) + xlabel + ' ' * (ncols - ((ncols - len(xlabel)) // 2) - len(xlabel) - 7) + max_x_str

    if do_print:
        print(s)

    return s


def hist(*xs, nrows=25, ncols=None, bins=20,
          left_censor=0, right_censor=0, do_print=False,
          sharex=True, gridcols=4,
          cumulative=False,
          density=True,
          labels=None,
          histtype='bar'):
    """ASCII histogram(s) for one or more 1-D samples, laid out in a text grid.

    Each series is optionally tail-trimmed with :func:`_transform`, binned into
    ``bins`` equal-width intervals on the value axis, then drawn with Unicode blocks
    (``histtype='bar'``) or ``♦`` step markers (``histtype='step'``). Multiple series
    appear side-by-side with ``gridcols`` panels per row. When ``sharex`` is True, all
    panels share the global min/max across series for the bin edges.

    Parameters
    ----------
    *xs : array-like
        One or more samples to histogram.
    nrows : int
        Vertical resolution in character rows (counts scaled to ``nrows * 8`` sub-rows
        using partial block characters).
    ncols : int, optional
        Total width per panel; default ``bins`` (must be divisible by ``bins``).
    bins : int
        Number of histogram bins.
    left_censor, right_censor : float
        Tail fractions in ``[0, 0.5)`` passed to :func:`_transform` per series.
    do_print : bool
        If True, ``print`` the returned string.
    sharex : bool
        If True, use a common value range for binning all series.
    gridcols : int
        Number of histogram panels per text row.
    cumulative : bool
        If True, use cumulative counts before density scaling.
    density : bool
        If True (with ``sharex``), scale counts so panels are comparable.
    labels : list of str, optional
        Title above each panel (default ``<x0>``, …).
    histtype : {'bar', 'step'}
        ``'bar'`` fills columns with block height; ``'step'`` marks non-zero bins with ``♦``.

    Returns
    -------
    str
        Multi-line string containing all panels.

    Raises
    ------
    Exception
        If ``histtype`` is invalid, ``ncols`` is not a multiple of ``bins``, or no
        series are passed.

    Examples
    --------
    Single-series ASCII histogram:

    >>> import numpy as np
    >>> from kanly.api import hist
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=1_000)
    >>> print(hist(x, bins=20, nrows=15))      # doctest: +SKIP

    Two-panel comparison with shared x-axis:

    >>> y = rng.normal(loc=1.0, scale=2.0, size=1_000)
    >>> print(hist(x, y, labels=['x', 'y'],
    ...            sharex=True, gridcols=2))    # doctest: +SKIP
    """
    blocks = ' ▁▂▃▄▅▆▇█'  # 1/8 to full
    fullblock = '█'

    assert histtype in ('bar', 'step')

    assert .5 > left_censor >= 0
    assert .5 > right_censor >= 0

    if ncols is None:
        ncols = bins
    assert ncols % bins == 0

    if len(xs) == 0:
        raise Exception("Must use at least one series with `hist`!")

    if labels is None:
        labels = [f'<x{j}>' for j in range(len(xs))]

    gridcols = min(len(xs), gridcols)
    gridrows = len(xs) // gridcols + (len(xs) % gridcols > 0)

    xs = [_transform(x, left_censor, right_censor) for x in xs]

    if sharex:
        min_x = np.min([z[1] for z in xs])
        max_x = np.max([z[2] for z in xs])
        bounds = [(min_x, max_x)] * len(xs)
    else:
        bounds = [(minn, maxx) for _, minn, maxx in xs]

    bindata = [
        dict(
            zip(
                ['col', 'count'],
                np.unique(
                    np.clip(bins * (s - bounds[i][0]) / (bounds[i][1] - bounds[i][0]), a_min=0, a_max=bins - 1).astype(
                        int),
                    return_counts=True)
            )
        )
        for i, (s, _, _) in enumerate(xs)
    ]

    if cumulative:
        for s in bindata:
            s['count'] = np.cumsum(s['count'])

    if density or not sharex:
        max_s_sum = max([sum(s['count']) for s in bindata])
        if cumulative or not sharex:
            for s in bindata:
                s['count'] = s['count'] * max_s_sum / max(s['count'])
        else:
            for s in bindata:
                s['count'] = s['count'] * max_s_sum / sum(s['count'])

    max_cnt = float(max([max(s['count']) for s in bindata]))
    for s in bindata:
        s['count'] = (s['count'] / max_cnt * nrows * 8).astype(int)

    plot_strs = []

    cols_per_bin = ncols // bins

    for s in bindata:
        temp = []
        heights = np.zeros(bins, dtype=int)
        heights[s['col']] = s['count']
        h_last = 0
        for h in heights:
            if cumulative and h < h_last:
                h = h_last
            h_last = h
            for _ in range(cols_per_bin):
                if histtype == 'bar':
                    if h == 0:
                        temp.append([' '] * nrows)
                    elif h % 8 == 0:
                        temp.append([fullblock] * int(h // 8) + [' '] * int(nrows - int(h // 8)))
                    else:
                        temp.append([fullblock] * int(h // 8) + [blocks[h % 8]] + [' '] * int(nrows - 1 - int(h // 8)))
                else:
                    h2 = int(np.round(h / 8))
                    if h2:
                        temp.append([' '] * (h2-1) + ['♦'] + [' '] * (nrows - h2))
                    else:
                        temp.append([' '] * nrows)

        temp = np.array(temp).transpose()[::-1]
        plot_strs.append([''.join(t) for t in temp])

    s = ''

    for gridr in range(gridrows):

        data_ind = [i for i in range(len(xs)) if i // gridcols == gridr]

        plot_strs_r = [plot_strs[i] for i in data_ind]

        # Plot labels
        for i in data_ind:
            name = labels[i]
            free_space = ncols - len(name[:ncols]) + 2
            s += ' ' * (free_space // 2) + name[:ncols] + ' ' * (free_space - free_space // 2)
            if i != data_ind[-1]:
                s += ' '
        s += '\n'

        # histogram
        s += ' '.join(['┌' + '─' * ncols + '┐'] * len(plot_strs_r))

        for r in range(nrows):
            s += '\n' + ' '.join(['│' + ps[r] + '│' for ps in plot_strs_r])

        s += '\n' + ' '.join(['└' + '─' * ncols + '┘'] * len(plot_strs_r))

        s += '\n'

        # Axis limits
        for i in data_ind:
            min_str = '%.3e' % bounds[i][0]
            max_str = '%.3e' % bounds[i][1]
            if min_str == max_str:
                max_str = '+' + '%.3e' % (bounds[i][1] - bounds[i][0])
            len_i = ncols + 2 - len(min_str) - len(max_str)
            s += f'{min_str}{" " * len_i}{max_str}'
            if i != data_ind[-1]:
                s += ' '

        s += '\n'

    if do_print:
        print(s)
    return s

# def hist(x, x1=None, x2=None, x3=None, *, nrows=25, ncols=100, bins=20,
#          left_censor=0, right_censor=0, do_print=False,
#          xlabel=None,
#          cumulative=False, histtype='bar', density=True, xnames=None):
#
#     assert .5 > left_censor >= 0
#     assert .5 > right_censor >= 0
#
#     x, xmin, xmax = _transform(x, left_censor, right_censor)
#     x1, ymin, ymax = _transform(x1, left_censor, right_censor)
#     x2, zmin, zmax = _transform(x2,  left_censor, right_censor)
#     x3, wmin, wmax = _transform(x3,  left_censor, right_censor)
#
#     assert histtype in ('bar', 'step')
#     if x1 is not None or x2 is not None or x3 is not None:
#         histtype = 'step'
#
#     assert ncols % bins == 0
#
#     blocks = ' ▁▂▃▄▅▆▇█' # 1/8 to full
#     markers = dict(zip('xyzw', '♦○+★'))
#
#     x_axis_min, x_axis_max = np.nanmin([xmin, ymin, zmin, wmin]), np.nanmax([xmax, ymax, zmax, wmax])
#     min_x_str = '%10.3e' % x_axis_min
#     max_x_str = '%10.3e' % x_axis_max
#     if min_x_str == max_x_str:
#         max_x_str = '+%9.3e' % (x_axis_max - x_axis_min)
#
#     transforms = [
#         ((a - x_axis_min) / (x_axis_max - x_axis_min) * (bins - 1)).astype(int)
#         for a in [x, x1, x2, x3]
#         if a is not None
#     ]
#
#     if xnames is None:
#         xnames = [f'x{j}' for j in range(len(transforms))]
#
#     uniques = [dict(zip(('unique', 'counts'), np.unique(a, return_counts=True)))
#                     for a in transforms] # unique, counts
#
#     if cumulative:
#         for u in uniques:
#             u['counts'] = np.cumsum(u['counts'])
#
#     if density:
#         max_n =  max([len(data) for data in [x, x1, x2, x3]
#                       if data is not None])
#         for u, data in zip(uniques,  [x, x1, x2, x3]):
#             u['counts'] = np.round(u['counts'] * max_n / len(data)).astype(int)
#
#     bars = []
#
#     s = ''
#
#     if histtype == 'bar':
#
#         counts = uniques[0]['counts']  # assumes if bar, only one series
#
#         counts = ((counts / max(counts)) * (nrows * 8)).astype(int)
#
#         for i, c in enumerate(counts):
#             bars.append(blocks[-1] * (c // 8) + blocks[c % 8])
#
#         bars = np.repeat(bars, ncols // bins)
#
#         for j in range(nrows):
#             temp = '\n'
#             for b in bars:
#                 temp += b[j] if j < len(b) else ' '
#             s = temp + s
#
#     else:
#
#         arr = np.zeros((ncols, len(uniques)))
#         bin_2_col = ncols // bins
#         for j, u in enumerate(uniques):
#             for i, c in zip(u['unique'], u['counts']):
#                 for k in range(i * bin_2_col, (i + 1) * bin_2_col):
#                     arr[k, j] = c
#
#         arr = np.round(arr / np.max(arr) * nrows).astype(int)
#
#         temp_strs = []
#         for i in range(ncols):
#             d = list(zip(('x', 'y', 'z', 'w')[:len(uniques)], arr[i]))
#             d = sorted(d, key=lambda p: p[1])
#             temp = ''
#             val_last = np.nan
#             cnt = 0
#             for lab, val in d:
#                 if val == val_last:
#                     continue
#                 if val > 0:
#                     temp += (val - len(temp) - 1) * ' ' + markers[lab]
#                     cnt += val
#                     val_last = val
#             if len(temp) < nrows:
#                 temp += (nrows - len(temp)) * ' '
#             temp_strs.append(temp)
#
#         arr_str = np.empty((nrows, ncols), dtype='str')
#         for i in range(nrows):
#             for j in range(ncols):
#                 arr_str[i, j] = temp_strs[j][nrows-i-1]
#             s += '\n' + ''.join(arr_str[i])
#
#     if xlabel is None:
#         xlabel = ''
#
#     s =  '─' * ncols + s + '\n' + '─' * ncols
#     s += '\n' + min_x_str + ' ' * ((ncols - len(xlabel) - 22) // 2) + xlabel + ' ' * (
#             (ncols - len(xlabel) - 22) // 2) + max_x_str
#
#     s += f'\n ({", ".join([f"{v}={s}" for v, s in zip(xnames, markers.values())])})'
#
#     splits = s.split('\n')
#     for i, st in enumerate(splits):
#         if i == 0:
#             pref, suff = '┌', '┐'
#         elif i < nrows + 1:
#             pref, suff = '│', '│'
#         elif i == nrows + 1:
#             pref, suff = '└', '┘'
#         else:
#             pref, suff = ' ', ' '
#         splits[i] = f'{pref}{st}{suff}'
#     s = '\n'.join(splits)
#
#     if do_print:
#         print(s)
#
#     return s

# TODO honestly not needed given scatter, should delete
# def ascii_hist2d(x, y, xbins=20, ybins=25, nrows=25, ncols=100, censor=0, do_print=False, xlabel=None, ylabel=None):
#
#     assert ncols % xbins == 0
#     assert nrows % ybins == 0
#     x, y = np.asarray(x), np.asarray(y)
#
#     xmin, xmax = min(x), max(x)
#     x_transform = ((x - xmin) / (xmax - xmin) * (xbins - 1)).astype(int)
#
#     ymin, ymax = min(y), max(y)
#     y_transform = ((y - ymin) / (ymax - ymin) * (ybins - 1)).astype(int)
#
#     arr = np.zeros((xbins, ybins))
#     for i, j in zip(x_transform, y_transform):
#         arr[i, j] += 1
#     arr /= np.max(arr)
#     arr_nz = arr[arr>0].ravel()
#     rng = min(arr_nz), max(arr_nz)
#
#     fills = '█▓▒░○·'[::-1]
#
#     s = ''
#     x_mult = ncols // xbins
#     y_mult = nrows // ybins
#     for j in range(nrows):
#         _j = j // y_mult
#         temp = ''
#         for i in range(ncols):
#             _i = i // x_mult
#             if arr[_i, _j] > 0:
#                 temp += fills[int(
#                     (arr[_i, _j] - rng[0]) / (rng[1] - rng[0]) * 5
#                 )]
#             else:
#                 temp += ' '
#         s = temp + '\n' + s
#
#     return s

# if __name__ == '__main__':
# #     # np.random.seed(0)
# #     # n = 100591
# #     # y = 3.4 * np.random.randn(n) + .0025 / (n / 1000) * np.arange(n) - .00005 / (n / 1000) ** 2 * np.arange(n) ** 2
# #     #
# #     #
# #     #
# #     # print(ascii_plot(y, coverage=.9, ncols=165, nrows=15))
# #     np.random.seed(0)
# # #
# #     n = 60041
# #     x = (5.5 + np.random.rand(n)*10)
# #     y = 5000 + 6 * np.sqrt(x) + 1.5 * np.random.randn(n)
# #     #print(ascii_scatter(x, y, nrows=14, ncols=50, xlabel='x stuffff', ylabel='y stu4424rsfff'))
#
#     #print(ascii_hist(x, bins=33, ncols=99, nrows=10, xlabel='hello world'))
#
#     import matplotlib.pyplot as plt
#     # plt.hist(x, bins=25)
#
#     #print(ascii_hist2d(x, y, xbins=10, ybins=10, ncols=50, nrows=20))
#     # print(scatter(x, y, nrows=50, ncols=160, xlabel='x stuff'))
#     #print(plot(sorted(y), ncols=100, coverage=.9, nrows=30))
#     # plt.hist2d(x,y, bins=20)
#     # plt.show()
#
#     n = 10000
#     y = np.exp(1.4 + .5 * np.random.randn(n))
#     # print(hist(y, nrows=15, ncols=160, bins=80,
#     #                xlabel='x stuff', censor=.005, cumulative=False, histtype='bar'))
#     #
#     # print(hist(y, nrows=15, ncols=160, bins=80,
#     #            xlabel='x stuff', censor=.005, cumulative=True))
#     #
#     # print(hist(y, (2 + 1.5 * y)[::2], nrows=15, ncols=150, bins=50,
#     #        xlabel='x stuff', censor=.005, cumulative=False, histtype='bar', density=False))
#     # print(hist(y, x2=(2 + 1.5 * y)[::2], x1=2*y, nrows=15, ncols=150, bins=75,
#     #            xlabel='x stuff', left_censor=0.0, right_censor=.01,
#     #            density=True, cumulative=False, histtype='bar'))
# if __name__ == '__main__':
#     n = 60041
#     x = np.random.randn(n)
#     y = x ** 2 + np.random.randn(n) + 1.7 * (x+1)
#     y = y[len(y)//2:]
#     #print(scatter(x, y, ncols=
# if __name__ == '__main__':
#     n = 1000
#     x = np.random.randn(n)
#     y = 1.6*x
#     y = y[len(y)//3:]
#     #y = y[len(y)//2:]
#     #print(scatter(x, y, ncols=25, nrows=15))
#     print(scatter(x, x, ncols=100, nrows=30, shade=False))
#     #print(plot(x))
#     # print(hist(y, ncols=60, nrows=10, bins=30, right_censor=.001, histtype='step'))
#     # print(hist(x, nrows=20, ncols=150, bins=30))
#     # print(hist(x, y, nrows=15, ncols=120, bins=40, histtype='step',
#     #            density=True, left_censor=.001, right_censor=.001))
#     # print(hist(x, y, nrows=15, ncols=120, bins=40, histtype='step',
#     #            density=False, left_censor=.001, right_censor=.001))
#     # print(hist(x, nrows=15, ncols=120, bins=10, histtype='bar',
#     #            density=True, left_censor=.0001, right_censor=.0001))
#     # print(hist(x, x[:len(x)//2], nrows=15, ncols=120, bins=10, histtype='step',
#     #            density=False, left_censor=.0001, right_censor=.0001))

#
# if __name__ == '__main__':
#
#     # import numpy as np
#     # x = np.random.randn(200005)
#     # x = np.hstack([x, x+3])
#     # print(hist(x, nrows=10, ncols=100, bins=100))
#
#     import numpy as np
#     n = 100
#     x = np.random.randn(n)
#     y = 1.2 + .36*x**3 + np.random.randn(n)
#
#     print(scatter(x, y, nrows=30, ncols=80, shade=False, marker='x' ))