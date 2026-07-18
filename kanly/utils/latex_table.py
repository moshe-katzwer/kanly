from __future__ import absolute_import, print_function


def fmt_number(num, sigfigs):
    """Format a number for inclusion in a LaTeX table cell.

    Uses scientific notation for values with very large or very small
    magnitudes; otherwise uses fixed-point notation.

    Args:
        num: Numeric value to format.
        sigfigs: Number of significant figures (controls decimal places).

    Returns:
        Formatted numeric string suitable for a LaTeX table cell.
    """
    if abs(num) > 10_000 or abs(num) < .001:
        return f"%.{sigfigs - 1}e" % num
    else:
        return f"%.{sigfigs}f" % num


def fmt_p_value(num, sigfigs):
    """Format a p-value for inclusion in a LaTeX table cell.

    Returns the string ``'<.001'`` for very small p-values; otherwise formats
    to three decimal places.

    Args:
        num: Scalar p-value in [0, 1].
        sigfigs: Unused; included for API symmetry with ``fmt_number``.

    Returns:
        Formatted p-value string.
    """
    if num < .001:
        return '<.001'
    else:
        return '%.3f' % num


def get_num_stars(p):
    """Map a p-value to the conventional number of significance stars.

    Thresholds:
        - 4 stars: p < 0.001
        - 3 stars: p < 0.01
        - 2 stars: p < 0.05
        - 1 star:  p < 0.10
        - 0 stars: p >= 0.10

    Args:
        p: Scalar p-value in [0, 1].

    Returns:
        Integer in {0, 1, 2, 3, 4}.
    """
    if p < .001:
        return 4
    if p < .01:
        return 3
    if p < .05:
        return 2
    if p < .1:
        return 1
    return 0


def latex_table(fits, sigfigs=3, show_bse=False, show_stars=True, show_p=False, show_t=False):
    """Generate a LaTeX ``tabular`` environment string for a set of regression results.

    Builds a table with one column per fit and one row per unique parameter,
    annotated with significance stars and optionally with standard errors,
    p-values, or t-statistics.  A footer lists the specification names of each
    model.

    Args:
        fits: List of ``RegressionResultsBase``-compatible result objects.
            Each must expose a ``params`` attribute (a ``pd.Series`` keyed by
            parameter name).
        sigfigs: Significant figures used when formatting numeric cells.
        show_bse: When ``True``, add a row of standard errors below each
            coefficient.  At most one of ``show_bse``, ``show_p``, ``show_t``
            may be ``True``.
        show_stars: When ``True``, append ``*`` symbols to coefficient cells
            according to the p-value of each parameter.
        show_p: When ``True``, add a row of formatted p-values.
        show_t: When ``True``, add a row of formatted t-statistics.

    Returns:
        String containing a complete LaTeX ``tabular`` environment (including
        surrounding ``\\begin{center}`` / ``\\end{center}`` wrappers).

    Examples
    --------
    Two-model LaTeX regression table with standard errors and stars:

    >>> import numpy as np, pandas as pd
    >>> from kanly.api import lm, latex_table
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({'x': rng.normal(size=200),
    ...                    'z': rng.normal(size=200)})
    >>> df['y'] = 1.0 + 2.0*df['x'] + rng.normal(size=200)
    >>> fit1 = lm('y ~ x',     df, specification_name='OLS')
    >>> fit2 = lm('y ~ x + z', df, specification_name='OLS+z')
    >>> print(latex_table([fit1, fit2],                    # doctest: +SKIP
    ...                   show_bse=True, show_stars=True))
    \\begin{center}
    \\begin{tabular}{ c c c  }
    \\hline
       & (1) & (2) \\
       \\hline
    \\textbf{Intercept} & 1.02*** & 1.02*** \\
    ...
    \\end{tabular}
    \\end{center}

    Paste the output directly into a LaTeX document.
    """
    assert 0 <= show_t + show_p + show_bse <= 1

    k = len(fits)
    param_names = dict()
    for f in fits:
        param_names |= {c: '' for c in f.params.index}

    result = f"""
\\begin{{center}}
\\begin{{tabular}}{{ {'c ' * (k + 1)} }}
\\hline
  {' & ' + ' & '.join([f'({j + 1})' for j in range(k)])} \\\\
  \\hline
"""

    for c in param_names:
        result += f"\n\\textbf{{{c}}}"
        for f in fits:
            c_in = c in f.params.index
            if c_in:
                result += f" & {fmt_number(f.params[c], sigfigs)}"
                if show_stars and hasattr(f, 'pvalues'):
                    result += "*" * get_num_stars(f.pvalues[c])
                result += " "
            else:
                result += f" & "

        result += f'\\\\'

        if show_bse or show_p or show_t:
            prefix = '' if show_bse else ('\\textit{p}=' if show_p else '\\textit{t}=')
            result += '\n'
            for f in fits:
                attr = 'bse' if show_bse else ('pvalues' if show_p else 'tvalues')
                c_in = c in f.params.index and hasattr(f, attr)
                fmt_temp = fmt_p_value if show_p else fmt_number
                if c_in:
                    num = getattr(f, attr)[c]
                    num = prefix + fmt_temp(num, sigfigs)
                    result += f" & \\small{{({num})}}"
                else:
                    result += f" & "

            result += "\\\\[1ex]"

    result += f"""
\\hline
\\end{{tabular}}

"""

    result += '\n'.join(
        [f'({j + 1}) {"" if getattr(f, "specification_name", "") is None else getattr(f, "specification_name", "")}'
         for j, f in enumerate(fits)])
    result += '\n\\hline'

    result += f"""
\\end{{center}}
"""

    return result
