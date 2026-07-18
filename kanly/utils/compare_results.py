from __future__ import absolute_import, print_function

import numpy as np
import pandas as pd
import textwrap

from kanly.regression.regression_results_base import \
    RegressionResultsBase, FOUR_STAR_LEVEL, THREE_STAR_LEVEL, TWO_STAR_LEVEL, ONE_STAR_LEVEL
from kanly import __version__


class TempResultObject(object):
    """
    Minimal duck-type stand-in for a ``RegressionResultsBase`` object.

    Used by ``compare_results`` to display a reference parameter vector
    alongside real regression results without requiring a full results object.
    All non-``params`` attributes are initialized to empty strings so that the
    summary-table code can call ``getattr`` on them without raising.
    """
    def __init__(self, params):
        """Initialize with a parameter Series and empty-string placeholders.

        Args:
            params: ``pd.Series`` (or dict-like) of parameter values keyed by
                parameter name.
        """
        self.params = params
        for attr in ('nobs', 'cov_type', 'rsquared', 'rsquared_adj',
                     'method', 'weights_name', 'df_resid', 'df_model'):
            self.__setattr__(attr, '')
        self.model = object()


def fmt_p_value(p):
    """Format a p-value for display in a results table.

    Values above 0.001 are shown as ``p=0.XXX``; values at or below 0.001
    are displayed as the string ``'p<.001'``.

    Args:
        p: Scalar p-value in [0, 1].

    Returns:
        Formatted string representation of the p-value.
    """
    return f'p={"%.3f" % p}' if p > .001 else 'p<.001'


def fmt_t_value(t, sigfigs):
    """Format a t-statistic for display in a results table.

    Uses scientific notation for very small or very large absolute values,
    otherwise fixed-point notation, both with ``sigfigs-1`` decimal places.

    Args:
        t: Scalar t-statistic.
        sigfigs: Total significant figures to display.

    Returns:
        Formatted string prefixed with ``'t='``.
    """
    if abs(t) < .001 or abs(t) > 10_000:
        return f't=%.{sigfigs - 1}e' % t
    else:
        return f't=%.{sigfigs - 1}f' % t


def fmt_bse_value(se, sigfigs):
    """Format a standard error for display in a results table (parenthesised).

    Uses scientific notation for very small or very large values, otherwise
    fixed-point notation, both with ``sigfigs-1`` decimal places.

    Args:
        se: Non-negative scalar standard error.
        sigfigs: Total significant figures to display.

    Returns:
        Formatted string enclosed in parentheses, e.g. ``'(0.0123)'``.
    """
    if se < .001 or se > 10_000:
        return f'(%.{sigfigs - 1}e)' % se
    else:
        return f'(%.{sigfigs - 1}f)' % se


def fmt_param_value(b, sigfigs):
    """Format a parameter estimate for display in a results table.

    Uses scientific notation for very small or very large absolute values,
    otherwise fixed-point notation, both with ``sigfigs-1`` decimal places.

    Args:
        b: Scalar parameter estimate.
        sigfigs: Total significant figures to display.

    Returns:
        Formatted numeric string without any prefix.
    """
    if abs(b) < .001 or abs(b) > 10_000:
        return f'%.{sigfigs - 1}e' % b
    else:
        return f'%.{sigfigs - 1}f' % b


def compare_results(
        fit_list, show_stars=False, show_bse=True, show_p=False, show_t=False, show_specification_name=True,
        parameter_subset=None, sig_figs=4, fit_titles=None, ref_param_values=None, ref_name=None,
        show_formulas=False, print_result=False, suptitle=None,
        show_cost=False, show_converged=False, show_penalty=False,
        one_star_level=ONE_STAR_LEVEL,
        two_star_level=TWO_STAR_LEVEL,
        three_star_level=THREE_STAR_LEVEL,
        four_star_level=FOUR_STAR_LEVEL,
        sort_param_names=False,
):
    """Build a publication-style side-by-side regression results table.

    Collates coefficient estimates (and optionally standard errors, t-stats,
    p-values, and significance stars) from multiple fitted models into a
    single formatted string, similar to Stata's ``esttab`` or R's ``stargazer``.
    A reference column of known true values can be appended for simulation studies.

    Args:
        fit_list: List of ``RegressionResultsBase``-compatible result objects to
            compare.  Each must expose at least a ``params`` attribute.
        show_stars: Whether to append significance star annotations to
            coefficient estimates.
        show_bse: Whether to display a row of standard errors below each
            coefficient.
        show_p: Whether to display a row of formatted p-values.
        show_t: Whether to display a row of formatted t-statistics.
        show_specification_name: Whether to append a model-name footer.
        parameter_subset: Optional list of parameter names to include;
            ``None`` shows all parameters found across all fits.
        sig_figs: Total significant figures for numeric formatting.
        fit_titles: Column header labels; auto-generated as ``(0)``, ``(1)``,
            … when ``None``.
        ref_param_values: Optional dict of ``{param_name: value}`` reference
            values to display as an extra column.
        ref_name: Display label for the reference column.
        show_formulas: Whether to append model formula strings below the table.
        print_result: When ``True`` prints the result; when ``False`` returns
            the formatted string.
        suptitle: Optional title line; defaults to ``'Regression Summary Table'``.
        show_cost: Whether to append the cost (SSR/2) row.
        show_converged: Whether to append a convergence-status row.
        show_penalty: Whether to append a penalty row.
        one_star_level: p-value threshold for ``*`` annotation.
        two_star_level: p-value threshold for ``**`` annotation.
        three_star_level: p-value threshold for ``***`` annotation.
        four_star_level: p-value threshold for ``****`` annotation.
        sort_param_names: Whether to sort parameter names alphabetically.

    Returns:
        Formatted multi-line string when ``print_result=False``; prints and
        returns ``None`` when ``print_result=True``.

    Examples
    --------
    Side-by-side comparison of three OLS specifications:

    >>> import numpy as np, pandas as pd
    >>> from kanly.api import lm, compare_results
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({'x':  rng.normal(size=200),
    ...                    'x2': rng.normal(size=200)})
    >>> df['y'] = 1.0 + 2.0*df['x'] - 0.5*df['x2'] + rng.normal(size=200)
    >>> fit1 = lm('y ~ x',         df, specification_name='simple')
    >>> fit2 = lm('y ~ x + x2',    df, specification_name='with x2')
    >>> fit3 = lm('y ~ x*x2',      df, specification_name='with x:x2')
    >>> print(compare_results([fit1, fit2, fit3],          # doctest: +SKIP
    ...                       show_stars=True, show_bse=True))
                          (0)         (1)         (2)
    Intercept           1.020       1.018       1.020
                       (0.071)     (0.071)     (0.071)
    x                   1.980       1.984       1.986
                       (0.071)     (0.071)     (0.072)
    x2                              -0.485      -0.487
                                   (0.071)     (0.071)
    x:x2                                         0.012
                                                (0.073)

    Pass a dict to ``ref_param_values`` to display known true values as an
    additional column for simulation studies.
    """

    assert 0 < four_star_level < three_star_level < two_star_level < one_star_level < 1

    if fit_titles is None:
        fit_titles = ['(%d)' % i for i in range(len(fit_list))]

    has_true_params = ref_param_values is not None
    if has_true_params:
        fit_list.append(TempResultObject(pd.Series(ref_param_values)))
        fit_titles.append('Reference' if ref_name is None else ref_name)

    len_title_max = np.max([len(t) for t in fit_titles])
    len_title_max = np.clip(len_title_max, 7, 20)

    param_names = dict()
    for f in fit_list:
        param_names |= {c: 0 for c in f.params.index
                       if parameter_subset is None or c in parameter_subset}
    param_names = list(param_names.keys())
    if sort_param_names:
        param_names = sorted(param_names)

    dfs = dict()
    attrs = ['params', 'bse', 'pvalues', 'tvalues']
    num_regressors = len(param_names)

    for attr in attrs:
        df_coef = pd.DataFrame(
            index=param_names,
            columns=fit_titles,
            data=[[''] * len(fit_list)] * len(param_names)
        )
        for i, (f, title) in enumerate(zip(fit_list, fit_titles)):
            if attr != 'stars':

                if has_true_params and attr != 'params' and i == len(fit_list) - 1:
                    continue

                if not hasattr(f, attr):
                    idx, vals = [], []
                else:
                    idx = list(set(getattr(f, attr).index) & set(param_names))
                    vals = getattr(f, attr)[idx]

                if attr == 'pvalues':
                    vals = [fmt_p_value(p) for p in vals]
                elif attr == 'tvalues':
                    vals = [fmt_t_value(t, sig_figs) for t in vals]
                elif attr == 'bse':
                    vals = [fmt_bse_value(b, sig_figs) for b in vals]
                elif attr == 'params':

                    if show_stars and hasattr(f, 'pvalues'):
                        stars = [
                            '  ****' if p < four_star_level
                            else ('  ***' if p < three_star_level
                                  else ('  **' if p < two_star_level
                                        else ("  *" if p < one_star_level else '')))
                            for p in getattr(f, 'pvalues')[idx]
                        ]
                    else:
                        stars = [''] * num_regressors

                    vals = [fmt_param_value(v, sig_figs) for v in vals]
                    vals = [s + " " + v for v, s in zip(vals, stars)]

                df_coef.loc[idx, title] = vals

        if attr != 'params':
            df_coef.index = [''] * num_regressors
        dfs[attr] = df_coef

    dfs_list = [dfs['params']]
    if show_bse:
        dfs_list.append(dfs['bse'])
    if show_t:
        dfs_list.append(dfs['tvalues'])
    if show_p:
        dfs_list.append(dfs['pvalues'])
    num_stats = len(dfs_list)

    def _get_fit_attr(attr_name):
        """Collect a scalar attribute from every fit object as an object array.

        Args:
            attr_name: Name of the attribute to retrieve from each result object.

        Returns:
            NumPy object array of length ``len(fit_list)`` containing the
            attribute value (or an empty string if the attribute is absent).
        """
        return np.array([getattr(f, attr_name, '') for f in fit_list], dtype=object)

    df_temp = pd.DataFrame(columns=fit_titles)
    df_temp.loc['Model:', :] = [getattr(f, 'get_result_type', lambda: '')() for f in fit_list]
    df_temp.loc['Outcome:', :] = [getattr(f.model, 'endog_name', '') for f in fit_list]
    df_temp.loc['No. Obs.', :] = _get_fit_attr('nobs')

    def get_rsquared(f):
        """Return a formatted R-squared (or score) string for a result object.

        Tries ``f.rsquared`` first, then falls back to ``f.score``; returns
        ``'-'`` when the attribute is present but not a numeric type, and
        ``''`` when neither attribute exists.

        Args:
            f: A regression result object (or ``TempResultObject``).

        Returns:
            4-decimal-place R-squared string, ``'-'``, or ``''``.
        """
        if hasattr(f, 'rsquared'):
            if not isinstance(f.rsquared, (float, int)):
                return '-'
            return "%.4f" % f.rsquared
        elif hasattr(f, 'score'):
            if not isinstance(f.score, (float, int)):
                return '-'
            return "%.4f" % f.score
        else:
            return ''

    df_temp.loc['R-squared: ', :] = [get_rsquared(f) for f in fit_list]
    df_temp.loc['R-squared Adj.: ', :] = ["%.4f" % f.rsquared_adj
                                          if (hasattr(f, 'rsquared_adj') and f.rsquared != '') else ''
                                          for f in fit_list]
    df_temp.loc['Pseudo R-squared: ', :] = ["%.4f" % f.pseudo_rsquared
                                            if (hasattr(f, 'pseudo_rsquared') and f.pseudo_rsquared != '') else ''
                                            for f in fit_list]

    df_temp.loc['Method:', :] = _get_fit_attr('method')
    df_temp.loc['Weights:', :] = _get_fit_attr('weights_name')
    df_temp.loc['Df Residuals: ', :] = _get_fit_attr('df_resid')
    df_temp.loc['Df Model:', :] = _get_fit_attr('df_model')
    df_temp.loc['Covariance Type:', :] = _get_fit_attr('cov_type')
    if show_converged:
        df_temp.loc['Converged:', :] = _get_fit_attr('converged')
    if show_cost:
        df_temp.loc['Cost (SSR/2):', :] = np.array(['%.4e' % getattr(f, 'cost', np.nan) for f in fit_list], dtype=object)
    if show_penalty:
        df_temp.loc['Penalty:', :] = np.array(['%.4e' % getattr(f, 'penalty', np.nan) for f in fit_list], dtype=object)

    df_sum = pd.concat(dfs_list + [df_temp])
    if has_true_params:
        df_sum[' | '] = ' | '
        df_sum = df_sum[fit_titles[:-1] + [" | "] + [fit_titles[-1]]]
    df_str = df_sum.to_string().split('\n')

    width = max(60, len(df_str[0]))
    bar = '─' * width
    dbl_bar = '═' * width

    ret_str = ""

    ret_str += "\n" + dbl_bar
    ret_str += "\n" + ("Regression Summary Table" if suptitle is None else str(suptitle))
    ret_str += "\n" + dbl_bar
    ret_str += "\n" + df_str[0]
    ret_str += "\n" + bar
    for i in range(num_regressors):
        for j in range(num_stats):
            ret_str += "\n" + df_str[1 + j * num_regressors + i]
        if i < num_regressors - 1 and (show_bse or show_p):
            ret_str += 2 * "\n"

    ret_str += "\n" + dbl_bar

    for k in range(1, len(df_temp) + 1)[::-1]:
        ret_str += "\n" + df_str[-k]

    if has_true_params:
        fit_titles = fit_titles[:-1]
        fit_list = fit_list[:-1]

    if show_formulas:
        ret_str += "\n" + bar

        for i, (f, title) in enumerate(zip(fit_list, fit_titles)):
            if isinstance(f, RegressionResultsBase):
                s = str(f.model.__dict__.get('endog_name', 'y') + " ~ "
                        + " + ".join(f.model.__dict__.get('exog_term_names', f.exog_names)))
            else:
                s = f.model.__dict__.get('formula', None)
            if hasattr(f.model, 'instrument_names') and f.model.instrument_names is not None:
                if len(f.model.instrument_names):
                    s += ", Instruments: {%s}" % ", ".join(f.model.instrument_names)
            if hasattr(f, 'absorb_info') and f.absorb_info is not None:
                s += ", Absorbed: " + str(f.absorb_info.absorb_name) + " [num=%d" % f.absorb_info.num_absorbed + "]"
            s = textwrap.wrap(s, width - 7)
            for n, l in enumerate(s):
                if n == 0:
                    ret_str += "\n" + (("%-" + ("%d" % len_title_max) + "s") % title) + l
                else:
                    ret_str += "\n" + ' ' * 7 + l

    if show_specification_name:
        print_bar = True
        for i, (f, title) in enumerate(zip(fit_list, fit_titles)):
            if f.__dict__.get('specification_name', None) is not None:
                if print_bar:
                    ret_str += "\n" + bar
                s = f'{title}  "{str(f.specification_name)}"'
                ret_str += "\n" + s
                print_bar = False
            elif f.model is not None and f.model.__dict__.get('specification_name', None) is not None:
                if print_bar:
                    ret_str += "\n" + bar
                s = title + " " + str(f.model.specification_name)
                ret_str += "\n" + s
                print_bar = False

    if show_stars:
        ret_str += "\n" + bar
        ret_str += f"\n" + f'**** p < {four_star_level}'
        ret_str += f"\n" + f' *** p < {three_star_level}'
        ret_str += f"\n" + f'  ** p < {two_star_level}'
        ret_str += f"\n" + f'   * p < {one_star_level}'

    ret_str += "\n" + dbl_bar
    ret_str += "\n" + (" " * max(width - 11 - len(__version__), 0)) + "[kanly, v=%s]\n" % __version__

    if print_result:
        print(ret_str)
    else:
        return ret_str
