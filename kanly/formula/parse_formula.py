"""Formula-string parsing utilities for sparse design-matrix construction.

This module splits high-level formula strings into endog/exog/instrument/
weight blocks, delegates core term parsing to Patsy ``ModelDesc``, and applies
kanly-specific extensions such as ``poly(...)`` expansion and token-based null
row detection.
"""
from __future__ import absolute_import, print_function

import re
import sys

from patsy.highlevel import ModelDesc

from kanly.formula.keys import (ENDOG_KEY, EXOG_KEY, INSTRUMENTS_KEY, WEIGHTS_KEY, FORMULA_KEY, ABSORB_KEY)
from kanly.formula.sparse_term import SparseTerm
from kanly.formula.polynomial import replace_poly_in_var_names_with_monomials_exploded
from kanly.formula.invalid_row_functions import (default_numerical_invalid_row_func,
                                                 default_categorical_invalid_row_func, default_weights_invalid_row_func,
                                                 is_numeric_dtype)


from kanly.stats.common import (
    jit, njit, numpy, np, np_linalg, scipy, sp, sp_linalg, sp_special, stats, logit, log_d_expit, expit, d_expit, sin,
    cos, tan, arcsin, arccos, arctan, cbrt, sqrt, log, log2, log10, exp, std_normal_cdf, normal_cdf, normal_pdf,
    normal_logpdf, log_normal_pdf, log_normal_logpdf, log_normal_cdf, logpdf_beta, logpdf_cauchy, logpdf_chi2,
    logpdf_expon, logpdf_f, logpdf_gamma, logpdf_genextreme, logpdf_halfcauchy, logpdf_halfnorm, logpdf_invgamma,
    logpdf_laplace, logpdf_logistic, logpdf_lognorm, logpdf_multivariate_normal, logpdf_multivariate_t, logpdf_norm,
    logpdf_pareto, logpdf_t, logpdf_truncnorm, logpdf_weibull_min, nopython_logpdf_beta, nopython_logpdf_cauchy,
    nopython_logpdf_chi2, nopython_logpdf_expon, nopython_logpdf_f, nopython_logpdf_gamma, nopython_logpdf_genextreme,
    nopython_logpdf_halfcauchy, nopython_logpdf_halfnorm, nopython_logpdf_invgamma, nopython_logpdf_laplace,
    nopython_logpdf_logistic, nopython_logpdf_lognorm, nopython_logpdf_multivariate_normal,
    nopython_logpdf_multivariate_t, nopython_logpdf_norm, nopython_logpdf_pareto, nopython_logpdf_t,
    nopython_logpdf_truncnorm, nopython_logpdf_weibull_min, nopython_pdf_beta, nopython_pdf_cauchy, nopython_pdf_chi2,
    nopython_pdf_expon, nopython_pdf_f, nopython_pdf_gamma, nopython_pdf_genextreme, nopython_pdf_halfcauchy,
    nopython_pdf_halfnorm, nopython_pdf_invgamma, nopython_pdf_laplace, nopython_pdf_logistic, nopython_pdf_lognorm,
    nopython_pdf_multivariate_normal, nopython_pdf_multivariate_t, nopython_pdf_norm, nopython_pdf_pareto,
    nopython_pdf_t, nopython_pdf_truncnorm, nopython_pdf_weibull_min, pdf_beta, pdf_cauchy, pdf_chi2, pdf_expon, pdf_f,
    pdf_gamma, pdf_genextreme, pdf_halfcauchy, pdf_halfnorm, pdf_invgamma, pdf_laplace, pdf_logistic, pdf_lognorm,
    pdf_multivariate_normal, pdf_multivariate_t, pdf_norm, pdf_pareto, pdf_t, pdf_truncnorm, pdf_weibull_min,
    __frozen_internal_logpdf_genextreme, __frozen_internal_logpdf_norm, __frozen_internal_logpdf_truncnorm,
    __frozen_internal_logpdf_beta, __frozen_internal_logpdf_cauchy, __frozen_internal_logpdf_laplace,
    __frozen_internal_logpdf_expon, __frozen_internal_logpdf_t, __frozen_internal_logpdf_gamma,
    __frozen_internal_logpdf_lognorm, __frozen_internal_logpdf_invgamma, __frozen_internal_logpdf_logistic,
    __frozen_internal_logpdf_chi2, __frozen_internal_logpdf_gennorm, __frozen_internal_logpdf_multivariate_normal,
    __frozen_internal_logpdf_multivariate_t, __frozen_internal_logpdf_halfnorm, __frozen_internal_logpdf_pareto,
    __frozen_internal_logpdf_halfcauchy, __frozen_internal_logpdf_loguniform, __frozen_internal_logpdf_f,
    __frozen_internal_logpdf_weibull_min, __frozen_internal_logpdf_dirichlet, __nopython_frozen_internal_logpdf_norm,
    __nopython_frozen_internal_logpdf_halfnorm, __nopython_frozen_internal_logpdf_beta,
    __nopython_frozen_internal_logpdf_cauchy, __nopython_frozen_internal_logpdf_laplace,
    __nopython_frozen_internal_logpdf_expon, __nopython_frozen_internal_logpdf_t,
    __nopython_frozen_internal_logpdf_multivariate_t, __nopython_frozen_internal_logpdf_gamma,
    __nopython_frozen_internal_logpdf_lognorm, __nopython_frozen_internal_logpdf_invgamma,
    __nopython_frozen_internal_logpdf_logistic, __nopython_frozen_internal_logpdf_chi2,
    __nopython_frozen_internal_logpdf_gennorm, __nopython_frozen_internal_logpdf_multivariate_normal,
    __nopython_frozen_internal_logpdf_truncnorm, __nopython_frozen_internal_logpdf_pareto,
    __nopython_frozen_internal_logpdf_halfcauchy, __nopython_frozen_internal_logpdf_loguniform,
    __nopython_frozen_internal_logpdf_genextreme, __nopython_frozen_internal_logpdf_f,
    __nopython_frozen_internal_logpdf_weibull_min, __nopython_frozen_internal_logpdf_dirichlet,
    get_frozen_logpdf_pareto, get_frozen_logpdf_norm, get_frozen_logpdf_truncnorm, get_frozen_logpdf_halfnorm,
    get_frozen_logpdf_beta, get_frozen_logpdf_cauchy, get_frozen_logpdf_laplace, get_frozen_logpdf_expon,
    get_frozen_logpdf_t, get_frozen_logpdf_gamma, get_frozen_logpdf_invgamma, get_frozen_logpdf_lognorm,
    get_frozen_logpdf_logistic, get_frozen_logpdf_gennorm, get_frozen_logpdf_chi2,
    get_frozen_logpdf_multivariate_normal, get_frozen_logpdf_halfcauchy, get_frozen_logpdf_multivariate_t,
    get_frozen_logpdf_uniform, get_frozen_logpdf_flat, get_frozen_logpdf_loguniform, get_frozen_logpdf_genextreme,
    get_frozen_logpdf_f, get_frozen_logpdf_weibull_min, get_frozen_logpdf_dirichlet, )


for c in ['numpy', 'np', 'scipy', 'sp']:
    if c not in locals():
        raise Exception


class SparseParseFormulaException(Exception):
    """Raised when formula parsing/validation fails in sparse parser helpers."""
    pass


def get_null_indices_for_formula(
        formula, data, absorb=None, numerical_invalid_row_func=None, categorical_invalid_row_func=None,
        weights_invalid_row_func=None, absorb_invalid_row_func=None, return_null_dict_by_col=False, _time=None,
        debug=False):
    """Compute row indices that are invalid for any formula-referenced column.

    Invalidity rules differ by variable type and can be customized via
    callback functions.

    Args:
        formula (str): Full formula string.
        data (DataFrame): Source data.
        absorb (str or list[str] or None): Optional absorb/fixed-effect columns.
        numerical_invalid_row_func (callable, optional): Predicate for numeric
            columns; returns boolean mask of invalid rows.
        categorical_invalid_row_func (callable, optional): Predicate for
            categorical columns.
        weights_invalid_row_func (callable, optional): Predicate for weights.
        absorb_invalid_row_func (callable, optional): Predicate for absorb cols.
        return_null_dict_by_col (bool): If True, also return per-block null map.
        _time (Any): Reserved/unused timing argument retained for compatibility.
        debug (bool): Debug flag propagated to parser.

    Returns:
        set[int] or tuple[set[int], dict]: Union of invalid row indices, and
        optionally a dict keyed by ENDOG/EXOG/INSTRUMENTS/WEIGHTS/ABSORB.
    """

    if numerical_invalid_row_func is None:
        numerical_invalid_row_func = default_numerical_invalid_row_func
    if categorical_invalid_row_func is None:
        categorical_invalid_row_func = default_categorical_invalid_row_func
    if absorb_invalid_row_func is None:
        absorb_invalid_row_func = default_categorical_invalid_row_func
    if weights_invalid_row_func is None:
        weights_invalid_row_func = default_weights_invalid_row_func

    formula_parsed = parse_formula(formula, debug)

    null_dict = dict()

    int_index = np.arange(data.shape[0])

    null_dict[ENDOG_KEY] = set()
    if isinstance(formula_parsed[ENDOG_KEY], str):
        endog_cols = SparseTerm._get_tokens_for_term(formula_parsed[ENDOG_KEY])['column_tokens']
        for c in endog_cols:
            null_dict[ENDOG_KEY] |= set(int_index[numerical_invalid_row_func(data[c])])
    else:
        for y in formula_parsed[ENDOG_KEY]:
            y_cols = SparseTerm._get_tokens_for_term(y)['column_tokens']
            for c in y_cols:
                if is_numeric_dtype(data.dtypes[c]):
                    null_dict[ENDOG_KEY] |= set(int_index[numerical_invalid_row_func(data[c])])
                else:
                    null_dict[ENDOG_KEY] |= set(int_index[categorical_invalid_row_func(data[c])])

    null_dict[EXOG_KEY] = set()
    for x in formula_parsed[EXOG_KEY]:
        x_cols = SparseTerm._get_tokens_for_term(x)['column_tokens']
        for c in x_cols:
            if is_numeric_dtype(data.dtypes[c]):
                null_dict[EXOG_KEY] |= set(int_index[numerical_invalid_row_func(data[c])])
            else:
                null_dict[EXOG_KEY] |= set(int_index[categorical_invalid_row_func(data[c])])

    null_dict[INSTRUMENTS_KEY] = set()
    if formula_parsed[INSTRUMENTS_KEY] is not None:
        for x in formula_parsed[INSTRUMENTS_KEY]:
            x_cols = SparseTerm._get_tokens_for_term(x)['column_tokens']
            for c in x_cols:
                if is_numeric_dtype(data.dtypes[c]):
                    null_dict[INSTRUMENTS_KEY] |= set(int_index[numerical_invalid_row_func(data[c])])
                else:
                    null_dict[INSTRUMENTS_KEY] |= set(int_index[categorical_invalid_row_func(data[c])])

    null_dict[WEIGHTS_KEY] = set()
    if formula_parsed[WEIGHTS_KEY] is not None:
        w_cols = SparseTerm._get_tokens_for_term(formula_parsed[WEIGHTS_KEY])['column_tokens']
        for c in w_cols:
            null_dict[INSTRUMENTS_KEY] |= set(int_index[weights_invalid_row_func(data[c])])

    null_dict[ABSORB_KEY] = set()
    if absorb is not None:
        if isinstance(absorb, str):
            absorb = [absorb]
        absb_cols = SparseTerm._get_tokens_for_term(":".join(absorb))['column_tokens']
        for c in absb_cols:
            null_dict[ABSORB_KEY] |= set(int_index[absorb_invalid_row_func(data[c])])

    null_int_index_rows = set()
    for k, v in null_dict.items():
        null_int_index_rows |= v

    if return_null_dict_by_col:
        return null_int_index_rows, null_dict
    else:
        return null_int_index_rows


def parse_patsy_term_list_to_var_names(patsy_term_list):
    """Convert Patsy term objects into kanly term strings.

    Args:
        patsy_term_list (list[patsy.Term]): Patsy term list from ModelDesc.

    Returns:
        list[str]: Ordered unique term strings with ``poly(...)`` expanded.
    """
    to_return_list = [":".join([r.code for r in z.factors]) #.replace(' ', '')  TODO remove
                      for z in patsy_term_list]
    to_return_list = replace_poly_in_var_names_with_monomials_exploded(to_return_list)
    return list(dict.fromkeys(to_return_list))


def temp_replace(formula, expr, dummy_char):
    """Temporarily replace regex matches with placeholder tokens.

    Args:
        formula (str): Original text.
        expr (str): Regex pattern to replace.
        dummy_char (str): Prefix char for placeholder keys.

    Returns:
        tuple[str, dict]: Updated text and map from original fragment to token.
    """
    dbl_qts = re.findall(expr, formula)
    dbl_qt_dict = {q: dummy_char * 10 + str(i).zfill(10) for i, q in enumerate(dbl_qts)}

    for k, v in dbl_qt_dict.items():
        formula = formula.replace(k, v)

    return formula, dbl_qt_dict


def temp_revert(formula, rep_dict_list):
    """Restore placeholder tokens back to original substrings.

    Args:
        formula (str or None): Candidate formula fragment.
        rep_dict_list (Iterable[dict]): Replacement dictionaries from
            ``temp_replace``.

    Returns:
        str or None: Restored formula fragment.
    """
    if formula is not None:
        for rep_dict in rep_dict_list:
            for k, v in rep_dict.items():
                formula = formula.replace(v, k)
    return formula


def fix_intercept(formula):
    """Normalize ``+ -`` intercept/no-intercept formatting edge cases.

    This avoids ambiguous parse patterns by rewriting accidental ``+   -``
    sequences to canonical `` -``.

    Args:
        formula (str): Raw formula text.

    Returns:
        str: Normalized formula text.
    """
    # careful about no intercept
    to_replace = []
    active = 0
    for c in formula:
        if c == '+':
            active = 1
        elif active:
            if c == ' ':
                active += 1
            elif c == '-':
                to_replace.append('+' + ' ' * (active - 1) + '-')
                active = 0
            else:
                active = 0
    for t in to_replace:
        formula = formula.replace(t, ' -')
    return formula


def parse_formula(formula, debug=False):
    """Parse a high-level formula string into structured term-name blocks.

    Supports kanly separators:
      - ``~``: endog/exog split
      - ``|``: IV instruments
      - ``$``: weights

    Parsing flow:
      1. Normalize intercept formatting and quoting.
      2. Temporarily shadow ``Q(...)`` and quoted strings.
      3. Split weights/instruments blocks.
      4. Revert shadows and delegate each block to Patsy ``ModelDesc``.
      5. Expand polynomial shorthand and enforce intercept/no-intercept marker.

    Args:
        formula (str): Input formula.
        debug (bool): Reserved debug flag for compatibility.

    Returns:
        dict: Keys ``FORMULA``, ``ENDOG``, ``EXOG``, ``INSTRUMENTS``,
        and ``WEIGHTS`` with parsed term-name values.

    Raises:
        Exception: If more than one ``$``, ``|``, or ``~`` token is present,
            or if weights parse to multiple columns.
    """

    if '~' not in formula:
        formula = '1 ~ ' + formula

    formula = fix_intercept(formula)

    formula = formula.replace('\'', '"')
    orig_formula = formula

    # shadow anything in Q(.) or double quotes
    formula, Q_qts_dict = temp_replace(formula, r'Q\(.*?\)', '#')
    formula, dbl_qt_dict = temp_replace(formula, r'".*?"', '@')

    # Make sure *at most one* weighting, endog/exog, and instruments
    if formula.count('$') > 1:
        raise Exception

    if formula.count('|') > 1:
        raise Exception

    if formula.count('~') > 1:
        raise Exception

    # Get Weights
    weights = None
    if formula.count('$'):
        formula, weights = formula.split('$')

    # Get Instruments
    instruments = None
    if formula.count('|'):
        formula, instruments = formula.split('|')

    # Add removed items back in
    formula = temp_revert(formula, (dbl_qt_dict, Q_qts_dict))
    instruments = temp_revert(instruments, (dbl_qt_dict, Q_qts_dict))
    weights = temp_revert(weights, (dbl_qt_dict, Q_qts_dict))

    rlmt = sys.getrecursionlimit()
    sys.setrecursionlimit(5000)
    model_desc = ModelDesc.from_formula(formula)
    sys.setrecursionlimit(rlmt)

    endog_names, exog_names = (
        parse_patsy_term_list_to_var_names(model_desc.lhs_termlist),
        parse_patsy_term_list_to_var_names(model_desc.rhs_termlist)
    )
    # if len(endog_names) != 1:
    #     raise Exception
    # endog_name = endog_names[0]

    if instruments is not None:
        desc = ModelDesc.from_formula(instruments)
        instrument_names = parse_patsy_term_list_to_var_names(desc.rhs_termlist)
    else:
        instrument_names = None

    if weights is not None:
        desc = ModelDesc.from_formula(weights + " -1")
        weights_names = parse_patsy_term_list_to_var_names(desc.rhs_termlist)
        if len(weights_names) != 1:
            raise Exception
        weights_name = weights_names[0]
    else:
        weights_name = None

    for term_list in (exog_names, instrument_names):
        if term_list is not None:
            if '' in term_list:
                term_list.remove('')
            else:
                term_list.append('-1')

    # TODO maybe not?
    # strip_Q((exog_names, endog_names, instrument_names, weights))

    return {
        FORMULA_KEY: orig_formula,
        ENDOG_KEY: endog_names, EXOG_KEY: exog_names,
        INSTRUMENTS_KEY: instrument_names,
        WEIGHTS_KEY: weights_name
    }


def test_tokens(tokens, data, debug=False):
    """Validate that token references resolve to data columns or known globals.

    Args:
        tokens (Iterable[str]): Candidate tokens from parsed expressions.
        data (DataFrame): Input data whose columns should satisfy references.
        debug (bool): Reserved debug flag for compatibility.

    Returns:
        set[str]: Data-column names required by the token set.

    Raises:
        Exception: If a ``Q("...")`` column is missing or an expression token
            does not resolve to a known global symbol.
    """
    data_cols_needed = set()
    for t in tokens:
        if t[:2] == 'Q(':
            z = re.findall(r'".*?"', t)[0].replace('"', '')
            if z not in data.columns:
                raise Exception(f"Q expression term \"{z}\" not found in data columns!")
            data_cols_needed.add(z)
        else:
            if t in data.columns:
                data_cols_needed.add(t)
            else:
                splt = t.split('.')
                if splt[0] not in globals():
                    raise Exception(f"Expression \"{t}\" not found in globals!")
    return data_cols_needed


def strip_Q(var_lists):
    """In-place replacement of ``Q("col name")`` expressions with bare names.

    Args:
        var_lists (Iterable[list[str] or None]): Lists of term strings to
            mutate.
    """
    for vlist in var_lists:
        if vlist is not None:
            for i, v in enumerate(vlist):
                if re.findall(r'Q\(.*?\)', v):
                    vlist[i] = re.findall(r'\".*?\"',
                                          re.findall(r'Q\(.*?\)', v)[0]
                                          )[0].replace('"', '')
