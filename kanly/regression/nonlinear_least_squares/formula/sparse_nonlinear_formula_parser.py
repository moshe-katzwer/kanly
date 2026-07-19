from __future__ import absolute_import, print_function

import re
import string
import time
from itertools import product

import numpy as np
# import autograd.numpy as np  --- maybe one day we'll add this
import pandas as pd
from pandas import DataFrame
from patsy.highlevel import dmatrix

from kanly import _IN_NOTEBOOK
from kanly.formula.sparse_term_to_data_methods import get_nobs_from_index, get_numerical_control_data
from kanly.regression.nonlinear_least_squares.formula.argument_parser import ArgumentParser
from kanly.regression.nonlinear_least_squares.formula.nonlinear_term_type import NonlinearTermType
from kanly.regression.nonlinear_least_squares.function_callables.prediction_function import PredictionFunction

from kanly.utils.linalg_utils import DEFAULT_DENSE_THRESHOLD_MB
from kanly.utils.util import get_eval_env_depth
from kanly.regression.nonlinear_least_squares.constants import DEFAULT_NLLS_JAC_METHOD

from kanly.automatic_differentiation.elementary_functions import *  # don't delete
from kanly.utils.util import dict_2_dataframe

alphabet_string = string.ascii_uppercase


def get_var_letter_prefix(num):
    """
    Returns sequentially variable/coefficient names for
    polynomials, e.g. [_A, _B, ..., _Z, _AA, _AB, ...]
    based on `num`
    """
    if num < 26:
        return '_' + alphabet_string[num % 26]
    else:
        return get_var_letter_prefix(num // 26 - 1) + alphabet_string[num % 26]


def build_prediction_function(exog_names, param_names, func_str, data, do_njit=True, debug=False, index=None,
                              custom_functions=dict(), dense_threshold_mb=DEFAULT_DENSE_THRESHOLD_MB,
                              jac_method=DEFAULT_NLLS_JAC_METHOD):
    """Build a ``PredictionFunction`` from parsed formula components.

    Rewrites formula tokens such as ``{alpha}``, ``[x]``, ``[poly(x, 2)]``, and
    ``[C(group, -1)]`` into Python code that indexes parameter vectors and
    compact data arrays.  The generated code is then wrapped in a
    ``PredictionFunction`` that can evaluate predictions and Jacobians.

    Args:
        exog_names: Data/exogenous term names extracted from ``[...]`` tokens.
        param_names: Explicit nonlinear parameter names extracted from
            ``{...}`` tokens.
        func_str: Formula expression string after the left-hand side has been
            removed.
        data: Input data as a pandas ``DataFrame``.
        do_njit: Whether to Numba-JIT compile the generated prediction function.
        debug: Whether to print parser timing and generated-code diagnostics.
        index: Optional row subset/indexer used when building data arrays.
        custom_functions: Optional mapping of names to functions available to
            generated prediction code.
        dense_threshold_mb: Dense-matrix threshold forwarded to the
            ``PredictionFunction`` Jacobian builder.
        jac_method: Jacobian method used by the resulting ``PredictionFunction``.

    Returns:
        Dict containing the ``PredictionFunction``, validity mask, generated
        code strings, and final parameter count.
    """

    _t = time.time()

    param_names = param_names.copy()
    exog_names = exog_names.copy()
    exog_names = [e for e in exog_names if e[:2] != 'C('] + [e for e in exog_names if e[:2] == 'C(']

    float_dict, int_dict = dict(), dict()

    _, index = get_nobs_from_index(data, index)

    # if 'Intercept' in param_names:
    #     func_str = f'params[{param_cnt}] + ' + func_str
    #     param_cnt += 1

    cnt_msk = dict()
    masks = dict()
    poly_dict = dict()

    for k, param_name in enumerate(param_names):
        func_str = func_str.replace(f'{{{param_name}}}', f"params[{k}]")
    param_cnt = len(param_names)

    num_masks = 0
    num_polynom = 0

    if debug:
        print()

    for i, var_name in enumerate(exog_names):

        _time_var = time.time()

        term_info = NonlinearTermType._get_term_type(var_name)
        term_type, term_arg_str = term_info['term_type'], term_info['term_arg_str']

        if debug:
            print(f"\t\tGetting column '{var_name}' of type '{term_type}'...", end='')

        # univariate polynomial
        if term_type == NonlinearTermType.POLYNOMIAL:

            poly_var_monomial_name, exponents, override_var_name = ArgumentParser._parse_poly_str_to_args(term_arg_str)

            if poly_var_monomial_name in poly_dict:
                suffix = f'_{poly_dict[poly_var_monomial_name]}'
                poly_dict[poly_var_monomial_name] += 1
            else:
                suffix = ''
                poly_dict[poly_var_monomial_name] = 1

            get_monomial_data(
                poly_var_monomial_name, data, float_dict, index)

            mask = f'___mask{num_masks}___'
            num_masks += 1
            masks[mask] = poly_var_monomial_name

            func_str = func_str.replace(
                f'[{var_name}]',
                '(' + ' + '.join([f"params[{param_cnt + d}]" +
                                  (f"*float_dict['{mask}']{'**' + str(e) if e > 1 else ''}" if e > 0 else '')
                                  for d, e in enumerate(exponents)]) + ')',
                1)

            param_cnt += len(exponents)
            coef_letter = get_var_letter_prefix(num_polynom) if override_var_name is None else override_var_name
            num_polynom += 1
            param_names += [f'{coef_letter}[{poly_var_monomial_name}{(f"**{e}" if e > 1 else "")}]' + suffix
                            if e > 0 else
                            f'{coef_letter}[{1}]' + suffix
                            for e in exponents]

        elif term_type == NonlinearTermType.MULTIVARIATE_POLYNOMIAL:

            var_names, exponent, drop1 = ArgumentParser._parse_polym_str_to_args(term_arg_str)

            [get_monomial_data(v, data, float_dict, index)[1] for v in var_names]

            varstr = lambda vname, power: -1 if power == 0 else (
                f"float_dict['{vname}']" if power == 1 else f"float_dict['{vname}']**{power}")

            coef_letter = get_var_letter_prefix(num_polynom)
            num_polynom += 1
            paramstr = lambda var_names, e: (
                    f'{coef_letter}['
                    + ','.join([f"{x}" + (f"**{e_x}" if e_x > 1 else "")
                                for x, e_x in zip(var_names, e) if e_x > 0])
                    + ']'
            )

            exponent_list = sorted([p for p in product(range(0, exponent + 1), repeat=len(var_names))],
                                   key=lambda x: sum(x))

            exponent_list = [e for e in exponent_list if sum(e) <= exponent]
            if drop1:
                exponent_list = [e for e in exponent_list if sum(e) > 1]
            # print(exponent_list)

            ret_strs = []
            for e in exponent_list:
                if sum(e) > 0:
                    ev_strings = np.array([varstr(v, e_v) for v, e_v in zip(var_names, e)], dtype=object)
                    ev_strings = ev_strings[ev_strings != -1]
                    ret_strs.append(f'params[{param_cnt}]*' + '*'.join(ev_strings.flatten()))
                    param_names.append(paramstr(var_names, e))
                    param_cnt += 1

            ret_str = f'({" + ".join(ret_strs)})'

            func_str = func_str.replace(f'[{var_name}]', f'({ret_str})', 1)

        elif term_type == NonlinearTermType.CHEBYSHEV:

            poly_var_monomial_name, max_exponent, drop1, override_var_name \
                = ArgumentParser._parse_cheb_str_to_args(term_arg_str)
            get_monomial_data(poly_var_monomial_name, data, float_dict, index)

            mask = f'___mask{num_masks}___'
            num_masks += 1
            masks[mask] = poly_var_monomial_name

            param_names_temp = []
            coef_letter = get_var_letter_prefix(num_polynom) if override_var_name is None else override_var_name
            num_polynom += 1
            if not drop1:
                param_names_temp.append(f'{coef_letter}[{poly_var_monomial_name},{0}]')
            param_names_temp.append(f'{coef_letter}[{poly_var_monomial_name},{1}]')

            prev0 = 1.0
            prev1 = float_dict[poly_var_monomial_name]
            for t in range(2, max_exponent + 1):
                param_names_temp.append(f'{coef_letter}[{poly_var_monomial_name}_cheb,{t}]')
                float_dict[f'cheb[{poly_var_monomial_name},{t}]'] = (
                        2 * float_dict[poly_var_monomial_name] * prev1 - prev0)
                prev0 = prev1
                prev1 = float_dict[f'cheb[{poly_var_monomial_name},{t}]']

            mask = f'___mask{num_masks}___'
            num_masks += 1
            masks[mask] = var_name

            get_var_for_formula_name \
                = lambda t: ('' if t == 0
                             else (f"*float_dict['{poly_var_monomial_name}']" if t == 1
                                   else f"*float_dict['cheb[{poly_var_monomial_name},{t}]']"
                                   )
                             )

            func_str = func_str.replace(
                f'[{var_name}]',
                '(' + ' + '.join([f'params[{param_cnt + cnt}]' + get_var_for_formula_name(i)
                                  for cnt, i in enumerate(range(drop1, max_exponent + 1))]) + ')',
                1)

            param_names += param_names_temp
            param_cnt += len(param_names_temp)

        # categorical
        elif term_type == NonlinearTermType.CATEGORICAL:

            cat_vals, variables, drop1, override_var_name = get_categorical_data(term_arg_str, data, index=index)
            cat_vals_unique, num_unique, cat_vals_idx = get_categorical_unique(cat_vals)

            if len(variables) > 1:
                var_join = ','.join(variables)
            else:
                var_join = variables[0]

            int_dict[var_join] = cat_vals_idx.values.astype(np.int64)

            if (var_join, override_var_name) in cnt_msk:
                cnt_msk[var_join] += 1
            else:
                cnt_msk[var_join] = 0

            mask = f'___mask{num_masks}___'
            num_masks += 1
            masks[mask] = var_join

            param_names_cat = [f"C({var_join})[{u}]_{cnt_msk[var_join]}" if cnt_msk[var_join]
                               else f"C({var_join})[{u}]"
                               for u in cat_vals_unique[drop1:]]
            if override_var_name is not None:
                param_names_cat = [f'{override_var_name}_{cc}' for cc in param_names_cat]
            param_names += param_names_cat

            start, end = param_cnt, param_cnt + num_unique

            func_str = func_str.replace(
                f'[{var_name}]',
                # Use categorical number to index param array!
                f"np.hstack((np.zeros(1), params[{start}:{end - 1}]))[int_dict['{mask}']]"
                if drop1 else
                f"params[{start}:{end}][int_dict['{mask}']]",
                1)

            param_cnt += num_unique - drop1

        # other
        elif term_type == NonlinearTermType.MONOMIAL:

            is_spline = term_arg_str[:3] in ('bs(', 'cr(', 'cc(')

            get_monomial_data(var_name, data, float_dict, index)

            mask = f'___mask{num_masks}___'
            num_masks += 1
            masks[mask] = var_name

            if is_spline:
                func_str = func_str.replace(
                    f'[{var_name}]',
                    f"np.dot(float_dict['{mask}'], params[{param_cnt}:{param_cnt + float_dict[var_name].shape[1]}])", 1)

                if len(float_dict[var_name].shape) > 1:
                    sub_params = [f'{var_name}_{j}' for j in range(float_dict[var_name].shape[1])]
                else:
                    sub_params = [var_name]

                param_names += sub_params
                param_cnt += len(sub_params)

            else:
                func_str = func_str.replace(f'[{var_name}]', f"float_dict['{mask}']", 1)

        if debug:
            print("%.2fs" % (time.time()-_time_var))

    for k, var_name in masks.items():
        func_str = func_str.replace(k, var_name)

    func_str = func_str.replace('{', '').replace('}', '')
    func_str = '(' + "\n\t + ".join(func_str.split('+')) + ')'

    # Temporarily protect generated newlines/tabs while removing whitespace
    # inside quoted dictionary keys; otherwise keys such as categorical masks
    # can be altered by formula prettification.
    func_str = func_str.replace("\n", '___N___').replace("\t", '___T___')
    func_str_split = re.split('\'(.+?)\'', func_str)
    func_str= ''.join(
        [m if i % 2 == 0 else "'" + m.replace('___N___', '').replace('___T___', '').replace(' ', '') + "'"
         for i, m in enumerate(func_str_split)])
    func_str = func_str.replace('___N___', '\n').replace('___T___', '\t')
    func_str_code = func_str

    func_str_code2 = func_str_code
    for tp, d in {'float': float_dict, 'int': int_dict}.items():
        i = 0
        for k, val in d.items():
            if len(val.shape) == 1 or val.shape[1] == 1:
                indices = f'{i}'
                i += 1
            else:
                indices = f'{i}:{i + val.shape[1]}'
                i += val.shape[1]
            # Replace dictionary lookups in generated code with array slices so
            # compiled prediction functions only close over dense numeric arrays.
            func_str_code2 = func_str_code2.replace(f'{tp}_dict[\'{k}\']', f'{tp}_arr[:, {indices}]')
    if len(float_dict):
        float_arr = np.hstack([float_dict[k].reshape((-1, 1)) if len(float_dict[k].shape) == 1 else float_dict[k]
                               for k in float_dict.keys()])
    else:
        float_arr = None
    if len(int_dict):
        int_arr = np.vstack([int_dict[k] for k in int_dict.keys()]).transpose()
    else:
        int_arr = None
    float_var_2_col = {k: i for i, k in enumerate(float_dict.keys())}
    int_var_2_col = {k: i for i, k in enumerate(int_dict.keys())}
    func_str = func_str_code2

    cnt_param_names = dict()
    param_names_new = []
    for p in param_names:
        cnt = cnt_param_names.get(p, 0)
        if cnt == 0:
            param_names_new.append(p)
            cnt_param_names[p] = 1
        else:
            param_names_new.append(f'{p}_{cnt}')
            cnt_param_names[p] += 1

    param_names = param_names_new

    func_str_pretty = func_str
    for i, nm in enumerate(param_names):
        func_str_pretty = func_str_pretty.replace(f'params[{i}]', nm)
    func_str_pretty = func_str_pretty.replace('float_dict', 'data')
    func_str_pretty = func_str_pretty.replace('int_dict', 'data')
    # print(func_str_pretty)

    # find null
    valid_indices = True
    for k, var in float_dict.items():
        if len(var.shape) > 1:
            for j in range(var.shape[1]):
                valid_indices = np.isfinite(var[:, j]) & valid_indices
        else:
            valid_indices = np.isfinite(var) & valid_indices

    prediction_function_callable = PredictionFunction.build_nonlinear_function_object(
        func_str, do_njit, param_names, float_arr, int_arr, float_var_2_col, int_var_2_col,
        custom_functions=custom_functions, dense_threshold_mb=dense_threshold_mb,
        jac_method=jac_method, debug=debug)

    return {
        'valid_indices_exog': valid_indices,
        'prediction_function_callable': prediction_function_callable,
        'func_str_code': func_str_code,
        'func_str_pretty': func_str_pretty,
        'num_params': param_cnt,
    }


def parse_str_to_var_names(func_str: str) -> list:
    """
    Example:
        (1)
            '[y] ~ {alpha}*[x] + [z]**{beta} $ [w]'
            -->
            (['y', 'x', 'z', 'w'], ['Intercept', 'alpha', 'beta'], '[y] ~ {alpha}*[x] + [z]**{beta} $ [w]')

        (2)
            '[y] ~ {alpha}*[x] + [z]**{beta} - 1'
            -->
            (['y', 'x', 'z'], ['alpha', 'beta'], '[y] ~ {alpha}*[x] + [z]**{beta} ')
    """

    func_str = func_str.replace('+', ' + ')
    while '  ' in func_str:
        func_str = func_str.replace('  ', ' ')

    #no_intercept = func_str.replace(' ', '')[-2:] == '-1'
    #if no_intercept:
    #    j = -1
    #    while func_str[j] != '-':
    #        j -= 1
    #    func_str = func_str[:j]

    exog_names = [x[1:-1] for x in re.findall(r"\[.*?\]", func_str)]
    param_names = (
            #([] if no_intercept else ['Intercept'])
            #+
            [p[1:-1] for p in re.findall(r"\{.*?\}", func_str)]
    )

    return exog_names, sorted(set(param_names)), func_str


def get_monomial_data(var_name: str, data: DataFrame, data_dict: dict, index=None, debug=False):
    """
    Extracts a column of data from a dataframe data

    Args:
        var_name: Column name or patsy-style numeric expression to evaluate.
        data: Source data frame.
        data_dict: Optional cache dict populated with extracted arrays.
        index: Optional row subset/indexer.
        debug: Whether to print diagnostics from lower-level data extraction.

    Returns:
        NumPy float64 array containing the requested regressor values.
    """

    _, index = get_nobs_from_index(data, index)

    var_name = var_name.replace(' ', '')
    if data_dict is not None and var_name in data_dict:
        return data_dict[var_name]
    elif var_name in data.columns:
        if index is None:
            xv = data[var_name].values.copy()
        else:
            xv = data[var_name].values[index].copy()
    else:
        is_spline = var_name[:3] in ('cc(', 'bs(', 'cr(')
        # if LOADED_IN_NB:
        #     xv = array(dmatrix(f'I({var_name}) -1', data, NA_action=NAAction(NA_types=[])))
        # else:
        #     xv = array(dmatrix(f'I({var_name}) -1', data, NA_action=NAAction(NA_types=[]),
        #                        eval_env=get_eval_env_depth()))
        ret = get_numerical_control_data(var_name, data, debug=debug, index=index)

        xv = ret.values.toarray()
        if len(np.shape(xv)) > 1 and not is_spline:
            xv = xv[:, -1]
            xv = xv.flatten()

    xv = xv.astype(np.float64)

    # if index is not None:
    #     xv = xv[np.arange(len(data))[index]]

    if data_dict is not None:
        data_dict[var_name] = xv

    return xv


def get_categorical_unique(cat_vals):
    """Returns actual int values `cat_vals_idx` used for indexing parameter (sub) vector
    for fixed effects, and the mapping `cat_vals_unique`

    Args:
        cat_vals: pandas Series of categorical labels.

    Returns:
        Tuple ``(cat_vals_unique, num_unique, cat_vals_idx)`` where
        ``cat_vals_unique`` are cleaned string labels, ``num_unique`` is the
        number of levels, and ``cat_vals_idx`` maps each row to its level index.
    """
    cat_vals_unique = sorted(cat_vals.unique())
    num_unique = len(cat_vals_unique)

    cat_vals_idx = cat_vals.map(dict(zip(cat_vals_unique, range(num_unique))))

    cat_vals_unique = [str(temp).replace(' ', '').replace('(', '').replace(')', '')
                       for temp in cat_vals_unique]

    return cat_vals_unique, num_unique, cat_vals_idx


def get_categorical_data(term_arg_str, data, index=None):
    """Extract categorical labels for a ``C(...)`` formula term.

    Supports one or more categorical variables; multiple variables are combined
    by converting each row to a tuple-like string.  Expressions not present as
    direct columns are evaluated through patsy.

    Args:
        term_arg_str: Inner argument string from ``C(...)``.
        data: Source pandas ``DataFrame``.
        index: Optional row subset/indexer.

    Returns:
        Tuple ``(cat_vals, variables, drop1, override_var_name)`` for
        categorical expansion in generated prediction code.
    """
    variables, drop1, override_var_name = ArgumentParser._parse_categorical_str_to_args(term_arg_str)

    df_temp = pd.DataFrame()

    # loop through all categorical variables
    for var in variables:
        if var in data.columns:
            xv = data[var]

        else:
            xv = dmatrix(f'I({var}) - 1', data, eval_env=get_eval_env_depth())
            if xv.shape[1] > 1:
                xv = xv[:, -1]
            xv = pd.Series(xv)

        if index is not None:
            xv = xv[index]

        df_temp[var] = xv

    if len(variables) > 1:
        cat_vals = df_temp.apply(lambda x: str(tuple(x)), axis=1)
    else:
        cat_vals = df_temp[variables[0]].astype(str)

    return cat_vals, variables, drop1, override_var_name


def build_prediction_function_from_formula(formula, data, debug=False, do_njit=True, index=None, _t=None,
                                           custom_functions=dict(), dense_threshold_mb=DEFAULT_DENSE_THRESHOLD_MB,
                                           jac_method=DEFAULT_NLLS_JAC_METHOD):
    """Parse an NLLS formula and build its prediction callable.

    Args:
        formula: NLLS formula string using ``[data]`` tokens and ``{parameter}``
            tokens.  The response/weight pieces may still be present; only the
            prediction-side sparse_terms are extracted here.
        data: Input data, either a pandas ``DataFrame`` or something coercible
            via ``dict_2_dataframe``.
        debug: Whether to print parsing and build diagnostics.
        do_njit: Whether to Numba-JIT compile the generated prediction function.
        index: Optional row subset/indexer.
        _t: Optional start time used for debug timing.
        custom_functions: Optional functions made available to generated code.
        dense_threshold_mb: Dense Jacobian threshold forwarded to callables.
        jac_method: Jacobian method used by the resulting prediction function.

    Returns:
        Tuple ``(prediction_function_callable, exog_result, valid_indices_exog)``.

    Examples
    --------
    Build a parameter-callable prediction function from an NLLS formula
    and evaluate it at trial parameters. This is the same parser used by
    :func:`nlls` / :func:`nlls_en`:

    >>> import numpy as np, pandas as pd
    >>> from kanly.api import build_prediction_function_from_formula
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({'x': rng.normal(size=100)})
    >>> pred_callable, exog_result, valid = \\
    ...     build_prediction_function_from_formula(
    ...         '{Intercept} + {beta} * exp({gamma} * [x])', df)   # doctest: +SKIP
    >>> yhat = pred_callable(np.array([1.0, 3.0, -0.5]))           # doctest: +SKIP
    """

    if _t is None:
        _t = time.time()

    data = dict_2_dataframe(data)

    # Get Exog
    if debug:
        print(f"\tParsing variable names and parameter names")
    exog_names, param_names, exog_func_str = parse_str_to_var_names(formula)

    if debug:
        print(f"\tBuilding prediction function callable...")
    exog_result = build_prediction_function(
        exog_names, param_names, exog_func_str, data, do_njit=do_njit, debug=debug, index=index,
        custom_functions=custom_functions, dense_threshold_mb=dense_threshold_mb, jac_method=jac_method)
    valid_indices_exog = exog_result['valid_indices_exog']
    if debug:
        print(f"\tPrediction function complete ({'%.2f' % (time.time() - _t)}s).")

    return exog_result['prediction_function_callable'], exog_result, valid_indices_exog
