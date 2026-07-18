"""
This file provides utility functions to parse categorical and numerical sparse_terms
into sparse data design matrices
"""

from __future__ import absolute_import, print_function

import math
import re

import numpy as np
from pandas import Series
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from patsy.missing import NAAction
from patsy.highlevel import dmatrix
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, isspmatrix
from scipy.sparse import hstack as sphstack
from scipy.linalg import qr as qr_scipy

from kanly.formula.seasonal_and_trend_matrices import generate_seasonal_matrix, generate_trend_matrix
from kanly.utils.util import get_eval_env_depth
from kanly.formula.sparse_formula_data_object import SparseFormulaDataObj
from kanly.formula.sparse_term import SparseTerm, NumericalControl
from kanly import _IN_NOTEBOOK
from kanly.formula.keys import RETURN_CONSTANT_COLUMN_TERM_NAME
from kanly.formula.lags import _do_lags
from kanly.nonparametric.bspline import bspline_design_matrix

from pandas.api.types import is_numeric_dtype, is_string_dtype



def get_numerical_control_data(term, data, debug=False, term_dict=None, invalid_row_func=None,
                               return_type='sparse', name=None, index=None) -> SparseFormulaDataObj:
    """Build data columns for a numerical term (raw, expression, lag, center, spline).

    Args:
        term (SparseTerm or str): Numerical term definition.
        data (DataFrame): Source data.
        debug (bool): Print diagnostic messages.
        term_dict (dict, optional): Cache for already-computed sparse_terms.
        invalid_row_func (callable, optional): Extra invalid-row predicate.
        return_type (str): ``'sparse'`` or ``'dense'`` output matrix format.
        name (str, optional): Block name stored on output object.
        index (array-like, optional): Optional subset index.

    Returns:
        SparseFormulaDataObj: Numerical design block and null-row metadata.

    Raises:
        Exception: If a referenced direct column is not numeric/bool, or when
            ``return_type`` is unsupported.
    """

    if isinstance(term, str):
        term = SparseTerm(numerical_controls=[NumericalControl(term)], var_name=term)

    # if term_dict is not None and term in term_dict.keys():
    #     return term_dict[term]

    if term.var_name == RETURN_CONSTANT_COLUMN_TERM_NAME:
        return SparseFormulaDataObj(csc_matrix(np.ones(len(data))).reshape((-1,1)),
                                    [RETURN_CONSTANT_COLUMN_TERM_NAME], set(),
                                    var_2_col_indices={term.var_name: [0]}, name=name)

    col_name = term.numerical_controls[0].term_name

    is_trend = term.numerical_controls[0].is_trend_term
    if is_trend:
        trend_values = term.numerical_controls[0].trend_exponents
        n = len(data)
        return generate_trend_matrix(trend_values, n, term_var_name=term.var_name,
                                     name=name, return_array=False, return_dense=False)

    is_seasonal = term.numerical_controls[0].is_seasonal_term
    if is_seasonal:
        seasonal_periods = term.numerical_controls[0].seasonal_periods
        n = len(data)
        return generate_seasonal_matrix(seasonal_periods, n, term_var_name=term.var_name, name=name,
                                        return_array=False, return_dense=False)

    if col_name[:2] == 'Q(':
        col_name = re.findall(r'\".*?\"', col_name)[0].replace('"', '')

    do_arith = col_name not in list(data.columns)

    # do we need to construct this column using computation
    # or is it actually a column of supplied data?
    if do_arith:

        term_name = term.numerical_controls[0].term_name.strip()
        if term_name[:2] != 'I(':
            col_name = f"I({col_name})"
        else:
            col_name = term_name

        if _IN_NOTEBOOK:
            vals = dmatrix(col_name + " -1", data, NA_action=NAAction(NA_types=[]))
        else:
            vals = dmatrix(col_name + " -1", data, NA_action=NAAction(NA_types=[]), eval_env=get_eval_env_depth())

        col_name = vals.design_info.column_names
        if vals.shape[1] > 1:
            if debug:
               print("%d columns returned -- taking the last one..." % vals.shape[1], end='')
            vals = vals[:, -1]
            col_name = col_name[-1:]

        col_name = [c.replace(' ', '') for c in col_name]

    else:
        if not (is_numeric_dtype(data.dtypes[col_name]) or is_bool_dtype(data.dtypes[col_name])):
            raise Exception("Term %s not a numeric or bool column!" % col_name)

        if isinstance(data[col_name], Series):
            vals = data[col_name].values.flatten()  # TODO?
        else:
            vals = data[col_name]

    vals, col_name = _do_lags(term, col_name, vals, data)

    n_rows, index = get_nobs_from_index(data, index)

    vals, col_name = _do_standardize(term, col_name, vals, data, term_dict=term_dict, index=index)

    is_bspline = term.numerical_controls[0].is_bspline_term
    vals_orig = vals  # keep this to check nan/inf on original array, not splines...
    if is_bspline:
        vals, col_name = _do_bspline(vals, col_name, term)

    if index is not None:
        vals = vals[index]

    nulls = set()
    vals_to_check = vals_orig if is_bspline else vals
    if len(vals_to_check.shape) > 1:
        for k in range(vals_to_check.shape[1]):
            nulls |= set(np.arange(n_rows)[~np.isfinite(vals_to_check[:,k])])
    else:
        nulls = set(np.arange(n_rows)[~np.isfinite(vals_to_check)])

    if invalid_row_func is not None:
        if len(vals_to_check.shape) > 1:
            for k in range(vals_to_check.shape[1]):
                nulls |= set(np.arange(n_rows)[invalid_row_func(vals_to_check[:, k])])
        else:
            nulls |= set(np.arange(n_rows)[invalid_row_func(vals_to_check)])

    if return_type == 'sparse':
        if len(vals.shape) == 1:
            vals = vals.reshape((-1, 1))
        values = vals
        if not isspmatrix(vals):
            values = csc_matrix(vals)
    elif return_type == 'dense':
        if isspmatrix(vals):
            values = vals.toarray()
        else:
            values = vals
    else:
        raise Exception("return type must be 'sparse' or 'dense'!")

    if isinstance(col_name, str):
        col_name = [col_name]

    length = values.shape[1] if np.ndim(values) == 2 else 1
    ret_val = SparseFormulaDataObj(
        values, col_name, nulls,
        sparse_terms=[term],
        var_2_col_indices={term.var_name: list(range(length))}, name=name,
    )

    if term_dict is not None:
        term_dict[term] = ret_val

    return ret_val


def _do_bspline(vals, col_name, term):
    numer_control = term.numerical_controls[0]
    assert numer_control.is_bspline_term

    if not isinstance(vals, np.ndarray):
        vals = np.asarray(vals)
    if np.ndim(vals) > 1:
        vals = vals[:, 0]

    if 'bspline' in term.state['numerical'][numer_control.term_name_original_string]:
        degree, bspline_df, knots, include_intercept = term.state['numerical'][numer_control.term_name_original_string]['bspline']
        vals, knots = bspline_design_matrix(
            vals, knots=knots, degree=degree, n_bases=bspline_df, include_intercept=include_intercept, return_dense=False)
    else:

        degree, bspline_df, include_intercept, knots, lower_bound, upper_bound \
            = (
            numer_control.bspline_degree, numer_control.bspline_df, numer_control.bspline_include_intercept,
            numer_control.bspline_knots, numer_control.bspline_lower_bound, numer_control.bspline_upper_bound
        )
        vals, knots = bspline_design_matrix(
            vals, degree=degree, n_bases=bspline_df, include_intercept=include_intercept, return_dense=False)

        term.state['numerical'][numer_control.term_name_original_string]['bspline'] = degree, bspline_df, knots, include_intercept

    if not isinstance(col_name, str):
        col_name = col_name[0]
    col_name = [f'bs_{col_name}_{j + 1}' for j in range(vals.shape[1])]

    return vals, col_name


def _do_standardize(term, col_name, vals, data, term_dict=None, index=None):
    """Apply ``DM(...)`` centering or standardizing transformation to one numerical column.

    Args:
        term (SparseTerm): Monomial numerical term.
        col_name (str): Current column label.
        vals (ndarray): Column values.
        data (DataFrame): Source data.
        term_dict (dict, optional): Shared cache used for nested term lookups.
        index (array-like, optional): Row subset used for weighted means.

    Returns:
        tuple: ``(demeaned_vals, updated_col_name)``.
    """
    assert term.is_monomial()
    numer_control_term = term.numerical_controls[0]

    if numer_control_term.center or numer_control_term.standardize:

        is_centering = numer_control_term.center # can only be one or the other

        weights_name = numer_control_term.center_weights if is_centering else numer_control_term.standardize_weights
        is_weighted = weights_name is not None

        # check if there is a stored state
        temp = term.state['numerical'][numer_control_term.term_name_original_string]
        if 'standardize' in temp:
            vals_mean, vals_std_dev = temp['standardize']['mean'], temp['standardize']['std_dev']

        else:
            wts = None
            if is_weighted:
                wts = get_numerical_control_data(
                    weights_name,
                    data, debug=False, term_dict=term_dict,
                    invalid_row_func=None,
                    return_type='dense', index=None).values

            if index is None:
                if wts is None:
                    vals_mean = np.nanmean(vals)
                    vals_std_dev = np.nanstd(vals)
                else:
                    vals_mean = np.average(vals, weights=wts.ravel())
                    vals_std_dev = np.sqrt(((vals - vals_mean) ** 2 * wts).sum() / wts.sum())
            else:
                if wts is None:
                    vals_mean = np.nanmean(vals[index])
                    vals_std_dev = np.nanstd(vals[index])
                else:
                    vals_mean = np.average(vals[index], weights=wts[index])
                    vals_std_dev = np.sqrt(((vals[index] - vals_mean) ** 2 * wts[index]).sum() / wts[index].sum())

        if is_centering:
            vals_std_dev = 1.0

        vals = (vals - vals_mean) / vals_std_dev
        if not isinstance(col_name, str):
            col_name = col_name[0]

        temp['standardize'] = {'mean': float(vals_mean), 'std_dev': float(vals_std_dev)}

        col_name = f'{"center" if is_centering else "standardize"}[{col_name}'\
                   f'{(";" + numer_control_term.center_weights.strip()) if is_weighted else ""}]'

    return vals, col_name


def get_nobs_from_index(data, index=None):
    """
    Converts a boolean integer index to an array of integer
    indexers

    Args:
        data (DataFrame): Source data used to validate index bounds/length.
        index (array-like[bool|int] or None): Optional row selector.

    Returns:
        tuple:
            - int: Number of observations selected.
            - ndarray or None: Integer row indices to keep.

    Raises:
        Exception: If index dtype is neither bool nor int.
    """

    if index is None:
        return data.shape[0], None

    else:
        index = np.asarray(index)
        dtype = index.dtype
        if dtype == 'bool':
            assert len(index) == len(data)
            index = np.arange(len(data))[index]
        elif dtype == 'int':
            index.sort()
            assert min(index) >= 0 and max(index) < len(data)
        else:
            raise Exception(f"`index` argument must be either dtype 'bool' or 'int', not {dtype}!")

        return len(index), index


def get_categorical_control_data(term, data, debug=False, term_dict=None, invalid_row_func=None,
                                 remove_null_columns=True, name=None, index=None) -> SparseFormulaDataObj:
    """
    Parses multiple interacted categorical controls into one big sparse matrix

    Args:
        term (SparseTerm or str): Categorical term or colon-joined interaction.
        data (DataFrame): Source data.
        debug (bool): Print diagnostics.
        term_dict (dict, optional): Cache for nested term computation.
        invalid_row_func (callable, optional): Optional invalid-row predicate.
        remove_null_columns (bool): Reindex to observed category combinations.
        name (str, optional): Block name saved in returned object.
        index (array-like, optional): Optional row subset.

    Returns:
        SparseFormulaDataObj: Sparse fixed effect/interaction matrix and metadata.
    """

    if isinstance(term, str):
        term = SparseTerm(categorical_controls=term.split(":"))

    col_data_list = []
    col_unique_data_list = []
    n_unique_list = []
    nulls = set()

    nobs, index = get_nobs_from_index(data, index)

    state = term.state['categorical']
    has_state = len(state) > 0
    if not has_state:
        state['val_2_ind'] = dict()

    for col_name in term.categorical_controls:

        if col_name not in list(data.columns):

            temp = get_numerical_control_data(
                SparseTerm.parse_to_term(col_name),
                data, term_dict=term_dict, debug=debug,
                invalid_row_func=invalid_row_func,
                return_type='dense',
                index=index,
            )

            col_data, col_name = Series(temp.values), temp.column_names[0].replace(" ", "")

            # TODO delete
            # if col_data.shape[1] > 1:
            #     raise Exception("Too many columns returned for term % s!" % term.var_name)

        else:

            col_data = data[str(col_name)]
            if not is_numeric_dtype(col_data):
                col_data = col_data.astype(str)

            if index is not None:
                col_data = col_data.iloc[index]

        col_data_list.append(col_data)

        try:
            cat_numeric = np.issubdtype(col_data.dtype, np.number)
        except:
            cat_numeric = False

        if cat_numeric:
            nulls |= set(np.arange(nobs)[~np.isfinite(col_data)])
        else:
            nulls |= set(np.arange(nobs)[Series(col_data).isnull()])

        if col_name not in state['val_2_ind']:
            col_data_unique = Series(Series(col_data).unique())
            col_data_unique = col_data_unique.sort_values().reset_index(drop=True)
            state['val_2_ind'][col_name] = col_data_unique
        else:
            col_data_unique = state['val_2_ind'][col_name]

        col_unique_data_list.append(col_data_unique)
        n_unique_list.append(len(col_data_unique))

    if index is None:
        col_indices = Series(np.zeros(nobs), index=data.index)
    else:
        col_indices = Series(np.zeros(nobs), index=data.index[index])

    mult = 1
    for i, col_name in enumerate(term.categorical_controls):
        mapping = dict(zip(col_unique_data_list[i], range(n_unique_list[i])))
        col_indices += mult * col_data_list[i].map(mapping).values
        mult *= n_unique_list[i]

    param_col_names = None

    if 'unique_indices' in state:
        unique_indices  = state['unique_indices']
    else:
        unique_indices = np.array(sorted(col_indices.unique())).astype(int)
        state['unique_indices'] = unique_indices

    unique_indices_copy = unique_indices.copy()
    for i, (n, col_name) in enumerate(zip(n_unique_list, term.categorical_controls)):
        div = n_unique_list[i]
        to_add = col_unique_data_list[i][np.mod(unique_indices_copy, div).astype(int)] \
            .apply(lambda x: f'C({col_name})[{x}]').reset_index(drop=True)
        if param_col_names is None:
            param_col_names = to_add
        else:
            param_col_names += ":" + to_add

        unique_indices_copy = (unique_indices_copy - np.mod(unique_indices_copy, div).astype(int)) / div

    if not has_state and remove_null_columns:
        col_indices = col_indices.map(dict(zip(unique_indices, np.arange(len(unique_indices)))))

    if not has_state:
        state['num_cat'] = int(col_indices.max() + 1)

    num_cat = state['num_cat']

    mat = coo_matrix((np.ones(nobs), (range(nobs), col_indices)),
                     shape=(nobs, num_cat)).tocsc()

    ret_val = SparseFormulaDataObj(mat, list(param_col_names), nulls, name=name, sparse_terms=[term])

    return ret_val

#
# if __name__ == '__main__':
#     from kanly.api import lm
#     import pandas as pd
#     from patsy import dmatrix
#     import matplotlib.pyplot as plt
#     import numpy as np
#
#     df = pd.DataFrame({'t': np.arange(200)})
#     df['y'] = np.exp(df.t / 20 - 1) + df.t ** 2 / 1000 - df.t
#     print(lm('y ~ L(bs(t,df=4),3)', df))
#     print(lm('y ~ bs(t,df=4)', df))
