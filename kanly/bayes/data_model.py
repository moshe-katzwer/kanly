"""
A class for building flexible statistical models in a python-like environment.
Relies on user supplying a "data_code_block" to describe how the model should
store data variables, and a "model code block" that describes the actual
statistical model.  The user supplies a dataframe or dict of data from which the
data code block draws data.

See docstring for DataModel class.
"""
from __future__ import absolute_import, print_function, annotations

import pprint
import re
import time
from collections import OrderedDict
from collections.abc import Iterable
from typing import Callable, List

import numpy as np
from patsy.highlevel import dmatrix
from numba import jit # don't delete

from kanly import __version__
from kanly.bayes.bayesian_model import BayesianModel
from kanly.parameter_collection import ParameterCollection
from kanly.bayes.parameter import Parameter, SCALAR_PARAM_TYPE, VECTOR_PARAM_TYPE, POLYNOMIAL_PARAM_TYPE, \
    DUMMY_PARAM_TYPE, SPLINE_PARAM_TYPE
from kanly.regression.nonlinear_least_squares.formula.sparse_nonlinear_formula_parser import \
    get_monomial_data, NonlinearTermType, get_categorical_data, get_categorical_unique
from kanly.stats.distributions.fit_distributions_mle import get_normal_pdf_x_y
from kanly.utils.parse_code_string import parse_code_str
from kanly.utils.util import dict_2_dataframe
from kanly.utils.parse_string_2_tuple import parse_str_2_tuple

# from kanly.stats import IMPORT_STR
# exec(IMPORT_STR)

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
    get_frozen_logpdf_f, get_frozen_logpdf_weibull_min, get_frozen_logpdf_dirichlet,
    beta, betaln, gamma, gammaln, erf, erfc, ndtr,
    IMPORT_STR)


DEFAULT_DATA_DELIMITER = '`'
DEFAULT_PARAMETER_DELIMITER = '$'

DEFAULT_PARALLEL = False
DEFAULT_FASTMATH = False
DEFAULT_NOPYTHON = True
DEFAULT_NOGIL = True

DUMMY_PREFIX = '_dummy'
POLY_PREFIX = '_poly'

assert 'jit' in locals()


def parse_param_str_to_object(param_string) -> Parameter:
    """Parse one ``$...$`` parameter token into a :class:`Parameter` description.

    Args:
        param_string: Raw parameter token extracted from model code.
    """

    if param_string[:5] == '_par[':
        return None

    # check boundedness
    bounds = None
    param_string_new = param_string
    if param_string[-1] == '>':
        args, _ = parse_code_str(param_string, '<', '>', unique=True)
        args = args[0]
        bounds = eval(f'_get_bounds({args})')
        param_string_new = param_string[:param_string.index('<')]

    # check if is dummy, or polynomial, or vector
    dim = None
    drop1 = None
    override_name = None
    other_info = None
    if param_string_new[-1] == ']':
        args, _ = parse_code_str(param_string_new, '[', ']', unique=True)
        args = parse_str_2_tuple(args[0])

        if param_string_new[:len(POLY_PREFIX) + 1] == f'{POLY_PREFIX}[':
            param_type = POLYNOMIAL_PARAM_TYPE

            args = ','.join(args).split(';')
            if len(args) == 2:
                override_name = args[1]
            else:
                override_name = None

            args = args[0].split(',')

            assert len(args) == 2
            if ',' in args[0]:
                args[0] = args[0].replace('[', '').replace(']', '').replace('(', '').replace(')', '').split(',')
                args[0] = [a.replace(" ", "") for a in args[0]]
            else:
                args[0] = [args[0]]

            vars_nms = args[0]
            power = int(args[1])

            name = ((override_name + '_') if override_name is not None else '') + 'poly_' + '_'.join(vars_nms) + '_' + str(power)
            other_info = {'var_names': vars_nms, 'power': power, 'override_name': override_name}

        elif param_string_new[:4] in ('_cr[', '_bs[', '_cc['):

            param_type = SPLINE_PARAM_TYPE
            name = param_string_new

            override_name = None
            spline_args = param_string_new.replace(' ', '')[4:-1].split(';')
            if len(spline_args) == 2:
                override_name = spline_args[1]
            spline_args = spline_args[0]

            other_info = {'spline_args': spline_args,
                          'spline_type': param_string_new.replace(' ', '')[1:3],
                          'override_name': override_name}

        elif param_string_new[:len(DUMMY_PREFIX) + 1] == f'{DUMMY_PREFIX}[':
            param_type = DUMMY_PARAM_TYPE

            args = ','.join(args).split(';')
            if len(args) == 2:
                override_name = args[1]
            else:
                assert len(args) == 1
                override_name = None

            args = args[0].split(',')
            if len(args) == 2:
                assert args[1].replace(' ', '') == '-1'
                drop1 = True
            else:
                assert len(args) == 1
                drop1 = False
            cat_name = args[0]

            name = param_string_new
            other_info = {'override_name': override_name, 'drop1': drop1, 'cat_name': cat_name}

        else:
            param_type = VECTOR_PARAM_TYPE
            assert len(args) == 1
            dim = int(args[0])
            name = param_string_new[:param_string_new.index('[')]

    else:
        name = param_string_new
        param_type = SCALAR_PARAM_TYPE
        dim = 1

    return Parameter(param_string, param_string_new, name, bounds, param_type, dim, other_info)


def nl_tab_split(string, num_tab=1):
    """Indent each line of ``string`` by ``num_tab`` tab-equivalents (4 spaces).

    Args:
        string: Input string or iterable of strings.
        num_tab: Number of indentation levels.
    """
    if not isinstance(string, str):
        assert isinstance(string, Iterable)
        string = '\n'.join(string)
    nltb = '\n' + '    ' * num_tab
    return nltb + nltb.join(string.split('\n'))


def parse_dict_2_tuples(dict_str):
    """divide a string of a dictionary into a list of key, value pairs

    Args:
        dict_str: Dictionary-like string expression.
    """

    dict_str = dict_str.replace("\n", "").replace("\t", "")
    parenth_stack = 0
    bracket_stack = 0
    for i, c in enumerate(dict_str):
        if c == '(':
            parenth_stack += 1
        elif c == ')':
            parenth_stack -= 1
        elif c == '[':
            bracket_stack += 1
        elif c == ']':
            bracket_stack -= 1
        elif c == ',' and parenth_stack == 0 and bracket_stack == 0:
            return [tuple(dict_str[:i].split(':'))] + parse_dict_2_tuples(dict_str[i + 1:])

    return []


def _get_bounds(lb=-np.inf, ub=np.inf) -> (float, float):
    """Helper used by parser ``eval`` calls for bound syntax like ``<lb,ub>``.

    Args:
        lb: Lower bound.
        ub: Upper bound.
    """
    return lb, ub


def remove_leading_whitespace_and_semicolon(string):
    """Normalize code-block formatting and treat top-level semicolons as newlines.

    Args:
        string: Raw code block string.
    """

    while '; ' in string:
        string = string.replace('; ', ';')

    string.replace('\t', '    ')
    strings = string.split('\n')
    while np.all([s[0] == ' ' for s in strings if len(s)]):
        strings = [s[1:] if len(s) else '' for s in strings]
    string = '\n'.join(strings)

    string_new = ''
    parenth_count = 0
    for c in string:
        if c in ['(', '[']:
            parenth_count += 1
        elif c in [')', ']']:
            parenth_count -= 1
        elif c == ';' and parenth_count == 0:
            c = '\n'
        string_new += c

    return string_new


def get_param_objects(model_code_block, parameter_delimiter_start, parameter_delimiter_end, unique=True
                      ) -> List[Parameter]:
    """Extract and normalize parameter declarations from a model code block.

    Args:
        model_code_block: Raw model code string.
        parameter_delimiter_start: Parameter start delimiter.
        parameter_delimiter_end: Parameter end delimiter.
        unique: Whether parsed code fragments should be de-duplicated.
    """

    param_names, _ = parse_code_str(model_code_block, parameter_delimiter_start, parameter_delimiter_end,
                                    unique=unique)

    for p in param_names:
        model_code_block = model_code_block.replace(f'${p}$', f'${p.replace(" ", "")}$')

    param_names = [x.replace(' ', '') for x in param_names]
    param_obj_list = [parse_param_str_to_object(x) for x in param_names]
    param_obj_list = [x for x in param_obj_list if x is not None]

    unique_names = list(OrderedDict.fromkeys([r.name for r in param_obj_list]).keys())
    param_obj_list_new = []

    string_replace_dict = dict()

    for un in unique_names:

        # make bounds consistent
        un_objs = [r for r in param_obj_list if r.name == un]

        if len(un_objs) == 1:
            param_obj_list_new += un_objs
            continue

        bnds = {r.bounds for r in un_objs} - {None}
        if len(bnds) > 1:
            raise Exception(f"Inconsistent bounds for '{un}'")
        elif len(bnds) == 1:
            bnds = bnds.pop()
            for r in un_objs:
                r.bounds = bnds

        # ensure type consistency
        un_types = {r.param_type for r in un_objs}
        un_objs_sub = un_objs

        if len(un_types) > 2:
            raise Exception(f"Inconsistent types {un_types} for '{un}'")
        elif len(un_types) == 2:
            assert 'scalar' in un_types
            un_objs_sub = [r for r in un_objs if r.param_type != 'scalar']

        param_obj_list_new.append(un_objs_sub[0])
        string_replace_dict[un_objs[0].param_string_unbounded] = [r.param_string for r in un_objs]

    param_obj_list = param_obj_list_new
    return param_obj_list, string_replace_dict, model_code_block


def get_data_variables(data_code_block, data, data_delimiter_start=DEFAULT_DATA_DELIMITER, data_delimiter_end=DEFAULT_DATA_DELIMITER,
                       debug=False):
    """Resolve ``data_code_block`` expressions against data into arrays and metadata.

    Args:
        data_code_block: Data block describing derived variables.
        data: Source dataframe-like input.
        data_delimiter_start: Opening delimiter for data expressions.
        data_delimiter_end: Closing delimiter for data expressions.
        debug: Debug print flag.
    """

    data_variables, formula = parse_code_str(data_code_block, data_delimiter_start, data_delimiter_end)

    if debug:
        print(f"Data variables found are:\n")
        pprint.pprint(data_variables)

    data_dict = dict()

    data_code_block_new = data_code_block
    cat_num_val_dict = dict()
    cat_val_name_dict = dict()
    cat_var_2_term_arg_str = dict()
    for i, dvar in enumerate(data_variables):
        term_info = NonlinearTermType._get_term_type(dvar)
        term_type, term_arg_str = term_info['term_type'], term_info['term_arg_str']
        if debug:
            print(f"Variable {i} is named '{dvar}' with type '{term_type}'")
        if term_type == NonlinearTermType.MONOMIAL:
            get_monomial_data(term_arg_str, data, data_dict)
            data_code_block_new = data_code_block_new.replace(f'{data_delimiter_start}{dvar}{data_delimiter_end}',
                                                              f'var{i}')
        elif term_type == NonlinearTermType.CATEGORICAL:

            cat_vals, variables, drop1, override_var_name = get_categorical_data(term_arg_str, data, index=None)

            # Recover attribute name from assignment line (for downstream dummy-parameter parsing).
            obj_var_name = None
            for s in data_code_block.split('\n'):
                if dvar in s:
                    obj_var_name = s.split('=')[0].split('.')[1].replace(' ', '')
                    break
            if obj_var_name is None:
                raise Exception
            cat_var_2_term_arg_str[obj_var_name] = term_arg_str

            cat_vals_unique, num_unique, cat_vals_idx = get_categorical_unique(cat_vals)
            data_dict[obj_var_name] = cat_vals_idx
            data_code_block_new = data_code_block_new.replace(
                f'{data_delimiter_start}{dvar}{data_delimiter_end}',
                f'var{i};\nself.num_{obj_var_name} = {num_unique};')
            cat_num_val_dict[f'{obj_var_name}'] = num_unique
            cat_val_name_dict[f'{obj_var_name}'] = cat_vals_unique

        else:
            raise Exception('Term Type must be MONOMIAL or CATEGORICAL')

    for dvar, v in data_dict.items():
        if not isinstance(v, np.ndarray):
            data_dict[dvar] = np.asarray(v)

    cat_num_val_dict = cat_num_val_dict
    cat_val_name_dict = cat_val_name_dict

    return data_code_block_new, data_dict, cat_val_name_dict, cat_num_val_dict, cat_var_2_term_arg_str


def build_intermediate_data_object(data_code_block, namespace_dict, temp_data_dict, other_variables, custom_import_str='',
                                   debug=False,
                                   ):
    """Modifies `namespace_dict` inplace.

    Args:
        data_code_block: Normalized data block source.
        namespace_dict: Namespace dictionary to mutate with resolved values.
        temp_data_dict: Parsed temporary data-variable dictionary.
        other_variables: Extra user-supplied variables/functions.
        custom_import_str: Extra import/source text prepended to generated class.
        debug: Debug print flag.
    """

    data_class_str_1 = remove_leading_whitespace_and_semicolon(f"""\n
{custom_import_str}\n\n
{IMPORT_STR}\n

class InternalDataClass(object):
    def __init__(
         self,
         {(', '.join([f'var{i}' for i in range(len(temp_data_dict))]) + ',') if len(temp_data_dict) else ''}
         {', '.join(other_variables.keys())}
    ):
        {nl_tab_split(data_code_block, num_tab=2)}

        {nl_tab_split([f'self.{k} = {k}' for k in other_variables], num_tab=2)}
    """)

    if debug:
        print("\n\nData Model Class str to be executed...\n")
        print("=" * 200)
        print(data_class_str_1)
        print("-" * 200)

    d = dict()
    exec(data_class_str_1, None, d)

    InternalDataClass = d['InternalDataClass']
    internal_data_object = InternalDataClass(
        *list(dict(**temp_data_dict, **other_variables).values())
    )

    namespace_dict.update(internal_data_object.__dict__.copy())
    namespace_dict.update(other_variables)

    del internal_data_object


def build_model_func_internal(
        model_code_block, namespace_dict, param_2_idx, custom_import_str='',
        nogil=DEFAULT_NOGIL, nopython=DEFAULT_NOPYTHON, fastmath=DEFAULT_FASTMATH, parallel=DEFAULT_PARALLEL,
        debug=False) -> Callable:
    """Compile the user model block into an executable internal function.

    Args:
        model_code_block: Model code string after parameter substitution.
        namespace_dict: Bound symbols passed as function defaults.
        param_2_idx: Mapping from parameter names to vector indices.
        custom_import_str: Extra import/source text inserted in generated function.
        nogil: Forwarded numba ``nogil`` option.
        nopython: Whether to compile with numba nopython mode.
        fastmath: Forwarded numba ``fastmath`` option.
        parallel: Forwarded numba ``parallel`` option.
        debug: Debug print flag.
    """

    model_func_str = remove_leading_whitespace_and_semicolon(f"""
{custom_import_str}\n
{IMPORT_STR}\n

{f'@jit(nogil={nogil}, nopython={nopython}, fastmath={fastmath}, parallel={parallel})' if nopython else ''}
def _model_func_internal(
    params,  
    {', '.join([f'{k}={k}' for k in namespace_dict.keys()])}
):

    {nl_tab_split(model_code_block, num_tab=1)}

    {nl_tab_split([f'# params[{i:3d}] is "{c}"' for c, i in param_2_idx.items()], num_tab=1)}
            """)

    if debug:
        print("\n\nModel Function str to be executed...\n")
        print("=" * 200)
        print(model_func_str)
        print("-" * 200)

    d = dict()
    d.update(namespace_dict)
    exec(model_func_str, None, d)

    _model_func_internal = d['_model_func_internal']
    _model_func_internal.__doc__ = model_func_str

    return _model_func_internal, model_code_block


def update_model_code_block_for_parameter_parsing(model_code_block, param_obj_list, namespace_dict,
                                                  cat_val_name_dict, cat_var_2_term_arg_str, cat_num_val_dict,
                                                  debug=False):
    """Replace parameter placeholders in code and build bounds/group/index metadata.

    Args:
        model_code_block: Model code string to rewrite.
        param_obj_list: Parsed parameter descriptor list.
        namespace_dict: Namespace with data variables and constants.
        cat_val_name_dict: Category value-name lookup.
        cat_var_2_term_arg_str: Mapping from categorical var to parser arg string.
        cat_num_val_dict: Mapping from categorical var to number of levels.
        debug: Debug print flag.
    """

    # Loop through parameter objects and build parameter set
    bounds = dict()
    parameter_groupings = dict()
    param_2_idx = dict()

    num_params_so_far = 0
    for par_obj in param_obj_list:

        if debug:
            print("\nParsing parameter:")
            print(par_obj)

        if par_obj.param_type == SCALAR_PARAM_TYPE:

            param_2_idx[par_obj.name] = num_params_so_far
            if par_obj.bounds is not None:
                bounds[par_obj.name] = tuple(par_obj.bounds)
            model_code_block = model_code_block.replace(f'${par_obj.name}$', f'params[{num_params_so_far}]')
            model_code_block = model_code_block.replace(f'${par_obj.param_string}$', f'params[{num_params_so_far}]')
            num_params_so_far += 1

        elif par_obj.param_type == VECTOR_PARAM_TYPE:

            sub_params = [f'{par_obj.name}[{d}]' for d in range(par_obj.dim)]
            param_2_idx.update({k: num_params_so_far + i for i, k in enumerate(sub_params)})
            for c in [par_obj.param_string, par_obj.param_string_unbounded, par_obj.name]:
                model_code_block = model_code_block.replace(
                    f'${c}$',
                    f'params[{num_params_so_far}:{num_params_so_far + par_obj.dim}]')

            num_params_so_far += par_obj.dim
            if par_obj.bounds is not None:
                bounds.update({k: tuple(par_obj.bounds) for k in sub_params})
            parameter_groupings[par_obj.name] = sub_params

        elif par_obj.param_type == DUMMY_PARAM_TYPE:

            drop1 = par_obj.other_info['drop1']
            cat_name = par_obj.other_info['cat_name']
            if cat_name not in namespace_dict:
                raise Exception(f"Category name must be in data model, you supplied '{cat_name}'!")
            override_var_name = par_obj.other_info['override_name']
            if override_var_name is None:
                override_var_name = ''
            else:
                override_var_name = '_' + override_var_name

            start, end = num_params_so_far, num_params_so_far + cat_num_val_dict[cat_name] - drop1
            if drop1:
                rep_str = f"np.hstack((np.array([0.0]), params[{start}:{end}]))[{cat_name}]"
            else:
                rep_str = f'params[{start}:{end}][{cat_name}]'

            for z in [par_obj.param_string, par_obj.param_string_unbounded]:
                model_code_block = model_code_block.replace(
                    f'${z}$',
                    rep_str
                )
            model_code_block = model_code_block.replace(
                f'$_par[{par_obj.param_string_unbounded}]$',
                f'params[{start}:{end}]'
            )

            sub_params = [f'C({cat_var_2_term_arg_str[cat_name]})[{u}]{override_var_name}'
                          for u in cat_val_name_dict[cat_name][drop1:]]

            param_2_idx.update({k: num_params_so_far + i for i, k in enumerate(sub_params)})
            parameter_groupings[par_obj.param_string_unbounded] = sub_params
            if par_obj.bounds is not None:
                bounds.update({k: tuple(par_obj.bounds) for k in sub_params})
            num_params_so_far += len(sub_params)

        elif par_obj.param_type == SPLINE_PARAM_TYPE:
            res = dmatrix(f"{par_obj.other_info['spline_type']}({par_obj.other_info['spline_args']})", namespace_dict,
                          return_type='dataframe')

            override_name = par_obj.other_info['override_name']
            sub_params = [c for c in res.columns if c != 'Intercept']
            arr = res[sub_params].to_numpy()
            sub_params = [c + (f'_{override_name}' if override_name is not None else '') for c in sub_params]
            spline_data_name = s = re.sub('[^0-9a-zA-Z]+', '_', par_obj.param_string_unbounded)
            namespace_dict[spline_data_name] = arr

            model_code_block = model_code_block.replace(
                f'${par_obj.param_string_unbounded}$',
                f'np.dot({spline_data_name}, params[{num_params_so_far}:{num_params_so_far+len(sub_params)}])'
            )
            model_code_block = model_code_block.replace(
                f'$_par[{par_obj.param_string_unbounded}]$',
                f'params[{num_params_so_far}:{num_params_so_far+len(sub_params)}]'
            )

            param_2_idx.update({k: num_params_so_far + i for i, k in enumerate(sub_params)})
            parameter_groupings[par_obj.param_string_unbounded] = sub_params
            num_params_so_far += len(sub_params)
            if par_obj.bounds is not None:
                bounds.update({k: tuple(par_obj.bounds) for k in sub_params})

        elif par_obj.param_type == POLYNOMIAL_PARAM_TYPE:
            vars_ = par_obj.other_info['var_names']
            power = par_obj.other_info['power']
            name = par_obj.name
            override_name = par_obj.other_info['override_name']
            if override_name is None:
                override_name = 'poly'

            expon_vals = [tuple()]
            for _ in vars_:
                expon_vals = [
                    (i, *e) for i in range(0, power + 1) for e in expon_vals
                    if sum((i, *e)) <= power]
            expon_vals = [e for e in expon_vals if sum(e) > 0]

            sub_params = [
                f'{override_name}[{",".join([f"{v}={expo}" for v, expo in zip(vars_, etuple) if expo > 0])}]'
                for etuple in expon_vals
            ]

            var_replace_str = ' + '.join(
                [
                    nm + ' * ' + '*'.join([
                        f'{v}**{expo}' if expo > 1 else f'{v}'
                        for v, expo in zip(vars_, etuple) if expo > 0
                    ]
                    )
                    for nm, etuple in zip(sub_params, expon_vals)
                ])
            var_replace_str = '(' + var_replace_str + ')'

            model_code_block = model_code_block.replace(f'${par_obj.param_string_unbounded}$', var_replace_str)

            param_2_idx.update({
                p: num_params_so_far + i for i, p in enumerate(sub_params)
            })

            for p in sub_params:
                model_code_block = model_code_block.replace(p, f'params[{param_2_idx[p]}]')

            parameter_groupings[par_obj.param_string_unbounded] = sub_params
            if par_obj.bounds is not None:
                bounds.update({k: tuple(par_obj.bounds) for k in sub_params})
            num_params_so_far += len(sub_params)

        else:
            raise Exception(f"Param Type {par_obj.param_type} not yet supported!")

    param_names = list(param_2_idx.keys())

    return param_names, param_2_idx, parameter_groupings, bounds, model_code_block


class DataModel(ParameterCollection):
    """
    A class for building flexible statistical models in a python-like environment.
    Relies on user supplying a "data_code_block" to describe how the model should
    store data variables, and a "model code block" that describes the actual
    statistical model.  The user supplies a dataframe or dict of data from which the
    data code block draws data.

    The user may also supply a dictionary "other_variables" which can contain data
    variables and functions to be used in the model.

    # --------------- #
    # data_code_block #
    # --------------- #

    The syntax for `data_code_block` is series of python expressions of the form

        self.x = `{patsy-like-expression}`

    Crucially, any patsy formulas must be inside `` characters.  For example

        self.x2 = `I(x**2)`
        self.x4 = self.x2 ** 2

        self.c_g = `C(g)`

    would build a data model with variables "x2", "x4", and "c_g" corresponding
    to the patsy expressions inside the ``.  Note that categorical controls like `C(g)`
    actually store in `c_g` a vector of ints in the range [0, N] where N is the number
    of unique "g" values. This is used later as described when we get to the
    "model_code_block"

    In the categorical controls, one can specify multiple categories to interact,
    for example `C(g1,g2>0)` is the same as `C(g1):C(g2>0)` in patsy.  But in
    this syntax it has to be a list.  The only acceptable patsy expressions here are
    either categorical `C(...)`, column names from "data", or arithmetic expressions
    like `I(x**2)`.

    You can use whatever numpy functions you want in the "data_code_block" and the
    "model_code_block". You can also define statistical functions here for use in
    the model code block.  For example,

        self.lp = get_frozen_logpdf_lognorm(0, 4)

    would store a callable in "lp" representing the log-pdf for a lognormal random
    variable with location 0 and scale 4.  Using "frozen" functions can save computation
    time since constants only need to be computed once.

    # ---------------- #
    # model_code_block #
    # ---------------- #

    In the model code block, parameters are indicated by $<code>$.

    The basic syntax for a parameter is

        Syntax               |   Description
        ---------------------|-------------------------------------------------------------
        $name$               |   A scalar parameter with name "name"
        $name<lb,ub>$        |   A scalar parameter with name "name", bounded in (lb, ub)
        $name[p]$            |   A vector parameter with name "name", length "p"
        $name[p]<lb,ub>$     |   A vector parameter with name "name", length "p",
                             |   bounded in (lb, ub)
        ---------------------|-------------------------------------------------------------

    So for example, a simple linear regression might look like

        return logpdf_norm(y, $a$ - $b$ * x, $sigma$)

    where "x" and "y" are presumably defined in the model code block or supplied via "other_variables"
    argument.

    Note that when *accessing* indices of a vector parameter, the code in your model code block
    should read `$beta$[2]` if we had a vector parameter $beta[p]$.

    See the functions in "kanly.bayes.utils.nopython_frozen_logpdf" and "kanly.bayes.utils.nopython_logpdf"
    for available statistical functions.  You can also supply your own via "other variables"!  In general though,
    for many scipy.stats distributions, you can use logpdf_{dist} for distribution {dist}, e.g. logpdf_gamma.
    The arguments are consistent with scipy.  See: https://docs.scipy.org/doc/scipy/reference/stats.html

    The model code block expects that the last line of the code block will be a return statement.
    By default, the DataModel object is a callable, and it returns the first item in the return-tuple
    specified in the code block, unless called with "return_first=False" argument.  So for example you
    might have

        return (log_posterior, log_likelihood, log_prior)

    and by default calling DataModel returns log_posterior, but can it also return all three.

    There are some convenience hard-coded "functions" one can use when defining parameters, as follows:

        $_dummy[x{,-1}{;name}]$   |   Dummy fixed-effects for x.  Optional "-1" arg to drop one for
                                  |   multicollinearity.  Optional "name" arg to override parameter naming
                                  |   in case that we use multiple dummy variables for "x" in the model and want
                                  |   to distinguish.

        $_bs[args{;name}]$        |   Splines from patsy.  Again, "name" is an optional override for disambiguation.
        $_cc[args{;name}]$        |   "args" are consistent with patsy syntax.  bs, cc and cr correspond to B-splines,
        $_cr[args{;name}]$        |    cyclic cubic splines, and natural cubic splines, respectively.
                                  |    See: https://patsy.readthedocs.io/en/latest/spline-regression.html

        $_poly[(x{,y,z,...}),power;{,name}]$    |   Polynomial of degree "power" in the variables x, y, z...
                                                |   Can be single or multivariable.  "name" is again the optional
                                                |   disambiguation name.
                                                |   $_poly[(x,y),2]$ would yield an expression
                                                |   "a0*x + a1*y + a2*x**2 + a3*y**2 + a4*x*y"

    The last convenience function is "_par[...]".  This essentially calls the parameters in isolation,
    independent of the actual model implied.  So for example $_poly[x,2]$ will add something to the code of the
    form "a_1*x + a_2*x**2" in places where $_poly[x,2]$ appeared. Writing $_par[_poly[x,2]]$ will
    instead reference the array [a_1, a_2].

    You can use bounds as above, e.g. $_dummy[x]<0,1>$ will bound all those fixed effects in (0,1).

    The parser is smart -- but not that smart.  You can define bounds or a vector parameter by
    putting (eg) $beta[5]<0,np.inf>$ at the top of your "model_code_block" and then reference
    $beta$ elsewhere, you don't need to repeat the bounds.

    # ------------- #
    # BayesianModel #
    # ------------- #

    Once a DataModel "dmo" object is constructed, the user can call "dmo.to_bayesian_model()" to
    convert "dmo" to a BayesianModel object from which you can sample via MCMC.  Use the "sample"
    method to draw from the distribution.

    This of course assumes that the "DataModel" callable is the log-density of the posterior.

    Examples
    --------
    Weighted regression with fixed effects and a polynomial, written as a
    posterior log-density block. Parameters appear in ``$...$`` braces;
    data columns appear in ```...``` backticks:

    >>> import numpy as np
    >>> from kanly.api import DataModel
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> x = rng.normal(size=n)
    >>> data = {'x': x,
    ...         'y': 3 + 10 * x + 2 * rng.standard_t(df=3, size=n),
    ...         'z': rng.standard_t(df=3, size=n),
    ...         'g': rng.integers(0, 4, n),
    ...         'wts': 0.01 + rng.uniform(size=n)}
    >>> data_string = '''
    ... self.x = `x`
    ... self.z = `z`
    ... self.y = `y`
    ... self.weights = `wts`
    ... self.g = `C(g)`                                # categorical
    ... self.root_weights = np.sqrt(self.weights)
    ... '''
    >>> model_string = '''
    ... pred = $Intercept$ + $x$ * x + $_dummy[g,-1]$ + $_poly[z,2]$
    ... resid = y - pred
    ... return logpdf_norm(resid, loc=0.0,
    ...                    scale=$sigma$ / root_weights).sum()
    ... '''
    >>> dmo = DataModel.build_data_model(data_string,        # doctest: +SKIP
    ...                                  model_string, data,
    ...                                  nopython=False)
    >>> model = dmo.to_bayesian_model(                       # doctest: +SKIP
    ...     priors={'x': 'norm(0, 10)'},
    ...     bounds={'sigma': [0, np.inf]})
    >>> fit = model.sample(np.ones(model.num_params),        # doctest: +SKIP
    ...                    n_samples=5_000, n_burnin=2_000,
    ...                    n_chains=4)
    >>> print(fit)                                            # doctest: +SKIP

    See ``examples/bayes/example_data_model.py`` and
    ``example_data_model_fit_beta.py`` for runnable scripts.
    """

    initialized = False

    def __init__(self, param_names, _model_func_internal, namespace_dict,
                 settings, data=None, time_elapsed=np.nan,
                 parameter_groupings=None, bounds=None, param_obj_list=None):
        """Initialize ``DataModel`` from pre-built internals.

        Args:
            param_names: Ordered parameter names.
            _model_func_internal: Tuple ``(callable, model_code_block)``.
            namespace_dict: Bound data/constants namespace.
            settings: Build settings dictionary.
            data: Optional original data object.
            time_elapsed: Model build elapsed seconds.
            parameter_groupings: Optional grouped parameter mapping.
            bounds: Optional parameter bounds dictionary.
            param_obj_list: Optional parsed parameter descriptor list.
        """

        super().__init__(param_names, parameter_groupings)

        self.settings = settings.copy()

        # the actual evaluation function for the model
        self._model_func_internal, self.model_code_block = _model_func_internal

        self.namespace_dict = namespace_dict.copy()

        self.data = data
        self.time_elapsed = time_elapsed

        for k, v in namespace_dict.items():
            setattr(self, k, v)

        if bounds is None:
            bounds = dict()
        self.bounds = bounds

        self.param_obj_list = param_obj_list

        self.initialized = True

    def __setattr__(self, name, value):
        """Restrict attribute mutation post-initialization to known namespace symbols.

        Args:
            name: Attribute name.
            value: New attribute value.
        """
        if not self.initialized:
            object.__setattr__(self, name, value)
        else:
            if name in self.namespace_dict:
                if type(value) == type(self.namespace_dict[name]) and np.shape(value) == np.shape(
                        self.namespace_dict[name]):
                    object.__setattr__(self, name, value)
                    self.namespace_dict[name] = value
                else:
                    raise Exception(f"Type or shape for variable {name} did not match!")
            else:
                raise Exception("Cannot set new attributes of `DataModel` after initialization")

    @staticmethod
    def build_data_model(
            data_code_block: str, model_code_block: str, data=None,
            nogil: bool = DEFAULT_NOGIL, nopython: bool = DEFAULT_NOPYTHON, fastmath: bool = DEFAULT_FASTMATH,
            parallel: bool = DEFAULT_PARALLEL, debug: bool = False,
            custom_import_str: str = None,
            data_delimiter_start: str = DEFAULT_DATA_DELIMITER,
            data_delimiter_end: str = DEFAULT_DATA_DELIMITER,
            parameter_delimiter_start: str = DEFAULT_PARAMETER_DELIMITER,
            parameter_delimiter_end: str = DEFAULT_PARAMETER_DELIMITER,
            other_variables: dict = None,
    ) -> DataModel:
        """Build a :class:`DataModel` from data/model code blocks and input data.

        Args:
            data_code_block: User data-transformation code.
            model_code_block: User model code returning log target components.
            data: Input dataframe/dict-like data source.
            nogil: Numba ``nogil`` option.
            nopython: Numba ``nopython`` option.
            fastmath: Numba ``fastmath`` option.
            parallel: Numba ``parallel`` option.
            debug: Debug mode flag.
            custom_import_str: Extra import/source code to inject.
            data_delimiter_start: Start delimiter for data expressions.
            data_delimiter_end: End delimiter for data expressions.
            parameter_delimiter_start: Start delimiter for parameter tokens.
            parameter_delimiter_end: End delimiter for parameter tokens.
            other_variables: Additional symbols available in generated code.

        Returns:
            DataModel: A callable data-aware model object. Use
            :meth:`to_bayesian_model` to convert it into a
            :class:`~kanly.bayes.bayesian_model.BayesianModel` for MCMC.

        Examples
        --------
        Fit a Beta distribution to univariate data using Numba-compiled
        likelihood code:

        >>> import numpy as np
        >>> from scipy.stats import beta
        >>> from kanly.api import DataModel
        >>> rng = np.random.default_rng(0)
        >>> data = {'x': beta.rvs(a=5, b=2, size=1_500, random_state=0)}
        >>> data_string = '''
        ... self.x = `x`
        ... '''
        >>> model_string = '''
        ... return nopython_logpdf_beta(x, a=$a$, b=$b$).sum()
        ... '''
        >>> dmo = DataModel.build_data_model(                   # doctest: +SKIP
        ...     data_string, model_string, data, nopython=True)
        >>> model = dmo.to_bayesian_model(                       # doctest: +SKIP
        ...     bounds={'a': [0, np.inf], 'b': [0, np.inf]})
        >>> fit = model.amha([1.0, 1.0], n_samples=5_000,        # doctest: +SKIP
        ...                  n_burnin=2_000, n_chains=4)
        >>> print(fit)                                            # doctest: +SKIP

        See ``examples/bayes/example_data_model.py`` and
        ``example_data_model_fit_beta.py`` for full runnable scripts and
        sample summary output.
        """

        time_start = time.time()

        namespace_dict = dict()

        if debug:
            print("Building kanly DataModel...\n\n")

        if data_code_block is None or data_code_block == '':
            data_code_block = 'pass'

        data_code_block = remove_leading_whitespace_and_semicolon(data_code_block)

        if data is None:
            data = dict()

        settings = dict(data_delimiter_start=data_delimiter_start, data_delimiter_end=data_delimiter_end,
                        parameter_delimiter_start=parameter_delimiter_start,
                        parameter_delimiter_end=parameter_delimiter_end,
                        parallel=parallel, nogil=nogil, fastmath=fastmath, nopython=nopython,
                        custom_import_str=custom_import_str,
                        data_code_block=data_code_block, model_code_block=model_code_block, debug=debug,
                        version=__version__,
                        )

        if other_variables is None:
            other_variables = dict()

        data = dict_2_dataframe(data)

        if custom_import_str is None:
            custom_import_str = ''

        # -----------------
        # ORGANIZE THE DATA
        # -----------------

        data_code_block_new, temp_data_dict, cat_val_name_dict, cat_num_val_dict, cat_var_2_term_arg_str \
            = get_data_variables(data_code_block, data, data_delimiter_start=data_delimiter_start,
                                 data_delimiter_end=data_delimiter_end,
                                 debug=debug)

        build_intermediate_data_object(data_code_block_new, namespace_dict, temp_data_dict, other_variables,
                                       custom_import_str=custom_import_str)

        del temp_data_dict

        # ------------------------
        # BUILD THE MODEL FUNCTION
        # ------------------------

        # assert model_code_block.count('return ') == 1 # todo keep this check?
        model_code_block = remove_leading_whitespace_and_semicolon(model_code_block)

        # get parameter objects from code
        param_obj_list, string_replace_dict, model_code_block = get_param_objects(
            model_code_block, '$', '$')

        # ensure name consistency in code block
        for dvar, v in string_replace_dict.items():
            for vsub in v:
                model_code_block = model_code_block.replace(f'${vsub}$', f'${dvar}$')

        param_names, param_2_idx, parameter_groupings, bounds, model_code_block = \
            update_model_code_block_for_parameter_parsing(
                model_code_block, param_obj_list, namespace_dict,
                cat_val_name_dict, cat_var_2_term_arg_str, cat_num_val_dict,
                debug=debug)

        _model_func_internal = build_model_func_internal(
            model_code_block, namespace_dict, param_2_idx, custom_import_str=custom_import_str,
            nogil=nogil, nopython=nopython, fastmath=fastmath, parallel=parallel, debug=debug)

        time_elapsed = time.time() - time_start
        if debug:
            print(f"Done! ({time_elapsed:.3f}s)")

        return DataModel(param_names, _model_func_internal, namespace_dict, settings, data=data,
                         time_elapsed=time_elapsed, parameter_groupings=parameter_groupings,
                         bounds=bounds, param_obj_list=param_obj_list)

    def model_func_internal(self, params):
        """Evaluate compiled model function with bound namespace values.

        Args:
            params: Parameter vector.
        """
        params = np.asarray(params)
        return self._model_func_internal(
            params,
            *self.namespace_dict.values()
        )

    def __call__(self, params, return_first=True):
        """See `self._model_func_internal.__doc__` for info
        `return_first=True` returns first value if return value if a tuple

        Args:
            params: Parameter vector or parameter dictionary.
            return_first: Whether to return first element when model returns tuple.
        """
        if isinstance(params, dict):
            params = self.dict_2_array(params)
        val = self.model_func_internal(params)
        if return_first:
            if isinstance(val, tuple):
                return val[0]
        return val

    def __str__(self):
        """Return verbose printable representation of model internals.

        Args:
            None.
        """
        newline = '\n'
        newtab = '\n\t'
        return (
            f'DataModel:'
            f'\n\nNamespace variables:\n{newtab}{newtab.join(pprint.pformat({k: (type(v), np.shape(v)) for k, v in self.namespace_dict.items()}).split(newline))}'
            f'\n\nParameters:\n{newtab}{newtab.join(pprint.pformat(self.param_2_idx).split(newline))}'
            f'\n\nData Formula:\n{newtab}{newtab.join(self.settings["data_code_block"].split(newline))}'
            f'\n\nModel Formula:\n{newtab}{newtab.join(self.settings["model_code_block"].split(newline))}'
            f'\n\nModel Function Code String:\n{newtab}{newtab.join(self._model_func_internal.__doc__.split(newline))}'
        )

    def __repr__(self):
        """Return representation string.

        Args:
            None.
        """
        return str(self)

    def to_bayesian_model(self, bounds: dict = None, do_bounded_transform: bool = True, priors: dict = None,
                          specification_name: str = None, other_info: dict = None,
                          debug: bool = False) -> BayesianModel:
        """Convert this data model into a :class:`BayesianModel`.

        Args:
            bounds: Optional bounds override/extension.
            do_bounded_transform: Whether to transform bounded parameters.
            priors: Optional prior dictionary.
            specification_name: Optional model label.
            other_info: Optional metadata dictionary.
            debug: Debug mode flag.
        """
        if other_info is None:
            other_info = dict()
        other_info = other_info.copy()
        other_info['data_model'] = self

        if bounds is None:
            bounds = self.bounds
        else:
            temp = bounds
            bounds = dict(self.bounds)
            bounds.update(temp)

        return BayesianModel(
            self, bounds=bounds, do_bounded_transform=do_bounded_transform, priors=priors,
            num_params=self.num_params, specification_name=specification_name, other_info=other_info, debug=debug,
            param_names=self.param_names, parameter_groupings=self.parameter_groupings)

#     def sample(self, x0, **kwargs) -> MCMCResults:
#         return self.to_bayesian_model().sample(x0, **kwargs)
#
#     def amha(self, x0, **kwargs) -> MCMCResults:
#         return self.to_bayesian_model().amha(x0, **kwargs)
#
#     def mala(self, x0, **kwargs) -> MCMCResults:
#         return self.to_bayesian_model().mala(x0, **kwargs)
#
# if __name__ == '__main__':
#
#
#     from kanly.api import lm
#     np.random.seed(0)
#     n = 1500
#     x = 1.56 * np.random.randn(n)
#     z = np.random.rand(n)
#     y = 3 + 10 * x + np.random.randn(n) * 3
#     wts = .01 + np.random.rand(n)
#     data = {'x': x, 'y': y, 'z': z}
#
#     data_code_block = '''
#     self.x = `x`
#     self.z = `z`
#     self.y = `y`
#     '''
#
#     model_code_block = '''
#     return logpdf_norm(y - $Intercept$ - $x$ * x - $z$ * z, 0, $sigma$).sum() + logpdf_norm($z$, -5, .1)
#     '''
#
#     model = DataModel.build_data_model(
#         data_code_block, model_code_block, data)
#
#     print(model)
#
#     fit = model.to_bayesian_model(bounds={'x': [9.7, np.inf]}).amha({'x': 10, 'sigma': 1}, n_samples=20_000)
#     print(fit)
#
#     print(lm('y ~ x + z', data))

# if __name__ == '__main__':
#     from kanly.api import lm
#     np.random.seed(0)
#     n = 1500
#     x = 1.56 * np.random.randn(n)
#     z = np.random.rand(n)
#     y = 3 + 10 * x + np.random.randn(n) * 3
#     g = np.random.randint(0, 10, n)
#     wts = .01 + np.random.rand(n)
#     data = {'x': x, 'y': y, 'z': z, 'g': g}
#
#     data_code_block = \
#     '''
# self.x = `x`
# self.z = `z`
# self.y = `y`
# self.g = `C(g)`
# self.X = np.vstack([self.x, self.z]).transpose()
#     '''
#
#     model_code_block = '''
# pred = $alpha$ + np.dot(X, $beta[2]$) + $dummy[g]$
# return logpdf_norm(y - pred, 0, $sigma$).sum()
#     '''
#
#     from scipy.stats import multivariate_normal
#
#     mnf = multivariate_normal([1, 1], [[1, .6], [.6, 1]]).logpdf
#
#     model = DataModel.build_data_model(
#         data_code_block,
#         model_code_block,
#         data
#     ).to_bayesian_model(
#         priors={'beta': 'multivariate_normal([10, 10], [[.01, -0.008], [-.008, .01]])',
#                 'C(g)': 'multivariate_normal(np.zeros(10), np.eye(10)*.05)',
#                 },
#     )
#
#     print(model)
#     print(model.parameter_groupings)
#     print(model.priors)
#     print(model.amha([0, 1, 0, 0] + [0] * 10))

# #
# if __name__ == '__main__':
#
#     from numba import njit
#
#     @njit
#     def func(x):
#         return x + 3
#
#     other_variables = {
#         'func': func
#     }
#
#     model_code = '''
#     $alpha<lb=-5,ub=5>$
#     return func(logpdf_norm($beta[5]$, $alpha$, .025+abs($alpha$)/4).sum())
#     '''
#
#     model = DataModel.build_data_model(
#         None,
#         model_code,
#         None,
#         other_variables=other_variables,
#         debug=True,
#     ).to_bayesian_model(
#         priors={'beta': 'multivariate_normal(5+np.zeros(5), np.eye(5)*.1)'},
#         #bounds={'alpha': [-5, 5]}
#     )
#     print(model)
#     fit = model.amha(np.zeros(6),
#                      n_samples=200_000,
#                      n_burnin=10_000,
#                      thinning=5,
#                      max_subchain_draws_sample=40_000,
#                      max_subchain_draws_burnin=2_000,
#                      do_diff_evolution_mc=True,
#                      diff_evolution_weight=.95)
#     print(fit)
#
#     fit.diagnostic_plot('beta__0', show=True)
#     fit.hist('beta__1', show=True)

#
# if __name__ == '__main__':
#
#     np.random.seed(0)
#     data = {'y': 10 + np.random.rand(1300), 'g': np.random.randint(0, 10, 1300)}
#
#     data_code_block = '''
# self.g = `C(g)`
# self.y = `np.log(y)`
# '''
#
#     model_code = '''
# # parameters
# $_dummy[g]<lb=-1,ub=1>$
#
# return logpdf_norm(y - $a$ - $_dummy[g]$, 0, $sigma<lb=0>$).sum()
# '''
#
#     model = DataModel.build_data_model(
#         data_code_block=data_code_block,
#         model_code_block=model_code,
#         data=data,
#         debug=False)
#     print(model)
#     print(model.bounds)
#
#     model2 = model.to_bayesian_model()
#
#     fit = model2.amha([.5]*12, n_samples=250_000, thinning=2)
#     print(fit)
#


# data = {'g': np.random.randint(0,10,100)}
#
# model_code = """
# $beta$
# $beta<0,1>$
# $beta[10]<0,1>$
# $x$
# $x[5]$
# $_dummy[g]$
# $_dummy[g]<-4>$
# $gamma$
# """
#
# DataModel.build_data_model('self.g = `C(g)`', model_code, data)

# if __name__ == '__main__':
#
#     np.random.seed(0)
#     data = {'y': 10 + np.random.rand(1300), 'g': np.random.randint(0, 10, 1300), 'x': np.random.rand(1300)}
#
#     data_code_block = '''
# self.g = `C(g)`
# self.y = `y`
# self.x = `x`
# '''
#
#     model_code = '''
# # parameters
# $_dummy[g]<lb=-1,ub=1>$
# intercept = $a$
#
# return logpdf_norm(y - intercept - $_dummy[g]$ + x*$_dummy[g,False,x]$, 0, $sigma<lb=0>$).sum()
# '''
#
#     model = DataModel.build_data_model(
#         data_code_block=data_code_block,
#         model_code_block=model_code,
#         data=data,
#         debug=False)
#     print(model)
#     print(model.bounds)
#     print('>>: ', model.parameter_groupings)
#
#     model2 = model.to_bayesian_model(priors={'_dummy[g]': 'norm(0,.01)', '_dummy[g,False,x]': 'norm(0,.1)'})
#
#     fit = model2.amha([.1]*22, n_samples=250_000, thinning=5)
#     print(fit)

# if __name__ == '__main__':
#
#     np.random.seed(0)
#     data = {'y': 10 + np.random.rand(1300), 'g': np.random.randint(0, 10, 1300), 'x': np.random.rand(1300)}
#
#     data_code_block = '''
# self.g = `C(g)`
# self.y = `y`
# self.x = `x`
# '''
#
#     model_code = '''
# intercept = $a$
# poly_x = $_poly[x,2]$
# return logpdf_norm(y - intercept - poly_x, 0, $sigma<lb=0>$).sum()
# '''
#
#     model = DataModel.build_data_model(
#         data_code_block=data_code_block,
#         model_code_block=model_code,
#         data=data,
#         debug=False)
#     print(model)
#     print(model.bounds)
#
#     model2 = model.to_bayesian_model()
#
#     fit = model2.amha([.1] * 4, n_samples=10_000, thinning=5)
#     print(fit)
#
# if __name__ == '__main__':
#     from kanly.api import build_data_model
#
#     import numpy as np
#
#     np.random.seed(0)
#     n = 1000
#
#     z = 4 * np.random.randn(n)
#     g1 = np.random.randint(0, 2, n)
#     g2 = np.random.randint(0, 8, n)
#
#     y = .3 * np.random.randn(n) + 4 * (z > 4)
#
#     data = {'y': y, 'g1': g1, 'g2': g2}
#
#     data_code = f'''
#     self.y = `y`
#     self.z = `z`
#     self.g1_g2 = `C(g1,z>4)`
#     self.z_gt_2 = `z>2`
#     '''
#
#     model_code = '''
#     return logpdf_norm(y - $_dummy[g1_g2]$ + $a$ * z_gt_2, 0, $sigma$).sum()
#     '''
#
#     model = build_data_model(data_code, model_code, data).to_bayesian_model()
#     fit = model.amha(np.ones(6))
#     print(fit)
# if __name__ == '__main__':
#     from kanly.api import build_data_model
#
#     import numpy as np
#
#     np.random.seed(0)
#     n = 1000
#
#     z = 4 * np.random.randn(n)
#     g1 = np.random.randint(0, 2, n)
#     g2 = np.random.randint(0, 8, n)
#
#     y = .3 * np.random.randn(n) + 4 * (z > 4)
#
#     data = {'y': y, 'g1': g1, 'g2': g2}
#
#     data_code = f'''
#     self.y = `y`
#     self.z = `z`
#     self.g1_g2 = `C(g1,z>4)`
#     self.z_gt_2 = `z>2`
#     '''
#
#     model_code = '''
#     return logpdf_norm(y - $a$ - $_bs[z, df=3]$, 0, 1).sum()
#     '''
#
#     model = build_data_model(data_code, model_code, data).to_bayesian_model()
#     fit = model.amha(np.ones(4))
#     print(fit)

# if __name__ == '__main__':
#     from kanly.api import build_data_model, lm, nlls
#
#     import numpy as np
#
#     np.random.seed(0)
#     n = 3000
#
#     z = 4 * np.random.randn(n)
#     x = 4 * np.random.randn(n)
#     g1 = np.random.randint(0, 2, n)
#     g2 = np.random.randint(0, 8, n)
#
#     y = .3 * np.random.randn(n) + 4 * (z > 4)
#
#     data = {'y': y, 'g1': g1, 'g2': g2, 'z': z, 'x': x}
#
#     data_code = f'''
#     self.y = `y`
#     self.z = `z`
#     self.g1_g2 = `C(g1,z>4)`
#     self.z_gt_2 = `z>2`
#     '''
#
#     model_code = '''
#     target = 0.0
#     target += logpdf_norm($_par[_bs[z, df=3; bs1]]$, 0, .01).sum()
#     target += logpdf_norm(y - $a$ - $_bs[z, df=3; bs1]$ - $_bs[z, df=3; bs2]$ - $_dummy[g1_g2,-1;hello]$, 0, $sigma$).sum()
#     return target
#     '''
#
#     model = build_data_model(data_code, model_code, data).to_bayesian_model()
#     fit = model.amha(np.ones(11), n_burnin=2000, n_samples=32000)
#     print(fit)


# if __name__ == '__main__':
#     from kanly.api import build_data_model, lm, nlls
#
#     import numpy as np
#
#     np.random.seed(0)
#     n = 3000
#
#     z = 4 * np.random.randn(n)
#     x = 4 * np.random.randn(n)
#     g1 = np.random.randint(0, 2, n)
#     g2 = np.random.randint(0, 8, n)
#
#     y = .3 * np.random.randn(n) + 4 * (z > 4)
#
#     data = {'y': y, 'g1': g1, 'g2': g2, 'z': z, 'x': x}
#
#     data_code = f'''
#     self.y = `y`
#     self.z = `z`
#     self.g1_g2 = `C(g1,z>4)`
#     self.z_gt_2 = `z>2`
#     '''
#
#     model_code = '''
#     target = 0.0
#     #target += logpdf_norm(y - $a$ - $_poly[z,2;poly_z]$, 0, $sigma$).sum()
#     target += logpdf_norm(y - $a$ - $_dummy[g1_g2,-1; g1_g2]<0,10>$, 0, $sigma$).sum()
#     return target
#     '''
#
#     model = build_data_model(data_code, model_code, data, debug=True).to_bayesian_model()
#     print(model)
#     fit = model.sample(np.ones(5), n_burnin=2000, n_samples=8000, thinning=2)
#     print(fit)
#
#     fit.multi_scatter(['a', 'a', 'sigma'], show=True)
#
if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from kanly.api import lm, blm, build_data_model
    from numpy.linalg import pinv
    from scipy.stats import invgamma

    n = 100
    k = 3
    np.random.seed(0)

    X = np.random.randn(n, k)
    beta = np.random.rand(k)
    y = X.dot(beta) + .15 * np.random.randn(n)

    df = pd.DataFrame(X, columns=[f'x{j}' for j in range(k)])
    df['y'] = y

    sigma = .0001

    data_code = f'''
    self.y = `y`
    self.x0 = `x0`
    self.x1 = `x1`
    self.x2 = `x2`
    '''

    model_code = f'''
    $sigma2<0,np.inf>$
    sigma = $sigma2$ ** .5

    $beta[3]$

    pred = x0 * $beta$[0] + x1 * $beta$[1] + x2 * $beta$[2]
    llf = logpdf_norm(y, pred, sigma).sum()
    prior = logpdf_norm($beta$[2], .3, {sigma})
    prior += logpdf_norm($beta$[2], 2.3, .01)

    return llf + prior
    '''

    model = build_data_model(data_code, model_code, df).to_bayesian_model()
    print(model)

    f = model.sample({'sigma2': 1}, debug=True, do_parallel=False)
    print(f)

    import matplotlib.pyplot as plt
    plt.hist(f.get_sample('beta[2]'), density=True, bins=30, alpha=.6)
    plt.plot(*get_normal_pdf_x_y(.3, sigma, num_sigma=5), lw=2)
    plt.show()
