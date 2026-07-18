from __future__ import absolute_import, print_function

import re
import numpy as np
import pprint

from numba import jit, vectorize  # don't delete
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
    get_frozen_logpdf_f, get_frozen_logpdf_weibull_min, get_frozen_logpdf_dirichlet, )


for z in ['logit', 'log_d_expit', 'expit', 'd_expit',
          'normal_pdf', 'normal_logpdf', 'normal_cdf',
          'log_normal_pdf', 'log_normal_logpdf', 'log_normal_cdf', 'jit', 'vectorize']:
    assert z in locals()


class FunctionCallable(object):
    """Lightweight wrapper around a compiled callable produced by
    ``get_callable_from_func_str``.

    Bundles the callable together with a human-readable description (the
    original function string and its 1-D / 2-D variants) and the ordered list
    of parameter names so that callers can inspect what the function expects.
    """

    def __init__(self, callable, description=None, param_names=None):
        """
        Args:
            callable: The underlying Python callable that accepts a parameter
                array (1-D or 2-D) and returns a scalar or array.
            description: Optional dict (or any object) describing the function,
                e.g. the original expression string, 1-D form, and 2-D form.
            param_names: Ordered list of parameter name strings corresponding
                to columns of the parameter array.
        """
        self.callable = callable
        self.description = description
        self.param_names = param_names

    def __call__(self, x, *args, **kwargs):
        """Evaluate the wrapped function on parameter array ``x``.

        Args:
            x: 1-D or 2-D array of parameter values.  A 1-D array is treated
                as a single parameter vector; a 2-D array is treated as a batch
                of rows.
            *args: Ignored (present for API compatibility).
            **kwargs: Ignored (present for API compatibility).

        Returns:
            Scalar or array depending on the function definition and the shape
            of ``x``.
        """
        return self.callable(x)

    def __repr__(self):
        """Return the string representation (delegates to ``__str__``)."""
        return str(self)

    def __str__(self):
        """Return a pretty-printed description of the function.

        Returns the ``pprint`` rendering of ``self.description`` when it is set,
        otherwise falls back to the default object string.
        """
        if self.description is not None:
            return pprint.pformat(self.description)
        else:
            return str(self)


def get_callable_from_func_str(function_str: str, param_names: list = None, debug: bool = False,
                               nopython: bool = False) -> FunctionCallable:
    """Compile a parameterised expression string into a callable ``FunctionCallable``.

    The expression string may reference named parameters enclosed in curly
    braces (e.g. ``"{mu} + {sigma} * x"``).  This function rewrites the
    template into two Python functions — one that accepts a 1-D parameter
    vector and one that accepts a 2-D array of row-vectors — then compiles
    them via ``exec`` (optionally JIT-compiled with Numba when
    ``nopython=True``).

    The resulting ``FunctionCallable`` dispatches automatically to the
    appropriate compiled function based on the dimensionality of the input.

    Args:
        function_str: Expression string with ``{param_name}`` placeholders for
            each free parameter, e.g. ``"return {beta} * x + {alpha}"``.
        param_names: Ordered list of parameter name strings defining the
            mapping from column index to name.  Inferred from the ``{...}``
            tokens in the string when ``None``.
        debug: When ``True``, print the rewritten 1-D and 2-D function strings
            before compilation.
        nopython: When ``True``, decorate the compiled functions with
            ``@jit(nopython=True)`` for Numba ahead-of-time compilation.

    Returns:
        A ``FunctionCallable`` that accepts a 1-D or 2-D NumPy array and
        evaluates the expression.
    """
    function_str_orig = function_str
    function_str = pre_format_func_str(function_str)

    function_str_1d = function_str
    function_str_2d = function_str

    text_in_brackets = re.findall('{(.+?)}', function_str)

    if param_names is None:
        param_names = text_in_brackets
    name_2_idx = {p: i for i, p in enumerate(param_names)}

    for c in text_in_brackets:
        function_str_1d = function_str_1d.replace(f'{{{c}}}', f'_____params_____[{name_2_idx[c]}]')
        function_str_2d = function_str_2d.replace(f'{{{c}}}', f'_____params_____[:, {name_2_idx[c]}]')

    # function_str_1d = function_str_1d.replace('{', '').replace('}', '')
    # function_str_2d = function_str_2d.replace('{', '').replace('}', '')

    if debug:
        print(function_str_1d)
        print(function_str_2d)

    nl_tab_split = lambda z: '\n\t' + '\n\t'.join(z.split('\n'))

    exec_dict = dict(globals())
    njit_str = f'@jit(nopython=True)\n' if nopython else ''
    exec(f'{njit_str}\ndef ______f1d(_____params_____): {nl_tab_split(function_str_1d)}', exec_dict)
    exec(f'{njit_str}\ndef ______f2d(_____params_____): {nl_tab_split(function_str_2d)}', exec_dict)

    _func_1d = exec_dict['______f1d']
    _func_2d = exec_dict['______f2d']

    _func_1d.__doc__ = function_str_orig
    _func_2d.__doc__ = function_str_orig

    def function_callable(params):
        """Dispatch to the 1-D or 2-D compiled function based on the rank of ``params``.

        Args:
            params: 1-D array of parameter values (single evaluation) or 2-D
                array of shape ``(n_samples, n_params)`` (batch evaluation).

        Returns:
            Scalar / 1-D result for 1-D input, or a 1-D array of length
            ``n_samples`` for 2-D input.

        Raises:
            Exception: If ``params`` is neither 1-D nor 2-D.
        """
        params = np.asarray(params)
        if np.ndim(params) == 2:
            return _func_2d(params)
        elif np.ndim(params) == 1:
            return _func_1d(params)
        else:
            raise Exception

    return FunctionCallable(function_callable,
                            description={'function_str': function_str,
                                         'param_names': param_names,
                                         'func1d': function_str_1d,
                                         'func2d': function_str_2d},
                            param_names=param_names)


def _check_func_for_test(func, param_names=None, debug=False):
    """Normalise ``func`` to a callable, compiling it if it is a string.

    Accepts either a ready-made callable or a function-expression string and
    returns a callable in both cases.

    Args:
        func: A callable object, or a string expression (see
            ``get_callable_from_func_str`` for format).
        param_names: Ordered parameter names, forwarded to
            ``get_callable_from_func_str`` when ``func`` is a string.
        debug: When ``True``, print intermediate compilation output.

    Returns:
        A callable that evaluates the function.

    Raises:
        Exception: If ``func`` is neither callable nor a string.
    """
    if hasattr(func, '__call__'):
        return func
    elif isinstance(func, str):
        return get_callable_from_func_str(func, param_names, debug=debug)
    else:
        raise Exception('`func` must be callable or string!')


def pre_format_func_str(func_str):
    """
    Make sure we return a value, basically treat as a `lambda`
    expressing when it's a single expression and append a return,
    else if multiline make sure last expression is a return
    statement.

    Last line of code *must* contain returnable values.
    """
    func_str = func_str.strip()
    if '\n' not in func_str and ';' not in func_str:
        if func_str[:7] != 'return ':
            func_str = 'return ' + func_str
        return func_str
    else:
        func_str_nl_split = func_str.split('\n')
        last_line_semi_split = func_str_nl_split[-1].split(';')[-1].strip()
        if last_line_semi_split[:7] != 'return ':
            raise Exception('Last expression of multline `func_str` must include return!')
        return func_str


def get_key(key, param_names, debug=False):
    """Resolve a ``key`` argument to a canonical form for use in result lookups.

    Used internally (e.g. in ``MCMCResults``) to accept flexible key
    specifications when indexing into parameter arrays or derived functions.

    Resolution rules (in order):
    1. If ``key`` is already callable, return it unchanged.
    2. If ``key`` is a string that matches a parameter name or the special
       string ``'log_posterior'``, return it as-is.
    3. If ``key`` is an integer, look up the corresponding name in
       ``param_names``.
    4. If ``key`` is any other string, compile it as a function expression via
       ``_check_func_for_test``.

    Args:
        key: A callable, parameter name string, integer index, or expression
            string.
        param_names: Ordered sequence of parameter name strings.
        debug: Forwarded to ``_check_func_for_test`` when expression
            compilation is triggered.

    Returns:
        A callable, a parameter name string, or an integer index, depending on
        what ``key`` resolves to.

    Raises:
        Exception: If ``key`` cannot be resolved by any of the above rules.
    """
    if callable(key):
        return key

    if key in list(param_names) or key == 'log_posterior':
        return key

    if isinstance(key, int):
        return param_names[key]

    if isinstance(key, str):
        return _check_func_for_test(key, param_names, debug=debug)

    else:
        raise Exception(f"{key} not valid!")

# if __name__ == '__main__':
#     func_str = """
# z = {x} + np.log(3)
# return nopython_logpdf_norm({x}, 0, 1)
# """
#     F = get_callable_from_func_str(func_str, nopython=True, param_names=['y', 'x'])
#     print(F)
#     print(F([0, 1.]))
