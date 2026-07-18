"""Build NumPy callables from formula strings and attach symbolic AD via :class:`AutoDiffGraphNode`.

``func_str_to_callable`` rewrites ``{name}`` placeholders to ``params[i]``, ``exec`` is a small
function body (optionally ``@jit``), and returns a :class:`FunctionCallable` that can emit
analytical partials, Jacobians, gradients, and Hessians for supported expressions.
"""
from __future__ import absolute_import, print_function

import pprint
import re

from kanly.automatic_differentiation.graph import AutoDiffGraphNode
from kanly.dill_object import DillObject

from kanly.utils.parse_code_string import parse_code_str

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
    gammaln, betaln, erf, ndtr, gamma, beta, erfc
)

DEFAULT_PARAM_DELIMITERS = '{', '}'


class FunctionCallable(DillObject):
    """Callable function built from a string, with optional analytical derivatives.

    Attributes mirror constructor arguments plus a lazily built ``auto_diff_node`` when
    derivative methods are used.
    """

    def __init__(self, callable_function, nobs, num_params, param_names=None, func_str=None, python_code_str=None,
                 other_args=''):
        """Wrap ``callable_function`` and record metadata for AD and composition.

        Parameters
        ----------
        callable_function : callable
            ``(params, *other) ->`` scalar or array function (after ``exec`` construction).
        nobs : int
            Number of observations (length of vector return when not scalar).
        num_params : int
            Length of ``params`` vector.
        param_names : list of str, optional
            Human-readable names; default ``_param_0``, ``_param_1``, …
        func_str : str, optional
            Expression after ``{name}`` → ``params[i]`` rewrite; used by :class:`AutoDiffGraphNode`.
        python_code_str : str, optional
            Full generated source including optional ``@jit`` (for debugging / persistence).
        other_args : str, optional
            Comma-separated names of extra arguments after ``params`` in the generated def.
        """
        self.callable_function = callable_function
        self.num_params = num_params
        self.nobs = nobs
        if param_names is None:
            param_names = [f'_param_{j}' for j in range(num_params)]
        self.param_names = param_names
        self.func_str = func_str
        self.python_code_str = python_code_str
        self.other_args = other_args

    def __call__(self, *args, **kwargs):
        """Forward to the underlying ``callable_function``."""
        return self.callable_function(*args, **kwargs)

    def __str__(self):
        """Pretty-print ``__dict__`` for interactive inspection."""
        return pprint.pformat(self.__dict__, indent=4)

    def __repr__(self):
        """Same as :meth:`__str__` (repr is verbose by design here)."""
        return str(self)

    def _check_has_auto_diff_node(self, debug=False):
        """Ensure ``auto_diff_node`` exists, parsing ``func_str`` on first derivative request."""
        if not hasattr(self, 'auto_diff_node'):
            self.auto_diff_node = AutoDiffGraphNode(self.func_str, debug=debug)

    # def get_analytical_gradient(self, do_jit=False):
    #     self._check_has_auto_diff_node()
    #     return self.auto_diff_node.get_analytical_jacobian(
    #         self.num_params, self.nobs, other_args=self.other_args, do_jit=do_jit)

    def get_analytical_partial_derivative(self, arg_num, do_jit=False, return_info=False, debug=False):
        """Return callable ∂f/∂``params[arg_num]`` (and optional metadata)."""
        self._check_has_auto_diff_node(debug=debug)
        return self.auto_diff_node.get_analytical_partial_derivative(
            arg_num, self.nobs, other_args=self.other_args, do_jit=do_jit, return_info=return_info,
            debug=debug)

    def get_analytical_partial_derivatives(self, do_jit=False, return_info=False, debug=False):
        """Return a list of partial-derivative callables, one per parameter index."""
        self._check_has_auto_diff_node(debug=debug)
        return self.auto_diff_node.get_analytical_partial_derivatives(
            self.num_params, self.nobs, other_args=self.other_args, debug=debug, do_jit=do_jit, return_info=return_info)

    def get_analytical_hessian(self, do_jit=False, nobs=None, agg_func=None, assume_symmetric=True, return_info=False,
                               debug=False):
        """Return Hessian of a scalar summary (mean / mean-squared) when ``nobs > 1`` requires ``agg_func``."""
        if nobs is None:
            nobs = self.nobs
        if agg_func:
            assert agg_func in ('mean', 'mean_squared')
        self._check_has_auto_diff_node(debug=debug)
        return self.auto_diff_node.get_analytical_hessian(
            self.num_params, nobs, other_args=self.other_args, do_jit=do_jit, agg_func=agg_func,
            assume_symmetric=assume_symmetric, return_info=return_info, debug=debug)

    def get_analytical_gradient(self, do_jit=False, nobs=None, agg_func=None, return_info=False, debug=False):
        """Return gradient of a scalar summary (mean / mean-squared) when ``nobs > 1`` requires ``agg_func``."""
        if nobs is None:
            nobs = self.nobs
        if agg_func:
            assert agg_func in ('mean', 'mean_squared')
        self._check_has_auto_diff_node(debug=debug)
        return self.auto_diff_node.get_analytical_gradient(
            self.num_params, nobs, other_args=self.other_args, do_jit=do_jit, agg_func=agg_func,
            return_info=return_info, debug=debug)

    def __neg__(self):
        """Negate the expression string and rebuild a :class:`FunctionCallable`."""
        func_str = f'-({self.func_str})'
        func_str = self._replace_param_arg_with_names(func_str)
        return func_str_to_callable(func_str, nopython=False, other_args=self.other_args, nobs=self.nobs)

    def __pos__(self):
        """Unary plus: rebuild from current expression (no structural change)."""
        func_str = self._replace_param_arg_with_names()
        return func_str_to_callable(func_str, nopython=False, other_args=self.other_args, nobs=self.nobs)

    def __add__(self, other):
        """Element-wise composition: ``self + other`` as a new string-based callable."""
        return self.__func_compose_internal(other, '+')

    __radd__ = __add__

    def __sub__(self, other):
        """``self - other``."""
        return self.__func_compose_internal(other, '-')

    def __rsub__(self, other):
        """``other - self`` (operand order reversed in string composition)."""
        return self.__func_compose_internal(other, '-', reverse_args=True)

    def __mul__(self, other):
        """``self * other``."""
        return self.__func_compose_internal(other, '*')

    __rmul__ = __mul__

    def __truediv__(self, other):
        """``self / other``."""
        return self.__func_compose_internal(other, '/')

    def __rtruediv__(self, other):
        """``other / self``."""
        return self.__func_compose_internal(other, '/', reverse_args=True)

    def __pow__(self, other):
        """``self ** other``."""
        return self.__func_compose_internal(other, '**')

    def __rpow__(self, other):
        """``other ** self``."""
        return self.__func_compose_internal(other, '**', reverse_args=True)

    def __func_compose_internal(self, other, operator, reverse_args=False):
        """Join two :class:`FunctionCallable` instances with ``operator`` and re-parse."""
        if isinstance(other, (int, float)):
            other = func_str_to_callable(f'{other}')
        if not isinstance(other, FunctionCallable):
            raise Exception
        func_str_self, func_str_other \
            = self._replace_param_arg_with_names(), other._replace_param_arg_with_names()

        if reverse_args:
            func_str_self, func_str_other = func_str_other, func_str_self

        func_str = f'({func_str_self}) {operator} ({func_str_other})'
        other_args = ','.join(
            [z for z in sorted(set(self.other_args.split(',') + other.other_args.split(',')))
             if not (z.isspace() or z == '')]
        )
        return func_str_to_callable(func_str, other_args=other_args, nobs=self.nobs)

    def _replace_param_arg_with_names(self, func_str=None):
        """Swap ``params[i]`` for ``{param_names[i]}`` placeholders (inverse of :func:`get_param_names`)."""
        if func_str is None:
            func_str = self.func_str
        for i, p in enumerate(self.param_names):
            func_str = func_str.replace(f'params[{i}]', f'{{{p}}}')
        return func_str

    def mean(self, *args, **kwargs):
        """Mean of the function value over the returned array (if any)."""
        return np.mean(self(*args, **kwargs))

    def mean_squared(self, *args, **kwargs):
        """Mean of squared function values."""
        return np.mean(self(*args, **kwargs) ** 2)

    # def sum(self, *args, **kwargs):
    #     return np.sum(self(*args, **kwargs))
    #
    # def sum_squared(self, *args, **kwargs):
    #     return np.sum(self(*args, **kwargs) ** 2)

    def get_frozen_function(self, *args, **kwargs):
        """``lambda params: self(params, *args, **kwargs)`` — only ``params`` vary."""
        return lambda params: self(params, *args, **kwargs)

    def get_frozen_mean_function(self, *args, **kwargs):
        """Frozen :meth:`mean` as a function of ``params`` only."""
        return lambda params: self.mean(params, *args, **kwargs)

    def get_frozen_mean_squared_function(self, *args, **kwargs):
        """Frozen :meth:`mean_squared` as a function of ``params`` only."""
        return lambda params: self.mean_squared(params, *args, **kwargs)

    def get_frozen_analytical_gradient(self, do_jit=False, nobs=None, agg_func=None, return_info=False, *args,
                                       **kwargs):
        """Gradient callable with ``*args``/``**kwargs`` bound (same semantics as :meth:`get_analytical_gradient`)."""
        result = self.get_analytical_gradient(do_jit=do_jit, nobs=nobs, agg_func=agg_func, return_info=return_info)
        if return_info:
            _grad_func, info = result
        else:
            _grad_func = result
        grad_func = lambda params: _grad_func(params, *args, **kwargs)
        if return_info:
            return grad_func, info
        else:
            return grad_func

    def get_frozen_analytical_hessian(self, do_jit=False, nobs=None, agg_func=None, assume_symmetric=True,
                                      return_info=False, *args, **kwargs):
        """Hessian callable with ``*args``/``**kwargs`` bound."""
        result = self.get_analytical_hessian(do_jit=do_jit, nobs=nobs, agg_func=agg_func, return_info=return_info,
                                             assume_symmetric=assume_symmetric)
        if return_info:
            _hess_func, info = result
        else:
            _hess_func = result
        hess_func = lambda params: _hess_func(params, *args, **kwargs)
        if return_info:
            return hess_func, info
        else:
            return hess_func


def get_param_names(func_str, param_delimiters=DEFAULT_PARAM_DELIMITERS):
    """Extract ordered parameter names and rewrite placeholders to ``params[i]``.

    Parameters
    ----------
    func_str : str
        Source containing parameter tokens between ``param_delimiters``.
    param_delimiters : tuple of (str, str), optional
        Opening and closing delimiter pair (default ``{`` / ``}``).

    Returns
    -------
    param_names : list of str
        Names in first-occurrence order from :func:`kanly.utils.parse_code_string.parse_code_str`.
    func_str : str
        Same expression with each ``{name}`` replaced by ``params[i]``.

    Raises
    ------
    Exception
        If parenthesis counts in the rewritten string do not match.
    """
    param_names, func_str = parse_code_str(func_str, *param_delimiters)
    if len(re.findall(r"\(", func_str)) != len(re.findall(r"\)", func_str)):
        raise Exception("Open- and closed-parentheses counts do not match!")
    #param_names = re.findall('{(.+?)}', func_str)
    #param_names = list(dict.fromkeys(param_names))
    for i, p in enumerate(param_names):
        func_str = func_str.replace(param_delimiters[0] + p + param_delimiters[1],
                                    f'params[{i}]')
    # print('*  ', param_names)
    # print('*  ', func_str)
    return param_names, func_str


def func_str_to_callable(func_str, nopython=False, other_args='', nobs=1, param_delimiters=DEFAULT_PARAM_DELIMITERS, debug=False):
    """Parse ``func_str`` into a :class:`FunctionCallable` (``exec`` of generated ``def``).

    Parameters
    ----------
    func_str : str
        Formula with parameter placeholders (see :func:`get_param_names`).
    nopython : bool, optional
        If True, prepend generated function with ``@jit`` from Numba.
    other_args : str, optional
        Comma-separated extra formal parameters after ``params`` (e.g. ``'x,y'``).
    nobs : int, optional
        Declared observation count (used by derivative helpers).
    param_delimiters : tuple, optional
        Delimiter pair for parameter tokens.
    debug : bool, optional
        If True, print generated ``python_code_str`` before ``exec``.

    Returns
    -------
    FunctionCallable
        Dill-serializable wrapper around the executed function.

    Examples
    --------
    Build a callable for a nonlinear residual expression and inspect its
    analytical Hessian:

    >>> import numpy as np
    >>> from kanly.api import func_str_to_callable
    >>> # parameters in curly braces are auto-discovered
    >>> func_str = 'c = {a}*{b}; y - {a} - {b}*x - c'
    >>> func = func_str_to_callable(func_str, other_args='x,y')
    >>> func.param_names
    ['a', 'b']
    >>> rng = np.random.default_rng(0)
    >>> n = 10
    >>> x = rng.normal(size=n)
    >>> y = 1.0 + 4.0 * x + rng.normal(size=n)
    >>> res = func.get_analytical_hessian(nobs=n, agg_func='mean_squared')   # doctest: +SKIP
    >>> H = res['hessian_callable']([1.0, 2.0], x=x, y=y)               # doctest: +SKIP
    """
    param_names, func_str = get_param_names(func_str, param_delimiters=param_delimiters)
    python_code_str = (f'{"@jit" if nopython else ""}\n'
                       f'def __func__temp__(params, {other_args}):\n'
                       f'   params = np.asarray(params);'
                       f'   return {func_str};')
    if debug:
        print(python_code_str)

    # 2. Pass the context to exec.
    # This forces the function to be created inside our dictionary.
    exec_context_dict = dict(globals())
    exec(python_code_str, exec_context_dict)

    # 3. Safely extract the function from the context dictionary
    dynamic_function = exec_context_dict['__func__temp__']

    # print(python_code_str)
    # exec(python_code_str)
    #print(locals())
    return FunctionCallable(dynamic_function, nobs, len(param_names), param_names,
                            func_str, python_code_str,
                            other_args)


# # TODO test code for vector-valued function
# if __name__ == '__main__':
#
#     # func_str = 'c = {a}*{b}-({a}-1.2)**2 - (.5+{b})**2 + ({a}-1.2)*{b}'
#     #
#     # func = func_str_to_callable(func_str)
#     # res = func.get_analytical_hessian(do_jit=True)
#
#     func_str = 'c = {a}*{b}; y - {a} - {b}*x - c'
#     func = func_str_to_callable(func_str, other_args='x,y')
#     print(func)
#
#     n = 10
#     np.random.seed(0)
#     x = np.random.randn(n)
#     y = 1 + 4 * x + np.random.randn(n)
#     res = func.get_analytical_hessian(nobs=n, agg_func='mean_squared')
#     print(res['func_str_code'])
#     print(res['hessian_callable']([1, 2], x=x, y=y))

# if __name__ == '__main__':
#
#     from kanly.api import bfgs_pqn
#     from kanly.api import func_str_to_callable
#
#
#     from kanly.utils.dict_2_array import dict_2_array
#
#     func = func_str_to_callable(f'({{x}}-1)**2 + ({{y}}+2)**2', debug=True)
#     x0 = dict_2_array({'x': 0, 'y': 2}, func.param_names)
#     result = bfgs_pqn(func, x0=x0, maxiter=10, maximize=False, ftol=1e-12, gtol=1e-12)
#
#     print(f'{func=}')
#     print('\n')
#     print(f'{result.x=}, {result.ferr=}, {result.gnorm=}')
