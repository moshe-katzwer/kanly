from __future__ import absolute_import, print_function

import numpy as np  # don't delete
import scipy as sp  # don't delete
import scipy        # don't delete
import numpy        # don't delete

from numba import njit  # do not delete njit!! -- it's used in `exec` block
# import numba_scipy # DO NOT DELETE!

from kanly.regression.nonlinear_least_squares.function_callables.jacobian import get_finite_diff_jacobian
from kanly.utils.linalg_utils import DEFAULT_DENSE_THRESHOLD_MB
from kanly.regression.nonlinear_least_squares.constants import DEFAULT_NLLS_JAC_METHOD
from kanly.automatic_differentiation.graph import \
    build_jacobian_from_string, build_partial_derivative_from_string
from kanly.dill_object import DillObject

from kanly.automatic_differentiation.elementary_functions import *  # don't delete

#from kanly.stats import IMPORT_STR
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


for c in ['numpy', 'np', 'scipy', 'sp', 'njit']:  # , 'numba_scipy']:
    if c not in locals():
        raise Exception


class PredictionFunction(DillObject):
    """Callable wrapper for generated nonlinear prediction code.

    Stores the generated prediction callable plus the compact float/int data
    arrays it reads from.  The object also exposes finite-difference and
    analytic Jacobian builders so optimizers can evaluate derivatives with a
    consistent API.
    """

    def __init__(self, func, num_params, float_arr, int_arr, float_var_2_col, int_var_2_col, func_str=None,
                 prediction_func_python_code_str=None, param_names=None, dense_threshold_mb=DEFAULT_DENSE_THRESHOLD_MB,
                 jac_method=DEFAULT_NLLS_JAC_METHOD, do_njit=False):
        """Initialise a prediction wrapper around generated code.

        Args:
            func: Callable accepting ``(params, float_arr, int_arr)`` and
                returning fitted values.
            num_params: Number of parameters expected by ``func``.
            float_arr: Dense float data array used by generated code.
            int_arr: Dense integer/categorical index array used by generated code.
            float_var_2_col: Mapping from float variable names to columns in
                ``float_arr``.
            int_var_2_col: Mapping from integer variable names to columns in
                ``int_arr``.
            func_str: Generated expression string before it is wrapped in a
                Python function.
            prediction_func_python_code_str: Full generated function code.
            param_names: Optional ordered parameter names.
            dense_threshold_mb: Dense Jacobian threshold in MB.
            jac_method: Finite-difference Jacobian method.
            do_njit: Whether the generated function was JIT-compiled.
        """

        self.func = func
        self.num_params = num_params

        self.float_arr = float_arr
        self.int_arr = int_arr
        self.float_var_2_col = float_var_2_col
        self.int_var_2_col = int_var_2_col
        self.nobs = self._get_nobs(float_arr, int_arr)

        self.func_str = func_str
        self.prediction_func_python_code_str = prediction_func_python_code_str
        self.do_njit = do_njit

        self.dense_threshold_mb = dense_threshold_mb
        self.jac_method = jac_method

        if param_names is None:
            param_names = [f'param{i}' for i in range(self.num_params)]
        self.param_names = param_names.copy()

        self.jacobian = get_finite_diff_jacobian(self, dense_threshold_mb=dense_threshold_mb, jac_method=jac_method)

    @staticmethod
    def _get_nobs(float_arr, int_arr):
        """Infer the number of observations from stored data arrays.

        Args:
            float_arr: Float data array, or ``None``.
            int_arr: Integer/categorical data array, or ``None``.

        Returns:
            Number of rows in the available data array.
        """
        if int_arr is None:
            return float_arr.shape[0]
        else:
            return int_arr.shape[0]

    def __call__(self, params, float_arr=None, int_arr=None, idx=None):
        """Evaluate predictions at ``params``.

        Args:
            params: Parameter vector of length ``self.num_params``.
            float_arr: Optional float data array override.
            int_arr: Optional integer data array override.
            idx: Optional row indexer for returning a subset of predictions.

        Returns:
            NumPy array of predicted values, optionally subset by ``idx``.
        """
        params = np.asarray(params)

        if len(params) != self.num_params:
            raise Exception(f"Must have {self.num_params} params!!")

        v = self.func(
            params,
            float_arr if float_arr is not None else self.float_arr,
            int_arr if int_arr is not None else self.int_arr,
        )
        if idx is not None:
            return v[idx]
        else:
            return v

    def _handle_null_data_dicts(self, float_arr, int_arr):
        """Replace missing data-array overrides with the arrays stored on ``self``.

        Args:
            float_arr: Optional float data array override.
            int_arr: Optional integer data array override.

        Returns:
            Tuple ``(float_arr, int_arr)`` with defaults filled in.
        """
        if float_arr is None:
            float_arr = self.float_arr
        if int_arr is None:
            int_arr = self.int_arr
        return float_arr, int_arr

    def __str__(self):
        """Return a readable expression with parameter indices replaced by names."""
        fs = self.func_str
        for j, nm in enumerate(self.param_names):
            fs = fs.replace(f'params[{j}]', nm)
        return fs

    @staticmethod
    def build_nonlinear_function_object(func_str, do_njit, param_names, float_arr, int_arr, float_var_2_col,
                                        int_var_2_col, custom_functions=dict(),
                                        dense_threshold_mb=DEFAULT_DENSE_THRESHOLD_MB,
                                        jac_method=DEFAULT_NLLS_JAC_METHOD, debug=False):
        """Compile generated expression code into a ``PredictionFunction``.

        Args:
            func_str: Generated expression string that returns fitted values.
            do_njit: Whether to decorate the generated function with Numba
                ``njit``.
            param_names: Ordered names for the parameter vector.
            float_arr: Float data array referenced by generated code.
            int_arr: Integer/categorical data array referenced by generated code.
            float_var_2_col: Mapping from float variable names to ``float_arr``
                columns.
            int_var_2_col: Mapping from integer variable names to ``int_arr``
                columns.
            custom_functions: Optional functions to expose to generated code.
            dense_threshold_mb: Dense Jacobian threshold in MB.
            jac_method: Finite-difference Jacobian method.
            debug: Reserved for generated-code diagnostics.

        Returns:
            ``PredictionFunction`` instance wrapping the generated callable.
        """

        for k, v in custom_functions.items():
            # Generated functions are executed in this module's namespace, so
            # custom functions must be registered globally before ``exec``.
            globals()[k] = v

        # TODO nogil=False, better or worse?
        jit_str = f'@{njit.__name__}(nogil=True)' if do_njit else ''
        prediction_func_python_code_str = (
            f'{jit_str}\n'
            f'def pred_func_internal(params, float_arr, int_arr):\n'
            f'\t\treturn ({func_str})\n'
        )

        exec_context = dict(globals())
        exec(prediction_func_python_code_str, exec_context)
        pred_func = exec_context['pred_func_internal']

        return PredictionFunction(
            pred_func, len(param_names), float_arr, int_arr, float_var_2_col, int_var_2_col, param_names=param_names,
            func_str=func_str, prediction_func_python_code_str=prediction_func_python_code_str,
            dense_threshold_mb=dense_threshold_mb, jac_method=jac_method, do_njit=do_njit)

    def copy(self, copy_data=True, debug=False):
        """Return a new ``PredictionFunction`` with optionally copied data arrays.

        Args:
            copy_data: When ``True``, copy ``float_arr`` and ``int_arr``;
                otherwise share them with the original object.
            debug: Forwarded to ``build_nonlinear_function_object``.

        Returns:
            New ``PredictionFunction`` instance.
        """

        if copy_data:
            float_arr = None if self.float_arr is None else self.float_arr.copy()
            int_arr = None if self.int_arr is None else self.int_arr.copy()

        else:
            float_arr = self.float_arr
            int_arr = self.int_arr

        return PredictionFunction.build_nonlinear_function_object(
            self.func_str, self.do_njit, self.param_names.copy(), float_arr, int_arr, self.float_var_2_col.copy(),
            self.int_var_2_col.copy(), custom_functions=dict(), dense_threshold_mb=self.dense_threshold_mb,
            jac_method=self.jac_method, debug=debug)

    def reindex(self, idx, inplace=False):
        """Subset the stored data arrays to a row indexer.

        Args:
            idx: Row indexer or boolean mask.
            inplace: When ``True``, mutate this object; otherwise return a new
                reindexed ``PredictionFunction``.

        Returns:
            ``None`` when ``inplace=True``; otherwise a reindexed copy.
        """

        if inplace:
            if self.float_arr is not None:
                self.float_arr = self.float_arr[idx]
            if self.int_arr is not None:
                self.int_arr = self.int_arr[idx]
            self.nobs = self._get_nobs(self.float_arr, self.int_arr)

        else:

            if self.float_arr is not None:
                float_arr_new = self.float_arr[idx]
            else:
                float_arr_new = None

            if self.int_arr is not None:
                int_arr_new = self.int_arr[idx]
            else:
                int_arr_new = None

            return PredictionFunction(
                self.func, self.num_params, float_arr_new, int_arr_new, self.float_var_2_col, self.int_var_2_col,
                func_str=self.func_str, prediction_func_python_code_str=self.prediction_func_python_code_str,
                param_names=np.array(self.param_names), jac_method=self.jac_method
            )

    def get_analytical_jacobian(self, dense_threshold_mb=DEFAULT_DENSE_THRESHOLD_MB, debug=False, do_jac_jit=True):
        """
        :return: jacobian callable, dict of info

        Args:
            dense_threshold_mb: Dense Jacobian threshold in MB.
            debug: Whether to print automatic-differentiation diagnostics.
            do_jac_jit: Whether to JIT compile generated analytic Jacobian code.
        """

        if hasattr(self, '__analytic_jac') and self.__analytic_dense_threshold_mb == dense_threshold_mb and self.__do_jit == do_jac_jit:
            return self.__analytic_jac, self.__jac_result

        try:
            jac_result = build_jacobian_from_string(
                self.func_str, self.num_params, self.nobs, other_args='float_arr=None, int_arr=None',
                dense_threshold_mb=dense_threshold_mb, debug=debug, do_jit=do_jac_jit)

            # print("\n\n"*10, "###\n", jac_result['func_str_code'], "\n\n"*10)

            def jacobian(params):
                """Evaluate the generated analytic Jacobian at ``params``.

                Args:
                    params: Parameter vector.

                Returns:
                    Dense or sparse Jacobian of predictions with respect to
                    parameters.
                """
                params = np.asarray(params)
                return jac_result['jacobian_callable'](params, float_arr=self.float_arr, int_arr=self.int_arr)

            self.__analytic_jac, self.__jac_result, self.__analytic_dense_threshold_mb, self.__do_jac_jit = \
                jacobian, jac_result, dense_threshold_mb, do_jac_jit

            return jacobian, jac_result

        except Exception as e:
            print("An analytic jacobian could not be computed!")
            raise e

    def get_analytical_partial_derivative(self, arg_number, debug=False, do_jit=True, return_info=False):
        """Build an analytic partial-derivative callable for one parameter.

        Args:
            arg_number: Parameter index to differentiate with respect to.
            debug: Whether to print automatic-differentiation diagnostics.
            do_jit: Whether to JIT compile generated derivative code.
            return_info: Whether to return derivative-generation metadata.

        Returns:
            Callable partial derivative, optionally paired with an info dict.
        """

        try:
            res = build_partial_derivative_from_string(
                self.func_str, arg_number, self.nobs, other_args='float_arr=None, int_arr=None',
                debug=debug, do_jit=do_jit, return_info=return_info)

            if return_info:
                _pd, pd_info = res
            else:
                _pd = res

            def partial_derivative(params):
                """Evaluate the generated partial derivative at ``params``.

                Args:
                    params: Parameter vector.

                Returns:
                    Derivative vector for one parameter.
                """
                params = np.asarray(params)
                return _pd(params, float_arr=self.float_arr, int_arr=self.int_arr)

            if return_info:
                return partial_derivative, pd_info
            else:
                return partial_derivative

        except Exception as e:
            print("An analytic partial derivative could not be computed!")
            raise e

    def get_analytical_partial_derivatives(self, debug=False, do_jit=True, return_info=False):
        """Build analytic partial-derivative callables for every parameter.

        Args:
            debug: Whether to print automatic-differentiation diagnostics.
            do_jit: Whether to JIT compile generated derivative code.
            return_info: Whether to return derivative-generation metadata.

        Returns:
            List of partial-derivative callables, optionally paired with a list
            of info dicts.
        """

        try:
            result = [self.get_analytical_partial_derivative(i, debug=debug, do_jit=do_jit, return_info=return_info)
                      for i in range(self.num_params)]

            if return_info:
                return [r[0] for r in result], [r[1] for r in result]
            else:
                return result

        except Exception as e:
            print("Analytic partial derivatives could not be computed!")
            raise e


# if __name__ == '__main__':
#
#     from kanly.api import nlls
#     from tqdm import tqdm
#
#     np.random.seed(0)
#     n = 50_000
#     x = 1.56 * np.random.randn(n)
#     z = np.random.rand(n)
#     y = 3 + 10 * x - 2 * z + np.random.randn(n) * 3
#     wts = .01 + np.random.rand(n)
#     g = np.random.randint(0, 25, n)
#     data = {'x': x, 'y': y, 'z': z, 'wts': wts, 'g': g}
#
#     fit = nlls('[y] ~ {a} + {b}*[x] + [C(g,-1)]', data)
#     print(fit)
#
#     r_func = fit.model.residual_function_callable.copy()
#
#     from pathos.multiprocessing import Pool
#     from multiprocessing import RLock
#
#     n_chains = 10
#     pool = Pool(processes=n_chains, initargs=(RLock(),), initializer=tqdm.set_lock)
#
#
#     def run_function(func, x):
#         for _ in tqdm(range(30_000)):
#             func(x)
#         return 1
#
#
#     jobs = [
#         pool.apply_async(run_function, _arg) for _arg in
#         [(#r_func.copy(),
#             r_func,
#           [1, 2] + [0] * 24) for i in range(n_chains)]
#     ]
#     pool.close()
#     pool.join()
