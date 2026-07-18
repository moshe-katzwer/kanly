import math
import numpy as np
import numba as nb
from numba.extending import overload
from scipy import special
from numba import vectorize, float64

RT_2 = 1.4142135623730951



def gamma(x):
    return special.gamma(x)

@vectorize([float64(float64)], cache=True)
def _gamma_ufunc(x):
    return math.gamma(x)

@overload(gamma)
def _nopython_gamma_nb(x):

    def impl(x):
        return _gamma_ufunc(x)

    return impl



def gammaln(x):
    return special.gammaln(x)


@vectorize([float64(float64)], cache=True)
def _gammaln_ufunc(x):
    return math.lgamma(x)


@overload(gammaln)
def _nopython_gammaln_nb(x):

    def impl(x):
        return _gammaln_ufunc(x)

    return impl




def erf(x):
    return special.erf(x)


@vectorize([float64(float64)], cache=True)
def _erf_ufunc(x):
    return math.erf(x)


@overload(erf)
def _nopython_erf_nb(x):

    def impl(x):
        return _erf_ufunc(x)

    return impl



def erfc(x):
    return special.erfc(x)


@vectorize([float64(float64)], cache=True)
def _erfc_ufunc(x):
    return math.erfc(x)


@overload(erfc)
def _nopython_erfc_nb(x):

    def impl(x):
        return _erfc_ufunc(x)

    return impl


def ndtr(x):
    return special.ndtr(x)


@vectorize([float64(float64)], cache=True)
def _ndtr_ufunc(x):
    return 0.5 * (1 + math.erf(x / RT_2))


@overload(ndtr)
def _nopython_ndtr_nb(x):

    def impl(x):
        return _ndtr_ufunc(x)

    return impl


    
    
def betaln(a, b):
    return special.betaln(a, b)

@vectorize([float64(float64, float64)], cache=True)
def _betaln_ufunc(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b) 


@overload(betaln)
def _nopython_betaln_nb(a, b):

    def impl(a, b):
        return _betaln_ufunc(a, b)

    return impl

    
def beta(a, b):
    return special.beta(a, b)

@vectorize([float64(float64, float64)], cache=True)
def _beta_ufunc(a, b):
    return np.exp(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))


@overload(beta)
def _nopython_beta_nb(a, b):

    def impl(a, b):
        return _beta_ufunc(a, b)

    return impl



# if __name__ == "__main__":
#     print(ndtr(.5))

#     from numba import njit
#     @njit
#     def f():
#         print(ndtr(0))
#         print(ndtr(np.array([0, 1.96])))

#     f()






# import math
# import numpy as np
# from numba import vectorize, float64, int64, int32, float32

# RT_2 = 1.4142135623730951


# # @vectorize([
# #     float64(float32),
# #     float64(float64),
# #     float64(int64),
# #     float64(int32)
# # ], cache=True)
# # def nopython_gammaln(x):
# #     return math.lgamma(x)


# @vectorize([
#     float64(float32),
#     float64(float64),
#     float64(int64),
#     float64(int32)
# ], cache=True)
# def nopython_erf(x):
#     return math.erf(x)

# @vectorize([
#     float64(float32),
#     float64(float64),
#     float64(int64),
#     float64(int32)
# ], cache=True)
# def nopython_ndtr(x):
#     'Cumulative distribution function for the standard normal distribution'
#     return (1.0 + nopython_erf(x / RT_2)) / 2.0


# @vectorize([
#     float64(float32),
#     float64(float64),
#     float64(int64),
#     float64(int32)
# ], cache=True)
# def nopython_gamma(x):
#     return math.gamma(x)


# @vectorize([
#     float64(float64, float64),
#     float64(float64, float32),
#     float64(float64, int64),
#     float64(float64, int32),
#     float64(float32, float64),
#     float64(float32, float32),
#     float64(float32, int64),
#     float64(float32, int32),
#     float64(int64, float64),
#     float64(int64, float32),
#     float64(int64, int64),
#     float64(int64, int32),
#     float64(int32, float64),
#     float64(int32, float32),
#     float64(int32, int64),
#     float64(int32, int32),
# ], cache=True)
# def nopython_betaln(a, b):
#     return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


# @vectorize([
#     float64(float64, float64),
#     float64(float64, float32),
#     float64(float64, int64),
#     float64(float64, int32),
#     float64(float32, float64),
#     float64(float32, float32),
#     float64(float32, int64),
#     float64(float32, int32),
#     float64(int64, float64),
#     float64(int64, float32),
#     float64(int64, int64),
#     float64(int64, int32),
#     float64(int32, float64),
#     float64(int32, float32),
#     float64(int32, int64),
#     float64(int32, int32),
# ], cache=True)
# def nopython_beta(a, b):
#     return np.exp(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))


# import math
# import numpy as np
# import numba as nb
# from numba import types
# from numba.extending import overload
# from scipy.special import gammaln


# def nopython_gammaln(x):
#     return gammaln(x)


# @overload(nopython_gammaln)
# def _nopython_gammaln_nb(x):
#     if isinstance(x, (types.Float, types.Integer)):
#         def impl(x):
#             return math.lgamma(nb.float64(x))
#         return impl

#     if isinstance(x, types.Array) and x.ndim == 1:
#         def impl(x):
#             out = np.empty(x.shape[0], dtype=np.float64)
#             for i in range(x.shape[0]):
#                 out[i] = math.lgamma(nb.float64(x[i]))
#             return out
#         return impl






# # """
# # Overload of numba-compliant scipy.special functions
# # The dependence on numba-scipy has been removed as that
# # repo is not maintained - we directly implement the key
# # functions here using math.
# # """
# # from __future__ import absolute_import, print_function

# # import math
# # import numpy as np

# # from numba import types
# # from numba.extending import overload

# # RT_2 = 1.4142135623730951


# # def nopython_gammaln(x):
# #     """Evaluate the log absolute gamma function.

# #     Args:
# #         x: Scalar or array-like input.

# #     Returns:
# #         ``scipy.special.gammaln(x)`` outside numba, with overload support for
# #         scalar and one-dimensional array inputs inside nopython functions.

# #     Examples
# #     --------
# #     Useful for safely evaluating log-densities with large shape parameters:

# #     >>> import numpy as np
# #     >>> from kanly.api import nopython_gammaln
# #     >>> nopython_gammaln(5.0)                              # doctest: +SKIP
# #     3.1780538303479458
# #     >>> nopython_gammaln(np.array([1.0, 2.0, 10.0]))       # doctest: +SKIP
# #     array([ 0.   ,  0.   , 12.802])
# #     """
# #     return math.lgamma(x)
# #     #return scsp.gammaln(x)


# # @overload(nopython_gammaln)
# # def gammaln_vec_impl(x):
# #     """Provide numba overloads for scalar and vector ``nopython_gammaln``.

# #     Args:
# #         x: Numba type for the input argument.

# #     Returns:
# #         Implementation callable specialized for scalar or array input.
# #     """
# #     if isinstance(x, (types.Integer, types.Float)):
# #         return lambda x: math.lgamma(float(x)) #scsp.loggamma(float(x))
# #     elif isinstance(x, types.Array):
# #         def _impl(x):
# #             """
# #             Numba implementation for the overload branch selected by argument types.

# #             Args:
# #                 x: Point or array of points at which to evaluate the function.

# #             Returns:
# #                 Scalar or vector result for the overloaded special function.
# #             """
# #             res = np.zeros(x.shape[0])
# #             for i, z in enumerate(x):
# #                 res[i] = math.lgamma(float(z))
# #                 #res[i] = scsp.loggamma(float(z))
# #             return res

# #         return _impl
# #     else:
# #         raise Exception(f'type {type(x)} not supported for `nopython_gammaln` implementation!')



# # def nopython_gamma(x):
# #     """Evaluate the gamma function with scipy.special semantics.

# #     Args:
# #         x: Scalar or array-like input.

# #     Returns:
# #         Gamma function value for ``x``.

# #     Examples
# #     --------
# #     Scalar and vector evaluation:

# #     >>> import numpy as np
# #     >>> from kanly.api import nopython_gamma
# #     >>> nopython_gamma(5.0)                                # doctest: +SKIP
# #     24.0
# #     >>> nopython_gamma(np.array([1.0, 2.0, 3.0, 4.0]))     # doctest: +SKIP
# #     array([1., 1., 2., 6.])
# #     """
# #     return math.gamma(x)
# #     #return scsp.gamma(x)



# # @overload(nopython_gamma)
# # def gamma_vec_impl(x):
# #     """Provide numba overloads for scalar and vector ``nopython_gamma``.

# #     Args:
# #         x: Numba type for the input argument.

# #     Returns:
# #         Implementation callable specialized for scalar or array input.
# #     """
# #     if isinstance(x, (types.Integer, types.Float)):
# #         return lambda x: math.gamma(float(x))
# #         #return lambda x: scsp.gamma(float(x))

# #     elif isinstance(x, types.Array):
# #         def _impl(x):
# #             """
# #             Numba implementation for the overload branch selected by argument types.

# #             Args:
# #                 x: Point or array of points at which to evaluate the function.

# #             Returns:
# #                 Scalar or vector result for the overloaded special function.
# #             """
# #             res = np.zeros(x.shape[0])
# #             for i, z in enumerate(x):
# #                 #res[i] = scsp.gamma(float(z))
# #                 res[i] = math.gamma(float(z))
# #             return res

# #         return _impl
# #     else:
# #         raise Exception(f'type {type(x)} not supported for `nopython_gamma` implementation!')


# # def nopython_betaln(a, b):
# #     """Evaluate the logarithm of the beta function.

# #     Args:
# #         a: First beta-function argument.
# #         b: Second beta-function argument.

# #     Returns:
# #         ``scipy.special.betaln(a, b)``.

# #     Examples
# #     --------
# #     Scalar and vector evaluation:

# #     >>> import numpy as np
# #     >>> from kanly.api import nopython_betaln
# #     >>> nopython_betaln(2.0, 5.0)                          # doctest: +SKIP
# #     -4.094344562222
# #     >>> nopython_betaln(np.array([2.0, 3.0]),
# #     ...                 np.array([5.0, 5.0]))              # doctest: +SKIP
# #     array([-4.094, -4.787])
# #     """
# #     return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
# #     # return scsp.betaln(a, b)


# # @overload(nopython_betaln)
# # def logbeta_vec_impl(a, b):
# #     """Provide numba overloads for scalar/vector beta log-normalizers.

# #     Args:
# #         a: Numba type for the first argument.
# #         b: Numba type for the second argument.

# #     Returns:
# #         Implementation callable for scalar-scalar, vector-vector,
# #         vector-scalar, or scalar-vector inputs.
# #     """
# #     if isinstance(a, (types.Integer, types.Float)) and isinstance(b, (types.Integer, types.Float)):
# #         #return lambda a, b: scsp.betaln(float(a), float(b))
# #         return lambda a, b: math.lgamma(float(a)) + math.lgamma(float(b)) - math.lgamma(float(a) + float(b))
# #     elif isinstance(a, types.Array) and isinstance(b, types.Array):
# #         def _impl(a, b):
# #             """
# #             Numba implementation for the overload branch selected by argument types.

# #             Args:
# #                 a: First shape or lower-bound parameter, matching scipy.stats naming for this distribution.
# #                 b: Second shape or upper-bound parameter, matching scipy.stats naming for this distribution.

# #             Returns:
# #                 Scalar or vector result for the overloaded special function.
# #             """
# #             res = np.zeros(a.shape[0])
# #             for i, (_a, _b) in enumerate(zip(a, b)):
# #                 res[i] = math.lgamma(_a) + math.lgamma(_b) - math.lgamma(_a + _b)
# #                 #res[i] = scsp.betaln(float(_a), float(_b))
# #             return res

# #         return _impl
# #     elif isinstance(a, types.Array) and isinstance(b, (types.Integer, types.Float)):
# #         def _impl(a, b):
# #             """
# #             Numba implementation for the overload branch selected by argument types.

# #             Args:
# #                 a: First shape or lower-bound parameter, matching scipy.stats naming for this distribution.
# #                 b: Second shape or upper-bound parameter, matching scipy.stats naming for this distribution.

# #             Returns:
# #                 Scalar or vector result for the overloaded special function.
# #             """
# #             b = float(b)
# #             res = np.zeros(a.shape[0])

# #             lgamma_b = math.lgamma(b)
# #             for i, _a in enumerate(a):
# #                 # res[i] = scsp.betaln(float(_a), b)
# #                 res[i] = math.lgamma(_a) + lgamma_b - math.lgamma(_a + b)
# #             return res

# #         return _impl
# #     elif isinstance(a, (types.Integer, types.Float)) and isinstance(b, types.Array):
# #         def _impl(a, b):
# #             """
# #             Numba implementation for the overload branch selected by argument types.

# #             Args:
# #                 a: First shape or lower-bound parameter, matching scipy.stats naming for this distribution.
# #                 b: Second shape or upper-bound parameter, matching scipy.stats naming for this distribution.

# #             Returns:
# #                 Scalar or vector result for the overloaded special function.
# #             """
# #             a = float(a)
# #             res = np.zeros(b.shape[0])
# #             lgamma_a = math.lgamma(a)
# #             for i, _b in enumerate(b):
# #                 # res[i] = scsp.betaln(a, float(_b))
# #                 res[i] = lgamma_a + math.lgamma(_b) - math.lgamma(a + _b)
# #             return res

# #         return _impl
# #     else:
# #         raise Exception(f'type {str((type(a), type(b)))} not supported for `nopython_betaln` implementation!')

# # def nopython_beta(a, b):
# #     """Evaluate the beta function.

# #     Args:
# #         a: First beta-function argument.
# #         b: Second beta-function argument.

# #     Returns:
# #         ``scipy.special.beta(a, b)``.

# #     Examples
# #     --------
# #     Scalar and vector evaluation:

# #     >>> import numpy as np
# #     >>> from kanly.api import nopython_beta
# #     >>> nopython_beta(2.0, 5.0)                            # doctest: +SKIP
# #     0.0166666666666...
# #     >>> nopython_beta(np.array([2.0, 3.0]),
# #     ...               np.array([5.0, 5.0]))                # doctest: +SKIP
# #     array([0.01667, 0.00833])
# #     """
# #     return np.exp(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
# #     #return #scsp.beta(a, b)


# # @overload(nopython_beta)
# # def beta_vec_impl(a, b):
# #     """Provide numba overloads for scalar/vector beta function calls.

# #     Args:
# #         a: Numba type for the first argument.
# #         b: Numba type for the second argument.

# #     Returns:
# #         Implementation callable for scalar-scalar, vector-vector,
# #         vector-scalar, or scalar-vector inputs.
# #     """
# #     if isinstance(a, (types.Integer, types.Float)) and isinstance(b, (types.Integer, types.Float)):
# #         # return lambda a, b: scsp.beta(float(a), float(b))
# #         return lambda a, b: np.exp(math.lgamma(float(a)) + math.lgamma(float(b)) - math.lgamma(float(a) + float(b)))
# #     elif isinstance(a, types.Array) and isinstance(b, types.Array):
# #         def _impl(a, b):
# #             """
# #             Numba implementation for the overload branch selected by argument types.

# #             Args:
# #                 a: First shape or lower-bound parameter, matching scipy.stats naming for this distribution.
# #                 b: Second shape or upper-bound parameter, matching scipy.stats naming for this distribution.

# #             Returns:
# #                 Scalar or vector result for the overloaded special function.
# #             """
# #             res = np.zeros(a.shape[0])
# #             for i, (_a, _b) in enumerate(zip(a, b)):
# #                 res[i] = np.exp(math.lgamma(_a) + np.lgamma(_b) - np.lgamma(_a + _b))
# #                 # res[i] = scsp.beta(float(_a), float(_b))
# #             return res

# #         return _impl
# #     elif isinstance(a, types.Array) and isinstance(b, (types.Integer, types.Float)):
# #         def _impl(a, b):
# #             """
# #             Numba implementation for the overload branch selected by argument types.

# #             Args:
# #                 a: First shape or lower-bound parameter, matching scipy.stats naming for this distribution.
# #                 b: Second shape or upper-bound parameter, matching scipy.stats naming for this distribution.

# #             Returns:
# #                 Scalar or vector result for the overloaded special function.
# #             """
# #             b = float(b)
# #             res = np.zeros(a.shape[0])
# #             lgamma_b = np.lgamma(b)
# #             for i, _a in enumerate(a):
# #                 np.exp(math.lgamma(float(_a)) + lgamma_b - math.lgamma(float(_a) + float(b)))
# #                 # res[i] = scsp.beta(float(_a), b)
# #             return res

# #         return _impl
# #     elif isinstance(a, (types.Integer, types.Float)) and isinstance(b, types.Array):
# #         def _impl(a, b):
# #             """
# #             Numba implementation for the overload branch selected by argument types.

# #             Args:
# #                 a: First shape or lower-bound parameter, matching scipy.stats naming for this distribution.
# #                 b: Second shape or upper-bound parameter, matching scipy.stats naming for this distribution.

# #             Returns:
# #                 Scalar or vector result for the overloaded special function.
# #             """
# #             a = float(a)
# #             res = np.zeros(b.shape[0])
# #             lgamma_a = math.lgamma(a)
# #             for i, _b in enumerate(a):
# #                 np.exp(lgamma_a + math.lgamma(float(_b)) - math.lgamma(a + float(_b)))
# #             return res

# #         return _impl
# #     else:
# #         raise Exception(f'type {str((type(a), type(b)))} not supported for `nopython_beta` implementation!')


# # def nopython_erf(z):
# #     """Evaluate the Gauss error function.

# #     Args:
# #         z: Scalar or array-like input.

# #     Returns:
# #         ``scipy.special.erf(z)``.

# #     Examples
# #     --------
# #     Scalar and vector evaluation:

# #     >>> import numpy as np
# #     >>> from kanly.api import nopython_erf
# #     >>> nopython_erf(1.0)                                  # doctest: +SKIP
# #     0.8427007929497149
# #     >>> nopython_erf(np.array([0.0, 1.0, 2.0])).round(3)   # doctest: +SKIP
# #     array([0.   , 0.843, 0.995])
# #     """
# #     #return scsp.erf(z)
# #     return math.erf(z)


# # @overload(nopython_erf)
# # def erf_vec_impl(z):
# #     """Provide numba overloads for scalar and vector ``nopython_erf``.

# #     Args:
# #         z: Numba type for the input argument.

# #     Returns:
# #         Implementation callable specialized for scalar or array input.
# #     """
# #     if isinstance(z, (types.Integer, types.Float)):
# #         return lambda z: math.erf(float(z))
# #     elif isinstance(z, types.Array):
# #         def _impl(z):
# #             """
# #             Numba implementation for the overload branch selected by argument types.

# #             Args:
# #                 z: Standardized input value or array.

# #             Returns:
# #                 Scalar or vector result for the overloaded special function.
# #             """
# #             res = np.zeros(z.shape[0])
# #             for i, _z in enumerate(z):
# #                 res[i] = math.erf(float(_z))
# #             return res

# #         return _impl
# #     else:
# #         raise Exception(f'type {type(z)} not supported for `nopython_erf` implementation!')


# # def nopython_ndtr(x):
# #     """Evaluate the standard normal CDF.

# #     Args:
# #         x: Scalar or array-like input.

# #     Returns:
# #         ``scipy.special.ndtr(x)``.

# #     Examples
# #     --------
# #     Standard-normal CDF at scalar and vector inputs:

# #     >>> import numpy as np
# #     >>> from kanly.api import nopython_ndtr
# #     >>> nopython_ndtr(1.645).round(4)                      # doctest: +SKIP
# #     0.95
# #     >>> nopython_ndtr(np.array([-2.0, 0.0, 2.0])).round(3) # doctest: +SKIP
# #     array([0.023, 0.5  , 0.977])
# #     """
# #     # return scsp.ndtr(x)
# #     return 0.5 * (1.0 + math.erf(x / RT_2))


# # @overload(nopython_ndtr)
# # def ndtr_vec_impl(x):
# #     """Provide numba overloads for scalar and vector ``nopython_ndtr``.

# #     Args:
# #         x: Numba type for the input argument.

# #     Returns:
# #         Implementation callable specialized for scalar or array input.
# #     """
# #     if isinstance(x, (types.Integer, types.Float)):
# #         return lambda x: 0.5 * (1.0 + math.erf(float(x) / RT_2))
# #     elif isinstance(x, types.Array):
# #         def _impl(x):
# #             """
# #             Numba implementation for the overload branch selected by argument types.

# #             Args:
# #                 x: Point or array of points at which to evaluate the function.

# #             Returns:
# #                 Scalar or vector result for the overloaded special function.
# #             """
# #             res = np.zeros(x.shape[0])
# #             for i, _z in enumerate(x):
# #                 res[i] = 0.5 * (1.0 + math.erf(float(_z) / RT_2))
# #             return res

# #         return _impl
# #     else:
# #         raise Exception(f'type {type(x)} not supported for `nopython_ndtr` implementation!')

# # #
# # # print(nopython_ndtr(0))
# # #
# # # from numba import njit
# # #
# # # @njit
# # # def f():
# # #     return nopython_ndtr(1.645), nopython_ndtr(np.array([-2, 0])), nopython_ndtr(1.964)
# # #
# # # print(f())
# # #
