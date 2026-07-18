"""
Distribution imports used in Bayesian Data Model class
"""

# DEPRECATED IMPORT STUFF
# from __future__ import absolute_import, print_function
#
# from kanly.stats.distributions.nopython_frozen_logpdf import (
#     __frozen_internal_logpdf_genextreme, __frozen_internal_logpdf_norm, __frozen_internal_logpdf_truncnorm,
#     __frozen_internal_logpdf_beta, __frozen_internal_logpdf_cauchy, __frozen_internal_logpdf_laplace,
#     __frozen_internal_logpdf_expon, __frozen_internal_logpdf_t, __frozen_internal_logpdf_gamma,
#     __frozen_internal_logpdf_lognorm, __frozen_internal_logpdf_invgamma, __frozen_internal_logpdf_logistic,
#     __frozen_internal_logpdf_chi2, __frozen_internal_logpdf_gennorm, __frozen_internal_logpdf_multivariate_normal,
#     __frozen_internal_logpdf_multivariate_t, __frozen_internal_logpdf_halfnorm, __frozen_internal_logpdf_pareto,
#     __frozen_internal_logpdf_halfcauchy, __frozen_internal_logpdf_loguniform, __frozen_internal_logpdf_f,
#     __frozen_internal_logpdf_weibull_min, __frozen_internal_logpdf_dirichlet, __nopython_frozen_internal_logpdf_norm,
#     __nopython_frozen_internal_logpdf_halfnorm, __nopython_frozen_internal_logpdf_beta,
#     __nopython_frozen_internal_logpdf_cauchy, __nopython_frozen_internal_logpdf_laplace,
#     __nopython_frozen_internal_logpdf_expon, __nopython_frozen_internal_logpdf_t,
#     __nopython_frozen_internal_logpdf_multivariate_t, __nopython_frozen_internal_logpdf_gamma,
#     __nopython_frozen_internal_logpdf_lognorm, __nopython_frozen_internal_logpdf_invgamma,
#     __nopython_frozen_internal_logpdf_logistic, __nopython_frozen_internal_logpdf_chi2,
#     __nopython_frozen_internal_logpdf_gennorm, __nopython_frozen_internal_logpdf_multivariate_normal,
#     __nopython_frozen_internal_logpdf_truncnorm, __nopython_frozen_internal_logpdf_pareto,
#     __nopython_frozen_internal_logpdf_halfcauchy, __nopython_frozen_internal_logpdf_loguniform,
#     __nopython_frozen_internal_logpdf_genextreme, __nopython_frozen_internal_logpdf_f,
#     __nopython_frozen_internal_logpdf_weibull_min, __nopython_frozen_internal_logpdf_dirichlet,
#     get_frozen_logpdf_pareto, get_frozen_logpdf_norm, get_frozen_logpdf_truncnorm, get_frozen_logpdf_halfnorm,
#     get_frozen_logpdf_beta, get_frozen_logpdf_cauchy, get_frozen_logpdf_laplace, get_frozen_logpdf_expon,
#     get_frozen_logpdf_t, get_frozen_logpdf_gamma, get_frozen_logpdf_invgamma, get_frozen_logpdf_lognorm,
#     get_frozen_logpdf_logistic, get_frozen_logpdf_gennorm, get_frozen_logpdf_chi2,
#     get_frozen_logpdf_multivariate_normal, get_frozen_logpdf_halfcauchy, get_frozen_logpdf_multivariate_t,
#     get_frozen_logpdf_uniform, get_frozen_logpdf_flat, get_frozen_logpdf_loguniform, get_frozen_logpdf_genextreme,
#     get_frozen_logpdf_f, get_frozen_logpdf_weibull_min, get_frozen_logpdf_dirichlet,
# )
#
# from kanly.stats.distributions.nopython_logpdf import (
#     logpdf_beta, logpdf_cauchy, logpdf_chi2, logpdf_expon, logpdf_f, logpdf_gamma, logpdf_genextreme, logpdf_halfcauchy,
#     logpdf_halfnorm, logpdf_invgamma, logpdf_laplace, logpdf_logistic, logpdf_lognorm, logpdf_norm, logpdf_pareto,
#     logpdf_t, logpdf_truncnorm, logpdf_weibull_min, logpdf_multivariate_normal, logpdf_multivariate_t,
#     nopython_logpdf_beta, nopython_logpdf_cauchy, nopython_logpdf_chi2, nopython_logpdf_expon, nopython_logpdf_f,
#     nopython_logpdf_gamma, nopython_logpdf_genextreme, nopython_logpdf_halfcauchy, nopython_logpdf_halfnorm,
#     nopython_logpdf_invgamma, nopython_logpdf_laplace, nopython_logpdf_logistic, nopython_logpdf_lognorm,
#     nopython_logpdf_norm, nopython_logpdf_multivariate_normal, nopython_logpdf_pareto, nopython_logpdf_t,
#     nopython_logpdf_multivariate_t, nopython_logpdf_truncnorm, nopython_logpdf_weibull_min,
# )
#
# from kanly.stats.distributions.nopython_scipy_special import (
#     nopython_ndtr, nopython_erf, nopython_gammaln, nopython_betaln
# )
#
# from kanly.stats.distributions.nopython_frozen_logpdf import NOPYTHON_FROZEN_LOGPDF_IMPORT_STRING
# from kanly.stats.distributions.nopython_logpdf import NOPYTHON_LOGPDF_IMPORT_STRING
#
# IMPORT_STR = f"""
# from numba import jit, njit
# import numpy
# import numpy as np
# import numpy.linalg as np_linalg
# import scipy
# import scipy as sp
# import scipy.linalg as sp_linalg
# import scipy.special as sp_special
# import scipy.stats as stats
# # import numba_scipy
# from kanly.utils.logit_functions import logit, log_d_expit, expit, d_expit
# from kanly.automatic_differentiation.elementary_functions import (
#     sin, cos, tan, arcsin, arccos, arctan, cbrt,
#     sqrt, log, log2, log10, exp)
# from kanly.utils.stats_functions import (
#     std_normal_cdf, normal_cdf, normal_pdf,
#     normal_logpdf, log_normal_pdf, log_normal_logpdf, log_normal_cdf)
# """
#
# IMPORT_STR += '\n' + '\n'.join([NOPYTHON_LOGPDF_IMPORT_STRING, NOPYTHON_FROZEN_LOGPDF_IMPORT_STRING])
