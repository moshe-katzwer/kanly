# prediction.py – DRAFT / NOT YET ACTIVE
#
# This file contains a commented-out ``Prediction`` mixin that would provide
# a generic ``predict`` method supporting sparse, dense-array, and DataFrame
# inputs.  It has not been integrated into the model hierarchy yet.  Do not
# uncomment or import from this file until the design is finalised.
#
# from __future__ import absolute_import, print_function
#
# from numpy import ndarray
# from pandas import DataFrame, Series
# from scipy.sparse import csc_matrix, isspmatrix
#
# from kanly.formula.data_getter import SparseDataGetter
#
#
# class Prediction(object):
#
#     def _predict_with_data_arg(self, exog, fail_on_column_difference=False, debug=False, params=None,
#                                override_iv_error=False):
#         if params is None:
#             params = self.params.copy()
#         else:
#             if isinstance(params, Series):
#                 params = params.values
#             params = Series(index=self.params.index, data=params)
#         return Prediction._predict_with_data_arg_internal(
#             self, exog, params=params, fail_on_column_difference=fail_on_column_difference, debug=debug,
#             override_iv_error=override_iv_error)
#
#     @staticmethod
#     def _predict_with_data_arg_internal(fit, exog, params, fail_on_column_difference=False, debug=False,
#                                         override_iv_error=False):
#         # TODO debug output
#         # TODO predict on subset of variables - e.g. not all variables found in formula itself
#
#         if hasattr(fit, 'is_iv') and fit.is_iv:
#             if not override_iv_error:
#                 raise NotImplementedError("generic predict function not supported for instrumental variables!")
#
#         if isspmatrix(exog):
#             return exog.dot(csc_matrix(fit.params).transpose()).toarray().flatten()
#
#         elif isinstance(exog, ndarray):
#             return exog.dot(fit.params)
#
#         elif isinstance(exog, DataFrame):
#
#             exog_term_names = fit.exog_term_names.copy()
#             if not fail_on_column_difference or 'Intercept' not in fit.params.index:
#                 exog_term_names.append('-1')
#
#             exog_obj = SparseDataGetter.dmatrix(exog_term_names, exog, drop_1_for_FE=fail_on_column_difference,
#                                                 debug=debug)
#
#             exog_col_names = set(exog_obj.column_names)
#             fit_param_index = set(fit.params.index)
#
#             if fail_on_column_difference:
#                 if exog_col_names != fit_param_index:
#                     raise Exception("Col mismatch %s, %s" % (fit_param_index-exog_col_names,
#                                                              exog_col_names-fit_param_index))  # TODO msg
#
#             param_values_sp = csc_matrix(
#                 [params.loc[v] if v in fit.params.index else 0.0 for v in exog_obj.column_names]).transpose()
#
#             y_hat = exog_obj.values.dot(param_values_sp).toarray().flatten()
#             if 'Intercept' in params.index and 'Intercept' not in exog_obj.column_names:
#                 y_hat += params['Intercept']
#
#             return y_hat
#
#         else:
#             raise Exception(
#                 "`exog` must be one of `None`, `pd.DataFrame` "
#                 "or `numpy.ndarray` "
#                 "or `scipy.sparse.spmatrix`!\n`exog` type %s not supported"
#                 % str(type(exog))
#             )
