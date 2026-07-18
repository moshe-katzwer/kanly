from __future__ import absolute_import, print_function

import time

import numpy as np

from kanly.formula.data_getter import SparseDataGetter
from kanly.formula.keys import ENDOG_KEY, EXOG_KEY, INSTRUMENTS_KEY, WEIGHTS_KEY
from kanly.regression.partial_least_squares.pls import PLS1, PLS2
from kanly.utils.linalg_utils import DenseThreshold


class SparsePLSModel(object): # LinearModelBase
    """
    WIP
    """

    def __init__(self):
        pass

    def accepts_multi_outcome(self):
        True

    @staticmethod
    def build_model_from_formula(
            formula, data, debug=False, index=None, check_constant_cols=False,
            fail_on_missing=False, cache_intermediate=True, dense_threshold_mb=1024):

        _time = time.time()
        center = "".join(formula.split())[-2:] != "-1"
        if center:
            formula += " -1"

        build_data_result = SparseDataGetter.get_data(
            data, formula, check_constant_cols=check_constant_cols, absorb=None, debug=debug, _time=_time,
            fail_on_missing=fail_on_missing, cache_intermediate=cache_intermediate, sum_to_n=False,
            index=index, test_formula_on_dummy=False,
            drop_1_for_FE=False,
        )

        endog_obj = build_data_result[ENDOG_KEY]

        exog_obj = build_data_result[EXOG_KEY]
        if build_data_result[INSTRUMENTS_KEY] is not None:
            raise Exception("Instrumental variables not possible for PLS!")
        if build_data_result[WEIGHTS_KEY] is not None:
            raise NotImplementedError("Weighted regression not implemented yet!")

        y = endog_obj.values
        endog_name = endog_obj.column_names[0]
        if np.ndim(y) == 1 or y.shape[1] == 1:
            y = endog_obj.values.toarray().flatten()
            endog_name = endog_name[0]

        X = exog_obj.values
        if DenseThreshold.is_convertible_to_dense(X, dense_threshold_mb=dense_threshold_mb):
            X = X.toarray()
        exog_names = exog_obj.column_names
        model_elapsed = time.time() - _time

        return y, X, center, endog_name, exog_names, model_elapsed

    def fit(self, l):
        return PLS1, PLS2

    def predict(self, data=None, params=None, index=None, debug=False, *args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def pls1(self):
        pass

    @staticmethod
    def pls2(self):
        pass

# if __name__ == '__main__':
#     np.random.seed(0)
#     n = 1000
#     k = 12
#     l = 3
#     T = np.exp(np.random.randn(n, l))
#     X = T.dot(np.random.randn(l, k)) + .3 * np.random.randn(n, k)
#     y = T.dot(np.random.randn(l)) + 4 * np.random.randn(n) + 120
#
#     import pandas as pd
#     df = pd.DataFrame(X, columns=[f'x{j}' for j in range(k)])
#     df['y1'] = y
#     df['y2'] = y
#
#     y, X, center, endog_name, exog_names, model_elapsed = SparsePLSModel.build_model_from_formula(
#         'y1 + y2 ~ ' + " + ".join([f'x{j}' for j in range(k)]), df)
#
#     import matplotlib.pyplot as plt
#     for j in range(2):
#         plt.scatter(PLS2(y, X, l=2)['fittedvalues'][:,j],
#                     )