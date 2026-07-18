from __future__ import absolute_import, print_function

import time

import numpy as np

from kanly.formula.data_getter import SparseDataGetter
from kanly.formula.keys import ENDOG_KEY, EXOG_KEY
from kanly.nonparametric.additive_models.results import GeneralizedAdditiveModelResults
from kanly.nonparametric.gaussian_kernel_smooth import gaussian_kernel_smooth
from kanly.regression.generalized_linear_models.families import GAUSSIAN, _get_family

SMOOTHER_KERNEL = 'KERNEL'
SMOOTHER_POLY = 'POLY'
DEFAULT_SMOOTHER = SMOOTHER_KERNEL
SMOOTHERS = [SMOOTHER_KERNEL, SMOOTHER_POLY]
TOL = 1e-6
MAX_ITER = 20


def am(formula, data, smoother=DEFAULT_SMOOTHER, smoother_options=None,
       max_iter=MAX_ITER, tol=TOL, test_formula_on_dummy=False,
       debug=False, keep_data=False, specification_name=None) -> GeneralizedAdditiveModelResults:
    formula_orig = formula
    formula_temp = formula.replace(' ', '').strip()
    if formula_temp[-2:] != '-1':
        formula += ' -1'

    data_res = SparseDataGetter.get_data(data, formula, test_formula_on_dummy=test_formula_on_dummy)
    y_obj, X_obj = data_res[ENDOG_KEY], data_res[EXOG_KEY]
    y = y_obj.values.toarray().flatten()
    y_col = y_obj.column_names[0]
    X = X_obj.values.toarray()
    X_cols = X_obj.column_names

    result = AM(y, X, smoother=smoother, smoother_options=smoother_options,
                max_iter=max_iter, tol=tol,
                debug=debug, specification_name=specification_name,
                exog_names=X_cols, endog_name=y_col)

    if keep_data:
        result.data = data
    result.formula = formula_orig
    result.from_formula = True

    return result


def AM(endog, exog, smoother=DEFAULT_SMOOTHER, smoother_options=None,
       max_iter=MAX_ITER, tol=TOL,
       debug=False,
       weights=None,
       exog_names=None, endog_name=None, specification_name=None,
       ) -> GeneralizedAdditiveModelResults:
    t0 = time.time()

    if smoother_options is None:
        smoother_options = dict()

    smoother = smoother.upper()
    assert smoother in SMOOTHERS

    n, num_cols = exog.shape
    alpha = endog.mean()

    f = np.zeros(exog.shape)
    if weights is None:
        weights = np.ones_like(endog)

    for itr in range(max_iter):
        for j in range(num_cols):
            r = endog - f.sum(axis=1) + f[:, j]
            err = -1
            if smoother == SMOOTHER_KERNEL:
                func = gaussian_kernel_smooth(exog[:, j], r, weights=weights, return_arrays=False, **smoother_options)
                f_new = func(exog[:, j])

            elif smoother == SMOOTHER_POLY:
                degree = smoother_options.get('degree', 3)
                b = np.polyfit(exog[:, j], r, deg=degree, w=weights)[::-1]
                x_new = np.ones(n)
                f_new = 0.0
                for c in b:
                    f_new += c * x_new
                    x_new *= exog[:, j]

            mean_f_j = f_new.mean()
            f_new -= mean_f_j
            err = max(err, np.abs(f[:, j] - f_new).max())
            f[:, j] = f_new
        if debug:
            print(f'{itr=}, {err:.2e}')
        if err < tol:
            break

    linear_predictor = alpha + f.sum(axis=1)
    fittedvalues = linear_predictor
    resid = endog - fittedvalues
    pseudo_rsquared = np.corrcoef([endog, fittedvalues])[0][1] ** 2

    interp_funcs = []
    for j in range(num_cols):
        idx = np.argsort(exog[:, j])
        interp_funcs.append(
            interp(exog[:, j][idx], f[:, j][idx])
        )

    def prediction_function(exog_):
        exog_ = np.asarray(exog_)
        if np.ndim(exog_) == 1:
            return alpha + np.sum([interp_funcs[j](x_) for j, x_ in enumerate(exog_)])
        else:
            return np.array([prediction_function(x_) for x_ in exog_])

    family = _get_family(GAUSSIAN)
    link = family.canonical_link()()

    return GeneralizedAdditiveModelResults(
        alpha, f, fittedvalues, resid, linear_predictor, weights, family, family.name(), link, link.name(), err,
        itr, time.time() - t0, pseudo_rsquared, itr, prediction_function, interp_funcs,
        exog, endog, exog_names, endog_name, specification_name
    )
