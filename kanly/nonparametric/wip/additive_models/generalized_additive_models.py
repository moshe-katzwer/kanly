from __future__ import absolute_import, print_function

import time

import numpy as np
from kanly.nonparametric.generalized_additive_models import GeneralizedAdditiveModelResults

from kanly.formula.data_getter import SparseDataGetter
from kanly.formula.keys import ENDOG_KEY, EXOG_KEY
from kanly.nonparametric.interpolate import interp
from kanly.nonparametric.lowess import LOWESS
from kanly.regression.generalized_linear_models.families import GAUSSIAN, _get_link, _get_family

SMOOTHER_LOWESS = 'LOWESS'
SMOOTHER_POLY = 'POLY'
MAX_ITER = 20
TOL = 1e-6


def gam(formula, data, smoother=SMOOTHER_LOWESS, smoother_options=None,
        family=GAUSSIAN, link=None, max_iter=MAX_ITER, tol=TOL, test_formula_on_dummy=False,
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

    result = GAM(y, X, smoother=smoother, smoother_options=smoother_options,
                 family=family, link=link,
                 max_iter=max_iter, tol=tol,
                 debug=debug, specification_name=specification_name,
                 exog_names=X_cols, endog_name=y_col)

    if keep_data:
        result.data = data
    result.formula = formula_orig
    result.from_formula = True

    return result


def GAM(endog, exog, smoother=SMOOTHER_LOWESS, smoother_options=None, family=GAUSSIAN, link=None,
        max_iter=MAX_ITER, tol=TOL,
        debug=False,
        exog_names=None, endog_name=None, specification_name=None,
        ) -> GeneralizedAdditiveModelResults:
    t0 = time.time()

    if smoother_options is None:
        smoother_options = dict()

    smoother = smoother.upper()

    family = _get_family(family)
    link = _get_link(link, family)
    is_canonical_link = link.name() == family.canonical_link().name()
    if debug:
        print(f'family = {family}, link = {link}')

    n, num_cols = exog.shape
    alpha = link.link(endog.mean())
    f = np.zeros(exog.shape)
    cnt_inner = 0

    for itr in range(max_iter):
        eta = alpha + f.sum(axis=1)
        mu = link.inverse_link(eta)
        g_prime = link.deriv(mu)
        z = eta + (endog - mu) * g_prime

        if is_canonical_link:
            w = 1.0 / family.variance(mu)
        else:
            w = 1.0 / (g_prime ** 2 * family.variance(mu))

        f_temp = f.copy()
        cnt_temp = 0
        while cnt_temp < 100:
            cnt_temp += 1
            err = -1.0
            for j in range(num_cols):
                f_temp_sum = f_temp.sum(axis=1)
                alpha = np.average(z - f_temp_sum, weights=w)
                r_j = z - (alpha + f_temp_sum - f_temp[:, j])

                cnt_inner += 1
                if smoother == SMOOTHER_LOWESS:
                    low = LOWESS(r_j, exog[:, j], weights=w, return_sorted=False, **smoother_options)
                    f_new = low[1]
                elif smoother == SMOOTHER_POLY:
                    degree = smoother_options.get('degree', 3)
                    b = np.polyfit(exog[:, j], r_j, deg=degree, w=w)[::-1]
                    x_new = np.ones(n)
                    f_new = 0.0
                    for c in b:
                        f_new += c * x_new
                        x_new *= exog[:, j]
                else:
                    raise Exception(f"smoother must be {SMOOTHER_LOWESS} or {SMOOTHER_POLY}")

                f_new -= f_new.mean()

                err = max(err, np.max(np.abs(f_new - f_temp[:, j])))
                f_temp[:, j] = f_new
                if debug:
                    print('\t', j, err)
            if err < np.sqrt(tol):
                break

        error_outer = np.max([np.abs(f_temp[:, j] - f[:, j]).max()
                              for j in range(num_cols)])
        if debug:
            print(itr, error_outer)

        f = f_temp
        if error_outer < tol:
            break

    linear_predictor = alpha + f.sum(axis=1)
    fittedvalues = link.inverse_link(linear_predictor)
    resid = endog - fittedvalues
    pseudo_rsquared = np.corrcoef([endog, fittedvalues])[0][1] ** 2

    interp_funcs = [interp(exog[:, j], f[:, j], assume_sorted=False)
                    for j in range(num_cols)]

    def prediction_function(exog_):
        exog_ = np.asarray(exog_)
        if np.ndim(exog_) == 1:
            return alpha + np.sum([interp_funcs[j](x_) for j, x_ in enumerate(exog_)])
        else:
            return np.array([prediction_function(x_) for x_ in exog_])

    return GeneralizedAdditiveModelResults(
        alpha, f, fittedvalues, resid, linear_predictor, w, family, family.name(), link, link.name(), error_outer,
        itr, time.time() - t0, pseudo_rsquared, cnt_inner, prediction_function, interp_funcs,
        exog, endog, exog_names, endog_name, specification_name
    )

# #
# if __name__ == '__main__':
#     import matplotlib.pyplot as plt


#     def run():
#         n = 2500
#         np.random.seed(0)
#         X = np.random.randn(n, 3)
#         X = X.dot(np.eye(3) + np.ones((3, 3)) / 10)
#         y = (
#                 3
#                 + np.exp(X[:, 0] + 1.5) / 10
#                 + 1.2 * X[:, 1] ** 2
#                 - 3.7 * X[:, 2]
#                 + .3 * np.random.randn(n)
#         )
#         y = np.exp(y)

#         res = gam('y ~ x0+x1+x2', dict(y=y, x0=X[:, 0], x1=X[:, 1], x2=X[:, 2]),
#                   smoother='LOWESS',
#                   family='POISSON',
#                   smoother_options={'frac': .33, 'degree': 1},
#                   debug=True)
#         # f = res['f']
#         # alpha = res['alpha']
#         # resid = y - alpha - f.sum(axis=1)
#         #
#         # fig, ax = plt.subplots(ncols=X.shape[1] + 1)
#         # for j in range(X.shape[1]):
#         #     ax[j].scatter(X[:, j], resid + f[:, j])
#         #     ax[j].scatter(X[:, j], f[:, j], c='r')
#         # ax[X.shape[1]].scatter(y, res['fitted_values'], alpha=.5)
#         # plt.plot((y.min(), y.max()), (y.min(), y.max()), lw=2, c='k')
#         #
#         # plt.show()
#         return res


#     run()
