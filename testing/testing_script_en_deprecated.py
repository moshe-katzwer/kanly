# import pandas as pd
# import numpy as np
# from sklearn.pipeline import make_pipeline, Pipeline
# from sklearn.preprocessing import StandardScaler
#
# from kanly.api import *
# import statsmodels.formula.api as smf
# import time
# import patsy
# from kanly.formula.data_getter import SparseDataGetter
# from scipy.sparse import hstack, csc_matrix
# from numpy.testing import assert_almost_equal, assert_array_almost_equal
#
# import warnings
#
# warnings.filterwarnings("ignore")
#
# from scipy.optimize import minimize
#
# from sklearn.linear_model import ElasticNet
#
# np.random.seed(0)
# n = 300
# df = pd.DataFrame({'x': np.random.randint(0, 4, n), 'z': np.arange(n),
#                    'w': np.random.randn(n), 'grp': np.random.randint(0, 4, n), 'city': np.random.randint(0, 3, n),
#                    'wtsvar': .5 + np.random.rand(n)},
#                   index=np.random.choice(np.arange(10 * n), n, replace=False)  # TODO
#                   )
#
# df['wtsvar'] *= n / df.wtsvar.sum()
#
# e = np.random.randn(n)
# df['z'] = -3 + .15 * df.w + .4 * e + 2 * np.random.randn(n)
# df['y'] = 3 + 1.2 * df['x'] + df.z + 3 * e + df.city
# df['q'] = np.random.randn(n)
# df['_one_'] = 1
#
# result_dict = dict()
# for normalize in [False, True]:
#     for fit_intercept in [True, False]:
#         for alpha in [.01, .1, .33, 1]:
#             for l1_ratio in [.15, .5, .85]:
#                 for positive in [False, True]:
#                     for do_weighted in [True, False]:
#
#                         key = ("wtd=%5s, norm=%5s, intercept=%5s, pos=%5s, alpha=%s, l1_ratio=%s"
#                                % (do_weighted, normalize, fit_intercept, positive,
#                                   "%.2e" % alpha if isinstance(alpha, float) else str(alpha),
#                                   "%.2e" % l1_ratio if isinstance(l1_ratio, float) else str(l1_ratio)))
#                         result_dict[key] = [True, True, None, None]
#
#                         print("\n" * 5, '#' * 100)
#                         print(key)
#                         print()
#                         fit = elastic_net('y ~ z * q + x'
#                                           # ' + C(grp)*I(w**2)'
#                                           + (' $ wtsvar' if do_weighted else '')
#                                           ,
#                                           df, alpha=alpha, l1_ratio=l1_ratio,  # weights='wtsvar',
#                                           fit_intercept=fit_intercept, normalize=normalize, positive=positive,
#                                           xtol=1e-10, max_iter=50_000,
#                                           debug=False, active_set=False, selection='random',
#                                           )
#
#                         fit2 = ELASTIC_NET(fit.model.endog, fit.model.exog, weights=fit.model.weights,
#                                            alpha=alpha, l1_ratio=l1_ratio,
#                                            fit_intercept=fit_intercept, normalize=normalize, positive=positive,
#                                            xtol=1e-10, max_iter=50_000,
#                                            debug=False, active_set=False, selection='random'
#                                            )
#
#                         X, y = fit.model.exog.toarray(), fit.model.endog.toarray().flatten()
#                         if fit.model.weights is not None:
#                             wts = fit.model.weights.flatten().copy()
#                             wts *= n / sum(wts)
#                             w_mean_x = np.array([np.average(X[:, l], weights=wts) for l in range(X.shape[1])])
#
#                         else:
#                             wts = np.ones(n)
#                             w_mean_x = np.array([np.average(X[:, l]) for l in range(X.shape[1])])
#
#                         if normalize and fit_intercept:
#                             if do_weighted:
#                                 w_mean = np.average(X, weights=wts, axis=0)
#                                 l2_norm_x = np.average((X - w_mean) ** 2, axis=0, weights=wts) ** .5
#                                 # l2_norm_x = np.array(
#                                 #     [
#                                 #         ((X[:, l] - np.average(X[:, l], weights=wts)) ** 2).sum() ** .5
#                                 #         for l in range(X.shape[1])
#                                 #     ]
#                                 # )
#                             else:
#                                 l2_norm_x = X.std(axis=0)
#
#                                 # l2_norm_x = np.array(
#                                 #     [
#                                 #         ((X[:, l] - np.average(X[:, l])) ** 2).sum() ** .5
#                                 #         for l in range(X.shape[1])
#                                 #     ]
#                                 # )
#
#                             l1_penalties = alpha * l1_ratio * l2_norm_x
#                             l2_penalties = alpha * (1 - l1_ratio) / 2 * l2_norm_x ** 2
#
#                         else:
#                             l2_norm_x = 1.0
#                             l1_penalties = alpha * l1_ratio
#                             l2_penalties = alpha * (1 - l1_ratio) / 2
#
#                         print("--------")
#                         print(w_mean_x)
#                         print(l2_norm_x)
#                         print(l1_penalties)
#                         print(l2_penalties)
#
#                         if normalize:
#                             en = Pipeline([
#                                 ('center', StandardScaler()),
#                                 ('model', ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept,
#                                                      positive=positive, tol=1e-10, max_iter=20000, selection='random',
#                                                      ))
#                             ])
#                         else:
#                             en = Pipeline([
#                                 ('model', ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept,
#                                                      positive=positive, tol=1e-10, max_iter=20000, selection='random',
#                                                      ))
#                             ])
#
#                         if do_weighted:
#                             if normalize:
#                                 fit_sk = en.fit(X.copy(), y, model__sample_weight=wts, center__sample_weight=wts)
#                             else:
#                                 fit_sk = en.fit(X.copy(), y, model__sample_weight=wts)
#
#                         else:
#                             fit_sk = en.fit(X.copy(), y)
#
#
#                         def penalty(p):
#                             p = np.asarray(p)
#                             return sum(l1_penalties * np.abs(p) + l2_penalties * (p ** 2))
#
#
#                         def ssr(p, fit_intercept):
#                             p = np.asarray(p)
#                             if fit_intercept:
#                                 return (wts * (p[0] + X.dot(p[1:]).flatten() - y.flatten()) ** 2).sum()
#                             else:
#                                 return (wts * (X.dot(p).flatten() - y.flatten()) ** 2).sum()
#
#
#                         if fit_intercept:
#                             def objective_function_(p):
#                                 p = np.asarray(p)
#                                 return ssr(p, fit_intercept) / (2 * n) + penalty(p[1:])
#                         else:
#                             def objective_function_(p):
#                                 p = np.asarray(p)
#                                 return ssr(p, fit_intercept) / (2 * n) + penalty(p)
#
#                         res_scipy = minimize(objective_function_, np.array(fit.params.values),
#                                              method='SLSQP',
#                                              bounds=([(-10_000, 10_000)] if fit_intercept else []) + [
#                                                  (0.0 if positive else -10_000., 10_000.)] * X.shape[1]
#                                              )
#
#                         b0 = float(fit_sk[-1].intercept_)
#                         bcoef = fit_sk[-1].coef_.flatten()
#                         if normalize:
#                             bcoef = bcoef / l2_norm_x
#                             if fit_intercept:
#                                 pass
#
#                         sk_coef = np.hstack([fit_sk[-1].intercept_, bcoef])[
#                             int(not fit_intercept):]
#
#                         # print(fit.summary(show_only_non_zero=False))
#                         # print(res_scipy)
#
#                         tbl = pd.DataFrame(
#                             {
#                                 'kanly': fit.params.values,
#                                 'kanly_EN': fit2.params.values,
#                                 'sklearn': sk_coef,
#                                 'scipy': res_scipy.x,
#                             },
#                             index=fit.params.index,
#                         )
#                         print(tbl.round(4))
#
#                         print(('model', 'obj', 'ssr', 'penalty'))
#                         for c in tbl.columns:
#                             print(c, objective_function_(tbl[c]), ssr(tbl[c], fit_intercept),
#                                   penalty(tbl[c][int(fit_intercept):]))
#
#                         try:
#                             assert_array_almost_equal(fit.params, res_scipy.x, decimal=4)
#                         except:
#                             result_dict[key][0] = False
#                         try:
#                             assert_array_almost_equal(fit.params, sk_coef, decimal=4)
#                         except:
#                             result_dict[key][1] = False
#
#                         if ~np.all(result_dict[key][:2]):
#                             print(tbl)
#
#                             result_dict[key][2] = tbl.copy()
#                             result_dict[key][3] = {
#                                 c: objective_function_(tbl[c].values)
#                                 for c in tbl.columns
#                             }
#
#                             # kanly has better objective_function_ value when results don't match
#                             if result_dict[key][3]['kanly'] > result_dict[key][3]['sklearn']:
#                                 pass
#                                 # raise Exception(
#                                 #     result_dict[key][3]['kanly'], result_dict[key][3]['sklearn']
#                                 # )
#
#                             print('\n')
#                         # make sure that over-penalization doesn't make the comparison pointless
#                         # assert sum(np.abs(fit_sk.coef_)) > 0
#
# print("$" * 100)
#
# mismatched = 0
# failed = 0
# passed = 0
# for k, v in result_dict.items():
#     # print(k, v[:2])
#     if ~np.all(v[:2]):
#         mismatched += 1
#         # # print(v[2])
#         # # print(v[3])
#         # # print('kanly minus sklearn: ', v[3]['kanly'] - v[3]['sklearn'])
#         # if v[3]['kanly'] > v[3]['sklearn']:
#         #     failed += 1
#         print(k, ' failed')
#     else:
#         print(k, ' passed')
#         passed += 1
#
# print('\nMISMATCHED = ' + str(mismatched))
# print('\nFAILED     = ' + str(failed))
# print('\nPASSED     = ' + str(passed))
