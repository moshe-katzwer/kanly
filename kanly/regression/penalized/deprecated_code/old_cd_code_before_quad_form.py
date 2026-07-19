
# def sparse_elastic_net_coordinate_descent(
#         X, y, weights=None, beta0=None, alpha=DEFAULT_EN_ALPHA, l1_ratio=DEFAULT_EN_L1_RATIO, max_iter=100, tol=1e-4,
#         fit_intercept=DEFAULT_EN_FIT_INTERCEPT, normalize=DEFAULT_EN_NORMALIZE, positive=DEFAULT_EN_POSITIVE,
#         debug=False, active_set=DEFAULT_EN_ACTIVE_SET,
#         apply_scaling=DEFAULT_EN_APPLY_SCALING):
#
#     tt = time.time()
#
#     if alpha < 0:
#         raise SparseElasticNetException("alpha must be non-negative!")
#     if l1_ratio < 0 or l1_ratio > 1:
#         raise SparseElasticNetException("l1_ratio must be in [0, 1]")
#
#     is_weighted = weights is not None
#     n, n_params = X.shape
#
#     if beta0 is None:
#         beta0 = np.zeros(n_params)
#     beta = beta0.copy()
#
#     if is_weighted:
#         if isinstance(weights, Series):
#             weights = weights.values
#         if isspmatrix(weights):
#             weights = weights.toarray().flatten()
#         weights = weights.astype(float)
#         weights *= n / weights.sum()
#     else:
#         weights = np.array([np.nan] * n)  # needed for numba, can't be None
#
#     if not isinstance(X, csc_matrix):
#         t = time.time()
#         if debug:
#             print("Converting design matrix to `scipy.sparse.csc_matrix`...", end='')
#         X = csc_matrix(X)
#         if debug:
#             print("%.2fs" % (time.time()-t))
#
#     if isspmatrix(y):
#         y = y.toarray()
#     elif isinstance(y, Series):
#         y = y.values
#     y = np.asarray(y).flatten()
#
#     from kanly.regression.linear_models.linear_model_2_quadratic_form import _linear_model_components_2_quadratic_form_and_likelihood
#     _, ssr_quad_form = _linear_model_components_2_quadratic_form_and_likelihood(y, X, weights)
#
#     _t = time.time()
#     if debug:
#         print('Getting scales and centers of regressors and adjusting penalties...', end='')
#     w_sum_squares_over_n, w_mean_x, l2_norm_X = _get_normalizing_factors(X, is_weighted, weights)
#     l1_penalties, l2_penalties = _get_penalties(alpha, l1_ratio, normalize, l2_norm_X)
#
#     if debug:
#         print('%.2fs' % (time.time()-_t))
#
#     intercept_ = 0.0
#
#     y_pred = np.zeros(n)
#     for j in range(n_params):
#         if beta[j] != 0:
#             y_pred[X.indices[X.indptr[j]:X.indptr[j + 1]]] += beta[j] * X.data[X.indptr[j]:X.indptr[j + 1]]
#
#     resid = y - y_pred
#     pbar = tqdm_kanly(range(max_iter), desc='Coordinate Descent Iterations', disable=not debug)
#
#     pause_coef_arr = np.array([False] * n_params)
#
#     last_updated = -1
#     beta_last = np.inf
#
#     param_index_rng = list(range(n_params))
#
#     if debug:
#         print('Beginning coordinate descent...')
#
#     full_update = True
#
#     _t = time.time()
#
#     converged = False
#     for itr in pbar:
#
#         # TODO iteration debug output
#
#         beta_old = beta.copy()
#
#         columns = np.array(param_index_rng)
#
#         last_updated, beta_last, intercept_ = \
#             sparse_coordinate_descent_update_iteration(
#                 columns, resid, X.data, X.indptr, X.indices, beta, is_weighted, w_sum_squares_over_n,
#                 w_mean_x, l1_penalties, l2_penalties, last_updated, beta_last, positive, fit_intercept, weights,
#                 )
#
#         with np.errstate(divide='ignore', invalid='ignore'):
#             diff = np.where(np.abs(beta_old) < 1, np.abs(beta_old - beta), np.abs(beta / beta_old - 1))
#
#         error = np.max(diff)
#         if itr > 1 and error < tol and full_update:
#             converged = True
#             break
#
#         full_update = (not active_set) or error < tol
#         if full_update:
#             param_index_rng = range(n_params)
#         else:
#             param_index_rng = np.arange(n_params)[diff > 0]
#
#         if debug:
#             pbar.set_description_str(
#                 "Coordinate Descent: eps=%.2e, zero=%.0f%%" % (error, 100.0 * np.sum(pause_coef_arr) / n_params))
#
#     pbar.close()
#
#     if debug:
#         print('Coordinate descent complete! {iters=%d, error=%.2e, time=%.2fs}' % (
#             itr+1, error, time.time() - _t))
#
#     resid -= intercept_
#     fitted_values = y - resid
#
#     if apply_scaling:
#         beta *= (1 + alpha * (1 - l1_ratio))
#         fitted_values = X.dot(csc_matrix(beta).reshape((-1,1))).toarray().flatten() + intercept_
#         resid = y - fitted_values
#
#     if is_weighted:
#         wtd_mean = np.average(y, weights=weights)
#         rsquared = 1.0 - np.average(resid ** 2, weights=weights) / np.average((y - wtd_mean) ** 2, weights=weights)
#     else:
#         rsquared = 1.0 - (resid ** 2).mean() / ((y - y.mean()) ** 2).mean()
#
#     return {
#         'intercept_': intercept_,
#         'coef_': beta.copy().flatten(),
#         'error': error,
#         'tol': tol,
#         'converged': converged,
#         'iters': itr+1,
#         'max_iter': max_iter,
#         'fit_time': time.time() - tt,
#         'fittedvalues': fitted_values,
#         'resid': resid,
#         'positive': positive,
#         'fit_intercept': fit_intercept,
#         'normalize': normalize,
#         'alpha': alpha,
#         'l1_ratio': l1_ratio,
#         'rsquared': rsquared,
#         'score': rsquared,
#         'l1_penalties': l1_penalties,
#         'l2_penalties': l2_penalties,
#     }
#
#
# @jit(nopython=True)
# def sparse_coordinate_descent_update_iteration(
#         columns, resid, X_data, X_indptr, X_indices, beta, is_weighted,
#         w_sum_squares_over_n, w_mean_x, l1_penalties, l2_penalties, last_updated, beta_last, positive, fit_intercept,
#         weights):
#
#     n = resid.shape[0]
#     for j in columns:
#
#         if last_updated != -1:
#             idx = X_indices[X_indptr[last_updated]:X_indptr[last_updated + 1]]
#             resid[idx] -= (beta[last_updated] - beta_last) * X_data[X_indptr[last_updated]:X_indptr[last_updated + 1]]
#
#         if j == last_updated and len(columns) > 1:
#             continue
#
#         if fit_intercept:
#             if is_weighted:
#                 intercept_ = resid.dot(weights) / n
#             else:
#                 intercept_ = resid.mean()
#
#         else:
#             intercept_ = 0.0
#
#         beta_last = beta[j]
#         last_updated = j
#
#         _x = X_data[X_indptr[j]:X_indptr[j + 1]]
#         _r = resid[X_indices[X_indptr[j]:X_indptr[j + 1]]]
#         if is_weighted:
#             _w = weights[X_indices[X_indptr[j]:X_indptr[j + 1]]]
#             rho = _x.dot(_w * _r) / n
#         else:
#             rho = _x.dot(_r) / n
#
#         rho += beta[j] * w_sum_squares_over_n[j] - intercept_ * w_mean_x[j]
#         rho = float(rho)
#
#         beta[j] = soft_threshold_en(rho, l1_penalties[j], l2_penalties[j], w_sum_squares_over_n[j])
#
#         if positive and beta[j] < 0:
#             beta[j] = 0.0
#
#     return last_updated, beta_last, intercept_
#
#
# @jit(nopython=True)
# def soft_threshold_en(partial_resid_corr, l1_penalty, l2_penalty, sum_squares_over_n):
#     if partial_resid_corr < -l1_penalty:
#         return (partial_resid_corr + l1_penalty) / (sum_squares_over_n + 2 * l2_penalty)
#     elif partial_resid_corr > l1_penalty:
#         return (partial_resid_corr - l1_penalty) / (sum_squares_over_n + 2 * l2_penalty)
#     else:
#         return 0


# if __name__ == '__main__':
#
#     max_iter = 200
#     normalize = False
#     alpha = .0001
#     l1_ratio = .25
#     tol = 1e-6
#
#     np.random.seed(0)
#     n = 1000000
#     k = 40
#     X = np.random.randn(n, k)
#     X = X.dot(np.random.randn(k, k))
#     X[X < 1.5] = 0
#     print(np.count_nonzero(X) / (n * k))
#     y = 1.2 + X.dot(np.arange(k)) + 5 * np.random.randn(n)
#     w = np.exp(np.random.randn(n)) * 20
#
#     X = csc_matrix(X)
#
#     _t = time.time()
#     res1 = sparse_elastic_net_coordinate_descent(X, y, weights=w, alpha=alpha, l1_ratio=l1_ratio, normalize=normalize,
#                                                  max_iter=max_iter, debug=True, tol=tol)
#     print(f'old = {time.time() - _t}')
#
#     _t = time.time()
#     res2 = sparse_elastic_net_coordinate_descent_quad_form(X, y, weights=w, alpha=alpha, l1_ratio=l1_ratio,
#                                                            normalize=normalize, prompt_user_for_more_iters=True,
#                                                            max_iter=max_iter, debug=True, tol=tol)
#     print(f'new = {time.time() - _t}')
#
#     print(np.max(np.abs(res1['coef_'] - res2['coef_'])))
#
#     # _tm = time.time()
#     # x0 = np.zeros(k)
#     #
#     # w = w * n / sum(w)
#     #
#     # XtX = X.T.dot(np.diag(w)).dot(X)
#     # XtX_diag = np.diag(XtX)
#     # sum_y = (w * y).sum()
#     # sum_X = np.array([sum(w * X[:, j]) for j in range(k)])
#     # sum_w = sum(w)
#     #
#     # Xty = X.T.dot(np.diag(w)).dot(y)
#     #
#     # w_sum_squares_over_n, w_mean_x, l2_norm_X = _get_normalizing_factors(X, True, w)
#     # l1_penalties, l2_penalties = _get_penalties(alpha, l1_ratio, normalize, l2_norm_X)
#     # print(l1_penalties, l2_penalties)
#     #
#     # # from tqdm import tqdm
#     # # print('====')
#     # # for t in tqdm(range(max_iter)):
#     # #     for j in range(k):
#     # #         a0 = (sum_y - np.dot(sum_X, x0)) / sum_w
#     # #
#     # #         numer = (Xty[j] - (XtX[j].dot(x0) + a0 * sum_X[j] - x0[j] * XtX_diag[j]))
#     # #         numer = np.sign(numer) * (np.abs(numer) - n * l1_penalties[j])
#     # #
#     # #         x0[j] = numer / (XtX_diag[j] + 2 * n * l2_penalties[j])
#     # #
#     # #         print("b: ", j, x0, a0)
#     # #
#     # # print(time.time() - _tm)
#
# if __name__ == '__main__':
#     np.random.seed(0)
#     X = csc_matrix(np.random.randn(10,4))
#     w = np.abs(.1+np.random.randn(10))
#     _get_normalizing_factors(X, True, w)
#     _get_normalizing_factors(X, True, 4*w)

#### GRADIENT DESCENT
# tol = 1e-6
#
# p0 = np.zeros(16)
# a0 = 0.
# step = 1
# X = fit_en.model.exog.toarray()
# y = fit_en.model.endog.toarray().flatten()
# y_mean = y.mean()
# X_mean = X.mean(axis=0)
# f0 = objective(a0, p0)
# err = np.inf
# df = 0.0
#
# flist = []
#
# for t in range(5000):
#
#     flist.append(f0)
#
#     accepted = 0
#     p0[np.abs(p0) < 1e-6] = 0.0
#
#     sg = subgrad(a0, p0)
#
#     a1 = y_mean - np.dot(X_mean, p0)
#     p1 = p0 - step * sg
#
#     f1 = objective(a1, p1)
#
#     if f1 < f0:
#         accepted += 1
#         m = 2
#         while True:
#             pnew = p0 - step * m * sg
#             anew = y_mean - np.dot(X_mean, pnew)
#             fnew = objective(anew, pnew)
#             if fnew < f1:
#                 accepted += 1
#                 m *= 2
#                 f1, p1, a1 = fnew, pnew, anew
#             else:
#                 break
#         step *= 2 ** (accepted - 1)
#         err = max(abs(p0 - p1))
#         df = f1 - f0
#         f0, p0, a0 = f1, p1, a1
#     else:
#         step /= 4
#
#     print("%6d" % t, "%6d" % accepted,
#           "%15.4e" % f0,
#           "%10.2e" % ((df / (f0 + df)) if accepted else np.nan),
#           "%10.2e" % err, "%.2e" % step)
#
#     if err < tol:
#         break
#
# if __name__ == '__main__':

#     def go():
#         import numpy as np
#         import pandas as pd

#         from kanly.api import elastic_net

#         n = 100
#         np.random.seed(0)
#         df = pd.DataFrame({
#             'x': np.random.randn(n),
#             'z': np.random.randint(0, 12, n),
#         })
#         df['y'] = 1.2 - 0.3 * df['x'] + .2 * np.random.randn(n)

#         fit = elastic_net('y ~ x + z', df, alpha=.1, debug=False,
#                           normalize=False,
#                           #relaxation_parameter=.15
#                           #regularize_to_values=[0,1.]
#                           )
#         print(fit.summary())


#         fit = elastic_net('y ~ x + z', df, alpha=.1, debug=False,
#                           normalize=False,
#                           relaxation_parameter=.15
#                           # regularize_to_values=[0,1.]
#                           )
#         print(fit.summary())
#     go()
