# import pandas as pd
#
# pd.set_option('display.max_rows', 60)
#
# from kanly.bayes.bayesian_regression_model import BayesianNonlinearLeastSquaresModel
# import numpy as np
# import pandas as pd
# from numpy.testing import assert_allclose
# from scipy.optimize import minimize
# from scipy.stats import norm
# # from kanly.bayes.utils.fast_logpdf import fast_normal_logpdf
#
# n = 1050
# np.random.seed(0)
# df = pd.DataFrame({'x': np.random.randn(n)})
# df['p'] = 1 / (1 + np.exp(-(.4 + .9 * df.x)))
# df['y'] = (np.random.rand(n) < df['p']).astype(float)
#
# bounds = {'beta': [.4, 1]}
#
# bmodel1 = BayesianNonlinearLeastSquaresModel.build_model_from_formula(
#     '[y] ~ 1.0 / (1.0 + np.exp(-({alpha} + {beta} * [x])))',
#     df,
#     bounds=bounds, do_bounded_transform=True)
#
# x0 = [0., 0, .5]
#
#
# def llf_by_hand(x):
#     alpha, beta, sigma2 = x
#     sigma2 = max(sigma2, 1e-10)
#     r = df.y - 1.0 / (1.0 + np.exp(-(alpha + beta * df.x)))
#     llf = -n / 2 * np.log(2 * np.pi * sigma2) - np.sum(r ** 2) / (2 * sigma2)
#     return llf
#
#
# def prior_log_pdf(x):
#     beta = x[1]
#     return norm.logpdf(beta, .6, .02)
#
#
# for priors in [None, {'beta': 'norm(.6, .02)'}]:
#     bmodel1.set_priors(priors)
#
#
#     def neg_log_posterior_by_hand(x):
#         # negative, maximize
#         if priors is None:
#             return -llf_by_hand(x)
#         else:
#             return -(llf_by_hand(x) + prior_log_pdf(x))
#
#
#     scipy_result = minimize(neg_log_posterior_by_hand, x0,
#                             bounds=[(-np.inf, np.inf), (-np.inf, np.inf), (0., np.inf)],
#                             method='slsqp')
#
#     for transformed in [True, False]:
#         x0_map = bmodel1.transform(x0) if transformed else x0
#         map_result = bmodel1.maximize_posterior(x0_map, transformed=transformed)
#         x_map = map_result['params']
#         if transformed:
#             x_map = bmodel1.transform(x_map)
#         assert_allclose(scipy_result.x, x_map, atol=1e-4, rtol=1e-3)
