# import numpy as np
# import pandas as pd
#
# from kanly.api import blm, elastic_net
# from kanly.bayes.bayesian_regression_model import BayesianLinearModel
# from kanly.bayes.mcmc.adaptive_metropolis.adaptive_metropolis_mcmc import amha
#
# n = 33
# np.random.seed(0)
# x = np.random.randn(n)
# y = 3 + 10 * x + np.random.randn(n) * 4.5
# data = {'x': x, 'y': y, 'wts': np.random.rand(n) * 20}
#
# # Fit a model where our prior is that
# #   sigma2 has pdf 1/sigma2
# #   beta|sigma2 ~ N([0,5], sigma2 * diag([0,10])^-1)
# fit = blm('y ~ x $ wts', data, mu0=[0, 5], Lambda0=[0, 10], a0=0, b0=0, specification_name='example')
# print(fit)
#
# # print(fit.fittedvalues)
# # print(fit.model.predict(params=fit.params))
# # print(fit.model.predict(params=fit.params, data=data))
# # print(fit.model.predict(params=fit.params, data=pd.concat([pd.DataFrame(z) for z in [data] * 2])))
# # print(fit.equitail_credible_interval(.01))
# #
# # # # compare with ridge
# # # fit_en = elastic_net('y ~ x $ wts', data, normalize=False, l1_ratio=0,
# # #                      alpha={'x': 10 / np.sum(data['wts'])},
# # #                      regularize_to_values={'x': 5},
# # #                      apply_scaling=False)
# # # print(fit_en)
# # #
# # #
# # compare with mcmc
# bmodel = BayesianLinearModel.build_model_from_formula('y ~ x $ wts', data, priors={'': fit.prior}) \
#     .amha([0, 0, 1], n_chains=8, n_samples=200_000, max_subchain_draws=100000)
# print(bmodel)
#
# print(amha(fit.model._log_pdf_posterior, [0, 0, 1], n_chains=8, n_samples=2_000, max_subchain_draws=100_000, debug=True))
#
#
