# from kanly.api import lm, glm, nlls, blm, bglm, bnlls
# from numpy.testing import assert_allclose
#
# import pandas as pd
# import numpy as np
#
# n = 10_000
# np.random.seed(0)
# df = pd.DataFrame({
#     'x': np.random.randn(n),
# })
# p = 1.0 / (1.0 + np.exp(-(-.2 + .16 * df['x'] - .048 * df['x'] ** 2)))
# df['y'] = (np.random.rand(n) < p).astype(int)
#
# #### #
# # LM #
# # ## #
#
# fit = lm('y ~ x', df)
# fitb = blm('y ~ x', df, debug=True,
#            mcmc_start_jitter=.25,
#            mcmc_options=dict(n_burnin=5000, n_samples=40_000, thinning=2, n_chains=8))
# print(fit)
# print(fitb['fit_mcmc_data_model'])
# assert_allclose(fit.params, fitb['fit_mcmc_data_model'].mean_params[:-1], rtol=.0025)
# assert_allclose(fit.bse, fitb['fit_mcmc_data_model'].bse[:-1], rtol=.01)
#
# # #### #
# # NLLS #
# # #### #
#
# # linear
# fit = nlls('[y] ~ {a}+{b}*[x]', df)
# fitb = bnlls('[y] ~ {a}+{b}*[x]', df, mcmc_options=dict(n_samples=25_000, thinning=2, n_chains=8))
# print(fit)
# print(fitb['fit_mcmc_data_model'])
# assert_allclose(fit.params, fitb['fit_mcmc_data_model'].mean_params[:-1], rtol=.0025)
# assert_allclose(fit.bse, fitb['fit_mcmc_data_model'].bse[:-1], rtol=.01)
#
# # nonlinear
# # linear
# fit = nlls('[y] ~ {a}+{b}*[x**3]', df)
# fitb = bnlls('[y] ~ {a}+{b}*[x**3]', df, mcmc_options=dict(n_samples=25_000, thinning=2, n_chains=8))
# print(fit)
# print(fitb['fit_mcmc_data_model'])
# assert_allclose(fit.params, fitb['fit_mcmc_data_model'].mean_params[:-1], rtol=.0025)
# assert_allclose(fit.bse, fitb['fit_mcmc_data_model'].bse[:-1], rtol=.01)
#
# # ### #
# # GLM #
# # ### #
#
# # binomial
# fit = glm('y ~ x', df, family='binomial')
# print(fit)
#
# fitb = bglm('y ~ x', df, family='binomial', debug=True,
#             mcmc_start_jitter=.25,
#             mcmc_options=dict(n_samples=30_000, thinning=2, n_chains=8))
# print(fitb['fit_mcmc_data_model'])
# assert_allclose(fit.params, fitb['fit_mcmc_data_model'].mean_params, rtol=.0025)
# assert_allclose(fit.bse, fitb['fit_mcmc_data_model'].bse, rtol=.01)
#
# # gaussian
# fit = glm('y ~ x', df)
# print(fit)
#
# fitb = bglm('y ~ x', df, debug=True,
#             mcmc_start_jitter=.25,
#             mcmc_options=dict(n_samples=30_000, thinning=2, n_chains=8))
# print(fitb['fit_mcmc_data_model'])
# assert_allclose(fit.params, fitb['fit_mcmc_data_model'].mean_params[:-1], rtol=.0025)
# assert_allclose(fit.bse, fitb['fit_mcmc_data_model'].bse[:-1], rtol=.01)
