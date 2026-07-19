# import pandas as pd
# import numpy as np
# from kanly.api import qr, lm
# from statsmodels.formula.api import quantreg
#
# import matplotlib.pyplot as plt
#
# n = 200_000
# np.random.seed(195)
#
# df = pd.DataFrame()
# df['x'] = np.random.randn(n)
# df['x'] -= df['x'].mean()
# df['z'] = (
#     #-np.exp(np.random.randn(n)) #n
#     (np.random.rand(n) < .3) * 1.5
# )
# # df['z'] = lm('z ~ x', df).resid
# df['e'] = np.random.randn(n)
# df['y'] = 1.7 ** (-3.5 + .4 * df.x + 1 * df.z + .25 * df.e)
# #df['y'] = -1 + 1.2 * df.x + 2 * df.z + .3 * np.random.randn(n) * (.3 + np.exp(df.x))
#
# tau = .9
#
# dx = 1e-5
# df['y2'] = 1.7 ** (-3.5 + .4 * (df.x + dx) + 1 * df.z + .25 * df.e)
# print("TRUTH = ", (np.quantile(df.y2, tau) - np.quantile(df.y, tau)) / dx)
#
# f, ax = plt.subplots(ncols=2)
# ax[0].scatter(df.x[:10_000], df.y[:10_000], alpha=.1)
# ax[1].hist(df.y[:10_000], bins=20)
# plt.show()
#
# print(df.corr())
#
# print(lm('y ~ x', df))
# print(lm('y ~ x + z', df))
#
# k = 1e-5
# print(fit := qr('y ~ x', df, tau, smoothing_k=k))
# # print(np.mean((tau - (fit.resid < 0)) * fit.resid))
# print(fit2:=qr('y ~ x + z', df, tau, smoothing_k=k))
#
# f, ax = plt.subplots(nrows=3, sharex=False)
# ax[0].scatter(df.x[:10000], fit.weights[:10000], alpha=.1)
# ax[1].scatter(df.x[:10000], fit2.weights[:10000], alpha=.1)
# ax[2].scatter(fit.weights[:10000], fit2.weights[:10000], alpha=.1)
# ax[2].set_title(str(np.corrcoef(fit.weights[:10000], fit2.weights[:10000])[0,1]))
# plt.show()
#
# # print((fit_sm := quantreg('y ~ x', df).fit(tau)).summary())
# # print(np.mean((tau - (fit_sm.resid < 0)) * fit_sm.resid))
