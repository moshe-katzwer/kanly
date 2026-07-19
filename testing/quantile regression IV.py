import pandas as pd

from kanly.api import QR, LM, qr
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# n = 15_000
#
#
# # def sim_data(seed, dx):
#     np.random.seed(seed)
#     Z = np.random.randn(n, 3)
#     e = np.random.randn(n) * np.exp(.2 * np.abs(Z.sum(axis=1)))
#     X = np.array([[4 + dx]]) + Z.dot(np.random.randn(3, 1)) + 1.2 * e.reshape((n, 1)).dot(
#         np.array([[1.5]])) + .3 * np.random.randn(n, 1)
#     y = (50 + X.dot([2.5]) + 5.6 * e + 15 + 3 * (np.exp(np.random.randn(n)) - np.exp(.5)).mean() * (
#                 1 + .3 * np.abs(np.sum(X, axis=1))))
#
#     return X, Z, y
#

# tau = .9
# dx = .0005
# results = []
# for t in tqdm_kanly([6664]):
#     X, Z, y1 = sim_data(t, 0)
#     _, _, y2 = sim_data(t, dx)
#     results.append((np.quantile(y2, tau) - np.quantile(y1, tau)) / dx)
# # plt.hist(results)
# plt.title(np.mean(results))
# plt.show()
# print((np.mean(results)))
#
# print(QR(y1, np.hstack((np.ones((n, 1)), X)), tau))
# print(QR(y1, np.hstack((np.ones((n, 1)), X)), tau, instruments=np.hstack((np.ones((n, 1)), Z))))
#
# #
# df = pd.DataFrame()
# for j in range(2):
#     df[f'x{j}'] = X[:,j]
# for j in range(3):
#     df[f'z{j}'] = Z[:,j]
#
# y = X.dot([1., 2.]) + 5.6 * e + 15 + 3 * np.exp(np.random.randn(n) - 0.5) * (1 + .3 * np.abs(np.sum(X, axis=1)))
#
# df['y'] = y
#
# X = np.hstack((np.ones((n, 1)), X))
# Z = np.hstack((np.ones((n, 1)), Z))
# tau = .99
#
# # fit = QR(y, X, tau=tau)
# # print(fit)
#
# Pi = np.linalg.pinv(Z.T.dot(Z)).dot(Z.T.dot(X))
# X_pred = Z.dot(Pi)
#
# # fit = QR(y, X_pred, tau=tau)
# # print(fit)
#
# X_ri = np.hstack((X_pred, X[:, 1:] - X_pred[:, 1:]))
# X_ri2 = np.hstack((X, X[:, 1:] - X_pred[:, 1:]))
#
# fit = LM(y, X_pred)
# print(fit)
# fit = LM(y, X_ri)
# print(fit)
#
# fit = QR(y, X_ri, tau=tau, line_search=True, debug=True, smoothing_k=.0001)
# print(fit)
#
# fit = QR(y, X_ri2, tau=tau, line_search=True, debug=True, smoothing_k=.0001)
# print(fit)
#
# fit = QR(y, X, tau=tau, line_search=True, debug=True, smoothing_k=.0001)
# print(fit)
#
# fit = QR(y, X, tau=tau, line_search=True, debug=True, smoothing_k=.0001, instruments=Z)
# print(fit)

def sim_data(seed, dx, tau, n):
    quants = []

    np.random.seed(seed)
    Z = np.random.randn(n, 3)
    e = np.random.randn(n) * np.exp(.2 * np.abs(Z.sum(axis=1)))

    for _dx in [-dx/2, dx/2]:
        np.random.seed(seed)
        X = _dx + 4 + Z.dot(np.random.randn(3, 1)) + 1.2 * e.reshape((n, 1)).dot(
            np.array([[1.5]])) + .3 * np.random.randn(n, 1)
        y = (-19.5 + X.dot([2.5]) + 5.6 * e + 15 + 3 * (np.exp(np.random.randn(n)) - np.exp(.5)).mean() * (
                1 + .3 * np.abs(np.sum(X, axis=1))))
        y /= 50
        # y = np.exp(y) ** .198593
        y = y + y ** 2
        quants.append(np.quantile(y, tau))

    effect = (quants[1] - quants[0]) / dx

    df = pd.DataFrame()
    for j in range(X.shape[1]):
        df[f'x{j}'] = X[:, j]
    for j in range(Z.shape[1]):
        df[f'z{j}'] = Z[:, j]
    df['y'] = y

    fit = qr('y ~ x0', df, tau=tau)
    effect_qr = fit.params['x0']

    fit = qr('y ~ x0 | ' + ' + '.join([f'z{j}' for j in range(Z.shape[1])]),
             df, tau=tau, residual_inclusion=True)
    effect_qr_iv_ri = fit.params['x0']

    fit = qr('y ~ x0 | ' + ' + '.join([f'z{j}' for j in range(Z.shape[1])]),
             df, tau=tau, residual_inclusion=False)
    effect_qr_iv = fit.params['x0']

    # plt.hist(y, bins=50, cumulative=True)
    # plt.show()

    # _Z = np.hstack((np.ones((n, 1)), Z))
    # _X = np.hstack((np.ones((n, 1)), X))
    # _Xhat = _Z.dot(np.linalg.pinv(_Z.T.dot(_Z)).dot(_Z.T.dot(_X)))
    # _X2 = _Xhat #np.hstack((_Xhat, (_X[:, 1] - _Xhat[:, 1]).reshape((-1, 1)), ((_X[:, 1] - _Xhat[:, 1]) ** 2).reshape((-1, 1))))
    # fit = QR(y, _X2, tau)
    #
    # _Z = np.hstack((np.ones((n, 1)), Z))
    # fit_1st = QR(X[:, 0], _Z, .5)
    # V = np.vstack([fit_1st.resid ** o for o in range(1, 2)]).T
    # _X2 = np.hstack((np.ones((n, 1)), fit_1st.fittedvalues.reshape((n,1)), V))
    # fit = QR(y, _X2, tau)
    #effect_qr_iv_manual = fit.params['<x1>']

    return effect, effect_qr, effect_qr_iv, effect_qr_iv_ri#, effect_qr_iv_manual


n = 5_000
tau = .85
dx = 5e-3

results = []
for s in tqdm(range(50)):
    results.append(sim_data(s, dx, tau, n))

results = pd.DataFrame(results, columns=['true', 'qr', 'iv', 'iv-ri'])#, 'manual'])#, 'manual'])
print(results.mean(axis=0))
print(results.std(axis=0))

for j in results.columns:
    plt.hist(results[j], label=j, alpha=.4, density=True, bins=20)
plt.axvline(results.true.mean(), color='k', lw=2)
plt.legend(loc='best')
plt.show()
