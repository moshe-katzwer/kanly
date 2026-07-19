import pandas as pd
import numpy as np
from kanly.api import lm, elastic_net, nlls_en
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(0)
n = 500
x = np.random.rand(n)
x - x.mean()
y = 1.3 + .5 * x + 1.6 * np.random.randn(n)
y = y - y.mean()
g = np.random.randint(0, 35, n)
# y += (g / 4) - (g / 4).mean()
data = {'x': x, 'y': y, 'w': 10 * np.ones(n), 'g': g}

alpha = 10.5
b0 = 0
l1_ratio = 0
# fit = elastic_net('y~x', data, alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, normalize=False,
#                   regularize_to_values={'x': b0})
#
# print(fit)

groups = None  # 'g'

fit0 = lm('y~x-1', data, ridge_kwds={'alpha': alpha, 'normalize': False}, cov_type='hc1',
          # 'cluster', cov_kwds={ 'groups': groups}
          )

fit1 = lm('y~x-1', data, ridge_kwds={'alpha': alpha, 'normalize': False}, cov_type='bootstrap',
          cov_kwds={'n_samples': 2_000, 'groups': groups})
print(fit1)

fit2 = nlls_en('[y] ~ {b}*[x]', data, alpha=alpha, l1_ratio=l1_ratio, regularize_to_values={'b': b0},
               debug=False, cov_type='bootstrap', cov_kwds={'n_samples': 2_000, 'groups': groups},
               bounds={'b': [0, np.inf]})
print(fit2['fit'])

plt.hist(np.array(fit1.bootstrapped_params.flatten()), alpha=.5, color='r', density=True, bins=35)
plt.axvline(fit1.bootstrapped_params.flatten().mean(), color='r', ls='--')
plt.axvline(fit1['x'], color='r')

plt.hist(np.array(fit2['bootstrap_result']['result']).flatten(), alpha=.5, color='b', density=True, bins=35)
plt.axvline(fit2['fit']['b'], color='b')
plt.axvline(np.mean(fit2['bootstrap_result']['result']), color='b', ls=':')

xx = np.linspace(fit0['x'] - 3 * fit0.bse['x'], fit0['x'] + 3 * fit0.bse['x'])
plt.plot(xx, norm.pdf(xx, fit0['x'], fit0.bse['x']), lw=2, color='k')

dff = pd.DataFrame({
    'lm': fit1.bootstrapped_params.flatten(),
    'en': np.array(fit2['bootstrap_result']['result']).flatten()
})
print(dff.describe())

plt.show()
