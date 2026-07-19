from kanly.api import lm, DataModel
from kanly.stats.distributions.nopython_logpdf import logpdf_norm
import numpy as np
from scipy.stats import t
from numba import njit


np.random.seed(0)
n = 500
x = 1.56 * np.random.randn(n)
z = 1.56 * np.random.randn(n) + .5 * x
y = 3 + 10 * x - 1 * z + t.rvs(df=3, size=n)
wts = .01 + np.random.rand(n)
data = {'x': x, 'y': y, 'wts': wts, 'g': np.random.randint(0, 4, n), 'z': z}

from kanly.api import bayes_lm_model

model = bayes_lm_model('y ~ x + z', data,
                       priors={'': '10+{z}; return logpdf_norm(np.abs({x})+3*np.abs({z}), 0, .05)'},
                       nopython=True
                       )

print(-(1 + 3 * 2.2) ** 2 / 2 - 0.5 * np.log(2 * np.pi))
print(model.log_pdf_prior([0, 1, 2.2, 1.]))

print(model.amha([0, 0, 0, 1], n_samples=10_000))

model = bayes_lm_model('y ~ x + z', data,
                       priors={'': lambda x: logpdf_norm(np.abs(x[1]) + 3 * np.abs(x[2]), 0, .05)},
                       nopython=True
                       )

print(-(1 + 3 * 2.2) ** 2 / 2 - 0.5 * np.log(2 * np.pi))
print(model.log_pdf_prior([0, 1, 2.2, 1.]))

print(model.amha([0, 0, 0, 1], n_samples=10_000))

model = bayes_lm_model('y ~ x + z', data,
                       priors={('x', 'z'): lambda x, z: logpdf_norm(np.abs(x) + 3 * np.abs(z), 0, .05)},
                       nopython=True
                       )

print(-(1 + 3 * 2.2) ** 2 / 2 - 0.5 * np.log(2 * np.pi))
print(model.log_pdf_prior([0, 1, 2.2, 1.]))

print(model.amha([0, 0, 0, 1], n_samples=10_000))
