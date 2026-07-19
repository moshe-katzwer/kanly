import pandas as pd
from scipy.stats import poisson
from kanly.api import build_data_model
import numpy as np


n = 5000
D = np.random.randint(0, 2, n)
pi = 1 / (1 + np.exp(-(.15 - .25 * D)))

ordered = np.random.rand(n) > pi

orders_ = poisson(np.exp(.25 + .1 * D[ordered])).rvs(np.count_nonzero(ordered))
orders = np.zeros(n)
orders[ordered] = orders_

truth = [.15, -.25, .25, .1]

model = build_data_model(
    data_code_block="""
    self.orders = `orders`;
    self.D = `D`;
    self.pos_idx = self.orders > 0
    self.orders_factorial = gammaln(self.orders+1)
    """,

    model_code_block="""
    pi = 1.0 / (1 + np.exp(-($pi0$ + $pi1$ * D)));
    lambda_ = np.exp($lambda0$ + $lambda1$ * D);

    llf = 0.0

    llf += np.log( pi + (1-pi) * np.exp(-lambda_))[~pos_idx].sum()
    llf += (np.log(1-pi) - lambda_ + orders * np.log(lambda_) - orders_factorial)[pos_idx].sum()

    return llf
    """,

    data=dict(D=D, orders=orders),

    nopython=True
)
model = model.to_bayesian_model()
res = model.map([0, 0, 0, 0], B0=100, maxiter=1000, ftol=1e-12, gtol=1e-6, momentum=.1)

res.x, res.optimization_result.ferr, res.optimization_result.iter
print(pd.DataFrame({
    'estimated map': res.x,
    'truth': truth,
}))

print(np.linalg.inv(res.optimization_result.hess))

"""
         estimated map  truth
pi0           0.158658   0.15
pi1          -0.260580  -0.25
lambda0       0.249129   0.25
lambda1       0.105144   0.10
"""

res = model.sample([0,0,0,0], n_samples=1000, do_parallel=False, n_chains=2)
print(res)
print(f'{res.map_params=}')