import numpy as np
import pandas as pd

from kanly.api import nlls

n = 10_000
np.random.seed(0)
df = pd.DataFrame({'x': np.random.randn(n)})
df['p'] = 1 / (1 + np.exp(-(.4 + .9 * df.x)))
df['y'] = (np.random.rand(n) < df['p']).astype(float)

fit = nlls('[y] ~ 1.0 / (1.0 + exp({alpha} + {beta} * [x]))',
           df,
           debug=True,
           jac_method='analytic',
           do_analytic_jac_jit=True,
           x_scale='jac',
           # cov_type='bootstrap',
           # start_params={'beta': -.5},
           # regularize_to_values={'beta': -.3},
           # l2_penalties={'beta': .1},
           # bounds={'alpha': [-.2, 1]},
           compute_cov=True,
           specification_name='logistic regression'
           )

print(fit)

"""
==========================================================================
Nonlinear Least Squares Results
logistic regression
==========================================================================

Dep. Variable: y

Date:                  Nov 07, 2023    R-squared:                   0.1395
Time:                      14:49:11    Adj. R-squared:              0.1394
Weights:                       None    Model Time:                   0.00s
Nobs:                         10000    Fit Time:                     0.21s
Df Residuals:                  9998    Cov Time:                     0.01s
Df Model:                         2    Iterations:                       4
Cost:                    1.0470e+03    Converged:                     True
Scale:                   2.0944e-01    Status:                           2
LLF:                    -6.3719e+03    Covariance Type:                HC1
Penalty:                 0.0000e+00    Active Constraints:               0
Objective:               1.0470e+03    Method:                          TR
Optimality:                2.10e-04                                       

==============================================================
          coef        std err      t   p>|t| [0.025,    0.975]
--------------------------------------------------------------
alpha  -0.3955  ****  0.02238 -17.67  <0.001  -0.4394  -0.3516
beta   -0.8832  ****   0.0264 -33.45  <0.001  -0.9349  -0.8314
==============================================================

formula:  [y] ~ 1.0 / (1.0 + exp({alpha} + {beta} * [x]))

Used t distribution with 9998 df at test level 0.0500.

message: Converged: |dF| < ftol * max(1, |F|)

                                                        [kanly v=0.0.514]
"""
