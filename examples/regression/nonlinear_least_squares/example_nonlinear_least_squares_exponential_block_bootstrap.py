import numpy as np
import pandas as pd

from kanly.api import nlls

n = 250
np.random.seed(0)
df = pd.DataFrame({
    'x': np.random.randn(n),
    'w': np.exp(np.random.randn(n)),
    'grp': np.random.randint(0, 10, n),
})
df['y'] = 1 + 3 * np.exp(-.5 * df.x) + .4 * np.random.randn(n)


fit = nlls('[y] ~ {Intercept} + {beta} * np.exp({gamma} * [x]) $ [w]',
           df,
           max_iter=100,
           cov_type='bootstrap', cov_kwds={'groups': 'grp', 'n_samples': 100},
           debug=True, specification_name='Example Exponential')

print(fit)

"""
==========================================================================
Nonlinear Least Squares Results
Example Exponential
==========================================================================

Dep. Variable: y

Date:                  Mar 29, 2023    R-squared:                   0.9573
Time:                      20:34:52    Adj. R-squared:              0.9569
Weights:                          w    Model Time:                   0.00s
Nobs:                           250    Fit Time:                     0.63s
Df Residuals:                   247    Cov Time:                     1.01s
Df Model:                         3    Iterations:                       7
Cost:                    2.7043e+01    Converged:                     True
Penalty:                 0.0000e+00    Status:                           2
Objective:               2.7043e+01    Covariance Type:          BOOTSTRAP
Optimality:                4.95e-06    Active Constraints:               0

==================================================================
              coef        std err      t   p>|t| [0.025,    0.975]
------------------------------------------------------------------
Intercept    1.109  **     0.3352   3.31   0.009   0.3506    1.867
beta         2.914  ****   0.3456   8.43  <0.001    2.132    3.695
gamma      -0.5172  ****  0.04219 -12.26  <0.001  -0.6126  -0.4217
==================================================================

formula:  [y] ~ {Intercept} + {beta} * np.exp({gamma} * [x]) $ [w]

Used t distribution with 9 df at test level 0.0500.
Did 100 Bayesian bootstrap repetitions, alpha=1.000, blocked on 'grp'.

message: |dF| < ftol * max(1, |F|)

                                                        [kanly v=0.0.384]
"""
