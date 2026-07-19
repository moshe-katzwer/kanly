import numpy as np
import pandas as pd

from kanly.api import nlls

n = 500
np.random.seed(0)
df = pd.DataFrame({
    'x': np.random.randn(n),
    'w': np.exp(np.random.randn(n)),
    'grp': np.random.randint(0, 20, n),
})
df['y'] = 1 + 3 * np.exp(-.5 * df.x) + .4 * np.random.randn(n)

fit = nlls('[y] ~ {Intercept}  + {beta} * np.exp({gamma} * [x]) $ [w]',
           df,
           max_iter=100,
           cov_type='cluster', cov_kwds={'groups': 'grp'},
           debug=True, specification_name='Example Exponential')

print(fit)

"""
==============================================================================
Nonlinear Least Squares Results
Example Exponential
==============================================================================

Dep. Variable: y

Date:                    Oct 10, 2022    Adj. R-squared:                0.9498
Time:                        09:49:03    Model Time:                     0.00s
Weights:                            w    Fit Time:                       0.54s
Nobs:                             250    Cov Time:                       0.70s
Df Residuals:                     247    Iterations:                         7
Df Model:                           3    Converged:                       True
Cost:                      3.2172e+01    Status:                             1
Optimality:                  3.21e-06    Covariance Type:       BOOTSTRAP(100)
R-squared:                     0.9502    Active Constraints:                 0

=================================================================
             coef        std err      t   p>|t| [0.025,    0.975]
-----------------------------------------------------------------
Intercept  0.7879  *       0.346   2.28   0.024   0.1065    1.469
beta        3.129  ****   0.3425   9.14  <0.001    2.454    3.804
gamma      -0.492  ****  0.03831 -12.84  <0.001  -0.5675  -0.4166
=================================================================

formula:  [y] ~ {Intercept} + {beta} * np.exp({gamma} * [x]) $ [w]

Used t distribution with 247 df at test level 0.050.

message: |dF| < ftol * max(1, |F|)

                                          [kanly package by moshe, v=0.0.299]
"""
