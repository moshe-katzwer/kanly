import numpy as np
import pandas as pd

from kanly.api import nlls

n = 250
np.random.seed(0)
df = pd.DataFrame({
    'x': np.random.randn(n),
    'w': np.exp(np.random.randn(n)),
})
df['y'] = 1 + 3 * np.exp(-.5 * df.x) + .4 * np.random.randn(n)
df.loc[3, 'y'] = np.nan

fit = nlls('[y] ~ {Intercept} + {beta} * exp({gamma} * [x]) $ [w]',
           df,
           max_iter=100,
           cov_type='bootstrap', cov_kwds={'n_samples': 5_000, 'max_processes': 5},
           debug=True, specification_name='Example Exponential',
           # dense_threshold_mb=0
           )

print(fit)

"""
══════════════════════════════════════════════════════════════════════════
Nonlinear Least Squares Results
Example Exponential
══════════════════════════════════════════════════════════════════════════

Dep. Variable: y

Date:                  Dec 27, 2024    R-squared:                   0.9500
Time:                      12:58:33    Adj. R-squared:              0.9496
Weights:                          w    Model Time:                   0.00s
Nobs:                           249    Fit Time:                     0.17s
Df Residuals:                   246    Cov Time:                     3.95s
Df Model:                         3    Iterations:                       7
Cost:                    3.2170e+01    Converged:                     True
Scale:                   2.6155e-01    Status:                           1
LLF:                    -1.9499e+02    Covariance Type:          BOOTSTRAP
Penalty:                 0.0000e+00    Active Constraints:               0
Objective:               3.2170e+01    Method:                          TR
Optimality:                3.05e-06                                       

══════════════════════════════════════════════════════════════════
              coef        std err      t   p>|t| [0.025,    0.975]
──────────────────────────────────────────────────────────────────
Intercept   0.7854  ****   0.1923   4.08  <0.001   0.4066    1.164
beta         3.131  ****   0.2004  15.63  <0.001    2.737    3.526
gamma      -0.4918  ****  0.02419 -20.33  <0.001  -0.5394  -0.4441
══════════════════════════════════════════════════════════════════

formula:  [y] ~ {Intercept} + {beta} * exp({gamma} * [x]) $ [w]

Used t distribution with 246 df at test level 0.0500.
Did 5000 Bayesian bootstrap repetitions, alpha=1.000.

message: Converged: |dF| < ftol * max(1, |F|)
"""
