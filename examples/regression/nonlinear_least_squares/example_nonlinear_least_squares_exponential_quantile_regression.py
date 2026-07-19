import numpy as np
import pandas as pd

from kanly.api import nlls
from kanly.regression.nonlinear_least_squares.function_callables.loss_functions import QuantileHuberLoss

n = 500
np.random.seed(0)
df = pd.DataFrame({
    'x': np.random.randn(n),
    'w': np.exp(np.random.randn(n))
})
df['y'] = 1 + 3 * np.exp(-.5 * df.x) + .4 * np.random.randn(n) * (1 + np.abs(df.x) / 3)

fit = nlls('[y] ~ {Intercept} + {beta} * np.exp({gamma} * [x])',
           df,
           root_loss_function=QuantileHuberLoss(tau=.15, k=.001),
           debug=True, specification_name='Example NLLS Quantile Regression')

print(fit)

"""
==========================================================================
Nonlinear Least Squares Results
Example Exponential
==========================================================================

Dep. Variable: y

Date:                  Oct 07, 2022    Adj. R-squared:              0.9129
Time:                      12:28:44    Model Time:                   0.00s
Weights:                       None    Fit Time:                     0.55s
Nobs:                           500    Cov Time:                     0.04s
Df Residuals:                   497    Iterations:                      20
Df Model:                         3    Converged:                     True
Cost:                    4.2373e+01    Status:                           4
Optimality:                1.66e+00    Covariance Type:                HC1
R-squared:                   0.9133    Active Constraints:               0

====================================================================
              coef         std err       t   p>|t| [0.025,    0.975]
--------------------------------------------------------------------
Intercept   0.4495  ****   0.03059   14.69  <0.001   0.3894   0.5096
beta         3.324  ****   0.03318  100.17  <0.001    3.259    3.389
gamma      -0.4645  ****  0.002868 -161.94  <0.001  -0.4701  -0.4588
====================================================================

formula:  [y] ~ {Intercept} + {beta} * np.exp({gamma} * [x])

Used t distribution with 497 df at test level 0.050.

Loss Function: QuantileHuberLoss(tau=0.3000, k=1.00e-03)

message: |dx| < xtol * max(1, |x|)

                                [kanly package by moshe, v=0.0.297]
"""

import matplotlib.pyplot as plt
plt.scatter(df.x, df.y)
plt.scatter(df.x, fit.fittedvalues)
plt.show()