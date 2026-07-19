import numpy as np
import pandas as pd

from kanly.api import lm

n = 1_000
np.random.seed(0)
df = pd.DataFrame({
    'x': np.random.randn(n),
    'grp': np.random.randint(0, 2, n)
})
df['y'] = 1.2 - 0.3 * df['x'] + .2 * (np.exp(-df['x']) * np.random.randn(n) * (1 + df['grp']))

fit = lm('y ~ x + C(grp)', df, cov_type='bootstrap', cov_kwds={'n_samples': 250, 'method': 'bayesian'})
print(fit)

"""
==============================================================================
Linear Model Results
==============================================================================

Dep. Variable:   y

Date:                    Oct 11, 2022    No. Obs.                         1000
Time:                        14:07:03    Df Residuals:                     997
Model Elapsed:                 0.01 s    Df Model:                           2
Fit Elapsed:                   0.00 s    R-squared:                      0.114
Cov Elapsed:                   0.72 s    Adj. R-squared:                 0.112
Method:                           OLS    F-statistic:                    26.00
Weights:                            -    Prob (F-statistic):             <.001
Intercept:                       True    Log-Likelihood:            -1186.0204
Implicit Intercept:             False    AIC:                          2378.04
Covariance Type:       BOOTSTRAP(250)    BIC:                          2392.76

===================================================================
               coef        std err      t   p>|t| [0.025,    0.975]
-------------------------------------------------------------------
Intercept     1.196  ****  0.02059  58.09  <0.001    1.155    1.236
x           -0.2878  ****  0.05457  -5.27  <0.001  -0.3949  -0.1807
C(grp)[1]  -0.02374        0.05037  -0.47   0.638  -0.1226   0.0751
===================================================================

formula:  y ~ x + C(grp)

Used t distribution with 997 df at test level 0.050.

                                          [kanly package by moshe, v=0.0.300]
"""
