import numpy as np
import pandas as pd

from kanly.api import sure

"""
A (seemingly) unrelated regression
"""

n = 500
np.random.seed(0)

df = pd.DataFrame({
    'x': np.random.randn(n),
    'user_id': range(n),
    'e': np.random.randn(n)
})
df['wts1'] = np.exp(.2*np.random.randn(n) + np.abs(df.x))
df['y1'] = 1.2 - 0.3 * df['x'] + .2 * np.random.randn(n) - .3 * df['e']
df['y2'] = 0.4 + .1 * df['x'] + .5 * np.random.randn(n) + .2 * df['e']

fit = sure(
    [
        {'formula': 'y1 ~ x $ wts1', 'data': df, 'specification_name': 'y1'},
        {'formula': 'y2 ~ x', 'data': df, 'specification_name': 'y2'},
    ],
    cov_type='cluster', cov_kwds={'groups': 'user_id'}
)

print(fit)

"""
==========================================================================
Linear Model Results
==========================================================================

Dep. Variable:   0) y1
                 1) y2

Date:                  Oct 12, 2022    No. Obs.                       1000
Time:                      16:29:38    Df Residuals:                   996
Model Elapsed:               0.06 s    Df Model:                         3
Fit Elapsed:                 0.01 s    R-squared:                    0.649
Cov Elapsed:                 0.00 s    Adj. R-squared:               0.648
Method:                  WLS (SURE)    F-statistic:                 175.49
Weights:                  <unknown>    Prob (F-statistic):           <.001
Intercept:                     True    Log-Likelihood:           -633.4691
Implicit Intercept:           False    AIC:                        1274.94
Covariance Type:            CLUSTER    BIC:                        1294.57

====================================================================
                coef        std err      t   p>|t| [0.025,    0.975]
--------------------------------------------------------------------
0_Intercept    1.218  ****   0.0208  58.57  <0.001    1.177    1.259
0_x          -0.3121  ****  0.01878 -16.62  <0.001   -0.349  -0.2752
1_Intercept   0.3723  ****  0.02311  16.11  <0.001   0.3269   0.4177
1_x           0.1005  ****  0.02163   4.65  <0.001  0.05803    0.143
====================================================================

formulas:
   [0] y1 ~ x $ wts1
   [1] y2 ~ x

Used t distribution with 499 df at test level 0.050.
Variance clustered on 'user_id'

                                                        [kanly v=0.0.302]
"""