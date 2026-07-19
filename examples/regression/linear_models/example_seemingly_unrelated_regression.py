import numpy as np
import pandas as pd

from kanly.api import sure

n = 100
np.random.seed(0)
df = pd.DataFrame({
    'x': np.random.randn(n),
    'e1': np.random.randn(n),
    'e2': np.random.randn(n),
    'user': np.arange(n),
})
df['wts'] = np.exp(df['x'] / 2)
df['y1'] = 1.2 - 0.3 * df['x'] + .2 * df['e1'] + .5 * df['e2']
df['y2'] = -.12 + .25 * df['x'] + .7 * df['e1'] + .3 * df['e2']

fit = sure(
    [{'formula': 'y1 ~ x $ wts'},
     {'formula': 'y2 ~ x'}]
    ,
    data=df,
    cov_type='cluster', cov_kwds={'groups': 'user'}
)
print(fit)

"""
======================================================================
Regression Results
======================================================================

Dep. Variable:   0) y1
                 1) y2

Date:               Feb 25, 2022   No. Obs.                     200
Time:                   12:36:16   Df Residuals:                196
Model Time:               0.11 s   Df Model:                      3
Fit Time:                 0.00 s   Covariance Type:         CLUSTER
Method:               OLS (SURE)   R-squared:                 0.514
Weights:                       -   Adj. R-squared:            0.506
                                   Intercept:                  True

======================================================================
                 coef        std err      t   p>|t|  [0.0250,  0.9750]
----------------------------------------------------------------------
0_Intercept     1.183  ****  0.05264  22.48   <.001     1.079    1.288
0_x           -0.2411  ****   0.0439  -5.49   <.001   -0.3282   -0.154
1_Intercept  -0.08645        0.07907  -1.09   0.277   -0.2433  0.07044
1_x            0.3519  ****  0.06947   5.07   <.001     0.214   0.4897
======================================================================
Used t distribution with 99 df at test level 0.050.
Variance clustered on 'user'

0) y1 ~ x
1) y2 ~ x
                                   [kanly package by moshe, v=0.0.90]
"""
