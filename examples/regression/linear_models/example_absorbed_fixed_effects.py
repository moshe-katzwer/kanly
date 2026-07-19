import numpy as np
import pandas as pd

from kanly.api import lm

n = 100
np.random.seed(0)
df = pd.DataFrame({
    'x': np.random.randn(n),
    'grp': np.random.randint(0, 4, n),
    'grp2': np.random.randint(0, 2, n),
})
df.x -= df.x.mean()

df['y'] = 1.2 - 0.3 * df['x'] + .2 * np.random.randn(n)

fit1 = lm('y ~ x', df, absorb=('grp', 'grp2'))
print(fit1)

"""
=========================================================================
Regression Results
=========================================================================

Dep. Variable:   y

Date:                  Aug 25, 2022   No. Obs.                        100
Time:                      17:21:14   Df Residuals:                    91
Model Time:                  0.05 s   Df Model:                         8
Fit Time:                    0.00 s   Covariance Type:          OLS_SMALL
Method:                         OLS   R-squared:                    0.728
Weights:                          -   Adj. R-squared:               0.704
                                      Intercept:                    False
                                      Implicit Intercept:            True

=========================================================================
     coef        std err       t   p>|t|  [0.0250,  0.9750]
-------------------------------------------------------------------------
x  -0.301  ****  0.01975  -15.25   <.001   -0.3403  -0.2618
=========================================================================
Absorbed: grp:grp2, num=8
    Within R**2=0.719, Adj. Within R**2=0.716

Used t distribution with 91 df at test level 0.050.

0) y ~ x
                                     [kanly package by moshe, v=0.0.122]
"""

fit2 = lm('y ~ x + C(grp):C(grp2) - 1', df)
print(fit2)

"""
===============================================================================
Regression Results
===============================================================================

Dep. Variable:   y

Date:                  Aug 25, 2022   No. Obs.                        100
Time:                      17:21:14   Df Residuals:                    91
Model Time:                  0.02 s   Df Model:                         8
Fit Time:                    0.00 s   Covariance Type:          OLS_SMALL
Method:                         OLS   R-squared:                    0.728
Weights:                          -   Adj. R-squared:               0.704
                                      Intercept:                    False
                                      Implicit Intercept:            True

===============================================================================
                         coef        std err       t   p>|t|  [0.0250,  0.9750]
-------------------------------------------------------------------------------
x                      -0.301  ****  0.01975  -15.25   <.001   -0.3403  -0.2618
C(grp2)[0]:C(grp2)[0]   1.217  ****   0.0733   16.61   <.001     1.072    1.363
C(grp2)[1]:C(grp2)[0]   1.222  ****  0.06133   19.92   <.001       1.1    1.343
C(grp2)[2]:C(grp2)[0]   1.139  ****  0.05603   20.34   <.001     1.028    1.251
C(grp2)[3]:C(grp2)[0]   1.124  ****  0.04854   23.16   <.001     1.028    1.221
C(grp2)[0]:C(grp2)[1]   1.228  ****  0.05179   23.72   <.001     1.126    1.331
C(grp2)[1]:C(grp2)[1]   1.163  ****  0.04885   23.82   <.001     1.066     1.26
C(grp2)[2]:C(grp2)[1]   1.116  ****  0.06887   16.21   <.001    0.9793    1.253
C(grp2)[3]:C(grp2)[1]   1.147  ****  0.04699   24.42   <.001     1.054    1.241
===============================================================================
Used t distribution with 91 df at test level 0.050.

0) y ~ x + C(grp):C(grp2) - 1
                                           [kanly package by moshe, v=0.0.122]
"""
