import numpy as np
import pandas as pd

from kanly.api import lm

n = 100
np.random.seed(0)
df = pd.DataFrame({
    'x': np.random.randn(n),
    'grp': np.random.randint(0, 2, n),
    'grp2': np.random.randint(0, 4, n).astype(str),
})
df.x -= df.x.mean()

df['y'] = 1.2 - 0.3 * df['x'] + .2 * np.random.randn(n)

fit1 = lm('y ~ x', df, absorb=('grp', 'grp2'), cov_type='cluster', cov_kwds={'groups': 'grp2'})
print(fit1)

'''
==========================================================================
Linear Model Results
==========================================================================

Dep. Variable:   y

Date:                  Jan 27, 2023    No. Obs.                        100
Time:                      13:34:57    Df Residuals:                    91
Model Elapsed:               0.01 s    Df Model:                         8
Fit Elapsed:                 0.00 s    R-squared:                   0.7246
Cov Elapsed:                 0.00 s    Adj. R-squared:              0.7004
Method:                         OLS    F-statistic:                  29.93
Weights:                          -    Prob (F-statistic):           <.001
Intercept:                    False    Log-Likelihood:             26.3998
Implicit Intercept:            True    AIC:                         -34.80
Covariance Type:            CLUSTER    BIC:                         -11.35
                                       Cond. No.:                 1.00e+00

=========================================================
      coef        std err      t   p>|t| [0.025,   0.975]
---------------------------------------------------------
x  -0.2904  ***   0.01456 -19.94  <0.001  -0.3367  -0.244
=========================================================

formula:  y ~ x

Absorbed: 'grp:grp2', num=8
	Within  R² = 0.6119
	Between R² = 0.1127

Used t distribution with 3 df at test level 0.0500.
Variance clustered on 'grp2'

(*) F-statistic computed assuming spherical errors.
    For robust F-statistic, use `fit.F_test()`

                                                        [kanly v=0.0.360]
'''