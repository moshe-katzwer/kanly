from kanly.api import ardl, ARDL
import pandas as pd
import numpy as np

np.random.seed(0)
n = 400
x = np.random.randn(n)
w = np.random.randn(n)
y = 10 + 1.2 * x - .16 * w + np.random.randn(n)
y[1:] += .5 * x[:-1]

fit_ARDL = ARDL(y, lags=1, exog=x.reshape((-1, 1)), fixed=w.reshape((-1, 1)), order=1, causal=False)
print(fit_ARDL)

df = pd.DataFrame(dict(y=y, x=x, w=w))
fit_ardl = ardl('y ~ x + w', df, lags=1, order={'x': 1}, causal=False)
print(fit_ardl)

"""
══════════════════════════════════════════════════════════════════════════
Linear Model Results
══════════════════════════════════════════════════════════════════════════

Dep. Variable:   <y>

Date:                  May 30, 2026    No. Obs.                        399
Time:                      09:35:12    Df Residuals:                   394
Model Elapsed:               0.00 s    Df Model:                         4
Fit Elapsed:                 0.00 s    R-squared:                   0.6398
Cov Elapsed:                 0.00 s    Adj. R-squared:              0.6361
Method:                         OLS    F-statistic:                 174.95
L2 Penalty:                    None    Prob (F-statistic):           <.001
Weights:                          -    Log-Likelihood:           -536.3755
Intercept:                     True    AIC:                        1082.75
Implicit Intercept:            True    BIC:                        1102.70
Covariance Type:          OLS_SMALL    scale:                   8.7226e-01
                                       

═══════════════════════════════════════════════════════════════════
              coef        std err      t   p>|t| [0.025,     0.975]
───────────────────────────────────────────────────────────────────
Intercept    9.775  ****   0.4561  21.43  <0.001    8.879     10.67
L1[<y>]    0.03017        0.04495   0.67   0.502  -0.0582    0.1185
<x0>         1.152  ****  0.04787  24.06  <0.001    1.058     1.246
L1[<x0>}    0.4418  ****  0.07037   6.28  <0.001   0.3034    0.5801
<w0>       -0.1335  ***   0.04681  -2.85   0.005  -0.2256  -0.04151
═══════════════════════════════════════════════════════════════════
Omnibus              4.301    Durbin-Watson:       2.055
Prob(Omnibus):       0.116    Skew:                0.248
Jarque-Bera(JB):     4.155    Kurtosis:            3.059
Prob(JB)             0.125    Cond. No.:          21.104
═══════════════════════════════════════════════════════════════════

formula:  <y> ~ Intercept + L1[<y>] + <x0> + L1[<x0>} + <w0>

Used t distribution with 394 df at test level 0.0500.

                                                       [kanly v=0.0.1041]

══════════════════════════════════════════════════════════════════════════
Linear Model Results
══════════════════════════════════════════════════════════════════════════

Dep. Variable:   y

Date:                  May 30, 2026    No. Obs.                        399
Time:                      09:35:12    Df Residuals:                   394
Model Elapsed:               0.00 s    Df Model:                         4
Fit Elapsed:                 0.00 s    R-squared:                   0.6398
Cov Elapsed:                 0.00 s    Adj. R-squared:              0.6361
Method:                         OLS    F-statistic:                 174.95
L2 Penalty:                    None    Prob (F-statistic):           <.001
Weights:                          -    Log-Likelihood:           -536.3755
Intercept:                     True    AIC:                        1082.75
Implicit Intercept:           False    BIC:                        1102.70
Covariance Type:          OLS_SMALL    scale:                   8.7226e-01
                                       

═══════════════════════════════════════════════════════════════════
              coef        std err      t   p>|t| [0.025,     0.975]
───────────────────────────────────────────────────────────────────
Intercept    9.775  ****   0.4561  21.43  <0.001    8.879     10.67
x            1.152  ****  0.04787  24.06  <0.001    1.058     1.246
w          -0.1335  ***   0.04681  -2.85   0.005  -0.2256  -0.04151
L[y]       0.03017        0.04495   0.67   0.502  -0.0582    0.1185
L[x]        0.4418  ****  0.07037   6.28  <0.001   0.3034    0.5801
═══════════════════════════════════════════════════════════════════
Omnibus              4.301    Durbin-Watson:       2.055
Prob(Omnibus):       0.116    Skew:                0.248
Jarque-Bera(JB):     4.155    Kurtosis:            3.059
Prob(JB)             0.125    Cond. No.:          21.104
═══════════════════════════════════════════════════════════════════

formula:  y ~ x + w + L(y,1) + L(x,1)

Used t distribution with 394 df at test level 0.0500.

                                                       [kanly v=0.0.1041]
"""
