from kanly.api import simulate_sarima, SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX as SARIMAX_SM

import numpy as np
import pandas as pd

# simulate an ARMA model
n = 1000
np.random.seed(0)
y = simulate_sarima(n, ar=[.4, .1], ma=[0, 0, -.34], seed=10, sigma2=1.3)
x = np.random.randn(n)
y = y + 3.2 * x + .02 * np.arange(n)
t = np.arange(0, n)

# fit an ARMAX model
fit = SARIMAX(y, exog=x.reshape((-1, 1)), order=(2, 0, (3,)), trend='ct', B0=1, gtol=1e-8, ftol=1e-12, xtol=1e-8,
              do_hannan_rissanen=True,
              concentrate_scale=False)
print(fit)

# compare to statsmodels
fit_statsmodels = SARIMAX_SM(y, exog=x.reshape((-1, 1)), order=(2, 0, (3,)), trend='ct',
                             concentrate_scale=False,
                             ).fit(disp=False)
print(fit_statsmodels.summary())

"""
════════════════════════════════════════════════════════════════════════════
SARIMAX Model Results
════════════════════════════════════════════════════════════════════════════

Dep. Variable: y

Model:               ARMAX(2,0,(3,))    Log Likelihood:           -1543.4067
                                        Avg. LL:                     -1.5434
Date:                   May 29, 2026    AIC:                        3100.813
Time:                       19:05:37    AICc:                       3100.926
model time:                    0.00s    BIC:                        3135.168
fit time:                      0.16s    HQIC:                       3113.871
cov time:                      0.00s    R-squared:                    0.9708
Covariance Type:                 opg    Adj. R-squared:               0.9706
No. Observations:               1000    converged:                      True
Likelihood Burn:                   0    max|grad|:                  2.59e-04
df Model:                          7    iter:                             11

══════════════════════════════════════════════════════════════════════
                coef        std err       t   p>|t|  [0.025,    0.975]
──────────────────────────────────────────────────────────────────────
const        0.09682        0.09403    1.03   0.303  -0.08747   0.2811
(t/1000)**1    19.77  ****   0.1643  120.29  <0.001     19.45    20.09
x1             3.225  ****  0.03357   96.08  <0.001      3.16    3.291
ar.L1          0.389  ****  0.03437   11.32  <0.001    0.3216   0.4564
ar.L2         0.1089  ****  0.03193    3.41  <0.001   0.04628   0.1715
ma.L3        -0.3345  ****  0.03313  -10.10  <0.001   -0.3995  -0.2696
sigma2         1.282  ****  0.05993   21.39  <0.001     1.165      1.4
══════════════════════════════════════════════════════════════════════
Ljung-Box(L1)(Q):     0.000    Durbin-Watson:        1.994
Prob(Q):              0.988    Skew:                -0.058
Jarque-Bera(JB):      1.592    Kurtosis:             2.843
Prob(JB)              0.451    
══════════════════════════════════════════════════════════════════════


Parameters estimated via maximum-likelihood
Uses Brockwell & Davis ARMA state space representation on
pre-differenced data.

                                                         [kanly v=0.0.1041]

                               SARIMAX Results                                
==============================================================================
Dep. Variable:                      y   No. Observations:                 1000
Model:             SARIMAX(2, 0, [3])   Log Likelihood               -1544.641
Date:                Fri, 29 May 2026   AIC                           3103.283
Time:                        19:05:38   BIC                           3137.637
Sample:                             0   HQIC                          3116.340
                               - 1000                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
intercept      0.0945      0.048      1.955      0.051      -0.000       0.189
drift          0.0091      0.001     12.755      0.000       0.008       0.010
x1             3.2487      0.034     94.931      0.000       3.182       3.316
ar.L1          0.4026      0.035     11.459      0.000       0.334       0.471
ar.L2          0.1350      0.033      4.125      0.000       0.071       0.199
ma.L3         -0.3414      0.034    -10.040      0.000      -0.408      -0.275
sigma2         1.3208      0.064     20.774      0.000       1.196       1.445
===================================================================================
Ljung-Box (L1) (Q):                   0.15   Jarque-Bera (JB):                 1.57
Prob(Q):                              0.70   Prob(JB):                         0.46
Heteroskedasticity (H):               0.96   Skew:                            -0.05
Prob(H) (two-sided):                  0.69   Kurtosis:                         2.84
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).

"""