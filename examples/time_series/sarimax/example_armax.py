from kanly.api import simulate_sarima, sarimax

import numpy as np
import pandas as pd

# simulate an ARMA model
n = 3_000
u = simulate_sarima(n, ar=[.4, .1], ma=[0, 0, -.5], seed=0, sigma2=1.3)
x = np.random.randn(n)
t = np.arange(0, n)
y = u + 1.5 * x + 1.2 + .001 * t - .00005 * t ** 2

df = pd.DataFrame({'y': y, 'x': x})

# fit an ARMAX model
fit = sarimax('y ~ x', df, order=(2, 0, (3,)), trend=(1, 1, 1))
print(fit)

"""
═════════════════════════════════════════════════════════════════════════════
SARIMAX Model Results
═════════════════════════════════════════════════════════════════════════════

Dep. Variable: y

Model:               ARMAX(2,0,(3,))    Log Likelihood:           -4647.6808
                                        Avg. LL:                     -1.5492
Date:                   May 29, 2026    AIC:                        9311.362
Time:                       18:15:47    AICc:                       9311.410
model time:                    0.00s    BIC:                        9359.413
fit time:                      0.37s    HQIC:                       9328.645
cov time:                      0.01s    R-squared:                    0.9999
Covariance Type:                 opg    Adj. R-squared:               0.9999
No. Observations:               3000    converged:                      True
Likelihood Burn:                   0    max|grad|:                  2.32e+08
df Model:                          8    iter:                              1

═════════════════════════════════════════════════════════════════════════════
              coef          std err         t   p>|t|    [0.025,       0.975]
─────────────────────────────────────────────────────────────────────────────
const        1.133  ****    0.04089     27.70  <0.001       1.052       1.213
t**1      0.001209  ****  2.448e-05     49.37  <0.001    0.001161    0.001257
t**2    -5.003e-05  ****   1.71e-09 -29259.63  <0.001  -5.004e-05  -5.003e-05
x            1.475  ****    0.01638     90.02  <0.001       1.443       1.507
ar.L1       0.3963  ****    0.01885     21.02  <0.001      0.3593      0.4332
ar.L2        0.107  ****    0.01981      5.40  <0.001     0.06818      0.1458
ma.L3      -0.5119  ****    0.01698    -30.15  <0.001     -0.5452     -0.4786
sigma2       1.308  ****    0.03412     38.32  <0.001       1.241       1.374
═════════════════════════════════════════════════════════════════════════════
Ljung-Box(L1)(Q):    0.000    Durbin-Watson:       2.000
Prob(Q):             0.992    Skew:                0.044
Jarque-Bera(JB):     0.948    Kurtosis:            3.000
Prob(JB)             0.623    
═════════════════════════════════════════════════════════════════════════════

formula:  y ~ x -1

Parameters estimated via maximum-likelihood
Uses Brockwell & Davis ARMA state space representation on
pre-differenced data.

                                                          [kanly v=0.0.1041]
"""
