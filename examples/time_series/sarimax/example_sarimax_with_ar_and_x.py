from kanly.api import simulate_sarima, sarimax

import numpy as np
import pandas as pd

# simulate an ARMA model
n = 3_000
u = simulate_sarima(n, ar=[.4, .1], seed=0, sigma2=1.3)
x = np.random.randn(n)
t = np.arange(0, n)
y = u + 1.5 * x + 1.2 + .001 * t - .00005 * t ** 2

df = pd.DataFrame({'y': y, 'x': x})

# fit an ARMAX model
fit = sarimax('y ~ x', df, order=(2, 0, 0), trend=(1, 1, 1))
print(fit)

"""
════════════════════════════════════════════════════════════════════════════
SARIMAX Model Results
════════════════════════════════════════════════════════════════════════════

Dep. Variable: y

Model:                 ARX(2,0,0)    Log Likelihood:        -4647.3024
                                     Avg. LL:                  -1.5491
Date:                May 29, 2026    AIC:                     9308.605
Time:                    18:09:31    AICc:                    9308.642
model time:                 0.00s    BIC:                     9350.649
fit time:                   0.33s    HQIC:                    9323.728
cov time:                   0.00s    R-squared:                 0.9999
Covariance Type:              opg    Adj. R-squared:            0.9999
No. Observations:            3000    converged:                   True
Likelihood Burn:                0    max|grad|:               2.39e+07
df Model:                       7    iter:                           1

════════════════════════════════════════════════════════════════════════════
              coef          std err        t   p>|t|    [0.025,       0.975]
────────────────────────────────────────────────────────────────────────────
const        1.066  ****    0.08107    13.15  <0.001       0.907       1.225
t**1      0.001319  ****   5.12e-05    25.77  <0.001    0.001219     0.00142
t**2    -5.007e-05  ****  6.677e-09 -7498.57  <0.001  -5.008e-05  -5.005e-05
x            1.496  ****    0.01937    77.22  <0.001       1.458       1.534
ar.L1       0.3943  ****    0.01864    21.16  <0.001      0.3578      0.4309
ar.L2      0.09501  ****     0.0189     5.03  <0.001     0.05796      0.1321
sigma2       1.297  ****    0.03364    38.56  <0.001       1.231       1.363
════════════════════════════════════════════════════════════════════════════
Ljung-Box(L1)(Q):    0.009    Durbin-Watson:       1.996
Prob(Q):             0.924    Skew:                0.043
Jarque-Bera(JB):     0.943    Kurtosis:            3.003
Prob(JB)             0.624    
════════════════════════════════════════════════════════════════════════════

formula:  y ~ x -1

Parameters estimated via maximum-likelihood
Uses Brockwell & Davis ARMA state space representation on
pre-differenced data.

                                                         [kanly v=0.0.1041]
"""
