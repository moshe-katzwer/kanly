"""
Autoregressive examples using both formula and array interfaces,
autoreg and AUTOREG.
"""

import numpy as np
import pandas as pd
from kanly.api import autoreg, AUTOREG

np.random.seed(0)
n, k = 36, 2
X = np.random.randn(n, k)
ar = [.6, .1]
p = len(ar)
b0 = 12

beta = np.arange(k)
Xbeta = X.dot(beta)
e = 1.2 * np.random.randn(n)

y = np.zeros(n)
trend = np.arange(n)

# AR structure on y
for j in range(n):
    y[j] = b0 + Xbeta[j] + e[j] + 1.5 * trend[j] - .006 * trend[j] ** 2 + 8 * ((trend[j] % 12) >= 10)
    if j >= p:
        y[j] += np.dot(ar[::-1], y[j - p:j])

df = pd.DataFrame()
for l in range(k):
    df[f'x{l}'] = X[:, l]

df['y'] = y

### FORMULA INTERFACE ###

formula = 'y ~ ' + ' + '.join([f'x{l}' for l in range(k)])

fit = autoreg(formula, df, lags=2, trend='ctt', seasonal_periods=12)
print(fit)

"""
══════════════════════════════════════════════════════════════════════════
Linear Model Results
══════════════════════════════════════════════════════════════════════════

Dep. Variable:   y

Date:                  May 26, 2026    No. Obs.                         34
Time:                      09:50:34    Df Residuals:                    16
Model Elapsed:               0.00 s    Df Model:                        17
Fit Elapsed:                 0.00 s    R-squared:                   0.9998
Cov Elapsed:                 0.00 s    Adj. R-squared:              0.9996
Method:                         OLS    F-statistic:                4432.47
L2 Penalty:                    None    Prob (F-statistic):           <.001
Weights:                          -    Log-Likelihood:            -35.1947
Intercept:                     True    AIC:                         106.39
Implicit Intercept:           False    BIC:                         133.86
Covariance Type:          OLS_SMALL    scale:                   9.8626e-01
                                       

═════════════════════════════════════════════════════════════════════════
                   coef         std err     t   p>|t|  [0.025,     0.975]
─────────────────────────────────────────────────────────────────────────
Intercept         14.05  ****      1.68  8.36  <0.001     10.49     17.61
x0               0.2268            0.23  0.99   0.339   -0.2608    0.7143
x1                1.285  ****    0.2251  5.71  <0.001    0.8076     1.762
L[y]             0.3984  ***     0.1021  3.90  <0.001    0.1819    0.6148
L2[y]            0.1683          0.1055  1.60    0.13  -0.05536     0.392
period[2/12]    0.01308           1.297  0.01   0.992    -2.736     2.762
period[3/12]     -1.568           1.184 -1.32   0.204    -4.077    0.9416
period[4/12]     -2.366  *        1.165 -2.03   0.059    -4.835    0.1032
period[5/12]     -2.086  *         1.19 -1.75   0.099    -4.609     0.437
period[6/12]     -2.866  **       1.281 -2.24    0.04    -5.582   -0.1504
period[7/12]     -3.471  **       1.294 -2.68   0.016    -6.215   -0.7278
period[8/12]     -1.554           1.314 -1.18   0.254    -4.339      1.23
period[9/12]     -3.294  **        1.31 -2.51   0.023    -6.072    -0.516
period[10/12]    -1.373           1.464 -0.94   0.362    -4.477     1.731
period[11/12]     6.086  ****     1.341  4.54  <0.001     3.243      8.93
period[12/12]     6.337  ****     1.086  5.83  <0.001     4.034      8.64
trend             2.605  ***     0.7283  3.58   0.003     1.061     4.149
trend[^2]      -0.01807  **    0.006971 -2.59    0.02  -0.03285  -0.00329
═════════════════════════════════════════════════════════════════════════
Omnibus               1.571    Durbin-Watson:        2.787
Prob(Omnibus):        0.456    Skew:                 0.466
Jarque-Bera(JB):      1.297    Kurtosis:             2.780
Prob(JB)              0.523    Cond. No.:          267.760
═════════════════════════════════════════════════════════════════════════

formula:  y ~ x0 + x1 + L(y, 1) + L(y, 2) + seasonal([12]) + trend([1,
          2])

Used t distribution with 16 df at test level 0.0500.

                                                       [kanly v=0.0.1033]
"""


### ARRAY INTERFACE ###

fit = AUTOREG(y, exog=X, lags=2, seasonal_periods=12, trend='ctt')
print(fit)

"""
══════════════════════════════════════════════════════════════════════════
Linear Model Results
══════════════════════════════════════════════════════════════════════════

Dep. Variable:   <y>

Date:                  May 26, 2026    No. Obs.                         34
Time:                      19:21:56    Df Residuals:                    16
Model Elapsed:               0.00 s    Df Model:                        17
Fit Elapsed:                 0.00 s    R-squared:                   0.9998
Cov Elapsed:                 0.00 s    Adj. R-squared:              0.9996
Method:                         OLS    F-statistic:                4432.47
L2 Penalty:                    None    Prob (F-statistic):           <.001
Weights:                          -    Log-Likelihood:            -35.1947
Intercept:                     True    AIC:                         106.39
Implicit Intercept:            True    BIC:                         133.86
Covariance Type:          OLS_SMALL    scale:                   9.8626e-01
                                       

═════════════════════════════════════════════════════════════════════════
                   coef         std err     t   p>|t|  [0.025,     0.975]
─────────────────────────────────────────────────────────────────────────
Intercept         14.05  ****      1.68  8.36  <0.001     10.49     17.61
trend             2.605  ***     0.7283  3.58   0.003     1.061     4.149
trend[^2]      -0.01807  **    0.006971 -2.59    0.02  -0.03285  -0.00329
<x0>             0.2268            0.23  0.99   0.339   -0.2608    0.7143
<x1>              1.285  ****    0.2251  5.71  <0.001    0.8076     1.762
period[2/12]    0.01308           1.297  0.01   0.992    -2.736     2.762
period[3/12]     -1.568           1.184 -1.32   0.204    -4.077    0.9416
period[4/12]     -2.366  *        1.165 -2.03   0.059    -4.835    0.1032
period[5/12]     -2.086  *         1.19 -1.75   0.099    -4.609     0.437
period[6/12]     -2.866  **       1.281 -2.24    0.04    -5.582   -0.1504
period[7/12]     -3.471  **       1.294 -2.68   0.016    -6.215   -0.7278
period[8/12]     -1.554           1.314 -1.18   0.254    -4.339      1.23
period[9/12]     -3.294  **        1.31 -2.51   0.023    -6.072    -0.516
period[10/12]    -1.373           1.464 -0.94   0.362    -4.477     1.731
period[11/12]     6.086  ****     1.341  4.54  <0.001     3.243      8.93
period[12/12]     6.337  ****     1.086  5.83  <0.001     4.034      8.64
L1[<y>]          0.3984  ***     0.1021  3.90  <0.001    0.1819    0.6148
L2[<y>]          0.1683          0.1055  1.60    0.13  -0.05536     0.392
═════════════════════════════════════════════════════════════════════════
Omnibus               1.571    Durbin-Watson:        2.787
Prob(Omnibus):        0.456    Skew:                 0.466
Jarque-Bera(JB):      1.297    Kurtosis:             2.780
Prob(JB)              0.523    Cond. No.:          267.760
═════════════════════════════════════════════════════════════════════════

formula:  <y> ~ Intercept + trend + trend[^2] + <x0> + <x1> +
          period[2/12] + period[3/12] + period[4/12] +
          period[5/12] + period[6/12] + period[7/12] +
          period[8/12] + period[9/12] + period[10/12] +
          period[11/12] + period[12/12] + L1[<y>] + L2[<y>]

Used t distribution with 16 df at test level 0.0500.

                                                       [kanly v=0.0.1034]
"""