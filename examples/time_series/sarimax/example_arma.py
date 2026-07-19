from kanly.api import simulate_sarima, SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX as SARIMAX_SM

import numpy as np
import pandas as pd

# simulate an ARMA model
n = 100
y = simulate_sarima(n, ar=[.4, .1], ma=[0, 0, -.34], seed=0, sigma2=1.3)
t = np.arange(0, n)

# fit an ARMAX model
fit = SARIMAX(y, order=(2, 0, (3,)), trend='c')
print(fit)

# compare to statsmodels
fit_statsmodels = SARIMAX_SM(y, order=(2, 0, (3,)), trend='c').fit(disp=False)

print('\nparameters:')
print(pd.DataFrame(
    data=np.array([fit.params, fit_statsmodels.params]).T,
    columns=['kanly', 'statsmodels'],
    index=fit.params.index,
))
print('\nlog likelihood function:')
print(pd.DataFrame(
    data=np.array([[fit.llf], [fit_statsmodels.llf]]).T,
    columns=['kanly', 'statsmodels'],
    index=['llf']
))

"""
════════════════════════════════════════════════════════════════════════════
SARIMAX Model Results
════════════════════════════════════════════════════════════════════════════

Dep. Variable: y

Model:               ARMAX(2,0,(3,))    Log Likelihood:            -157.0398
                                        Avg. LL:                     -1.5704
Date:                   May 29, 2026    AIC:                         324.080
Time:                       18:32:49    AICc:                        324.718
model time:                    0.00s    BIC:                         337.105
fit time:                      0.15s    HQIC:                        329.351
cov time:                      0.00s    R-squared:                    0.2170
Covariance Type:                 opg    Adj. R-squared:               0.1840
No. Observations:                100    converged:                      True
Likelihood Burn:                   0    max|grad|:                  6.74e-05
df Model:                          5    iter:                             19

══════════════════════════════════════════════════════════════
           coef       std err     t   p>|t|  [0.025,    0.975]
──────────────────────────────────────────────────────────────
const   0.02657        0.1963  0.14   0.892   -0.3581   0.4112
ar.L1    0.4138  ****  0.1046  3.96  <0.001    0.2088   0.6188
ar.L2    0.1843  *     0.1027  1.80   0.073  -0.01693   0.3856
ma.L3    -0.341  ***   0.1145 -2.98   0.003   -0.5654  -0.1166
sigma2    1.347  ****  0.2544  5.29  <0.001    0.8481    1.845
══════════════════════════════════════════════════════════════
Ljung-Box(L1)(Q):    0.012    Durbin-Watson:       1.985
Prob(Q):             0.914    Skew:                0.217
Jarque-Bera(JB):     3.607    Kurtosis:            2.177
Prob(JB)             0.165    
══════════════════════════════════════════════════════════════


Parameters estimated via maximum-likelihood
Uses Brockwell & Davis ARMA state space representation on
pre-differenced data.

                                                         [kanly v=0.0.1041]
                                                         
                                                         
parameters:
           kanly  statsmodels
const   0.026566     0.010672
ar.L1   0.413780     0.413781
ar.L2   0.184321     0.184330
ma.L3  -0.340985    -0.340996
sigma2  1.346779     1.346767

log likelihood function:
          kanly  statsmodels
llf -157.039762  -157.039762          
"""
