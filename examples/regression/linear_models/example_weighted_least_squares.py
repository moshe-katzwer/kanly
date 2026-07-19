import numpy as np
import pandas as pd

from kanly.api import lm

n = 100
np.random.seed(0)
df = pd.DataFrame({
    'x': np.random.randn(n),
    'grp': np.random.randint(0, 12, n),
    'obs': np.random.randint(1, 6, n),
})
df['y'] = 1.2 - 0.3 * df['x'] + .2 * np.random.randn(n)

fit = lm('y ~ x + C(grp) $ obs', df, scale_design_matrix=False)
print(fit)

"""
══════════════════════════════════════════════════════════════════════════
Linear Model Results
══════════════════════════════════════════════════════════════════════════

Dep. Variable:   y

Date:                  Feb 22, 2025    No. Obs.                        100
Time:                      11:41:58    Df Residuals:                    87
Model Elapsed:               0.00 s    Df Model:                        12
Fit Elapsed:                 0.01 s    R-squared:                   0.7639
Cov Elapsed:                 0.00 s    Adj. R-squared:              0.7313
Method:                         WLS    F-statistic:                  23.46
L2 Penalty:                    None    Prob (F-statistic):           <.001
Weights:                        obs    Log-Likelihood:             21.7740
Intercept:                     True    AIC:                         -17.55
Implicit Intercept:           False    BIC:                          16.32
Covariance Type:          OLS_SMALL    Cond. No.:             Not Computed
                                       scale:                   1.1422e-01

═════════════════════════════════════════════════════════════════════
                coef        std err      t   p>|t| [0.025,     0.975]
─────────────────────────────────────────────────────────────────────
Intercept      1.313  ****  0.07058  18.61  <0.001    1.173     1.454
C(grp)[1]    -0.0869         0.1064  -0.82   0.416  -0.2984    0.1246
C(grp)[2]    -0.1899  *      0.1009  -1.88   0.063  -0.3905   0.01069
C(grp)[3]   -0.08235        0.09551  -0.86   0.391  -0.2722    0.1075
C(grp)[4]    -0.1874  *      0.1082  -1.73   0.087  -0.4024   0.02764
C(grp)[5]    -0.2334  **    0.09134  -2.56   0.012   -0.415  -0.05188
C(grp)[6]   0.002292         0.1034   0.02   0.982  -0.2032    0.2077
C(grp)[7]    -0.2116  **    0.09981  -2.12   0.037  -0.4099  -0.01317
C(grp)[8]   -0.08593        0.09106  -0.94   0.348  -0.2669   0.09507
C(grp)[9]   -0.07086        0.09319  -0.76   0.449  -0.2561    0.1144
C(grp)[10]   -0.1106         0.1123  -0.98   0.327  -0.3337    0.1126
C(grp)[11]  -0.08983        0.08977  -1.00    0.32  -0.2683    0.0886
x            -0.3098  ****   0.0192 -16.13  <0.001   -0.348   -0.2717
═════════════════════════════════════════════════════════════════════
Omnibus                    5.683    Durbin-Watson:             1.893
Prob(Omnibus):             0.058    Skew:                     -0.295
Jarque-Bera(JB):           6.506    Kurtosis:                  4.101
Prob(JB)                   0.039    Cond.No.:           not computed
═════════════════════════════════════════════════════════════════════

formula:  y ~ C(grp) + x $ obs

Used t distribution with 87 df at test level 0.0500.

Eigenvalues and condition number not computed.

                                                        [kanly v=0.0.874]
"""