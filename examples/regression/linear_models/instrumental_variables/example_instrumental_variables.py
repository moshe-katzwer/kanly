import numpy as np
import pandas as pd

from kanly.api import lm, compare_results

n = 250
np.random.seed(0)
df = pd.DataFrame({
    'z': np.random.randn(n),
    'e': np.random.randn(n),
    'grp': np.random.randint(0, 12, n),
    'obs': np.random.randint(1, 6, n),
})
df['x'] = .5 * df['z'] + .7 * df['e']
df['y'] = 1.2 - 0.3 * df['x'] + .2 * df['e'] / df['obs']

fit_ols = lm('y ~ x + C(grp) $ obs', df, specification_name='WLS')
fit_iv = lm('y ~ x + C(grp) | z + C(grp) $ obs', df, specification_name='WLS-IV', debug=False)
fit_iv_absorb = lm('y ~ x | z  $ obs', df, specification_name='WLS-IV-Absorb', absorb='grp')

print(fit_ols)
print(fit_iv)
print(fit_iv_absorb)

print(compare_results(
    [fit_iv, fit_iv_absorb, fit_ols],
    ref_param_values={'x': -.3},
    parameter_subset=['x'],
    show_bse=True, show_formulas=True))


"""
══════════════════════════════════════════════════════════════════════════
Linear Model Results
WLS
══════════════════════════════════════════════════════════════════════════

Dep. Variable:   y

Date:                  May 05, 2026    No. Obs.                        250
Time:                      10:28:52    Df Residuals:                   237
Model Elapsed:               0.00 s    Df Model:                        12
Fit Elapsed:                 0.00 s    R-squared:                   0.9123
Cov Elapsed:                 0.00 s    Adj. R-squared:              0.9079
Method:                         WLS    F-statistic:                 205.45
L2 Penalty:                    None    Prob (F-statistic):           <.001
Weights:                        obs    Log-Likelihood:            338.8064
Intercept:                     True    AIC:                        -651.61
Implicit Intercept:           False    BIC:                        -605.83
Covariance Type:          OLS_SMALL    scale:                   1.1055e-02
                                       

═══════════════════════════════════════════════════════════════════════
                 coef         std err      t   p>|t|  [0.025,    0.975]
───────────────────────────────────────────────────────────────────────
Intercept       1.197  ****   0.01347  88.92  <0.001     1.171    1.224
x             -0.2347  ****  0.004888 -48.01  <0.001   -0.2443  -0.2251
C(grp)[1]   -0.004733         0.01854  -0.26   0.799  -0.04126   0.0318
C(grp)[2]    -0.01415         0.01826  -0.77   0.439  -0.05013  0.02183
C(grp)[3]     0.00201         0.01874   0.11   0.915  -0.03491  0.03893
C(grp)[4]   -0.004767         0.02009  -0.24   0.813  -0.04434   0.0348
C(grp)[5]   -0.000346         0.01884  -0.02   0.985  -0.03746  0.03677
C(grp)[6]    0.007767         0.01968   0.39   0.693  -0.03101  0.04654
C(grp)[7]     0.01643         0.01744   0.94   0.347  -0.01792  0.05078
C(grp)[8]    -0.01035         0.02251  -0.46   0.646  -0.05469    0.034
C(grp)[9]     0.00975         0.01833   0.53   0.595  -0.02636  0.04586
C(grp)[10]   0.003632          0.0192   0.19    0.85   -0.0342  0.04147
C(grp)[11]   -0.01756         0.01774  -0.99   0.323   -0.0525  0.01738
═══════════════════════════════════════════════════════════════════════
Omnibus              42.630    Durbin-Watson:        1.977
Prob(Omnibus):        0.000    Skew:                -0.793
Jarque-Bera(JB):    101.959    Kurtosis:             5.697
Prob(JB)              0.000    Cond. No.:           13.336
═══════════════════════════════════════════════════════════════════════

formula:  y ~ x + C(grp) $ obs

Used t distribution with 237 df at test level 0.0500.

                                                       [kanly v=0.0.1012]

══════════════════════════════════════════════════════════════════════════
Linear Model Results
WLS-IV
══════════════════════════════════════════════════════════════════════════

Dep. Variable:   y

Date:                  May 05, 2026    No. Obs.                        250
Time:                      10:28:52    Df Residuals:                   237
Model Elapsed:               0.00 s    Df Model:                        12
Fit Elapsed:                 0.00 s    R-squared:                    0.797
Cov Elapsed:                 0.00 s    Adj. R-squared:              0.7868
Method:                  IV (W2SLS)    F-statistic:                      -
L2 Penalty:                    None    Prob (F-statistic):               -
Weights:                        obs    Log-Likelihood:                   -
Intercept:                     True    AIC:                              -
Implicit Intercept:           False    BIC:                              -
Covariance Type:          OLS_SMALL    scale:                   2.5584e-02
                                       

═══════════════════════════════════════════════════════════════════════
                  coef        std err      t   p>|t|  [0.025,    0.975]
───────────────────────────────────────────────────────────────────────
Intercept        1.192  ****   0.0205  58.12  <0.001     1.151    1.232
x               -0.321  ****  0.01468 -21.86  <0.001   -0.3499   -0.292
C(grp)[1]   -0.0006298        0.02822  -0.02   0.982  -0.05622  0.05496
C(grp)[2]      0.02486        0.02837   0.88   0.382  -0.03102  0.08075
C(grp)[3]     0.002119        0.02851   0.07   0.941  -0.05405  0.05829
C(grp)[4]     -0.02335        0.03068  -0.76   0.447  -0.08378  0.03709
C(grp)[5]     -0.01712        0.02876  -0.60   0.552  -0.07379  0.03954
C(grp)[6]     -0.01317         0.0301  -0.44   0.662  -0.07247  0.04612
C(grp)[7]      0.01855        0.02653   0.70   0.485  -0.03372  0.07081
C(grp)[8]     -0.01951        0.03427  -0.57    0.57  -0.08702  0.04799
C(grp)[9]      0.02903        0.02803   1.04   0.301  -0.02619  0.08424
C(grp)[10]     0.01174        0.02924   0.40   0.688  -0.04586  0.06934
C(grp)[11]    -0.01092          0.027  -0.40   0.686  -0.06411  0.04226
═══════════════════════════════════════════════════════════════════════
Omnibus             12.061    Durbin-Watson:       2.095
Prob(Omnibus):       0.002    Skew:               -0.308
Jarque-Bera(JB):    18.365    Kurtosis:            4.176
Prob(JB)             0.000    Cond. No.:          13.339
═══════════════════════════════════════════════════════════════════════

formula:  y ~ x + C(grp) | z + C(grp) $ obs

Endogenous Regressors: x

Excluded Regressors:   z

Used t distribution with 237 df at test level 0.0500.

                                                       [kanly v=0.0.1012]

══════════════════════════════════════════════════════════════════════════
Linear Model Results
WLS-IV-Absorb
══════════════════════════════════════════════════════════════════════════

Dep. Variable:   y

Date:                  May 05, 2026    No. Obs.                        250
Time:                      10:28:52    Df Residuals:                   237
Model Elapsed:               0.00 s    Df Model:                        12
Fit Elapsed:                 0.00 s    R-squared:                    0.797
Cov Elapsed:                 0.00 s    Adj. R-squared:              0.7868
Method:                  IV (W2SLS)    F-statistic:                      -
L2 Penalty:                    None    Prob (F-statistic):               -
Weights:                        obs    Log-Likelihood:                   -
Intercept:                    False    AIC:                              -
Implicit Intercept:            True    BIC:                              -
Covariance Type:          OLS_SMALL    scale:                   2.5584e-02
                                       

════════════════════════════════════════════════════════
     coef        std err      t   p>|t| [0.025,   0.975]
────────────────────────────────────────────────────────
x  -0.321  ****  0.01468 -21.86  <0.001  -0.3499  -0.292
════════════════════════════════════════════════════════
Omnibus             12.061    Durbin-Watson:       2.095
Prob(Omnibus):       0.002    Skew:               -0.308
Jarque-Bera(JB):    18.365    Kurtosis:            4.176
Prob(JB)             0.000    Cond. No.:           1.000
════════════════════════════════════════════════════════

formula:  y ~ x | z  $ obs

Absorbed: 'grp', num=12
	Within  R² = 0.4094
	Between R² = 0.0594

Endogenous Regressors: x

Excluded Regressors:   z

Used t distribution with 237 df at test level 0.0500.

                                                       [kanly v=0.0.1012]


════════════════════════════════════════════════════════════════════
Regression Summary Table
════════════════════════════════════════════════════════════════════
                           (0)         (1)        (2)   |  Reference
────────────────────────────────────────────────────────────────────
x                       -0.321      -0.321     -0.235   |     -0.300
                       (0.015)     (0.015)    (0.005)   |           
════════════════════════════════════════════════════════════════════
Model:                     LLS         LLS        LLS   |           
Outcome:                     y           y          y   |           
No. Obs.                   250         250        250   |           
R-squared:              0.7970      0.7970     0.9123   |          -
R-squared Adj.:         0.7868      0.7868     0.9079   |           
Pseudo R-squared:                                       |           
Method:             IV (W2SLS)  IV (W2SLS)        WLS   |           
Weights:                   obs         obs        obs   |           
Df Residuals:              237         237        237   |           
Df Model:                   12          12         12   |           
Covariance Type:     OLS_SMALL   OLS_SMALL  OLS_SMALL   |           
────────────────────────────────────────────────────────────────────
(0)      y ~ x + C(grp), Instruments: {Intercept, z, C(grp)[1],
       C(grp)[2], C(grp)[3], C(grp)[4], C(grp)[5], C(grp)[6],
       C(grp)[7], C(grp)[8], C(grp)[9], C(grp)[10], C(grp)[11]}
(1)      y ~ x, Instruments: {z}, Absorbed: grp [num=12]
(2)      y ~ x + C(grp)
────────────────────────────────────────────────────────────────────
(0)  "WLS-IV"
(1)  "WLS-IV-Absorb"
(2)  "WLS"
════════════════════════════════════════════════════════════════════
                                                 [kanly, v=0.0.1012]
"""
