import numpy as np
import pandas as pd

from kanly.api import glm, compare_results

n = 5_000_000
np.random.seed(0)
df = pd.DataFrame({
     'z': np.random.randn(n),
     'e': np.random.randn(n)
})

df['x'] = 1.2 - .9 * df['z'] + .03 * df['z'] ** 2 + .4 * df['e']
p = 1.0 / (1.0 + np.exp(-(-.2 + .17 * df['x'] + .5 * df['e'])))
df['y'] = (np.random.rand(n) < p).astype(int)

fit = glm('y ~ x', df, specification_name='"OLS"', residual_inclusion=False)
fit_iv = glm('y ~ x | z', df, specification_name='IV, No RI', residual_inclusion=False)
fit_ri = glm('y ~ x | z', df, family='binomial',
             debug=True, specification_name='IV, RI',
             start_params={'x': .2},
             residual_inclusion_order=3, residual_inclusion=True)

print(fit_ri)

"""
====================================================================
Generalized Linear Model Results
IV, RI
====================================================================

Dep. Variable: y

Date:              Dec 01, 2023    Deviance:            1.3129e+05
Time:                  09:48:07    Pearson chi2:            0.0528
Family:                BINOMIAL    Scale:               1.0000e+00
Link:                     LOGIT    Converged:                 True
Var Weights:                  -    Iterations:                   6
Method:                    IRLS    Rel. Err.:             9.86e-09
Nobs:                    100000    Abs. Err.:             1.58e-13
Df Residuals:             99995    Cov. Type:            NONROBUST
Df Model:                     5    Model Elapsed:            0.06s
Log-Likelihood:     -6.5646e+04    Fit Elapsed:              0.14s
Pseudo Rsq:              0.0528    Cov Elapsed:              0.01s

====================================================================
              coef         std err      t   p>|t|  [0.025,    0.975]
--------------------------------------------------------------------
Intercept  -0.1794  ****    0.0121 -14.82  <0.001   -0.2031  -0.1557
x           0.1641  ****  0.007363  22.29  <0.001    0.1497   0.1785
x_r(1)       1.333  ****   0.02774  48.04  <0.001     1.278    1.387
x_r(2)     0.01014         0.03317   0.31    0.76  -0.05487  0.07515
x_r(3)      0.1036  *      0.05128   2.02   0.043  0.003059   0.2041
====================================================================

formula:  y ~ x | z

fit_intercept = True
Link Function: g(x) = log(x/(1-x))

IV residual_inclusion=True

*** Note: IV is complicated in non-linear settings,
    experts only! ***

*** Note: NON-BOOTSTRAP INFERENCE MAY BE 
    UNRELIABLE FOR INSTRUMENTAL VARIABLES!! ***
Used t distribution with 99995 df at test level 0.0500.

                                                  [kanly v=0.0.529]
"""

print(compare_results([fit, fit_iv, fit_ri]))

"""
============================================================
Regression Summary Table
============================================================
                          (0)        (1)        (2)
------------------------------------------------------------
Intercept               0.401      0.459     -0.179
                     (0.0025)  (0.00266)   (0.0121)


x                       0.085      0.038      0.164
                    (0.00159)  (0.00174)  (0.00736)


x_r(1)                                         1.33
                                           (0.0277)


x_r(2)                                       0.0101
                                           (0.0332)


x_r(3)                                        0.104
                                           (0.0513)
============================================================
Outcome:                    y          y          y
No. Obs.               100000     100000     100000
R-squared:                                         
R-squared Adj.:                                    
Pseudo R-squared:      0.0198     0.0033     0.0528
Method:                   GLM        GLM        GLM
Weights:                    -          -          -
Df Residuals:           99998      99998      99995
Df Model:                   2          2          5
Covariance Type:    NONROBUST  NONROBUST  NONROBUST
Converged:               True       True       True
------------------------------------------------------------
(0)  ""OLS""
(1)  "IV, No RI"
(2)  "IV, RI"
============================================================
                                          [kanly, v=0.0.529]
"""
