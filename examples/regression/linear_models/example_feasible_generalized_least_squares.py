import numpy as np
import pandas as pd

from kanly.api import lm, compare_results

n = 100
np.random.seed(0)
df = pd.DataFrame({
    'x': np.random.randn(n),
})
df['y'] = 1.2 - 0.3 * df['x'] + .2 * np.random.randn(n) * (10 * (df.x > 1))

fit_fgls = lm('y ~ x', df, do_fgls=True, fgls_kwds={'maxiter': 10, 'tol': 1e-4}, specification_name='fgls')
print(fit_fgls)

fit_ols = lm('y ~ x', df, specification_name='ols')

print(compare_results([fit_fgls, fit_ols], ref_param_values={'Intercept': 1.2, 'x': -.3}))

"""
==========================================================================
Linear Model Results
fgls
==========================================================================

Dep. Variable:   y

Date:                  Oct 14, 2022    No. Obs.                        100
Time:                      12:02:51    Df Residuals:                    98
Model Elapsed:               0.01 s    Df Model:                         1
Fit Elapsed:                 0.05 s    R-squared:                    0.803
Cov Elapsed:                 0.00 s    Adj. R-squared:               0.801
Method:                         OLS    F-statistic:                 399.91
Weights:                   FGLS[10]    Prob (F-statistic):           <.001
Intercept:                     True    Log-Likelihood:              1.8293
Implicit Intercept:           False    AIC:                           0.34
Covariance Type:          OLS_SMALL    BIC:                           5.55

==================================================================
              coef        std err      t   p>|t| [0.025,    0.975]
------------------------------------------------------------------
Intercept    1.223  ****   0.0242  50.54  <0.001    1.175    1.271
x          -0.2866  ****  0.01433 -20.00  <0.001   -0.315  -0.2581
==================================================================

formula:  y ~ x

Used t distribution with 98 df at test level 0.050.

                                                        [kanly v=0.0.304]


============================================================
Regression Summary Table
============================================================
                           (0)        (1)   |  Reference
------------------------------------------------------------
Intercept                 1.22       1.32   |        1.2
                      (0.0242)    (0.082)   |           


x                       -0.287     -0.135   |       -0.3
                      (0.0143)   (0.0812)   |           
============================================================
Outcome:                     y          y   |           
No. Obs.                   100        100   |           
R-squared: :            0.8012     0.0174   |           
R-squared Adj.:         0.8012     0.0174   |           
Pseudo R-squared: :                         |           
Method:                    OLS        OLS   |           
Weights:              FGLS[10]          -   |           
Df Residuals:               98         98   |           
Df Model:                    1          1   |           
Covariance Type:     OLS_SMALL  OLS_SMALL   |           
------------------------------------------------------------
(0)      y ~ x
(1)      y ~ x
------------------------------------------------------------
(0): fgls
(1): ols
============================================================
                                          [kanly, v=0.0.304]
"""