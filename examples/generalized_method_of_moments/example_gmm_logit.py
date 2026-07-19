import numpy as np
import pandas as pd

from kanly.api import gmm, glm, compare_results

n = 1_000
np.random.seed(10)

data = pd.DataFrame()

data['x'] = np.random.randn(n)
data['prob'] = 1 / (1 + np.exp(-(-.295 + .3 * data.x)))
data['y'] = np.random.rand(n) < data.prob

resid_string = '[y] - 1.0 / (1.0 + np.exp(-{Intercept} - {x}*[x]))'
fit_gmm = gmm(
    [
        resid_string,
        (resid_string, '[x]'),
    ],
    data,
    specification_name='GMM Logit MLE',
)

print(fit_gmm)

fit_glm = glm('y~x', data, family='binomial', specification_name='GLM Logit')
print(compare_results([fit_gmm, fit_glm]))


"""
====================================================================
Generalized Method of Moments Results
GMM Logit MLE
====================================================================

Dep. Variable: {Not Applicable in GMM}

Date:               Oct 19, 2022    Converged:                  True
Time:                   21:28:20    No. Iters:                     3
Nobs:                       2500    Objective:            7.3369e-13
No. Moments:                   2    Cov Type:               SANDWICH
No. Params:                    2    Model Elapsed:             0.00s
Over Identified:           False    Fit Elapsed:               0.56s
Method:                 ONE_STEP    

=================================================================
              coef        std err     t   p>|t| [0.025,    0.975]
-----------------------------------------------------------------
Intercept  -0.2357  ****  0.04084 -5.77  <0.001  -0.3158  -0.1556
x            0.345  ****  0.04297  8.03  <0.001   0.2607   0.4292
=================================================================

Moment 0:   ([y] - 1.0/(1.0 + np.exp(-{Intercept}-{x}*[x])))
Moment 1:   ([y] - 1.0/(1.0 + np.exp(-{Intercept}-{x}*[x]))) * ([x])

Converged: relative change in objective_function -1.6e-10 size below f_tol=1.0e-08
Used t distribution with 2498 df at test level nan.

                                                  [kanly v=0.0.313]
                                                  
                                                  
============================================================
Regression Summary Table
============================================================
                          (0)       (1)
------------------------------------------------------------
Intercept              -0.236    -0.236
                     (0.0408)  (0.0408)


x                       0.345     0.345
                      (0.043)  (0.0429)
============================================================
Outcome:                              y
No. Obs.                 2500      2500
R-squared: :                           
R-squared Adj.:                        
Pseudo R-squared: :              0.0194
Method:              ONE_STEP       GLM
Weights:                 None         -
Df Residuals:            2498      2498
Df Model:                   2         2
Covariance Type:     SANDWICH       HC1
------------------------------------------------------------
(0)    y ~ Intercept + x
(1)    y ~ x
------------------------------------------------------------
(0): GMM Logit MLE
(1): GLM Logit
============================================================
                                          [kanly, v=0.0.313]
"""
