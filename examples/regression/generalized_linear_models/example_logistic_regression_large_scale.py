import numpy as np
import pandas as pd

from kanly.api import glm

n = 100_000
np.random.seed(0)
df = pd.DataFrame({
     'x': np.random.randn(n),
     'g': np.random.randint(0, 100, n),
})
p = 1.0 / (1.0 + np.exp(-(-.2 + .16 * df['x'] - .048 * df['x']**2
                          + df['g'].map(dict(zip(range(100), np.random.randn(n))))))
           )
df['y'] = (np.random.rand(n) < p).astype(int)

fit = glm('y ~ poly(x,2) + C(g)', df, family='binomial',
          specification_name='logistic regression example', debug=True)
print(fit)


"""
==========================================================================
Generalized Linear Model Results
logistic regression example
--------------------------------------------------------------------------

Dep. Variable: y

Date:              Sep 26, 2022    Deviance:            13665.4423
Time:                  09:33:53    Pearson chi2:            0.0041
Family:                BINOMIAL    Scale:                      1.0
Link:                     LOGIT    Converged:                 True
Var Weights:                  -    Iterations:                   3
Method:                    IRLS    Rel. Err.:             3.10e-09
Nobs:                     10000    Abs. Err.:             2.85e-11
Df Residuals:              9997    Cov. Type:                  HC1
Df Model:                     3    Model Elapsed:            0.02s
Log-Likelihood:      -6832.7212    Fit Elapsed:              0.02s
Pseudo Rsq:              0.0041    Cov Elapsed:              0.00s


==========================================================================
                coef           std err     t   p>|t|   [0.025,      0.975]
--------------------------------------------------------------------------
Intercept   -0.19746    ****  0.024755 -7.98  <0.001   -0.24599   -0.14894
x            0.14376    ****  0.020638  6.97  <0.001    0.10331    0.18422
I(x**2)    -0.041403      **  0.014935 -2.77   0.006  -0.070678  -0.012128
==========================================================================

fit_intercept = True
Link Function: g(x) = log(x/(1-x))
                                      [kanly package by moshe, v=0.0.270]
"""
