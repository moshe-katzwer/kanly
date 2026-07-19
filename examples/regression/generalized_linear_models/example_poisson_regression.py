import numpy as np
import pandas as pd

from kanly.api import glm

n = 5_000
np.random.seed(0)
df = pd.DataFrame({
     'x': np.random.randn(n),
     'e': np.random.randn(n),
})
df['y'] = np.exp(-1.5 + 2 * df.x + df.e)

fit = glm('y ~ x', df, family='poisson', debug=True)
print(fit)

"""
==================================================================
Generalized Linear Model Results
==================================================================

Dep. Variable: y

Date:              Oct 11, 2022    Deviance:             8810.8613
Time:                  09:45:09    Pearson chi2:            0.7057
Family:                 POISSON    Scale:                      1.0
Link:                       LOG    Converged:                 True
Var Weights:                  -    Iterations:                   7
Method:                    IRLS    Rel. Err.:             1.15e-09
Nobs:                      5000    Abs. Err.:             9.61e-10
Df Residuals:              4998    Cov. Type:                  HC1
Df Model:                     2    Model Elapsed:            0.01s
Log-Likelihood:      -7709.9799    Fit Elapsed:              0.03s
Pseudo Rsq:              0.7057    Cov Elapsed:              0.00s

=================================================================
             coef        std err      t   p>|t| [0.025,    0.975]
-----------------------------------------------------------------
Intercept  -0.922  ****  0.07739 -11.91  <0.001   -1.074  -0.7703
x           1.882  ****  0.05976  31.50  <0.001    1.765        2
=================================================================

formula:  y ~ x

fit_intercept = True
Link Function: g(x) = log(x)

                              [kanly package by moshe, v=0.0.299]
"""
