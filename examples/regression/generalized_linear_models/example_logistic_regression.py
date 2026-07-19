import numpy as np
import pandas as pd

from kanly.api import glm

n = 10_000
np.random.seed(0)
df = pd.DataFrame({
     'x': np.random.randn(n),
})
p = 1.0 / (1.0 + np.exp(-(-.2 + .16 * df['x'] - .048 * df['x'] ** 2)))
df['y'] = (np.random.rand(n) < p).astype(int)

fit = glm('y ~ poly(x,2)', df, family='binomial',
          specification_name='logistic regression example', cov_type='bootstrap', cov_kwds={'n_samples': 500})

print(fit)
print(fit.get_marginal_effects())

"""
══════════════════════════════════════════════════════════════════
Generalized Linear Model Results
logistic regression example
══════════════════════════════════════════════════════════════════

Dep. Variable: y

Date:              Dec 09, 2024    Deviance:            1.3665e+04
Time:                  15:25:27    Pearson chi2:            0.0041
Family:                BINOMIAL    Scale:               1.0000e+00
Link:                     LOGIT    Converged:                 True
Var Weights:                  -    Iterations:                   4
Method:                    IRLS    Rel. Err.:             3.10e-09
Nobs:                     10000    Abs. Err.:             2.85e-11
Df Residuals:              9997    Cov. Type:            BOOTSTRAP
Df Model:                     3    Model Elapsed:            0.01s
Log-Likelihood:     -6.8327e+03    Fit Elapsed:              0.01s
Pseudo Rsq:              0.0041    Cov Elapsed:              3.19s

══════════════════════════════════════════════════════════════════
              coef        std err     t   p>|t|  [0.025,    0.975]
──────────────────────────────────────────────────────────────────
Intercept  -0.1975  ****    0.024 -8.23  <0.001   -0.2445  -0.1504
x           0.1438  ****   0.0209  6.88  <0.001    0.1028   0.1847
I(x**2)    -0.0414  **    0.01525 -2.71   0.007  -0.07131  -0.0115
══════════════════════════════════════════════════════════════════

formula:  y ~ poly(x,2)

fit_intercept = True
Link Function: g(x) = log(x/(1-x))
Used t distribution with 9997 df at test level 0.0500.
Did 500 Bayesian bootstrap repetitions, alpha=1.000.

                                                [kanly v=0.0.826]
                                                
═══════════════════════════════════════════════════════════════════════
GLM Marginal Effects
logistic regression example
───────────────────────────────────────────────────────────────────────

Dep. Var.               y
Method:              dydx
At:               overall
Date:        Jun 11, 2026
Time:            15:20:46

═══════════════════════════════════════════════════════════════════════
            dy/dx   std err         z         p>|z|  [0.025,     0.975]
───────────────────────────────────────────────────────────────────────
x        0.035238  0.005077  6.941010  3.893068e-12  0.025287  0.045188
I(x**2) -0.010148  0.003733 -2.718413  6.559592e-03 -0.017465 -0.002831
───────────────────────────────────────────────────────────────────────
                                                     [kanly v=0.0.1047]
"""
