from kanly.api import rlm, lm, compare_results
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

n = 50
np.random.seed(0)
df = pd.DataFrame({
    'x': np.random.rand(n),
    'e': np.random.randn(n),
})

df.loc[np.random.choice(df.index, 5, replace=False), 'e'] += 15
df['y'] = 1.2 - .8 * df['x'] + df['e']

fit_rlm = rlm('y ~ x', df,
              cov_type='BOOTSTRAP',
              debug=True)
print(fit_rlm)

fit_ols_boot = lm('y ~ x', df, cov_type='BOOTSTRAP')

print(compare_results([fit_ols_boot, fit_rlm],
                   fit_titles=['ols', 'rlm'],
                   ref_param_values={'Intercept': 1.2, 'x': -.8}
                   )
      )

"""
========================================
Robust Regression (M-Estimation)
----------------------------------------

loss ("M"):     <statsmodels.robust.norms.HuberT object at 0x158f7e4f0>

nobs:           50
params:         2

max_iter:       50
x_tol:          1.0e-06
========================================



==================================================================
  iter          cost     % dCost        |dx|        scale     time
------------------------------------------------------------------
     0    1.5120e+02         nan         inf   2.2892e+00    0.00s
     1    4.4587e+01    -7.1e-01    5.05e-01   1.0057e+00    0.01s
     2    1.1715e+02     1.6e+00    1.74e-01   9.6662e-01    0.01s
     3    1.2263e+02     4.7e-02    3.32e-02   9.7835e-01    0.01s
     4    1.2093e+02    -1.4e-02    4.55e-03   9.8213e-01    0.01s
     5    1.2039e+02    -4.5e-03    8.03e-04   9.8218e-01    0.01s
     6    1.2038e+02    -5.8e-05    8.51e-05   9.8220e-01    0.02s
     7    1.2038e+02    -2.0e-05    1.04e-05   9.8220e-01    0.02s
     8    1.2038e+02    -2.0e-06    1.23e-06   9.8220e-01    0.02s
==================================================================
Bootstrap regression: 100it [00:02, 48.79it/s]
================================================================
Robust Linear Model (M-estimation)
================================================================

Dep. Variable: y

dep. variable:               y    scale (est):             0.982
weights:                     -    force scale:              None
nobs:                       50    cost:               1.2038e+02
model:                     rlm    converged:                True
method:                   IRLS    iterations:                  9
norm:                      str    model elapsed:           0.01s
df resid:                   48    fit elapsed:             0.02s
df model:                    1    max_iters:                  50
use_t:                    True    error:                1.46e-07
covariance type:     BOOTSTRAP    tol:                  1.00e-06

==============================================================
             coef       std err     t  p>|t| [0.025,    0.975]
--------------------------------------------------------------
Intercept   1.366  **    0.4421  3.09  0.003   0.4768    2.255
x          -1.289        0.6465 -1.99  0.052   -2.589  0.01103
==============================================================

formula:  y ~ x


                                              [kanly v=0.0.304]


============================================================
Regression Summary Table
============================================================
                           ols        rlm   |  Reference
------------------------------------------------------------
Intercept                 3.47       1.37   |        1.2
                        (1.59)    (0.442)   |           


x                        -2.55      -1.29   |       -0.8
                        (2.64)    (0.647)   |           
============================================================
Outcome:                     y          y   |           
No. Obs.                    50         50   |           
R-squared: :            0.0002              |           
R-squared Adj.:         0.0002              |           
Pseudo R-squared: :                0.0201   |           
Method:                    OLS       IRLS   |           
Weights:                     -          -   |           
Df Residuals:               48         48   |           
Df Model:                    1          1   |           
Covariance Type:     BOOTSTRAP  BOOTSTRAP   |           
------------------------------------------------------------
ols      y ~ x
rlm      y ~ Intercept + x
============================================================
                                          [kanly, v=0.0.304]
"""

plt.scatter(df.x, df.y, label='data', alpha=.5, color='k')
x_rng = np.linspace(df.x.min(), df.x.max(), 5)
plt.plot(x_rng, fit_rlm['Intercept'] + fit_rlm['x'] * x_rng,
         label='RLM', lw=2)
plt.plot(x_rng, fit_ols_boot['Intercept'] + fit_ols_boot['x'] * x_rng,
         label='OLS', lw=2)
plt.legend(loc='best')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
