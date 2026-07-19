import numpy as np
import pandas as pd

from kanly.api import lm
from kanly.api import compare_results


n = 2_500
num_groups = 20
np.random.seed(0)
df = pd.DataFrame({
    'grp': np.random.randint(0, num_groups, n)
})

grp_to_x_mean_dict = dict(zip(range(num_groups), np.random.randn(num_groups)))

df['x'] = np.random.randn(n) + df['grp'].map(grp_to_x_mean_dict)
df['y'] = 1.2 - 0.3 * df['x'] + .2 * np.exp(-df['x']) + np.random.randn(n)

fit_bb = lm('y ~ x', df, cov_type='bootstrap', cov_kwds={'n_samples': 1_000, 'seed': 0, 'groups': 'grp'})
print(fit_bb)

"""
==========================================================================
Linear Model Results
==========================================================================

Dep. Variable:   y

Date:                  Dec 08, 2023    No. Obs.                       2500
Time:                      11:07:42    Df Residuals:                  2498
Model Elapsed:               0.00 s    Df Model:                         1
Fit Elapsed:                 0.00 s    R-squared:                   0.3639
Cov Elapsed:                 1.12 s    Adj. R-squared:              0.3636
Method:                         OLS    F-statistic:                  28.47
L2 Penalty:                    None    Prob (F-statistic):           <.001
Weights:                          -    Log-Likelihood:          -4277.7769
Intercept:                     True    AIC:                        8559.55
Implicit Intercept:           False    BIC:                        8571.20
Covariance Type:          BOOTSTRAP    Cond. No.:             Not Computed
                                       scale:                   1.7953e+00

=================================================================
              coef       std err      t   p>|t| [0.025,    0.975]
-----------------------------------------------------------------
Intercept    1.765  ****  0.1075  16.41  <0.001     1.54     1.99
x          -0.7071  ****  0.1325  -5.34  <0.001  -0.9845  -0.4297
=================================================================

formula:  y ~ x

Used t distribution with 19 df at test level 0.0500.
Did 1000 Bayesian bootstrap repetitions, alpha=1.000, blocked on 'grp'.

Eigenvalues and condition number not computed.

                                                        [kanly v=0.0.532]
"""

fit_boot = lm('y ~ x', df, cov_type='bootstrap', cov_kwds={'n_samples': 1_000})
fit_ols = lm('y ~ x', df)
fit_hc1 = lm('y ~ x', df, cov_type='hc1')
fit_cluster = lm('y ~ x', df, cov_type='cluster', cov_kwds={'groups': 'grp'})

print(compare_results([fit_ols, fit_hc1, fit_boot, fit_cluster, fit_bb]))

"""
======================================================================
Regression Summary Table
======================================================================
                          (0)       (1)        (2)      (3)        (4)
----------------------------------------------------------------------
Intercept                1.76      1.76       1.76     1.76       1.76
                     (0.0277)  (0.0368)   (0.0367)  (0.128)    (0.108)


x                      -0.707    -0.707     -0.707   -0.707     -0.707
                     (0.0187)  (0.0394)   (0.0404)   (0.15)    (0.133)
======================================================================
Outcome:                    y         y          y        y          y
No. Obs.                 2500      2500       2500     2500       2500
R-squared:             0.3639    0.3639     0.3639   0.3639     0.3639
R-squared Adj.:        0.3636    0.3636     0.3636   0.3636     0.3636
Pseudo R-squared:                                                     
Method:                   OLS       OLS        OLS      OLS        OLS
Weights:                    -         -          -        -          -
Df Residuals:            2498      2498       2498     2498       2498
Df Model:                   1         1          1        1          1
Covariance Type:    OLS_SMALL       HC1  BOOTSTRAP  CLUSTER  BOOTSTRAP
Converged:                 NA        NA         NA       NA         NA
======================================================================
                                                    [kanly, v=0.0.532]
"""
