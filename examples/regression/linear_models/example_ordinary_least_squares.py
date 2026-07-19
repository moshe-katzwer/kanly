import numpy as np
import pandas as pd
from kanly.api import lm

n = 100
np.random.seed(0)
df = pd.DataFrame({
    'x': np.random.randn(n),
    'grp': np.random.randint(0, 12, n),
})
df['y'] = 1.2 - 0.3 * df['x'] + .2 * np.random.randn(n)

fit = lm('y ~ x + C(grp)', df, use_t=True,
         debug=False,
         # cov_type='bootstrap',
         # cov_kwds={'max_processes': 6, 'n_samples': 10_000},
         # inverse_method=np.linalg.inv
         # compute_eigenvalues=True,
         )

print(fit.summary())
print(fit.summary_df())


"""
==========================================================================
Linear Model Results
==========================================================================

Dep. Variable:   y

Date:                  Jun 12, 2023    No. Obs.                        100
Time:                      09:47:08    Df Residuals:                    87
Model Elapsed:               0.01 s    Df Model:                        12
Fit Elapsed:                 0.01 s    R-squared:                   0.7551
Cov Elapsed:                 0.00 s    Adj. R-squared:              0.7213
Method:                         OLS    F-statistic:                  22.35
Weights:                          -    Prob (F-statistic):           <.001
Intercept:                     True    Log-Likelihood:             28.6145
Implicit Intercept:           False    AIC:                         -31.23
Covariance Type:          OLS_SMALL    BIC:                           2.64
                                       Cond. No.:                 1.52e+01
                                       scale:                   3.7972e-02

=====================================================================
                 coef        std err      t   p>|t| [0.025,    0.975]
---------------------------------------------------------------------
Intercept       1.148  ****  0.07958  14.43  <0.001     0.99    1.306
x              -0.313  ****  0.01993 -15.70  <0.001  -0.3526  -0.2734
C(grp)[1]     0.04559         0.1053   0.43   0.666  -0.1637   0.2549
C(grp)[2]     0.02104         0.1027   0.20   0.838  -0.1831   0.2252
C(grp)[3]     0.05604        0.09762   0.57   0.567   -0.138   0.2501
C(grp)[4]    -0.02239         0.1126  -0.20   0.843  -0.2461   0.2013
C(grp)[5]   -0.009628         0.1028  -0.09   0.926   -0.214   0.1947
C(grp)[6]     0.07483         0.1085   0.69   0.492  -0.1408   0.2905
C(grp)[7]     0.08615         0.1052   0.82   0.415   -0.123   0.2953
C(grp)[8]     0.01355        0.09768   0.14    0.89  -0.1806   0.2077
C(grp)[9]     0.02682          0.103   0.26   0.795  -0.1778   0.2315
C(grp)[10]   -0.09459         0.1262  -0.75   0.455  -0.3453   0.1561
C(grp)[11]    0.09414         0.1007   0.94   0.352  -0.1059   0.2942
=====================================================================

formula:  y ~ x + C(grp)

Used t distribution with 87 df at test level 0.0500.

                                                        [kanly v=0.0.423]
"""