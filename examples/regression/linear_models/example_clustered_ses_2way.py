import numpy as np
import pandas as pd

from kanly.api import lm, compare_results

n = 10_000
np.random.seed(0)

df = pd.DataFrame({
    'x': np.random.randn(n),
    'grp1': np.random.randint(0, 60, n),
    'grp2': np.random.randint(0, 40, n),
})
df['y'] = 1.2 - 0.3 * df['x'] + .2 * (np.exp(-df['x']) * np.random.randn(n) * (1 + df['grp1']))

fit_2way = lm('y ~ x', df, cov_type='cluster',
              cov_kwds={'groups': ('grp1', 'grp2')}, specification_name='2way-Cluster')
fit1 = lm('y ~ x', df, cov_type='cluster',
          cov_kwds={'groups': 'grp1'}, specification_name='1way-Cluster')
fit2 = lm('y ~ x', df, cov_type='cluster',
          cov_kwds={'groups': 'grp2'}, specification_name='1way-Cluster')

print(compare_results([fit_2way, fit1, fit2]))

"""
============================================================
Regression Summary Table
============================================================
                        (0)      (1)      (2)
------------------------------------------------------------
Intercept              1.23     1.23     1.23
                    (0.168)   (0.19)  (0.158)


x                     -0.34    -0.34    -0.34
                    (0.331)  (0.388)  (0.319)
============================================================
Outcome:                  y        y        y
No. Obs.              10000    10000    10000
R-squared:           0.0003   0.0003   0.0003
R-squared Adj.:      0.0002   0.0002   0.0002
Pseudo R-squared:                            
Method:                 OLS      OLS      OLS
Weights:                  -        -        -
Df Residuals:          9998     9998     9998
Df Model:                 1        1        1
Covariance Type:    CLUSTER  CLUSTER  CLUSTER
Converged:               NA       NA       NA
------------------------------------------------------------
(0)  "2way-Cluster"
(1)  "1way-Cluster"
(2)  "1way-Cluster"
============================================================
                                          [kanly, v=0.0.461]
"""
