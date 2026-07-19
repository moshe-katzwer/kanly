import numpy as np
import pandas as pd

from kanly.api import nlls

n, k = 5_000, 25
np.random.seed(0)

df = pd.DataFrame({f'x{j:02}': np.random.randn(n) for j in range(k)})
beta = np.random.randn(k)
df['y'] = 1.6 + df.dot(beta) + np.random.randn(n)

fit = nlls('[y] ~ {Intercept} + ' + ' + '.join([f'{{alpha{j:02}}}*[x{j:02}]' for j in range(k)]),
           df,
           bounds=[(-np.inf, np.inf)] + [(0, np.inf)] * k,
           specification_name='bounded least squares example',
           debug=True)

print(fit)

"""
==========================================================================
Nonlinear Least Squares Results
bounded least squares example
==========================================================================

Dep. Variable: y

Date:                  Sep 22, 2022    Adj. R-squared:              0.5836
Time:                      13:41:27    Model Time:                   0.00s
Weights:                       None    Fit Time:                     1.70s
Nobs:                          5000    Cov Time:                     0.00s
Df Residuals:                  4974    Iterations:                       8
Df Model:                        26    Converged:                     True
Cost:                    2.8199e+04    Status:                           1
Optimality:                1.94e-06    Covariance Type:               None
R-squared:                   0.5857    Active Constraints:              10

==========================================================================
              coef
--------------------------------------------------------------------------
Intercept  1.56661
alpha00    0.00000
alpha01    1.96261
alpha02    1.02538
alpha03    0.37454
alpha04    0.74942
alpha05    0.43069
alpha06    0.15565
alpha07    0.00000
alpha08    0.08199
alpha09    0.00000
alpha10    0.49426
alpha11    0.00000
alpha12    0.57985
alpha13    1.44986
alpha14    0.00000
alpha15    0.40746
alpha16    0.00000
alpha17    1.34733
alpha18    0.00000
alpha19    0.00000
alpha20    0.15328
alpha21    1.00157
alpha22    0.00000
alpha23    0.00000
alpha24    2.21391
==========================================================================


[y] ~ {Intercept} + {alpha00}*[x00] + {alpha01}*[x01] + {alpha02}*[x02]
+ {alpha03}*[x03] + {alpha04}*[x04] + {alpha05}*[x05] + {alpha06}*[x06]
+ {alpha07}*[x07] + {alpha08}*[x08] + {alpha09}*[x09] + {alpha10}*[x10]
+ {alpha11}*[x11] + {alpha12}*[x12] + {alpha13}*[x13] + {alpha14}*[x14]
+ {alpha15}*[x15] + {alpha16}*[x16] + {alpha17}*[x17] + {alpha18}*[x18]
+ {alpha19}*[x19] + {alpha20}*[x20] + {alpha21}*[x21] + {alpha22}*[x22]
+ {alpha23}*[x23] + {alpha24}*[x24]

message: |dF| < ftol * max(1, |F|)

                                      [kanly package by moshe, v=0.0.255]
"""
