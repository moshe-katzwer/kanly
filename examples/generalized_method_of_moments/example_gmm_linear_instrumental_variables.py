import numpy as np
import pandas as pd

from kanly.api import gmm

n = 300
np.random.seed(10)

data = pd.DataFrame()
data['z1'] = np.random.randn(n)
data['z2'] = np.random.randn(n)
data['e'] = np.random.randn(n)
data['x'] = .5 * data.z1 + 1.2 * data.z2 + .5 * data.e
data['y'] = 1.2 + .4 * data.x + .15 * data.z1 + data.e * (1 + .1 * np.exp(data.z1))

fit_gmm = gmm(
    [
         '[y] - ({Intercept} + {x}*[x] + {z1}*[z1])',
        ('[y] - ({Intercept} + {x}*[x] + {z1}*[z1])', '[z1]'),
        ('[y] - ({Intercept} + {x}*[x] + {z1}*[z1])', '[z2]'),
    ],
    data,
    specification_name='GMM IV2SLS',
    debug=True,
)

print(fit_gmm)


"""
====================================================================
Generalized Method of Moments Results
GMM IV2SLS
====================================================================

Dep. Variable: {Not Applicable in GMM}

Date:               Oct 19, 2022    Converged:                  True
Time:                   20:44:31    No. Iters:                     1
Nobs:                        300    Objective:            4.4157e-20
No. Moments:                   3    Cov Type:               SANDWICH
No. Params:                    3    Model Elapsed:             0.00s
Over Identified:           False    Fit Elapsed:               0.48s
Method:                 ONE_STEP    

==================================================================
              coef        std err      t   p>|t|  [0.025,   0.975]
------------------------------------------------------------------
Intercept    1.114  ****  0.06306  17.66  <0.001    0.9895   1.238
x           0.4048  ****  0.05487   7.38  <0.001    0.2968  0.5128
z1         0.05824        0.07331   0.79   0.428  -0.08603  0.2025
==================================================================

Moment 0:   ([y] - ({Intercept} + {x}*[x] + {z1}*[z1]))
Moment 1:   ([y] - ({Intercept} + {x}*[x] + {z1}*[z1])) * ([z1])
Moment 2:   ([y] - ({Intercept} + {x}*[x] + {z1}*[z1])) * ([z2])

Converged: max gradient 1.9e-10 below g_tol=1.0e-08
Used t distribution with 297 df at test level nan.

                                                  [kanly v=0.0.313]
"""

# from kanly.api import lm
# fit_iv = lm('y ~ x + z1 | z1 + z2', data)
# print(fit_iv)
