import numpy as np
import pandas as pd

from kanly.api import gmm

n = 25_000
np.random.seed(10)

data = pd.DataFrame()
data['z1'] = np.random.randn(n)
data['z2'] = np.random.randn(n)
data['e'] = np.random.randn(n)
data['x'] = .5 * data.z1 + 1.2 * data.z2 + .5 * data.e
data['y'] = np.exp(-.1 + .25 * data.x + .1 * data.z1) + data.e

# These moment conditions aren't "optimal", just chosen to represent
# some weird but valid moments.  The system is also over-identified.
fit_gmm = gmm(
    [
         '[y] - np.exp({Intercept} + {x}*[x] + {z1}*[z1])',
        ('[y] - np.exp({Intercept} + {x}*[x] + {z1}*[z1])', '[z1**3]'),
        ('[y] - np.exp({Intercept} + {x}*[x] + {z1}*[z1])', '[z2]'),
        ('[y] - np.exp({Intercept} + {x}*[x] + {z1}*[z1])', '[z2**2] + [z1]'),
        ('[y] - np.exp({Intercept} + {x}*[x] + {z1}*[z1])', '[z2**2]'),
    ],
    data,
    debug=True,
    specification_name='Nonlinear GMM IV Example',
    method='iterative',
    cov_type='bootstrap', cov_kwds={'n_samples': 100}
)

print(fit_gmm)

"""
=====================================================================
Generalized Method of Moments Results
=====================================================================

Dep. Variable: {Not Applicable in GMM}

Date:             Oct 19, 2022    Converged:                True
Time:                 12:37:10    No. Iters:                   8
Nobs:                    15000    Objective:          1.0430e-04
No. Moments:                 5    Cov Type:            BOOTSTRAP
No. Params:                  3    Model Elapsed:           0.01s
Method:              ITERATIVE    Fit Elapsed:             1.16s

=====================================================================
               coef         std err      t   p>|t| [0.025,     0.975]
---------------------------------------------------------------------
Intercept  -0.08794  ****  0.008482 -10.37  <0.001  -0.1046  -0.07131
x            0.2432  ****  0.006143  39.60  <0.001   0.2312    0.2553
z1           0.0989  ****  0.008585  11.52  <0.001  0.08207    0.1157
=====================================================================

Moment 0:   ([y] - np.exp({Intercept} + {x}*[x] + {z1}*[z1]))
Moment 1:   ([y] - np.exp({Intercept} + {x}*[x] + {z1}*[z1])) * ([z1**3])
Moment 2:   ([y] - np.exp({Intercept} + {x}*[x] + {z1}*[z1])) * ([z2])
Moment 3:   ([y] - np.exp({Intercept} + {x}*[x] + {z1}*[z1])) * ([z2**2] + [z1])
Moment 4:   ([y] - np.exp({Intercept} + {x}*[x] + {z1}*[z1])) * ([z2]) * ([z2])

Converged: predicted decrease in objective_function too close to zero
Used t distribution with 14997 df at test level nan.
Did 100 Bayesian bootstrap repetitions, alpha=1.000.

                                                   [kanly v=0.0.311]
"""
