import numpy as np
import pandas as pd

from kanly.api import elastic_net

n = 10_000
k = 200
np.random.seed(0)

df = pd.DataFrame(index=range(n))
X = np.hstack([np.random.randn(n, 1) for i in range(k)])
X = np.dot(X, np.random.randint(0, 5, (k, k)))

coefs = np.zeros(k)
coefs[[2, 6, 70, 30]] = 1.5

y = -2.6 + X.dot(coefs) + .3 * np.random.randn(n)

df.loc[:, ['x%d' % i for i in range(k)]] = X
df['y'] = y

formula = 'y ~ ' + ' + '.join(['x%d' % i for i in range(k)])

fit = elastic_net(formula, df, alpha=.025, l1_ratio=1.0, debug=False,
                  specification_name='example elastic net', selection='cyclic',
                  ftol=1e-12, xtol=1e-8, active_set=True, max_iter=2_000,
                  )

print(fit.summary(show_only_non_zero=True, show_formula=False))
print(f'\nrelative dual gap = {fit.dual_gap()/fit.objective_function_:.2e}')

"""
════════════════════════════════════════════════════════════════
Penalized Linear Model Results
example elastic net
════════════════════════════════════════════════════════════════

Dep. Variable: y

Date:             Jun 09, 2026    |dx|:                 2.80e-08
Time:                 09:46:39    |dF/F|                7.56e-13
Method:                  LASSO    max|subgrad|:         3.12e-05
Nobs:                    10000    alpha:                2.50e-02
Params:                    201    l1_ratio:             1.00e+00
Score:                  1.0000    fit_intercept:            True
SSR:                9.1367e+02    normalize:                True
Penalty:            5.1384e+00    positive:                False
Objective:          5.1841e+00    scaled:                  False
Weights:                     -    relaxation:                   
Converged:                True    active_set:               True
Iters:                     233    selection:              random
Max Iter:                 2000    Tolerance:            1.00e-08
                                  Model Time:              3.27s
                                  Fit Time:                0.05s

════════════════════
                coef
────────────────────
Intercept     -2.599
x2               1.5
x6             1.499
x30              1.5
x70              1.5
x88        3.027e-05
════════════════════

195 parameter estimates suppressed in output that are zero.

Converged in 233 iterations: 
	f_error = 7.6e-13 < 1.0e-10 = ftol, 
	g_error = 3.1e-05 < 1.0e-04 = gtol.

                                             [kanly v=0.0.1046]
"""
