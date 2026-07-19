"""
GLSAR vs OLS when errors follow AR(p).

Demonstrates :func:`~kanly.api.glsar` on simulated data with **serially correlated
errors**:

1. Draw innovations ``e`` from AR(2) via :func:`~kanly.api.simulate_sarima`
   (``ar=[0.6, 0.2]``, ``sigma2=3``).
2. Build ``y = 1.6 + 0·x0 + 1·x1 + e`` with i.i.d. regressors ``x0``, ``x1``.
3. Fit OLS, GLSAR(1), and GLSAR(2) and compare with :func:`~kanly.api.compare_results`.

**Takeaway:** While OLS is consistent for ``x0`` and ``x1`` when errors are AR —
coefficients are biased and R² is low in small samples. GLSAR(1) and GLSAR(2) mostly
 recover the true slopes (Intercept ≈ 1.6, ``x0`` ≈ 0, ``x1`` ≈ 1) 
 and raise R² substantially.
 
With AR(2) errors, ``glsar(..., nlags=2)`` is the correctly specified order;
``nlags=1`` still improves on OLS but may leave mild misspecification in the
AR structure.

Default whitening is **Prais-Winsten** (``full_information=True``). Pass
``full_information=False`` for Cochrane-Orcutt (closer to statsmodels ``GLSAR``).

Run::

    python examples/regression/linear_models/example_glsar.py
"""
from kanly.api import simulate_sarima
import pandas as pd
import numpy as np
from kanly.api import lm, glsar, compare_results

np.random.seed(1)
n = 500

# AR(2) error process (true coefficients known for compare_results reference column)
ar = [.6, .2]
e = simulate_sarima(n=n, seed=0, burnin=500, ar=ar, sigma2=3.0)

# Linear model: Intercept=1.6, x0 coef=0, x1 coef=1
k = 2
X = np.random.randn(n, k)
y = 1.6 + X.dot(np.arange(k)) + e

df = pd.DataFrame(X, columns=[f'x{j}' for j in range(k)])
df['y'] = y

fit_ols = lm('y ~ x0 + x1', df, specification_name='OLS')
fit_glsar1 = glsar('y ~ x0 + x1', df, nlags=1, specification_name='GLSAR[1]')
fit_glsar2 = glsar('y ~ x0 + x1', df, nlags=2, specification_name='GLSAR[2]')
fit_glsar3 = glsar('y ~ x0 + x1', df, nlags=3, specification_name='GLSAR[3]')

# Compare GLSAR results to OLS
true_param_values = {'Intercept': 1.6, 'x0': 0, 'x1': 1.0}
print(compare_results(
    [fit_ols, fit_glsar1, fit_glsar2, fit_glsar3],
    ref_param_values=true_param_values)
)

# Treat asymptotic distribution as "Bayesian posterior"
# look at MSE relative to "truth"
mse_tbl = []
for key, truth in true_param_values.items():
    for fit in [fit_ols, fit_glsar1, fit_glsar2, fit_glsar3]:
        mse_tbl.append({'parameter': key, 'model': fit.specification_name,
                        'mse': (fit.params[key] - truth)**2 + fit.bse[key]**2
        })

print(pd.DataFrame(mse_tbl).pivot(index='model', columns='parameter'))

"""
═══════════════════════════════════════════════════════════════════════════════
Regression Summary Table
═══════════════════════════════════════════════════════════════════════════════
                          (0)        (1)        (2)        (3)   |    Reference
───────────────────────────────────────────────────────────────────────────────
Intercept               1.589      1.594      1.594      1.593   |        1.600
                      (0.118)    (0.296)    (0.385)    (0.391)   |             


x0                      0.124      0.019      0.016      0.020   |    0.000e+00
                      (0.121)    (0.070)    (0.071)    (0.072)   |             


x1                      1.175      1.103      1.104      1.103   |        1.000
                      (0.120)    (0.066)    (0.068)    (0.068)   |             
═══════════════════════════════════════════════════════════════════════════════
Model:                    LLS        LLS        LLS        LLS   |             
Outcome:                    y          y          y          y   |             
No. Obs.                  500        500        500        500   |             
R-squared:             0.1615     0.3621     0.3491     0.3468   |            -
R-squared Adj.:        0.1582     0.3596     0.3465     0.3441   |             
Pseudo R-squared:                                                |             
Method:                   OLS   GLSAR[1]   GLSAR[2]   GLSAR[3]   |             
Weights:                    -       None       None       None   |             
Df Residuals:             497        497        497        497   |             
Df Model:                   2          2          2          2   |             
Covariance Type:    OLS_SMALL  OLS_SMALL  OLS_SMALL  OLS_SMALL   |             
───────────────────────────────────────────────────────────────────────────────
(0)  "OLS"
(1)  "GLSAR[1]"
(2)  "GLSAR[2]"
(3)  "GLSAR[3]"
═══════════════════════════════════════════════════════════════════════════════
                                                            [kanly, v=0.0.1031]

                mse                    
parameter Intercept        x0        x1
model                                  
GLSAR[1]   0.087886  0.005201  0.014955
GLSAR[2]   0.148025  0.005377  0.015387
GLSAR[3]   0.152989  0.005521  0.015193
OLS        0.014105  0.030066  0.044909
"""
