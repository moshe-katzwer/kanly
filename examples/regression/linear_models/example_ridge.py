import numpy as np
import pandas as pd

from kanly.api import lm, compare_results, elastic_net

n = 100
np.random.seed(0)
df = pd.DataFrame({
    'x': 2 + np.random.randn(n),
    'z': np.random.randn(n) * 5,
})
df['y'] = 1.2 - 0.3 * df['x'] + .2 * np.random.randn(n)

fit_ridge = lm('y ~ x + z', df, debug=False, ridge_kwds={'alpha': .6, 'normalize': True})
fit_en = elastic_net('y ~ x + z', df, debug=False, alpha=.6, l1_ratio=0, normalize=True, ftol=1e-15)
fit_ols = lm('y ~ x + z', df, debug=False)

print(compare_results([fit_ridge, fit_en, fit_ols]))

"""
════════════════════════════════════════════════════════════
Regression Summary Table
════════════════════════════════════════════════════════════
                            (0)          (1)        (2)
────────────────────────────────────────────────────────────
Intercept                 0.938        0.938      1.160
                        (0.042)        (nan)    (0.044)


x                        -0.178       -0.178     -0.287
                        (0.017)        (nan)    (0.019)


z                    -3.656e-04   -3.656e-04      0.002
                        (0.003)        (nan)    (0.004)
════════════════════════════════════════════════════════════
Model:                      LLS        EN_sk        LLS
Outcome:                      y            y          y
No. Obs.                    100          100        100
R-squared:               0.5987       0.5987     0.6976
R-squared Adj.:          0.5905          nan     0.6914
Pseudo R-squared:                                      
Method:                   RIDGE   RIDGE (CD)        OLS
Weights:                      -            -          -
Df Residuals:                97          NaN         97
Df Model:                     2          NaN          2
Covariance Type:      OLS_SMALL          N/A  OLS_SMALL
════════════════════════════════════════════════════════════
                                         [kanly, v=0.0.1046]
"""
