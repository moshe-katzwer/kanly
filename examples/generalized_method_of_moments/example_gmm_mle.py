import numpy as np
import pandas as pd

from kanly.api import gmm_mle, glm, compare_results

n = 5_000
np.random.seed(10)

data = pd.DataFrame()

data['x'] = np.random.rand(n)
data['prob'] = .3 + .25 * data.x
data['y'] = np.random.rand(n) < data.prob

fit_gmm = gmm_mle('[y]*np.log({Intercept}+{x}*[x]) + (1-[y])*np.log(1-{Intercept}-{x}*[x])',
                  data, is_log_llf=True, start_params=[.5, 0])
print(fit_gmm)

fit_glm = glm('y ~ x', data, family='binomial', link='identity', start_params=[.5, 0])

print(compare_results([fit_gmm, fit_glm]))

'''
============================================================
Regression Summary Table
============================================================
                          (0)       (1)
------------------------------------------------------------
Intercept               0.319     0.319
                     (0.0133)  (0.0133)


x                       0.225     0.225
                     (0.0237)  (0.0237)
============================================================
Outcome:                              y
No. Obs.                 5000      5000
R-squared: :                           
R-squared Adj.:                        
Pseudo R-squared: :              0.0129
Method:              ONE_STEP       GLM
Weights:                 None         -
Df Residuals:            4998      4998
Df Model:                   2         2
Covariance Type:     SANDWICH       HC1
------------------------------------------------------------
(0)    y ~ Intercept + x
(1)    y ~ x
============================================================
                                          [kanly, v=0.0.355]
'''