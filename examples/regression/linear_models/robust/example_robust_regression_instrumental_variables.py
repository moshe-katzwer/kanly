# from kanly.api import rlm, lm, compare_fits
# import pandas as pd
# import numpy as np
#
# n = 50
# np.random.seed(0)
# df = pd.DataFrame({
#     'x': np.random.rand(n),
#     'e': np.random.randn(n),
# })
#
# df.loc[np.random.choice(df.index, 5, replace=False), 'e'] += 15
# df['y'] = 1.2 - .8 * df['x'] + df['e']
#
# fit_rlm = rlm('y ~ x', df, cov_type='BOOTSTRAP', debug=True)
# print(fit_rlm)
#
# fit_ols_boot = lm('y ~ x', df, cov_type='BOOTSTRAP')
#
# print(compare_fits([fit_ols_boot, fit_rlm],
#                    fit_titles=['ols', 'rlm'],
#                    ref_param_values={'Intercept': 1.2, 'x': -.8}
#                    )
#       )
#
# """
# ========================================================================
# Robust Linear Model (M-estimation)
# ========================================================================
#
# Dep. Variable: y
#
# dep. variable:                   y    df resid:                       48
# weights:                         -    df model:                        1
# nobs:                           50    use_t:                        True
# model:                         rlm    covariance type:    BOOTSTRAP(100)
# method:                       IRLS    scale (est):                 0.982
# norm:                       HuberT    cost:                   1.2038e+02
#
# ==============================================================
#              coef       std err     t  p>|t| [0.025,    0.975]
# --------------------------------------------------------------
# Intercept   1.366  ★★    0.4564  2.99  0.004    0.448    2.283
# x          -1.289        0.6847 -1.88  0.066   -2.666  0.08775
# ==============================================================
#
# formula:  y ~ x
#
# num_iters:           9
# max_iter:           50
# tol:          0.000001
# error:             0.0
# converged:        True
#
#
#                                     [kanly package by moshe, v=0.0.275]
#
#
# ==================================================================
# Regression Summary Table
# ==================================================================
#                                 ols             rlm   |  Reference
# ------------------------------------------------------------------
# Intercept                      3.47            1.37   |        1.2
#                              (1.82)         (0.456)   |
#
#
# x                             -2.55           -1.29   |       -0.8
#                              (2.95)         (0.685)   |
# ==================================================================
# Outcome:                          y               y   |
# No. Obs.                         50              50   |
# R-squared: :                 0.0002                   |
# R-squared Adj.:              0.0002                   |
# Pseudo R-squared: :                          0.0201   |
# Method:                         OLS            IRLS   |
# Weights:                          -               -   |
# Df Residuals:                    48              48   |
# Df Model:                         1               1   |
# Covariance Type:     BOOTSTRAP(100)  BOOTSTRAP(100)   |
# ------------------------------------------------------------------
# ols      y ~ x
# rlm      y ~ Intercept + x
# ==================================================================
#                          [kanly package by moshe, v=0.0.275]
# """
