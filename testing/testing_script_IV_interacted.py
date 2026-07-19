import numpy as np
import pandas as pd

from kanly.api import lm, compare_results

n = 2100
np.random.seed(0)
df = pd.DataFrame({
    'z': np.random.randn(n),
    'e': np.random.randn(n),
    'grp': np.random.randint(0, 12, n),
    'obs': np.random.randint(1, 6, n),
    'newbie': np.random.randint(0, 2, n),
})
df['x'] = .1 * df['z'] + .9 * df['e'] + .6 * np.random.randn(n)
df['y'] = 1.2 - 0.3 * df['x'] * (1-df['newbie']) - .6 * df['x'] * df['newbie'] + .2 * df['e'] / df['obs']

for force_iv in [True, False]:

    fit_iv = lm('y ~ x * C(newbie) + C(grp) | z * C(newbie) + C(grp) $ obs', df,
                specification_name='WLS-IV', debug=True,
                force_iv_projection=force_iv)

    print(fit_iv)

    X = fit_iv.model.exog.toarray()
    W = np.diag(fit_iv.model.weights)
    Z = fit_iv.model.instruments.toarray()
    y = fit_iv.model.endog.toarray().reshape((-1, 1))

    X_hat = Z.dot(np.linalg.inv(Z.T.dot(W).dot(Z)).dot(Z.T.dot(W).dot(X)))
    beta = np.linalg.inv(X_hat.T.dot(W).dot(X_hat)).dot(X_hat.T.dot(W).dot(y)).flatten()

    from numpy.testing import assert_array_almost_equal
    assert_array_almost_equal(beta, fit_iv.params)

'''
===============================================================================
Linear Model Results
WLS-IV
===============================================================================

Dep. Variable:   y

Date:                  Feb 13, 2023    No. Obs.                       2100
Time:                      09:27:23    Df Residuals:                  2085
Model Elapsed:               0.04 s    Df Model:                        14
Fit Elapsed:                 0.03 s    R-squared:                   0.9594
Cov Elapsed:                 0.00 s    Adj. R-squared:              0.9591
Method:                  IV (W2SLS)    F-statistic:                      -
Weights:                        obs    Prob (F-statistic):               -
Intercept:                     True    Log-Likelihood:                   -
Implicit Intercept:           False    AIC:                              -
Covariance Type:          OLS_SMALL    BIC:                              -
                                       Cond. No.:                 2.27e+01

===============================================================================
                      coef         std err       t   p>|t|   [0.025,     0.975]
-------------------------------------------------------------------------------
Intercept            1.197  ****  0.007633  156.78  <0.001      1.182     1.212
C(newbie)[1]      0.004859        0.004993    0.97   0.331  -0.004932   0.01465
C(grp)[1]         -0.01278         0.01165   -1.10   0.273   -0.03563   0.01007
C(grp)[2]          0.00831         0.01024    0.81   0.417   -0.01177   0.02839
C(grp)[3]       -0.0009654        0.009978   -0.10   0.923   -0.02053    0.0186
C(grp)[4]        -0.009575         0.01054   -0.91   0.364   -0.03024   0.01109
C(grp)[5]        -0.001403          0.0101   -0.14    0.89   -0.02122   0.01841
C(grp)[6]         -0.01222         0.01114   -1.10   0.273   -0.03407  0.009634
C(grp)[7]          0.00227         0.01043    0.22   0.828   -0.01819   0.02273
C(grp)[8]         0.006482         0.01035    0.63   0.531   -0.01382   0.02679
C(grp)[9]        -0.001091         0.01055   -0.10   0.918   -0.02178    0.0196
C(grp)[10]       -0.007122        0.009855   -0.72    0.47   -0.02645    0.0122
C(grp)[11]       -0.003078         0.01006   -0.31    0.76   -0.02281   0.01666
x:C(newbie)[0]     -0.3367  ****   0.03761   -8.95  <0.001    -0.4104   -0.2629
x:C(newbie)[1]     -0.5858  ****   0.02914  -20.10  <0.001    -0.6429   -0.5286
===============================================================================

formula:  y ~ x * C(newbie) + C(grp) | z * C(newbie) + C(grp) $ obs

Endogenous Regressors: x:C(newbie)[0], x:C(newbie)[1]

Excluded Regressors:   z:C(newbie)[0], z:C(newbie)[1]

Used t distribution with 2085 df at test level 0.0500.

                                                             [kanly v=0.0.367]
'''
