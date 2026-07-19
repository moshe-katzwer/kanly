import numpy as np
import pandas as pd
from kanly.api import elastic_net, lm

np.random.seed(0)
n = 500
df = pd.DataFrame({
    'x': np.random.randn(n),
    'w': np.random.rand(n),
    'geo': np.random.randint(0, 100, n).astype(str),
    'e': 3 * np.random.randn(n),
})
df['y'] = 1.56 + 2.6 * df.x ** 2 - 3 * np.log(1 + df.w) \
          + (df.geo == 1) * 4 + (df.geo == 2) * -3 \
          + (df.geo == 7) * 1.2 + df.e

fit_en, fit_ols = elastic_net('y ~ poly(x,4) + np.log(1+w) + C(geo)', df, alpha=.01, refit=True)
print(fit_en)
print(fit_ols)

"""
================================================================
Penalized Linear Model Results
================================================================

Dep. Variable: y

Date:             Mar 29, 2023    alpha:               1.000e-02
Time:                 20:43:44    l1_ratio:            1.000e+00
Method:                  lasso    fit_intercept:            True
Nobs:                      500    normalize:                True
Params:                    106    positive:                False
Score:                  0.6030    scaled:                  False
Weights:                     -    Model Time:              0.04s
                                  Fit Time:                1.39s

========================
                    coef
------------------------
Intercept         0.9403
x                      0
I(x**2)             2.44
I(x**3)                0
I(x**4)                0
I(np.log(1+w))    -1.003
C(geo)[0]       -0.05768
C(geo)[1]              0
C(geo)[10]             0
C(geo)[11]             0
C(geo)[12]             0
C(geo)[13]             0
C(geo)[14]             0
C(geo)[15]             0
C(geo)[16]             0
C(geo)[17]             0
C(geo)[18]             0
C(geo)[19]        0.2802
C(geo)[2]              0
C(geo)[20]       -0.6332
C(geo)[21]             0
C(geo)[22]             0
C(geo)[23]             0
C(geo)[24]             0
C(geo)[25]             0
C(geo)[26]             0
C(geo)[27]             0
C(geo)[28]             0
C(geo)[29]             0
C(geo)[3]              0
C(geo)[30]             0
C(geo)[31]             0
C(geo)[32]             0
C(geo)[33]             0
C(geo)[34]             0
C(geo)[35]             0
C(geo)[36]             0
C(geo)[37]             0
C(geo)[38]             0
C(geo)[39]             0
C(geo)[4]              0
C(geo)[40]             0
C(geo)[41]             0
C(geo)[42]             0
C(geo)[43]             0
C(geo)[44]             0
C(geo)[45]             0
C(geo)[46]             0
C(geo)[47]             0
C(geo)[48]       -0.7329
C(geo)[49]        0.2575
C(geo)[5]              0
C(geo)[50]             0
C(geo)[51]             0
C(geo)[52]             0
C(geo)[53]             0
C(geo)[54]             0
C(geo)[55]             0
C(geo)[56]             0
C(geo)[57]             0
C(geo)[58]        -2.092
C(geo)[59]             0
C(geo)[6]              0
C(geo)[60]             0
C(geo)[61]             0
C(geo)[62]             0
C(geo)[63]             0
C(geo)[64]             0
C(geo)[65]             0
C(geo)[66]             0
C(geo)[67]             0
C(geo)[68]             0
C(geo)[69]             0
C(geo)[7]              0
C(geo)[70]             0
C(geo)[71]             0
C(geo)[72]       -0.0934
C(geo)[73]             0
C(geo)[74]             0
C(geo)[75]             0
C(geo)[76]             0
C(geo)[77]             0
C(geo)[78]             0
C(geo)[79]       -0.1084
C(geo)[8]              0
C(geo)[80]             0
C(geo)[81]             0
C(geo)[82]             0
C(geo)[83]             0
C(geo)[84]             0
C(geo)[85]             0
C(geo)[86]             0
C(geo)[87]             0
C(geo)[88]             0
C(geo)[89]             0
C(geo)[9]              0
C(geo)[90]             0
C(geo)[91]             0
C(geo)[92]             0
C(geo)[93]             0
C(geo)[94]             0
C(geo)[95]             0
C(geo)[96]             0
C(geo)[97]             0
C(geo)[98]             0
C(geo)[99]        -1.354
========================

formula:  y ~ poly(x,4) + np.log(1+w) + C(geo)


                                              [kanly v=0.0.384]

==========================================================================
Linear Model Results
==========================================================================

Dep. Variable:   y

Date:                  Mar 29, 2023    No. Obs.                        500
Time:                      20:43:44    Df Residuals:                   488
Model Elapsed:               0.00 s    Df Model:                        11
Fit Elapsed:                 0.01 s    R-squared:                   0.6295
Cov Elapsed:                 0.00 s    Adj. R-squared:              0.6211
Method:                         OLS    F-statistic:                  75.37
Weights:                          -    Prob (F-statistic):           <.001
Intercept:                     True    Log-Likelihood:          -1228.3794
Implicit Intercept:            True    AIC:                        2480.76
Covariance Type:          OLS_SMALL    BIC:                        2531.33
                                       Cond. No.:                 1.84e+01

=======================================================================
                  coef        std err      t   p>|t| [0.025,     0.975]
-----------------------------------------------------------------------
Intercept        1.211  ****    0.301   4.02  <0.001   0.6192     1.802
I(x**2)          2.635  ****  0.09531  27.64  <0.001    2.447     2.822
I(np.log(1+w))  -1.937  **     0.6452  -3.00   0.003   -3.205   -0.6695
C(geo)[0]       -2.951  *        1.44  -2.05   0.041   -5.781   -0.1215
C(geo)[19]       2.917  *       1.438   2.03   0.043  0.09126     5.743
C(geo)[20]      -4.281  *       2.025  -2.11   0.035    -8.26   -0.3013
C(geo)[48]      -4.284  *       2.034  -2.11   0.036   -8.281   -0.2874
C(geo)[49]       2.712           1.44   1.88    0.06  -0.1177     5.542
C(geo)[58]      -4.604  **      1.436  -3.21  <0.001   -7.426    -1.783
C(geo)[72]      -1.852  *      0.9142  -2.03   0.043   -3.649  -0.05615
C(geo)[79]      -2.937          1.657  -1.77   0.077   -6.192    0.3178
C(geo)[99]      -3.532  **      1.176  -3.00   0.003   -5.842    -1.221
=======================================================================

formula:  y ~ Intercept + I(x**2) + I(np.log(1+w)) + C(geo)[0] +
          C(geo)[19] + C(geo)[20] + C(geo)[48] + C(geo)[49]
          + C(geo)[58] + C(geo)[72] + C(geo)[79] +
          C(geo)[99]

Used t distribution with 488 df at test level 0.0500.

                                                        [kanly v=0.0.384]
"""