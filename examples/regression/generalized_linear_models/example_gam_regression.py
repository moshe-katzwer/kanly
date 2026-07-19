"""Poisson GAM example: B-spline smooth of x2 with tunable roughness penalty.

Compares unpenalized spline GLM (``penalty=0``) vs penalized GAM fits.
See ``kanly/regression/generalized_linear_models/README.md`` (GAM section).
"""
import numpy as np
import pandas as pd

from kanly.api import glm, gam
import matplotlib.pyplot as plt

np.random.seed(0)
n2 = 300
x2 = np.random.randn(n2)
x2.sort()
y2 = np.exp(-4.5 + .5 * x2 + .05 * x2 ** 2 - .5 * np.sin(x2 * 4) + .4 * np.random.randn(n2))
data2 = pd.DataFrame(dict(x2=x2, y2=y2))
df_2 = 20

fit_glm = gam(
    'y2 ~ x2', data2,
    df=dict(x2=df_2),
    penalty=dict(x2=0.0),  # no penalization - just GLM
    family='poisson')

fit_gam1 = gam(
    'y2 ~ x2', data2,
    penalty=dict(x2=.0010),
    df=dict(x2=df_2), family='poisson', cov_type='nonrobust')
print(fit_gam1)

fit_gam2 = gam(
    'y2 ~ x2', data2,
    penalty=dict(x2=.05),
    df=dict(x2=df_2), family='poisson', cov_type='nonrobust')
print(fit_gam2)

plt.figure(dpi=150)
plt.scatter(x2, y2, alpha=.3, c='y')
plt.plot(x2, fit_glm.fittedvalues, lw=3, label=f'glm [edf={fit_glm.edf.sum():.1f}]')
plt.plot(x2, fit_gam1.fittedvalues, lw=3,
         label=f'gam(penalization={.0001:.1e}) [edf={fit_gam1.edf.sum():.1f}]')
plt.plot(x2, fit_gam2.fittedvalues, lw=3,
         label=f'gam(penalization={.05:.1e}) [edf={fit_gam2.edf.sum():.1f}]')
plt.legend(loc='best')
plt.title("Generalized Additive Model Example\nPoisson")
plt.yscale('log')
plt.ylabel('log(y)')
plt.show()

"""
══════════════════════════════════════════════════════════════════
Generalized Additive Model Results
══════════════════════════════════════════════════════════════════

Dep. Variable: y2

Date:              Jun 04, 2026    Deviance:            2.6314e+00
Time:                  13:34:06    Pearson chi2:            3.0423
Family:                 POISSON    Scale:               1.0000e+00
Link:                       LOG    Converged:                 True
Var Weights:                  -    Iterations:                   6
Method:                    IRLS    Rel. Err.:             4.22e-10
Nobs:                       300    Abs. Err.:             5.40e-11
Df Residuals:               279    Cov. Type:            NONROBUST
Df Model:                  3.20    Model Elapsed:            0.00s
Log-Likelihood:     -2.0930e+01    Fit Elapsed:              0.00s
Pseudo Rsq:              0.0218    Cov Elapsed:              0.00s

════════════════════════════════════════════════════════
             coef   std err     t  p>|t| [0.025,  0.975]
────────────────────────────────────────────────────────
Intercept  -5.885     4.033 -1.46  0.146   -13.82  2.053
bs_x2_1    0.2697     1.292  0.21  0.835   -2.274  2.814
bs_x2_2    0.6126     2.692  0.23  0.820   -4.686  5.911
bs_x2_3     1.016     3.467  0.29  0.770    -5.81  7.841
bs_x2_4     1.196     3.738  0.32  0.749   -6.163  8.554
bs_x2_5     1.317     3.879  0.34  0.734   -6.318  8.952
bs_x2_6     1.422     3.982  0.36  0.721   -6.417  9.261
bs_x2_7     1.499     4.046  0.37  0.711   -6.466  9.463
bs_x2_8      1.57     4.098  0.38  0.702   -6.497  9.636
bs_x2_9     1.626     4.133  0.39  0.694    -6.51  9.763
bs_x2_10    1.676     4.154  0.40  0.687   -6.501  9.852
bs_x2_11    1.727     4.164  0.41  0.679    -6.47  9.925
bs_x2_12    1.776     4.167  0.43  0.670   -6.426  9.978
bs_x2_13    1.841     4.165  0.44  0.659   -6.359  10.04
bs_x2_14    1.912     4.161  0.46  0.646   -6.279   10.1
bs_x2_15    2.004     4.153  0.48  0.630   -6.172  10.18
bs_x2_16    2.103     4.146  0.51  0.612   -6.058  10.26
bs_x2_17    2.237     4.143  0.54  0.590   -5.918  10.39
bs_x2_18    2.245     4.165  0.54  0.590   -5.954  10.44
bs_x2_19    2.239     4.248  0.53  0.599   -6.124   10.6
bs_x2_20    2.237     4.332  0.52  0.606    -6.29  10.76
════════════════════════════════════════════════════════

formula:  y2 ~ bs(x2, degtee=3, df=20, include_intercept=False)

fit_intercept = True
Link Function: g(x) = log(x)
Used t distribution with 279 df at test level 0.0500.

                                               [kanly v=0.0.1046]
"""
