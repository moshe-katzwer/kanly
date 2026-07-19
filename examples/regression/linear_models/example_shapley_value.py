"""Shapley R² decomposition: exact enumeration vs permutation sampling.

Demonstrates:
- ``fit.shapley_value()`` on a formula fit (sparse_terms = RHS pieces like ``x6``, ``C(grp)``)
- ``shapley_value('y ~ ...', df)`` and tuple ``('y', [sparse_terms])`` standalone APIs
- ``LM`` matrix API with column-level term names
- ``return_full=True`` for timing and regression-count metadata
"""

import numpy as np
import pandas as pd

from kanly.api import lm, LM, shapley_value

# --- Synthetic data: 12 continuous predictors + one categorical term ---
n = 100_000
np.random.seed(0)
p = 12
X = np.random.randn(n, p)
X = X.dot(np.random.randint(0, 3, (p, p)))  # induce collinearity
df = pd.DataFrame(X, columns=[f'x{j}' for j in range(p)])
df['grp'] = np.random.randint(0, 12, n)
y = 1.2 - 0.3 * df['x6'] + 1.2 * df['x2'] - 5.6 * (df.grp == 1) + np.random.randn(n)
df['y'] = y

# Formula fit: Shapley players are formula *sparse_terms* (13 here: x0..x11 + C(grp))
fit = lm('y ~ ' + '+'.join([f'x{j}' for j in range(p)]) + ' + C(grp)', df)

# Exact mode: enumerate all 2^p - 1 non-empty term subsets (8191 subset R² evals for p=13)
print('\nExact Shapley Values:')
result = fit.shapley_value(return_full=True)
print(f"{result['shapley_values'].round(5)}"
      f"\ntime={result['time_elapsed']:.4f}s\nnum_regressions={result['num_regressions']}")

# Permutation mode: Monte Carlo over random term orderings (much faster for large p)
sample = 50
print(f'\nShapley values with sampling only {sample} permutations:')
result = fit.shapley_value(sample=sample, return_full=True)
print(f"{result['shapley_values'].round(5)}"
      f"\ntime={result['time_elapsed']:.4f}s\nnum_regressions={result['num_regressions']}")

print('\nfit:')
print(fit)

# Standalone API without a pre-existing fit object
print('\nStandalone shapley_value (formula string):')
tab = shapley_value('y ~ x2 + x6 + C(grp)', df)
print(tab.round(5))

print('\nStandalone shapley_value (tuple specification):')
tab = shapley_value(('y', ['x2', 'x6', 'C(grp)']), df)
print(tab.round(5))

"""
Exact Shapley Values:
        shapley_value      pct
x2            0.36754  0.38343
C(grp)        0.10162  0.10601
x9            0.06636  0.06923
x3            0.06415  0.06693
x5            0.05925  0.06181
x10           0.05513  0.05751
x7            0.05453  0.05689
x11           0.05048  0.05266
x8            0.04527  0.04723
x6            0.03322  0.03466
x1            0.02345  0.02446
x4            0.01921  0.02004
x0            0.01836  0.01915
time=0.3110s
num_regressions=8191

Shapley values with sampling only 50 permutations:
        shapley_value      pct
x2            0.41019  0.42792
C(grp)        0.10171  0.10610
x9            0.07207  0.07518
x10           0.06080  0.06343
x7            0.05555  0.05795
x5            0.04830  0.05039
x6            0.04011  0.04184
x3            0.03896  0.04064
x8            0.03781  0.03944
x11           0.03453  0.03602
x1            0.02595  0.02707
x4            0.01955  0.02040
x0            0.01304  0.01360
time=0.0173s
num_regressions=489

fit:
════════════════════════════════════════════════════════════════════════════
Linear Model Results
════════════════════════════════════════════════════════════════════════════

Dep. Variable:   y

Date:                  May 20, 2026    No. Obs.                     100000
Time:                      08:53:54    Df Residuals:                 99976
Model Elapsed:               0.08 s    Df Model:                        23
Fit Elapsed:                 0.07 s    R-squared:                   0.9586
Cov Elapsed:                 0.00 s    Adj. R-squared:              0.9586
Method:                         OLS    F-statistic:              1.006e+05
L2 Penalty:                    None    Prob (F-statistic):           <.001
Weights:                          -    Log-Likelihood:        -141617.3163
Intercept:                     True    AIC:                      283282.63
Implicit Intercept:           False    BIC:                      283510.94
Covariance Type:          OLS_SMALL    scale:                   9.9472e-01


════════════════════════════════════════════════════════════════════════════
                  coef         std err       t   p>|t|   [0.025,      0.975]
────────────────────────────────────────────────────────────────────────────
Intercept        1.214  ****     0.011  110.36  <0.001      1.192      1.235
x0            0.003318         0.00273    1.22   0.224  -0.002033    0.00867
x1              0.0046        0.004183    1.10   0.271  -0.003598     0.0128
x2               1.207  ****  0.003391  356.04  <0.001      1.201      1.214
x3           -0.006589        0.004166   -1.58   0.114   -0.01475   0.001576
x4           -0.003483        0.004529   -0.77   0.442   -0.01236   0.005394
x5            0.007872  *     0.004532    1.74   0.082  -0.001011    0.01676
x6             -0.3018  ****   0.00211 -143.03  <0.001    -0.3059    -0.2976
x7            0.004045        0.003197    1.27   0.206  -0.002221    0.01031
x8           -0.003246         0.00255   -1.27   0.203  -0.008244   0.001753
x9           -0.001728        0.002469   -0.70   0.484  -0.006568   0.003112
x10         -0.0002959        0.003126   -0.09   0.925  -0.006424   0.005832
x11           -0.01043  *     0.005743   -1.82   0.069   -0.02168  0.0008283
C(grp)[1]       -5.619  ****   0.01543 -364.13  <0.001     -5.649     -5.589
C(grp)[2]     -0.02068         0.01552   -1.33   0.183    -0.0511   0.009748
C(grp)[3]     -0.01864         0.01548   -1.20   0.229   -0.04899    0.01171
C(grp)[4]     0.005258         0.01553    0.34   0.735   -0.02518    0.03569
C(grp)[5]     -0.01047         0.01547   -0.68   0.498   -0.04078    0.01984
C(grp)[6]    -0.009038         0.01548   -0.58   0.559   -0.03939    0.02131
C(grp)[7]     -0.01021         0.01556   -0.66   0.512   -0.04071    0.02029
C(grp)[8]     -0.02363         0.01549   -1.53   0.127   -0.05398   0.006731
C(grp)[9]      -0.0363  **     0.01554   -2.34    0.02   -0.06676  -0.005833
C(grp)[10]    -0.01972         0.01551   -1.27   0.204   -0.05011    0.01068
C(grp)[11]    -0.02802  *      0.01547   -1.81    0.07   -0.05835   0.002301
════════════════════════════════════════════════════════════════════════════
Omnibus              1.893    Durbin-Watson:       1.997
Prob(Omnibus):       0.388    Skew:                0.003
Jarque-Bera(JB):     1.889    Kurtosis:            2.980
Prob(JB)             0.389    Cond. No.:          77.683
════════════════════════════════════════════════════════════════════════════

formula:  y ~ x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11 + C(grp)

Used t distribution with 99976 df at test level 0.0500.

                                                         [kanly v=0.0.1026]
"""

# Matrix API: each column is one Shapley player (no C(grp) expansion)
fit = LM(y, X, add_constant=True)
result = fit.shapley_value(return_full=True)
print(f"{result['shapley_values'].round(5)}"
      f"\ntime={result['time_elapsed']:.4f}s\nnum_regressions={result['num_regressions']}")

print('\nfit:')
print(fit)

"""
       shapley_value      pct
<x3>         0.36754  0.42891
<x10>        0.06640  0.07749
<x4>         0.06416  0.07487
<x6>         0.05921  0.06909
<x11>        0.05508  0.06428
<x8>         0.05454  0.06364
<x12>        0.05054  0.05898
<x9>         0.04520  0.05274
<x7>         0.03325  0.03880
<x2>         0.02344  0.02735
<x5>         0.01921  0.02241
<x1>         0.01836  0.02143
time=0.2235s
num_regressions=4095

fit:
══════════════════════════════════════════════════════════════════════════
Linear Model Results
══════════════════════════════════════════════════════════════════════════

Dep. Variable:   <y>

Date:                  May 20, 2026    No. Obs.                     100000
Time:                      11:51:39    Df Residuals:                 99987
Model Elapsed:               0.00 s    Df Model:                        12
Fit Elapsed:                 0.03 s    R-squared:                   0.8569
Cov Elapsed:                 0.00 s    Adj. R-squared:              0.8569
Method:                         OLS    F-statistic:               49898.85
L2 Penalty:                    None    Prob (F-statistic):           <.001
Weights:                          -    Log-Likelihood:        -203586.3359
Intercept:                     True    AIC:                      407198.67
Implicit Intercept:            True    BIC:                      407322.34
Covariance Type:          OLS_SMALL    scale:                   3.4349e+00
                                       

══════════════════════════════════════════════════════════════════════════
                 coef         std err       t   p>|t|   [0.025,     0.975]
──────────────────────────────────────────────────────────────────────────
Intercept      0.7225  ****  0.005861  123.26  <0.001      0.711     0.734
<x1>        0.0006314        0.005074    0.12   0.901  -0.009313   0.01058
<x2>         0.002521        0.007772    0.32   0.746   -0.01271   0.01775
<x3>            1.205  ****  0.006301  191.27  <0.001      1.193     1.218
<x4>        -0.002284        0.007741   -0.30   0.768   -0.01746   0.01289
<x5>       -0.0004093        0.008415   -0.05   0.961    -0.0169   0.01608
<x6>         0.005113        0.008422    0.61   0.544   -0.01139   0.02162
<x7>           -0.302  ****   0.00392  -77.02  <0.001    -0.3096   -0.2943
<x8>         0.003868        0.005941    0.65   0.515  -0.007777   0.01551
<x9>        -0.003828        0.004739   -0.81   0.419   -0.01312   0.00546
<x10>       0.0003153        0.004588    0.07   0.945  -0.008678  0.009308
<x11>       -0.009748  *     0.005809   -1.68   0.093   -0.02113  0.001638
<x12>      -0.0003582         0.01067   -0.03   0.973   -0.02127   0.02056
══════════════════════════════════════════════════════════════════════════
Omnibus              33773.369    Durbin-Watson:           1.999
Prob(Omnibus):           0.000    Skew:                   -1.781
Jarque-Bera(JB):    102617.716    Kurtosis:                6.456
Prob(JB)                 0.000    Cond. No.:              74.850
══════════════════════════════════════════════════════════════════════════

formula:  <y> ~ Intercept + <x1> + <x2> + <x3> + <x4> + <x5> + <x6> +
          <x7> + <x8> + <x9> + <x10> + <x11> + <x12>

Used t distribution with 99987 df at test level 0.0500.

                                                       [kanly v=0.0.1026]
"""
