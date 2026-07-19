"""Timing comparison: kanly ``lm`` vs statsmodels ``ols`` on a large fixed-effects model.

This script fits the same OLS model with HC1 standard errors three ways:

1. **kanly** with explicit dummies: ``y ~ x1 + ... + x20 + C(g)``
2. **kanly** with absorbed fixed effects: ``y ~ x1 + ... + x20``, ``absorb='g'``
3. **statsmodels** ``smf.ols`` on the full dummy-expanded formula

On a representative run (3M rows, 20 slopes, 180 group levels → 201 regressors
with dummies), kanly finishes in roughly **13–16 seconds** while statsmodels
takes on the order of **18 minutes** (~1056 s). Coefficients, standard errors,
and *t*-statistics on the slope sparse_terms agree across all three fits.

Why kanly is faster
-------------------
**(a) Sparse design matrices.** Categorical ``C(g)`` and other sparse_terms build a
sparse ``X``. Forming ``X'X`` and ``X'y`` uses sparse structure instead of
materialising a dense ``n × k`` matrix for every algebra step.

**(b) Smaller linear algebra for β.** A common statsmodels path evaluates
``(inv(X'X) @ X') @ y``. The intermediate ``inv(X'X) @ X'`` is ``k × n`` and
**dense** even when ``X`` is sparse — expensive in both memory and multiply count.
Kanly solves ``β = inv(X'X) @ (X'y)`` where ``X'X`` is ``k × k`` and ``X'y`` is
``k × 1``, which is the standard efficient normal-equations form.

Kanly still provides the same core inference as statsmodels on ``lm`` fits —
coefficients, robust/cluster/HAC/bootstrap SEs, *t*-tests, confidence intervals,
predictions, Wald tests — plus IV, absorbed FE, FGLS, ridge, SURE, Shapley R²,
and other extensions documented in the package README.

Requirements: ``statsmodels`` (not a kanly dependency) for the comparison arm.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from kanly.api import lm, timer as kanly_timer, clear_timers

clear_timers()

# Problem size: 3M rows, 20 slopes, 180 group levels (201 regressors with C(g))
n = 3_000_000
p = 20
X = np.random.randn(n, p)
g = np.random.randint(0, 180, n)
y = X.dot(np.random.randn(p)) + np.random.randn(n)

df = pd.DataFrame(X, columns=[f'x{i+1}' for i in range(p)])
df['y'] = y
df['g'] = g

# burnin for some numba dependence
lm('y ~ x1', df)

formula_absorb = 'y ~ ' + ' + '.join([f'x{i+1}' for i in range(p)])
formula_full = formula_absorb + ' + C(g)'

print('fitting and timing kanly and statsmodels...')

# --- Timed fits (HC1 heteroskedasticity-robust SEs on all three) ---
kanly_timer('kanly')
fit_kanly = lm(formula_full, df, cov_type='HC1')
kanly_timer('kanly', display=True)

kanly_timer('kanly_absorb')
fit_kanly_absorb = lm(formula_absorb, df, absorb='g', cov_type='HC1')
kanly_timer('kanly_absorb', display=True)

kanly_timer('statsmodels')
fit_statsmodels = smf.ols(formula_full, df).fit(cov_type='HC1')
kanly_timer('statsmodels', display=True)

# --- Numerical agreement on slope coefficients (FE dummies differ in naming) ---
param_names = [f'x{i+1}' for i in range(p)]
for attribute in ['params', 'bse', 'tvalues']:
    print('\n\n', attribute, ':')
    print(
        pd.DataFrame({
            'kanly': getattr(fit_kanly, attribute)[param_names],
            'kanly_absorb': getattr(fit_kanly_absorb, attribute)[param_names],
            'statsmodels': getattr(fit_statsmodels, attribute)[param_names],
        }, index=param_names)
    )

"""
fitting and timing kanly and statsmodels...
name='kanly', elapsed=15.512027s
name='kanly_absorb', elapsed=12.565556s
name='statsmodels', elapsed=1055.766994s


 params :
        kanly  kanly_absorb  statsmodels
x1  -0.305575     -0.305575    -0.305575
x2   1.188714      1.188714     1.188714
x3  -0.279424     -0.279424    -0.279424
x4   1.050721      1.050721     1.050721
x5  -1.059178     -1.059178    -1.059178
x6   0.059970      0.059970     0.059970
x7  -0.084896     -0.084896    -0.084896
x8   1.247267      1.247267     1.247267
x9  -0.156490     -0.156490    -0.156490
x10  0.571808      0.571808     0.571808
x11 -1.240365     -1.240365    -1.240365
x12  1.052359      1.052359     1.052359
x13 -0.314501     -0.314501    -0.314501
x14  1.295816      1.295816     1.295816
x15  0.419746      0.419746     0.419746
x16  0.256979      0.256979     0.256979
x17  1.239984      1.239984     1.239984
x18  0.299124      0.299124     0.299124
x19 -1.325716     -1.325716    -1.325716
x20 -2.443112     -2.443112    -2.443112


 bse :
        kanly  kanly_absorb  statsmodels
x1   0.000577      0.000577     0.000577
x2   0.000577      0.000577     0.000577
x3   0.000578      0.000578     0.000578
x4   0.000578      0.000578     0.000578
x5   0.000577      0.000577     0.000577
x6   0.000578      0.000578     0.000578
x7   0.000577      0.000577     0.000577
x8   0.000577      0.000577     0.000577
x9   0.000578      0.000578     0.000578
x10  0.000578      0.000578     0.000578
x11  0.000577      0.000577     0.000577
x12  0.000578      0.000578     0.000578
x13  0.000578      0.000578     0.000578
x14  0.000578      0.000578     0.000578
x15  0.000578      0.000578     0.000578
x16  0.000577      0.000577     0.000577
x17  0.000578      0.000578     0.000578
x18  0.000578      0.000578     0.000578
x19  0.000578      0.000578     0.000578
x20  0.000577      0.000577     0.000577


 tvalues :
           kanly  kanly_absorb  statsmodels
x1   -529.634200   -529.634200  -529.634200
x2   2060.288388   2060.288388  2060.288388
x3   -483.547407   -483.547407  -483.547407
x4   1819.148243   1819.148243  1819.148243
x5  -1834.300635  -1834.300635 -1834.300635
x6    103.708230    103.708230   103.708230
x7   -147.061544   -147.061544  -147.061544
x8   2159.830805   2159.830805  2159.830805
x9   -270.773656   -270.773656  -270.773656
x10   989.499668    989.499668   989.499668
x11 -2150.274395  -2150.274395 -2150.274395
x12  1820.375900   1820.375900  1820.375900
x13  -544.265850   -544.265850  -544.265850
x14  2243.527035   2243.527035  2243.527035
x15   726.791417    726.791417   726.791417
x16   445.154835    445.154835   445.154835
x17  2147.020466   2147.020466  2147.020466
x18   517.594628    517.594628   517.594628
x19 -2295.418379  -2295.418379 -2295.418379
x20 -4234.744431  -4234.744431 -4234.744431
"""
