"""
Poisson IV GLM: three GLM specifications compared side by side to show when
and why residual inclusion (control-function approach) is needed.

Setup
-----
``x`` is endogenous: it depends on ``e``, which also enters the outcome
directly (``y = exp(-4.5 + 2*x + e)``).  ``z`` is a valid instrument
(correlated with ``x``, independent of ``e``).

Background: linear vs nonlinear IV
-----------------------------------
In a **linear** model, 2SLS (substitute first-stage fitted values for ``x``)
and the control-function approach (add first-stage residuals as a regressor)
are numerically equivalent — either gives a consistent estimator.

In a **nonlinear** model this equivalence breaks down.  Substituting ``x̂``
into the nonlinear link (``E[g⁻¹(β₀ + β₁ x̂)] ≠ E[g⁻¹(β₀ + β₁ x)]`` for
non-identity links) is *not* consistent in general.  The control-function
approach — appending the first-stage residuals to the outcome equation —
remains consistent.  This example demonstrates that asymmetry using a
Poisson log-link model.

The three Poisson GLM specifications
--------------------------------------
- **GLM** (``fit``): no IV at all; naive Poisson regression ignoring that
  ``x`` is endogenous.  Biased due to omitted-variable confounding.

- **GLM-IV** (``fit_iv``): formula uses ``y ~ x | z`` but
  ``residual_inclusion=False``.  The endogenous regressor is replaced by its
  first-stage fitted values before entering the log link.  Not consistent for
  nonlinear families — included as a diagnostic baseline.

- **GLM-IV-RI** (``fit_iv_ri``): formula uses ``y ~ x | z`` with
  ``residual_inclusion=True``.  First-stage residuals (up to order 2) are
  appended as extra regressors in the Poisson outcome equation.  This is the
  consistent control-function estimator.  The coefficient on ``x_ri`` (the
  appended residual) also serves as an endogeneity test: a large, significant
  value confirms that ``x`` is endogenous.

Expected results (true parameters: Intercept = -4.5, x = 2.0)
--------------------------------------------------------------
- ``GLM``      — biased:       ~x=2.69, Intercept=-2.37
- ``GLM-IV``   — inconsistent: ~x=1.81, Intercept=0.19
- ``GLM-IV-RI``— consistent:   ~x=2.05, Intercept=-4.18  ✓

Bootstrap covariance (``cov_type='BOOTSTRAP'``) is recommended for all IV GLM
fits because asymptotic sandwich formulas do not account for first-stage
estimation uncertainty.

References
----------
Terza, J.V., Basu, A., & Rathouz, P.J. (2008). Two-stage residual inclusion
    estimation: Addressing endogeneity in health econometric modeling.
    *Journal of Health Economics*, 27(3), 531–543.
    (Coins the 2SRI estimator; proves consistency for nonlinear models including
    Poisson; contrasts with the inconsistent two-stage predictor substitution.)

Wooldridge, J.M. (2015). Control function methods in applied econometrics.
    *Journal of Human Resources*, 50(2), 420–445.
    (Survey of control-function approaches across nonlinear model classes with
    clear exposition of when 2SLS and control functions agree vs. diverge.)

Rivers, D., & Vuong, Q.H. (1988). Limited information estimators and
    exogeneity tests for simultaneous probit models.
    *Journal of Econometrics*, 39(3), 347–366.
    (Foundational proof of residual inclusion for nonlinear limited-dependent-
    variable models.)
"""
import numpy as np
import pandas as pd

from kanly.api import glm, compare_results

n = 100_000
np.random.seed(0)
df = pd.DataFrame({
     'z': np.random.randn(n),
     'e': np.random.randn(n)
})
df['x'] = -2.8 + 0.5 * df['z'] + 0.025 * df['z'] ** 2 + 1.1 * df['e'] + 0.3 * np.random.randn(n)
df['y'] = np.exp(-4.5 + 2.0 * df.x + df.e)

fit_iv_ri = glm('y ~ x | z', df, family='poisson', residual_inclusion=True,
                residual_inclusion_order=2,
                cov_type='BOOTSTRAP')
print(fit_iv_ri)

# Compare specifications

fit = glm('y ~ x', df, family='poisson', cov_type='BOOTSTRAP')
fit_iv = glm('y ~ x | z', df, family='poisson', residual_inclusion=False, cov_type='BOOTSTRAP')

print(compare_results(
     fit_list=[fit, fit_iv, fit_iv_ri],
     fit_titles=['GLM', 'GLM-IV', 'GLM-IV-RI'],
     ref_param_values={'Intercept': -4.5, 'x': 2.0}
))

"""
==============================================================
GLM Regression Results
--------------------------------------------------------------

Dependent Variable: y

nobs:                10000   Pearson chi2:        7.493e+00
df resid:             9997   Scale:                     1.0
Family:            POISSON   Converged:                True
Link:                  LOG   Iterations:                 11
Var Weights:             -   Rel. Err.:            4.62e-11
Method:               IRLS   Abs. Err.:            5.12e-12
Log-Likelihood:  -157.8572   Cov. Type:      BOOTSTRAP(100)
Pseudo Rsq:         0.7601   Model Time:              0.01s
Deviance:            7.202   Fit Time:                0.06s

==============================================================
               coef  std err      t   P>|t|  [0.025,    0.975]
--------------------------------------------------------------
Intercept -4.178833  0.22910 -18.24  <0.001  -4.62792 -3.72975
x          2.053127  0.07409  27.71  <0.001   1.90790  2.19836
x_ri       2.793935  0.04171  66.99  <0.001   2.71218  2.87569
==============================================================

fit_intercept = True
Link Function: g(x) = log(x)
Instruments = ['Intercept', 'z'] (residual_inclusion=True)
Converged on 100 Bootstrap repetitions

                          [kanly package by moshe, v=0.0.209]




==================================================================================
Regression Summary Table
==================================================================================
                                GLM          GLM-IV       GLM-IV-RI   |  Reference
----------------------------------------------------------------------------------
Intercept                     -2.37           0.185           -4.18   |       -4.5
                           (0.0255)         (0.511)         (0.229)   |           


x                              2.69            1.81            2.05   |          2
                           (0.0443)         (0.199)        (0.0741)   |           


x_ri                                                           2.79   |           
                                                           (0.0417)   |           
==================================================================================
Outcome:                          y               y               y   |           
No. Obs.                      10000           10000           10000   |           
R-squared: :                                                          |           
R-squared Adj.:                                                       |           
Pseudo R-squared: :          0.7529          0.0663          0.7601   |           
Method:                         GLM             GLM             GLM   |           
Weights:                          -               -               -   |           
Df Residuals:                  9998            9998            9997   |           
Df Model:                         2               2               3   |           
Covariance Type:     BOOTSTRAP(100)  BOOTSTRAP(100)  BOOTSTRAP(100)   |           
----------------------------------------------------------------------------------
GLM      y ~ x
GLM-IV   y ~ x, Instruments: {Intercept, z}
GLM-IV-RIy ~ x, Instruments: {Intercept, z}
==================================================================================
                                         [kanly package by moshe, v=0.0.209]
"""