import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from kanly.api import glm, gam
from statsmodels.gam.api import GLMGam, BSplines
import statsmodels.api as sm
from kanly.api import bfgs_pqn
from kanly.nonparametric.bspline import bspline_penalty

from kanly.regression.generalized_linear_models.regression_results import SparseGLMRegressionResults

np.random.seed(0)
n2 = 300
x2 = np.random.randn(n2)
x2.sort()
y2 = np.exp(-4.5 + .5 * x2 + .05 * x2 ** 2 - .5 * np.sin(x2 * 4) + .4 * np.random.randn(n2))
data2 = pd.DataFrame(dict(x2=x2, y2=y2))
df_2 = 20

# GLM and GAM with no penalization should be identical

fit_glm = glm(f'y2 ~ bs(x2, degree=3, df={df_2})', data2,
              family='poisson')

fit_gam0 = gam('y2 ~ x2', data2,
               df=dict(x2=df_2),
               penalty=dict(x2=0.0),  # no penalization - just GLM
               family='poisson')

assert_allclose(fit_glm.params, fit_gam0.params)
assert_allclose(fit_glm.bse, fit_gam0.bse)
assert_allclose(fit_glm.fittedvalues, fit_gam0.fittedvalues)
assert_allclose(fit_glm.predict(data=data2[:10]), fit_gam0.predict(data=data2[:10]))

# Compare kanly to statsmodels
penalty_ = .05

fit_gam1: SparseGLMRegressionResults = gam(
    'y2 ~ x2', data2,
    penalty=dict(x2=penalty_),
    df=dict(x2=df_2), family='poisson', cov_type='hc1', tol=1e-8, max_iter=100)

bs = BSplines(data2[['x2']], df=[df_2 + 1], degree=[3])
alpha = np.array([penalty_])

fit_gamsm = GLMGam.from_formula('y2 ~ 1', data=data2, smoother=bs, alpha=alpha,
                                family=sm.families.Poisson()).fit(cov_type='hc1')

assert_allclose(fit_gamsm.params, fit_gam1.params, atol=1e-6, rtol=1e-4)
assert_allclose(fit_gamsm.bse, fit_gam1.bse, atol=1e-5, rtol=1e-3)
assert_allclose(fit_gamsm.fittedvalues, fit_gam1.fittedvalues, atol=1e-6, rtol=1e-4)

# Kanly estimation by hand
llf = fit_glm.get_log_likelihood_function()

S = bspline_penalty(
    knots=fit_gam1.model.formula_design_info.exog_terms[1].state['numerical']
    ['bs(x2, degree=3, df=20, include_intercept=False)']['bspline']['knots'],
    degree=3, include_intercept=False)

res = bfgs_pqn(lambda x: -llf(x, 1.0) + penalty_ * x[1:].dot(S).dot(x[1:]),
               x0=fit_glm.params, xtol=1e-8, ftol=1e-15)

assert_allclose(res.x, fit_gam1.params, atol=1e-2, rtol=5e-3)
assert_allclose(np.exp(fit_gam1.model.exog.toarray() @ res.x),
                fit_gam1.fittedvalues, atol=1e-4, rtol=5e-3)
