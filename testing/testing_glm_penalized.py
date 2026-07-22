"""Cross-check the package's different penalized GLM fitting paths.

The script generates synthetic data, fits Gaussian and Poisson models with
several L2 penalty strengths, and compares the results with a hand-written
penalized likelihood optimized by BFGS.
"""

from kanly.regression.generalized_linear_models.sparse_glm_internal import glm_internal
from kanly.regression.generalized_linear_models.families import Poisson
from kanly.api import glm
from kanly.api import sparse_dmatrix, sparse_dmatrices

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from patsy import dmatrix

from scipy.stats import norm, poisson
from kanly.api import bfgs

from numpy.testing import assert_allclose

# Create positive response data for two groups. Group membership changes both
# the intercept and the slope with respect to log(x).
n = 130
np.random.seed(0)
x = np.exp(np.random.randn(n))
g = np.random.randint(0,2,n)
y = np.exp(.15 + .1 * g + (.26 + .15 * g) * np.log(x) + .2*np.random.randn(n))
plt.scatter(x,y)
df = pd.DataFrame({'x': x, 'g': g})

# x2 deliberately duplicates x so the design includes equivalent predictors
# in a main effect and in group-specific interaction terms.
df = pd.DataFrame({'x': x, 'g': g, 'y': y})
df['x2'] = df.x

# Build response and design arrays for the independent reference objective.
endog, exog = sparse_dmatrices(
    'y ~ C(g) + I(np.log(x2)) + C(g):I(np.log(x))', df, return_dense=True,
    drop_1_for_FE=True
)
endog = endog.values
exog = exog.values

stds = exog.std(axis=0)

for normalize in [False, True]:
    # Exercise no, weak, moderate, and strong L2 regularization.
    for a in [0, .5, 1, 10]:
        # Penalize only the last two coefficients. The matrix representation is
        # scaled by the number of observations to match glm's alpha convention.
        G = np.zeros((5, 5))
        G[-1, -1] = a * n
        G[-2, -2] = a * n
        if normalize:
            G[-1,-1] /= stds[-1]
            G[-2,-2] /= stds[-2]
        alpha = [0]*3 + [a]*2

        # Shrink the penalized coefficients toward 1 instead of the usual 0.
        regvals = [0] * 3 + [1] * 2

        # Define the same penalized likelihood explicitly, providing a reference
        # independent of glm's matrix, IRLS, and coordinate-descent interfaces.
        def objective_function(params):
            params = np.asarray(params)
            lin_pred = exog @ params
            if family == 'poisson':
                # Convert the linear predictor to the Poisson mean via the log link.
                lin_pred = np.exp(lin_pred)
                llf = np.sum(endog * np.log(lin_pred) - lin_pred)
            elif family == 'gaussian':
                # Use a unit-variance Gaussian likelihood for this comparison.
                llf = norm.logpdf(endog, lin_pred, 1.0).sum() 
            # Subtract the sample-size-scaled L2 penalty from the log-likelihood.
            return llf - n/2*sum(alpha * (params - regvals)**2)


        for family in ['gaussian', 'poisson']:
            # First fit: supply the L2 penalty as an explicit matrix.
            fit0 = glm(
                'y ~ C(g) + I(np.log(x2)) + C(g):I(np.log(x))', df, 
                L2_penalty_matrix=G, regularize_to_values=regvals,
                normalize = False,
                family=family,
                max_iter=2000,
                tol=1e-8,
            )

            # Second fit: express the same penalty per coefficient and use IRLS.
            fit1 = glm(
                'y ~ C(g) + I(np.log(x2)) + C(g):I(np.log(x))', df, 
                alpha=alpha, l1_ratio=0,
                regularize_to_values=regvals,
                normalize = normalize,
                opt_method='IRLS',
                family=family,
                max_iter=2000,
                tol=1e-8,
            )

            # Third fit: use the same setup with coordinate descent.
            fit2 = glm(
                'y ~ C(g) + I(np.log(x2)) + C(g):I(np.log(x))', df, 
                alpha=alpha, l1_ratio=0,
                regularize_to_values=regvals,
                normalize = normalize,
                opt_method='COORDINATE_DESCENT',
                family=family,
                max_iter=2000,
                tol=1e-8,
            )

            fits = [fit0, fit1, fit2]
            if a == 0:
                # With no penalty, include glm's ordinary unregularized fit too.
                fits.append(
                    glm(
                        'y ~ C(g) + I(np.log(x2)) + C(g):I(np.log(x))', df, 
                        regularize_to_values=regvals,
                        family=family,
                    )
                )

            # Optimize the explicit objective as a solver-independent reference.
            optim_res = bfgs(
                objective_function, [.1]*5, maximize=True, debug=False,
                maxiter=1000, xtol=1e-6
            )

            print()
            print(f'{a=}, {normalize=}, {family=}')
            for f in fits:
                print(f.params.values)
                if a == 0:
                    # Duplicate predictors make the individual slope coefficients
                    # non-unique. Their group-specific sums are identifiable, so
                    # compare those sums along with the first two coefficients.
                    assert_allclose(fit0.params[:2], f.params[:2], atol=1e-4, rtol=1e-3)
                    assert_allclose(fit0.params[:2], optim_res.x[:2], atol=1e-4, rtol=1e-3)
                    for j in (3,4):
                        assert_allclose(fit0.params.iloc[2]+fit0.params.iloc[j], 
                                        f.params.iloc[2]+f.params.iloc[j], atol=1e-4, rtol=1e-3)
                        assert_allclose(fit0.params.iloc[2]+fit0.params.iloc[j], 
                                        optim_res.x[2]+ optim_res.x[j], atol=1e-4, rtol=1e-3)
                else:
                    # A positive penalty selects a unique coefficient vector.
                    assert_allclose(fit0.params, f.params, atol=1e-4, rtol=1e-3)
    
                # print(optim_res.x)
                # if a == 0:
                #     # Make the same identifiable-combination check against BFGS.
                #     assert_allclose(fit0.params[:2], optim_res.x[:2], atol=1e-4, rtol=1e-3)
                #     for j in (3,4):
                #         assert_allclose(fit0.params.iloc[2]+fit0.params.iloc[j], 
                #                          optim_res.x[2]+ optim_res.x[j], atol=1e-4, rtol=1e-3)
                # else:
                #     assert_allclose(fit0.params, optim_res.x, atol=1e-4, rtol=1e-3)

                
print('\nALL PASSED!')
