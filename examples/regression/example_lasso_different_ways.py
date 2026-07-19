"""
Lasso Regression Methodology Comparison via the Kanly Package

This script demonstrates and benchmarks three fundamentally different approaches 
to achieving Lasso (L1-penalized) regularization on a specific feature ('x') 
using the `kanly` package:

0. Standard OLS (for reference)
1. Frequentist Elastic Net: Traditional coordinate descent approach utilizing the 
   standard formula interface, isolated to pure Lasso via `l1_ratio=1`.
2. Non-linear Least Squares (NLLS): A coordinate descent implementation wrapped 
   in kanly's non-linear least squares solver.
3. Bayesian MCMC (Bayesian Lasso): A sampling approach using an explicit Laplace 
   prior on the regularized feature to mirror the L1 penalty mathematically, 
   paired with a Jeffreys prior on the error variance.

All three models incorporate observation weights ('wts') and isolate the 
penalization strictly to the 'x' parameter, leaving the intercept and 'z' unpenalized.
"""

import numpy as np
from kanly.api import elastic_net, nlls_en, compare_results, lm
from kanly.bayes.bayesian_regression_model import BayesianLinearModel

# =====================================================================
# 0. Data Generation & Setup
# =====================================================================
np.random.seed(0)
n = 50

# Generate features and a target variable driven strongly by 'x'
x = np.random.randn(n)
z = np.random.rand(n)
y = 3 + 10 * x + np.random.randn(n) * 3
wts = .01 + np.random.rand(n)  # Heteroscedastic/observation weights

data = {'x': x, 'y': y, 'z': z, 'wts': wts}

# Define the penalty strength. 
# Note: Though named l2_penalty_x, setting l1_ratio=1 makes this a pure L1/Lasso penalty.
l2_penalty_x = 3.0

# =====================================================================
# 1. Model 0: Standard OLS
# =====================================================================
# Uses standard formula syntax. '$ wts' appends the weighting vector.
fit_ols = lm('y ~ x + z $ wts', data, specification_name='OLS')

# =====================================================================
# 2. Model 1: Standard Elastic Net Interface (Frequentist Lasso)
# =====================================================================
# Uses standard formula syntax. '$ wts' appends the weighting vector.
fit_en = elastic_net('y ~ x + z $ wts', data,
                     alpha={'x': l2_penalty_x}, l1_ratio=1,  # l1_ratio=1 forces Lasso
                     normalize=False, specification_name='Elastic Net')

# =====================================================================
# 3. Model 2: NLLS Coordinate Descent
# =====================================================================
# Uses explicit parameter bracket notation. Solves the regularization path 
# via an underlying non-linear optimization framework.
fit_nlls_en = nlls_en('[y] ~ {Intercept} + {x}*[x] + {z}*[z] $ [wts]', data,
                      alpha={'x': l2_penalty_x}, l1_ratio=1, scale_penalties=True,
                      ftol=1e-12, xtol=1e-6,
                      specification_name='NLLS Coord Descent')

# =====================================================================
# 4. Model 3: Bayesian Linear Model (Bayesian Lasso via MCMC)
# =====================================================================
# Build the model structure using the formula interface
model_mcmc = BayesianLinearModel.build_model_from_formula(
    'y ~ x + z $ wts', data,

    # Define custom log-probability density functions for the priors.
    # - Laplace marginal on {x} creates the Bayesian equivalent of an L1 penalty.
    # - Jeffreys prior on {__sigma2} provides an uninformative scale baseline.
    priors={'': f'-{sum(wts) * l2_penalty_x}/({{__sigma2}}) * np.abs({{x}})'
                f'- {3} * np.log({{__sigma2}}) - np.log({{__sigma2}})'
            },
    specification_name='MCMC'
)

# Execute the sampling using Adaptive Metropolis-Hastings (AMHA)
fit_mcmc = model_mcmc.amha(
    start_params=[2, 3, 3, 40.],
    n_samples=200_000,
    n_burnin=5_000,
    n_chains=6,
    max_subchain_draws_sample=33_000,
    thinning=10,  # Thinning to reduce autocorrelation in posterior draws
    do_diff_evolution_mc=True,  # Use Differential Evolution MC for better global space exploration
    debug=False
)

# =====================================================================
# 5. Model Comparison
# =====================================================================
# Compile point estimates across all three paradigms for structural validation
print(compare_results([fit_ols, fit_en, fit_nlls_en, fit_mcmc], show_bse=False))

"""
═══════════════════════════════════════════════════════════════
Regression Summary Table
═══════════════════════════════════════════════════════════════
                          (0)        (1)           (2)      (3)
───────────────────────────────────────────────────────────────
Intercept               4.189      3.567         3.567    3.566
x                       9.737      6.791         6.791    6.791
z                      -0.015      1.391         1.391    1.390
__sigma2                                                 27.027
═══════════════════════════════════════════════════════════════
Model:                    LLS         EN_sk          NLLS     MCMC
Outcome:                    y          y             y         
No. Obs.                   50         50            50         
R-squared:             0.9414     0.8570        0.8570         
R-squared Adj.:        0.9389        nan        0.8509         
Pseudo R-squared:                                              
Method:                   WLS  WTD LASSO            CD      AMH
Weights:                  wts        wts           wts         
Df Residuals:              47        NaN            47         
Df Model:                   2        NaN             3         
Covariance Type:    OLS_SMALL        N/A  NOT COMPUTED         
───────────────────────────────────────────────────────────────
(0)  "OLS"
(1)  "Elastic Net"
(2) NLLS Coord Descent
(3)  "MCMC"
═══════════════════════════════════════════════════════════════
                                            [kanly, v=0.0.1026]
"""