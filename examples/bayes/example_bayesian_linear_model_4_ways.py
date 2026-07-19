"""
Bayesian weighted linear regression with a categorical fixed effect — four ways to
specify the same model in kanly, then compare posterior summaries with ``compare_results``.

1. ``bayes_lm_model`` — regression-style formula (``y ~ x + C(g) $ wts``).
2. ``bayes_nlls_model`` — NLLS-style formula (bracket/brace syntax).
3. ``build_data_model`` — data and model code blocks, converted to a sampler with
   ``to_bayesian_model()``.
4. ``BayesianModel`` — explicit log-likelihood function and parameter names.

Each fit uses adaptive Metropolis (``.amha()``); the four chains should agree (see output below).
"""
from kanly.api import build_data_model, BayesianModel, bayes_lm_model, bayes_nlls_model, compare_results
import numpy as np
from scipy.stats import t, norm

np.random.seed(0)
n = 500
x = 1.56 * np.random.randn(n)
g = np.random.randint(0, 4, n)
wts = .01 + np.random.rand(n)

y = 3 + 10 * x + (g == 2) * 3 + t.rvs(df=3, size=n)

data = {'x': x, 'y': y, 'wts': wts, 'g': g}

fit1 = bayes_lm_model(
    'y ~ x + C(g) $ wts', data, priors={'x': 'norm(0, .3)'}
).amha(start_params={'__sigma2': 1.}, n_samples=10_000, n_burnin=1_500)

fit2 = bayes_nlls_model(
    '[y] ~ {Intercept} + {x}*[x] + [C(g,-1)] $ [wts]', data, priors={'x': 'norm(0, .3)'}
).amha(start_params={'__sigma2': 1.}, n_samples=10_000, n_burnin=1_500)

fit3 = build_data_model(
    data_code_block='self.y = `y`; self.x = `x`; self.root_weights = `np.sqrt(wts)`; self.g = `C(g)`',
    model_code_block='return logpdf_norm($x$, 0., .3) + logpdf_norm(y, $Intercept$ + $x$*x + $_dummy[g,-1]$,'
                     '$__sigma2<0,np.inf>$**.5 / root_weights).sum();',
    data=data
).to_bayesian_model().amha(start_params={'__sigma2': 1.}, n_samples=50_000, n_burnin=1_500)

fit4 = BayesianModel(
    log_likelihood_function=lambda params: norm.logpdf(
        y,
        loc=params[0] + params[1] * x + sum([params[2 + j - 1] * (g == j) for j in range(1, 4)]),
        scale=params[-1] ** .5 / np.sqrt(wts)
    ).sum(),
    param_names=['Intercept', 'x', 'C(g)[1]', 'C(g)[2]', 'C(g)[3]', '__sigma2'],
    priors={'x': 'norm(0, .3)'},
    bounds={'__sigma2': (0, np.inf)},
).amha(start_params={'__sigma2': 1.}, n_samples=10_000, n_burnin=1_500)

print(compare_results([fit1, fit2, fit3, fit4]))


"""
══════════════════════════════════════════════════════════════════════════════════════════════════════
Regression Summary Table
══════════════════════════════════════════════════════════════════════════════════════════════════════
                                    (0)                  (1)                  (2)                  (3)
──────────────────────────────────────────────────────────────────────────────────────────────────────
C(g)[1]                          0.1301               0.1301               0.1301               0.1301
                               (0.2095)             (0.2095)             (0.2095)             (0.2095)


C(g)[2]                           2.988                2.988                2.988                2.988
                               (0.2068)             (0.2068)             (0.2068)             (0.2068)


C(g)[3]                          0.2177               0.2177               0.2177               0.2177
                               (0.2066)             (0.2066)             (0.2066)             (0.2066)


Intercept                         3.043                3.043                3.043                3.043
                               (0.1536)             (0.1536)             (0.1536)             (0.1536)


__sigma2                          1.342                1.342                1.342                1.342
                              (0.09003)            (0.09003)            (0.09003)            (0.09003)


x                                 9.725                9.725                9.725                9.725
                              (0.04953)            (0.04953)            (0.04953)            (0.04953)
══════════════════════════════════════════════════════════════════════════════════════════════════════
Outcome:                                                                                              
No. Obs.                                                                                              
R-squared:                                                                                            
R-squared Adj.:                                                                                       
Pseudo R-squared:                                                                                     
Method:             Adaptive Metropolis  Adaptive Metropolis  Adaptive Metropolis  Adaptive Metropolis
Weights:                                                                                              
Df Residuals:                                                                                         
Df Model:                                                                                             
Covariance Type:                                                                                      
Converged:                                                                                            
══════════════════════════════════════════════════════════════════════════════════════════════════════
                                                                                    [kanly, v=0.0.795]
"""
