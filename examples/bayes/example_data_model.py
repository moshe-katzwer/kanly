from kanly.api import DataModel
import numpy as np
from scipy.stats import t

from kanly.bayes.mcmc.mcmc_results import MCMCResults

np.random.seed(0)
n = 100
x = 1.56 * np.random.randn(n)
y = 3 + 10 * x + 2*t.rvs(df=3, size=n)
z = t.rvs(df=3, size=n)
wts = .01 + np.random.rand(n)
data = {'x': x, 'y': y, 'wts': wts, 'g': np.random.randint(0, 4, n), 'z': z}

data_string = """
self.x = `x`
self.z = `z`
self.y = `y`
self.weights = `wts`
self.g = `C(g)`                                   # categorical data
self.root_weights = np.sqrt(self.weights)
"""

model_string = """
pred = $Intercept$ + $x$ * x + $_dummy[g,-1]$ + $_poly[z,2]$      # WLS on 'x' and fixed effects for 'g'
resid = y - pred

return logpdf_norm(
    resid, 
    loc=0.0, 
    scale=$sigma$ / root_weights
).sum()     # normal log-likelihood
"""

dataobj = DataModel.build_data_model(
    data_string, model_string, data, debug=False, nopython=False,
)

model = dataobj.to_bayesian_model(
    priors={'x': 'norm(-23.6, 1.33)'},
    bounds={'x': [0, 10]},
    debug=True,
)

print(model)

model.map( 1.2 * np.ones(8))

fit: MCMCResults = model.sample(
    1.2 * np.ones(8),
    n_samples=10_000, n_burnin=3_000,
    thinning=5,
    debug=True,
    n_chains=2,
    do_diff_evolution_mc=True,
    max_subchain_draws_burnin=2_000,
    max_subchain_draws_sample=5_000,
    # step_cov_initial_samples=400,
    # do_parallel=True,
    do_mala_cd_warmup=True,
    frac_burnin_mala=.6,
    n_chains_mala=2,
    # n_samples_mala=10_000,
    # max_subchain_draws_mala=1_000,
    # diff_evolution_step_cadence_mala=5,
    # diff_evolution_jump_cadence=10,
    # callback_function=lambda x: x
)

print(fit)

"""
═══════════════════════════════════════════════════════════════════════════════════════
MCMC Results
───────────────────────────────────────────────────────────────────────────────────────

Num Parameters:     8
Method:             AMH

Date:                May 16, 2026        Acceptance Rate:                0.2317
Time:                    13:08:07        Adaptive:                         True
                                                                               
Total Iterations:           78000        Efficiency:                           
MCMC Draw Time:             5.56s            Min:                        0.0072
R_hat Time:                 0.01s            Avg:                        0.0567
ESS Time:                   0.06s            Max:                        0.0946
Summary Time:               0.16s                                              
                                         Gelman-Rubin:                         
No. Chains:                     6            R_hat > 1.01:                  3/8
    Thinning:                   5            Avg Split R_hat:            1.0567
    Iterations:             13000            Median Split R_hat:         1.0567
    Burnin:                  3000            Max Split R_hat:            1.1810
    Samples:                10000                                              
                                         Maximum Log Posterior:     -5.7470e+02

═══════════════════════════════════════════════════════════════════════════════════════
              mean     std      MCSE    median   [.05,    .95] Rel MCSE     ESS   R_hat
───────────────────────────────────────────────────────────────────────────────────────
Intercept    1.506   1.996   0.02649     1.432   -1.728  5.117  0.01759  5678.0  1.0019
x            6.343   4.287     0.206     9.184  0.03976  9.704  0.03248   433.0  1.1810
C(g)[1]       2.07   2.663   0.03669     1.848   -1.878  7.248  0.01772  5267.0  1.0035
C(g)[2]       3.16   2.974   0.03988     3.012   -1.543  8.662  0.01262  5561.0  1.0017
C(g)[3]      2.838   3.011   0.04474     2.453   -1.168  8.919  0.01577  4529.0  1.0084
poly[z=1]  -0.2706  0.7824   0.01272   -0.3787   -1.271   1.31   -0.047  3785.0  1.0095
poly[z=2]  0.05413  0.2689  0.008496  -0.04322  -0.1873  0.651    0.157  1002.0  1.0710
sigma        5.568   4.615    0.1496     2.483    2.068  13.06  0.02687   951.0  1.1767
───────────────────────────────────────────────────────────────────────────────────────
Some R_hat are above 1.01, the MCMC chains have not converged!
Some effective sample sizes are below 5000!

                                                                    [kanly, v=0.0.1018]
"""