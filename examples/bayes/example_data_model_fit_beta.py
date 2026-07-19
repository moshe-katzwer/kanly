from kanly.api import DataModel
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

np.random.seed(0)
n = 1500
data = {
    'x': beta.rvs(a=5, b=2, size=n),
    'y': 3 + beta.rvs(a=2, b=4, size=n)
}

data_string = """
self.x = `x`
self.y = `y`
"""

model_string = """
logpdfval = nopython_logpdf_beta(x, a=$a$, b=$b$).sum()
return logpdfval
"""

dataobj = DataModel.build_data_model(
    data_string, model_string, data,
    debug=True, nopython=True,
)

model = dataobj.to_bayesian_model(
    bounds={'a': [0, np.inf], 'b': [0, np.inf]},
)

fit = model.amha(
    [1., 1],
    n_samples=20_000,
    n_burnin=10_000, debug=False,
    do_diff_evolution_mc=False,
    max_subchain_draws_burnin=4_000,
    max_subchain_draws_sample=20_000,
    step_cov_initial_samples=400,
    do_parallel=True,
)

print(fit)

fit.diagnostic_plot('b')
plt.show()

"""
════════════════════════════════════════════════════════════════════════════                                                                        
MCMC Results                                                              
────────────────────────────────────────────────────────────────────────────                                                                        
                                                                          
Num Parameters:     2                                                     
Method:             AMH                                                   
                                                                          
Date:                May 16, 2026        Acceptance Rate:               0.2378                                                                      
Time:                    13:09:03        Adaptive:                        True                                                                      
                                                                              
Total Iterations:          120000        Efficiency:                          
MCMC Draw Time:             4.32s            Min:                       0.1092
R_hat Time:                 0.01s            Avg:                       0.1104
ESS Time:                   0.02s            Max:                       0.1117
Summary Time:               0.11s                                             
                                         Gelman-Rubin:                        
No. Chains:                     4            R_hat > 1.01:                 0/2
    Thinning:                   1            Avg Split R_hat:           1.0002
    Iterations:             30000            Median Split R_hat:        1.0002
    Burnin:                 10000            Max Split R_hat:           1.0002
    Samples:                20000                                             
                                         Maximum Log Posterior:     7.1045e+02

════════════════════════════════════════════════════════════════════════════
    mean      std       MCSE median [.05,    .95]   Rel MCSE     ESS   R_hat
────────────────────────────────────────────────────────────────────────────
a  4.961   0.1807   0.001934  4.958  4.673  5.265  0.0003898  8733.0  1.0002
b  2.051  0.06951  0.0007354  2.049  1.938  2.168  0.0003586  8935.0  1.0002
────────────────────────────────────────────────────────────────────────────
                                                         [kanly, v=0.0.1018]
"""