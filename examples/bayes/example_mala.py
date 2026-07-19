from kanly.api import bayes_nlls_model, compare_results
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
n = 100
x = 1.56 * np.random.randn(n)
z = np.random.rand(n)
y = 3 + 10 * x - 2 * z + np.random.randn(n) * 3
wts = .01 + np.random.rand(n)
data = {'x': x, 'y': y, 'z': z, 'wts': wts}

fit1 = bayes_nlls_model(
    '[y] ~ {a}+{b}*[x] + {c}*[z]', data, bounds={'b': [5, 8], 'a': [1, 6]}
).amha(
    [2, 7, 1., 1.], user_prompt_for_more_iters=False, debug=False, n_samples=40_000,
    thinning=5,
)

#
fit2 = bayes_nlls_model(
    '[y] ~ {a}+{b}*[x] + {c}*[z]', data, bounds={'b': [5, 8], 'a': [1, 6]}
).mala(
    [2, 7, 1., 1.], user_prompt_for_more_iters=False, debug=False, n_samples=40_000,
    thinning=5,
)

print(fit1)
print(fit2)

fig, ax = plt.subplots(ncols=fit2.num_params, figsize=(10,5))

for i, nm in enumerate(fit2.param_names):
    ax[i].hist(fit1.get_sample(nm), alpha=.5, bins=30, density=True)
    ax[i].hist(fit2.get_sample(nm), alpha=.5, bins=30, density=True)
plt.show()

print(compare_results([fit1, fit2]))

"""
═════════════════════════════════════════════════════════════════════════════════════════
MCMC Results
─────────────────────────────────────────────────────────────────────────────────────────

Num Parameters:     4
Method:             AMH

Date:                May 26, 2026        Acceptance Rate:                0.2346
Time:                    10:34:47        Adaptive:                         True
                                                                               
Total Iterations:          200000        Efficiency:                           
MCMC Draw Time:             7.96s            Min:                        0.2721
R_hat Time:                 0.01s            Avg:                        0.3121
ESS Time:                   0.08s            Max:                        0.3595
Summary Time:               0.24s                                              
                                         Gelman-Rubin:                         
No. Chains:                     4            R_hat > 1.01:                  0/4
    Thinning:                   5            Avg Split R_hat:            1.0000
    Iterations:             50000            Median Split R_hat:         1.0000
    Burnin:                 10000            Max Split R_hat:            1.0000
    Samples:                40000                                              
                                         Maximum Log Posterior:     -2.8674e+02

═════════════════════════════════════════════════════════════════════════════════════════
            mean      std       MCSE  median  [.05,      .95]   Rel MCSE      ESS   R_hat
─────────────────────────────────────────────────────────────────────────────────────────
a          3.244   0.8667   0.004154   3.236   1.815    4.691   0.001281  43534.0  1.0000
b          7.962  0.03817  0.0001592   7.974   7.886    7.998   2.00e-05  57523.0  1.0000
c         -3.119     1.53   0.007046  -3.104  -5.667  -0.6146  -0.002259  47180.0  1.0000
__sigma2   19.68    2.943    0.01297   19.39   15.37    24.94  0.0006593  51477.0  1.0000
─────────────────────────────────────────────────────────────────────────────────────────
                                                                      [kanly, v=0.0.1033]

════════════════════════════════════════════════════════════════════════════════════════
MCMC Results
────────────────────────────────────────────────────────────────────────────────────────

Num Parameters:     4
Method:             CD-MALA

Date:                May 26, 2026        Acceptance Rate:                0.6015
Time:                    10:34:59        Adaptive:                         True
                                                                               
Total Iterations:          160000        Efficiency:                           
MCMC Draw Time:            11.76s            Min:                        0.0727
R_hat Time:                 0.01s            Avg:                        0.2203
ESS Time:                   0.03s            Max:                        0.4377
Summary Time:               0.16s                                              
                                         Gelman-Rubin:                         
No. Chains:                     4            R_hat > 1.01:                  0/4
    Thinning:                   5            Avg Split R_hat:            1.0000
    Iterations:             40000            Median Split R_hat:         1.0000
    Burnin:                 10000            Max Split R_hat:            1.0000
    Samples:                30000                                              
                                         Maximum Log Posterior:     -2.8674e+02

════════════════════════════════════════════════════════════════════════════════════════
            mean      std       MCSE median  [.05,      .95]   Rel MCSE      ESS   R_hat
────────────────────────────────────────────────────────────────────────────────────────
a          3.249   0.8651   0.009265  3.236   1.834    4.709   0.002851   8719.0  1.0000
b          7.963  0.03592  0.0001905  7.974   7.892    7.998   2.39e-05  35540.0  1.0000
c         -3.126    1.527    0.01614  -3.11  -5.673  -0.6277  -0.005162   8957.0  1.0000
__sigma2   19.64    2.946    0.01286  19.36   15.35    24.87  0.0006545  52523.0  1.0000
────────────────────────────────────────────────────────────────────────────────────────
                                                                     [kanly, v=0.0.1033]


════════════════════════════════════════════════════════════
Regression Summary Table
════════════════════════════════════════════════════════════
                        (0)      (1)
────────────────────────────────────────────────────────────
a                     3.244    3.249
                    (0.867)  (0.865)


b                     7.962    7.963
                    (0.038)  (0.036)


c                    -3.119   -3.126
                    (1.530)  (1.527)


__sigma2             19.678   19.643
                    (2.943)  (2.946)
════════════════════════════════════════════════════════════
Model:                 MCMC     MCMC
Outcome:                            
No. Obs.                            
R-squared:                          
R-squared Adj.:                     
Pseudo R-squared:                   
Method:                 AMH  CD-MALA
Weights:                            
Df Residuals:                       
Df Model:                           
Covariance Type:                    
════════════════════════════════════════════════════════════
                                         [kanly, v=0.0.1033]
"""
