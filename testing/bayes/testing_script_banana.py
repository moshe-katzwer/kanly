import numpy as np
from kanly.bayes.mcmc.adaptive_metropolis.adaptive_metropolis_mcmc import amha
import matplotlib.pyplot as plt


def log_likelihood(x):
    return -(2.5 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2)


fit = amha(log_likelihood, [0, 0],
           debug=True,
           param_names=['a', 'b'],
           n_burnin=10_000,
           n_chains=6,
           n_samples=50_000,
           max_subchain_draws_sample=30_000,
           user_prompt_for_more_iters=False
           )
print(fit)

fit.scatter('a', 'b')
plt.show()
