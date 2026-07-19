from kanly.api import gmm_mle
import numpy as np
import pandas as pd

"""
Estimating MLE for a chi-squared using GMM
"""

np.random.seed(0)
n = 50
k = 6

df = pd.DataFrame({'x': (np.random.randn(n, k) ** 2).sum(axis=1)})

# MLE supplying log-likelihood function
log_likelihood_formula = '-{k}/2 * np.log(2) - np.log(scipy.special.gamma({k}/2)) + ({k}/2-1)*np.log([x]) - [x]/2'

print(gmm_mle(log_likelihood_formula, df, is_log_llf=True, do_njit=False,
              start_params=[1.], debug=True))


# MLE supplying likelihood function
likelihood_formula = '([x]**({k}/2-1)*np.exp(-[x]/2))/(2**({k}/2) * scipy.special.gamma({k}/2) )'

print(gmm_mle(likelihood_formula, df, is_log_llf=False, do_njit=False,
              start_params=[1.], debug=True))
