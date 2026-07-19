from kanly.bayes.mcmc.diagnostics.gelman_rubin_split_rhat import get_rhat
import numpy as np
from numpy.testing import assert_almost_equal

n_chains = 4
np.random.seed(0)
runs_per_chain = 1000
x = [np.random.randn(runs_per_chain) + np.random.randn()*.954 for _ in range(n_chains)]
result = get_rhat(x)
R_hat = result[0]

x_sub = []
for _x in x:
    x_sub += [_x[:len(_x) // 2], _x[len(_x) // 2:]]

chain_means = np.array([_x.mean() for _x in x_sub])
grand_mean = np.mean(chain_means)
N = len(x_sub[0])  # draws per chain
M = len(x_sub)  # number of chains

B = N / (M - 1.) * np.sum((chain_means - grand_mean) ** 2)
s_m = np.array([np.sum((_x - chain_means[m]) ** 2) for m, _x in enumerate(x_sub)]) / (N - 1)
W = 1. / M * np.sum(s_m)

var_plus = (N - 1.) / N * W + 1. / N * B

print(">>> ", N)
print('\t > W = ', W)
print('\t > B = ', B)
print('\t > N = ', N)
print('\t > var_plus = ', (N - 1.) / N * W + 1. / N * B)


R_hat_expected = np.sqrt(var_plus / W)
print(R_hat, var_plus, B, W)

assert_almost_equal(R_hat_expected, R_hat)