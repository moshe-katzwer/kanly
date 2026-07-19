import pandas as pd
import numpy as np
from kanly.bayes.mcmc.adaptive_metropolis.adaptive_metropolis_mcmc import amha


def log_post(x):
    """log_post function.

    Args:
        x: TODO.
    """
    return (-(x[0] - 5) ** 2 / 10) + (-(x[1] + 2) ** 2 / 2)


result = amha(log_posterior=log_post, start_params=[4, 0], step_cov=np.eye(2) * .5, n_samples=25000, n_burnin=5000, thinning=1,
              bounds={'<x0>': (1, 10)},
              max_scaler=np.inf,
              min_scaler=1e-4,
              scaler0=1.5,
              # min_unique_draws_for_cov=15000,
              scaler_adjust_rate=.25, scaler_adjust_denom_power=.33,  # proposal_df=3,
              draw_size=900, step_cov_initial_samples=.9,
              n_chains=10, do_adaptive=True, do_parallel=True, user_prompt_for_more_iters=True, debug=True)

result.set_param_names(['x1', 'x2'])
print(result)
print(result.pids)

f = result.diagnostic_plot(0)

# import matplotlib.pyplot as plt
# plt.show()
#
# f = result.diagnostic_plot(1)

import matplotlib.pyplot as plt

# plt.show()

# print(result.apply_function_to_sample('2*{x1}-{x2}')
#       - (2*result.sample_df.x1 - result.sample_df.x2)[~result.sample_info_df.is_burnin])

# f = result.kde()
result.multi_trace()
result.diagnostic_plot('x1')
result.scatter('x1', 'x2')
result.hpdi_plot('x1', level=.6)
result.rank_plot('x1')

plt.show()

# print(result.hpdi(0, .8))
# #
#
# def moving_average(a, n=3):
#     ret = np.cumsum(a, dtype=float)
#     ret[n:] = ret[n:] - ret[:-n]
#     return ret[n - 1:] / n
#
# f, ax = plt.subplots(nrows=max(len(result.chain_results),2))
# for i, c in enumerate(result.chain_results):
#     ax[i].plot(moving_average(c['acceptance_probs'], 50))
#     ax[i].twinx().plot(moving_average(c['scalers'], 50), color='r')
#
# plt.show()
#
# from kanly.wip____bayes.bayesian_regression import get_neg_inverse_observed_information
# print(pd.DataFrame(get_neg_inverse_observed_information(log_post, [5, -2])))


