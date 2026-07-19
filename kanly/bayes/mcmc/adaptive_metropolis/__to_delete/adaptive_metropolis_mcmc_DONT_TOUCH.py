# OLD VERSION USING PATHOS, NOT RAY

# from __future__ import absolute_import, print_function
#
# import os
# import pprint
# import time
# import warnings
#
# from multiprocessing import RLock
#
# import numpy as np
# import pandas as pd
# import psutil
# #from pathos.multiprocessing import Pool
# from pathos.multiprocessing import Pool
# #from pathos.pools import ProcessPool as Pool
# #from pathos.pools import ParallelPool as Pool
# #from multiprocess.pool import Pool
# #from multiprocessing import Pool
# # from pathos.multiprocessing import ThreadPool as Pool # <-- todo is this faster?, probably not...
# from scipy.stats import multivariate_normal, multivariate_t
# from tqdm import tqdm
#
# import dill # don't delete
# #dill.settings['recurse'] = True
# from dill import loads, dumps
#
# from kanly.bayes.mcmc.adaptive_metropolis.constants import DEFAULT_AM_N_BURNIN, DEFAULT_AM_N_SAMPLES, \
#     DEFAULT_AM_N_CHAINS, DEFAULT_AM_THINNING, DEFAULT_AM_DRAW_SIZE, DEFAULT_AM_TARGET_ACCEPTANCE_RATE, \
#     DEFAULT_AM_MAX_PROCESSES, DEFAULT_AM_PBAR_UPDATE_CADENCE, DEFAULT_AM_DO_ADAPTIVE, \
#     DEFAULT_AM_DO_PARALLEL, DEFAULT_AM_USER_PROMPT_FOR_MORE_ITERS, \
#     DEFAULT_AM_SCALER_ADJUST_RATE, DEFAULT_AM_SCALER_ADJUST_DENOM_POWER, DEFAULT_AM_MIN_SCALER, DEFAULT_AM_MAX_SCALER, \
#     DEFAULT_AM_MAX_SUBCHAIN_DRAWS_BURNIN, DEFAULT_AM_MAX_SUBCHAIN_DRAWS_SAMPlE, \
#     DEFAULT_AM_PROPOSAL_DF, DEFAULT_AM_RESAMPLE_K, DEFAULT_AM_STEP_COV_ADJUST_RATE, DEFAULT_AM_NORMALIZE_STEP_COV, \
#     DEFAULT_AM_DO_DIFF_EVOLUTION_MC, DEFAULT_AM_DIFF_EVOLUTION_FRAC_BURNIN, \
#     DEFAULT_AM_DIFF_EVOLUTION_MAX_DRAWS, DEFAULT_AM_DIFF_EVOLUTION_WEIGHT, DEFAULT_AM_STOP_ADAPTATION_AFTER_BURNIN
# from kanly.bayes.mcmc.aggregate_covariances import aggregate_covs
# from kanly.bayes.mcmc.check_starting_point import check_starting_point
# from kanly.bayes.mcmc.diagnostics.diagnostics import get_diagnostic_update_message
# from kanly.bayes.mcmc.format_starting_point_for_mcmc import format_starting_point_for_mcmc
# from kanly.bayes.mcmc.mcmc_results import MCMCResults
# from kanly.bayes.parameter_transformations import convert_samples_to_unbounded_space
# from kanly.utils.user_prompt_for_more_iters import user_prompt_for_more_iters_method
#
#
# def _get_process_memory_mb(pid=None):
#     """rss, vms from psutil memory info"""
#     if pid is None:
#         pid = os.getpid()
#     try:
#         proc = psutil.Process(pid)
#         return proc.memory_info().rss / 1024 ** 2, proc.memory_info().vms / 1024 ** 2
#     except:
#         return -1, -1
#
#
# def normalize_matrix_det(V, not_fixed_param_index=None, normalizer_value=1.0):
#     """
#     Normalizes a matrix V (assumed positive semi-definite) to have
#     determinant `normalizer_value`
#
#     (Not *currently* used, but some schemes normalize the proposal
#     covariance.)
#
#     not_fixed_param_index - list of free moving parameters, only ones included
#     in det() calculation
#     """
#     assert normalizer_value > 0
#     V = np.asarray(V)
#     nn = V.shape[0]
#     if not_fixed_param_index is None or len(not_fixed_param_index) == nn:
#         V_sub = V
#     else:
#         V_sub = V[not_fixed_param_index][:, not_fixed_param_index]
#     slog_det = np.linalg.slogdet(V_sub)
#     assert slog_det[0] > 0
#     normalizer = np.exp((np.log(normalizer_value) - slog_det[1]) / len(V_sub))
#     return normalizer * V, normalizer, np.linalg.slogdet(normalizer * V_sub)[1]
#
#
# def make_pos_semidef(step_cov):
#     # Make a matrix positive semidefinite by adding to it's diagonal
#     is_pos_def = False
#     incr = 1e-6
#     while not is_pos_def:
#         try:
#             np.linalg.cholesky(step_cov)
#             is_pos_def = True
#             break
#         except np.linalg.LinAlgError:
#             np.fill_diagonal(step_cov, step_cov.diagonal() + incr)
#             incr *= 1.5
#     return step_cov
#
#
# def _acceptance_choice(x0, x1, lp0, lp1, lp_adj0, lp_adj1, uniform_draw):
#     """Metropolis-Hasting accept/reject step"""
#
#     if not np.isfinite(lp1):
#         accepted_bool = False
#         log_acceptance_prob = -np.inf
#
#     else:
#
#         lp_gap = (lp1 + lp_adj1) - (lp0 + lp_adj0)
#         log_acceptance_prob = min(lp_gap, 0.0)
#
#         if lp_gap >= 0 or uniform_draw <= lp_gap:
#             accepted_bool = True
#
#         else:
#             accepted_bool = False
#
#     if accepted_bool:
#         x_new, lp_new, lp_adj_new = x1, lp1, lp_adj1
#     else:
#         x_new, lp_new, lp_adj_new = x0, lp0, lp_adj0
#
#     return x_new, lp_new, lp_adj_new, accepted_bool, log_acceptance_prob
#
#
# def run_adaptive_metropolis_chain(seed, log_posterior, x0, step_cov, log_posterior_jacobian_adjustment=None,
#                                   n_burnin=DEFAULT_AM_N_BURNIN, n_samples=DEFAULT_AM_N_SAMPLES, itr0=0,
#                                   target_acceptance_rate=DEFAULT_AM_TARGET_ACCEPTANCE_RATE, chain_no=None,
#                                   draw_size=DEFAULT_AM_DRAW_SIZE, scaler0=None, min_scaler=DEFAULT_AM_MIN_SCALER,
#                                   max_scaler=DEFAULT_AM_MAX_SCALER, scaler_adjust_rate=DEFAULT_AM_SCALER_ADJUST_RATE,
#                                   scaler_adjust_denom_power=DEFAULT_AM_SCALER_ADJUST_DENOM_POWER,
#                                   thinning=DEFAULT_AM_THINNING, pbar_update_cadence=DEFAULT_AM_PBAR_UPDATE_CADENCE,
#                                   do_adaptive=DEFAULT_AM_DO_ADAPTIVE, debug=False,
#                                   proposal_df=DEFAULT_AM_PROPOSAL_DF, resample_k=DEFAULT_AM_RESAMPLE_K, bloc=0,
#                                   fix_params_arg=None, diff_evolulution_past_samples=None, diff_evolution_weight=None,
#                                   ) -> dict:
#     __time_chn_master = time.time()
#     #
#     # log_posterior = loads(log_posterior)
#     # log_posterior_jacobian_adjustment = loads(log_posterior_jacobian_adjustment)
#
#     rand = np.random.RandomState(seed)
#
#     if resample_k is None or resample_k == 0:
#         resample_k = np.inf
#
#     do_lp_adjustment = log_posterior_jacobian_adjustment is not None
#     x0 = np.array(x0, dtype=float).flatten().copy()
#     num_params = len(x0)
#
#     step_cov0 = step_cov.copy()
#     step_cov = make_pos_semidef(step_cov)
#
#     if max_scaler is None:
#         max_scaler = 100000.0
#     if min_scaler is None:
#         min_scaler = .00001
#
#     if scaler0 is None:
#         scaler = 1.0
#     else:
#         scaler = scaler0
#
#     gelman_optimal_scaler_value = 2.38 / np.sqrt(num_params)
#
#     if diff_evolulution_past_samples is not None:
#         assert 0 < diff_evolution_weight <= 1
#         scaler_draw_denom = (
#                 np.sqrt(diff_evolution_weight ** 2 + (1 - diff_evolution_weight) ** 2)
#                 * gelman_optimal_scaler_value
#         )
#         de_idx = rand.randint(0, len(diff_evolulution_past_samples), ((n_samples + n_burnin) * thinning, 2))
#     else:
#         diff_evolution_weight = 0.0
#         scaler_draw_denom = gelman_optimal_scaler_value
#
#     if chain_no is None:
#         chain_no = seed
#
#     draws_nc = np.zeros((n_burnin + n_samples, num_params))
#     log_posterior_arr = np.zeros(n_burnin + n_samples)
#
#     x0 = np.array(x0, copy=True, dtype=float)
#     if fix_params_arg is not None:
#         x0[fix_params_arg['index']] = fix_params_arg['values']
#
#     lp0 = log_posterior(x0)
#     lp_adj0 = log_posterior_jacobian_adjustment(x0) if do_lp_adjustment else 0.0
#
#     if not np.isfinite(lp0) or np.isnan(lp0):
#         raise Exception('Starting point must generating finite log-posterior!')
#
#     last_update_time = 0
#     # last_update_cov = itr0 + 1
#     # num_cov_updates = 0
#
#     lp_time = 0
#     cov_time = 0
#     draw_time = 0
#     accept_decision_time = 0
#
#     _t_draw = time.time()
#     if proposal_df == np.inf:
#         rand_dist = multivariate_normal(mean=[0.0] * num_params, cov=step_cov.copy(), allow_singular=True)
#     else:
#         raise NotImplementedError
#         #rand_dist = multivariate_t(loc=[0.0] * num_params, shape=step_cov.copy(), df=proposal_df)
#
#     draw_generator = (d for d in rand_dist.rvs(random_state=rand, size=draw_size))
#     #draw_generator = (d for d in rand.multivariate_normal(mean=[0.0] * num_params, cov=step_cov.copy(), size=draw_size))
#     draw_time += time.time() - _t_draw
#
#     accepted_rate = 0.0
#
#     # S2 = 0.0
#     # S1 = 0.0
#
#     err_settings = np.seterr()
#     np.seterr(all='ignore')
#
#     setup_time = time.time() - __time_chn_master
#
#     accept = 0
#     accepteds = []
#     log_acceptance_probs = []
#     scalers = []
#
#     itr = 0
#
#     log_random_uniform = np.log(rand.rand(thinning * (n_samples + n_burnin)))
#
#     with tqdm(total=(n_burnin + n_samples) * thinning, desc=f'chain {"%6d" % chain_no}, accept = {"%6.4f" % 0}',
#               disable=not debug,
#               # position=seed, leave=True
#               ) as pbar:
#
#         for itr in range(itr0, itr0 + (n_burnin + n_samples) * thinning):
#
#             # scales on the AM draw from covariance and
#             coef_am = scaler * (1.0 - diff_evolution_weight) / scaler_draw_denom
#             coef_de = scaler * diff_evolution_weight / (scaler_draw_denom * np.sqrt(2))
#
#             scalers.append(scaler)
#
#             _t_draw = time.time()
#             try:
#                 am_draw_next = next(draw_generator)
#
#             except:
#                 draw_generator = (d for d in rand_dist.rvs(random_state=rand, size=draw_size))
#                 # draw_generator = (d for d in rand.multivariate_normal(mean=[0.0] * num_params, cov=step_cov.copy(), size=draw_size))
#                 am_draw_next = next(draw_generator)
#
#             draw_time += time.time() - _t_draw
#
#             if (itr - itr0) // thinning > 0 and itr0 > 0 and itr % resample_k == 0:
#                 x1 = draws_nc[rand.randint(0, ((itr - itr0) // thinning))] + coef_am * am_draw_next
#             else:
#                 x1 = x0 + coef_am * am_draw_next
#
#             # TODO remove this Differential Evolution MCMC stuff...?
#             if diff_evolulution_past_samples is not None:
#                 de_draw_next = (diff_evolulution_past_samples[de_idx[itr - itr0, 0]]
#                                 - diff_evolulution_past_samples[de_idx[itr - itr0, 1]])
#                 x1 += coef_de * de_draw_next
#
#             if fix_params_arg is not None:
#                 x1[fix_params_arg['index']] = fix_params_arg['values']
#
#             _t_lp = time.time()
#             lp1 = log_posterior(x1)
#             lp_adj1 = log_posterior_jacobian_adjustment(x1) if do_lp_adjustment else 0.0
#             lp_time += time.time() - _t_lp
#
#             _t_accept = time.time()
#             x0, lp0, lp_adj0, accepted_bool, log_acceptance_prob = \
#                 _acceptance_choice(x0, x1, lp0, lp1, lp_adj0, lp_adj1, uniform_draw=log_random_uniform[itr-itr0])
#
#             accept += accepted_bool
#             accepteds.append(accepted_bool)
#             log_acceptance_probs.append(log_acceptance_prob)
#
#             if itr % thinning == 0:
#                 draws_nc[(itr - itr0) // thinning] = x0
#                 log_posterior_arr[(itr - itr0) // thinning] = lp0
#
#             accept_decision_time += time.time() - _t_accept
#
#             accepted_rate = ((itr - itr0) * accepted_rate + accepteds[-1]) / (itr + 1 - itr0)
#             if do_adaptive:
#                 # gamma_i = 1 / (itr+10)
#                 # scaler = min(max(min_scaler, scaler * np.exp(gamma_i * (accept_prob - target_acceptance_rate))), max_scaler)
#                 # accepted_rate_last_100 = np.mean(accepteds[-100:])
#                 # if accepted_rate_last_100 > 1.025 * target_acceptance_rate:
#                 #     scaler = min(scaler * scaler_scale_up, max_scaler)
#                 # elif accepted_rate_last_100 < .975 * target_acceptance_rate:
#                 #     scaler = max(scaler * scaler_scale_down, min_scaler)
#                 #accepted_rate_last_250 = np.mean(accepteds[-500:])
#                 cur_adust_coef = scaler_adjust_rate / max(itr - 100, 1) ** scaler_adjust_denom_power
#                 scaler = min(
#                     max(
#                         scaler * (1.0 + cur_adust_coef * (accepted_bool - target_acceptance_rate)),
#                         min_scaler
#                     ),
#                     max_scaler
#                 )
#
#             if time.time() - last_update_time > pbar_update_cadence:
#                 last_update_time = time.time()
#                 pbar.update((itr + 1) - itr0 - pbar.n)
#                 if debug:
#                     pbar_str = f'chain {"%4d" % chain_no}, accept = {"%5.3f" % accepted_rate}, ' \
#                                f'accept_L200 = {"%5.2f" % np.mean(accepteds[-200:])}, scaler = {"%5.2e" % scaler}'
#                     pbar.set_description(pbar_str)
#
#         pbar.update((itr + 1) - itr0 - pbar.n)
#         if debug:
#             pbar.set_description(pbar_str)
#
#     np.seterr(**err_settings)
#
#     log_acceptance_probs = np.array(log_acceptance_probs, dtype=float)
#
#     return {
#         'samples': draws_nc,
#         'n_burnin': n_burnin,
#         'n_samples': n_samples,
#         'acceptance_probs': np.exp(log_acceptance_probs),
#         'seed': seed,
#         'accepteds': np.array(accepteds),
#         'scalers': np.array(scalers),
#         'step_cov': step_cov,
#         'step_cov0': step_cov0,
#         # 'cov_samples': np.cov(draws_nc, rowvar=False, ddof=0),
#         # 'mean_samples': np.mean(draws_nc, axis=0),
#         # 'S1': S1,
#         # 'S2': S2,
#         'lp_time': lp_time,
#         'cov_time': cov_time,
#         'draw_time': draw_time,
#         'accept_decision_time': accept_decision_time,
#         'setup_time': setup_time,
#         'fit_elapsed': time.time() - __time_chn_master,
#         'thinning': thinning,
#         'log_posterior': log_posterior_arr,
#         'itr': itr,
#         'bloc': [bloc] * (n_samples + n_burnin),
#         'pid': [os.getpid()] * (n_samples + n_burnin),
#     }
#
#
# # global RUN_AMHA_CHAIN_FUNC_STR
# # RUN_AMHA_CHAIN_FUNC_STR = dumps(run_adaptive_metropolis_chain)
#
#
# def run_adaptive_metropolis_chain_wrapper(args):
#     func = run_adaptive_metropolis_chain
#     return func(*args)
#
#
# def get_subchain_n_draws(n_burnin, n_samples, max_subchain_draws_burnin=5_000, max_subchain_draws_sample=10_000):
#     n_sub_chains_burnin = max(n_burnin // max_subchain_draws_burnin, n_burnin > 0)
#     subchain_n_burnin_iters = [max_subchain_draws_burnin * (n_burnin >= max_subchain_draws_burnin)] * n_sub_chains_burnin
#     if n_burnin:
#         if n_burnin % max_subchain_draws_burnin:
#             if n_burnin > max_subchain_draws_burnin:
#                 n_sub_chains_burnin += 1
#                 subchain_n_burnin_iters.append(0)
#             subchain_n_burnin_iters[-1] += n_burnin % max_subchain_draws_burnin
#
#     n_sub_chains_samples = max(n_samples // max_subchain_draws_sample, n_samples > 0)
#     subchain_n_samples_iters = [max_subchain_draws_sample * (n_samples >= max_subchain_draws_sample)] * n_sub_chains_samples
#     if n_samples:
#         subchain_n_samples_iters[-1] += n_samples % max_subchain_draws_sample
#
#     n_subchains = n_sub_chains_burnin + n_sub_chains_samples
#     subchain_n_burnin_iters = subchain_n_burnin_iters + [0] * n_sub_chains_samples
#     subchain_n_samples_iters = [0] * n_sub_chains_burnin + subchain_n_samples_iters
#     is_burnin_subchain = [True] * n_sub_chains_burnin + [False] * n_sub_chains_samples
#
#     subchain_cnt = 0
#
#     return n_subchains, subchain_n_burnin_iters, subchain_n_samples_iters, is_burnin_subchain, subchain_cnt
#
#
# def amha(log_posterior, x0, x0_is_original_scale=True,
#          step_cov=None, log_posterior_jacobian_adjustment=None, param_names=None, specification_name=None,
#          n_chains=DEFAULT_AM_N_CHAINS,
#          n_burnin=DEFAULT_AM_N_BURNIN, n_samples=DEFAULT_AM_N_SAMPLES, max_processes=DEFAULT_AM_MAX_PROCESSES,
#          seed=None, target_acceptance_rate=DEFAULT_AM_TARGET_ACCEPTANCE_RATE, draw_size=DEFAULT_AM_DRAW_SIZE,
#          scaler0=None, min_scaler=DEFAULT_AM_MIN_SCALER, max_scaler=DEFAULT_AM_MAX_SCALER,
#          scaler_adjust_rate=DEFAULT_AM_SCALER_ADJUST_RATE,
#          thinning=DEFAULT_AM_THINNING, scaler_adjust_denom_power=DEFAULT_AM_SCALER_ADJUST_DENOM_POWER,
#          pbar_update_cadence=DEFAULT_AM_PBAR_UPDATE_CADENCE,
#          bounds=None, do_adaptive=DEFAULT_AM_DO_ADAPTIVE,
#          max_subchain_draws_burnin=DEFAULT_AM_MAX_SUBCHAIN_DRAWS_BURNIN,
#          max_subchain_draws_sample=DEFAULT_AM_MAX_SUBCHAIN_DRAWS_SAMPlE,
#          do_parallel=DEFAULT_AM_DO_PARALLEL, user_prompt_for_more_iters=DEFAULT_AM_USER_PROMPT_FOR_MORE_ITERS,
#          debug=False, proposal_df=DEFAULT_AM_PROPOSAL_DF,
#          resample_k=DEFAULT_AM_RESAMPLE_K, step_cov_adjust_rate=DEFAULT_AM_STEP_COV_ADJUST_RATE,
#          # ridge_epsilon=DEFAULT_AM_RIDGE_EPSILON,
#          # min_unique_draws_for_cov=DEFAULT_AM_MIN_UNIQUE_DRAWS_FOR_COV, cov_update_frequency=DEFAULT_AM_COV_UPDATE_FREQUENCY,
#          model=None, fix_params=None, transformations=None, show_r_hat_ever_subchain=False, starting_iter=0,
#          # do_metropolis_de=False, de_scaler_past_samples=1.0, de_past_samples=None, de_frac_burnin=.33,
#          normalize_step_cov=DEFAULT_AM_NORMALIZE_STEP_COV,
#
#          do_diff_evolution_mc=DEFAULT_AM_DO_DIFF_EVOLUTION_MC,
#          diff_evolution_past_samples=None,
#          diff_evolution_frac_burnin=DEFAULT_AM_DIFF_EVOLUTION_FRAC_BURNIN,
#          diff_evolution_max_draws=DEFAULT_AM_DIFF_EVOLUTION_MAX_DRAWS,
#          diff_evolution_weight=DEFAULT_AM_DIFF_EVOLUTION_WEIGHT,
#
#          stop_adaptation_after_burnin=DEFAULT_AM_STOP_ADAPTATION_AFTER_BURNIN,
#
#          ) -> MCMCResults:
#     """
#
#     :param log_posterior:
#     :param x0:
#     :param step_cov:
#     :param log_posterior_jacobian_adjustment:
#     :param param_names:
#     :param specification_name:
#     :param n_chains:
#     :param n_burnin:
#     :param n_samples:
#     :param max_processes:
#     :param seed:
#     :param target_acceptance_rate:
#     :param draw_size:
#     :param scaler0:
#     :param min_scaler:
#     :param max_scaler:
#     :param scaler_adjust_rate:
#     :param thinning:
#     :param scaler_adjust_denom_power:
#     :param pbar_update_cadence:
#     :param bounds:
#     :param do_adaptive:
#     :param max_subchain_draws:
#     :param do_parallel:
#     :param user_prompt_for_more_iters:
#     :param debug:
#     :param proposal_df:
#     :param resample_k:
#     :param step_cov_adjust_rate:
#     :param model:
#     :param fix_params: dict mapping index to fixed parameter value
#     :return:
#     """
#
#     _time_master = time.time()
#
#     if do_diff_evolution_mc:
#         assert isinstance(diff_evolution_weight, float) and 0 < diff_evolution_weight < 1
#
#     if debug:
#         print("----------------------------------")
#         print("Beginning Adaptive Metropolis MCMC")
#         print("----------------------------------")
#
#     if not debug and show_r_hat_ever_subchain:
#         raise Exception("show_r_hat_ever_subchain=True and debug=False not allowed!")
#
#     x0s_formated, transformations, fix_params, fix_params_transformed, param_names, num_params, transformation_function =\
#         format_starting_point_for_mcmc(x0, x0_is_original_scale, n_chains, transformations, fix_params, param_names,
#                                        debug=debug)
#     check_starting_point(x0s_formated, log_posterior, log_posterior_jacobian_adjustment, debug)
#
#     has_fixed_params = fix_params_transformed is not None and len(fix_params_transformed)
#     if has_fixed_params:
#         fixed_param_idx, fixed_param_vals = list(fix_params_transformed.keys()), list(fix_params_transformed.values())
#         not_fixed_param_index = [a for a in range(num_params) if a not in fixed_param_idx]
#         fix_params_arg = {'index': fixed_param_idx, 'values': fixed_param_vals}
#     else:
#         fixed_param_idx = []
#         not_fixed_param_index = range(num_params)
#         fix_params_arg = None
#         fix_params = None
#
#     if step_cov is None:
#         step_cov = np.eye(len(x0s_formated[0])) * min(len(x0s_formated[0]) ** -2, .01)
#
#     scaler0_orig = scaler0
#     step_cov0_orig = step_cov.copy()
#
#     log_posterior_bounded = log_posterior
#     if bounds is not None:
#         if isinstance(bounds, dict):
#             bounds_arr = np.ones((2, num_params))
#             for i, nm in enumerate(param_names):
#                 bounds_arr[:, i] = bounds.get(nm, [-np.inf, np.inf])
#             bounds = bounds_arr
#         bounds = np.array(bounds.copy())
#         assert np.prod(np.shape(bounds)) == 2 * num_params
#         assert bounds.shape[0] == 2
#
#         def log_posterior_bounded(x):
#             if np.any(x < bounds[0]) or np.any(x > bounds[1]):
#                 return -np.inf
#             return log_posterior(x)
#
#     if debug:
#         if specification_name is not None:
#             print('\nSpecification Name: ', specification_name, '\n')
#
#         info = np.array([
#             ('Settings', ''),
#             ('   n_params:', num_params),
#             ('   n_chains:', n_chains),
#             ('   max_subchain_draws_burnin:', max_subchain_draws_burnin),
#             ('   max_subchain_draws_sample:', max_subchain_draws_sample),
#             ('   do_parallel:', do_parallel),
#             ('   thinning:', thinning),
#             ('   n_burnin:', n_burnin),
#             ('   n_samples:', n_samples),
#             ('   seed:', seed),
#             ('   draw_size:', draw_size),
#             ('   target_acceptance_rate:', target_acceptance_rate),
#             ('   adaptive:', do_adaptive),
#             ('   scaler0:', scaler0),
#             ('   min_scaler:', min_scaler),
#             ('   max_scaler:', max_scaler),
#             ('   scaler_adjust_rate:', scaler_adjust_rate),
#             ('   scaler_adjust_denom_power:', scaler_adjust_denom_power),
#             ('   proposal_df:', proposal_df),
#             ('   resample_k:', resample_k),
#             ('   step_cov_adjust_rate:', step_cov_adjust_rate),
#             ('   fixed_params:', fix_params is not None),
#             ('   normalize step cov: ', normalize_step_cov),
#
#             ('   do_diff_evolution_mc:', do_diff_evolution_mc),
#             ('   diff_evolution_max_draws:', diff_evolution_max_draws),
#             ('   diff_evolution_frac_burnin:', diff_evolution_frac_burnin),
#             ('   diff_evolution_weight:', diff_evolution_weight),
#             ('   stop_adaptation_after_burnin:', stop_adaptation_after_burnin),
#
#             # ('   cov_update_frequency:', cov_update_frequency),
#             # ('   min_unique_draws_for_cov:', min_unique_draws_for_cov),
#         ])
#         print()
#         print(pd.Series(info[:, 1], index=info[:, 0]).to_string())
#         print()
#
#     if seed is None:
#         seed = 0
#     rand = np.random.RandomState(seed)
#
#     chain_nos = range(n_chains)
#
#     if debug:
#         print('\nBeginnning MCMC chains now...')
#
#     chain_results_master = [None] * n_chains
#
#     n_subchains, subchain_n_burnin_iters, subchain_n_samples_iters, is_burnin_subchain, subchain_cnt \
#         = get_subchain_n_draws(n_burnin, n_samples, max_subchain_draws_burnin=max_subchain_draws_burnin,
#                                max_subchain_draws_sample=max_subchain_draws_sample)
#
#     if do_diff_evolution_mc:
#         if stop_adaptation_after_burnin and diff_evolution_past_samples is None:
#             if np.count_nonzero(subchain_n_burnin_iters) < 2:
#                 raise Exception("`do_diff_evolution_mc` ineffective if less than two burnin sub-chains. "
#                                 "\nConsider ratio of `n_burnin` to `max_subchain_draws_burnin`!")
#
#     mean_master = None
#     n_burnin_master = 0
#     n_samples_master = 0
#
#     if starting_iter is None:
#         itr0 = 0
#     else:
#         itr0 = starting_iter
#
#     bloc = 0
#
#     #global log_posterior_bounded_str, log_posterior_jacobian_adjustment_str
#     #log_posterior_bounded_str = dumps(log_posterior_bounded)
#     #log_posterior_jacobian_adjustment_str = dumps(log_posterior_jacobian_adjustment)
#     log_posterior_bounded_str, log_posterior_jacobian_adjustment_str = log_posterior_bounded, log_posterior_jacobian_adjustment
#
#     while True:
#
#         if normalize_step_cov:
#             # todo normalize by something else?
#             step_cov_iter, _, _ = normalize_matrix_det(step_cov, not_fixed_param_index, 1.0)
#         else:
#             step_cov_iter = step_cov
#
#         bloc += 1
#         chain_args = [
#             (
#                 seed, log_posterior_bounded_str, x0.copy(),
#                 step_cov_iter.copy(), log_posterior_jacobian_adjustment_str,
#                 subchain_n_burnin_iters[subchain_cnt], subchain_n_samples_iters[subchain_cnt], itr0,
#                 target_acceptance_rate, ch_num, draw_size, scaler0, min_scaler, max_scaler, scaler_adjust_rate,
#                 scaler_adjust_denom_power, thinning, pbar_update_cadence, do_adaptive, debug,
#                 # cov_update_frequency, min_unique_draws_for_cov, ridge_epsilon,
#                 proposal_df, resample_k, bloc, fix_params_arg,
#                 diff_evolution_past_samples, diff_evolution_weight if diff_evolution_past_samples is not None else 0,
#             )
#             for seed, x0, ch_num, step_cov in zip(rand.randint(0, 100_000, n_chains),
#                                                   x0s_formated, chain_nos, [step_cov.copy()] * n_chains)
#         ]
#
#         if debug:
#             print(dict(subchain=f'{subchain_cnt + 1}/{n_subchains}', n_burnin=subchain_n_burnin_iters[subchain_cnt],
#                        n_sample=subchain_n_samples_iters[subchain_cnt], is_burnin=is_burnin_subchain[subchain_cnt]))
#
#         if do_parallel:
#             time.sleep(.0005)
#
#             num_processes = min(max_processes, n_chains)
#             pool = Pool(processes=num_processes, initargs=(RLock(),), initializer=tqdm.set_lock)
#             #pool = ProcessPool(num_processes, initargs=(RLock(),), initializer=tqdm.set_lock)
#             try:
#                 # jobs = [
#                 #     pool.apply_async(run_adaptive_metropolis_chain, _arg) for _arg in chain_args
#                 # ]
#                 #
#                 # pool.close()
#                 # pool.join()
#
#                 #chain_results = [c for c in pool.imap(lambda x: run_adaptive_metropolis_chain(*x), chain_args)]
#
#                 #
#                 # chain_results = [job.get(0) for job in jobs]
#
#                 results = pool.map(
#                     run_adaptive_metropolis_chain_wrapper,
#                     chain_args)
#                 chain_results = list(results)
#
#                 pool.close()
#                 pool.join()
#
#                 #chain_results = [job.get() for job in jobs]
#
#                 if debug:
#                     print('\nJoining the data from the parallel jobs...')
#
#                 pool.terminate()
#
#
#             except KeyboardInterrupt:
#                 with warnings.catch_warnings():
#                     warnings.simplefilter("ignore")
#                     pool.terminate()
#                 if n_samples_master == 0:
#                     n_samples_master = n_burnin_master // 3
#                     n_burnin_master = n_burnin_master - n_samples_master
#                 warnings.warn("\n\nKeyboard Interrupt! Stopping draws.\n\n")
#                 break
#
#             except Exception as e:
#                 pool.terminate()
#                 print("\n!!!"*4)
#                 print(e.__traceback__)
#                 raise e
#
#         else:
#             try:
#                 chain_results = [run_adaptive_metropolis_chain(*_arg) for _arg in chain_args]
#             except KeyboardInterrupt:
#                 if n_samples_master == 0:
#                     n_samples_master = n_burnin_master // 3
#                     n_burnin_master = n_burnin_master - n_samples_master
#                 warnings.warn("\n\nKeyboard Interrupt! Stopping draws.\n\n")
#                 break
#             except Exception as e:
#                 raise e
#
#         for i, (chn_master, chn_new) in enumerate(zip(chain_results_master, chain_results)):
#
#             if i == 0:  # we only count burnin *per* chain
#                 n_burnin_master += chn_new['n_burnin']
#
#             n_samples_master += chn_new['n_samples']
#
#             if chn_master is None:
#                 chain_results_master[i] = chain_results[i]
#
#                 # add mean and cov
#                 chain_results_master[i]['mean_draws'] = np.mean(chain_results[i]['samples'], axis=0)
#                 chain_results_master[i]['cov_draws'] = np.cov(chain_results[i]['samples'], ddof=0, rowvar=False)
#
#             else:
#
#                 new_mean, new_cov = (
#                     np.mean(chn_new['samples'], axis=0),
#                     np.cov(chn_new['samples'], ddof=0, rowvar=False)
#                 )
#
#                 cov_chn, mean_chn = aggregate_covs(
#                     [chn_master['cov_draws'], new_cov],
#                     [chn_master['mean_draws'], new_mean],
#                     [len(chn_master['samples']), len(chn_new['samples'])]
#                 )
#
#                 new_ch_dict = {
#                     **{
#
#                         'samples': np.vstack([chn_master['samples'], chn_new['samples']]),
#
#                         'seed': chn_master['seed'],
#                         'accepteds': np.hstack([chn_master['accepteds'], chn_new['accepteds']]),
#                         'scalers': np.hstack([chn_master['scalers'], chn_new['scalers']]),
#                         'acceptance_probs': np.hstack([chn_master['acceptance_probs'], chn_new['acceptance_probs']]),
#
#                         'cov_draws': cov_chn,
#                         'mean_draws': mean_chn,
#
#                         'step_cov0': chn_master['step_cov0'],
#
#                         'thinning': thinning,
#                         'log_posterior': np.hstack([chn_master['log_posterior'], chn_new['log_posterior']]),
#                         'itr': chn_new['itr'],
#
#                     },
#                     **{
#                         kk: chn_master[kk] + chn_new[kk]
#                         for kk in ['lp_time', 'cov_time', 'draw_time', 'accept_decision_time', 'fit_elapsed',
#                                    'setup_time', 'n_burnin', 'n_samples', 'bloc', 'pid']
#                     }
#                 }
#
#                 chain_results_master[i] = new_ch_dict
#
#             x0s_formated[i] = chain_results[i]['samples'][-1].copy()
#             #temp = chain_results[i]['samples'][-1].copy()
#             #if log_posterior_bounded(x0s[i]) < log_posterior(temp):
#             #    x0s[i] = temp
#
#         subchain_cnt += 1
#
#         cov_new_master, _ = aggregate_covs(
#             [c['cov_draws'] for c in chain_results_master],
#             [c['mean_draws'] for c in chain_results_master],
#             [len(c['samples']) for c in chain_results_master],
#         )
#
#         step_cov_old = step_cov
#         if is_burnin_subchain or not stop_adaptation_after_burnin:
#             step_cov = step_cov_adjust_rate * cov_new_master + (1.0 - step_cov_adjust_rate) * step_cov_old
#
#         if has_fixed_params:
#             for j in fix_params_arg['index']:
#                 step_cov[:, j] = 0.
#                 step_cov[j, :] = 0.
#                 step_cov[j, j] = 1.
#
#         if debug and show_r_hat_ever_subchain:
#
#             print("\nMaximum Log Posterior (so far..):")
#             print(pd.DataFrame(
#                 {'max lp': [c['log_posterior'].max() for c in chain_results_master]}
#             ).transpose().to_string())
#
#             if is_burnin_subchain or not stop_adaptation_after_burnin:
#                 step_cov_slog_dets = {
#                     'step_cov_old': np.linalg.slogdet(step_cov_old[not_fixed_param_index][:, not_fixed_param_index])[1],
#                     'step_cov_update': np.linalg.slogdet(cov_new_master[not_fixed_param_index][:, not_fixed_param_index])[1],
#                     'step_cov_updated': np.linalg.slogdet(step_cov[not_fixed_param_index][:, not_fixed_param_index])[1],
#                 }
#                 step_cov_slog_dets['relative_scales'] = np.exp(
#                     step_cov_slog_dets['step_cov_update'] - step_cov_slog_dets['step_cov_old'])
#
#                 print("\nStep Cov slogdet (log Generalized Variance):")
#                 pprint.pprint(step_cov_slog_dets)
#
#         if subchain_cnt < n_subchains:
#             if debug:
#
#                 if show_r_hat_ever_subchain:
#                     prompt_message = get_diagnostic_update_message(
#                         [c['samples'] for c in chain_results_master], param_names, n_chains,
#                         _time_master, transformation_function, fix_param_idx_subset=fixed_param_idx)
#                     print(prompt_message)
#
#                 print(f'\n(Current run time is {"%.2fs" % (time.time() - _time_master)})\n')
#
#         else:
#             if user_prompt_for_more_iters:
#
#                 prompt_message = get_diagnostic_update_message(
#                     [c['samples'] for c in chain_results_master], param_names, n_chains,
#                     _time_master, transformation_function, fix_param_idx_subset=fixed_param_idx)
#
#                 new_draws = user_prompt_for_more_iters_method(prompt_message, do_prompt=True,
#                                                               print_not_converged=False, assert_even=True)
#                 if new_draws <= 0:
#                     break
#
#                 n_samples = new_draws
#                 n_burnin = 0
#
#                 n_subchains, subchain_n_burnin_iters, subchain_n_samples_iters, is_burnin_subchain, subchain_cnt \
#                     = get_subchain_n_draws(n_burnin, n_samples,
#                                            max_subchain_draws_burnin=max_subchain_draws_burnin,
#                                            max_subchain_draws_sample=max_subchain_draws_sample)
#
#             else:
#                 break
#
#         if do_diff_evolution_mc:
#             if not stop_adaptation_after_burnin or is_burnin_subchain:
#                 diff_evolution_past_samples = np.vstack([
#                     c['samples'][rand.randint(
#                         low=int(diff_evolution_frac_burnin * len(c['samples'])),
#                         high=len(c['samples']),
#                         size=min(
#                             diff_evolution_max_draws // n_chains,
#                             int((1 - diff_evolution_frac_burnin) * len(c['samples']))
#                         )
#                     )]
#                     for c in chain_results_master
#                 ])
#
#         scaler0 = np.mean([np.mean(c['scalers'][-100:]) for c in chain_results_master])
#         itr0 = chain_results_master[0]['itr']
#
#         mean_master_new = [c['mean_draws'] for c in chain_results_master]
#         # if debug and mean_master is not None:
#         #     print("Diff in Means")
#         #     print(f"{'Chain':7s}{'Max Abs':10s}{'Max Rel':10s}")
#         #     print("-"*27)
#         #     for i, (c1, c2) in enumerate(zip(mean_master, mean_master_new)):
#         #         print(f"{i:7d}{np.abs(c1 - c2).max():10.2e}{(np.abs(c1 - c2) / (np.abs(c2) + 1e-8)).max():10.2e}")
#         #     print("-" * 27)
#         mean_master = mean_master_new
#
#     cov_params_unbndd, mean_params_unbndd = convert_samples_to_unbounded_space(
#         chain_results_master, transformations, num_params, debug=debug, key='samples')
#
#     options = {'method': 'Adaptive Metropolis',
#                'min_scaler': min_scaler, 'max_scaler': max_scaler, 'scaler_adjust_rate': scaler_adjust_rate,
#                'scaler_adjust_denom_power': scaler_adjust_denom_power,
#                'scaler0': scaler0_orig, 'draw_size': draw_size,
#                'target_acceptance_rate': target_acceptance_rate, 'seed': seed,
#                'max_processes': max_processes, 'log_posterior_bounded': log_posterior_bounded,
#                'log_posterior': log_posterior,
#                'max_subchain_draws_burnin': max_subchain_draws_burnin,
#                'max_subchain_draws_sample': max_subchain_draws_sample,
#                'n_burnin': n_burnin_master, 'n_samples': n_samples_master, 'n_chains': n_chains,
#                'x0': x0.copy(),
#                'step_cov': step_cov0_orig.copy(), 'do_adaptive': do_adaptive,
#                'bounds': bounds.copy() if bounds is not None else None,
#                'proposal_df': proposal_df,
#                'do_parallel': do_parallel,
#                'resample_k': resample_k,
#                'show_r_hat_ever_subchain': show_r_hat_ever_subchain,
#                'step_cov_adjust_rate': step_cov_adjust_rate,
#                'pbar_update_cadence': pbar_update_cadence,
#                'normalize_step_cov': normalize_step_cov,
#                'thinning': thinning,
#                'stop_adaptation_after_burnin': stop_adaptation_after_burnin,
#
#                # DE-MC params
#                'do_diff_evolution_mc': do_diff_evolution_mc,
#                'diff_evolution_frac_burnin': diff_evolution_frac_burnin,
#                'diff_evolution_max_draws': diff_evolution_max_draws,
#                'diff_evolution_weight': diff_evolution_weight,
#                }
#
#     other_info = dict(
#         scaler=scaler0,
#         step_cov=step_cov,
#         cov_params_unbounded_space=cov_params_unbndd,
#         mean_params_unbounded_space=mean_params_unbndd,
#         diff_evolution_past_samples=None if diff_evolution_past_samples is None else diff_evolution_past_samples.copy(),
#     )
#     mcmc_time = time.time() - _time_master
#
#     if debug:
#         print(f'\nMCMC Drawing Complete... {"%.2fs" % mcmc_time}')
#
#     return MCMCResults('Adaptive Metropolis', num_params, log_posterior, log_posterior_jacobian_adjustment, param_names,
#                        chain_results_master,
#                        mcmc_time, n_burnin_master, n_chains, thinning, options, specification_name, debug=False,
#                        model=model, fix_params=fix_params, other_info=other_info, transformations=transformations
#                        )
#
#
#
#
# # if __name__ == '__main__':
# #
# #     n = 500
# #     np.random.seed(0)
# #     x = np.random.randn(n)
# #     y = 1.3 * x + np.random.randn(n) * .5
# #     data = {'x': x, 'y': y}
# #
# #     from kanly.regression.nonlinear_least_squares.model import SparseNonlinearLeastSquaresModel
# #     nlls_model = SparseNonlinearLeastSquaresModel.build_model_from_formula(
# #         '[y]~{x}*[x]', data
# #     )
# #     llf = nlls_model.get_log_likelihood_function()
# #
# #     fit = mcmc(llf, [0,1], n_chains=4, n_burnin=100, n_samples=100, param_names=['x', '__sigma2'],
# #                fix_params={1: 1.1}, user_prompt_for_more_iters=True,
# #                debug=True)
# #     print(fit)
# #
# #     print(fit.sample_df)
# #     print(fit.sample_df.cov())
# #     print(fit.sample_df['__sigma2'].unique())
# #     print(fit.sample_df[fit.sample_df['__sigma2']==1])
# #
# #     # l2_penalty = {'x': 50.}
# #     # reg_2_vals = {'x': 5}
# #
# # if __name__ == '__main__':
# #     from kanly.api import bayes_nonlinear_regression_model
# #     import numpy as np
# #     import pandas as pd
# #     import matplotlib.pyplot as plt
# #
# #     np.random.seed(0)
# #     n = 30
# #     x = 3.7 * np.random.rand(n)
# #     y = 1.2 + .9 * x ** .5 + .3 * np.random.randn(n)
# #     data = dict(x=x, y=y)
# #     plt.scatter(x, y)
# #
# #     model = bayes_nonlinear_regression_model('[y] ~ {a} + {b}*[x]**(1-{c})', data, bounds={'c': [0, .995]},
# #                                              do_bounded_transform=False, do_njit=False)
# #
# #     t = time.time()
# #     M = 5000
# #     for i in range(M):
# #         model.log_likelihood_function([1, 1, .7, .5])
# #     print(">  ", M/(time.time() - t))
# #
# #     M = 5000
# #     _acceptance_choice(1, 1, 1, 1, 1.4, 1, 1)
# #     for i in range(M):
# #         _acceptance_choice(1,1,1,1,1.4,1,1)
# #     print(">  ", M/(time.time() - t))
# #
# #     fit = model.mcmc([1, 1, .7, .5], n_burnin=30000, n_samples=50_000, max_subchain_draws=20_000,
# #                      user_prompt_for_more_iters=True, do_parallel=True, debug=True)
# #
# # #     print(fit.chain_results)
# # if __name__ == '__main__':
# #     from kanly.api import bayes_nonlinear_regression_model
# #
# #     np.random.seed(0)
# #     n = 600
# #     x = .56 * 1.2 *np.random.rand(n)
# #     z = np.random.rand(n)
# #     y = 3 + 10 * x - 2 * z + np.random.randn(n) * 3
# #     wts = .01 + np.random.rand(n)
# #     data = {'x': x, 'y': y, 'z': z, 'wts': wts}
# #
# #     model = bayes_nonlinear_regression_model(
# #         '[y] ~ {a}+{b}*[x]**{c}', data, bounds={'b': [5, 8], 'a': [1,6], 'c': [.05, .98]}
# #     )
# #
# #     n_burn, n_samp = 5_000, 50_000
# #
# #     fit1 = model.amha(
# #         [2, 6, .5, 1.], user_prompt_for_more_iters=False,
# #         debug=True,
# #         n_chains=4,
# #         n_burnin=n_burn,
# #         n_samples=n_samp,
# #         show_r_hat_ever_subchain=True,
# #         max_subchain_draws_burnin=5_000,
# #         max_subchain_draws_sample=15_000,
# #         thinning=2,
# #         # fix_params={'a': 2.6},
# #         # do_metropolis_de=True,
# #         # de_frac_burnin=.33,
# #     )
# #     print(fit1)
# #
# #     fit2 = model.amha(
# #         [2, 6, .5, 1.], user_prompt_for_more_iters=False,
# #         debug=True,
# #         n_chains=4,
# #         n_burnin=n_burn,
# #         n_samples=n_samp,
# #         show_r_hat_ever_subchain=True,
# #         max_subchain_draws_burnin=5_000,
# #         max_subchain_draws_sample=35_000,
# #         do_diff_evolution_mc=True,
# #         diff_evolution_weight=.9,
# #         thinning=2,
# #     )
# #     print(fit2)
# #
# #     print()
# #     print('$' * 200)
# #     print()
# #
# #     print(fit1)
# #     print(fit2)
# #
# #     fit1.multi_hist(['b', 'c'], show=True, suptitle='fit1')
# #     fit2.multi_hist(['b', 'c'], show=True, suptitle='fit2')
