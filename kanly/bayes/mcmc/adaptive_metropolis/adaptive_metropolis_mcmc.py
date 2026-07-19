"""
Adaptive Metropolis–Hastings (AMH) MCMC sampler with optional Differential Evolution.

This module implements the ``amha`` function—the primary entry point for running
parallel Adaptive Metropolis MCMC chains—and the lower-level helpers it relies on:

- ``run_adaptive_metropolis_chain``: single-chain kernel that draws proposals from a
  multivariate normal with an adaptively scaled covariance, applies the
  Metropolis–Hastings accept/reject step, and optionally mixes in Differential
  Evolution (DE-MC) jumps between chains.
- ``_acceptance_choice``: pure Metropolis–Hastings decision given log-posteriors
  and an optional Jacobian adjustment for bounded-to-unbounded transforms.
- ``normalize_matrix_det`` / ``make_pos_semidef``: numerical utilities for
  keeping the proposal covariance matrix well-conditioned.
- ``get_subchain_n_draws``: breaks the requested total draw count into sub-chains
  whose length is capped at ``max_subchain_draws_{burnin,sample}`` so that Ray
  remote tasks stay responsive and covariance estimates can be updated between runs.

Parallelism is handled by Ray: when ``do_parallel=True`` each chain is dispatched
as a Ray remote task via ``run_adaptive_metropolis_chain_remote``.
"""
from __future__ import absolute_import, print_function

import os
import time
import warnings

import numpy as np
import psutil
# import ray
# from ray.experimental import tqdm_ray
from scipy.stats import multivariate_normal

from kanly.bayes.mcmc.adaptive_metropolis.constants import DEFAULT_AM_N_BURNIN, DEFAULT_AM_N_SAMPLES, \
    DEFAULT_AM_N_CHAINS, DEFAULT_AM_THINNING, DEFAULT_AM_DRAW_SIZE, DEFAULT_AM_TARGET_ACCEPTANCE_RATE, \
    DEFAULT_AM_MAX_PROCESSES, DEFAULT_AM_PBAR_UPDATE_CADENCE, DEFAULT_AM_DO_ADAPTIVE, \
    DEFAULT_AM_DO_PARALLEL, DEFAULT_AM_USER_PROMPT_FOR_MORE_ITERS, \
    DEFAULT_AM_SCALER_ADJUST_RATE, DEFAULT_AM_SCALER_ADJUST_DENOM_POWER, DEFAULT_AM_MIN_SCALER, DEFAULT_AM_MAX_SCALER, \
    DEFAULT_AM_MAX_SUBCHAIN_DRAWS_BURNIN, DEFAULT_AM_MAX_SUBCHAIN_DRAWS_SAMPlE, \
    DEFAULT_AM_PROPOSAL_DF, DEFAULT_AM_STEP_COV_INITIAL_SAMPLES, DEFAULT_AM_NORMALIZE_STEP_COV, \
    DEFAULT_AM_DO_DIFF_EVOLUTION_MC, DEFAULT_AM_DIFF_EVOLUTION_FRAC_BURNIN, \
    DEFAULT_AM_DIFF_EVOLUTION_MAX_DRAWS, DEFAULT_AM_DIFF_EVOLUTION_WEIGHT, DEFAULT_AM_STOP_ADAPTATION_AFTER_BURNIN, \
    DEFAULT_AM_DIFF_EVOLUTION_JUMP_CADENCE, DEFAULT_AM_SCALAR_JITTER_BOUNDS, METHOD as AMH_METHOD
from kanly.bayes.mcmc.aggregate_covariances import aggregate_covs, get_total_covariance_from_batches
from kanly.bayes.mcmc.check_starting_point import check_starting_point
from kanly.bayes.mcmc.diagnostics.diagnostics import get_diagnostic_update_message
from kanly.bayes.mcmc.format_starting_point_for_mcmc import format_starting_point_for_mcmc
from kanly.bayes.mcmc.mcmc_results import MCMCResults
from kanly.bayes.parameter_transformations import convert_samples_to_unbounded_space
from kanly.utils.print_options import print_options
from kanly.utils.user_prompt_for_more_iters import user_prompt_for_more_iters_method


# from tqdm import tqdm


def _get_process_memory_mb(pid=None):
    """Return the RSS and VMS memory usage of a process in megabytes.

    Uses ``psutil`` to query the process memory info.  Returns ``(-1, -1)``
    if the query fails (e.g. the process has already exited).

    Args:
        pid: Process ID to query.  Defaults to the current process when ``None``.

    Returns:
        2-tuple ``(rss_mb, vms_mb)`` of resident-set and virtual-memory sizes
        in megabytes, or ``(-1, -1)`` on error.
    """
    if pid is None:
        pid = os.getpid()
    try:
        proc = psutil.Process(pid)
        return proc.memory_info().rss / 1024 ** 2, proc.memory_info().vms / 1024 ** 2
    except:
        return -1, -1


def normalize_matrix_det(V, not_fixed_param_index=None, normalizer_value=1.0):
    """Scale a positive semi-definite matrix so its determinant equals ``normalizer_value``.

    Computes the signed log-determinant of the free-parameter sub-matrix of ``V``
    and multiplies the full matrix by the appropriate scalar factor.  This is not
    currently used in the default sampling path but is available for schemes that
    normalize the proposal covariance determinant.

    Args:
        V: Square positive semi-definite matrix (NumPy array) to normalize.
        not_fixed_param_index: List of free (non-fixed) parameter indices used to
            compute the determinant sub-matrix.  When ``None`` or equal in length
            to the full matrix, the full matrix is used.
        normalizer_value: Target determinant value (must be > 0).

    Returns:
        3-tuple ``(V_normalized, normalizer, log_det_normalized)`` where
        ``V_normalized`` is the rescaled matrix, ``normalizer`` is the scalar
        multiplier applied, and ``log_det_normalized`` is the log-determinant
        of the normalized sub-matrix.
    """
    assert normalizer_value > 0
    V = np.asarray(V)
    nn = V.shape[0]
    if not_fixed_param_index is None or len(not_fixed_param_index) == nn:
        V_sub = V
    else:
        V_sub = V[not_fixed_param_index][:, not_fixed_param_index]
    slog_det = np.linalg.slogdet(V_sub)
    assert slog_det[0] > 0
    normalizer = np.exp((np.log(normalizer_value) - slog_det[1]) / len(V_sub))
    return normalizer * V, normalizer, np.linalg.slogdet(normalizer * V_sub)[1]


def make_pos_semidef(step_cov):
    """Ensure a covariance matrix is positive semi-definite by diagonal perturbation.

    Repeatedly attempts a Cholesky factorization of ``step_cov``.  On failure
    (``LinAlgError``) adds a small identity multiple to the diagonal, growing
    the increment by 1.5× each attempt until the factorization succeeds.
    Scalar (0-D array) inputs are handled as a degenerate 1×1 case.

    Args:
        step_cov: Covariance matrix (2-D NumPy array) or scalar to repair.

    Returns:
        Positive semi-definite version of ``step_cov``, possibly with small
        diagonal additions.
    """
    # Make a matrix positive semidefinite by adding to it's diagonal
    is_pos_def = False
    incr = 1e-6

    if np.shape(step_cov) == tuple():
        if step_cov < 0:
            step_cov = np.array([[incr]])

    else:
        while not is_pos_def:
            try:
                np.linalg.cholesky(step_cov)
                is_pos_def = True
                break
            except np.linalg.LinAlgError:
                step_cov = step_cov + np.diag([incr] * len(step_cov))
                incr *= 1.5

    return step_cov


def _acceptance_choice(x0, x1, lp0, lp1, lp_adj0, lp_adj1, uniform_draw):
    """Execute one Metropolis–Hastings accept/reject step.

    The acceptance probability is ``min(1, exp((lp1 + lp_adj1) - (lp0 + lp_adj0)))``.
    The Jacobian adjustment sparse_terms ``lp_adj0`` / ``lp_adj1`` account for
    bounded-to-unbounded parameter transformations so that the chain targets
    the correct posterior on the original (bounded) scale.

    Args:
        x0: Current parameter vector (NumPy array, unbounded space).
        x1: Proposed parameter vector (NumPy array, unbounded space).
        lp0: Log-posterior evaluated at ``x0``.
        lp1: Log-posterior evaluated at ``x1``; automatically rejected if not finite.
        lp_adj0: Jacobian log-determinant adjustment at ``x0`` (0.0 if no transform).
        lp_adj1: Jacobian log-determinant adjustment at ``x1`` (0.0 if no transform).
        uniform_draw: Pre-drawn log-uniform random variate for the accept/reject comparison.

    Returns:
        5-tuple ``(x_new, lp_new, lp_adj_new, accepted_bool, log_acceptance_prob)``
        where the first three are the retained state, ``accepted_bool`` indicates
        whether the proposal was accepted, and ``log_acceptance_prob`` is
        ``min(0, lp_gap)``.
    """

    if not np.isfinite(lp1):
        accepted_bool = False
        log_acceptance_prob = -np.inf

    else:

        lp_gap = (lp1 + lp_adj1) - (lp0 + lp_adj0)
        log_acceptance_prob = min(lp_gap, 0.0)

        if lp_gap >= 0 or uniform_draw <= lp_gap:
            accepted_bool = True

        else:
            accepted_bool = False

    if accepted_bool:
        x_new, lp_new, lp_adj_new = x1, lp1, lp_adj1
    else:
        x_new, lp_new, lp_adj_new = x0, lp0, lp_adj0

    return x_new, lp_new, lp_adj_new, accepted_bool, log_acceptance_prob


def run_adaptive_metropolis_chain(seed, log_posterior, x0, step_cov, log_posterior_jacobian_adjustment=None,
                                  n_burnin=DEFAULT_AM_N_BURNIN, n_samples=DEFAULT_AM_N_SAMPLES, itr0=0,
                                  target_acceptance_rate=DEFAULT_AM_TARGET_ACCEPTANCE_RATE, chain_no=None,
                                  draw_size=DEFAULT_AM_DRAW_SIZE, scaler0=None, min_scaler=DEFAULT_AM_MIN_SCALER,
                                  max_scaler=DEFAULT_AM_MAX_SCALER, scaler_adjust_rate=DEFAULT_AM_SCALER_ADJUST_RATE,
                                  scaler_adjust_denom_power=DEFAULT_AM_SCALER_ADJUST_DENOM_POWER,
                                  thinning=DEFAULT_AM_THINNING, pbar_update_cadence=DEFAULT_AM_PBAR_UPDATE_CADENCE,
                                  do_adaptive=DEFAULT_AM_DO_ADAPTIVE, debug=False,
                                  proposal_df=DEFAULT_AM_PROPOSAL_DF,
                                  #resample_k=DEFAULT_AM_RESAMPLE_K,
                                  bloc=0,
                                  fix_params_arg=None, diff_evolution_past_samples=None, diff_evolution_weight=None,
                                  diff_evolution_jump_cadence=np.inf,
                                  scalar_jitter_bounds=None,
                                  position=0
                                  ) -> dict:
    """Run a single Adaptive Metropolis–Hastings chain (optionally with DE-MC jumps).

    Executes ``(n_burnin + n_samples) * thinning`` Metropolis iterations,
    thinning by ``thinning`` when recording draws.  The global scalar
    multiplier on the proposal covariance is adapted after every step using a
    stochastic-approximation rule aimed at the ``target_acceptance_rate``.
    When Differential Evolution (DE) jumps are enabled, every
    ``diff_evolution_jump_cadence`` iterations the proposal is replaced by a
    scaled difference of two randomly drawn past samples.

    The chain is designed to be called either directly (serial mode) or as a
    Ray remote task via ``run_adaptive_metropolis_chain_remote``.

    Args:
        seed: Integer random seed for ``numpy.random.RandomState``.
        log_posterior: Callable ``log_posterior(x)`` → scalar log-posterior on
            the unbounded/transformed parameter space.
        x0: Starting parameter vector (1-D array-like).
        step_cov: Initial proposal covariance matrix (NumPy array).  A scalar
            ``0`` is treated as a 1×1 case; the matrix is repaired to positive
            semi-definite by ``make_pos_semidef`` if needed.
        log_posterior_jacobian_adjustment: Optional callable that returns the
            log-determinant of the Jacobian of the bounded→unbounded transform;
            ``None`` disables the adjustment.
        n_burnin: Number of burn-in iterations to run (draws are stored but
            excluded from the effective sample).
        n_samples: Number of post-burn-in sample iterations.
        itr0: Starting iteration counter offset (used when restarting a chain
            to keep the adaptation schedule consistent).
        target_acceptance_rate: Desired Metropolis acceptance rate; the scalar
            is nudged up/down relative to this target.
        chain_no: Display identifier for the progress bar.  Defaults to ``seed``.
        draw_size: Number of proposal draws to pre-generate in each batch from
            the multivariate-normal proposal distribution.
        scaler0: Initial scalar multiplier on the proposal covariance.
            Defaults to 1.0 when ``None``.
        min_scaler: Hard lower bound on the adaptive scalar.
        max_scaler: Hard upper bound on the adaptive scalar.
        scaler_adjust_rate: Learning rate for the scaler adaptation rule.
        scaler_adjust_denom_power: Exponent applied to the iteration count in
            the diminishing-adaptation denominator.
        thinning: Store every ``thinning``-th draw; the remaining draws are
            evaluated but discarded.
        pbar_update_cadence: Seconds between progress-bar updates (when ``debug=True``).
        do_adaptive: Whether to adapt the scalar at each iteration.
        debug: Whether to show a ``tqdm`` progress bar via ``tqdm_ray``.
        proposal_df: Degrees of freedom for the proposal distribution.
            Currently only ``np.inf`` (Gaussian) is implemented.
        bloc: Integer bloc/segment identifier appended to the result dict for
            bookkeeping when chains are split into sub-chains.
        fix_params_arg: Dict ``{'index': [...], 'values': [...]}`` of parameter
            indices and their fixed values; those dimensions are overwritten on
            every draw if provided.
        diff_evolution_past_samples: 2-D array of past draws used as the DE-MC
            gene pool.  ``None`` disables DE-MC.
        diff_evolution_weight: Mixture weight in ``[0, 1]`` controlling the
            relative contribution of DE jumps vs. normal proposal moves.
        diff_evolution_jump_cadence: How often (in iterations) to force a
            pure DE jump (temporarily setting ``coef_am`` to near-zero).
        scalar_jitter_bounds: Optional ``(lo, hi)`` tuple for per-step random
            scaling of the proposal; provides extra exploration.
        position: ``tqdm`` bar position (for multi-chain parallel display).

    Returns:
        Dict with keys: ``'samples'`` (array of shape
        ``(n_burnin + n_samples, num_params)``), ``'accepteds'``,
        ``'scalers'``, ``'acceptance_probs'``, ``'log_posterior'``,
        timing keys (``'lp_time'``, ``'draw_time'``, etc.), and metadata
        (``'seed'``, ``'thinning'``, ``'bloc'``, ``'pid'``, ``'itr'``,
        ``'num_jump_tries'``, ``'num_jump_successes'``).
    """
    __time_chn_master = time.time()

    do_diff_evolution = diff_evolution_past_samples is not None
    if not do_diff_evolution:
        diff_evolution_jump_cadence = None
    if diff_evolution_jump_cadence is None:
        diff_evolution_jump_cadence = np.inf
    num_jump_tries = 0
    num_jump_successes = 0

    if scalar_jitter_bounds is not None:
        scalar_jitter_bounds = tuple(scalar_jitter_bounds)
        assert len(scalar_jitter_bounds) == 2
        assert 0 < scalar_jitter_bounds[0] < scalar_jitter_bounds[1]

    if debug:
        from ray.experimental import tqdm_ray
        pbar = tqdm_ray.tqdm(position=position, total=thinning * (n_samples + n_burnin))

    rand = np.random.RandomState(seed)

    # if resample_k is None or resample_k == 0:
    #     resample_k = np.inf

    do_lp_adjustment = log_posterior_jacobian_adjustment is not None
    x0 = np.array(x0, dtype=float).flatten().copy()
    num_params = len(x0)

    step_cov0 = step_cov.copy()
    step_cov = make_pos_semidef(step_cov)

    if max_scaler is None:
        max_scaler = DEFAULT_AM_MAX_SCALER
    if min_scaler is None:
        min_scaler = DEFAULT_AM_MIN_SCALER

    if scaler0 is None:
        scaler = 1.0
    else:
        scaler = scaler0

    gelman_optimal_scaler_value = 2.38 / np.sqrt(num_params)

    if do_diff_evolution:
        assert 0 <= diff_evolution_weight <= 1
        scaler_draw_denom = (
                np.sqrt(diff_evolution_weight ** 2 + (1 - diff_evolution_weight) ** 2)
                * gelman_optimal_scaler_value
        )
        de_idx = rand.randint(0, len(diff_evolution_past_samples), ((n_samples + n_burnin) * thinning, 2))
    else:
        diff_evolution_weight = 0.0
        scaler_draw_denom = gelman_optimal_scaler_value

    if chain_no is None:
        chain_no = seed

    draws_nc = np.zeros((n_burnin + n_samples, num_params))
    log_posterior_arr = np.zeros(n_burnin + n_samples)

    x0 = np.array(x0, copy=True, dtype=float)
    if fix_params_arg is not None:
        x0[fix_params_arg['index']] = fix_params_arg['values']

    lp0 = log_posterior(x0)
    lp_adj0 = log_posterior_jacobian_adjustment(x0) if do_lp_adjustment else 0.0

    if not np.all(np.isfinite(lp0)) or np.any(np.isnan(lp0)):
        raise Exception('Starting point must generating finite log-posterior!')

    last_update_time = 0
    # last_update_cov = itr0 + 1
    # num_cov_updates = 0

    lp_time = 0
    cov_time = 0
    draw_time = 0
    accept_decision_time = 0

    _t_draw = time.time()
    if proposal_df == np.inf:
        rand_dist = multivariate_normal(mean=[0.0] * num_params, cov=step_cov.copy(), allow_singular=True)
    else:
        raise NotImplementedError
        # rand_dist = multivariate_t(loc=[0.0] * num_params, shape=step_cov.copy(), df=proposal_df)

    draw_generator = (d for d in rand_dist.rvs(random_state=rand, size=draw_size))
    # draw_generator = (d for d in rand.multivariate_normal(mean=[0.0] * num_params, cov=step_cov.copy(), size=draw_size))
    draw_time += time.time() - _t_draw

    accepted_rate = 0.0

    # S2 = 0.0
    # S1 = 0.0

    err_settings = np.seterr()
    np.seterr(all='ignore')

    setup_time = time.time() - __time_chn_master

    accept = 0
    accepteds = []
    log_acceptance_probs = []
    scalers = []

    itr = 0

    log_random_uniform = np.log(rand.rand(thinning * (n_samples + n_burnin)))

    if scalar_jitter_bounds is not None:
        jump_scaler_rand = rand.rand(thinning * (n_samples + n_burnin))
        jump_scaler_rand = scalar_jitter_bounds[0] + jump_scaler_rand * (scalar_jitter_bounds[1] - scalar_jitter_bounds[0])

    for itr in range(itr0, itr0 + (n_burnin + n_samples) * thinning):

        # scales on the AM draw from covariance and from differential evolution
        # past draws
        coef_am = scaler * (1.0 - diff_evolution_weight) / scaler_draw_denom
        coef_de = scaler * (diff_evolution_weight / np.sqrt(2)) / scaler_draw_denom

        if do_diff_evolution and itr % diff_evolution_jump_cadence == 0:
            coef_am, coef_de = .001, 1.0
            num_jump_tries += 1

        if scalar_jitter_bounds is not None:
            coef_am *= jump_scaler_rand[itr - itr0]
            coef_de *= jump_scaler_rand[itr - itr0]

        # if itr == itr0:
        #     print(bloc, type(diff_evolution_past_samples), diff_evolution_weight, gelman_optimal_scaler_value,
        #           scaler_draw_denom,  (1.0 - diff_evolution_weight) / scaler_draw_denom,
        #          (diff_evolution_weight / np.sqrt(2)) / scaler_draw_denom)

        scalers.append(scaler)

        _t_draw = time.time()
        try:
            am_draw_next = next(draw_generator)
        except:
            draw_generator = (d for d in rand_dist.rvs(random_state=rand, size=draw_size))
            # draw_generator = (d for d in rand.multivariate_normal(mean=[0.0] * num_params, cov=step_cov.copy(), size=draw_size))
            am_draw_next = next(draw_generator)

        draw_time += time.time() - _t_draw

        # TODO remove `resample_k` stuff?
        # if (itr - itr0) // thinning > 0 and itr0 > 0 and itr % resample_k == 0:
        #     x1 = draws_nc[rand.randint(0, ((itr - itr0) // thinning))] + coef_am * am_draw_next
        # else:

        x1 = x0 + coef_am * am_draw_next

        if do_diff_evolution:
            de_draw_next = (diff_evolution_past_samples[de_idx[itr - itr0, 0]]
                                - diff_evolution_past_samples[de_idx[itr - itr0, 1]])
            x1 += coef_de * de_draw_next

        if fix_params_arg is not None:
            x1[fix_params_arg['index']] = fix_params_arg['values']

        _t_lp = time.time()
        lp1 = log_posterior(x1)
        lp_adj1 = log_posterior_jacobian_adjustment(x1) if do_lp_adjustment else 0.0
        lp_time += time.time() - _t_lp

        _t_accept = time.time()
        x0, lp0, lp_adj0, accepted_bool, log_acceptance_prob = \
            _acceptance_choice(x0, x1, lp0, lp1, lp_adj0, lp_adj1,
                               uniform_draw=log_random_uniform[itr - itr0])

        # if do_diff_evolution and itr % diff_evolution_jump_cadence == 0:
        #     print(">>>> ", itr, 'accept = ' + "%5s" % accepted_bool,
        #           "%10.4f" % np.exp(
        #               lp1 + lp_adj1 - lp0 - lp_adj0
        #           ),
        #           '%10.4f' % log_acceptance_prob,
        #           '%10.4f' % np.exp(log_acceptance_prob),
        #     )

        num_jump_successes += accepted_bool and do_diff_evolution and itr % diff_evolution_jump_cadence == 0
        accept += accepted_bool
        accepteds.append(accepted_bool)
        log_acceptance_probs.append(log_acceptance_prob)

        if itr % thinning == 0:
            draws_nc[(itr - itr0) // thinning] = x0
            log_posterior_arr[(itr - itr0) // thinning] = lp0

        accept_decision_time += time.time() - _t_accept

        accepted_rate = ((itr - itr0) * accepted_rate + accepteds[-1]) / (itr + 1 - itr0)
        if do_adaptive and not (do_diff_evolution and itr % diff_evolution_jump_cadence == 0):
            cur_adust_coef = scaler_adjust_rate / max(itr - 100, 1) ** scaler_adjust_denom_power
            scaler = min(
                max(
                    scaler * (1.0 + cur_adust_coef * (accepted_bool - target_acceptance_rate)),
                    min_scaler
                ),
                max_scaler
            )

        if debug and time.time() - last_update_time > pbar_update_cadence:
            last_update_time = time.time()
            pbar.update((itr + 1) - itr0 - pbar._x)
            if debug:
                pbar_str = f'chain {chain_no:2d}, lp = {lp0:8.4e}, accept = {accepted_rate:5.2f}, ' \
                           f'accept_L200 = {np.mean(accepteds[-200:]):5.2f}, scaler = {scaler:6.2e}'
                pbar.set_description(pbar_str)

    if debug:
        pbar.update((itr + 1) - itr0 - pbar._x)
        pbar_str = f'chain {chain_no:2d}, lp = {lp0:8.4e}, accept = {accepted_rate:5.2f}, ' \
                   f'accept_L200 = {np.mean(accepteds[-200:]):5.2f}, scaler = {scaler:6.2e}'
        pbar.set_description(pbar_str)

    np.seterr(**err_settings)

    log_acceptance_probs = np.array(log_acceptance_probs, dtype=float)

    #print(f"JUMP {num_jump_successes} / {num_jump_tries}, ({n_burnin*thinning}, {n_samples*thinning})")

    return {
        'samples': draws_nc,
        'n_burnin': n_burnin,
        'n_samples': n_samples,
        'acceptance_probs': np.exp(log_acceptance_probs),
        'seed': seed,
        'accepteds': np.array(accepteds),
        'scalers': np.array(scalers),
        'step_cov': step_cov,
        'step_cov0': step_cov0,
        # 'cov_samples': np.cov(draws_nc, rowvar=False, ddof=0),
        # 'mean_samples': np.mean(draws_nc, axis=0),
        # 'S1': S1,
        # 'S2': S2,
        'lp_time': lp_time,
        'cov_time': cov_time,
        'draw_time': draw_time,
        'accept_decision_time': accept_decision_time,
        'setup_time': setup_time,
        'fit_elapsed': time.time() - __time_chn_master,
        'thinning': thinning,
        'log_posterior': log_posterior_arr,
        'itr': itr,
        'bloc': [bloc] * (n_samples + n_burnin),
        'pid': [os.getpid()] * (n_samples + n_burnin),
        'num_jump_tries': num_jump_tries,
        'num_jump_successes': num_jump_successes,
        'scalar_jitter_bounds': scalar_jitter_bounds,
    }


# run_adaptive_metropolis_chain_remote = ray.remote(run_adaptive_metropolis_chain)


def get_subchain_n_draws(n_burnin, n_samples, max_subchain_draws_burnin=5_000, max_subchain_draws_sample=10_000):
    """Partition total burn-in and sample draws into bounded sub-chains.

    To keep Ray remote tasks short and allow the proposal covariance to be
    updated between sub-chains, this function splits ``n_burnin`` and
    ``n_samples`` into segments of at most ``max_subchain_draws_burnin`` and
    ``max_subchain_draws_sample`` respectively.  Each sub-chain is described
    by the number of burn-in and sample draws it should run.

    Args:
        n_burnin: Total number of burn-in draws requested.
        n_samples: Total number of post-burn-in sample draws requested.
        max_subchain_draws_burnin: Maximum draws allowed in a single burn-in
            sub-chain segment.
        max_subchain_draws_sample: Maximum draws allowed in a single sample
            sub-chain segment.

    Returns:
        5-tuple ``(n_subchains, subchain_n_burnin_iters, subchain_n_samples_iters,
        is_burnin_subchain, subchain_cnt)`` where ``n_subchains`` is the total
        number of sub-chains, ``subchain_n_burnin_iters`` and
        ``subchain_n_samples_iters`` are lists of per-sub-chain draw counts
        (zeros for the opposite phase), ``is_burnin_subchain`` is a boolean
        list flagging burn-in sub-chains, and ``subchain_cnt`` is initialized
        to 0 as a running counter.
    """
    n_sub_chains_burnin = max(n_burnin // max_subchain_draws_burnin, n_burnin > 0)
    subchain_n_burnin_iters = [max_subchain_draws_burnin * (
            n_burnin >= max_subchain_draws_burnin)] * n_sub_chains_burnin
    if n_burnin:
        if n_burnin % max_subchain_draws_burnin:
            if n_burnin > max_subchain_draws_burnin:
                n_sub_chains_burnin += 1
                subchain_n_burnin_iters.append(0)
            subchain_n_burnin_iters[-1] += n_burnin % max_subchain_draws_burnin

    n_sub_chains_samples = max(n_samples // max_subchain_draws_sample, n_samples > 0)
    subchain_n_samples_iters = [max_subchain_draws_sample * (
                n_samples >= max_subchain_draws_sample)] * n_sub_chains_samples
    if n_samples:
        subchain_n_samples_iters[-1] += n_samples % max_subchain_draws_sample

    n_subchains = n_sub_chains_burnin + n_sub_chains_samples
    subchain_n_burnin_iters = subchain_n_burnin_iters + [0] * n_sub_chains_samples
    subchain_n_samples_iters = [0] * n_sub_chains_burnin + subchain_n_samples_iters
    is_burnin_subchain = [True] * n_sub_chains_burnin + [False] * n_sub_chains_samples

    subchain_cnt = 0

    return n_subchains, subchain_n_burnin_iters, subchain_n_samples_iters, is_burnin_subchain, subchain_cnt


def amha(log_posterior, start_params, start_params_is_original_scale=True,
         step_cov=None, log_posterior_jacobian_adjustment=None, param_names=None, specification_name=None,
         n_chains=DEFAULT_AM_N_CHAINS,
         n_burnin=DEFAULT_AM_N_BURNIN, n_samples=DEFAULT_AM_N_SAMPLES, max_processes=DEFAULT_AM_MAX_PROCESSES,
         seed=None, target_acceptance_rate=DEFAULT_AM_TARGET_ACCEPTANCE_RATE, draw_size=DEFAULT_AM_DRAW_SIZE,
         scaler0=None, min_scaler=DEFAULT_AM_MIN_SCALER, max_scaler=DEFAULT_AM_MAX_SCALER,
         scaler_adjust_rate=DEFAULT_AM_SCALER_ADJUST_RATE,
         thinning=DEFAULT_AM_THINNING, scaler_adjust_denom_power=DEFAULT_AM_SCALER_ADJUST_DENOM_POWER,
         pbar_update_cadence=DEFAULT_AM_PBAR_UPDATE_CADENCE,
         bounds=None, do_adaptive=DEFAULT_AM_DO_ADAPTIVE,
         max_subchain_draws_burnin=DEFAULT_AM_MAX_SUBCHAIN_DRAWS_BURNIN,
         max_subchain_draws_sample=DEFAULT_AM_MAX_SUBCHAIN_DRAWS_SAMPlE,
         do_parallel=DEFAULT_AM_DO_PARALLEL, user_prompt_for_more_iters=DEFAULT_AM_USER_PROMPT_FOR_MORE_ITERS,
         debug=False, proposal_df=DEFAULT_AM_PROPOSAL_DF,
         # resample_k=DEFAULT_AM_RESAMPLE_K,
         step_cov_initial_samples=DEFAULT_AM_STEP_COV_INITIAL_SAMPLES,
         # ridge_epsilon=DEFAULT_AM_RIDGE_EPSILON,
         # min_unique_draws_for_cov=DEFAULT_AM_MIN_UNIQUE_DRAWS_FOR_COV, cov_update_frequency=DEFAULT_AM_COV_UPDATE_FREQUENCY,
         model=None, fix_params=None, transformations=None,
         starting_iter=0,
         # do_metropolis_de=False, de_scaler_past_samples=1.0, de_past_samples=None, de_frac_burnin=.33,
         normalize_step_cov=DEFAULT_AM_NORMALIZE_STEP_COV,

         do_diff_evolution_mc=DEFAULT_AM_DO_DIFF_EVOLUTION_MC,
         diff_evolution_past_samples=None,
         diff_evolution_frac_burnin=DEFAULT_AM_DIFF_EVOLUTION_FRAC_BURNIN,
         diff_evolution_max_draws=DEFAULT_AM_DIFF_EVOLUTION_MAX_DRAWS,
         diff_evolution_weight=DEFAULT_AM_DIFF_EVOLUTION_WEIGHT,
         diff_evolution_jump_cadence=DEFAULT_AM_DIFF_EVOLUTION_JUMP_CADENCE,

         stop_adaptation_after_burnin=DEFAULT_AM_STOP_ADAPTATION_AFTER_BURNIN,

         scalar_jitter_bounds=DEFAULT_AM_SCALAR_JITTER_BOUNDS,

         callback_function=None

         ) -> MCMCResults:

    """Run Adaptive Metropolis–Hastings MCMC with optional Differential Evolution.

    This is the primary user-facing entry point for the AMH sampler.  It:

    1. Formats and validates the starting parameter vectors (one per chain),
       applying any bounded-to-unbounded transformations via
       ``format_starting_point_for_mcmc``.
    2. Checks that all starting points yield finite log-posteriors.
    3. Initializes the proposal covariance (either user-supplied or a scaled
       identity) and optionally normalizes its determinant.
    4. Applies any hard parameter bounds as log-posterior barriers.
    5. Splits the full draw count into sub-chains via ``get_subchain_n_draws``
       and iteratively dispatches them—either serially or in parallel via Ray—
       updating the pooled proposal covariance and DE past-sample pool between
       sub-chains.
    6. Prompts the user for more iterations when ``user_prompt_for_more_iters``
       is True and convergence criteria are not met.
    7. Returns a fully initialized ``MCMCResults`` object.

    Args:
        log_posterior: Callable ``log_posterior(x)`` → scalar.  Evaluated on
            the (possibly unbounded-transformed) parameter space.
        start_params: Starting values.  Either a 1-D array (shared across all
            chains, with per-chain jitter applied automatically), a 2-D array
            of shape ``(n_chains, num_params)``, or a list of 1-D arrays.
        start_params_is_original_scale: When ``True`` (default), starting
            values are in the original (possibly bounded) parameter space and
            are mapped to the unbounded sampling space by the transformations.
        step_cov: Initial proposal covariance matrix.  Defaults to a small
            scaled identity if ``None``.
        log_posterior_jacobian_adjustment: Optional callable returning the
            log-determinant of the Jacobian for the bounded→unbounded transform.
            Required when ``transformations`` are supplied.
        param_names: List of parameter name strings.  Auto-generated when ``None``.
        specification_name: Optional display label for the model run.
        n_chains: Number of parallel MCMC chains.
        n_burnin: Number of burn-in draws per chain (discarded from posterior
            summaries but stored for diagnostics).
        n_samples: Number of post-burn-in draws per chain.
        max_processes: Maximum number of Ray worker processes to use when
            ``do_parallel=True``.
        seed: Base integer random seed; each chain receives ``seed + chain_idx``.
        target_acceptance_rate: Target Metropolis acceptance rate for the
            adaptive scalar.  Typically around 0.234 (Gelman optimal) or 0.44
            for 1-D chains.
        draw_size: Batch size for pre-drawing proposals from the multivariate
            normal.
        scaler0: Initial scalar multiplier on the proposal covariance.
        min_scaler: Minimum allowed adaptive scalar.
        max_scaler: Maximum allowed adaptive scalar.
        scaler_adjust_rate: Learning rate for the adaptive scalar.
        thinning: Store every ``thinning``-th iteration.
        scaler_adjust_denom_power: Exponent for the diminishing-adaptation rule.
        pbar_update_cadence: Seconds between ``tqdm`` bar updates.
        bounds: Optional dict ``{param_name: [lo, hi]}`` or 2-D array of shape
            ``(2, num_params)`` imposing hard bounds as -∞ log-posterior barriers.
        do_adaptive: Whether to adapt the scalar each iteration.
        max_subchain_draws_burnin: Maximum sub-chain length during burn-in.
        max_subchain_draws_sample: Maximum sub-chain length during sampling.
        do_parallel: Whether to dispatch chains in parallel via Ray.
        user_prompt_for_more_iters: Whether to ask the user for more iterations
            if convergence is not achieved.
        debug: Whether to print progress information and timing.
        proposal_df: Proposal distribution degrees of freedom (``np.inf`` for
            Gaussian; other values not yet implemented).
        step_cov_initial_samples: Number of initial draws used to estimate the
            starting covariance when ``step_cov`` is ``None``; auto-computed
            from ``n_burnin`` when ``None``.
        model: Optional reference to the originating ``BayesianModel``.
        fix_params: Dict ``{param_name_or_index: value}`` of parameters held
            fixed during sampling.
        transformations: Dict ``{param_index: TransformedParameter}`` mapping
            bounded parameters to unbounded equivalents.
        starting_iter: Initial iteration counter offset (non-zero when
            continuing sampling from a previous run).
        normalize_step_cov: Whether to normalize the proposal covariance
            determinant to 1 before sampling.
        do_diff_evolution_mc: Whether to enable Differential Evolution MC jumps.
        diff_evolution_past_samples: Pre-supplied DE past-sample pool (2-D array).
            If ``None``, the pool is built from the burn-in draws.
        diff_evolution_frac_burnin: Fraction of burn-in draws to use for the
            initial DE sample pool.
        diff_evolution_max_draws: Maximum number of draws kept in the DE pool.
        diff_evolution_weight: Mixture weight for DE vs. normal proposals
            (must be in ``(0, 1)``).
        diff_evolution_jump_cadence: Iteration interval for forced DE jumps.
        stop_adaptation_after_burnin: Whether to freeze the scalar and covariance
            after burn-in finishes.
        scalar_jitter_bounds: Optional ``(lo, hi)`` for per-step random jitter
            of the proposal scalar.
        callback_function: Optional callable called after each sub-chain with
            the current ``MCMCResults`` object; useful for checkpointing.

    Returns:
        ``MCMCResults`` object containing all draws, diagnostics, and metadata.

    Examples
    --------
    Sample from a 2-parameter Beta posterior via :class:`DataModel` (the
    convenient wrapper that builds the model code for you):

    >>> import numpy as np
    >>> from scipy.stats import beta
    >>> from kanly.api import DataModel
    >>> rng = np.random.default_rng(0)
    >>> data = {'x': beta.rvs(a=5, b=2, size=1_500, random_state=0)}
    >>> dmo = DataModel.build_data_model(                       # doctest: +SKIP
    ...     'self.x = `x`',
    ...     'return nopython_logpdf_beta(x, a=$a$, b=$b$).sum()',
    ...     data, nopython=True)
    >>> model = dmo.to_bayesian_model(                           # doctest: +SKIP
    ...     bounds={'a': [0, np.inf], 'b': [0, np.inf]})
    >>> fit = model.amha(                                        # doctest: +SKIP
    ...     [1.0, 1.0],
    ...     n_samples=10_000, n_burnin=3_000,
    ...     n_chains=4, do_diff_evolution_mc=True,
    ...     max_subchain_draws_burnin=2_000,
    ...     max_subchain_draws_sample=5_000,
    ...     do_parallel=True)
    >>> print(fit)                                               # doctest: +SKIP
    ════════════════════════════════════════════════════════════════════════════
    MCMC Results
    ════════════════════════════════════════════════════════════════════════════
    ...
    a  4.961   0.181   ...   1.0002
    b  2.051   0.070   ...   1.0002

    Calling :meth:`BayesianModel.sample` defaults to this AMH sampler and
    can additionally schedule a MALA warm-up phase first via
    ``do_mala_cd_warmup=True``.

    See Also
    --------
    :func:`mala` : Coordinate Metropolis-Adjusted Langevin sampler.
    """
    _time_master = time.time()

    if do_diff_evolution_mc:
        assert isinstance(diff_evolution_weight, float) and 0 < diff_evolution_weight < 1

    if debug:
        print("----------------------------------")
        print("Beginning Adaptive Metropolis MCMC")
        print("----------------------------------")

    if step_cov_initial_samples is None:
        step_cov_initial_samples = max(n_burnin // 100, 50)

    x0s_formatted, transformations, fix_params, fix_params_transformed, param_names, num_params, transformation_function = \
        format_starting_point_for_mcmc(start_params, start_params_is_original_scale, n_chains, transformations, fix_params, param_names,
                                       debug=debug)

    check_starting_point(x0s_formatted, log_posterior, log_posterior_jacobian_adjustment, debug)

    has_fixed_params = fix_params_transformed is not None and len(fix_params_transformed)
    if has_fixed_params:
        fixed_param_idx, fixed_param_vals = list(fix_params_transformed.keys()), list(fix_params_transformed.values())
        not_fixed_param_index = [a for a in range(num_params) if a not in fixed_param_idx]
        fix_params_arg = {'index': fixed_param_idx, 'values': fixed_param_vals}
    else:
        fixed_param_idx = []
        not_fixed_param_index = range(num_params)
        fix_params_arg = None
        fix_params = None

    if step_cov is None:
        step_cov = np.eye(len(x0s_formatted[0])) * min(len(x0s_formatted[0]) ** -2, .01)

    scaler0_orig = scaler0

    step_cov0_orig = make_pos_semidef(step_cov.copy())

    log_posterior_bounded = log_posterior
    if bounds is not None:
        if isinstance(bounds, dict):
            bounds_arr = np.ones((2, num_params))
            for i, nm in enumerate(param_names):
                bounds_arr[:, i] = bounds.get(nm, [-np.inf, np.inf])
            bounds = bounds_arr
        bounds = np.array(bounds.copy())
        assert np.prod(np.shape(bounds)) == 2 * num_params
        assert bounds.shape[0] == 2

        def log_posterior_bounded(x):
            """log_posterior_bounded function.

            Args:
                x: TODO.
            """
            if np.any(x < bounds[0]) or np.any(x > bounds[1]):
                return -np.inf
            return log_posterior(x)

    if do_parallel:

        import ray

        t_put = time.time()
        n_cpu = min(max_processes, n_chains)
        run_adaptive_metropolis_chain_remote = ray.remote(run_adaptive_metropolis_chain)
                
        if ray.is_initialized():
            try:
                if ray.available_resources()['CPU'] < n_cpu:
                    if debug:
                        print(f"Shutting down `ray` and reinitializing, available CPUs ({ray.available_resources()['CPU']})"
                              f" less than specified ({n_cpu})")
                    ray.shutdown()
                    ray.init(num_cpus=n_cpu, log_to_driver=debug)
            except:
                ray.shutdown()
                ray.init(num_cpus=n_cpu, log_to_driver=debug)
        else:
            ray.init(num_cpus=n_cpu, log_to_driver=debug)

        if debug:
            print("Putting log posterior in `ray` storage...", end="")
        log_posterior_bounded_ray_id = ray.put(log_posterior_bounded)
        log_posterior_jacobian_adjustment_id = ray.put(log_posterior_jacobian_adjustment)
        diff_evolution_past_samples_id = ray.put(diff_evolution_past_samples)
        if debug:
            print(f"Done! ({time.time() - t_put:.2f}s)")

    else:
        log_posterior_bounded_ray_id = log_posterior_bounded
        log_posterior_jacobian_adjustment_id = log_posterior_jacobian_adjustment
        diff_evolution_past_samples_id = diff_evolution_past_samples

    if debug:
        if specification_name is not None:
            print('\nSpecification Name: ', specification_name, '\n')

        settings_dict = dict(
            debug=debug,
            pbar_update_cadence=pbar_update_cadence,
            user_prompt_for_more_iters=user_prompt_for_more_iters,

            seed=seed,
            do_adaptive=do_adaptive,
            n_chains=n_chains,
            n_burnin=n_burnin,
            n_samples=n_samples,
            target_acceptance_rate=target_acceptance_rate,
            thinning=thinning,
            max_subchain_draws_burnin=max_subchain_draws_burnin,
            max_subchain_draws_sample=max_subchain_draws_sample,
            draw_size=draw_size,

            scaler0=scaler0, min_scaler=min_scaler, max_scaler=max_scaler,
            scaler_adjust_rate=scaler_adjust_rate,
            scaler_adjust_denom_power=scaler_adjust_denom_power,

            do_parallel=do_parallel,
            max_processes=max_processes,

            x0_is_original_scale=start_params_is_original_scale,

            normalize_step_cov=normalize_step_cov,
            stop_adaptation_after_burnin=stop_adaptation_after_burnin,

            do_diff_evolution_mc=do_diff_evolution_mc,
            diff_evolution_max_draws=diff_evolution_max_draws,
            diff_evolution_past_samples=None if diff_evolution_past_samples is None else len(
                diff_evolution_past_samples),
            diff_evolution_weight=diff_evolution_weight,
            diff_evolution_frac_burnin=diff_evolution_frac_burnin,

            scalar_jitter_bounds=str(scalar_jitter_bounds),

            diff_evolution_jump_cadence=diff_evolution_jump_cadence,
        )
        print_options(settings_dict, title='ADAPTIVE METROPOLIS SAMPLING SETTINGS')

        # info = np.array([
        #     ('Settings', ''),
        #     ('   n_params:', num_params),
        #     ('   n_chains:', n_chains),
        #     ('   max_subchain_draws_burnin:', max_subchain_draws_burnin),
        #     ('   max_subchain_draws_sample:', max_subchain_draws_sample),
        #     ('   do_parallel:', do_parallel),
        #     ('   thinning:', thinning),
        #     ('   n_burnin:', n_burnin),
        #     ('   n_samples:', n_samples),
        #     ('   seed:', seed),
        #     ('   draw_size:', draw_size),
        #     ('   target_acceptance_rate:', target_acceptance_rate),
        #     ('   adaptive:', do_adaptive),
        #     ('   scaler0:', scaler0),
        #     ('   min_scaler:', min_scaler),
        #     ('   max_scaler:', max_scaler),
        #     ('   scaler_adjust_rate:', scaler_adjust_rate),
        #     ('   scaler_adjust_denom_power:', scaler_adjust_denom_power),
        #     ('   proposal_df:', proposal_df),
        #     # ('   resample_k:', resample_k),
        #     ('   step_cov_initial_samples:', step_cov_initial_samples),
        #     ('   fixed_params:', fix_params is not None),
        #     ('   normalize step cov: ', normalize_step_cov),
        #
        #     ('   do_diff_evolution_mc:', do_diff_evolution_mc),
        #     ('   diff_evolution_max_draws:', diff_evolution_max_draws),
        #     ('   diff_evolution_frac_burnin:', diff_evolution_frac_burnin),
        #     ('   diff_evolution_weight:', diff_evolution_weight),
        #     ('   diff_evolution_jump_cadence:', diff_evolution_jump_cadence),
        #     ('   stop_adaptation_after_burnin:', stop_adaptation_after_burnin),
        #
        #     ('   scalar_jitter_bounds:', str(scalar_jitter_bounds)),
        #
        #     # ('   cov_update_frequency:', cov_update_frequency),
        #     # ('   min_unique_draws_for_cov:', min_unique_draws_for_cov),
        # ])
        # print()
        # print(pd.Series(info[:, 1], index=info[:, 0]).to_string())
        # print()

    if seed is None:
        seed = 0
    rand = np.random.RandomState(seed)

    chain_nos = range(n_chains)

    if debug:
        print('\nBeginnning MCMC chains now...')

    chain_results_master = [None] * n_chains

    n_subchains, subchain_n_burnin_iters, subchain_n_samples_iters, is_burnin_subchain, subchain_cnt \
        = get_subchain_n_draws(n_burnin, n_samples, max_subchain_draws_burnin=max_subchain_draws_burnin,
                               max_subchain_draws_sample=max_subchain_draws_sample)

    if do_diff_evolution_mc:
        if stop_adaptation_after_burnin and diff_evolution_past_samples is None:
            if np.count_nonzero(subchain_n_burnin_iters) < 2:
                raise Exception("`do_diff_evolution_mc` ineffective if less than two burnin sub-chains. "
                                "\nConsider ratio of `n_burnin` to `max_subchain_draws_burnin`!")

    mean_master = None
    n_burnin_master = 0
    n_samples_master = 0

    if starting_iter is None:
        itr0 = 0
    else:
        itr0 = starting_iter

    bloc = 0

    _time_mcmc_draws = time.time()
    _wait_time = 0

    while True:

        if normalize_step_cov:
            # todo normalize by something else?
            step_cov_iter, _, _ = normalize_matrix_det(step_cov, not_fixed_param_index, 1.0)
        else:
            step_cov_iter = step_cov

        bloc += 1
        chain_args = [
            (
                seed, log_posterior_bounded_ray_id, start_params.copy(),
                step_cov_iter.copy(), log_posterior_jacobian_adjustment_id,
                subchain_n_burnin_iters[subchain_cnt], subchain_n_samples_iters[subchain_cnt], itr0,
                target_acceptance_rate, ch_num, draw_size, scaler0, min_scaler, max_scaler, scaler_adjust_rate,
                scaler_adjust_denom_power, thinning, pbar_update_cadence,
                do_adaptive and (is_burnin_subchain or not stop_adaptation_after_burnin),
                debug,
                # cov_update_frequency, min_unique_draws_for_cov, ridge_epsilon,
                proposal_df, bloc, fix_params_arg,
                diff_evolution_past_samples_id, diff_evolution_weight if diff_evolution_past_samples is not None else 0,
                diff_evolution_jump_cadence, scalar_jitter_bounds,
                position,
            )
            for position, (seed, start_params, ch_num, step_cov) in enumerate(
                zip(rand.randint(0, 100_000, n_chains), x0s_formatted, chain_nos, [step_cov.copy()] * n_chains)
            )
        ]

        if debug:
            print()
            print(dict(bloc=f'{subchain_cnt + 1}/{n_subchains}', n_burnin=subchain_n_burnin_iters[subchain_cnt],
                       n_sample=subchain_n_samples_iters[subchain_cnt], is_burnin=is_burnin_subchain[subchain_cnt]))

        if do_parallel:
            time.sleep(.01)

            try:
                
                chain_results = ray.get([run_adaptive_metropolis_chain_remote.remote(*arg) for arg in chain_args])

                time.sleep(.01)
                if debug:
                    print('\nJoining the data from the parallel jobs...')

            except KeyboardInterrupt:
                warnings.warn("\n\nKeyboard Interrupt! Stopping draws.\n\n")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                ray.shutdown()
                if n_samples_master == 0:
                    n_samples_master = n_burnin_master // 3
                    n_burnin_master = n_burnin_master - n_samples_master
                break

            except Exception as e:
                ray.shutdown()
                print("\n!!!" * 4)
                print(e.__traceback__)
                raise e

        else:
            try:
                chain_results = [run_adaptive_metropolis_chain(*_arg) for _arg in chain_args]
            except KeyboardInterrupt:
                if n_samples_master == 0:
                    n_samples_master = n_burnin_master // 3
                    n_burnin_master = n_burnin_master - n_samples_master
                warnings.warn("\n\nKeyboard Interrupt! Stopping draws.\n\n")
                break
            except Exception as e:
                raise e

        for i, (chn_master, chn_new) in enumerate(zip(chain_results_master, chain_results)):

            if i == 0:  # we only count burnin *per* chain
                n_burnin_master += chn_new['n_burnin']
                n_samples_master += chn_new['n_samples']

            if chn_master is None:
                chain_results_master[i] = chain_results[i]

                # Ray returns immutable arrays, so on first draw (where we don't vstack),
                # copy it to mutable
                chain_results[i]['samples'] = np.array(chain_results[i]['samples'])

                # add mean and cov
                chain_results_master[i]['mean_draws'] = np.mean(chain_results[i]['samples'], axis=0)
                chain_results_master[i]['cov_draws'] = np.cov(chain_results[i]['samples'], ddof=0, rowvar=False)

            else:

                new_mean, new_cov = (
                    np.mean(chn_new['samples'], axis=0),
                    np.cov(chn_new['samples'], ddof=0, rowvar=False)
                )

                cov_chn, mean_chn = aggregate_covs(
                    [chn_master['cov_draws'], new_cov],
                    [chn_master['mean_draws'], new_mean],
                    [len(chn_master['samples']), len(chn_new['samples'])]
                )

                new_ch_dict = {
                    **{

                        'samples': np.vstack([chn_master['samples'], chn_new['samples']]),

                        'seed': chn_master['seed'],
                        'accepteds': np.hstack([chn_master['accepteds'], chn_new['accepteds']]),
                        'scalers': np.hstack([chn_master['scalers'], chn_new['scalers']]),
                        'acceptance_probs': np.hstack([chn_master['acceptance_probs'], chn_new['acceptance_probs']]),

                        'cov_draws': cov_chn,
                        'mean_draws': mean_chn,

                        'step_cov0': chn_master['step_cov0'],

                        'thinning': thinning,
                        'log_posterior': np.hstack([chn_master['log_posterior'], chn_new['log_posterior']]),
                        'itr': chn_new['itr'],

                    },
                    **{
                        kk: chn_master[kk] + chn_new[kk]
                        for kk in ['lp_time', 'cov_time', 'draw_time', 'accept_decision_time', 'fit_elapsed',
                                   'setup_time', 'n_burnin', 'n_samples', 'bloc', 'pid',
                                   'num_jump_successes',  'num_jump_tries',
                                   ]
                    }
                }

                chain_results_master[i] = new_ch_dict

            x0s_formatted[i] = chain_results[i]['samples'][-1].copy()
            # temp = chain_results[i]['samples'][-1].copy()
            # if log_posterior_bounded(x0s[i]) < log_posterior(temp):
            #    x0s[i] = temp

        subchain_cnt += 1

#         cov_new_master, _ = aggregate_covs(
#             [c['cov_draws'] for c in chain_results_master],
#             [c['mean_draws'] for c in chain_results_master],
#             [len(c['samples']) for c in chain_results_master],
#         )

        _time_agg_cov = time.time()
        cov_window_start = len(chain_results_master[0]['samples']) // 4
        if debug:
            print(f"Updating covariance structure on samples {cov_window_start} to "
                  f"{len(chain_results_master[0]['samples'])}...", end='')

        cov_new_master, _ = get_total_covariance_from_batches(
            [c['samples'] for c in chain_results_master],
            window_start=cov_window_start)

        step_cov_old = step_cov
        if is_burnin_subchain[0] or not stop_adaptation_after_burnin:
            # step_cov = step_cov_initial_samples * cov_new_master + (1.0 - step_cov_initial_samples) * step_cov_old
            n_draws = (n_samples_master + n_burnin_master) * n_chains
            # print("WE UPDATED!!!!!", (is_burnin_subchain, not stop_adaptation_after_burnin, 'n_draws', n_draws, step_cov_initial_samples))
            step_cov = (
                    (n_draws * cov_new_master + step_cov_initial_samples * step_cov0_orig)
                    / (n_draws + step_cov_initial_samples)
            )

        if has_fixed_params:
            for j in fix_params_arg['index']:
                step_cov[:, j] = 0.
                step_cov[j, :] = 0.
                step_cov[j, j] = 1.

        if debug:
            print(f"done! ({time.time()-_time_agg_cov:.2f}s)")

        # if debug and show_r_hat_every_subchain:
        #
        #     print("\nMaximum Log Posterior (so far..):")
        #     print(pd.DataFrame(
        #         {'max lp': [c['log_posterior'].max() for c in chain_results_master]}
        #     ).transpose().to_string())
        #
        #     if is_burnin_subchain or not stop_adaptation_after_burnin:
        #         step_cov_slog_dets = {
        #             'step_cov_old': np.linalg.slogdet(step_cov_old[not_fixed_param_index][:, not_fixed_param_index])[1],
        #             'step_cov_update':
        #                 np.linalg.slogdet(cov_new_master[not_fixed_param_index][:, not_fixed_param_index])[1],
        #             'step_cov_updated': np.linalg.slogdet(step_cov[not_fixed_param_index][:, not_fixed_param_index])[1],
        #         }
        #         # step_cov_slog_dets['relative_scales'] = np.exp(
        #         #     step_cov_slog_dets['step_cov_update'] - step_cov_slog_dets['step_cov_old'])
        #
        #         print("\nStep Cov slogdet (log Generalized Variance):")
        #         pprint.pprint(step_cov_slog_dets)

        if subchain_cnt < n_subchains:
            if debug:
                prompt_message = get_diagnostic_update_message(
                    chain_results_master, param_names, n_chains, _time_mcmc_draws, _time_master,
                    transformation_function,
                    fixed_param_idx, thinning, callback_function=callback_function,
                    title_string='Adaptive Metropolis-Hastings')
                print(prompt_message)

        else:
            if user_prompt_for_more_iters:

                prompt_message = get_diagnostic_update_message(
                    chain_results_master, param_names, n_chains, _time_mcmc_draws, _time_master,
                    transformation_function,
                    fixed_param_idx, thinning, callback_function=callback_function,
                    title_string='Adaptive Metropolis-Hastings')
                wait_time_start = time.time()

                new_draws = user_prompt_for_more_iters_method(prompt_message, do_prompt=True,
                                                              print_not_converged=False, assert_even=True)

                _wait_time += time.time() - wait_time_start

                if new_draws <= 0:
                    break

                n_samples = new_draws
                n_burnin = 0

                n_subchains, subchain_n_burnin_iters, subchain_n_samples_iters, is_burnin_subchain, subchain_cnt \
                    = get_subchain_n_draws(n_burnin, n_samples,
                                           max_subchain_draws_burnin=max_subchain_draws_burnin,
                                           max_subchain_draws_sample=max_subchain_draws_sample)

            else:
                break

        if do_diff_evolution_mc:
            if not stop_adaptation_after_burnin or is_burnin_subchain:
                diff_evolution_past_samples = np.vstack([
                    c['samples'][rand.randint(
                        low=int(diff_evolution_frac_burnin * len(c['samples'])),
                        high=len(c['samples']),
                        size=min(
                            diff_evolution_max_draws // n_chains,
                            int((1 - diff_evolution_frac_burnin) * len(c['samples']))
                        )
                    )]
                    for c in chain_results_master
                ])
                if ray.is_initialized():
                    diff_evolution_past_samples_id = ray.put(diff_evolution_past_samples)
                else:
                    diff_evolution_past_samples_id = diff_evolution_past_samples

        # if show_cov_every_subchain:
        #     with warnings.catch_warnings():
        #         warnings.simplefilter("ignore")
        #         df_temp = pd.DataFrame({
        #             'cov': np.abs(np.diag(cov_new_master)) ** .5,
        #             'step_old': np.abs(np.diag(step_cov_old)) ** .5,
        #             'step_new': np.abs(np.diag(step_cov)) ** .5,
        #         }, index=param_names)
        #         if diff_evolution_past_samples is not None:
        #             df_temp['past_samp'] = np.std(diff_evolution_past_samples, axis=0)
        #         print(df_temp.to_string())

        scaler0 = np.mean([np.mean(c['scalers'][-100:]) for c in chain_results_master])
        itr0 = chain_results_master[0]['itr']

    cov_params_unbndd, mean_params_unbndd = convert_samples_to_unbounded_space(
        chain_results_master, transformations, num_params, debug=debug, key='samples')

    options = {'method': AMH_METHOD,
               'min_scaler': min_scaler, 'max_scaler': max_scaler, 'scaler_adjust_rate': scaler_adjust_rate,
               'scaler_adjust_denom_power': scaler_adjust_denom_power,
               'scaler0': scaler0_orig, 'draw_size': draw_size,
               'target_acceptance_rate': target_acceptance_rate, 'seed': seed,
               'max_processes': max_processes, 'log_posterior_bounded': log_posterior_bounded,
               'log_posterior': log_posterior,
               'max_subchain_draws_burnin': max_subchain_draws_burnin,
               'max_subchain_draws_sample': max_subchain_draws_sample,
               'n_burnin': n_burnin_master, 'n_samples': n_samples_master, 'n_chains': n_chains,
               'x0': start_params.copy(),
               'step_cov': step_cov0_orig.copy(), 'do_adaptive': do_adaptive,
               'bounds': bounds.copy() if bounds is not None else None,
               'proposal_df': proposal_df,
               'do_parallel': do_parallel,
               # 'resample_k': resample_k,
               # 'show_cov_every_subchain': show_cov_every_subchain,
               # 'step_cov_adjust_rate': step_cov_initial_samples,
               'pbar_update_cadence': pbar_update_cadence,
               'normalize_step_cov': normalize_step_cov,
               'thinning': thinning,
               'stop_adaptation_after_burnin': stop_adaptation_after_burnin,

               # DE-MC params
               'do_diff_evolution_mc': do_diff_evolution_mc,
               'diff_evolution_frac_burnin': diff_evolution_frac_burnin,
               'diff_evolution_max_draws': diff_evolution_max_draws,
               'diff_evolution_weight': diff_evolution_weight,
               'diff_evolution_jump_cadence': diff_evolution_jump_cadence,

               'scalar_jitter_bounds': scalar_jitter_bounds,

               'callback_function': callback_function,
               }

    other_info = dict(
        scaler=scaler0,
        step_cov=step_cov,
        cov_params_unbounded_space=cov_params_unbndd,
        step_cov_initial_samples=DEFAULT_AM_STEP_COV_INITIAL_SAMPLES,
        mean_params_unbounded_space=mean_params_unbndd,
        diff_evolution_past_samples=None if diff_evolution_past_samples is None else diff_evolution_past_samples.copy(),
    )
    mcmc_time = time.time() - _time_mcmc_draws - _wait_time
    total_time = time.time() - _time_master - _wait_time

    if ray.is_initialized():
        if debug:
            print('\nShutting down `ray`...', end='')
        ray.shutdown()
        if debug:
            print('done!')

    if debug:
        print(f'\nMCMC Drawing Complete... '
              f'\n\tTotal draws:      {thinning * chain_results_master[0]["samples"].shape[0] * n_chains}'
              f'\n\tTotal samples:    {chain_results_master[0]["samples"].shape[0] * n_chains}'
              f'\n\tMCMC draw time:   {"%.2fs" % mcmc_time}'
              f'\n\tWait time:        {"%.2fs" % _wait_time}'
              f'\n\tTime per draw:    {1000 * mcmc_time / (thinning * chain_results_master[0]["samples"].shape[0] * n_chains):.3f}ms'
              f'\n\tTime per sample:  {1000 * mcmc_time / (chain_results_master[0]["samples"].shape[0] * n_chains):.3f}ms'
              f'\n\tTotal time:       {"%.2fs" % total_time}\n')

    return MCMCResults(AMH_METHOD, num_params, log_posterior, log_posterior_jacobian_adjustment,
                       param_names, chain_results_master,
                       total_time, n_burnin_master, n_chains, thinning, options, specification_name, debug=debug,
                       model=model, fix_params=fix_params, other_info=other_info, transformations=transformations
                       )

# if __name__ == '__main__':
#
#     n = 500
#     np.random.seed(0)
#     x = np.random.randn(n)
#     y = 1.3 * x + np.random.randn(n) * .5
#     data = {'x': x, 'y': y}
#
#     from kanly.regression.nonlinear_least_squares.model import SparseNonlinearLeastSquaresModel
#     nlls_model = SparseNonlinearLeastSquaresModel.build_model_from_formula(
#         '[y]~{x}*[x]', data
#     )
#     llf = nlls_model.get_log_likelihood_function()
#
#     fit = mcmc(llf, [0,1], n_chains=4, n_burnin=100, n_samples=100, param_names=['x', '__sigma2'],
#                fix_params={1: 1.1}, user_prompt_for_more_iters=True,
#                debug=True)
#     print(fit)
#
#     print(fit.sample_df)
#     print(fit.sample_df.cov())
#     print(fit.sample_df['__sigma2'].unique())
#     print(fit.sample_df[fit.sample_df['__sigma2']==1])
#
#     # l2_penalty = {'x': 50.}
#     # reg_2_vals = {'x': 5}
#
# if __name__ == '__main__':
#     from kanly.api import bayes_nonlinear_regression_model
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt
#
#     np.random.seed(0)
#     n = 30
#     x = 3.7 * np.random.rand(n)
#     y = 1.2 + .9 * x ** .5 + .3 * np.random.randn(n)
#     data = dict(x=x, y=y)
#     plt.scatter(x, y)
#
#     model = bayes_nonlinear_regression_model('[y] ~ {a} + {b}*[x]**(1-{c})', data, bounds={'c': [0, .995]},
#                                              do_bounded_transform=False, do_njit=False)
#
#     t = time.time()
#     M = 5000
#     for i in range(M):
#         model.log_likelihood_function([1, 1, .7, .5])
#     print(">  ", M/(time.time() - t))
#
#     M = 5000
#     _acceptance_choice(1, 1, 1, 1, 1.4, 1, 1)
#     for i in range(M):
#         _acceptance_choice(1,1,1,1,1.4,1,1)
#     print(">  ", M/(time.time() - t))
#
#     fit = model.mcmc([1, 1, .7, .5], n_burnin=30000, n_samples=50_000, max_subchain_draws=20_000,
#                      user_prompt_for_more_iters=True, do_parallel=True, debug=True)
#
# #     print(fit.chain_results)
# if __name__ == '__main__':
#     from kanly.api import bayes_nlls_model
#
#     np.random.seed(0)
#     n = 600
#     x = .56 * 1.2 *np.random.rand(n)
#     z = np.random.rand(n)
#     y = 3 + 10 * x - 2 * z + np.random.randn(n) * 3
#     wts = .01 + np.random.rand(n)
#     data = {'x': x, 'y': y, 'z': z, 'wts': wts}
#
#     model = bayes_nlls_model(
#         '[y] ~ {a}+{b}*[x]**{c}', data, bounds={'b': [5, 8], 'a': [1,6], 'c': [.05, .98]}
#     )
#
#     n_burn, n_samp = 5_000, 10_000
#
#     fit1 = model.amha(
#         [2, 6, .5, 1.], user_prompt_for_more_iters=False,
#         debug=True,
#         n_chains=4,
#         n_burnin=n_burn,
#         n_samples=n_samp,
#         #show_r_hat_ever_subchain=True,
#         max_subchain_draws_burnin=5_000,
#         max_subchain_draws_sample=15_000,
#         thinning=2,
#         # fix_params={'a': 2.6},
#         # do_metropolis_de=True,
#         # de_frac_burnin=.33,
#     )
#     fit1.kde('b', show=True)
#     fit1.diagnostic_plot('b', show=True)
#     fit1.hpdi_plot('b', .8, show=True)
#     # print(fit1)
#     #
#     # fit2 = model.amha(
#     #     [2, 6, .5, 1.], user_prompt_for_more_iters=False,
#     #     debug=True,
#     #     n_chains=4,
#     #     n_burnin=n_burn,
#     #     n_samples=n_samp,
#     #     show_r_hat_ever_subchain=True,
#     #     max_subchain_draws_burnin=5_000,
#     #     max_subchain_draws_sample=35_000,
#     #     do_diff_evolution_mc=True,
#     #     diff_evolution_weight=.9,
#     #     thinning=2,
#     # )
#     # print(fit2)
#     #
#     # print()
#     # print('$' * 200)
#     # print()
#     #
#     # print(fit1)
#     # print(fit2)
#
#     # fit1.multi_hist(['b', 'c'], show=True, suptitle='fit1')
#     # fit2.multi_hist(['b', 'c'], show=True, suptitle='fit2')
