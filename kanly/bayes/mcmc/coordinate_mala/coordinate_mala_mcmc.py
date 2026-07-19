"""
Coordinate-wise Metropolis-Adjusted Langevin Algorithm (MALA) MCMC sampler.

This module implements ``mala``—the user-facing entry point—and the per-chain
kernel ``run_mala_chain``.  Unlike the full-dimensional AMH sampler, this
algorithm updates one coordinate at a time using a Langevin-drift proposal:

    x1[coord] = x0[coord] + tau[coord] * ∇_coord log_π(x0) + √(2·tau[coord]) · ε

where ε is a standard normal (or Cauchy) innovation, and ``tau[coord]`` is
an individual per-coordinate step size.  The Metropolis acceptance probability
corrects for the asymmetry of the Langevin proposal kernel, yielding a valid
reversible Markov chain.

Each coordinate's tau is adapted after every iteration using a simple
stochastic approximation rule aimed at ``target_acceptance_rate``::

    tau[coord] *= 1 + tau_adjust * (accepted - target_accept)

Optionally, periodic Differential Evolution (DE) jumps are applied at
a cadence ``diff_evolution_step_cadence`` to help escape correlated or
multimodal regions.

Parallelism across chains is provided by Ray.
"""
from __future__ import absolute_import, print_function

import os
import time

import numpy as np
from scipy.stats import cauchy

import warnings
# import ray
# from ray.experimental.tqdm_ray import tqdm

from kanly.bayes.mcmc.check_starting_point import check_starting_point
from kanly.bayes.mcmc.coordinate_mala.constants import (
    DEFAULT_COORD_MALA_THINNING,
    DEFAULT_COORD_MALA_N_CHAINS, DEFAULT_COORD_MALA_DEBUG,
    DEFAULT_COORD_MALA_DO_CAUCHY, DEFAULT_COORD_MALA_DO_MALA, DEFAULT_COORD_MALA_N_SAMPLES,
    DEFAULT_COORD_MALA_TAU_ADJUST, DEFAULT_COORD_MALA_P_BAR_UPDATE_CADENCE, DEFAULT_COORD_MALA_TARGET_ACCEPTANCE_RATE,
    DEFAULT_COORD_MALA_USER_PROMPT_FOR_MORE_ITERS,
    DEFAULT_COORD_MALA_MAX_SUBCHAIN_DRAWS,

    DEFAULT_COORD_MALA_DIFF_EVOLUTION_STEP_CADENCE, DEFAULT_COORD_MALA_DIFF_EVOLUTION_FRAC_BURNIN,
    DEFAULT_COORD_MALA_DIFF_EVOLUTION_MAX_DRAWS,

    METHOD as CD_MALA_METHOD, DEFAULT_COORD_MALA_FRAC_BURNIN
)

from kanly.bayes.mcmc.format_starting_point_for_mcmc import format_starting_point_for_mcmc
from kanly.bayes.mcmc.diagnostics.diagnostics import get_diagnostic_update_message
from kanly.bayes.mcmc.mcmc_results import MCMCResults
from kanly.bayes.parameter_transformations import convert_samples_to_unbounded_space
from kanly.utils.print_options import print_options
from kanly.utils.user_prompt_for_more_iters import user_prompt_for_more_iters_method


def run_mala_chain(lp_func, lp_adj_func, x0_start, seed, n_samples, tau_start=None,
                   thinning=DEFAULT_COORD_MALA_THINNING,
                   target_accept=DEFAULT_COORD_MALA_TARGET_ACCEPTANCE_RATE, do_cauchy=DEFAULT_COORD_MALA_DO_CAUCHY,
                   pbar_update_cadence=DEFAULT_COORD_MALA_P_BAR_UPDATE_CADENCE, do_mala=DEFAULT_COORD_MALA_DO_MALA,
                   debug=False, tau_adjust=DEFAULT_COORD_MALA_TAU_ADJUST, bloc=0, fix_param_set=None,
                   position=0, diff_evolution_step_cadence=DEFAULT_COORD_MALA_DIFF_EVOLUTION_STEP_CADENCE,
                   past_samples=None, diff_evolution_scale=1.0):

    """Run a single coordinate-wise MALA (or random-walk Metropolis) chain.

    Iterates through ``n_samples * thinning`` sub-steps.  Each sub-step
    picks one coordinate at random (excluding fixed parameters), forms
    a Langevin-drift proposal in that coordinate, evaluates the
    Metropolis–Hastings acceptance probability (with the MALA correction
    for proposal asymmetry), and updates ``tau[coord]`` to steer the
    per-coordinate acceptance rate toward ``target_accept``.

    Optionally, every ``diff_evolution_step_cadence`` iterations a
    full-dimensional DE jump is attempted instead of a coordinate update.

    Args:
        lp_func: Callable ``lp_func(x)`` → scalar log-posterior (including
            Jacobian adjustment if supplied via ``lp_adj_func``).
        lp_adj_func: Optional callable returning the Jacobian log-determinant
            of the bounded→unbounded transform.  Treated as zero when ``None``.
        x0_start: Starting parameter vector (1-D array-like).
        seed: Integer random seed for ``numpy.random.RandomState``.
        n_samples: Number of thinned draws to store.
        tau_start: Initial per-coordinate step sizes (1-D array of length
            ``num_params``).  Defaults to ``1e-4`` for all coordinates when ``None``.
        thinning: Store every ``thinning``-th iteration.
        target_accept: Desired per-coordinate acceptance rate for adaptation.
        do_cauchy: Whether to use a Cauchy (instead of Gaussian) innovation;
            the Langevin correction is adjusted accordingly.
        pbar_update_cadence: Seconds between tqdm refresh cycles.
        do_mala: Whether to use the Langevin drift in the proposal.  When
            ``False`` this reduces to a coordinate random-walk Metropolis.
        debug: Whether to display a tqdm progress bar.
        tau_adjust: Learning rate for the per-coordinate tau adaptation rule.
        bloc: Integer segment identifier appended to the result dict.
        fix_param_set: Optional set of parameter indices to hold fixed;
            those coordinates are excluded from the coordinate selection.
        position: tqdm bar position for multi-chain display.
        diff_evolution_step_cadence: How often (in iterations) to perform a
            full-dimensional DE jump instead of a coordinate MALA step.
            ``None`` or ``0`` disables DE jumps.
        past_samples: Optional pre-supplied 2-D array of past draws used as
            the DE gene pool.  When ``None``, draws from the current chain's
            history are used.
        diff_evolution_scale: Initial global scalar for the DE jump size;
            adapted using a 0.234-target rule after each DE step.

    Returns:
        Dict with keys: ``'samples'`` (2-D array of shape ``(n_samples, p)``),
        ``'log_posterior'``, ``'tau'`` (final per-coordinate step sizes),
        ``'num_tries'``, ``'num_accepted'``, ``'acceptance_rate'``,
        ``'diff_evolution_scale'``, ``'chain_elapsed'``, ``'bloc'``, ``'pid'``,
        and a ``'settings'`` sub-dict.
    """
    __t = time.time()

    if lp_adj_func is None:
        lp_adj_func = lambda x: 0.0

    err_settings = np.seterr()
    np.seterr(all='ignore')

    x0 = np.array(x0_start).astype(float)

    last_update_time = 0

    def deriv_lp(i, value, f0=None, dx=1e-6):
        """Compute the finite-difference partial derivative of the log-posterior at coordinate ``i``.

        Uses a relative step size ``di = max(|value[i]| * dx, 1e-12)`` to
        avoid numerical issues near zero.  The adjustment log-determinant is
        included in the evaluation at ``value1`` so the gradient reflects the
        full adjusted log-posterior.

        Args:
            i: Coordinate index at which to differentiate.
            value: Current parameter vector (1-D array).
            f0: Pre-evaluated log-posterior (+ Jacobian adjustment) at ``value``;
                re-evaluated from ``lp_func`` when ``None``.
            dx: Relative finite-difference step size.

        Returns:
            Scalar finite-difference estimate of ``d/dx_i log_π(value)``.
        """
        if f0 is None:
            f0 = lp_func(value)
        di = max(min(np.abs(value[i]), 1.) * dx, 1e-12)
        value1 = np.array(value)
        value1[i] += di
        fi = lp_func(value1) + lp_adj_func(value1)

        val = (fi - f0) / di
        return val

    p = len(x0)

    rand = np.random.RandomState(seed)
    if fix_param_set:
        rand_coord = rand.choice(list(set(range(p)) - set(fix_param_set)),
                                 n_samples * thinning, replace=True)
    else:
        rand_coord = rand.randint(0, p, n_samples * thinning)
    rand_log_uniform = np.log(rand.rand(n_samples * thinning))

    if do_cauchy:
        rand_innovation = rand.rand(n_samples * thinning)
        rand_innovation = cauchy.ppf(rand_innovation)
    else:
        rand_innovation = rand.randn(n_samples * thinning)

    if tau_start is None:
        tau = np.full(p, 1e-4)
    else:
        tau = np.array(tau_start, copy=True)

    draws = np.zeros((n_samples, p))
    log_posteriors = np.zeros(n_samples)

    lp0 = lp_func(x0)
    lp_adj0 = lp_adj_func(x0)

    num_tries = np.zeros(p)
    num_accepted = np.zeros(p)

    # do_resample = resample_k is not None

    if debug:
        from ray.experimental.tqdm_ray import tqdm
        pbar = tqdm(total=thinning * n_samples, position=position)

    # taus = [tau]

    for itr, (coord, innov, log_uniform) in enumerate(zip(rand_coord, rand_innovation, rand_log_uniform)):

        if diff_evolution_step_cadence and itr >= 100 and itr % diff_evolution_step_cadence == 0:
            # differential evolution step:

            if past_samples is None:
                rnd_draw_idx = rand.randint(int(1 / 2 * itr / thinning), itr // thinning, size=2)
                direction = draws[rnd_draw_idx[0]] - draws[rnd_draw_idx[1]]
            else:
                draw_past_bool = rand.rand(2) < .8
                draws_past = [
                    past_samples[rand.randint(len(past_samples))]
                    if d else
                    draws[rand.randint(int(1 / 2 * itr / thinning), itr // thinning)]
                    for d in draw_past_bool
                ]
                direction = draws_past[1] - draws_past[0]

            x1 = x0 + diff_evolution_scale * direction
            lp1 = lp_func(x1)
            lp_adj1 = lp_adj_func(x1)

            accepted = int(log_uniform < lp1 + lp_adj1 - (lp0 + lp_adj0))
            if accepted:
                x0, lp0, lp_adj0 = x1, lp1, lp_adj1
            diff_evolution_scale *= (1 + .05 * (accepted - .234))

        else:
            # coordinate mala step
            if do_mala:
                deriv0 = deriv_lp(coord, x0, f0=lp0 + lp_adj0)
            else:
                deriv0 = 0

            x1 = np.array(x0)
            x1[coord] += tau[coord] * deriv0 + np.sqrt(2 * tau[coord]) * innov

            lp1 = lp_func(x1)
            lp_adj1 = lp_adj_func(x1)

            accepted = 0
            if np.isfinite(lp1):

                if do_mala:
                    deriv1 = deriv_lp(coord, x1, f0=lp1 + lp_adj1)

                    if do_cauchy:

                        # prob of jumping to x0 from x1
                        q_x1_to_x0 = -np.log(1 + (x0[coord] - (x1[coord] + tau[coord] * deriv1)) ** 2 / (2 * tau[coord]))

                        # prob of jumping to x1 from x0
                        q_x0_to_x1 = -np.log(1 + (x1[coord] - (x0[coord] + tau[coord] * deriv0)) ** 2 / (2 * tau[coord]))
                    else:

                        # prob of jumping to x0 from x1
                        q_x1_to_x0 = -1. / (4 * tau[coord]) * (x0[coord] - (x1[coord] + tau[coord] * deriv1)) ** 2

                        # prob of jumping to x1 from x0
                        q_x0_to_x1 = -1. / (4 * tau[coord]) * (x1[coord] - (x0[coord] + tau[coord] * deriv0)) ** 2
                else:
                    q_x1_to_x0, q_x0_to_x1 = 0, 0

                log_accept_prob = (
                        + (lp1 + lp_adj1)
                        - (lp0 + lp_adj0)
                        + q_x1_to_x0
                        - q_x0_to_x1
                )

                if log_uniform < log_accept_prob:
                    x0, lp0, lp_adj0 = x1, lp1, lp_adj1
                    accepted = 1

            # not_burned_in = still_burnin and (np.any(num_accepted < 100) or np.any(num_tries - num_accepted < 100))
            # if itr < thinning * n_burnin or not_burned_in:
            tau[coord] *= 1 + tau_adjust * (accepted - target_accept)

            # if still_burnin and not not_burned_in:
            #     still_burnin = False
            #     if debug:
            #         print('burn in end at itr ', itr)

            num_tries[coord] += 1
            num_accepted[coord] += accepted

        if itr % thinning == 0:
            draws[itr // thinning] = x0
            log_posteriors[itr // thinning] = lp0

        if debug:
            if time.time() - last_update_time > pbar_update_cadence:
                last_update_time = time.time()
                pbar.update(itr - pbar._x)
                pbar.set_description(
                    f'lp = {lp0:8.4e}, coord = {coord:3d}, tau = {tau[coord]:6.2e}, accept = {num_accepted[coord] / num_tries[coord]:6.3e}')

    np.seterr(**err_settings)

    return {
        'settings': {
            'seed': seed,
            'n_samples': n_samples,
            'thinning': thinning,
            'target_accept': target_accept,
            'x0_start': x0_start,
            # 'resample_k': resample_k,
            'do_cauchy': do_cauchy,
            'diff_evolution_step_cadence': diff_evolution_step_cadence,
            'diff_evolution_scale': diff_evolution_scale,
        },
        'samples': draws,
        'log_posterior': log_posteriors,
        'tau': tau,
        'diff_evolution_scale': diff_evolution_scale,
        'num_tries': num_tries,
        'num_accepted': num_accepted,
        'acceptance_rate': num_accepted / (num_tries + 1e-15),
        'chain_elapsed': time.time() - __t,
        'bloc': [bloc] * n_samples,
        'pid': [os.getpid()] * n_samples,
    }


# run_mala_chain_remote = ray.remote(run_mala_chain)


def get_past_samples(chain_results_master, rand, n_chains, diff_evolution_frac_burnin, diff_evolution_max_draws):
    """Build the Differential Evolution past-sample pool from the current chain history.

    For each chain, randomly selects ``diff_evolution_max_draws // n_chains``
    draws from the latter ``(1 - diff_evolution_frac_burnin)`` portion of the
    chain (i.e., skipping the early warm-up draws) and stacks them into a
    single 2-D array.  The resulting pool is used by subsequent sub-chains as
    the DE gene pool for cross-chain jump proposals.

    Args:
        chain_results_master: List of per-chain result dicts, each containing
            a ``'samples'`` array of shape ``(n_draws, num_params)``.
        rand: ``numpy.random.RandomState`` instance for reproducible sampling.
        n_chains: Number of chains (controls the per-chain allocation from
            the total pool size).
        diff_evolution_frac_burnin: Fraction of each chain's draws to treat
            as early warm-up and exclude from the pool.
        diff_evolution_max_draws: Total maximum pool size; each chain
            contributes ``diff_evolution_max_draws // n_chains`` draws.

    Returns:
        2-D array of shape approximately ``(diff_evolution_max_draws, num_params)``
        containing randomly selected post-warm-up draws from all chains.
    """
    return np.vstack(
        [
            C['samples'][rand.choice(
                range(int(len(C['samples']) * diff_evolution_frac_burnin), len(C['samples'])),
                diff_evolution_max_draws // n_chains,
                replace=True
            )]
            for C in chain_results_master
        ]
    )


def mala(log_posterior, start_params, seed=0,
         log_posterior_jacobian_adjustment=None,
         start_params_is_original_scale=True,
         transformations=None,
         param_names=None,
         specification_name=None,
         n_chains=DEFAULT_COORD_MALA_N_CHAINS,
         n_samples=DEFAULT_COORD_MALA_N_SAMPLES,
         frac_burnin=DEFAULT_COORD_MALA_FRAC_BURNIN,
         thinning=DEFAULT_COORD_MALA_THINNING,
         # resample_k=DEFAULT_COORD_MALA_RESAMPLE_K,
         target_acceptance_rate=DEFAULT_COORD_MALA_TARGET_ACCEPTANCE_RATE, do_cauchy=DEFAULT_COORD_MALA_DO_CAUCHY,
         do_mala=DEFAULT_COORD_MALA_DO_MALA,
         debug=DEFAULT_COORD_MALA_DEBUG,
         pbar_update_cadence=DEFAULT_COORD_MALA_P_BAR_UPDATE_CADENCE,
         user_prompt_for_more_iters=DEFAULT_COORD_MALA_USER_PROMPT_FOR_MORE_ITERS,
         tau_adjust=DEFAULT_COORD_MALA_TAU_ADJUST,
         fix_params=None, model=None, max_processes=12,
         diff_evolution_step_cadence=DEFAULT_COORD_MALA_DIFF_EVOLUTION_STEP_CADENCE,
         max_subchain_draws=DEFAULT_COORD_MALA_MAX_SUBCHAIN_DRAWS,
         diff_evolution_frac_burnin=DEFAULT_COORD_MALA_DIFF_EVOLUTION_FRAC_BURNIN,
         diff_evolution_max_draws=DEFAULT_COORD_MALA_DIFF_EVOLUTION_MAX_DRAWS,
         callback_function=None,
         tau0=None
         ) -> MCMCResults:
    """Run Coordinate MALA MCMC across multiple parallel chains.

    This is the primary user-facing entry point for the coordinate-wise MALA
    sampler.  It:

    1. Formats and validates starting parameter vectors via
       ``format_starting_point_for_mcmc``, applying any bounded-to-unbounded
       transformations.
    2. Splits the total draw count into sub-chains of at most
       ``max_subchain_draws`` draws so that Ray tasks stay responsive and the
       per-coordinate tau values and DE pool can be updated between runs.
    3. Dispatches each sub-chain to all chains in parallel via Ray.
    4. After each sub-chain merges results, updates the DE past-sample pool,
       re-computes the starting states, and (when ``debug=True``) prints a
       convergence summary.
    5. When ``user_prompt_for_more_iters=True``, asks the user whether to
       continue sampling after all sub-chains complete.
    6. Assembles the final ``MCMCResults`` object.

    Args:
        log_posterior: Callable ``log_posterior(x)`` → scalar log-posterior
            on the (possibly unbounded-transformed) parameter space.
        start_params: Starting values as a 1-D array (one set shared, with
            per-chain jitter), 2-D array of shape ``(n_chains, num_params)``,
            or list of 1-D arrays.
        seed: Base integer random seed; per-chain seeds are derived from this.
        log_posterior_jacobian_adjustment: Optional callable returning the
            Jacobian log-determinant of the bounded→unbounded transform.
        start_params_is_original_scale: When ``True`` (default), starting
            values are in the original (bounded) space and are transformed
            before sampling.
        transformations: Dict ``{param_index: TransformedParameter}`` for
            parameters reparameterized to an unbounded space.
        param_names: List of parameter name strings.  Auto-generated when ``None``.
        specification_name: Optional display label for the model run.
        n_chains: Number of parallel MCMC chains.
        n_samples: Total number of draws to collect per chain (including burn-in).
        frac_burnin: Fraction of draws to classify as burn-in in the
            ``MCMCResults`` object (e.g. 0.25 → first 25% discarded).
        thinning: Store every ``thinning``-th iteration.
        target_acceptance_rate: Per-coordinate target acceptance rate for
            adapting ``tau``.
        do_cauchy: Whether to use Cauchy innovations for proposals.
        do_mala: Whether to use Langevin gradient drift; ``False`` gives a
            plain coordinate random-walk Metropolis.
        debug: Whether to print progress and convergence messages.
        pbar_update_cadence: Seconds between tqdm progress-bar refreshes.
        user_prompt_for_more_iters: Whether to ask the user to continue after
            all sub-chains finish.
        tau_adjust: Learning rate for the per-coordinate tau adaptation.
        fix_params: Dict ``{param_name_or_index: value}`` of fixed parameters.
        model: Optional reference to the originating ``BayesianModel``.
        max_processes: Maximum Ray worker processes.
        diff_evolution_step_cadence: Interval (in iterations) for DE jumps.
            ``None`` or ``0`` disables DE jumps.
        max_subchain_draws: Maximum draws per sub-chain.
        diff_evolution_frac_burnin: Fraction of chain history to skip when
            building the DE pool (avoids including early warm-up draws).
        diff_evolution_max_draws: Maximum DE pool size.
        callback_function: Optional callable invoked after each sub-chain
            with the current chain results dict; useful for checkpointing.
        tau0: Optional initial tau value; scalar (applied to all coordinates)
            or 1-D array of per-coordinate values.

    Returns:
        ``MCMCResults`` object containing all draws, diagnostics, and metadata.

    Examples
    --------
    Coordinate-wise MALA sampling for a 2-parameter Beta posterior:

    >>> import numpy as np
    >>> from scipy.stats import beta
    >>> from kanly.api import DataModel
    >>> rng = np.random.default_rng(0)
    >>> data = {'x': beta.rvs(a=5, b=2, size=1_500, random_state=0)}
    >>> dmo = DataModel.build_data_model(                      # doctest: +SKIP
    ...     'self.x = `x`',
    ...     'return nopython_logpdf_beta(x, a=$a$, b=$b$).sum()',
    ...     data, nopython=True)
    >>> model = dmo.to_bayesian_model(                          # doctest: +SKIP
    ...     bounds={'a': [0, np.inf], 'b': [0, np.inf]})
    >>> fit = model.mala(                                       # doctest: +SKIP
    ...     [1.0, 1.0],
    ...     n_samples=10_000, n_chains=4,
    ...     frac_burnin=0.3, do_mala=True)
    >>> print(fit)                                              # doctest: +SKIP

    For plain coordinate Metropolis without the Langevin drift term, set
    ``do_mala=False``. ``BayesianModel.sample(..., do_mala_cd_warmup=True)``
    runs MALA as a warm-up phase before switching to AMH.

    See Also
    --------
    :func:`amha` : Adaptive Metropolis–Hastings sampler.
    """

    time_start = time.time()

    if debug:
        print("------------------------------------------------------")
        print("Beginning Coordinate Metropolis-adjusted Langevin MCMC")
        print("------------------------------------------------------")

    x0s_formatted, transformations, fix_params, fix_params_transformed, param_names, num_params, transformation_function = \
        format_starting_point_for_mcmc(start_params, start_params_is_original_scale, n_chains, transformations, fix_params, param_names,
                                       debug=debug)

    fix_param_idx = fix_params.keys() if fix_params else None
    check_starting_point(x0s_formatted, log_posterior, log_posterior_jacobian_adjustment, debug)

    if debug:
        if specification_name is not None:
            print('\nSpecification Name: ', specification_name, '\n')

        print_options(
            dict(
                num_params=num_params,
                n_chains=n_chains,
                max_processes=max_processes,
                do_mala=do_mala,
                do_cauchy=do_cauchy,
                thinning=thinning,
                n_samples=n_samples,
                frac_burnin=frac_burnin,
                max_subchain_draws=max_subchain_draws,
                target_acceptance_rate=target_acceptance_rate,
                adaptive=True,
                seed=seed,
                tau0=tau0,
                tau_adjust=tau_adjust,
                fix_params=fix_params is not None,
                diff_evolution_step_cadence=diff_evolution_step_cadence,
                diff_evolution_max_draws=diff_evolution_max_draws,
                diff_evolution_frac_burnin=diff_evolution_frac_burnin,
                x0_is_original_scale=start_params_is_original_scale,
            ),
            title='COORDINATE MALA SETTINGS'
        )

    chain_results_master = None

    outer_cnt = 0

    diff_evolution_scale = 1.0
    past_samples_id = None

    bloc = 0
    rand = np.random.RandomState(seed)

    t_put = time.time()
    n_cpu = min(max_processes, n_chains)

    import ray

    if ray.is_initialized():
        try:
            if ray.available_resources()['CPU'] < n_cpu:
                ray.shutdown()
                ray.init(num_cpus=n_cpu, log_to_driver=debug)
        except:
            ray.shutdown()
            ray.init(num_cpus=n_cpu, log_to_driver=debug)
    else:
        ray.init(num_cpus=n_cpu, log_to_driver=debug)

    t_put = time.time()
    run_mala_chain_remote = ray.remote(run_mala_chain)

    if debug:
        print("Putting log posterior in `ray` storage...", end="")
    log_posterior_id = ray.put(log_posterior)
    log_posterior_jacobian_adjustment_id = ray.put(log_posterior_jacobian_adjustment)
    if debug:
        print(f"Done! ({time.time() - t_put:.2f}s)")

    n_samples_split_list = [max_subchain_draws] * (n_samples // max_subchain_draws) + [n_samples % max_subchain_draws]

    if tau0 is None:
        tau_start = None
    elif isinstance(tau0, (int, float)):
        tau_start = np.array([tau0]*num_params).astype(float)
    else:
        tau_start = np.array(tau0)

    while True:

        for n_samples_bloc in n_samples_split_list:
            bloc += 1

            # print(str({'bloc': bloc, 'n_samples': n_samples_bloc}))

            try:

                chain_args = [
                    (
                        log_posterior_id, log_posterior_jacobian_adjustment_id, start_params, seed_chain,
                        n_samples_bloc, tau_start, thinning, target_acceptance_rate,
                        do_cauchy,
                        pbar_update_cadence, do_mala, debug, tau_adjust, bloc,
                        fix_param_idx, pos, diff_evolution_step_cadence,
                        past_samples_id, diff_evolution_scale
                    )
                    for pos, (start_params, seed_chain) in enumerate(zip(x0s_formatted, rand.randint(0, 3_000_000, n_chains)))
                ]

                chain_results = ray.get(
                    [run_mala_chain_remote.remote(*arg) for arg in chain_args])

                time.sleep(.01)
                if debug:
                    print('\nJoining the data from the parallel jobs...')

            except KeyboardInterrupt:
                warnings.warn("\n\nKeyboard Interrupt! Stopping draws.\n\n")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                ray.shutdown()
                break
            except Exception as e:
                ray.shutdown()
                print("\n!!!" * 4)
                print(e.__traceback__)
                raise e

            outer_cnt += 1

            if chain_results_master is None:
                chain_results_master = chain_results
                for c in chain_results:
                    c['samples'] = np.array(c['samples'], copy=True)

            else:
                for old, new in zip(chain_results_master, chain_results):
                    old['tau'] = np.array(new['tau'], copy=True)
                    old['diff_evolution_scale'] = new['diff_evolution_scale']
                    for k in ['log_posterior', 'pid', 'bloc']:
                        old[k] = np.hstack([old[k], new[k]])
                    old['chain_elapsed'] += new['chain_elapsed']

                    for k in ['num_tries', 'num_accepted']:
                        old[k] = old[k] + new[k]

                    old['acceptance_rate'] = old['num_accepted'] / np.clip(old['num_tries'], 1, np.inf)
                    old['acceptance_rate'][old['num_tries'] == 0] = np.nan

                    for k in ['n_samples']:
                        old['settings'][k] += new['settings'][k]

                    old['samples'] = np.vstack([old['samples'], new['samples']])

                    del new['samples']

            x0s_formatted = [C['samples'][-1] for C in chain_results_master]
            tau_start = np.mean([c['tau'] for c in chain_results_master], axis=0)
            diff_evolution_scale = np.mean([c['diff_evolution_scale'] for c in chain_results_master])

            past_samples = get_past_samples(
                chain_results_master, rand, n_chains, diff_evolution_frac_burnin, diff_evolution_max_draws)
            if ray.is_initialized():
                past_samples_id = ray.put(past_samples)
            else:
                past_samples_id = past_samples

            if debug:
                message = get_diagnostic_update_message(
                    chain_results_master, param_names, n_chains, time_start, time_start, transformation_function,
                    fix_param_idx, thinning, callback_function=callback_function,
                    title_string='Coordinate Metropolis Adjusted Langevin (CD-MALA)'
                )

                print(message)

        if user_prompt_for_more_iters:

            tau_start = np.mean([c['tau'] for c in chain_results_master], axis=0)

            message = get_diagnostic_update_message(
                chain_results_master, param_names, n_chains, time_start, time_start, transformation_function,
                fix_param_idx, thinning, callback_function=callback_function,
                title_string='Coordinate Metropolis Adjusted Langevin (CD-MALA)'
            )

            n_samples_new = user_prompt_for_more_iters_method(message, do_prompt=True, assert_even=True)

            if n_samples_new <= 0 or n_samples_new is None or n_samples_new == '':
                break

            n_samples_split_list = ([max_subchain_draws] * (n_samples_new // max_subchain_draws) \
                               + [n_samples_new % max_subchain_draws])

            x0s_formatted = [C['samples'][-1] for C in chain_results_master]

        else:
            break

    for c in chain_results_master:
        c['accepteds'] = np.mean(c['num_accepted'][c['num_tries'] > 0] / c['num_tries'][c['num_tries'] > 0])

    num_samps = len(chain_results_master[0]['samples'])

    cov_params_unbndd, mean_params_unbndd = convert_samples_to_unbounded_space(
        chain_results_master, transformations, num_params, debug=debug, key='samples', window_start=num_samps // 4)

    options_dict = dict(
        method=CD_MALA_METHOD,
        seed=seed, do_adaptive=True, x0=start_params.copy(), n_chains=n_chains,
        n_samples=n_samples, thinning=thinning, frac_burnin=frac_burnin,
        # resample_k=resample_k,
        target_accept_rate=target_acceptance_rate, do_cauchy=do_cauchy, do_mala=do_mala,
        x0_is_original_scale=start_params_is_original_scale, debug=debug,
        tau_adjust=tau_adjust,
        max_subchain_draws=max_subchain_draws,
        diff_evolution_step_cadence=diff_evolution_step_cadence,
        diff_evolution_max_draws=diff_evolution_max_draws,
        diff_evolution_frac_burnin=diff_evolution_frac_burnin,
        callback_function=callback_function,
        tau0=tau0,
    )
    other_info = dict(
        tau_final=tau_start,
        cov_params_unbounded_space=cov_params_unbndd,
        mean_params_unbounded_space=mean_params_unbndd
    )

    if ray.is_initialized():
        if debug:
            print('\nShutting down `ray`...', end='')
        ray.shutdown()
        if debug:
            print('done!')

    mcmc_result = MCMCResults(
        CD_MALA_METHOD,
        num_params, log_posterior, log_posterior_jacobian_adjustment=log_posterior_jacobian_adjustment,
        param_names=param_names, chain_results=chain_results_master,
        mcmc_time=time.time() - time_start,
        n_burnin=int(frac_burnin * num_samps), n_chains=n_chains, thinning=thinning, options=options_dict,
        specification_name=specification_name,
        debug=debug, model=model, fix_params=fix_params, other_info=other_info,
        transformations=transformations
    )

    return mcmc_result

# if __name__ == '__main__':
#     from kanly.api import bayes_nonlinear_regression_model
#     import matplotlib.pyplot as plt
#     from kanly.bayes.mcmc.mcmc_results import MCMCResults
#
#     np.random.seed(0)
#     n = 50
#     x = 1.56 * np.random.randn(n)
#     z = np.random.rand(n)
#     y = 3 + 10 * x - 2 * z + np.random.randn(n) * 3
#     wts = .01 + np.random.rand(n)
#     data = {'x': x, 'y': y, 'z': z, 'wts': wts}
#
#     fit = bayes_nonlinear_regression_model(
#         '[y] ~ {a}+{b}*[x]', data, bounds={'b': [5, 8], 'a': [1, 6]}
#     ).mala(
#         [1.5, 6, 1.], user_prompt_for_more_iters=True, debug=False, n_samples=1_000,
#         #fix_params={'a': 2.6}
#     )
#
#     print(fit)
#     print(fit.__dict__.keys())
