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

During warm-up, each coordinate's tau is adapted after its updates using a
stochastic approximation rule aimed at ``target_acceptance_rate``::

    tau[coord] *= 1 + tau_adjust * (accepted - target_accept)

Step sizes, DE scale, and the DE history pool are fixed for retained sampling.
Optionally, periodic Differential Evolution (DE) jumps are applied at
a cadence ``diff_evolution_step_cadence`` to help escape correlated or
multimodal regions.

Parallelism across chains is provided by Ray.
"""
from __future__ import absolute_import, print_function

import os
import time
from math import isfinite

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

from kanly.bayes.mcmc.format_starting_point_for_mcmc import (
    format_starting_point_for_mcmc, validate_parameter_index,
)
from kanly.bayes.mcmc.diagnostics.diagnostics import get_diagnostic_update_message
from kanly.bayes.mcmc.mcmc_results import MCMCResults
from kanly.bayes.parameter_transformations import convert_samples_to_original_space
from kanly.utils.print_options import print_options
from kanly.utils.user_prompt_for_more_iters import user_prompt_for_more_iters_method


def _check_sampling_bounds(bounds, starts, num_params):
    """Validate hard bounds and formatted starts before evaluating the target."""
    if bounds is None:
        return None
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (2, num_params):
        raise ValueError('bounds must contain lower and upper vectors of length num_params')
    lower, upper = bounds
    if not np.all(lower <= upper):
        raise ValueError('bounds must be ordered and may not contain NaN')
    if not np.all((lower <= starts) & (starts <= upper)):
        raise ValueError('Starting point is outside the hard sampling bounds')
    if np.all(np.isneginf(lower)) and np.all(np.isposinf(upper)):
        return None
    return bounds


@np.errstate(all='ignore')
def run_mala_chain(lp_func, lp_adj_func, x0_start, seed, n_samples, tau_start=None,
                   thinning=DEFAULT_COORD_MALA_THINNING,
                   target_accept=DEFAULT_COORD_MALA_TARGET_ACCEPTANCE_RATE, do_cauchy=DEFAULT_COORD_MALA_DO_CAUCHY,
                   pbar_update_cadence=DEFAULT_COORD_MALA_P_BAR_UPDATE_CADENCE, do_mala=DEFAULT_COORD_MALA_DO_MALA,
                   debug=False, tau_adjust=DEFAULT_COORD_MALA_TAU_ADJUST, bloc=0, fix_param_set=None,
                   position=0, diff_evolution_step_cadence=DEFAULT_COORD_MALA_DIFF_EVOLUTION_STEP_CADENCE,
                   past_samples=None, diff_evolution_scale=1.0, n_warmup=0, bounds=None):

    """Run a single coordinate-wise MALA (or random-walk Metropolis) chain.

    Iterates through ``n_samples * thinning`` sub-steps.  Each sub-step
    picks one coordinate at random (excluding fixed parameters), forms
    a Langevin-drift proposal in that coordinate, evaluates the
    Metropolis–Hastings acceptance probability (with the MALA correction
    for proposal asymmetry).  During the first ``n_warmup`` stored draws,
    ``tau[coord]`` is updated to steer the per-coordinate acceptance rate
    toward ``target_accept``.  Proposal tuning is fixed afterward.

    Optionally, every ``diff_evolution_step_cadence`` iterations a
    full-dimensional DE jump is attempted instead of a coordinate update.

    Args:
        lp_func: Log-posterior evaluated at the model parameters corresponding
            to the unbounded vector x, before adding the Jacobian adjustment.
        lp_adj_func: Optional callable returning the Jacobian log-determinant
            of the map from unbounded x to the original model parameters.  Treated as zero when ``None``.
        x0_start: Starting parameter vector (1-D array-like).
        seed: Integer random seed for ``numpy.random.RandomState``.
        n_samples: Number of thinned draws to store.
        tau_start: Initial per-coordinate step sizes (1-D array of length
            ``num_params``).  Defaults to ``1e-4`` for all coordinates when ``None``.
        thinning: Store after every ``thinning`` iterations.
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
            a fixed DE gene pool.  When ``None``, only this call's warm-up
            draws are used; retained draws never enter the pool.  DE is
            skipped until at least two pool rows are available.
        diff_evolution_scale: Initial global scalar for the DE jump size;
            adapted using a 0.234-target rule during warm-up only.
        n_warmup: Number of the ``n_samples`` stored draws reserved for
            adaptation.  Defaults to zero (fixed-proposal sampling).
        bounds: Optional pair of inclusive lower and upper bound vectors in
            sampling coordinates. Impossible probes and proposals skip the target.

    Returns:
        Dict with keys: ``'samples'`` (2-D array of shape ``(n_samples, p)``),
        ``'log_posterior'``, ``'tau'`` (final per-coordinate step sizes),
        ``'num_tries'``, ``'num_accepted'``, ``'acceptance_rate'``,
        ``'diff_evolution_scale'``, ``'chain_elapsed'``, ``'bloc'``, ``'pid'``,
        and a ``'settings'`` sub-dict.
    """
    __t = time.time()

    x0 = np.array(x0_start, dtype=float, copy=True)
    if x0.ndim != 1 or not len(x0) or not np.all(np.isfinite(x0)):
        raise ValueError('x0_start must be a nonempty finite one-dimensional vector')
    p = len(x0)
    bounds = _check_sampling_bounds(bounds, x0, p)
    if bounds is not None:
        lower, upper = bounds
    if not isinstance(n_samples, (int, np.integer)) or n_samples < 0:
        raise ValueError('n_samples must be a nonnegative integer')
    if not isinstance(thinning, (int, np.integer)) or thinning < 1:
        raise ValueError('thinning must be a positive integer')
    if not isinstance(n_warmup, (int, np.integer)) or not 0 <= n_warmup <= n_samples:
        raise ValueError('n_warmup must be an integer between zero and n_samples')
    tau = np.full(p, 1e-4) if tau_start is None else np.array(tau_start, dtype=float, copy=True)
    if tau.shape != (p,) or not np.all(np.isfinite(tau)) or np.any(tau <= 0):
        raise ValueError('tau_start must contain one positive finite step size per parameter')
    if not np.isfinite(target_accept) or not 0 <= target_accept <= 1:
        raise ValueError('target_accept must be between zero and one')
    if not np.isfinite(tau_adjust) or tau_adjust < 0 or tau_adjust * target_accept >= 1:
        raise ValueError('tau_adjust must be finite, nonnegative, and preserve positive step sizes')
    if not np.isfinite(diff_evolution_scale) or diff_evolution_scale <= 0:
        raise ValueError('diff_evolution_scale must be positive and finite')
    if past_samples is not None:
        past_samples = np.array(past_samples, dtype=float, copy=True)
        if past_samples.ndim != 2 or past_samples.shape[1] != p or not np.all(np.isfinite(past_samples)):
            raise ValueError('past_samples must be a finite two-dimensional array with one column per parameter')

    last_update_time = 0

    def evaluate(x):
        lp = lp_func(x)
        adjustment = lp_adj_func(x) if lp_adj_func is not None and isfinite(lp) else 0.0
        return lp, adjustment

    def deriv_lp(i, x, adjusted_lp):
        """Use the same forward, backward, then zero drift rule at both endpoints."""
        xi = x[i]
        di = max(min(abs(xi), 1.) * 1e-6, 1e-12)
        probe = None
        for step in (di, -di):
            candidate = xi + step
            if not isfinite(candidate) or (bounds is not None and not lower[i] <= candidate <= upper[i]):
                continue
            if candidate == xi:
                return 0.0
            if probe is None:
                probe = x.copy()
            probe[i] = candidate
            lp, adjustment = evaluate(probe)
            slope = (lp + adjustment - adjusted_lp) / step
            if isfinite(slope):
                return slope
        return 0.0

    # These derivatives belong to the entire current state. A change in
    # any coordinate can change all partials of a coupled log-posterior.
    current_derivatives = {}

    if fix_param_set is not None:
        fix_param_set = {validate_parameter_index(i, p, 'fix_param_set') for i in fix_param_set}
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

    draws = np.zeros((n_samples, p))
    log_posteriors = np.zeros(n_samples)

    lp0, lp_adj0 = evaluate(x0)
    if not np.isfinite(lp0 + lp_adj0):
        raise ValueError('Starting point must have a finite adjusted log-posterior')

    num_tries = np.zeros(p)
    num_accepted = np.zeros(p)

    # do_resample = resample_k is not None

    if debug:
        from ray.experimental.tqdm_ray import tqdm
        pbar = tqdm(total=thinning * n_samples, position=position)

    # taus = [tau]

    for itr, (coord, innov, log_uniform) in enumerate(zip(rand_coord, rand_innovation, rand_log_uniform)):

        adapting = itr < n_warmup * thinning
        de_pool = None
        if diff_evolution_step_cadence and itr >= 100 and itr % diff_evolution_step_cadence == 0:
            # Only completed warm-up draws can enter the proposal pool.
            # Once warm-up ends, both slice limits stay fixed.
            de_pool = past_samples
            if de_pool is None:
                history_end = min(itr // thinning, n_warmup)
                de_pool = draws[history_end // 2:history_end]

        if de_pool is not None and len(de_pool) >= 2:
            # differential evolution step:
            rnd_draw_idx = rand.randint(len(de_pool), size=2)
            direction = de_pool[rnd_draw_idx[0]] - de_pool[rnd_draw_idx[1]]
            if fix_param_set:
                direction[list(fix_param_set)] = 0.0

            x1 = x0 + diff_evolution_scale * direction
            # Repeated history rows produce a valid self-proposal.
            if np.array_equal(x1, x0):
                accepted = 1
            elif np.all(np.isfinite(x1)) and (bounds is None or np.all((lower <= x1) & (x1 <= upper))):
                lp1, lp_adj1 = evaluate(x1)
                accepted = int(isfinite(lp1 + lp_adj1) and log_uniform < lp1 + lp_adj1 - (lp0 + lp_adj0))
                if accepted:
                    x0, lp0, lp_adj0 = x1, lp1, lp_adj1
                    current_derivatives.clear()
            else:
                accepted = 0
            if adapting:
                diff_evolution_scale *= (1 + .05 * (accepted - .234))

        else:
            # coordinate mala step
            if do_mala:
                if coord not in current_derivatives:
                    current_derivatives[coord] = deriv_lp(coord, x0, lp0 + lp_adj0)
                deriv0 = current_derivatives[coord]
            else:
                deriv0 = 0

            candidate = x0[coord] + (tau[coord] * deriv0 + np.sqrt(2 * tau[coord]) * innov)
            lp1, lp_adj1 = -np.inf, 0.0
            # An increment can round away at the scale of this coordinate.
            accepted = int(candidate == x0[coord])
            if not accepted and isfinite(candidate) and (bounds is None or lower[coord] <= candidate <= upper[coord]):
                x1 = x0.copy()
                x1[coord] = candidate
                lp1, lp_adj1 = evaluate(x1)

            if isfinite(lp1 + lp_adj1):

                if do_mala:
                    deriv1 = deriv_lp(coord, x1, lp1 + lp_adj1)

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
                    if do_mala:
                        current_derivatives.clear()
                        current_derivatives[coord] = deriv1

            if adapting:
                tau[coord] *= 1 + tau_adjust * (accepted - target_accept)

            num_tries[coord] += 1
            num_accepted[coord] += accepted

        if (itr + 1) % thinning == 0:
            draws[itr // thinning] = x0
            log_posteriors[itr // thinning] = lp0

        if debug:
            if time.time() - last_update_time > pbar_update_cadence:
                last_update_time = time.time()
                pbar.update(itr - pbar._x)
                pbar.set_description(
                    f'lp = {lp0:8.4e}, coord = {coord:3d}, tau = {tau[coord]:6.2e}, accept = {num_accepted[coord] / num_tries[coord]:6.3e}')

    return {
        'settings': {
            'seed': seed,
            'n_samples': n_samples,
            'n_warmup': n_warmup,
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
        containing randomly selected late warm-up draws from all chains.
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
         tau0=None, bounds=None
         ) -> MCMCResults:
    """Run Coordinate MALA MCMC across multiple parallel chains.

    This is the primary user-facing entry point for the coordinate-wise MALA
    sampler.  It:

    1. Formats and validates starting parameter vectors via
       ``format_starting_point_for_mcmc``, applying any bounded-to-unbounded
       transformations.
    2. Splits the total draw count into sub-chains of at most
       ``max_subchain_draws`` draws, with a separate boundary at the end of
       warm-up. Tau, DE scale, and the DE history pool are adapted only during
       warm-up and remain fixed throughout retained sampling.
    3. Dispatches each sub-chain to all chains in parallel via Ray.
    4. After each sub-chain merges results, updates the starting states and,
       during warm-up only, the DE past-sample pool. When ``debug=True`` it
       prints a convergence summary.
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
            Jacobian log-determinant of the map from unbounded x to the original model parameters.
        start_params_is_original_scale: When ``True`` (default), starting
            values are in the original (bounded) space and are transformed
            before sampling.
        transformations: Dict ``{param_index: TransformedParameter}`` for
            parameters reparameterized to an unbounded space.
        bounds: Optional pair of inclusive lower and upper vectors on the
            sampling scale, after any reparameterization.
        param_names: List of parameter name strings.  Auto-generated when ``None``.
        specification_name: Optional display label for the model run.
        n_chains: Number of parallel MCMC chains.
        n_samples: Total number of draws to collect per chain (including burn-in).
        frac_burnin: Fraction of the initially requested draws used for
            adaptation and discarded as burn-in (e.g. 0.25 → first 25%).
            Must be in [0, 1). Additional draws requested interactively use
            the frozen proposal and do not increase this burn-in count.
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
        diff_evolution_frac_burnin: Fraction of warm-up history to skip when
            building the DE pool (avoids including early warm-up draws).
            The pool is frozen when warm-up ends. Without warm-up, DE is
            disabled because no history pool is available.
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

    for name, value in [('n_samples', n_samples), ('max_subchain_draws', max_subchain_draws)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if not np.isfinite(frac_burnin) or not 0 <= frac_burnin < 1:
        raise ValueError('frac_burnin must be finite and in [0, 1)')
    if diff_evolution_step_cadence:
        if not np.isfinite(diff_evolution_frac_burnin) or not 0 <= diff_evolution_frac_burnin < 1:
            raise ValueError('diff_evolution_frac_burnin must be finite and in [0, 1)')
        if (isinstance(diff_evolution_max_draws, (bool, np.bool_))
                or not isinstance(diff_evolution_max_draws, (int, np.integer))
                or diff_evolution_max_draws <= 0):
            raise ValueError('diff_evolution_max_draws must be a positive integer')

    n_burnin = int(frac_burnin * n_samples)

    def split_draws(count):
        return [min(max_subchain_draws, count - start)
                for start in range(0, count, max_subchain_draws)]

    if debug:
        print("------------------------------------------------------")
        print("Beginning Coordinate Metropolis-adjusted Langevin MCMC")
        print("------------------------------------------------------")

    x0s_formatted, transformations, fix_params, fix_params_transformed, param_names, num_params, transformation_function = \
        format_starting_point_for_mcmc(start_params, start_params_is_original_scale, n_chains, transformations, fix_params, param_names,
                                       debug=debug)

    fix_param_idx = fix_params.keys() if fix_params else None
    bounds = _check_sampling_bounds(bounds, x0s_formatted, num_params)
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

    # No block may straddle the warm-up boundary. This also avoids empty
    # blocks when a requested draw count is exactly divisible by the limit.
    n_samples_split_list = split_draws(n_burnin) + split_draws(n_samples - n_burnin)
    n_completed = 0

    if tau0 is None:
        tau_start = None
    elif np.ndim(tau0) == 0:
        tau_start = np.full(num_params, tau0, dtype=float)
    else:
        tau_start = np.array(tau0, dtype=float)

    interrupted = False
    while True:

        for n_samples_bloc in n_samples_split_list:
            bloc += 1
            n_warmup_bloc = min(n_samples_bloc, max(n_burnin - n_completed, 0))

            # print(str({'bloc': bloc, 'n_samples': n_samples_bloc}))

            try:

                chain_args = [
                    (
                        log_posterior_id, log_posterior_jacobian_adjustment_id, start_params, seed_chain,
                        n_samples_bloc, tau_start, thinning, target_acceptance_rate,
                        do_cauchy,
                        pbar_update_cadence, do_mala, debug, tau_adjust, bloc,
                        fix_param_idx, pos, diff_evolution_step_cadence,
                        past_samples_id, diff_evolution_scale, n_warmup_bloc, bounds
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
                interrupted = True
                ray.shutdown()
                break
            except Exception as e:
                ray.shutdown()
                print("\n!!!" * 4)
                print(e.__traceback__)
                raise e

            outer_cnt += 1
            n_completed += n_samples_bloc

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

                    for k in ['n_samples', 'n_warmup']:
                        old['settings'][k] += new['settings'][k]

                    old['samples'] = np.vstack([old['samples'], new['samples']])

                    del new['samples']

            x0s_formatted = [C['samples'][-1] for C in chain_results_master]
            if n_warmup_bloc:
                tau_start = np.mean([c['tau'] for c in chain_results_master], axis=0)
                diff_evolution_scale = np.mean([c['diff_evolution_scale'] for c in chain_results_master])

                if diff_evolution_step_cadence:
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

        if interrupted:
            break

        if user_prompt_for_more_iters:

            message = get_diagnostic_update_message(
                chain_results_master, param_names, n_chains, time_start, time_start, transformation_function,
                fix_param_idx, thinning, callback_function=callback_function,
                title_string='Coordinate Metropolis Adjusted Langevin (CD-MALA)'
            )

            n_samples_new = user_prompt_for_more_iters_method(message, do_prompt=True, assert_even=True)

            if n_samples_new is None or n_samples_new == '' or n_samples_new <= 0:
                break

            n_samples_split_list = split_draws(n_samples_new)

            x0s_formatted = [C['samples'][-1] for C in chain_results_master]

        else:
            break

    if interrupted and (chain_results_master is None or n_completed <= n_burnin):
        raise RuntimeError('MALA interrupted before any post-warmup draws were collected')

    for c in chain_results_master:
        c['accepteds'] = np.mean(c['num_accepted'][c['num_tries'] > 0] / c['num_tries'][c['num_tries'] > 0])

    num_samps = len(chain_results_master[0]['samples'])

    cov_params_unbndd, mean_params_unbndd = convert_samples_to_original_space(
        chain_results_master, transformations, num_params, debug=debug, key='samples', window_start=n_burnin)

    options_dict = dict(
        method=CD_MALA_METHOD,
        seed=seed, do_adaptive=True, x0=start_params.copy(), n_chains=n_chains,
        n_samples=num_samps, n_burnin=n_burnin, thinning=thinning, frac_burnin=frac_burnin,
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
        bounds=bounds,
    )
    other_info = dict(
        tau_final=np.mean([c['tau'] for c in chain_results_master], axis=0),
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
        n_burnin=n_burnin, n_chains=n_chains, thinning=thinning, options=options_dict,
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
