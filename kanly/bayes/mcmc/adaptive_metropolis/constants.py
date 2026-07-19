"""
Default constants and tuning parameters for the Adaptive Metropolis–Hastings (AMH) sampler.

These values are used as keyword-argument defaults throughout
``adaptive_metropolis_mcmc.py`` and can be overridden by the caller.
"""
from __future__ import absolute_import, print_function

from numpy import inf

# String identifier used in MCMCResults.method and to distinguish AMH from MALA results.
METHOD = 'AMH'  # Adaptive Metropolis

# ── Chain topology ────────────────────────────────────────────────────────────
DEFAULT_AM_N_CHAINS = 4        # Number of parallel chains run.
DEFAULT_AM_N_BURNIN = 10_000   # Burn-in draws per chain (discarded from posterior summaries).
DEFAULT_AM_N_SAMPLES = 20_000  # Post-burn-in draws per chain stored for inference.
DEFAULT_AM_THINNING = 1        # Store every k-th draw; 1 = keep all.

# ── Proposal distribution ─────────────────────────────────────────────────────
# Draws are pre-generated in batches to amortize numpy call overhead.
DEFAULT_AM_DRAW_SIZE = 2_000
# Degrees of freedom of the proposal distribution; inf = multivariate Gaussian.
DEFAULT_AM_PROPOSAL_DF = inf

# ── Adaptive scalar tuning ────────────────────────────────────────────────────
# Target Metropolis acceptance rate (Gelman et al. optimal for high-dimensional Gaussians).
DEFAULT_AM_TARGET_ACCEPTANCE_RATE = .234
DEFAULT_AM_DO_ADAPTIVE = True  # Toggle adaptation on/off.
# The scaler is multiplied by (1 + rate * (accepted - target) / itr^power) each step.
DEFAULT_AM_SCALER_ADJUST_RATE = .5           # Learning rate coefficient.
DEFAULT_AM_SCALER_ADJUST_DENOM_POWER = .35   # Exponent for the diminishing-adaptation denominator.
DEFAULT_AM_MIN_SCALER = 1e-12  # Hard lower bound on the adaptive scalar (prevents step collapse).
DEFAULT_AM_MAX_SCALER = inf    # Hard upper bound (inf = unconstrained).
# Whether to freeze the scalar and covariance after burn-in; keeps the sample phase stationary.
DEFAULT_AM_STOP_ADAPTATION_AFTER_BURNIN = False
# Optional per-step random jitter applied to the scalar; (lo, hi) or None to disable.
DEFAULT_AM_SCALAR_JITTER_BOUNDS = None

# ── Parallelism ───────────────────────────────────────────────────────────────
DEFAULT_AM_MAX_PROCESSES = 8       # Ray worker processes.
DEFAULT_AM_DO_PARALLEL = True      # Dispatch chains via Ray when True; serial otherwise.
# Seconds between tqdm progress-bar refreshes.
DEFAULT_AM_PBAR_UPDATE_CADENCE = .7
DEFAULT_AM_USER_PROMPT_FOR_MORE_ITERS = False  # Ask user to continue if not converged.

# ── Sub-chain chunking ────────────────────────────────────────────────────────
# The full draw count is split into sub-chains so that the proposal covariance
# can be updated between runs and Ray tasks stay short.
DEFAULT_AM_MAX_SUBCHAIN_DRAWS_SAMPlE = 25_000  # Max draws per sample sub-chain.
DEFAULT_AM_MAX_SUBCHAIN_DRAWS_BURNIN = 5_000   # Max draws per burn-in sub-chain.

# ── Proposal covariance initialization ───────────────────────────────────────
# None → auto-compute from n_burnin // 100 (with a floor of 50 draws).
DEFAULT_AM_STEP_COV_INITIAL_SAMPLES = None
# When True, normalizes the proposal covariance determinant to 1; not currently used.
DEFAULT_AM_NORMALIZE_STEP_COV = False

# ── Differential Evolution Monte Carlo (DE-MC) ───────────────────────────────
# DE-MC augments normal proposals with jumps drawn as (x_a - x_b) from past samples,
# which can escape multimodal regions more efficiently than pure Gaussian proposals.
DEFAULT_AM_DO_DIFF_EVOLUTION_MC = True
# Fraction of burn-in draws used to build the initial DE sample pool.
DEFAULT_AM_DIFF_EVOLUTION_FRAC_BURNIN = .25
# Maximum pool size; oldest draws are dropped when exceeded.
DEFAULT_AM_DIFF_EVOLUTION_MAX_DRAWS = 10_000
# Mixture weight in (0,1): fraction of the proposal coming from DE jumps vs. normal draws.
DEFAULT_AM_DIFF_EVOLUTION_WEIGHT = .95
# How often (in iterations) to force a pure DE jump (None = never force).
DEFAULT_AM_DIFF_EVOLUTION_JUMP_CADENCE = None

DEFAULT_AM_STEP_COV_ADJUST_RATE = .8  # TODO DELETE