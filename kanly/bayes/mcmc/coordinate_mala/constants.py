"""
Default constants and tuning parameters for the Coordinate MALA (CD-MALA) sampler.

These values are keyword-argument defaults in ``coordinate_mala_mcmc.py``
and can all be overridden by the caller.
"""
from __future__ import absolute_import, print_function

# String identifier stored in MCMCResults.method to distinguish CD-MALA from AMH.
METHOD = 'CD-MALA'  # Coordinate MALA

# ── Chain topology ────────────────────────────────────────────────────────────
DEFAULT_COORD_MALA_N_CHAINS = 4         # Number of parallel chains.
DEFAULT_COORD_MALA_N_SAMPLES = 20_000   # Total draws per chain (burn-in included).
# Fraction of total draws treated as burn-in; first (frac_burnin * n_samples) draws
# are discarded from posterior summaries in MCMCResults.
DEFAULT_COORD_MALA_FRAC_BURNIN = .25
DEFAULT_COORD_MALA_THINNING = 1         # Store every k-th draw; 1 = keep all.
# Maximum draws per sub-chain; total draws are split into batches of this size
# so that tau and the DE pool can be updated between sub-chains.
DEFAULT_COORD_MALA_MAX_SUBCHAIN_DRAWS = 5_000

# ── Proposal and adaptation ───────────────────────────────────────────────────
# Target per-coordinate acceptance rate; 0.574 is near the optimal for the
# Langevin diffusion on a Gaussian target.
DEFAULT_COORD_MALA_TARGET_ACCEPTANCE_RATE = .57
# Whether to use the Langevin gradient drift in proposals.  False = plain
# coordinate random-walk Metropolis (ignores gradient information).
DEFAULT_COORD_MALA_DO_MALA = True
# Whether to use Cauchy (heavy-tailed) innovations instead of Gaussian.
# Cauchy proposals can improve mixing when the target has heavy tails.
DEFAULT_COORD_MALA_DO_CAUCHY = False
# Learning rate for the per-coordinate tau adaptation rule:
#   tau[coord] *= 1 + tau_adjust * (accepted - target_accept)
# Larger values adapt faster but may overshoot.
DEFAULT_COORD_MALA_TAU_ADJUST = .25

# ── Output and display ────────────────────────────────────────────────────────
DEFAULT_COORD_MALA_DEBUG = False           # Whether to show progress bars and print diagnostics.
DEFAULT_COORD_MALA_P_BAR_UPDATE_CADENCE = .5  # Seconds between tqdm bar refreshes.
DEFAULT_COORD_MALA_USER_PROMPT_FOR_MORE_ITERS = False  # Ask user to continue sampling.

# ── Differential Evolution (DE) jumps ────────────────────────────────────────
# Every `diff_evolution_step_cadence` iterations, perform a full-dimensional
# DE jump (difference of two past samples) instead of a coordinate MALA step.
# This helps escape correlated or multimodal regions that coordinate moves miss.
DEFAULT_COORD_MALA_DIFF_EVOLUTION_STEP_CADENCE = 20
# Fraction of each chain's history to skip when building the DE pool;
# avoids using very early warm-up draws that may be far from the mode.
DEFAULT_COORD_MALA_DIFF_EVOLUTION_FRAC_BURNIN = .5
# Maximum size of the DE past-sample pool.
DEFAULT_COORD_MALA_DIFF_EVOLUTION_MAX_DRAWS = 2_500
