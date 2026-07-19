"""Default tuning constants for the nonlinear least-squares (NLLS) solvers.

Two solver families share this module:

* The **trust-region NLLS** loop used by `nlls` / `NLLS` (see
  `kanly.regression.nonlinear_least_squares.optimize.nlls_minimize_internal`).
* The **elastic-net NLLS** coordinate-descent loop used by `nlls_en` /
  `NLLS_EN`. Constants prefixed with ``DEFAULT_NLLS_EN_`` only affect this
  second solver.

Most numeric tolerances default to ``sqrt(machine eps)`` (`EPS`), the standard
choice for first-order convergence checks in finite-precision arithmetic.
Override any of these at call time via the corresponding keyword argument on
the public API functions rather than mutating this module.
"""
from __future__ import absolute_import, print_function

import numpy

from kanly.regression.cov_types import HC0, HC1, NONROBUST, OLS_SMALL, BOOTSTRAP, CLUSTER

# ---------------------------------------------------------------------------
# Numeric primitives
# ---------------------------------------------------------------------------
# Square root of machine epsilon (~1.49e-8 on IEEE 754 double). Used as the
# default tolerance for x/f/gradient convergence checks; full machine eps is
# typically too tight given finite-difference Jacobian noise.
EPS = numpy.finfo(float).eps ** .5

# ---------------------------------------------------------------------------
# Inference defaults (covariance estimator + hypothesis-test level)
# ---------------------------------------------------------------------------
DEFAULT_NLLS_COV_TYPE = OLS_SMALL  # small-sample OLS-style sigma^2*(J'J)^-1
DEFAULT_NLLS_TEST_LEVEL = .05      # two-sided alpha used by .summary() tables

# Allow-list for `cov_type` arguments. Anything outside this list is rejected
# by the public NLLS entry points.
NLLS_COV_TYPES = [HC0, HC1, NONROBUST, OLS_SMALL, BOOTSTRAP, CLUSTER]

# ---------------------------------------------------------------------------
# Bootstrap covariance settings (when cov_type == BOOTSTRAP)
# ---------------------------------------------------------------------------
DEFAULT_NLLS_BOOTSTRAP_N_SAMPLES = 250        # number of resample refits
DEFAULT_NLLS_BOOTSTRAP_SEED = 0               # base seed for reproducibility
DEFAULT_NLLS_BOOTSTRAP_USE_CORRECTION = True  # small-sample bias correction

# ---------------------------------------------------------------------------
# Trust-region NLLS convergence tolerances
# ---------------------------------------------------------------------------
# Convergence is declared if any one of these tolerances is met.
DEFAULT_NLLS_XTOL = EPS  # relative parameter change ||dx|| / ||x||
DEFAULT_NLLS_GTOL = EPS  # projected gradient infinity norm
DEFAULT_NLLS_FTOL = EPS  # relative objective_function change |df| / |f|

# ---------------------------------------------------------------------------
# Trust-region NLLS iteration budget and step bookkeeping
# ---------------------------------------------------------------------------
DEFAULT_NLLS_DELTA = None                       # trust-region radius (None => auto)
DEFAULT_NLLS_DELTA_FLOOR = DEFAULT_NLLS_XTOL    # min radius before declaring stall
DEFAULT_NLLS_MAX_ITER = 100                     # max outer iterations
DEFAULT_NLLS_PROMPT_USER_FOR_MORE_ITERS = False # interactive "continue?" prompt

# Parameter scaling for the trust-region step. ``None`` keeps natural units;
# ``'jac'`` scales by the diagonal of J'J (Levenberg–Marquardt style).
DEFAULT_NLLS_X_SCALE = None  # None or 'jac'

# ---------------------------------------------------------------------------
# Trust-region radius update policy (rho = actual / predicted reduction)
# ---------------------------------------------------------------------------
DEFAULT_NLLS_RHO_QUAD_MODEL_ACCEPT = .75   # rho >= this => expand radius
DEFAULT_NLLS_RHO_QUAD_MODEL_REJECT = .25   # rho <= this => shrink radius
DEFAULT_NLLS_DELTA_INCREASE_FACTOR = 1.5   # multiplier when expanding
DEFAULT_NLLS_DELTA_DECREASE_FACTOR = 3     # divisor when shrinking
DEFAULT_NLLS_RHO_STEP_ACCEPT_FLOOR = .1    # min rho to keep the proposed step

# ---------------------------------------------------------------------------
# Bound handling (reflective trust region)
# ---------------------------------------------------------------------------
DEFAULT_NLLS_NUM_REFLECTIONS = 25     # max reflections off active bounds per step
DEFAULT_NLLS_REFLECTION_THETA = .995  # fraction of step kept before reflection

# ---------------------------------------------------------------------------
# Jacobian construction (trust-region NLLS)
# ---------------------------------------------------------------------------
DEFAULT_NLLS_DO_BROYDEN_JAC_UPDATE = False      # rank-1 Broyden updates between refits
DEFAULT_NLLS_BROYDEN_JAC_UPDATE_CADENCE = 3     # full Jacobian refit every N iters
DEFAULT_NLLS_DO_LINE_SEARCH = False             # backtracking line search after step
DEFAULT_NLLS_JAC_METHOD = 'fwd'                 # 'fwd' / 'mid' / 'analytic'
DEFAULT_NLLS_TRY_NEWTON_STEP = True             # try unconstrained Newton step first

# ---------------------------------------------------------------------------
# Elastic-net NLLS (`nlls_en` / `NLLS_EN`) — coordinate-descent solver
# ---------------------------------------------------------------------------
# Convergence tolerances (same semantics as the trust-region versions above)
DEFAULT_NLLS_EN_GTOL = EPS
DEFAULT_NLLS_EN_XTOL = EPS
DEFAULT_NLLS_EN_FTOL = EPS
DEFAULT_NLLS_EN_MAX_ITER = 100              # max outer coordinate-descent sweeps

# Step-size shrinkage inside a coordinate update when the proposed step fails
DEFAULT_NLLS_EN_NUM_SHRINKAGE = 4           # max shrinks before giving up on coord
DEFAULT_NLLS_EN_SHRINK_FACTOR = .33         # multiplicative shrink per attempt

DEFAULT_NLLS_EN_ACTIVE_SET = True           # restrict sweeps to nonzero coords once warm
DEFAULT_NLLS_EN_ALPHA = 1.0                 # overall penalty strength
DEFAULT_NLLS_EN_L1_RATIO = 0.0              # 1.0 = LASSO, 0.0 = ridge, in (0,1) = EN_sk
DEFAULT_NLLS_EN_NORMALIZE = False           # standardize features before fitting

# JIT-compile the analytic Jacobian via numba when available
DEFAULT_NLLS_DO_ANALYTIC_JAC_JIT = True

# --- Coordinate-selection strategies for elastic-net CD ---
NLLS_EN_GREEDY = 'greedy'   # pick coord with largest |gradient|
NLLS_EN_CYCLIC = 'cyclic'   # sweep coords in order (deterministic)
NLLS_EN_RANDOM = 'random'   # sweep coords in random order
EN_SELECTION_TYPES = (NLLS_EN_CYCLIC, NLLS_EN_RANDOM, NLLS_EN_GREEDY)
DEFAULT_NLLS_EN_SELECTION = NLLS_EN_CYCLIC

# ---------------------------------------------------------------------------
# Diagnostics / memory
# ---------------------------------------------------------------------------
DEFAULT_NLLS_KEEP_OPTIMIZATION_PATH = False  # store every accepted iterate
DEFAULT_NLLS_DENSE_THRESHOLD_MB = 1024       # switch sparse->dense Jacobian above this

# ---------------------------------------------------------------------------
# Jacobian construction (elastic-net NLLS)
# ---------------------------------------------------------------------------
NLLS_EN_JAC_METHOD_ANALYTIC = 'analytic'  # symbolic derivatives via the AD graph
NLLS_EN_JAC_METHOD_FWD = 'fwd'            # forward finite difference (cheap)
NLLS_EN_JAC_METHOD_MID = 'mid'            # centered finite difference (accurate)
NLLS_EN_JAC_METHODS = (NLLS_EN_JAC_METHOD_ANALYTIC, NLLS_EN_JAC_METHOD_FWD, NLLS_EN_JAC_METHOD_MID)

DEFAULT_NLLS_EN_JAC_METHOD = NLLS_EN_JAC_METHOD_FWD

# ---------------------------------------------------------------------------
# Penalty scaling (objective_function normalization)
# ---------------------------------------------------------------------------
# The NLLS objective_function is SSR/2; scaling penalties by nobs keeps the L2 penalty
# strength on the same units as the residual variance.
DEFAULT_NLLS_SCALE_L2_PENALTIES = True   # scales penalties by nobs since NLLS objective_function is SSR/2
# The elastic-net NLLS objective_function is SSR/(2n); when False, penalties are
# divided by n so the user-facing alpha keeps the same meaning across n.
DEFAULT_NLLS_EN_SCALE_PENALTIES = True   # if False, we divide penalties by nobs since NLLS_EN objective_function is SSR/(2*n)

# ---------------------------------------------------------------------------
# Optional 1-D line search along the coordinate-descent trajectory
# ---------------------------------------------------------------------------
# Parameters for looking on either side, 1d directionally
# of the current CD iterate, using the trajectory between the previous
# iter and this one. Useful when CD is slow-converging along a narrow valley.
DEFAULT_NLLS_EN_ONE_DIM_SEARCH_CADENCE = None   # None disables; otherwise run every N sweeps
DEFAULT_NLLS_EN_ONE_DIM_SEARCH_INIT_VAL = .01   # initial step size along the search dir
DEFAULT_NLLS_EN_ONE_DIM_SEARCH_MULTIPLIER = 5   # expansion factor between trial points
