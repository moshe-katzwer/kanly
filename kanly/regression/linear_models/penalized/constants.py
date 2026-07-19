from __future__ import absolute_import, print_function

"""Default settings and display labels for penalized linear models."""

# Baseline elastic-net penalty and convergence controls used by the public APIs.
DEFAULT_EN_ALPHA = .01
DEFAULT_EN_L1_RATIO = 1.0
DEFAULT_EN_X_TOL = 1e-6
DEFAULT_EN_F_TOL = 1e-10
DEFAULT_EN_G_TOL = 1e-4
DEFAULT_EN_MAX_ITER = 500

# Default model-shape and solver behavior.  When ``normalize=True`` (the
# default), penalties are scaled by each predictor's standard deviation — the
# same scale-invariance goal as sklearn's ``StandardScaler`` workflow.  See
# ``sparse_elastic_net_internal.OLD_SKLEARN_NORMALIZATION`` for the legacy
# demeaned L2-norm convention.
DEFAULT_EN_FIT_INTERCEPT = True
DEFAULT_EN_NORMALIZE = True
DEFAULT_EN_POSITIVE = False
DEFAULT_EN_APPLY_SCALING = False
DEFAULT_EN_ACTIVE_SET = True
DEFAULT_EN_PROMPT_USER_FOR_MORE_ITERS = False
DEFAULT_EN_RELAXATION_PARAMETER = None

# Coordinate ordering modes for the coordinate-descent loop.
EN_GREEDY = 'greedy'
EN_CYCLIC = 'cyclic'
EN_RANDOM = 'random'
EN_SELECTION_TYPES = (EN_CYCLIC, EN_RANDOM, EN_GREEDY)
DEFAULT_EN_SELECTION = EN_RANDOM

# Internal method labels used when routing special ridge/least-squares cases.
EN_COORDINATE_DESCENT = 'COORDINATE_DESCENT'
EN_LEAST_SQUARES = 'LEAST_SQUARES'
DEFAULT_EN_RIDGE_METHOD = EN_COORDINATE_DESCENT

# Summary labels for unweighted fits.
ELASTIC_NET = 'ELASTIC NET'
LASSO = 'LASSO'
OLS = 'OLS (CD)'
RIDGE = 'RIDGE (CD)'

# Summary labels for weighted fits.
W_ELASTIC_NET = 'WTD ELASTIC NET'
W_LASSO = 'WTD LASSO'
WLS = 'WLS (CD)'
W_RIDGE = 'WTD RIDGE (CD)'

# Parameters for looking on either side, one-dimensionally, of the current CD
# iterate using the trajectory between the previous iterate and this one.
DEFAULT_EN_ONE_DIM_SEARCH_CADENCE = 4
DEFAULT_EN_ONE_DIM_SEARCH_INIT_VAL = .1
DEFAULT_EN_ONE_DIM_SEARCH_MULTIPLIER = 10
