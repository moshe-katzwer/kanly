"""Default settings and symbolic names for generalized linear model fitting."""
from __future__ import absolute_import, print_function

from kanly.regression.generalized_linear_models.families import Gaussian
from kanly.regression.cov_types import BOOTSTRAP, HC1, NONROBUST


# Covariance estimators supported by the high-level GLM API.
GLM_COV_TYPES = [BOOTSTRAP, HC1, NONROBUST]

# Gaussian is the most permissive default because it accepts real-valued
# outcomes and uses the identity link by default.
DEFAULT_GLM_FAMILY = Gaussian

# Iterative optimizer stopping controls.
DEFAULT_GLM_TOL = 1.0e-8
DEFAULT_GLM_MAX_ITER = 100

# Inference defaults used when building RegressionResults-style summaries.
DEFAULT_GLM_TEST_LEVEL = 0.05
DEFAULT_GLM_USE_T = True

# Elastic-net defaults. alpha=0 disables regularization, and l1_ratio=0 means
# any nonzero regularization defaults to pure ridge unless the caller opts in.
DEFAULT_GLM_ALPHA = 0.0
DEFAULT_GLM_L1_RATIO = 0.0

# Coordinate-descent / fallback line-search controls. The intercept shrinkage
# gradually removes an intercept contribution when fit_intercept is disabled.
SHRINK_INTERCEPT = .9
LINE_SEARCH_SHRINK = .5
MAX_LINE_SEARCH = 10

# Model-based covariance is the default because it is cheap and matches the GLM
# likelihood assumptions when those assumptions are trusted.
DEFAULT_GLM_COV_TYPE = NONROBUST

# IV-GLM defaults: residual inclusion augments nonlinear IV fits with first-stage
# residual sparse_terms, with order 1 meaning only the raw residuals are included.
DEFAULT_GLM_RESIDUAL_INCLUSION = True
DEFAULT_GLM_RESIDUAL_INCLUSION_ORDER = 1

# Optimizer method labels. IRLS is the standard GLM routine; coordinate descent
# is used for elastic-net penalties; the one-iteration variant warms up IRLS.
METHOD_IRLS = 'IRLS'
METHOD_COORD_DESC = 'COORDINATE_DESCENT'
METHOD_COORD_DESCENT_1_ITER = 'COORDINATE_DESCENT_1_ITER'

GLM_METHODS = [METHOD_IRLS, METHOD_COORD_DESCENT_1_ITER, METHOD_COORD_DESC]

# Formula/design safety defaults.
DEFAULT_GLM_FORCE_IV_PROJECTION = False
DEFAULT_GLM_CHECK_CONSTANT_COLS = True

# Long fits can optionally ask the user whether to continue after max_iter.
DEFAULT_GLM_PROMPT_USER_FOR_MORE_ITERS = False

