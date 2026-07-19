"""Default tuning constants for kanly linear regression models.

This module is the single source of truth for every flag and default
value that controls estimation and inference behaviour across
``SparseLinearModel``, ``lm_internal``, ``variance_covariance2``, and
``sparse_iv_first_stage2``.  Importers should reference these names
rather than hard-coding scalar values so that package-wide defaults can
be changed in one place.
"""

from __future__ import absolute_import, print_function

from kanly.regression.cov_types import OLS_SMALL

# ---------------------------------------------------------------------------
# FGLS (Feasible Generalized Least Squares) iteration controls
# ---------------------------------------------------------------------------

# Key name used to look up the maximum iteration count in fgls_kwds dicts.
FGLS_MAX_ITER_KEY = 'maxiter'
# Key name used to look up the convergence tolerance in fgls_kwds dicts.
FGLS_TOL_KEY = 'tol'
# Default maximum number of FGLS re-weighting iterations.
FGLS_MAX_ITER_DEFAULT = 10
# Default convergence tolerance (max abs change in beta across iterations).
FGLS_TOL_DEFAULT = 1e-6

# ---------------------------------------------------------------------------
# Inference defaults
# ---------------------------------------------------------------------------

# Two-sided significance level for confidence intervals and Wald/F tests.
DEFAULT_LM_TEST_LEVEL = .05
# Use t-distribution (True) vs standard normal (False) for inference.
DEFAULT_LM_USE_T = True

# Default number of bootstrap resamples when cov_type='bootstrap'.
DEFAULT_LM_BOOTSTRAP_N_SAMPLES = 100

# Default covariance type: OLS_SMALL applies the n/(n-k) finite-sample
# correction to the sandwich estimator.
DEFAULT_LM_COV_TYPE = OLS_SMALL

# Sentinel weight name used when no weights are present in a formula.
NULL_WEIGHTS_NAME = '-'

# ---------------------------------------------------------------------------
# Formula / model-building flags
# ---------------------------------------------------------------------------

# If True, test the formula on a small dummy DataFrame before parsing the
# full data; helps surface formula errors cheaply.
DEFAULT_LM_TEST_FORMULA_ON_DUMMY = False

# If True, force every regressor column through the IV projection step
# even when it is classified as exogenous.
DEFAULT_LM_FORCE_IV_PROJECTION = False

# If True, check for and drop constant columns in the design matrix
# after formula evaluation.
DEFAULT_LM_CHECK_CONST_COLS = True

# If True, scale each regressor column to unit L2 norm before solving,
# then unscale parameters and covariance afterwards.  Improves numerical
# stability for poorly-conditioned design matrices.
DEFAULT_LM_SCALE_DESIGN_MATRIX = True

# ---------------------------------------------------------------------------
# Eigenvalue / condition-number computation flags
# ---------------------------------------------------------------------------

# None means "compute automatically when number of regressors < threshold".
DEFAULT_LM_COMPUTE_EIGENVALUES = None
# Whether to compute eigenvalues of the instrument Gram matrix.
DEFAULT_LM_COMPUTE_EIGENVALUES_INSTRUMENTS = False
# Auto-compute eigenvalues only when the design has fewer columns than this.
DEFAULT_LM_COMPUTE_EIGENVALUES_UNDER_MAX_DIM = 200

# ---------------------------------------------------------------------------
# Matrix inversion
# ---------------------------------------------------------------------------

# None lets linalg_utils choose between scipy sparse and numpy dense
# inversion based on matrix size and sparsity.
DEFAULT_LM_INVERSE_METHOD = None
