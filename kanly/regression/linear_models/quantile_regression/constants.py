"""Tuning defaults and flag constants for the IRLS quantile regression solver.

Every configurable knob used by ``qr_internal``, ``model``, and the result
classes lives here so that callers can override individual settings without
importing internals directly.
"""
from __future__ import absolute_import, print_function

# --- Inference ---
# Use the t-distribution (rather than the normal) for hypothesis tests by default.
DEFAULT_QR_USE_T = True
# Two-sided significance level for confidence intervals and p-values.
DEFAULT_QR_TEST_LEVEL = .05

# --- IRLS solver convergence ---
# Maximum number of IRLS iterations before declaring non-convergence.
DEFAULT_QR_MAX_ITER = 200
# Stop when the max elementwise relative/absolute change in β is below this.
DEFAULT_QR_X_TOL = 1e-6
# Stop when the relative change in the objective_function (check-function cost) is below this.
DEFAULT_QR_F_TOL = 1e-8

# --- Loss smoothing ---
# Minimum residual magnitude used in the IRLS weight denominator to avoid 1/0.
DEFAULT_QR_MIN_RESID_CLIP = 1e-10
# Half-width k of the smooth quadratic region in the surrogate loss.
# Smaller values give a closer approximation to the true check function.
DEFAULT_QR_SMOOTHING_K = 1e-5

# --- Line search ---
# Whether to run a grid line search at each IRLS step to ensure cost reduction.
DEFAULT_QR_LINE_SEARCH = True
# Step-size grid evaluated during line search (convex combinations of old and new β).
DEFAULT_QR_STEP_GRID = [.05, .25, .5, 1, 1.5, 3, 5, 10, 20]

# --- Covariance type strings ---
# Robust (heteroscedastic) sandwich covariance: (X'X)^{-1} X'DX (X'X)^{-1} / f0^2.
QR_COV_TYPE_ROBUST = 'ROBUST'
# IID (homoscedastic) covariance: tau(1-tau)/f0^2 * (X'X)^{-1}.
QR_COV_TYPE_IID = 'IID'
# Bootstrap covariance (Bayesian or block bootstrap; computed in model.py).
QR_COV_TYPE_BOOTSTRAP = 'BOOTSTRAP'
# Default covariance estimator applied when cov_type is not specified by the caller.
DEFAULT_QR_COV_TYPE = QR_COV_TYPE_IID
# Complete list of accepted cov_type strings (validated at fit time).
QR_COV_TYPES = [QR_COV_TYPE_BOOTSTRAP, QR_COV_TYPE_ROBUST, QR_COV_TYPE_IID]

# --- Bootstrap ---
# Default number of bootstrap repetitions for BOOTSTRAP covariance estimation.
DEFAULT_QR_BOOTSTRAP_N_SAMPLES = 100

# --- Loss function ---
# Default surrogate loss; one of 'huber', 'softl1', 'smoothcup'.
DEFAULT_QR_LOSS_FUNCTION = 'huber'

# --- Instrumental variables (control-function approach) ---
# Whether to include IV residuals as additional regressors (control function).
DEFAULT_QR_RESIDUAL_INCLUSION = True  # Still experimenting on this
# Polynomial order of residual powers to include in the control-function augmentation.
DEFAULT_QR_RESIDUAL_INCLUSION_ORDER = 1

# --- Design matrix ---
# Whether to column-scale X before solving (not yet wired into qr_internal).
DEFAULT_QR_SCALE_DESIGN_MATRIX = True

# --- Sparsity / density threshold ---
# If the design matrix fits within this many MB after densifying, convert it
# to a dense array before the IRLS loop to exploit BLAS routines.
DEFAULT_QR_DENSE_THRESHOLD_MB = 1_024
