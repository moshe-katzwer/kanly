"""Tuning defaults and flag constants for the RLM (M-estimation) IRLS solver.

All configurable knobs used by ``rlm_internal``, ``model``, and
``regression_results`` live here so that callers can override individual
settings without importing internals directly.
"""
from __future__ import absolute_import, print_function

from kanly.regression.linear_models.robust.variance_covariance import H1

# --- IRLS solver convergence ---
# Stop when the max elementwise relative/absolute change in β falls below this.
DEFAULT_RLM_X_TOL = 1e-6
# Maximum number of IRLS iterations before declaring non-convergence.
DEFAULT_RLM_MAX_ITER = 50

# --- Default norm (M) function ---
# String key resolved by get_norm(); 'HuberT' with c=1.345 is the standard choice.
DEFAULT_RLM_M = 'HuberT'

# --- Covariance and inference defaults ---
# H1 is the most commonly used M-estimator sandwich variant (Huber 1981).
DEFAULT_RLM_COV_TYPE = H1
# Two-sided significance level for confidence intervals and p-values.
DEFAULT_RLM_TEST_LEVEL = .05
# Use t-distribution (rather than normal) for hypothesis tests.
DEFAULT_RLM_USE_T = True

# --- Bootstrap ---
# Default number of bootstrap repetitions for BOOTSTRAP covariance estimation.
DEFAULT_RLM_BOOTSTRAP_N_SAMPLES = 100

# --- IV / control-function (reserved; not yet wired into RLM) ---
DEFAULT_RLM_RESIDUAL_INCLUSION = True
