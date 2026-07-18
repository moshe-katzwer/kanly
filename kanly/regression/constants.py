"""Default inference settings shared across all kanly regression models.

These constants supply fallback values for hypothesis-test level and the
choice of t-distribution versus normal-distribution critical values whenever
callers do not override them explicitly.
"""

from __future__ import absolute_import, print_function

# Two-sided significance level used for confidence intervals, p-value stars,
# and Wald/F-test critical values when the caller does not specify one.
DEFAULT_TEST_LEVEL = .05

# When True, inference uses the t-distribution with finite df_resid degrees of
# freedom; when False the standard normal is used (appropriate for large
# samples or models that report asymptotic SEs).
DEFAULT_USE_T = True
