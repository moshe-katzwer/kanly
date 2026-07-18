"""kanly.regression – linear and generalised linear regression models.

This package provides:
  - Abstract base classes for model construction and results (``ModelBase``,
    ``LinearModelBase``, ``RegressionResultsBase``).
  - Covariance estimation utilities: HC-robust, HAC, cluster-robust, and
    bootstrap sandwich estimators (``sandwich_tools``, ``cov_types``).
  - Fixed-effects absorption via within-group de-meaning (``absorb_tools``).
  - Diagnostic plotting for fitted-value and residual analysis
    (``plot_diagnostics``).
  - Sub-packages for penalised linear models, generalised linear models
    (GLM), and nonlinear least squares (NLLS).
"""
