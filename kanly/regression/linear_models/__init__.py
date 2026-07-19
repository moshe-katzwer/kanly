"""kanly linear models — OLS, WLS, IV, FGLS, absorbed FE, SURE, sparse and fast paths.

This package provides the primary public API (``SparseLinearModel``) and all
supporting utilities for fitting linear regression models:

- ``model.py``              — ``SparseLinearModel``: formula and matrix APIs,
                              FGLS, ridge, SURE, multi-outcome, fast (lsmr) path.
- ``regression_results.py`` — ``SparseLinearRegressionResults``, ``InstrumentInfo``,
                              ``AbsorbInfo2``: result summaries, tests, diagnostics.
- ``lm_internal.py``        — Core numerical pipeline: absorb → IV → solve → unscale.
- ``variance_covariance2.py``— Covariance estimators: HC, HAC, cluster, multi-way.
- ``sparse_iv_first_stage2.py``— IV first stage: project endogenous regressors
                                 onto instruments.
- ``fast_lm_internal.py``   — Sparse iterative LS via ``scipy.sparse.linalg.lsmr``
                               (no inference, no matrix inverse).
- ``linear_model_2_quadratic_form.py`` — SSR quadratic form and Gaussian LLF
                                         callable.
- ``constants.py``          — All tuning defaults and flag constants.
- ``exceptions.py``         — Custom exception classes.
- ``shapley.py``            — Shapley R² decomposition across regressors.
- ``permutation_test.py``   — Randomisation inference via treatment permutation.
- ``two_way_cluster.py``    — Two-way clustered SEs (Cameron–Gelbach–Miller).
"""
