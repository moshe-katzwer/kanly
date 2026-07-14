[<p align="center"><img src="images/kanly_logo_white_border.svg" width="300"></p>](images/kanly_logo.svg)


# kanly

**Author:** Moshe Katzwer ([`richard.katzwer@gmail.com`](richard.katzwer@gmail.com))

**kanly** is a Python toolkit for statistical modeling with an emphasis on **sparse matrix–first** implementations. It was built initially for **large sparse OLS and instrumental-variables regression**—designs with **tens of millions of rows** and **hundreds of predictors**—using one formula-driven sparse design pipeline. The package has since grown to cover GLM, robust and quantile linear models, nonlinear least squares, GMM, time series (SARIMAX), nonparametrics, **extensive bootstrap inference** (Bayesian bootstrap, classical bootstrap, block/cluster resampling), and **Bayesian** posterior sampling, while keeping the same Patsy-like formula interface.

Typical applications include:

- **Large sparse linear models** — OLS, WLS, absorbed fixed effects, ridge, FGLS, GLSAR (feasible GLS with AR errors; Prais–Winsten by default via `glsar`), multiple outcomes, and a fast coefficient-only path (`lm_fast`) for huge designs. 
  - On multi-million-row models with many fixed effects, kanly is often **much faster than statsmodels** while returning the same coefficients and inference (see [Performance vs statsmodels](#performance-vs-statsmodels)).
- **Instrumental variables** — 2SLS / W2SLS in linear models; IV GLM with optional **residual inclusion** (control-function approach, required for consistent nonlinear IV); control-function quantile regression; linear and nonlinear **GMM** (including MLE-as-GMM).  IV estimators have a patsy-like interface, same as vanilla OLS/WLS models.
- **Nonlinear least squares (NLLS)** — Trust-region fitting from formulas with analytic or numeric Jacobians, weights, bounds, robust/quantile-style losses, and elastic-net variants.  
   - Uses sparse Jacobians for perfomance for non-linear models with fixed effects.  
   - Includes a custom patsy like syntax for formula-defined NLLS problems, e.g. `[y] ~ np.exp({a} + {b}*[x])` for data `x,y` and parameters `a,b`.
- **Bayesian modeling** — A full Bayesian estimation package, including 
   - A custom, high-performance, gradient-free MCMC sampling algorithm (a mixture of [Metropolis-Adjusted Langevin Algorithm](https://m-clark.github.io/docs/ld_mcmc/#metropolis-adjusted-langevin-algorithm), [Adaptive Metropolis](https://m-clark.github.io/docs/ld_mcmc/#adaptive-metropolis), and [Differential Evolution MCMC](https://m-clark.github.io/docs/ld_mcmc/#differential-evolution-markov-chain).  
   - A custom modeling syntax for defining large models.  It is analogous to [STAN](https://mc-stan.org/docs/reference-manual/mcmc.html), but generates numba-optimized python code.  (See [`DataModel`](#datamodel-data-block-model-block-then-mcmc) class documentation.)
   - Convenience wrappers for Bayesian Linear Regression, Bayesian GLM, etc..
- **Generalized linear models (GLM)** — Binomial, Poisson, Gaussian, Gamma, inverse Gaussian, negative binomial; IRLS with optional elastic-net regularization; **generalized additive models (GAM)** via ``gam`` (penalized B-spline smooths — see [GLM README](kanly/regression/generalized_linear_models/README.md#generalized-additive-models-gam)); IV-style formulas with residual inclusion; bootstrap and robust covariances.
- **Robust and quantile regression** — M-estimation (IRLS) and quantile regression (IRLS with smooth check losses) at large scale.
- **Penalized linear models** — Elastic net / LASSO / ridge via coordinate descent with optional OLS refit on selected support.
- **Bootstrap inference** — Built-in resampling for covariance and tests: **Bayesian bootstrap** (Dirichlet weights), **classical bootstrap** (with-replacement row or cluster resampling), and **block / cluster bootstrap** when observations are correlated within groups. Available on `lm`, `glm`, `qr`, `nlls`, GMM, and related fits via `cov_type='bootstrap'`; optional **Ray** parallelism for large `n_samples`; **joint covariance** across multiple bootstrapped models (`get_joint_bootstrapped_distribution`). See [`kanly/bootstrap/README.md`](kanly/bootstrap/README.md).
- **Time series** — `acf` / `pacf`, `simulate_sarima`, and SARIMAX-style fitting with exogenous regressors (see [`kanly/time_series/README.md`](kanly/time_series/README.md)).
- **Nonparametrics** — KDE, LOWESS, STL / MSTL, Gaussian kernel smoothing, and piecewise splines / interpolation.
- **Densities and classical tests** — Numba-friendly PDFs / log-PDFs (`kanly.stats.distributions`) and helpers for [`Wald`](https://en.wikipedia.org/wiki/Wald_test), [`delta-method`](https://en.wikipedia.org/wiki/Delta_method), and [`Fieller`](https://en.wikipedia.org/wiki/Fieller%27s_theorem) inference (`StatisticalTests` in `kanly.stats.statistical_tests`; `kanly.api` re-exports the distribution functions but not the test class).

For a runnable end-to-end tour of some of these workflows, start with the notebook
[`example_quick_start.ipynb`](example_quick_start.ipynb) at the repository root (OLS/IV, GLM-IV, NLLS, LASSO, Bayesian MCMC, linear block bootstrap, KDE/LOWESS).

Most user-facing **model fitting** APIs are re-exported from **`kanly.api`**. 

Some utilities (for example `StatisticalTests`) are imported from their submodules — see [`kanly/stats/README.md`](kanly/stats/README.md).

---

## `kanly` Table of Contents

### This README

- [Installation](#installation)
  - [Recommended: editable install in a virtual environment](#recommended-editable-install-in-a-virtual-environment)
  - [Notes](#notes)
  - [Python version](#python-version)
- [Navigating the repository](#navigating-the-repository)
- [User guide overview](#user-guide-overview)
- [`kanly.formula`](#kanlyformula)
  - [What this subpackage does](#what-this-subpackage-does)
  - [Comparison: SciPy / statsmodels / scikit-learn](#comparison-scipy-statsmodels-scikit-learn)
  - [Core behaviour](#core-behaviour)
  - [Core entry points](#core-entry-points)
  - [Formula DSL (common)](#formula-dsl-common)
  - [Minimal usage](#minimal-usage)
  - [Examples in this repo](#examples-in-this-repo)
- [`kanly.regression`](#kanlyregression)
  - [What this subpackage does](#what-this-subpackage-does-1)
  - [Comparison: SciPy / statsmodels / scikit-learn](#comparison-scipy-statsmodels-scikit-learn-1)
    - [Syntax comparisons](#syntax-comparisons)
  - [`linear_models/`](#linear_models)
    - [Features](#features)
    - [Quick start (OLS)](#quick-start-ols)
    - [Formula syntax](#formula-syntax)
    - [WLS, fixed effects, and IV](#wls-fixed-effects-and-iv)
    - [Cluster-robust, HC, HAC, and bootstrap](#cluster-robust-hc-hac-and-bootstrap)
    - [FGLS and ridge](#fgls-and-ridge)
    - [GLSAR (AR errors)](#glsar-ar-errors)
    - [SURE](#sure)
    - [Large sparse / fast path](#large-sparse-fast-path)
    - [Performance vs statsmodels](#performance-vs-statsmodels)
    - [Matrix API](#matrix-api)
    - [Result objects](#result-objects)
    - [Examples in this repo (`examples/regression/linear_models/`)](#examples-in-this-repo-examplesregressionlinear_models)
  - [`linear_models/robust/`](#linear_modelsrobust)
    - [Quick start](#quick-start)
    - [Norm functions (`M=`)](#norm-functions-m)
    - [Covariance](#covariance)
    - [Limitations](#limitations)
    - [Examples in this repo](#examples-in-this-repo-1)
  - [`linear_models/quantile_regression/`](#linear_modelsquantile_regression)
    - [Quick start](#quick-start-1)
    - [Multiple quantiles](#multiple-quantiles)
    - [Covariance](#covariance-1)
    - [IV (experimental)](#iv-experimental)
    - [Examples in this repo](#examples-in-this-repo-2)
  - [`linear_models/penalized/`](#linear_modelspenalized)
    - [Formula API](#formula-api)
    - [Post-selection OLS refit](#post-selection-ols-refit)
    - [Examples in this repo](#examples-in-this-repo-3)
  - [`generalized_linear_models/`](#generalized_linear_models)
    - [Basic usage](#basic-usage)
    - [Families (string names)](#families-string-names)
    - [Links (examples)](#links-examples)
    - [Weights and covariance](#weights-and-covariance)
    - [Regularization](#regularization)
    - [IV GLM and residual inclusion](#iv-glm-and-residual-inclusion)
    - [Examples in this repo](#examples-in-this-repo-4)
  - [`generalized_method_of_moments/`](#generalized_method_of_moments)
    - [Entry points](#entry-points)
    - [Linear IV-GMM](#linear-iv-gmm)
    - [Methods](#methods)
    - [Examples in this repo (`examples/generalized_method_of_moments/`)](#examples-in-this-repo-examplesgeneralized_method_of_moments)
  - [`nonlinear_least_squares/`](#nonlinear_least_squares)
    - [Basic usage](#basic-usage-1)
    - [Conventions](#conventions)
    - [Jacobian modes](#jacobian-modes)
    - [Elastic-net NLLS](#elastic-net-nlls)
    - [Examples in this repo (`examples/regression/nonlinear_least_squares/`)](#examples-in-this-repo-examplesregressionnonlinear_least_squares)
  - [`partial_least_squares/`](#partial_least_squares)
- [`kanly.time_series`](#kanlytime_series)
  - [What this subpackage does](#what-this-subpackage-does-2)
  - [Comparison: SciPy / statsmodels / scikit-learn](#comparison-scipy-statsmodels-scikit-learn-2)
  - [SARIMAX quick start](#sarimax-quick-start)
  - [Formula interface](#formula-interface)
  - [Simulation](#simulation)
  - [ACF / PACF](#acf-pacf)
  - [Examples in this repo](#examples-in-this-repo-5)
- [`kanly.bayes`](#kanlybayes)
  - [What this subpackage does](#what-this-subpackage-does-3)
  - [Comparison: SciPy / statsmodels / scikit-learn](#comparison-scipy-statsmodels-scikit-learn-3)
  - [Prefer imports from `kanly.api`](#prefer-imports-from-kanlyapi)
  - [Minimal custom model](#minimal-custom-model)
  - [`DataModel`: data block, model block, then MCMC](#datamodel-data-block-model-block-then-mcmc)
  - [Conjugate Bayesian LM](#conjugate-bayesian-lm)
  - [Examples in this repo (`examples/bayes/`)](#examples-in-this-repo-examplesbayes)
- [`kanly.stats`](#kanlystats)
  - [What this subpackage does](#what-this-subpackage-does-4)
  - [Imports](#imports)
  - [Comparison: SciPy / statsmodels / scikit-learn / kanly](#comparison-scipy-statsmodels-scikit-learn-kanly)
- [`kanly.nonparametric`](#kanlynonparametric)
  - [What this subpackage does](#what-this-subpackage-does-5)
  - [Comparison: SciPy / statsmodels / scikit-learn](#comparison-scipy-statsmodels-scikit-learn-4)
  - [Modules](#modules)
  - [KDE](#kde)
  - [LOWESS](#lowess)
  - [STL / MSTL](#stl-mstl)
  - [Splines / interpolation](#splines-interpolation)
  - [Examples in this repo](#examples-in-this-repo-6)
- [`kanly.optimize`](#kanlyoptimize)
  - [What this subpackage does](#what-this-subpackage-does-6)
  - [Quick usage](#quick-usage)
  - [Comparison: SciPy](#comparison-scipy)
  - [Examples in this repo](#examples-in-this-repo-7)
- [`kanly.automatic_differentiation`](#kanlyautomatic_differentiation)
  - [What this subpackage does](#what-this-subpackage-does-7)
  - [Quick usage](#quick-usage-1)
  - [Comparison: JAX / PyTorch](#comparison-jax-pytorch)
- [Other modules (supporting)](#other-modules-supporting)
- [License](#license)

### Package README files

- [`kanly/automatic_differentiation/README.md`](kanly/automatic_differentiation/README.md)
- [`kanly/bayes/README.md`](kanly/bayes/README.md)
- [`kanly/bootstrap/README.md`](kanly/bootstrap/README.md)
- [`kanly/formula/README.md`](kanly/formula/README.md)
- [`kanly/nonparametric/README.md`](kanly/nonparametric/README.md)
- [`kanly/optimize/README.md`](kanly/optimize/README.md)
- [`kanly/plot/README.md`](kanly/plot/README.md)
- [`kanly/regression/README.md`](kanly/regression/README.md)
- [`kanly/regression/generalized_linear_models/README.md`](kanly/regression/generalized_linear_models/README.md)
- [`kanly/regression/generalized_method_of_moments/README.md`](kanly/regression/generalized_method_of_moments/README.md)
- [`kanly/regression/linear_models/README.md`](kanly/regression/linear_models/README.md)
- [`kanly/regression/linear_models/penalized/README.md`](kanly/regression/linear_models/penalized/README.md)
- [`kanly/regression/linear_models/quantile_regression/README.md`](kanly/regression/linear_models/quantile_regression/README.md)
- [`kanly/regression/linear_models/robust/README.md`](kanly/regression/linear_models/robust/README.md)
- [`kanly/regression/nonlinear_least_squares/README.md`](kanly/regression/nonlinear_least_squares/README.md)
- [`kanly/regression/partial_least_squares/README.md`](kanly/regression/partial_least_squares/README.md)
- [`kanly/stats/README.md`](kanly/stats/README.md)
- [`kanly/time_series/README.md`](kanly/time_series/README.md)
- [`kanly/utils/README.md`](kanly/utils/README.md)

---

## Installation

The package is built with **setuptools**; see [`setup.py`](setup.py). Dependencies are listed in **requirements.in** at the repository root (also included as package data when installed).

### Recommended: editable install in a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -U pip setuptools wheel
pip install -r requirements.in
pip install -e .
```

`pip install -e .` installs the `kanly` package from the local tree in editable mode.

### Notes

- `setup.py` reads dependencies via a `requirements()` helper. If dependencies are not picked up automatically by your setuptools version, run `pip install -r requirements.in` before `pip install -e .` (as shown above).
- `requirements.in` pins `scipy == 1.10.1` and `numba-scipy == 0.4.0`; match those if you hit binary or API compatibility issues.

### Python version

Use **Python 3** consistent with the stack in `requirements.in` (packages such as `numba`, `pandas 2.x`, and `scipy 1.10.1`).

---

## Navigating the repository

- `kanly/` — Library source. Subfolders match the user guide below (e.g. `kanly/regression/linear_models/`).
- `examples/` — Runnable scripts; paths mirror `kanly/` (e.g. `examples/regression/linear_models/`). Use them as copy-paste starting points.
- **Sub-readmes** — Long-form API notes live next to the code (linked in each section). This root readme is a map and condensed guide.

Trees such as `kanly/__sandbox__/`, `___to_delete/`, `to_delete2/`, and `wip/` under some packages are **not** stable public API unless explicitly documented.

---

## User guide overview


| Package path                                          | Primary `kanly.api` entry points                                                                                   | Detailed readme                                                                                                                |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| `kanly/formula/`                                      | `sparse_dmatrix`, `sparse_dmatrices`, `SparseDataGetter`                                                           | [`kanly/formula/README.md`](kanly/formula/README.md)                                                                           |
| `kanly/regression/` (overview)                        | Shared formula stack; comparison table across estimators                                                             | [`kanly/regression/README.md`](kanly/regression/README.md)                                                                     |
| `kanly/regression/linear_models/`                     | `lm`, `lm_fast`, `sure`, `shapley_value`, …                                                                        | [`kanly/regression/linear_models/README.md`](kanly/regression/linear_models/README.md)                                         |
| `kanly/regression/linear_models/robust/`              | `rlm`, `RLM`                                                                                                       | [`kanly/regression/linear_models/robust/README.md`](kanly/regression/linear_models/robust/README.md)                           |
| `kanly/regression/linear_models/quantile_regression/` | `qr`, `QR`                                                                                                         | [`kanly/regression/linear_models/quantile_regression/README.md`](kanly/regression/linear_models/quantile_regression/README.md) |
| `kanly/regression/linear_models/penalized/`           | `elastic_net`, `EN`, …                                                                                             | [`kanly/regression/linear_models/penalized/README.md`](kanly/regression/linear_models/penalized/README.md)                     |
| `kanly/regression/generalized_linear_models/`         | `glm`, `GLM`                                                                                                       | [`kanly/regression/generalized_linear_models/README.md`](kanly/regression/generalized_linear_models/README.md)                 |
| `kanly/regression/generalized_method_of_moments/`     | `gmm`, `GMM`, `gmm_iv_linear`, `gmm_iv_nonlinear`, `gmm_mle`                                                       | [`kanly/regression/generalized_method_of_moments/README.md`](kanly/regression/generalized_method_of_moments/README.md)         |
| `kanly/regression/nonlinear_least_squares/`           | `nlls`, `nlls_en`, …                                                                                               | [`kanly/regression/nonlinear_least_squares/README.md`](kanly/regression/nonlinear_least_squares/README.md)                     |
| `kanly/regression/partial_least_squares/`             | `pls1`, `PLS1`, `PLS2`                                                                                             | [`kanly/regression/partial_least_squares/README.md`](kanly/regression/partial_least_squares/README.md)                         |
| `kanly/bootstrap/`                                    | `cov_type='bootstrap'` on fits; `get_joint_bootstrapped_distribution`                                              | [`kanly/bootstrap/README.md`](kanly/bootstrap/README.md)                                                                       |
| `kanly/time_series/` (SARIMAX in `sarimax/`)          | `SARIMAX`, `sarimax`, `ARIMA`, `arima`, `simulate_sarima`, `acf`, `pacf`                                           | [`kanly/time_series/README.md`](kanly/time_series/README.md) — package readme (not under `sarimax/`)                           |
| `kanly/bayes/`                                        | `bmodel`, `DataModel`, `amha`, `mala`, `blm`, `bayes_lm_model`, …                                                  | [`kanly/bayes/README.md`](kanly/bayes/README.md)                                                                               |
| `kanly/stats/`                                        | `logpdf_`*, `pdf_`*, `nopython_gammaln`, … on `kanly.api`; `StatisticalTests` from `kanly.stats.statistical_tests` | [`kanly/stats/README.md`](kanly/stats/README.md)                                                                               |
| `kanly/nonparametric/`                                | `kde`, `lowess`, `LOWESS`, `stl`, `mstl`, `gaussian_kernel_smooth`, `cubic_spline`, …                              | [`kanly/nonparametric/README.md`](kanly/nonparametric/README.md)                                                               |
| `kanly/optimize/`                                     | `bfgs_pqn` (`bfgs`), `cdb`                                                                                         | [`kanly/optimize/README.md`](kanly/optimize/README.md)                                                                         |
| `kanly/automatic_differentiation/`                    | `func_str_to_callable`, `FunctionCallable` (graph helpers in submodule)                                            | [`kanly/automatic_differentiation/README.md`](kanly/automatic_differentiation/README.md)                                       |


The sections below follow this filesystem order. For exhaustive option lists and edge-case notes, follow the readme links in the table.

---

## `kanly.formula`

**See also:** [`kanly/formula/README.md`](kanly/formula/README.md)

### What this subpackage does

`kanly.formula` is the **design-matrix engine** of the package: it turns R-style strings into **sparse** numeric matrices suitable for large problems. Conceptually it overlaps with [`Patsy`](https://patsy.readthedocs.io/) and the formula layer used by [`statsmodels`](https://www.statsmodels.org/stable/user.html#using-formulas-to-specify-models) (`from_formula`), but kanly’s grammar and internals are tuned for **sparsity**, **IV / weights**, and **fixed-effect absorption** hooks consumed by `SparseDataGetter`.

**Useful background:** [design matrix](https://en.wikipedia.org/wiki/Design_matrix), [dummy variable](https://en.wikipedia.org/wiki/Dummy_variable_(statistics)).

### Comparison: SciPy / statsmodels / scikit-learn


| Tool             | Role                                                                                                                                                                                                                                       | Syntax difference                                                                                                                                             |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **SciPy**        | [`sparse.linalg`](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html), [`linalg.lstsq`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html) for numeric `y`, `X` after **you** build the design matrix | **No** formula DSL, IV, weights, or absorb — only linear algebra on arrays you construct.                                                                     |
| **statsmodels**  | `smf.ols('y ~ x', data)` builds dense or sparse paths depending on API                                                                                                                                                                     | Uses **patsy**; instruments are not expressed as `y ~ x                                                                                                       |
| **scikit-learn** | `Pipeline` + transformers; no algebraic formula strings                                                                                                                                                                                    | You build `numpy` arrays yourself (`StandardScaler`, `OneHotEncoder`, …) — **no** single `y ~ x + C(g)` string.                                               |
| **kanly**        | `sparse_dmatrix('x + C(g)', df)` or model formulas on `DataFrame`                                                                                                                                                                          | [`patsy`](https://patsy.readthedocs.io/en/latest/)-style formulas with sparse designs, IV / weights / absorb, and one `formula, data` workflow across models. |


### Core behaviour

- Parse `~` (response / regressors), `|` (instruments), `$` (weights).
- Build sparse designs; drop invalid rows consistently across endog, exog, instruments, and weights.
- Support categoricals, `I(...)`, polynomials, lags, and more.

### Core entry points

- `kanly.api.sparse_dmatrix(formula_rhs, data)`
- `kanly.api.sparse_dmatrices(full_formula, data)`
- `kanly.formula.data_getter.SparseDataGetter.get_data(data, formula, ...)`

### Formula DSL (common)

- Intercept control: `-1` (drop intercept)
- Categorical coding: `C(col)`
- Quoted column names: `Q("column with spaces")`
- Arithmetic wrapper: `I(...)`
- Polynomial shorthand: `poly(x, 3)`, `poly(x, [1,3,5])`, `poly(x, 3, -1)`
- Lags/leads: `L(x, 1)`, `L(x, [1,2,4])`
  * grouped lags `L(x, 1, grp)`
- [B-Splines](https://en.wikipedia.org/wiki/B-spline): `bs(x,degree=..,df=..,include_intercept=..)` 
  * (see [patsy](https://patsy.readthedocs.io/en/latest/spline-regression.html))
- Demeaning: `center(x)` or weighted `center(x, w)`
- Standardizing: `standardize(x)` or `standardize(x, w)` (for weights `w`, demeans and divides by standard deviation).
- Trends: `trend([1,3])` for `t + t**3`
  * (assumes ordered data)
- Seasonal fixed effects: `seasonality(12)` will generate 12 fixed effects
  * (assumes ordered data)

These transforms are "stateful" and are safe for predicting on other data sets - that is if we demean, the model remembers the mean.

**Pitfalls:** At most one `~`, one `|`, and one `$` per formula. Null rows are unioned across blocks. With absorb/fixed effects, `-1` in exog is disallowed. Use `sparse_dmatrices` or model APIs for full `y ~ X`.

### Minimal usage

```python
from kanly.api import sparse_dmatrix, sparse_dmatrices

X_obj = sparse_dmatrix('x*C(g) + I(x**2) + poly(z, 3)', df)
X = X_obj.values
cols = X_obj.column_names

y_obj, X_obj = sparse_dmatrices('y ~ x1 + C(city) + poly(x2, 2)', df)
```

```python
from kanly.formula.data_getter import SparseDataGetter

bundle = SparseDataGetter.get_data(
    df,
    'y ~ x + C(grp) | z + C(grp) $ w',
    absorb=None,
    check_constant_cols=True,
)
# bundle keys include ENDOG, EXOG, INSTRUMENTS, WEIGHTS, VALID_OBS_ROWS, ...
```

### Examples in this repo

- [`examples/formula/example_sparse_dmatrix.py`](examples/formula/example_sparse_dmatrix.py) — building sparse designs with `sparse_dmatrix` / debug output.
- [`examples/example_formula.py`](examples/example_formula.py) — rich formula: `poly`, interactions, `Q("column name")`, `np.log`, `-1` to drop intercept.

**Tip:** Complex RHS formulas work the same in `lm`; this example is a good stress test for parser + sparse expansion.

---

## `kanly.regression`

**See also:** [`kanly/regression/README.md`](kanly/regression/README.md) (regression-wide comparison table and syntax comparisons)

### What this subpackage does

`kanly.regression` hosts almost all **parametric frequentist** estimators: linear, GLM, GMM, nonlinear least squares, robust and quantile linear models, elastic net, and partial least squares. They share [`kanly.formula`](kanly/formula/README.md)-based sparse matrices, so adding high-cardinality factors or long formulas stays memory-efficient compared to always densifying.

**Useful background:** [`linear regression`](https://en.wikipedia.org/wiki/Linear_regression), [`generalized linear model`](https://en.wikipedia.org/wiki/Generalized_linear_model), [`instrumental variables`](https://en.wikipedia.org/wiki/Instrumental_variables_estimation), [`generalized method of moments`](https://en.wikipedia.org/wiki/Generalized_method_of_moments), [`nonlinear least squares`](https://en.wikipedia.org/wiki/Non-linear_least_squares).

### Comparison: SciPy / statsmodels / scikit-learn


| Area                       | SciPy                                                                                                                                                                                                                                                                     | statsmodels                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | scikit-learn                                                                | kanly                                                                                                                                                                                            |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **OLS / WLS / GLS / FGLS / GLSAR** | [`linalg.lstsq`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html), [`sparse.linalg.lsmr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsmr.html) — numeric `y`, `X` only; **no** formula DSL or inference layer | [`OLS`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html) / [`WLS`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.WLS.html) / [`GLS`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.GLS.html) (and `smf.`*); [`GLSAR`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.GLSAR.html) (experimental, **Cochrane–Orcutt** only — `whiten()` drops first `p` obs); FGLS-style heteroskedasticity via iterative workflows | `LinearRegression` — prediction-focused; no classical inference summaries   | `lm` — WLS, GLS (`sigma`), `do_fgls=True` (FGLS for heteroskedasticity); **`glsar` / `GLSAR`** for AR(`p`) errors with **`full_information=True`** (Prais–Winsten) or **`False`** (Cochrane–Orcutt); HC / cluster / HAC / bootstrap; **absorb**; sparse formulas |
| **IV (2SLS / W2SLS)**      | —                                                                                                                                                                                                                                                                         | Sandbox [`IV2SLS`](https://www.statsmodels.org/stable/generated/statsmodels.sandbox.regression.gmm.IV2SLS.html) — array API (`endog`, `exog`, `instrument`); not unified formula IV syntax                                                                                                                                                                                                                                                                                                                                                         | —                                                                           | `lm` with `y ~ x | z` formula syntax; 2SLS / W2SLS with HC / cluster / bootstrap SEs; `absorb=` for Frisch-Waugh IV                                                                              |
| **GLM**                    | — (assemble log-likelihood + [`optimize`](https://docs.scipy.org/doc/scipy/reference/optimize.html) yourself)                                                                                                                                                             | [`smf.glm`](https://www.statsmodels.org/stable/glm.html) — families, links, `summary()`; [`GLMGam`](https://www.statsmodels.org/stable/generated/statsmodels.gam.generalized_additive_model.GLMGam.html) for smooths                                                                                                                                                                                                                                                                                                                                                                                              | `LogisticRegression` etc. — **no** full GLM family objects (e.g. Gamma, NB) | `glm` — same covariance / IV / absorb stack as `lm` on **sparse** formula designs; **`gam`** for penalized B-spline GAMs ([README](kanly/regression/generalized_linear_models/README.md#generalized-additive-models-gam)) |
| **Quantile**               | —                                                                                                                                                                                                                                                                         | [`QuantReg`](https://www.statsmodels.org/stable/generated/statsmodels.regression.quantile_regression.QuantReg.html) / [`smf.quantreg`](https://www.statsmodels.org/stable/examples/notebooks/generated/quantile_regression.html) (IRLS; formula or arrays)                                                                                                                                                                                                                                                                                         | —                                                                           | `qr` — sparse IRLS, smooth check loss; control-function IV path ([`kanly/regression/linear_models/quantile_regression/README.md`](kanly/regression/linear_models/quantile_regression/README.md)) |
| **Robust**                 | —                                                                                                                                                                                                                                                                         | [`RLM`](https://www.statsmodels.org/stable/generated/statsmodels.robust.robust_linear_model.RLM.html)                                                                                                                                                                                                                                                                                                                                                                                                                                              | `HuberRegressor` (different algorithmic focus)                              | `rlm` sparse IRLS                                                                                                                                                                                |
| **Penalized**              | — (use third-party or `optimize` with penalty by hand)                                                                                                                                                                                                                    | Limited built-ins                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | `ElasticNet`, `Lasso`, `Ridge` on dense `ndarray`                           | `elastic_net` on **sparse** formula designs, optional OLS refit                                                                                                                                  |
| **NLLS**                   | [`optimize.least_squares`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html) — you code residuals/Jacobians                                                                                                                         | [`bayesNonlinearLS`](https://www.statsmodels.org/stable/generated/statsmodels.miscmodels.nonlinls.NonlinearLS.html) in `statsmodels.miscmodels` (subclass `_predict`; array-oriented — **not** kanly-style formula NLLS)                                                                                                                                                                                                                                                                                                                           | —                                                                           | `nlls` / `nlls_en` with formula parsers, bounds, robust/quantile root losses                                                                                                                     |
| **GMM**                    | [`optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/optimize.html) only — **no** moment API                                                                                                                                                                  | [`sandwich`](https://www.statsmodels.org/stable/sandwich.html) ecosystem; sandbox [`IV2SLS`](https://www.statsmodels.org/stable/generated/statsmodels.sandbox.regression.gmm.IV2SLS.html) and related GMM helpers                                                                                                                                                                                                                                                                                                                                  | —                                                                           | `gmm`, `gmm_iv_linear`, `gmm_iv_nonlinear`, `gmm_mle`                                                                                                                                            |


#### Syntax comparisons

The snippets below assume a `pandas` `DataFrame` `df` with columns such as `y`, `x`, `z`, `grp`, and `w` (weights). Kanly, statsmodels, sklearn, and SciPy differ mainly in **how you specify the model** (formula vs arrays vs residual callable) and in **built-in inference** (SE types, IV, sparse designs).

**OLS / WLS — `lm` vs statsmodels `ols` / `wls`**

```python
import statsmodels.formula.api as smf
from kanly.api import lm

# statsmodels — formula API
fit_sm = smf.ols('y ~ x + C(grp)', data=df).fit()
fit_sm_w = smf.wls('y ~ x + C(grp)', data=df, weights=df['w']).fit()

# kanly — same formula strings; optional robust / cluster / bootstrap SEs
fit = lm('y ~ x + C(grp)', df)
fit_w = lm('y ~ x + C(grp) $ w', df, cov_type='HC1')
```

**IV (2SLS) — `lm` vs sandbox `IV2SLS`**

```python
from patsy import dmatrices
import numpy as np
from statsmodels.sandbox.regression.gmm import IV2SLS
from kanly.api import lm

# statsmodels — IV2SLS takes numeric arrays only. Categorical `grp` must be expanded
# to dummies (e.g. with patsy) before fitting; exogenous controls belong in BOTH
# exog and instrument.
y, X = dmatrices('y ~ x + C(grp)', df, return_type='matrix')
_, Z = dmatrices('y ~ z + C(grp)', df, return_type='matrix')
y = np.squeeze(np.asarray(y))
X = np.asarray(X)
Z = np.asarray(Z)
fit_sm = IV2SLS(y, X, Z).fit()

# kanly — same model syntax as OLS; C(grp) handled in the formula parser
fit_iv = lm('y ~ x + C(grp) | z + C(grp)', df)
fit_iv_w = lm('y ~ x + C(grp) | z + C(grp) $ w', df)
fit_iv_absorb = lm('y ~ x | z', df, absorb='grp')
```

**GLM — `glm` vs statsmodels `glm`**

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf
from kanly.api import glm

# statsmodels
fit_sm = smf.glm(
    'y ~ poly(x, 2)',
    data=df,
    family=sm.families.Binomial(),
).fit()

# kanly — family as string; same formula grammar as lm (IV, weights, absorb, cov_type)
fit = glm('y ~ poly(x, 2)', df, family='binomial', cov_type='HC1')
fit_iv = glm('y ~ x | z', df, family='poisson')
```

**Penalized linear — `elastic_net` vs sklearn `ElasticNet`**

```python
import numpy as np
from sklearn.linear_model import ElasticNet
from kanly.api import elastic_net

# sklearn — dense design matrix X and response y (no formula DSL)
# X = np.column_stack([np.ones(n), df['x'].values, ...])  # build manually
fit_sk = ElasticNet(alpha=1e-4, l1_ratio=1.0, fit_intercept=False).fit(X, y)

# kanly — formula on DataFrame; sparse design; per-coefficient alpha dict
fit = elastic_net('y ~ x0 + x1 + x2 + ...', df, alpha=1e-4, l1_ratio=1.0)
fit = elastic_net('y ~ x + z $ w', df, alpha={'x': 3.0}, l1_ratio=1.0, refit=True)
```

**Nonlinear least squares — `nlls` vs `scipy.optimize.least_squares`**

```python
import numpy as np
from scipy.optimize import least_squares
from kanly.api import nlls

# scipy — residual vector; x0 is required
def residuals(params):
    intercept, beta, gamma = params
    return df['y'].to_numpy() - (intercept + beta * np.exp(gamma * df['x'].to_numpy()))

res = least_squares(residuals, x0=[1.0, 3.0, -0.5])

# kanly — {braces} name parameters; [brackets] name data columns; start_params is the
# analogue of x0 (dict keyed by name or vector in parameter order). If omitted, the
# solver starts from 0 and may run a subsample pre-fit (subsample=...) first.
fit = nlls(
    '[y] ~ {Intercept} + {beta} * exp({gamma} * [x])',
    df,
    start_params={'Intercept': 1.0, 'beta': 3.0, 'gamma': -0.5},
)
fit = nlls(
    '[y] ~ {Intercept} + {beta} * exp({gamma} * [x]) $ [w]',
    df,
    start_params=[1.0, 3.0, -0.5],
    max_iter=100,
)
```

The subsections below follow the `kanly/regression/` directory layout and include example code with printed summaries from the `examples/` tree.

### `linear_models/`

`kanly.regression.linear_models` provides a unified, sparse-first linear regression framework.

#### Features

- **OLS / WLS**; **IV (2SLS / W2SLS)** with **inference on the IV estimator** (standard errors, *t*-tests, CIs—not OLS-on-endogenous regressors); **absorbed fixed effects** (Frisch-Waugh).
- **FGLS**; **GLSAR** (`glsar` / `GLSAR`) — feasible GLS with AR(`p`) errors, iterative whitening, Prais–Winsten (`full_information=True`, default) or Cochrane–Orcutt (`full_information=False`); **ridge**; **SURE** (seemingly unrelated regressions).
- **Multiple outcomes**; **fast path** `lm_fast` (LSMR, no inverse — no SEs).
- **Cluster / HC / HAC / bootstrap** standard errors; **two-way clustering** via a tuple of group columns in `cov_kwds`.
- **Shapley R²** decomposition by formula term (Owen-style; exact or permutation sampling; quadratic-form fast path); **permutation tests**.
- **Numba-accelerated** pieces in the estimation stack where applicable (alongside sparse linear algebra).

#### Quick start (OLS)

```python
from kanly.api import lm

fit = lm('y ~ x + C(grp)', df, use_t=True)
print(fit.summary())
```

Example output (from [`examples/regression/linear_models/example_ordinary_least_squares.py`](examples/regression/linear_models/example_ordinary_least_squares.py)):

```text
==========================================================================
Linear Model Results
==========================================================================

Dep. Variable:   y
Method:                         OLS    R-squared:                   0.7551
Covariance Type:          OLS_SMALL    Adj. R-squared:              0.7213

=====================================================================
                 coef        std err      t   p>|t| [0.025,    0.975]
---------------------------------------------------------------------
Intercept       1.148  ****  0.07958  14.43  <0.001     0.99    1.306
x              -0.313  ****  0.01993 -15.70  <0.001  -0.3526  -0.2734
... (C(grp) dummies omitted) ...
=====================================================================

formula:  y ~ x + C(grp)
```

#### Formula syntax


| Syntax                     | Effect                      |
|----------------------------|-----------------------------|
| `y ~ x + z`                | OLS                         |
| `y1 + y2 ~ x + z`          | Multiple Outcome OLS        |
| `y ~ x + z $ w`            | WLS (`w` = weight column)   |
| `y ~ x + z \| z1 + z2`     | IV (instruments `z1`, `z2`) |
| `y ~ x + z \| z1 + z2 $ w` | Weighted IV                 |
| `C(grp)`                   | Categorical dummies         |
| `I(x**2)`                  | Inline transformation       |
| `poly(x, 3)`               | Degree-3 polynomial         |
| `L(x)`                     | Lag of `x`                  |
| `L(x,3)`                   | 3rd lag of `x`              |
| `L(x,[2,3])`               | 2nd and 3rd lag of `x`      |
| `x*z`                      | Main effects + interaction  |
| `x:z`                      | Interaction only            |

For a categorical variable like `grp`, pass `absorb='grp'` (or `absorb=('grp2', 'grp1')` for multiple interacted) to absorb fixed effects without expanding dummies.

#### WLS, fixed effects, and IV

**Syntax**

```python
from kanly.api import lm

fit = lm('y ~ x + C(grp) $ obs', df)

fit = lm('y ~ x', df, absorb='grp')
fit = lm('y ~ x', df, absorb=('grp', 'period'))

fit_iv = lm('y ~ x + C(grp) | z + C(grp)', df)
print(fit_iv.summary_iv())
```

**Instrumental variables example** ([`examples/regression/linear_models/instrumental_variables/example_instrumental_variables.py`](examples/regression/linear_models/instrumental_variables/example_instrumental_variables.py)):

```python
from kanly.api import lm, compare_results

fit_ols = lm('y ~ x + C(grp) $ obs', df, specification_name='WLS')
fit_iv = lm('y ~ x + C(grp) | z + C(grp) $ obs', df, specification_name='WLS-IV')
fit_iv_absorb = lm('y ~ x | z $ obs', df, specification_name='WLS-IV-Absorb', absorb='grp')

print(compare_results(
    [fit_iv, fit_iv_absorb, fit_ols],
    ref_param_values={'x': -.3},
    parameter_subset=['x'],
    show_bse=True, show_formulas=True))
```

When `x` is endogenous, WLS on `x` is biased; IV recovers the causal slope. Standard errors and *t*-tests on `x` in the IV fit refer to the **2SLS / W2SLS** estimator.

Example output — **WLS (biased)** on endogenous `x` (same example file):

```text
Linear Model Results
WLS
Method:                         WLS    R-squared:                   0.9123
...
Intercept       1.197  ****   0.01347  88.92  <0.001     1.171    1.224
x             -0.2347  ****  0.004888 -48.01  <0.001   -0.2443  -0.2251
...
formula:  y ~ x + C(grp) $ obs
```

Example output — **WLS-IV** (2SLS with weights):

```text
Linear Model Results
WLS-IV
Method:                  IV (W2SLS)    R-squared:                    0.797
...
Intercept        1.192  ****   0.0205  58.12  <0.001     1.151    1.232
x               -0.321  ****  0.01468 -21.86  <0.001   -0.3499   -0.292
...
formula:  y ~ x + C(grp) | z + C(grp) $ obs

Endogenous Regressors: x
Excluded Regressors:   z
```

Example output — **WLS-IV with absorbed `grp`**:

```text
Linear Model Results
WLS-IV-Absorb
Method:                  IV (W2SLS)
...
x  -0.321  ****  0.01468 -21.86  <0.001  -0.3499  -0.292

formula:  y ~ x | z  $ obs
Absorbed: 'grp', num=12
Endogenous Regressors: x
Excluded Regressors:   z
```

`compare_results` on the three fits (subset `x`):

```text
                           (0)         (1)        (2)   |  Reference
x                       -0.321      -0.321     -0.235   |     -0.300
Method:             IV (W2SLS)  IV (W2SLS)        WLS
(0)  "WLS-IV"     y ~ x + C(grp) | z + C(grp) $ obs
(1)  "WLS-IV-Absorb"   y ~ x | z, Absorbed: grp
(2)  "WLS"        y ~ x + C(grp) $ obs
```

#### Cluster-robust, HC, HAC, and bootstrap

```python
fit = lm('y ~ x', df, cov_type='cluster', cov_kwds={'groups': 'firm_id'})

# Two-way clustering: pass a tuple of grouping columns
fit = lm('y ~ x', df, cov_type='cluster',
         cov_kwds={'groups': ('grp1', 'grp2')})

fit_hc1 = lm('y ~ x', df, cov_type='HC1')

fit = lm('y ~ x', df, cov_type='hac',
         cov_kwds={'maxlags': 4, 'kernel': 'bartlett'})

# Bayesian bootstrap (default method)
fit = lm('y ~ x + C(grp)', df,
         cov_type='bootstrap',
         cov_kwds={'n_samples': 500, 'method': 'bayesian', 'alpha': 1.0})

# Classical bootstrap with replacement
fit = lm('y ~ x', df,
         cov_type='bootstrap',
         cov_kwds={'n_samples': 1000, 'method': 'classical', 'seed': 42})

# Block (cluster) bootstrap — resample clusters, not rows
fit = lm('y ~ x', df,
         cov_type='bootstrap',
         cov_kwds={'n_samples': 500, 'method': 'bayesian',
                   'groups': 'firm_id', 'max_processes': 4})
```

**Bootstrap methods** (see [`kanly/bootstrap/README.md`](kanly/bootstrap/README.md)):


| `cov_kwds['method']`   | Weights                                                          | Typical use                                                         |
| ---------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------- |
| `'bayesian'` (default) | Dirichlet(`alpha`, …, `alpha`), scaled to `nobs` or `#` clusters | Smooth, strictly positive weights; default for many kanly workflows |
| `'classical'`          | Integer replication counts (rows or clusters)                    | Traditional with-replacement bootstrap                              |


With `groups` in `cov_kwds`, both methods **resample clusters** (block bootstrap): one weight vector per cluster, broadcast to member rows. Fits store `bootstrapped_params` and a bootstrap covariance; use `get_joint_bootstrapped_distribution([fit_a, fit_b])` when the same bootstrap scheme was applied to several models and you need a **joint** parameter covariance.

A helper `two_way_cluster` also exists in [`kanly/regression/linear_models/two_way_cluster.py`](kanly/regression/linear_models/two_way_cluster.py) but is **not** exported from `kanly.api`; use the tuple-`groups` pattern above or import that module directly.

Supported `cov_type` values include `'OLS'`, `'OLS_SMALL'`, `'HC0'`–`'HC3'`, `'HAC'`, `'HAC_PANEL'`, `'CLUSTER'`, `'BOOTSTRAP'`, etc.

#### FGLS and ridge

```python
fit = lm('y ~ x + z', df, do_fgls=True,
         fgls_kwds={'maxiter': 20, 'tol': 1e-8})

fit = lm('y ~ x + z', df, ridge_kwds={'alpha': 0.5})
```

Ridge: inference is intentionally questionable (biased estimator); summaries warn accordingly.

#### GLSAR (AR errors)

**GLSAR** fits linear models when residuals follow AR(`p`): each iteration estimates AR coefficients on residuals, whitens `y` and `X`, and refits GLS on the whitened system until AR parameters stabilize. Entry points: `glsar(formula, data, nlags)` (formula) and `GLSAR(endog, exog, nlags)` (arrays), exported from `kanly.api`.

- **`full_information=True`** (default): **Prais–Winsten** whitening — uses stationary initial covariance for the first `nlags` observations.
- **`full_information=False`**: **Cochrane–Orcutt** — innovation rows only (closer to statsmodels `GLSAR`, which drops the first `p` observations in `whiten()`).

AR diagnostics are on `fit.glsar_info` (`ar_params`, `numiter`, `full_information`, etc.). Not compatible with absorb, IV, WLS weights, or known `sigma` GLS. See [`kanly/regression/linear_models/README.md`](kanly/regression/linear_models/README.md).

#### SURE

```python
from kanly.api import sure

fit = sure(
    [
        {'formula': 'y1 ~ x $ wts1', 'data': df},
        {'formula': 'y2 ~ x', 'data': df},
    ],
    cov_type='cluster',
    cov_kwds={'groups': 'user_id'},
)
```

#### Large sparse / fast path

```python
from kanly.api import lm, lm_fast

fit = lm('y ~ x*C(g) + poly(z, 3)', df)
fit_fast = lm_fast('y ~ x*C(g) + poly(z, 3)', df)  # coefficients only
```

On very large `n`, compare `fit.fit_elapsed` and `fit_fast.fit_elapsed`:

```python
# see examples/regression/linear_models/example_fast_lm.py
fit = lm('y ~ x*C(g) + I(x**2) + poly(z, 3)', df, debug=False)
fit_fast = lm_fast('y ~ x*C(g) + I(x**2) + poly(z, 3)', df, debug=False)
print(fit.fit_elapsed, fit_fast.fit_elapsed)
```

#### Performance vs statsmodels

For large linear models—especially with high-cardinality fixed effects—kanly is
often **orders of magnitude faster** than [`statsmodels`](https://www.statsmodels.org/)
`smf.ols`, while producing the **same** slope coefficients, standard errors, and
*t*-statistics (plus kanly-only features: IV, `absorb=`, cluster/HAC/bootstrap SEs,
SURE, Shapley R², etc.).

Runnable benchmark:
[`examples/regression/linear_models/example_kanly_vs_statsmodels_time.py`](examples/regression/linear_models/example_kanly_vs_statsmodels_time.py).
It fits 3M observations with 20 slopes and 180 group levels (201 regressors with
dummies), all with `cov_type='HC1'`. Representative timings from that script:

| Fit | Elapsed (approx.) |
|-----|-------------------|
| kanly `lm('y ~ x1 + ... + C(g)', ...)` | ~16 s |
| kanly `lm(..., absorb='g')` | ~13 s |
| statsmodels `smf.ols(...).fit(cov_type='HC1')` | ~18 min |

Two implementation differences explain most of the gap:

1. **Sparse designs** — kanly builds sparse `X` and forms `X'X` / `X'y` without
   densifying the full `n × k` design matrix.
2. **Normal equations** — kanly solves `β = inv(X'X) @ (X'y)` (`k × k` and `k × 1`).
   A common statsmodels path uses `(inv(X'X) @ X') @ y`, where `inv(X'X) @ X'` is
   `k × n` and **dense** even when `X` is sparse—far more memory and arithmetic.

You do not trade inference for speed: `lm` still returns `params`, `bse`, `pvalues`,
`conf_int()`, `predict()`, `wald_test()`, and the same robust covariance types as
statsmodels (HC, cluster, HAC, bootstrap), with additional estimators documented
below.

#### Matrix API

```python
from kanly.regression.linear_models.model import SparseLinearModel

fit = SparseLinearModel.LM(y, X, weights=w, instruments=Z,
                           exog_names=['Intercept', 'x1', 'x2'],
                           has_constant=True)
result = SparseLinearModel.LM_fast(y, X, weights=w, exog_names=['Intercept', 'x1', 'x2'])
```

#### Result objects

Key attributes: `params`, `bse`, `pvalues`, `rsquared`, `rsquared_adj`, `resid`, `fittedvalues`, `llf`, `aic`, `bic`, `wald_test`, `conf_int()`, `summary()`, `summary_df()`, `plot_diagnostics()`, etc.

**Prediction:** `fit.predict()` and `fit.predict(data=new_df)`; out-of-sample prediction is not supported for all IV / absorb cases (see submodule readme).

*Shapley R² (formula terms, `fit.shapley_value()` or `shapley_value`), permutation tests, and lift tests: [`kanly/regression/linear_models/README.md`](kanly/regression/linear_models/README.md).*

#### Examples in this repo (`examples/regression/linear_models/`)

- [`example_ordinary_least_squares.py`](examples/regression/linear_models/example_ordinary_least_squares.py), [`example_weighted_least_squares.py`](examples/regression/linear_models/example_weighted_least_squares.py) — basic OLS / WLS.
- [`example_absorbed_fixed_effects.py`](examples/regression/linear_models/example_absorbed_fixed_effects.py) — `absorb=`.
- [`instrumental_variables/example_instrumental_variables.py`](examples/regression/linear_models/instrumental_variables/example_instrumental_variables.py) — 2SLS formulas.
- [`example_clustered_ses.py`](examples/regression/linear_models/example_clustered_ses.py), [`example_clustered_ses_2way.py`](examples/regression/linear_models/example_clustered_ses_2way.py) — one- vs two-way cluster SEs (with `compare_results`).
- [`example_bootstrap.py`](examples/regression/linear_models/example_bootstrap.py), [`example_block_bootstrap.py`](examples/regression/linear_models/example_block_bootstrap.py) — bootstrap covariances.
- [`example_fast_lm.py`](examples/regression/linear_models/example_fast_lm.py) — `lm` vs `lm_fast` timing on millions of rows.
- [`example_kanly_vs_statsmodels_time.py`](examples/regression/linear_models/example_kanly_vs_statsmodels_time.py) — `lm` vs statsmodels `ols` on 3M rows with 180 fixed effects (timing + numerical agreement).
- [`example_large_sparse_regression.py`](examples/regression/linear_models/example_large_sparse_regression.py) — huge sparse `SparseDataFrame` + `lm`.
- [`example_feasible_generalized_least_squares.py`](examples/regression/linear_models/example_feasible_generalized_least_squares.py), [`example_ridge.py`](examples/regression/linear_models/example_ridge.py).
- [`example_sure.py`](examples/regression/linear_models/example_sure.py), [`example_seemingly_unrelated_regression.py`](examples/regression/linear_models/example_seemingly_unrelated_regression.py).
- [`example_multiple_outcomes.py`](examples/regression/linear_models/example_multiple_outcomes.py), [`example_linear_model_prediction.py`](examples/regression/linear_models/example_linear_model_prediction.py).
- [`example_shapley_value.py`](examples/regression/linear_models/example_shapley_value.py), [`example_polynomial_regression.py`](examples/regression/linear_models/example_polynomial_regression.py).
- [`example_ordinary_least_squares_with_indexing.py`](examples/regression/linear_models/example_ordinary_least_squares_with_indexing.py) — row `index` subsets.

Also: [`examples/regression/example_ridge_different_ways.py`](examples/regression/example_ridge_different_ways.py), [`examples/regression/example_lasso_different_ways.py`](examples/regression/example_lasso_different_ways.py) — ridge / LASSO comparisons across APIs.

---

### `linear_models/robust/`

Sparse-first **M-estimation** via IRLS.

#### Quick start

```python
from kanly.api import rlm

fit = rlm('y ~ x', df)
print(fit)
```

#### Norm functions (`M=`)

Examples: `'HuberT'` (default), `'TukeyBiweight'`, `'LeastSquares'`, `'AndrewWave'`, `'RamsayE'`, `'TrimmedMean'`.

#### Covariance


| `cov_type`     | Role                      |
| -------------- | ------------------------- |
| `'H1'`         | Default Huber-style       |
| `'H2'`, `'H3'` | Alternatives              |
| `'SANDWICH'`   | Heteroscedasticity-robust |
| `'BOOTSTRAP'`  | Resampling                |


#### Limitations

Single outcome only; **no IV**; **no absorb** in `rlm` (use explicit dummies or pre-partialling). Bootstrap with integer `index` is not supported.

*Details: [`kanly/regression/linear_models/robust/README.md`](kanly/regression/linear_models/robust/README.md).*

#### Examples in this repo

- [`examples/regression/linear_models/robust/example_robust_regression.py`](examples/regression/linear_models/robust/example_robust_regression.py)
- [`examples/regression/linear_models/robust/example_robust_regression_instrumental_variables.py`](examples/regression/linear_models/robust/example_robust_regression_instrumental_variables.py) — illustrates limitation / error path for IV with RLM (read script header).

---

### `linear_models/quantile_regression/`

Quantile regression via IRLS and smooth surrogate losses (Huber, SoftL1, SmoothCup); sparse CSC designs; optional IV via **control functions** (experimental).

#### Quick start

```python
from kanly.api import qr

fit = qr('y ~ x + z', df, tau=0.9)
print(fit.summary())
```

#### Multiple quantiles

```python
taus = (0.1, 0.5, 0.9)
fits = qr('y ~ poly(x, 3)', df, taus,
          cov_type='bootstrap',
          cov_kwds={'seed': 1, 'n_samples': 500, 'max_processes': 6},
          line_search=True)
```

#### Covariance


| `cov_type`    | Notes                              |
| ------------- | ---------------------------------- |
| `'IID'`       | KDE-based, homoscedastic           |
| `'ROBUST'`    | Heteroscedastic sandwich           |
| `'BOOTSTRAP'` | Preferred for hard inference cases |


#### IV (experimental)

```python
fit_iv = qr('y ~ x + C(grp) | z + C(grp)', df, tau=0.9,
            residual_inclusion=True, residual_inclusion_order=1)
```

Prefer bootstrap covariance for IV quantile workflows.

*Details: [`kanly/regression/linear_models/quantile_regression/README.md`](kanly/regression/linear_models/quantile_regression/README.md).*

#### Examples in this repo

- [`examples/regression/linear_models/quantile_regression/example_quantile_regression.py`](examples/regression/linear_models/quantile_regression/example_quantile_regression.py) — example output in the file docstring

(Sketch / WIP: [`examples/work_in_progress/example_quantile_regression_instrumental_variables.py`](examples/work_in_progress/example_quantile_regression_instrumental_variables.py).)

---

### `linear_models/penalized/`

Coordinate-descent **elastic net** on sparse least squares: LASSO (`l1_ratio=1`), ridge (`l1_ratio=0`), or mixtures.

#### Formula API

```python
from kanly.api import elastic_net

fit = elastic_net(
    "y ~ " + " + ".join([f"x{i}" for i in range(k)]),
    df,
    alpha=0.0001,
    l1_ratio=1.0,
    selection="random",
    active_set=True,
)
print(fit.summary(show_only_non_zero=True, show_formula=False))
```

#### Post-selection OLS refit

```python
fit_en, fit_ols = elastic_net(
    "y ~ poly(x,4) + C(geo)",
    df,
    alpha=0.01,
    l1_ratio=1.0,
    refit=True,
)
```

Classical SEs for penalized estimates are not provided; use `refit=True` when you need OLS inference on selected variables.

*Objective definition and array API: [`kanly/regression/linear_models/penalized/README.md`](kanly/regression/linear_models/penalized/README.md).*

#### Examples in this repo

- [`examples/regression/linear_models/penalized/example_elastic_net.py`](examples/regression/linear_models/penalized/example_elastic_net.py)
- [`examples/regression/linear_models/penalized/example_elastic_net_with_refit.py`](examples/regression/linear_models/penalized/example_elastic_net_with_refit.py)

---

### `generalized_linear_models/`

Sparse GLMs with IRLS, canonical or custom links, frequency weights, robust covariances, bootstrap, elastic-net penalties, IV-style formulas, and **GAM** smooths via ``gam`` ([details](kanly/regression/generalized_linear_models/README.md#generalized-additive-models-gam)).

#### Basic usage

```python
from kanly.api import glm

fit = glm(
    "y ~ x + poly(z, 2)",
    data=df,
    family="binomial",
)
print(fit.summary())
pred = fit.predict(df)
```

#### Families (string names)

- `binomial` / `bernoulli`, `poisson`, `gaussian`, `gamma`, `inverse_gaussian`, `negative_binomial` (optional dispersion in string, e.g. `"negative_binomial(0.5)"`).

#### Links (examples)

`logit`, `probit`, `identity`, `log`, `cloglog`, `sqrt`, `negative_inverse`, `inverse_squared`, etc. Each family restricts allowed links via `safe_links()`.

#### Weights and covariance

```python
fit = glm("y ~ x $ w", df, family="poisson")

fit = glm("y ~ x", df, family="binomial", cov_type="hc1")
fit = glm("y ~ x", df, family="binomial", cov_type="bootstrap",
          cov_kwds={"n_samples": 250, "seed": 123})
```

#### Regularization

```python
fit = glm("y ~ x + z", df, family="binomial",
          alpha=0.1, l1_ratio=0.5, normalize=True)
```

#### IV GLM and residual inclusion

For **linear models**, 2SLS (substitute fitted values from the first stage) and
the control-function (residual inclusion) approach are equivalent — either gives
a consistent estimator. For **nonlinear GLMs** the two diverge:

- **IV without residual inclusion** (`residual_inclusion=False`): substitutes
the predicted endogenous variable into the nonlinear link function. This is
inconsistent for non-identity links (the "forbidden regression") and should be
treated only as a diagnostic or baseline comparison.
- **IV with residual inclusion** (`residual_inclusion=True`): appends
first-stage residuals as extra regressors (control-function approach). This is
the consistent estimator for nonlinear models. A significant coefficient on
the appended residual also tests for endogeneity.

```python
# Naive GLM — biased when x is endogenous
fit = glm("y ~ x", df, family="poisson", cov_type="bootstrap")

# IV without residual inclusion — NOT consistent for nonlinear families
fit_iv = glm("y ~ x | z", df, family="poisson",
             residual_inclusion=False, cov_type="bootstrap")

# IV with residual inclusion — consistent control-function estimator
fit_iv_ri = glm("y ~ x | z", df, family="poisson",
                residual_inclusion=True, residual_inclusion_order=2,
                cov_type="bootstrap")

print(compare_results([fit, fit_iv, fit_iv_ri],
                      fit_titles=["GLM", "GLM-IV", "GLM-IV-RI"],
                      ref_param_values={"x": 2.0}))
```

Bootstrap covariance is strongly recommended for IV GLM: asymptotic sandwich
formulas do not account for first-stage estimation uncertainty.

*Full explanation, example output, and endogeneity test interpretation:
[`kanly/regression/generalized_linear_models/README.md`](kanly/regression/generalized_linear_models/README.md).*

#### GAM (smooth covariates)

Penalized B-spline smooths use the same IRLS core as ``glm``; the roughness
matrix is added to ``X'WX`` each iteration. Entry point: ``gam`` / ``GAM``.

```python
from kanly.api import gam

fit = gam("y ~ x", df, penalty=dict(x=0.05), df=dict(x=20), family="poisson")
```

*Details and tuning guidance:
[`kanly/regression/generalized_linear_models/README.md#generalized-additive-models-gam`](kanly/regression/generalized_linear_models/README.md#generalized-additive-models-gam).*

#### Examples in this repo

- [`examples/regression/generalized_linear_models/example_logistic_regression.py`](examples/regression/generalized_linear_models/example_logistic_regression.py)
- [`examples/regression/generalized_linear_models/example_logistic_regression_large_scale.py`](examples/regression/generalized_linear_models/example_logistic_regression_large_scale.py) — large `n` + high-cardinality `C(g)` + `debug=True`.
- [`examples/regression/generalized_linear_models/example_poisson_regression.py`](examples/regression/generalized_linear_models/example_poisson_regression.py)
- [`examples/regression/generalized_linear_models/example_logistic_regression_instrumental_variables.py`](examples/regression/generalized_linear_models/example_logistic_regression_instrumental_variables.py)
- [`examples/regression/generalized_linear_models/example_poisson_regression_instrumental_variables.py`](examples/regression/generalized_linear_models/example_poisson_regression_instrumental_variables.py)
- [`examples/regression/generalized_linear_models/example_gam_regression.py`](examples/regression/generalized_linear_models/example_gam_regression.py)

**Tip:** For big sparse logistic designs, follow the large-scale example’s pattern (`poly` + `C(group)` + `specification_name` / `debug`).

---

### `generalized_method_of_moments/`

GMM from raw moment functions or formulas; linear and nonlinear IV-GMM; MLE score equations as moments.

#### Entry points

- `gmm` — formula-defined moments.
- `GMM` — raw callable returning `(nobs, n_moments)`.
- `gmm_iv_linear` — `y ~ x | z`.
- `gmm_iv_nonlinear` — nonlinear residual + instrument design.
- `gmm_mle` — score-based GMM.

#### Linear IV-GMM

```python
from kanly.api import gmm_iv_linear, lm, compare_results

fit_gmm = gmm_iv_linear("y ~ x | z", data, cov_type="SANDWICH")
fit_iv = lm("y ~ x | z", data, cov_type="NONROBUST")
print(compare_results([fit_gmm, fit_iv]))
```

#### Methods

`ONE_STEP`, `TWO_STEP`, `ITERATIVE`. Covariance: `SANDWICH`, `CLUSTER`, `BOOTSTRAP`, etc.

*Moment formula syntax: [`kanly/regression/generalized_method_of_moments/README.md`](kanly/regression/generalized_method_of_moments/README.md).*

#### Examples in this repo (`examples/generalized_method_of_moments/`)

- [`example_gmm_linear.py`](examples/generalized_method_of_moments/example_gmm_linear.py), [`example_gmm_linear_instrumental_variables.py`](examples/generalized_method_of_moments/example_gmm_linear_instrumental_variables.py)
- [`example_gmm_nonlinear.py`](examples/generalized_method_of_moments/example_gmm_nonlinear.py), [`example_gmm_logit.py`](examples/generalized_method_of_moments/example_gmm_logit.py)
- [`example_gmm_mle.py`](examples/generalized_method_of_moments/example_gmm_mle.py), [`example_gmm_mle_chi_squared.py`](examples/generalized_method_of_moments/example_gmm_mle_chi_squared.py)

---

### `nonlinear_least_squares/`

Trust-region **nonlinear least squares** from formulas, with **inference** on `{parameters}`: standard errors, *t*-tests, and confidence intervals via `cov_type` (`'hc1'`, cluster, bootstrap, etc.)—not only point estimates. Supports weights, bounds, robust/quantile-style **root losses**, and `nlls_en` for elastic-net regularization.

#### Basic usage

```python
import pandas as pd
from kanly.api import nlls

fit = nlls(
    '[y] ~ {Intercept} + {beta} * exp({gamma} * [x]) $ [w]',
    df,
    start_params={'Intercept': 1.0, 'beta': 3.0, 'gamma': -0.5},
    max_iter=100,
    cov_type='bootstrap',
    cov_kwds={'n_samples': 5_000, 'max_processes': 5},
)
print(fit)
```

`start_params` is the analogue of SciPy’s `x0` (dict keyed by `{parameter}` names or a vector in parameter order). If omitted, the solver defaults to `0` and may run a **subsample** pre-fit first.

Example output (from [`examples/regression/nonlinear_least_squares/example_nonlinear_least_squares_exponential.py`](examples/regression/nonlinear_least_squares/example_nonlinear_least_squares_exponential.py)):

```text
Nonlinear Least Squares Results
Example Exponential

Dep. Variable: y
Weights:                          w    R-squared:                   0.9500
Nobs:                           249    Converged:                     True
Covariance Type:          BOOTSTRAP    Method:                          TR

              coef        std err      t   p>|t| [0.025,    0.975]
Intercept   0.7854  ****   0.1923   4.08  <0.001   0.4066    1.164
beta         3.131  ****   0.2004  15.63  <0.001    2.737    3.526
gamma      -0.4918  ****  0.02419 -20.33  <0.001  -0.5394  -0.4441

formula:  [y] ~ {Intercept} + {beta} * exp({gamma} * [x]) $ [w]
Did 5000 Bayesian bootstrap repetitions, alpha=1.000.
message: Converged: |dF| < ftol * max(1, |F|)
```

#### Conventions

- `[expr]` — data.
- `{param}` — parameters.
- `$ [w]` — weights.
- `C(group)`, `poly`, `cheb`, `polym`, etc.

#### Jacobian modes

`jac_method="analytic"` | `"mid"` | `"fwd"`; optional JIT for analytic Jacobians.

#### Elastic-net NLLS

```python
from kanly.api import nlls_en

fit = nlls_en(
    "[y] ~ {a} + {b} * [x] + [C(group, -1)]",
    df,
    alpha={"b": 0.1},
    l1_ratio=0.5,
    start_params={"a": 0.0, "b": 1.0},
)
```

*Large-scale tips: [`kanly/regression/nonlinear_least_squares/README.md`](kanly/regression/nonlinear_least_squares/README.md).*

#### Examples in this repo (`examples/regression/nonlinear_least_squares/`)

- [`example_nonlinear_least_squares_exponential.py`](examples/regression/nonlinear_least_squares/example_nonlinear_least_squares_exponential.py), [`example_nonlinear_least_squares_logistic.py`](examples/regression/nonlinear_least_squares/example_nonlinear_least_squares_logistic.py)
- [`example_bounded_least_squares.py`](examples/regression/nonlinear_least_squares/example_bounded_least_squares.py), [`example_nonlinear_least_squares_elastic_net.py`](examples/regression/nonlinear_least_squares/example_nonlinear_least_squares_elastic_net.py)
- [`example_nonlinear_least_squares_large_scale.py`](examples/regression/nonlinear_least_squares/example_nonlinear_least_squares_large_scale.py), [`example_nonlinear_least_squares_large_scale_bounded.py`](examples/regression/nonlinear_least_squares/example_nonlinear_least_squares_large_scale_bounded.py)
- [`example_nonlinear_least_squares_exponential_block_bootstrap.py`](examples/regression/nonlinear_least_squares/example_nonlinear_least_squares_exponential_block_bootstrap.py), [`example_nonlinear_least_squares_exponential_block_cluster.py`](examples/regression/nonlinear_least_squares/example_nonlinear_least_squares_exponential_block_cluster.py)
- [`example_nonlinear_least_squares_exponential_quantile_regression.py`](examples/regression/nonlinear_least_squares/example_nonlinear_least_squares_exponential_quantile_regression.py)

---

### `partial_least_squares/`

**Partial least squares** (PLS) finds latent components that maximise the covariance between predictors and response(s). It is useful when predictors are highly collinear or when dimensionality reduction before regression is desired. `PLS1` handles a single response; `PLS2` handles multiple responses simultaneously.

`pls1`, `PLS1`, and `PLS2` are exported from `kanly.api` and implemented under `kanly/regression/partial_least_squares/`. **Detailed guide:** [`kanly/regression/partial_least_squares/README.md`](kanly/regression/partial_least_squares/README.md) and docstrings in [`kanly/regression/partial_least_squares/pls.py`](kanly/regression/partial_least_squares/pls.py).

**Examples:** none under `examples/` at present.

---

## `kanly.time_series`

**See also:** [`kanly/time_series/README.md`](kanly/time_series/README.md)

### What this subpackage does

`kanly.time_series` provides **univariate** time-series tools: sample **ACF** and **PACF**, **SARIMA simulation** (`simulate_sarima`), and **SARIMAX** estimation with optional exogenous regressors (implementation under [`kanly/time_series/sarimax/`](kanly/time_series/sarimax/)).

The **user guide** for all of the above is [`kanly/time_series/README.md`](kanly/time_series/README.md) at the **package root** (it is not in `sarimax/`; that subfolder only holds a short pointer back to the package readme).

**Useful background:** [`ARIMA`](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average), [`autocorrelation`](https://en.wikipedia.org/wiki/Autocorrelation), [`SARIMA`](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average#Extensions). Theory and notation for state-space ARMA/SARIMA align with Brockwell & Davis (2016) — see the references section in the time-series readme.

### Comparison: SciPy / statsmodels / scikit-learn


|                           | SciPy                                                                                                                                                                                                  | statsmodels                                                                                                                                                     | scikit-learn                                                                      | kanly                                                                                  |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **Time-series modelling** | [`signal`](https://docs.scipy.org/doc/scipy/reference/signal.html), [`fft`](https://docs.scipy.org/doc/scipy/reference/fft.html) — useful primitives, **no** unified SARIMAX likelihood / Kalman stack | [`tsa.statespace.SARIMAX`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html), `acf`, `pacf` — broad `tsa` ecosystem | No first-class SARIMAX; not the usual tool for ARIMA                              | `SARIMAX`, `sarimax(formula, data=...)`, `simulate_sarima`, `acf` / `pacf`             |
| **Practical note**        | You compose filters and fits yourself                                                                                                                                                                  | Mature reference; **kanly** aims for similar *spirit*, not guaranteed numerical parity (see [`kanly/time_series/README.md`](kanly/time_series/README.md))       | Often paired with **statsmodels**, **pmdarima**, or **Prophet** for this workload | Formula path for exogenous regressors matches regression-style APIs elsewhere in kanly |


### SARIMAX quick start

```python
import numpy as np
from kanly.api import SARIMAX

res = SARIMAX(
    y,
    exog=x.reshape(-1, 1),
    order=(1, 1, 1),
    seasonal_order=(1, 0, 1, 12),
    trend="c",
)
print(res)
forecast = res.get_forecast(steps=12, exog=np.zeros((12, 1)))
```

### Formula interface

```python
from kanly.api import sarimax

res = sarimax(
    "sales ~ price + promo",
    data=df,
    order=(1, 0, 1),
    seasonal_order=(0, 1, 1, 52),
)
```

### Simulation

```python
from kanly.api import simulate_sarima

y = simulate_sarima(
    n=500, ar=[0.5], ma=[-0.2], d=1,
    sar=[0.3], sma=[], D=0, s=12, sigma2=1.0, seed=0,
)
```

### ACF / PACF

`kanly.api` also exposes `acf` and `pacf` (autocorrelation / partial autocorrelation) from `kanly.time_series`.

*Exhaustive API notes, `simulate_sarima` options, SARIMAX fitting and forecasting, and references: [`kanly/time_series/README.md`](kanly/time_series/README.md) (sections on ACF/PACF, simulation, and SARIMAX).*

### Examples in this repo

- [`examples/time_series/sarimax/example_arma.py`](examples/time_series/sarimax/example_sarimax_with_ar_and_x.py) — `simulate_sarima` + `sarimax`
- [`examples/time_series/example_acf.py`](examples/time_series/example_acf.py) — `acf` / `pacf` on simulated data

---

## `kanly.bayes`

**See also:** [`kanly/bayes/README.md`](kanly/bayes/README.md)

### What this subpackage does

`kanly.bayes` is the **Bayesian** layer: arbitrary log-posteriors (`BayesianModel`), **compiled / string-based** `DataModel` for quick prototyping, MCMC samplers (**Adaptive Metropolis**, coordinate **MALA**, combined workflows), **MAP / MLE**, and shortcuts that wrap frequentist formula modules with priors (`bayes_lm_model`, `bayes_glm_model`, `bayes_nlls_model`). It also includes **conjugate** normal–inverse-gamma linear regression (`blm`).

**Useful background:** [`Bayes' theorem`](https://en.wikipedia.org/wiki/Bayes%27_theorem), [`Markov chain Monte Carlo`](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo), [`Metropolis–Hastings algorithm`](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm).

### Comparison: SciPy / statsmodels / scikit-learn


| Tool             | Overlap                                                                                                                                                                                                           | Difference                                                                                                                          |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **SciPy**        | [`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html) priors / likelihood pieces; [`scipy.optimize`](https://docs.scipy.org/doc/scipy/reference/optimize.html) for **MAP**-style point estimates | **No** adaptive MCMC, `DataModel`, or sampler orchestration — you wire `logpost` + `minimize` or roll your own chain                |
| **statsmodels**  | Mostly **frequentist** inference; some generic likelihood with `GenericLikelihoodModel`                                                                                                                           | No built-in adaptive MCMC stack comparable to `BayesianModel.sample`; priors are not first-class like kanly’s string/`scipy` priors |
| **scikit-learn** | `BayesianRidge` (only a narrow conjugate-like ridge)                                                                                                                                                              | Not a general-purpose MCMC / `DataModel` system — use **PyMC**, **Stan**, or kanly for full custom posteriors                       |
| **kanly**        | Custom log-density for parameters given data, plus samplers and formula-wrapped likelihoods                                                                                                                       |                                                                                                                                     |


### Prefer imports from `kanly.api`

Examples: `bmodel`, `DataModel`, `build_data_model`, `amha`, `mala`, `bayes_lm_model`, `bayes_glm_model`, `bayes_nlls_model`, `blm`.

### Minimal custom model

```python
import numpy as np
from kanly.api import bmodel

def log_likelihood(params):
    mu, sigma = params
    resid = y - mu
    return -0.5 * np.sum((resid / sigma) ** 2) - len(y) * np.log(sigma)

model = bmodel(
    log_likelihood,
    param_names=["mu", "sigma"],
    bounds={"sigma": (0.0, np.inf)},
    priors={"mu": "norm(0, 10)"},
)
mcmc_fit = model.sample([0.0, 1.0], n_samples=10_000, n_burnin=2_000)
```

### `DataModel`: data block, model block, then MCMC

A [`DataModel`](kanly/bayes/data_model.py) pairs two **strings** of Python code: a **data block** (pull columns from the dict/DataFrame into `self` via backticks) and a **model block** that returns a log-posterior or log-likelihood using `$param$` placeholders. Build a [`BayesianModel`](kanly/bayes/bayesian_model.py) with `to_bayesian_model`, then call `sample` or `amha`.

**WLS regression + MCMC** ([`examples/bayes/example_data_model.py`](examples/bayes/example_data_model.py)):

```python
from kanly.api import DataModel
import numpy as np

data = {'x': x, 'y': y, 'wts': wts, 'g': g, 'z': z}

data_string = """
self.x = `x`
self.z = `z`
self.y = `y`
self.weights = `wts`
self.g = `C(g)`
self.root_weights = np.sqrt(self.weights)
"""

model_string = """
pred = $Intercept$ + $x$ * x + $_dummy[g,-1]$ + $_poly[z,2]$
resid = y - pred
return logpdf_norm(resid, loc=0.0, scale=$sigma$ / root_weights).sum()
"""

dataobj = DataModel.build_data_model(data_string, model_string, data, nopython=False)
model = dataobj.to_bayesian_model(priors={'x': 'norm(-23.6, 1.33)'}, bounds={'x': [0, 10]})
fit = model.sample(1.2 * np.ones(8), n_samples=10_000, n_burnin=3_000, n_chains=6, ...)
print(fit)
```

Example output (from the same file):

```text
MCMC Results
Method:             AMH
No. Chains:                     6    Samples:                10000
Maximum Log Posterior:     -5.7470e+02

              mean     std      MCSE    median   [.05,    .95]     ESS   R_hat
Intercept    1.506   1.996   0.02649     1.432   -1.728  5.117  5678.0  1.0019
x            6.343   4.287     0.206     9.184  0.03976  9.704   433.0  1.1810
C(g)[1]       2.07   2.663   0.03669     1.848   -1.878  7.248  5267.0  1.0035
... (other fixed effects and sigma) ...
Some R_hat are above 1.01, the MCMC chains have not converged!
```

**Custom likelihood (beta)** ([`examples/bayes/example_data_model_fit_beta.py`](examples/bayes/example_data_model_fit_beta.py)):

```python
from kanly.api import DataModel

data_string = """
self.x = `x`
self.y = `y`
"""

model_string = """
return nopython_logpdf_beta(x, a=$a$, b=$b$).sum()
"""

dataobj = DataModel.build_data_model(data_string, model_string, data, nopython=True)
model = dataobj.to_bayesian_model(bounds={'a': [0, np.inf], 'b': [0, np.inf]})
fit = model.amha([1., 1], n_samples=20_000, n_burnin=10_000, do_parallel=True, ...)
print(fit)
```

Example output (from the same file):

```text
MCMC Results
Method:             AMH
No. Chains:                     4    Samples:                20000
Maximum Log Posterior:     7.1045e+02

    mean      std       MCSE median [.05,    .95]     ESS   R_hat
a  4.961   0.1807   0.001934  4.958  4.673  5.265  8733.0  1.0002
b  2.051  0.06951  0.0007354  2.049  1.938  2.168  8935.0  1.0002
```

### Conjugate Bayesian LM

```python
import numpy as np
from kanly.api import blm

fit = blm(
    "y ~ x + z $ wts",
    data,
    Lambda0=np.diag([0, 3.3 * np.sum(data["wts"]), 0]),
    mu0=np.zeros(3),
    a0=0,
    b0=0,
)
```

The submodule readme documents the `DataModel` backticks / `$parameter$` DSL, spline expansions, sampler tuning, bounded transforms, and pitfalls (experimental folders, `__sigma2` in regression wrappers).

*Full guide: [`kanly/bayes/README.md`](kanly/bayes/README.md).*

### Examples in this repo (`examples/bayes/`)

- [`example_data_model.py`](examples/bayes/example_data_model.py), [`example_data_model_fit_beta.py`](examples/bayes/example_data_model_fit_beta.py)
- [`example_bayesian_linear_regression.py`](examples/bayes/example_bayesian_linear_regression.py), [`example_bayesian_linear_regression_non_conjugate.py`](examples/bayes/example_bayesian_linear_regression_non_conjugate.py), [`example_bayesian_linear_regression_mcmc.py`](examples/bayes/example_bayesian_linear_regression_mcmc.py)
- [`example_bayesian_linear_model_4_ways.py`](examples/bayes/example_bayesian_linear_model_4_ways.py), [`example_bayesian_linear_regression_model_conjugate.py`](examples/bayes/example_bayesian_linear_regression_model_conjugate.py)
- [`example_bayesian_logistic_model.py`](examples/bayes/example_bayesian_logistic_model.py), [`example_bayesian_elastic_net.py`](examples/bayes/example_bayesian_elastic_net.py)
- [`example_mala.py`](examples/bayes/example_mala.py), [`example_bayes_ols_mcmc.py`](examples/bayes/example_bayes_ols_mcmc.py)

---

## `kanly.stats`

**See also:** [`kanly/stats/README.md`](kanly/stats/README.md)

### What this subpackage does

`kanly.stats` has two roles: **(1)** fast **density helpers** that mirror [`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html) parameterisations but skip heavy validation (for MCMC, MLE inner loops, and `DataModel` code generation), and **(2)** `**StatisticalTests*`* — classical post-estimation tools ([`delta method`](https://en.wikipedia.org/wiki/Delta_method), simulation from a multivariate-normal limit, [`Fieller intervals`](https://en.wikipedia.org/wiki/Fieller%27s_theorem), [`Wald tests`](https://en.wikipedia.org/wiki/Wald_test)).

Detailed documentation: `**[kanly/stats/README.md](kanly/stats/README.md)**` (covers `statistical_tests.py`, `nopython_logpdf.py`, `nopython_frozen_logpdf.py`, `nopython_scipy_special.py`; **not** the MLE / normal-array helpers).

### Imports

```python
# Density helpers (also re-exported from kanly.api)
from kanly.api import logpdf_norm, pdf_norm, nopython_gammaln

# Inference helpers (submodule only)
from kanly.stats.statistical_tests import StatisticalTests

# Frozen log-pdf factories (priors, fixed hyperparameters)
from kanly.stats.distributions.nopython_frozen_logpdf import get_frozen_logpdf_norm

logp = get_frozen_logpdf_norm(loc=0.0, scale=1.0, nopython=False)
print(logp(0.0))
```

### Comparison: SciPy / statsmodels / scikit-learn / kanly


| Task                   | SciPy                                                      | statsmodels                    | scikit-learn | kanly                                                                                |
| ---------------------- | ---------------------------------------------------------- | ------------------------------ | ------------ | ------------------------------------------------------------------------------------ |
| `logpdf` / `pdf`       | `scipy.stats.<dist>.logpdf(x, *params)` frozen or unfrozen | Usually inside model `loglike` | —            | Free functions `logpdf_*` / `pdf_*`; optional **Numba**; **no** distribution objects |
| Wald / Fieller / delta | —                                                          | `results.wald_test(r_matrix)`  | —            | `StatisticalTests` (submodule import) with explicit arrays                           |
| Fast frozen prior      | `scipy.stats.norm(loc, scale).logpdf`                      | —                              | —            | `get_frozen_logpdf_`* + optional **Numba** callables                                 |


**sklearn** does not expose Wald / Fieller / delta-method utilities; use **statsmodels** results objects, **SciPy** for densities, or **kanly**’s `StatisticalTests`.

---

## `kanly.nonparametric`

**See also:** [`kanly/nonparametric/README.md`](kanly/nonparametric/README.md)

### What this subpackage does

`kanly.nonparametric` implements **smoothing and density estimation** without a full parametric likelihood: [`kernel density estimation`](https://en.wikipedia.org/wiki/Kernel_density_estimation) (FFT or direct), [`LOWESS / LOESS`](https://en.wikipedia.org/wiki/Local_regression), STL / MSTL-style seasonal smoothing, Gaussian kernel regression, and **piecewise** splines / interpolators with analytic derivatives.

### Comparison: SciPy / statsmodels / scikit-learn


| Method                  | SciPy                                                                                                                                            | statsmodels                                                            | scikit-learn                                     | kanly                                                                                   |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------- | ------------------------------------------------ | --------------------------------------------------------------------------------------- |
| KDE                     | [`gaussian_kde`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html) (Gaussian kernel; Silverman’s rule default) | `statsmodels.nonparametric.KDEUnivariate`, `KDEMultivariate`           | `KernelDensity` (multiple kernels via `kernel=`) | `kde` with FFT path, explicit kernel name strings, power-of-two grid constraints        |
| LOWESS                  | — (not in SciPy)                                                                                                                                 | `statsmodels.nonparametric.lowess`                                     | —                                                | `LOWESS` / `lowess` with Numba path                                                     |
| Splines / interpolation | [`interpolate`](https://docs.scipy.org/doc/scipy/reference/interpolate.html) (`CubicSpline`, `UnivariateSpline`, `interp1d`, …)                  | Some formula / smoothing support                                       | Limited vs general `scipy.interpolate`           | `cubic_spline`, `linear_spline`, `interp` with **derivative** hooks on returned objects |
| STL / seasonal split    | — (use `signal` or third-party for ad hoc filters)                                                                                               | Seasonal decomposition utilities in `tsa` (API differs from kanly STL) | —                                                | `stl`, `mstl`; LOESS-driven STL workflow                                                |


### Modules


| Module                      | Role                                                  |
| --------------------------- | ----------------------------------------------------- |
| `lowess.py`                 | LOWESS / LOESS (Numba-accelerated)                    |
| `kde.py`                    | KDE — FFT and direct, multiple kernels                |
| `stl.py`                    | STL seasonal-trend decomposition                      |
| `mstl.py`                   | MSTL (multiple seasonalities) — `mstl` on `kanly.api` |
| `gaussian_kernel_smooth.py` | Gaussian kernel regression                            |
| `interpolate.py`            | Splines / step interpolators with derivatives         |


### KDE

```python
import numpy as np
from kanly.api import kde

support, density = kde(data)
support, density = kde(data, kernel='epa', bw='scott')
f = kde(data, return_arrays=False)
print(f(0.0))
```

Kernels include `'gau'`, `'epa'`, `'uni'`, `'tri'`, `'biw'`, `'triw'`, `'cos'`, `'tric'`. The default FFT path requires `gridsize` to be a power of two.

### LOWESS

```python
from kanly.api import LOWESS

x_smooth, y_smooth = LOWESS(y, x, frac=0.3, it=1, degree=1)
```

### STL / MSTL

```python
from kanly.api import stl

trend, seasonality, resid = stl(y, period=12)
```

For multiple seasonal periods, `kanly.api` also exposes `mstl` (see docstring / implementation under `kanly/nonparametric/`).

### Splines / interpolation

```python
from kanly.api import cubic_spline, linear_spline, interp

f = cubic_spline(x, y)
print(f.derivative(2.0))
g = interp(x, y, kind='nearest')
```

*Pitfalls (FFT grid, STL missing data, LOWESS JIT): [`kanly/nonparametric/README.md`](kanly/nonparametric/README.md).*

### Examples in this repo

- No dedicated `examples/nonparametric/` tree; see [`lowess_example.py`](lowess_example.py) and [`testing/testing_script_lowess.py`](testing/testing_script_lowess.py) (referenced in the nonparametric readme), plus tests under `tests/`.

---

## `kanly.optimize`

**See also:** [`kanly/optimize/README.md`](kanly/optimize/README.md)

### What this subpackage does

`kanly.optimize` provides **box-constrained** black-box optimizers: **projected BFGS quasi-Newton** (`bfgs_pqn`) with line search and optional analytic gradient/Hessian, and **bounded coordinate descent** (`cdb`). Results share a common [`OptimizationResult`](kanly/optimize/optimization_results.py) base (subclasses add Hessian approximations, binding-bound masks, etc.).

**Detailed guide:** [`kanly/optimize/README.md`](kanly/optimize/README.md) (documents only top-level `.py` files in that folder, not `__TO_DELETE/`).

### Quick usage

**BFGS with a string objective** (full script: [`examples/optimize/example_bfgs.py`](examples/optimize/example_bfgs.py)):

```python
from kanly.api import bfgs_pqn, func_str_to_callable
from kanly.utils.dict_2_array import dict_2_array

func = func_str_to_callable("({x}-1)**2 + ({y}+2)**2")
x0 = dict_2_array({"x": 0, "y": 2}, func.param_names)
result = bfgs_pqn(func, x0=x0, maxiter=10, maximize=False, ftol=1e-12, gtol=1e-12)
print(result.x, result.ferr, result.gnorm)
```

**Arbitrary callables and box constraints** — pass any `fun(x)` and optional **bounds** shaped **2 × p** (lower row, upper row). **Coordinate descent** uses the same vector API:

```python
import numpy as np
from kanly.api import bfgs_pqn, cdb

def rosen(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

x0 = np.array([-1.0, 1.0])
bounds = np.array([[-2.0, -2.0], [2.0, 2.0]])

res_qn = bfgs_pqn(rosen, x0, bounds=bounds, maxiter=500)
res_cd = cdb(rosen, x0, bounds=bounds, maxiter=2000)
```

`func_str_to_callable` is implemented in **kanly.automatic_differentiation** and re-exported from **kanly.api** for convenience.

### Comparison: SciPy


|                                     | SciPy                                                                                                                                          | kanly                                                                                        |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Box-constrained smooth problems** | [`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) (`L-BFGS-B`, `TNC`, `SLSQP`, …) | `bfgs_pqn` — in-house projected BFGS + Wolfe-style line search; same **2 × p** bounds layout |
| **Coordinate descent**              | No direct analogue in `minimize`                                                                                                               | `cdb` sweeps coordinates with feasible finite differences                                    |


SciPy covers more method families (trust-region, global, general constraints); **kanly**’s solvers are tailored for internal use (MAP, nonlinear fitting) with a small, consistent API on `numpy` vectors.

### Examples in this repo

- [`examples/optimize/example_bfgs.py`](examples/optimize/example_bfgs.py) — `func_str_to_callable`, `dict_2_array`, `bfgs_pqn`, diagnostics (`ferr`, `gnorm`).

---

## `kanly.automatic_differentiation`

**See also:** [`kanly/automatic_differentiation/README.md`](kanly/automatic_differentiation/README.md)

### What this subpackage does

**kanly.automatic_differentiation** turns **formula strings** (with `{param}` or custom delimiter pairs) into callables and, for supported expressions, **symbolically** derives gradients, Jacobians, and Hessians by building an expression graph and generating Numba-optional Python via `exec`.

**Detailed guide:** [`kanly/automatic_differentiation/README.md`](kanly/automatic_differentiation/README.md).

### Quick usage

```python
from kanly.api import func_str_to_callable

f = func_str_to_callable("({a}-1)**2 + ({b}+2)**2")
partials, info = f.get_analytical_partial_derivatives(return_info=True)
```

For optimization with named dict initial values, pair with `dict_2_array` (see [`examples/optimize/example_bfgs.py`](examples/optimize/example_bfgs.py)). For delimiter customization and `return_info` inspection, see [`examples/autodiff/example_autodiff.py`](examples/autodiff/example_autodiff.py).

### Comparison: JAX / PyTorch


|                 | JAX / PyTorch              | kanly autodiff                                                   |
| --------------- | -------------------------- | ---------------------------------------------------------------- |
| **Modeling**    | Python functions / modules | **Strings** rewritten to `params[i]` + optional `other_args`     |
| **Derivatives** | Trace / reverse-mode AD    | **Symbolic** rules + generated source (subset of numpy-like ops) |


---

## Other modules (supporting)

These live under `kanly/` and are re-exported from `kanly.api` for convenience. Paths listed here **exclude** packages that have their own section or readme above (e.g. **kanly.optimize**, **kanly.stats**, **kanly.automatic_differentiation**).


| Path                    | Role                                                                                         | Notable `kanly.api` symbols                                                                                                  |
| ----------------------- | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `kanly/bootstrap/`      | Bayesian & classical bootstrap; block/cluster resampling; joint covariances; Ray parallelism | `get_joint_bootstrapped_distribution` — [`kanly/bootstrap/README.md`](kanly/bootstrap/README.md)                             |
| `kanly/plot/`           | ASCII line / scatter / histogram plots                                                       | `plot`, `scatter`, `hist` — [`kanly/plot/README.md`](kanly/plot/README.md)                                                   |
| `kanly/utils/`          | Tables, HDI, timers, comparing fits                                                          | `compare_results`, `get_highest_density_interval`, `latex_table`, `timer` — [`kanly/utils/README.md`](kanly/utils/README.md) |
| `kanly/general_models/` | Generic callable fitting                                                                     | `fit_general_model_callable`                                                                                                 |
| `kanly/dill_object.py`  | Serialization helpers                                                                        | `read`, `save`                                                                                                               |


**Distributions / likelihood fragments:** see the **kanly.stats** section above and [`kanly/stats/README.md`](kanly/stats/README.md) (`kanly.api` also exposes plotting-oriented helpers such as `get_mle_x_y` from `fit_distributions_mle` — documented in source, not the stats readme).

**Autodiff:** covered under **kanly.automatic_differentiation** above; [`examples/autodiff/example_autodiff.py`](examples/autodiff/example_autodiff.py) complements that section.

---

## License

If the repository contains a `LICENSE` file, refer to it for terms. (Not duplicated here.)
