# `kanly.stats`

**See also:** [kanly README](../../README.md) · [bayes](../bayes/README.md) (uses `logpdf_*` in `DataModel`)

The **`kanly.stats`** package supplies (1) reusable **classical inference helpers** for nonlinear transformations of estimated parameters, and (2) **fast, Numba-friendly** PDF / log-PDF building blocks aligned with [`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html) parameterisations.

This readme covers:

- [`statistical_tests.py`](statistical_tests.py) — `StatisticalTests` static methods.
- [`distributions/nopython_logpdf.py`](distributions/nopython_logpdf.py) — explicit `logpdf_*` / `pdf_*` (parameters vary per call; suitable when the density parameters are **unknown** and part of the objective, e.g. MCMC or MLE).
- [`distributions/nopython_frozen_logpdf.py`](distributions/nopython_frozen_logpdf.py) — `get_frozen_logpdf_*` factories returning callables with **fixed** parameters (typical for **priors** and fixed hyperparameters).
- [`distributions/nopython_scipy_special.py`](distributions/nopython_scipy_special.py) — `gammaln`, `betaln`, `gamma`, `beta`, `erf`, `erfc`, `ndtr`, etc., for use inside Numba [`@njit`](https://numba.readthedocs.io/en/stable/user/jit.html) kernels.

It does **not** document [`distributions/fit_distributions_mle.py`](distributions/fit_distributions_mle.py) or [`distributions/get_normal_arrays.py`](distributions/get_normal_arrays.py) here (per package docs scope).

Background reading: [probability density function](https://en.wikipedia.org/wiki/Probability_density_function), [log-likelihood](https://en.wikipedia.org/wiki/Likelihood_function), [delta method](https://en.wikipedia.org/wiki/Delta_method), [Wald test](https://en.wikipedia.org/wiki/Wald_test), [Fieller's theorem](https://en.wikipedia.org/wiki/Fieller%27s_theorem), [central limit theorem](https://en.wikipedia.org/wiki/Central_limit_theorem).

## Imports

```python
# Density helpers (also re-exported from kanly.api)
from kanly.api import logpdf_norm, pdf_norm, gammaln

# Inference helpers (submodule only)
from kanly.stats.statistical_tests import StatisticalTests

# Frozen log-pdf factories (priors, fixed hyperparameters)
from kanly.stats.distributions.nopython_frozen_logpdf import get_frozen_logpdf_norm

logp = get_frozen_logpdf_norm(loc=0.0, scale=1.0, nopython=False)
print(logp(0.0))
```

## Comparison: SciPy / statsmodels / scikit-learn / kanly

| Task | SciPy | statsmodels | scikit-learn | kanly |
| ---- | ----- | ----------- | ------------ | ----- |
| `logpdf` / `pdf` | `scipy.stats.<dist>.logpdf(x, *params)` frozen or unfrozen | Usually inside model `loglike` | — | Free functions `logpdf_*` / `pdf_*`; optional **Numba**; **no** distribution objects |
| Wald / Fieller / delta | — | `results.wald_test(r_matrix)` | — | `StatisticalTests` (submodule import) with explicit arrays |
| Fast frozen prior | `scipy.stats.norm(loc, scale).logpdf` | — | — | `get_frozen_logpdf_*` + optional **Numba** callables |

**sklearn** does not expose Wald / Fieller / delta-method utilities; use **statsmodels** results objects, **SciPy** for densities, or **kanly**’s `StatisticalTests`.

---

## `StatisticalTests` (`statistical_tests.py`)

[`StatisticalTests`](statistical_tests.py) is a namespace of **static methods** for inference on scalar (or vector-to-scalar) functions of an asymptotic multivariate normal estimator.

Import:

```python
from kanly.stats.statistical_tests import StatisticalTests
```

### Comparable functionality

| kanly | statsmodels | Notes |
|-------|---------------|--------|
| `StatisticalTests.wald_test` | `statsmodels.regression.linear_model.RegressionResults.wald_test`, `statsmodels.base.model.GenericLikelihoodModelResults.wald_test` | statsmodels builds **R** and **q** from formula or column names; kanly expects explicit `r_matrix`, `q`, and uses your `cov_params` array. |
| `StatisticalTests.delta_method_test` | Often hand-rolled; related to [`statsmodels.stats.Oaxaca_Blinder`](https://www.statsmodels.org/stable/generated/statsmodels.stats.oaxaca.OaxacaBlinder.html) style decomposition in spirit only | kanly uses **central finite differences** for the gradient of `func(mean)`. |
| `StatisticalTests.clt_simulation_test` / `simulate_from_clt` | Bootstrapping / simulation in various `statsmodels` modules | kanly draws **`mean + MVN(0, cov)`** and applies `func`; not the same as residual bootstrap. |

**scikit-learn** does not provide Wald / Fieller / delta-method utilities; use **scipy.stats** for distributions or **statsmodels** for model-based tests.

### Methods

#### `sparse_to_dense(*args)`

Converts `scipy.sparse.spmatrix` inputs to dense `numpy.ndarray` (other inputs are coerced with `np.array`). Use when passing covariance matrices that may be sparse.

#### `simulate_from_clt(func, mean, cov, n_trials=100_000, seed=0)`

Draws `n_trials` samples from **MVN(mean, cov)**, applies scalar `func` row-wise, returns a 1-D array of simulated `func(θ)` values. Useful for visualising the sampling distribution of a nonlinear transformation under [asymptotic normality](https://en.wikipedia.org/wiki/Asymptotic_distribution).

#### `delta_method_test(func, mean, cov, null_hypothesis=0, step=.002, test_level=.05)`

[Delta-method](https://en.wikipedia.org/wiki/Delta_method) standard error for `func(mean)` via numerical gradient; returns `estimate`, `ci_lo` / `ci_hi`, `pvalue`, `test_stat`, `std_err`.

#### `clt_simulation_test(..., tail='two'|'one', include_samples=False)`

Empirical CI and p-value using `simulate_from_clt`. `tail` controls one- vs two-sided **empirical** p-values based on how simulated mass relates to the null.

#### `ratio_fieller_test(params, cov_params, top, bottom, null_hypothesis=0, test_level=.05, top_constant=0, bottom_constant=0)`

[Fieller interval](https://en.wikipedia.org/wiki/Fieller%27s_theorem) for a ratio of linear forms in `params`, plus a Wald-style p-value. If the Fieller discriminant is negative, the interval is **unbounded** (returned as `(-inf, inf)`).

#### `wald_test(params, cov_params, degrees_freedom, r_matrix=None, q=None, use_f=False)`

Quadratic form **(Rβ−q)′(RVR′)⁻¹(Rβ−q)**. Default `r_matrix=I`, `q=0` tests all coefficients jointly. `use_f=True` uses an **F** distribution (with your `degrees_freedom` as denominator df); otherwise **chi-square**.

---

## Nopython distributions

### Design: frozen vs non-frozen

- **`logpdf_*` / `pdf_*` in `nopython_logpdf.py`** — Parameters (`loc`, `scale`, shape, etc.) are **arguments every call**. Use when parameters change across optimisation / MCMC iterations. Matching **full** densities (not “up to a constant”) is intended for inference on those parameters.
- **`get_frozen_logpdf_*` in `nopython_frozen_logpdf.py`** — Returns a **callable** `f(x)` with parameters **fixed at construction**. Docstring notes these can be correct **up to an additive constant** for speed; intended heavily for **priors** where only relative density matters. Pass `nopython=True` to obtain a Numba-jit-friendly closure where supported.

See module docstrings in [`nopython_logpdf.py`](distributions/nopython_logpdf.py) and [`nopython_frozen_logpdf.py`](distributions/nopython_frozen_logpdf.py).

### Distribution reference table

SciPy-style names; **`logpdf_*` / `pdf_*`** live in [`nopython_logpdf.py`](distributions/nopython_logpdf.py). **`get_frozen_logpdf_*`** factories are in [`nopython_frozen_logpdf.py`](distributions/nopython_frozen_logpdf.py). For **`logpdf_*` / `pdf_*`**, **`x`** is always the first argument (scalar or array); remaining arguments are in the table. For **`get_frozen_*`**, the table lists arguments **before** the optional **`nopython`** flag (default `False`).

For **`logpdf_multivariate_normal` / `pdf_multivariate_normal`**, supply **`cov`** (covariance) or **`tau`** (precision), not both. For **`logpdf_multivariate_t` / `pdf_multivariate_t`**, use **`shape`** (dispersion matrix) or **`tau`**.

| Distribution | Suffix | `logpdf_*` / `pdf_*` args (after `x`) | `get_frozen_logpdf_*` args (before `nopython`) |
|--------------|--------|----------------------------------------|------------------------------------------------|
| [Beta](https://en.wikipedia.org/wiki/Beta_distribution) | `beta` | `a`, `b`, `loc=0.`, `scale=1.` | `a`, `b`, `loc=0.0`, `scale=1.0` |
| [Cauchy](https://en.wikipedia.org/wiki/Cauchy_distribution) | `cauchy` | `loc=0.`, `scale=1.` | `loc=0.0`, `scale=1.0` |
| [Chi-squared](https://en.wikipedia.org/wiki/Chi-squared_distribution) | `chi2` | `df`, `loc=0.`, `scale=1.` | `df`, `loc=0.0`, `scale=1.0` |
| [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution) | `dirichlet` | — | `alpha` |
| [Exponential](https://en.wikipedia.org/wiki/Exponential_distribution) | `expon` | `loc=0.`, `scale=1.` | `loc=0.0`, `scale=1.0` |
| [F](https://en.wikipedia.org/wiki/F-distribution) | `f` | `dfn`, `dfd`, `loc=0.0`, `scale=0.0` | `dfn`, `dfd`, `loc=0.0`, `scale=1.0` |
| [Gamma](https://en.wikipedia.org/wiki/Gamma_distribution) | `gamma` | `a`, `loc=0.`, `scale=1.` | `a`, `loc=0.0`, `scale=1.0` |
| [Generalized extreme value](https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution) | `genextreme` | `c`, `loc=0.`, `scale=1.` | `c`, `loc=0.0`, `scale=1.0` |
| [Generalized normal](https://en.wikipedia.org/wiki/Generalized_normal_distribution) / exponential power | `gennorm` | — | `beta`, `loc=0.0`, `scale=1.0` |
| [Half-Cauchy](https://en.wikipedia.org/wiki/Cauchy_distribution#Related_distributions) | `halfcauchy` | `loc=0.`, `scale=1.` | `loc=0.0`, `scale=1.0` |
| [Half-normal](https://en.wikipedia.org/wiki/Half-normal_distribution) | `halfnorm` | `loc=0.`, `scale=1.` | `loc=0.0`, `scale=1.0` |
| [Inverse gamma](https://en.wikipedia.org/wiki/Inverse-gamma_distribution) | `invgamma` | `a`, `loc=0.`, `scale=1.` | `a`, `loc=0.0`, `scale=1.0` |
| [Laplace](https://en.wikipedia.org/wiki/Laplace_distribution) | `laplace` | `loc=0.`, `scale=1.` | `loc=0.0`, `scale=1.0` |
| [Logistic](https://en.wikipedia.org/wiki/Logistic_distribution) | `logistic` | `loc=0.`, `scale=1.` | `loc=0.0`, `scale=1.0` |
| [Log-normal](https://en.wikipedia.org/wiki/Log-normal_distribution) | `lognorm` | `s`, `loc=0.`, `scale=1.` | `s`, `loc=0.0`, `scale=1.0` |
| [Multivariate normal](https://en.wikipedia.org/wiki/Multivariate_normal_distribution) | `multivariate_normal` | `mean=None`, `cov=None`, `tau=None` | `mean`, `cov` |
| [Multivariate t](https://en.wikipedia.org/wiki/Multivariate_t-distribution) | `multivariate_t` | `df`, `mean=None`, `shape=None`, `tau=None` | `loc`, `shape`, `df=1.0` |
| [Normal](https://en.wikipedia.org/wiki/Normal_distribution) | `norm` | `loc=0.`, `scale=1.` | `loc=0.0`, `scale=1.0` |
| [Pareto](https://en.wikipedia.org/wiki/Pareto_distribution) | `pareto` | `b`, `loc=0.0`, `scale=1.` | `b`, `loc=0.0`, `scale=1.0` |
| [Student t](https://en.wikipedia.org/wiki/Student%27s_t-distribution) | `t` | `df`, `loc=0.`, `scale=1.` | `df`, `loc=0.0`, `scale=1.0` |
| [Truncated normal](https://en.wikipedia.org/wiki/Truncated_normal_distribution) | `truncnorm` | `a`, `b`, `loc=0.`, `scale=1.` | `a`, `b`, `loc=0`, `scale=1` |
| [Weibull](https://en.wikipedia.org/wiki/Weibull_distribution) (`weibull_min`) | `weibull_min` | `c`, `loc=0.`, `scale=1.` | `c`, `loc=0.0`, `scale=1.0` |
| [Uniform](https://en.wikipedia.org/wiki/Continuous_uniform_distribution) on `[loc, loc+scale]` | `uniform` | — | `loc`, `scale` |
| Flat / uniform prior helper (`flat` class) | `flat` | — | `a`, `b` |
| Log-uniform ([reciprocal](https://en.wikipedia.org/wiki/Reciprocal_distribution)) | `loguniform` | — | `a`, `b`, `loc=0.0`, `scale=1.0` |

Callable names: **`logpdf_{suffix}`**, **`pdf_{suffix}`**, **`get_frozen_logpdf_{suffix}`**. Defaults above match the source; **`logpdf_f`** / **`pdf_f`** use **`scale=0.0`** by default (see [`nopython_logpdf.py`](distributions/nopython_logpdf.py)). Frozen factories may assert on inputs (e.g. `scale > 0`, `a < b` for truncated normal); see [`nopython_frozen_logpdf.py`](distributions/nopython_frozen_logpdf.py).

### Comparable functionality: SciPy / statsmodels

| kanly | SciPy | Syntax / behaviour |
|-------|--------|-------------------|
| `logpdf_norm(x, loc=0., scale=1.)` | `scipy.stats.norm.logpdf(x, loc, scale)` | Same **loc/scale** convention; kanly skips domain checks for speed. |
| `get_frozen_logpdf_norm(loc, scale, nopython=False)` | `scipy.stats.norm(loc, scale).logpdf` | kanly returns a **dedicated** callable; SciPy uses frozen distribution objects. |
| Multivariate | `scipy.stats.multivariate_normal.logpdf` | kanly exposes `logpdf_multivariate_normal` / `logpdf_multivariate_t` with optional precision `tau` vs cov `cov`; verify argument names when porting formulas. |

**statsmodels** generally evaluates log-likelihoods **inside** specific models (e.g. GLM, discrete), not as free-standing `logpdf_*` functions; kanly’s functions are closer to **SciPy** primitives.

### `nopython_scipy_special.py`

Wraps `scipy.special` functions with [**Numba overloads**](https://numba.readthedocs.io/en/stable/extending/overloads-guide.html) so they compile inside `njit` blocks. Prefer these over raw `scipy.special` calls inside Numba kernels.  Supported functions are `gammaln`, `betaln`, `gamma`, `beta`, `erf`, `erfc`, `ndtr`.

### Safety

These routines **do not** replicate SciPy’s input validation. Invalid parameters or `x` outside support can yield `nan`, `inf`, or garbage — same trade-off as many handwritten likelihood fragments.

---

## Exports via `kanly.api`

Many `logpdf_*`, `pdf_*`, and `nopython_*` symbols are re-exported from **`kanly.api`** for convenience (`StatisticalTests` is **not** currently on `kanly.api`; import from `kanly.stats.statistical_tests`).

Frozen factories such as `get_frozen_logpdf_norm` live in **`kanly.stats`** submodules; use `from kanly.stats.distributions.nopython_frozen_logpdf import get_frozen_logpdf_norm` or rely on code-generation import strings bundled with `DataModel` (see [`kanly/stats/__init__.py`](__init__.py) `IMPORT_STR`).

---

## Examples

- Bayesian `DataModel` code blocks often concatenate `IMPORT_STR` from [`kanly/stats/__init__.py`](__init__.py) so `logpdf_norm`, `get_frozen_logpdf_beta`, etc. appear in the generated namespace — see [`examples/bayes/example_data_model.py`](../../examples/bayes/example_data_model.py).
