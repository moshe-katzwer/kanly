# Generalized Linear Models User Guide

**See also:** [kanly README](../../../README.md) · [regression overview](../README.md) · [formula](../../formula/README.md)

This package fits sparse generalized linear models (GLMs) with canonical or
custom links, variance weights, optional instruments, robust covariance,
elastic-net style regularization, **marginal effects** on the fitted mean
(see [Marginal effects](#marginal-effects)), and **generalized additive models (GAM)**
via ``gam`` (penalized B-spline smooths — see [GAM section](#generalized-additive-models-gam)).

For a runnable tour that includes Poisson GLM vs GLM-IV (residual inclusion), see the repository-root notebook
[`example_quick_start.ipynb`](../../../example_quick_start.ipynb).

## Mathematical Setup

A GLM specifies the conditional mean through a link function:

```text
eta_i = x_i' beta
g(mu_i) = eta_i
mu_i = E[y_i | x_i] = g^{-1}(eta_i)
```

The response distribution is represented in exponential-dispersion form:

```text
log L_i = (y_i * theta_i - b(theta_i)) / phi_i + c(y_i, phi_i)
mu_i = b'(theta_i)
Var(y_i | x_i) = phi_i * V(mu_i)
```

Here `theta_i` is the canonical parameter, `b(theta)` is the cumulant
function, `V(mu)` is the variance function, and `phi_i` is the scale/dispersion
adjusted by variance weights when supplied.

IRLS solves the local weighted least-squares problem implied by the current
mean and link derivative.  For non-canonical links the working weights are:

```text
w_i = 1 / (g'(mu_i)^2 * V(mu_i))
```

For canonical links this simplifies because the link and family match.  The
penalized objective used when `alpha > 0` adds an elastic-net term:

```text
- mean(log L_i) + alpha * [l1_ratio * ||beta||_1
                         + (1 - l1_ratio) / 2 * ||beta||_2^2]
```

### IRLS and line search

Each outer iteration is a **weighted least-squares (WLS) solve** — the expensive
part of IRLS.  After that update, kanly evaluates the penalized negative
log-likelihood.  If the objective did not improve, it **backs off** along the
segment from the previous iterate toward the new coefficients (halving the step
by default, up to 10 tries).  Each backoff only recomputes the likelihood on
interpolated `(params, intercept, linear predictor)` — **not** another WLS
factorization — so the extra cost is usually small compared with a full IRLS step.

That fallback line search is on by default (`line_search_fallback=True` on
`glm` / `GLMModel.fit`) and can materially reduce the number of outer iterations
relative to accepting every WLS step as-is (as in many **statsmodels** GLM fits),
especially with non-canonical links or stiff starting values.

## Basic Formula API

Use `kanly.api.glm` for formula-based models:

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

Formula parsing is handled by the shared sparse formula utilities. Common terms
include:

- `x + z` for numeric regressors.
- `poly(x, 2)` for polynomial terms.
- `C(group)` for categorical/fixed-effect expansions.
- `y ~ x | z` for instrumental-variable style formulas.

## Families and Links

Families can be supplied by name, class, or instance. String names are
case-insensitive and underscores are optional, so `"negative_binomial"` and
`"negativebinomial"` both resolve to the same family. Negative binomial also
supports an overdispersion parameter in the string form, for example
`"negative_binomial(0.5)"`.

### Registered Families

The `Default link` is used when `link=None`. The `Canonical link` is the
exponential-family canonical link. `Legal links` are the link classes returned by
each family's `safe_links()` and accepted by the model validation layer.

| Family | Response support / role | Dispersion | Default link | Canonical link | Legal links |
|--------|--------------------------|------------|--------------|----------------|-------------|
| `binomial` | Outcomes in `[0, 1]`; binary or fractional proportions | Fixed | `logit` | `logit` | `probit`, `logit`, `cloglog`, `identity`, `cauchy` |
| `bernoulli` | Alias-style subclass of `binomial` for binary outcomes | Fixed | `logit` | `logit` | `probit`, `logit`, `cloglog`, `identity`, `cauchy` |
| `poisson` | Non-negative counts | Fixed (`scale = 1`) | `log` | `log` | `log`, `sqrt`, `identity` |
| `gaussian` | Real-valued continuous outcomes | Estimated | `identity` | `identity` | `log`, `identity` |
| `gamma` | Strictly positive continuous outcomes | Estimated | `negative_inverse` | `negative_inverse` | `log`, `identity`, `inverse`, `negative_inverse` |
| `inverse_gaussian` | Strictly positive continuous outcomes | Estimated | `negative_two_inverse_squared` | `negative_two_inverse_squared` | `inverse_squared`, `negative_two_inverse_squared`, `identity`, `log` |
| `negative_binomial` | Overdispersed non-negative counts; accepts `alpha`, e.g. `"negative_binomial(0.5)"` | Estimated | `log` | `negative_binomial_canonical_link` | `log`, `sqrt`, `identity` |

The negative-binomial canonical link is implemented for family internals, but it
is **not** included in `NegativeBinomial.safe_links()`. In normal user-facing
fits, use the default `log` link or another listed legal link.

### Registered Link Names

| Link name | Link function `g(mu)` | Common role |
|-----------|------------------------|-------------|
| `logit` | `log(mu / (1 - mu))` | Canonical/default for `binomial` and `bernoulli` |
| `probit` | `Phi^{-1}(mu)` | Alternative binary-response link |
| `cloglog` | `log(-log(1 - mu))` | Alternative binary-response link |
| `cauchy` | Cauchy inverse-CDF link | Alternative binary-response link |
| `identity` | `mu` | Canonical/default for `gaussian`; legal for several families |
| `log` | `log(mu)` | Canonical/default for `poisson`; default for `negative_binomial`; legal for positive-mean families |
| `sqrt` | `sqrt(mu)` | Legal for `poisson` and `negative_binomial` |
| `negative_inverse` | `-1 / mu` | Canonical/default for `gamma` |
| `inverse` | `1 / mu` | Legal for `gamma` |
| `negative_two_inverse_squared` | `-1 / (2 * mu**2)` | Canonical/default for `inverse_gaussian` |
| `inverse_squared` | `1 / mu**2` | Legal for `inverse_gaussian` |
| `exponential` | `exp(mu)` | Registered link name; not listed as safe for the registered families above |
| `negative_binomial_canonical_link` | `log(alpha * mu / (1 + alpha * mu))` | Canonical negative-binomial link; implemented but not user-facing safe by default |

The `Power` link class exists in `links.py`, but it is not registered in
`LINK_NAME_2_CLS` and is not listed as safe for any family.

Common examples:

```python
glm("y ~ x", df, family="binomial")   # logistic/probit-style binary models
glm("y ~ x", df, family="poisson")    # count models
glm("y ~ x", df, family="gaussian")   # linear Gaussian GLM
glm("y ~ x", df, family="gamma")      # positive continuous outcomes
```

If `link=None`, the family default link is used. You can supply a safe link by
name, class, or instance:

```python
fit = glm("y ~ x", df, family="binomial", link="probit")
fit = glm("y ~ x", df, family="poisson", link="log")
```

## Array API

Use `SparseGeneralizedLinearModel.GLM` or `kanly.api.GLM` when you already have
arrays:

```python
from kanly.api import GLM

fit = GLM(
    endog=y,
    exog=X,
    family="poisson",
    exog_names=["x1", "x2"],
    endog_name="orders",
)
```

## Weights, Covariance, and Bootstrap

Pass variance weights through formula weights or the array API:

```python
fit = glm("y ~ x $ w", df, family="poisson")
```

Supported covariance types include:

```python
fit_nonrobust = glm("y ~ x", df, family="binomial", cov_type="nonrobust")
fit_hc1 = glm("y ~ x", df, family="binomial", cov_type="hc1")
fit_boot = glm(
    "y ~ x",
    df,
    family="binomial",
    cov_type="bootstrap",
    cov_kwds={"n_samples": 250, "seed": 123},
)
```

Bootstrap covariance refits the GLM on bootstrap-weighted samples and stores the
resulting empirical covariance on the returned result object.

## Marginal Effects

For nonlinear links, a coefficient ``beta_k`` is **not** the change in the fitted
mean ``mu = g^{-1}(X beta)`` when ``x_k`` moves by one unit.  After a GLM fit,
call :meth:`~kanly.regression.generalized_linear_models.regression_results.SparseGLMRegressionResults.get_marginal_effects`
on the result object to obtain **response-scale** effects, in the spirit of
statsmodels
[`GLMResults.get_margeff`](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLMResults.get_margeff.html).

Implementation lives in
[`marginal_effects.py`](marginal_effects.py).  Standard errors use the
**delta method**: ``cov(me) = J @ cov(beta) @ J'``.

```python
from kanly.api import glm

fit = glm("y ~ x1 + x2 + treat", df, family="binomial", link="logit")

me = fit.get_marginal_effects(at="overall", dummy=True)
print(me.summary())           # formatted table (default __str__)
print(me.summary_df())        # pandas DataFrame with dy/dx, SEs, z, p, CI
```

The returned
[`GLMMarginalEffects`](marginal_effects.py) object exposes ``margeff``,
``margeff_se``, ``margeff_cov``, and related fields.

### Evaluation point (`at`)

| Value | Meaning |
|-------|---------|
| ``'overall'`` (default) | Average of observation-level effects over the sample |
| ``'mean'`` | Effect at ``x* =`` column means |
| ``'median'`` | Effect at ``x* =`` column medians (not with dummy detection) |
| ``'all'`` | ``(nobs, nparams)`` matrix of per-observation effects; no SEs |

### Dummy (0/1) regressors

With ``dummy=True`` (default), columns whose entries are all ``0`` or ``1`` are
treated as **discrete** indicators: the reported effect is the average change
``E[g(eta | x_k=1) - g(eta | x_k=0)]`` on the mean scale (secant), not the
tangent ``g'(eta) * beta_k``.  Set ``dummy=False`` to force the continuous
formula for every column.

The internal helper
:func:`~kanly.regression.generalized_linear_models.marginal_effects._get_marginal_effects`
also supports semi-elasticities and elasticities (``effect_type`` in
``{'dydx', 'eydx', 'eyex', 'dyex'}``) and ``dummy_method`` in
``{'secant', 'tangent'}``; the public ``get_marginal_effects`` entry point
currently returns ``dydx`` effects with secant dummies.

> **Note:** Marginal-effect inference assumes the coefficient covariance from
> the GLM fit is appropriate (unpenalized fits with a computed ``cov_params``).
> Penalized GLM estimates are biased; delta-method SEs are not reliable there.

## Regularization

Set `alpha > 0` to use coordinate descent with elastic-net penalties:

```python
fit = glm(
    "y ~ x + z",
    df,
    family="binomial",
    alpha=0.1,
    l1_ratio=0.5,
    normalize=True,
)
```

Parameters:

- `alpha`: overall penalty strength.
- `l1_ratio`: L1 share of the elastic-net penalty.
- `normalize`: whether to scale penalties by predictor standard deviation.
- `penalize_scale`: whether penalties are multiplied by estimated scale.

Penalized estimates are biased; inference is intentionally limited in summaries.

## Instrumental Variables and Residual Inclusion

### Why residual inclusion matters for nonlinear models

For **linear models**, the 2SLS estimator (substitute first-stage fitted values
for the endogenous regressor) and the control-function (residual-inclusion)
approach are numerically equivalent — both yield consistent estimates.

For **nonlinear models** (Poisson, logistic, etc.) the two approaches diverge:

- **IV without residual inclusion**: substituting the first-stage predicted
  ``x̂`` into the nonlinear link is sometimes called the "forbidden regression".
  Because ``E[g⁻¹(β₀ + β₁ x̂)] ≠ E[g⁻¹(β₀ + β₁ x)]`` for non-identity links,
  this estimator is *not* consistent in general even with large samples.

- **IV with residual inclusion (control-function approach)**: the first-stage
  residuals ``v̂ = x − x̂`` (and optional polynomial terms) are appended as
  extra regressors in the outcome equation.  Conditioning on ``v̂`` makes the
  remaining variation in ``x`` exogenous, yielding a consistent estimator.
  A significant coefficient on the residual term also serves as a Hausman-style
  endogeneity test.

### Usage

Use a vertical bar in the formula to supply instruments:

```python
# IV without residual inclusion — formula syntax only; NOT consistent for
# nonlinear families (use as a naive benchmark, not a final estimator)
fit_iv = glm(
    "y ~ x + control | z + control",
    df,
    family="poisson",
    residual_inclusion=False,
    cov_type="bootstrap",
)

# IV with residual inclusion — control-function approach; consistent for
# nonlinear families
fit_iv_ri = glm(
    "y ~ x + control | z + control",
    df,
    family="poisson",
    residual_inclusion=True,
    residual_inclusion_order=1,   # polynomial order of the appended residual; 1 or 2
    cov_type="bootstrap",
)
```

yields

```
==================================================================================
Regression Summary Table
==================================================================================
                                GLM          GLM-IV       GLM-IV-RI   |  Reference
----------------------------------------------------------------------------------
Intercept                     -2.37           0.185           -4.18   |       -4.5
                           (0.0255)         (0.511)         (0.229)   |           


x                              2.69            1.81            2.05   |          2
                           (0.0443)         (0.199)        (0.0741)   |           


x_ri                                                           2.79   |           
                                                           (0.0417)   |           
==================================================================================
```

`residual_inclusion_order` controls the degree of the polynomial appended for
the first-stage residuals.  Order 1 adds a single linear residual term; order 2
adds both a linear and a quadratic term.

Bootstrap covariance (`cov_type='bootstrap'`) is strongly recommended for IV
GLM fits because asymptotic sandwich formulas do not account for first-stage
estimation uncertainty.

### Comparing specifications

```python
from kanly.api import glm, compare_results

# Naive GLM (endogeneity ignored), IV without RI, IV with RI
fit     = glm("y ~ x",      df, family="poisson", cov_type="bootstrap")
fit_iv  = glm("y ~ x | z",  df, family="poisson", residual_inclusion=False,
              cov_type="bootstrap")
fit_ri  = glm("y ~ x | z",  df, family="poisson", residual_inclusion=True,
              residual_inclusion_order=2, cov_type="bootstrap")

print(compare_results(
    fit_list=[fit, fit_iv, fit_ri],
    fit_titles=["GLM", "GLM-IV", "GLM-IV-RI"],
    ref_param_values={"x": 2.0},
))
```

See [`examples/regression/generalized_linear_models/example_poisson_regression_instrumental_variables.py`](../../../../examples/regression/generalized_linear_models/example_poisson_regression_instrumental_variables.py)
for a complete worked example with captured output showing the bias of each
specification.

## Generalized Additive Models (GAM)

The same GLM stack supports **smooth covariates** via penalized IRLS, in the
spirit of statsmodels
[`GLMGam`](https://www.statsmodels.org/stable/generated/statsmodels.gam.generalized_additive_model.GLMGam.html).
Use ``kanly.api.gam`` (alias ``GAM``) instead of ``glm``:

```python
from kanly.api import gam

fit = gam(
    "y ~ x1 + x2",
    df,
    penalty=dict(x2=0.05),   # roughness weight per smooth variable
    df=dict(x2=20),          # spline basis dimension per smooth variable
    family="poisson",
)
print(fit.summary())  # fit.is_gam is True; Df Model uses effective d.f. (edf)
```

For each key in ``penalty`` / ``df``, the corresponding formula column is
expanded to a cubic B-spline basis before fit. The internal **``gam_penalty``**
matrix (integrated squared second derivative on each spline block) is added to
``X'WX`` at every IRLS iteration — the same mechanism documented in
``sparse_glm_internal`` — rather than using coordinate descent.

- ``penalty[var]=0``: unpenalized GLM on the full spline expansion (flexible, can overfit).
- Larger ``penalty[var]``: smoother fitted curves; summary reports **edf** (effective degrees of freedom) per coefficient.

Linear terms and GLM options (family, link, ``cov_type``, IV syntax where supported)
behave as in ``glm``. Do not combine GAM with ``alpha > 0`` elastic-net in the
current implementation.

See [`examples/regression/generalized_linear_models/example_gam_regression.py`](../../../../examples/regression/generalized_linear_models/example_gam_regression.py)
for Poisson GAM fits at several penalty strengths with a plot of fitted curves.

## Large-Scale Models

The sparse formula path supports large row counts and high-cardinality
categoricals:

```python
fit = glm(
    "y ~ x + poly(z, 2) + C(group)",
    df,
    family="binomial",
    debug=True,
)
```

Use `debug=True` to print parsing and optimization progress, and consider
categorical terms through `C(...)` to keep design construction sparse.

## External References

Wikipedia has concise background pages for the core GLM concepts used here:

- [Generalized linear model](https://en.wikipedia.org/wiki/Generalized_linear_model)
- [Exponential family](https://en.wikipedia.org/wiki/Exponential_family)
- [Link function](https://en.wikipedia.org/wiki/Generalized_linear_model#Link_function)
- [Iteratively reweighted least squares](https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares)
- [Logistic regression](https://en.wikipedia.org/wiki/Logistic_regression)
- [Poisson regression](https://en.wikipedia.org/wiki/Poisson_regression)

### IV and residual inclusion (control-function approach)

- Terza, J.V., Basu, A., & Rathouz, P.J. (2008). Two-stage residual inclusion estimation: Addressing endogeneity in health econometric modeling. *Journal of Health Economics*, 27(3), 531–543.
  Coins the 2SRI estimator and proves its consistency for nonlinear models (including Poisson); contrasts with the inconsistent two-stage predictor substitution.

- Wooldridge, J.M. (2015). Control function methods in applied econometrics. *Journal of Human Resources*, 50(2), 420–445.
  Survey of control-function approaches across nonlinear model classes (binary choice, count, censored); clear exposition of when 2SLS and control functions agree (linear case) and diverge (nonlinear case).

- Rivers, D., & Vuong, Q.H. (1988). Limited information estimators and exogeneity tests for simultaneous probit models. *Journal of Econometrics*, 39(3), 347–366.
  Foundational proof of the residual-inclusion approach for nonlinear limited-dependent-variable models.

## Examples

**Quick start (multi-topic):** [`example_quick_start.ipynb`](../../../example_quick_start.ipynb) at the repository root — Poisson GLM / GLM-IV, plus OLS/IV, NLLS, LASSO, Bayesian, linear block bootstrap, and nonparametrics.

See also `examples/regression/generalized_linear_models/`:

- `example_logistic_regression.py`: logistic/binomial GLM, polynomial terms, bootstrap covariance.
- `example_logistic_regression_large_scale.py`: large sparse design with categorical terms.
- `example_logistic_regression_instrumental_variables.py`: binary GLM with instruments and residual inclusion.
- `example_poisson_regression.py`: Poisson regression with log link.
- `example_poisson_regression_instrumental_variables.py`: Poisson IV GLM with bootstrap covariance and result comparison.
- `example_gam_regression.py`: Poisson GAM with B-spline smooths and penalty tuning.
