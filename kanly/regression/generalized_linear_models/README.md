# Generalized Linear Models User Guide

**See also:** [kanly README](../../../README.md) · [regression overview](../README.md) · [formula](../../formula/README.md)

This package fits sparse generalized linear models (GLMs), including ordinary
and zero-truncated count models, with canonical or custom links, variance
weights, optional instruments, robust covariance, elastic-net style
regularization, **marginal effects** on the fitted mean
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
function, `V(mu)` is the family-specific variance function, and `phi_i` is
the observation-level GLM scale.

When variance weights `v_i` are supplied:

```text
phi_i = phi / v_i
Var(y_i | x_i, v_i) = phi * V(mu_i) / v_i
```

The global scale value used by the fit is returned as `fit.scale`, but it is
not always estimated. For families declaring a fixed GLM scale it is exactly
`1`. For Gaussian, Gamma, and inverse-Gaussian families it is a Pearson moment
estimate.
GLM scale, family-specific distribution parameters, and elastic-net
regularization parameters are distinct quantities; see
[Scale and dispersion terminology](#scale-and-dispersion-terminology).

IRLS solves the local weighted least-squares problem implied by the current
mean and link derivative. For non-canonical links, the effective working
weights, including the supplied variance weights, are proportional to:

```text
W_i = v_i / (g'(mu_i)^2 * V(mu_i))
```

For canonical links this simplifies because the link and family match.  The
penalized objective used when `alpha > 0` adds an elastic-net term:

```text
- mean(log L_i) + alpha * [l1_ratio * ||beta||_1
                         + (1 - l1_ratio) / 2 * ||beta||_2^2]
```

The `alpha` in this objective is the regularization strength, not a
negative-binomial overdispersion parameter.

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

## Zero-Truncated Poisson

Use `zero_truncated_poisson` when the response is a count observed conditional
on being positive, as in the positive-count component of a hurdle model:

```python
from kanly.api import glm

fit = glm(
    "positive_count ~ x + z",
    data=df,
    family="zero_truncated_poisson",
)
```

The regression models the underlying, untruncated Poisson rate rather than the
conditional mean directly:

```text
eta_i = x_i' beta = log(lambda_i)
mu_i = E[Y_i | Y_i > 0, x_i]
     = lambda_i / (1 - exp(-lambda_i))
```

Consequently, `exp(x_i' beta)` is `lambda_i`, while fitted values are the
zero-truncated conditional means `mu_i > 1`. An observed count may still equal
one. The conditional probability mass function is:

```text
P(Y_i = y | Y_i > 0, x_i)
    = exp(-lambda_i) * lambda_i**y
      / (y! * (1 - exp(-lambda_i))),  y = 1, 2, ...
```

Important behavior:

- Outcomes must be finite positive integers. Zero, negative, fractional, and
  non-finite values raise `ValueError` before fitting.
- The family has fixed GLM scale: `fit.scale == 1.0`. It has no estimated
  dispersion parameter.
- `lambda_i` varies with the regressors and is the modeled Poisson rate; it is
  not a scale or dispersion parameter.
- Fitting ordinary Poisson after deleting zero observations is not equivalent,
  because ordinary Poisson omits the truncation normalizer.
- The only registered safe/default link is
  `zero_truncated_poisson_link`, its canonical log-rate link.

A complete simulation and parameter-recovery example is available in
[`example_zero_truncated_poisson.py`](../../../examples/regression/generalized_linear_models/example_zero_truncated_poisson.py).

No separate zero-truncated Gamma family is needed. Gamma already has support
on `(0, inf)` and assigns probability zero to an exact zero; the shared support
validation rejects zero-valued Gamma outcomes. A continuous hurdle model should
fit ordinary `gamma` to its positive observations, typically with `link="log"`.

## Families and Links

Families can be supplied by name, class, or instance. String names are
case-insensitive and underscores are optional, so `"negative_binomial"` and
`"negativebinomial"` both resolve to the same family. Negative binomial also
supports a **fixed, user-specified** overdispersion parameter in the string
form, for example `"negative_binomial(0.5)"`. This syntax specifies the
parameter; it does not estimate it.

### Registered Families

The `Default link` is used when `link=None`. The `Canonical link` is the
exponential-family canonical link. `Legal links` are the link classes returned by
each family's `safe_links()` and accepted by the model validation layer.

| Family | Response support / role | Variance function `V(mu)` | GLM scale `phi` | Additional distribution parameter | Default link | Canonical link | Legal links |
|--------|--------------------------|---------------------------|-----------------|-----------------------------------|--------------|----------------|-------------|
| `binomial` | Outcomes in `[0, 1]`; binary or fractional proportions | `mu * (1 - mu)` | Fixed at `1` | None | `logit` | `logit` | `probit`, `logit`, `cloglog`, `identity`, `cauchy` |
| `bernoulli` | Alias-style subclass of `binomial` for binary outcomes | `mu * (1 - mu)` | Fixed at `1` | None | `logit` | `logit` | `probit`, `logit`, `cloglog`, `identity`, `cauchy` |
| `poisson` | Non-negative counts | `mu` | Fixed at `1` | None | `log` | `log` | `log`, `sqrt`, `identity` |
| `zero_truncated_poisson` | Positive integer counts `{1, 2, ...}` | `mu * (1 + lambda - mu)`, where `mu=lambda/(1-exp(-lambda))` | Fixed at `1` | None; modeled rate `lambda_i=exp(x_i' beta)` is not dispersion | `zero_truncated_poisson_link` | `zero_truncated_poisson_link` | `zero_truncated_poisson_link` |
| `gaussian` | Real-valued continuous outcomes | `1` | Pearson-estimated | None | `identity` | `identity` | `log`, `identity` |
| `gamma` | Strictly positive continuous outcomes | `mu**2` | Pearson-estimated | Shape implied by effective scale: `1 / phi_i`; not independently fitted | `negative_inverse` | `negative_inverse` | `log`, `identity`, `inverse`, `negative_inverse` |
| `inverse_gaussian` | Strictly positive continuous outcomes | `mu**3` | Pearson-estimated | Shape implied by effective scale: `1 / phi_i`; not independently fitted | `negative_two_inverse_squared` | `negative_two_inverse_squared` | `inverse_squared`, `negative_two_inverse_squared`, `identity`, `log` |
| `negative_binomial` | Overdispersed non-negative counts | `mu + alpha_nb * mu**2` | Fixed at `1` | Fixed `alpha_nb > 0`; default `1` | `log` | `negative_binomial_canonical_link` | `log`, `sqrt`, `identity` |

For Gaussian, Gamma, and inverse-Gaussian families, the reported scale is the
weighted Pearson moment estimate:

```text
phi_hat
    = sum_i [
          v_i * (y_i - mu_i)^2 / V(mu_i)
      ] / df_resid
```

This is not, in general, joint maximum-likelihood estimation of a separate
scale or shape parameter. For Gamma and inverse Gaussian, the implied
observation-level shape is `1 / phi_i`. It is `1 / fit.scale` without variance
weights and `v_i / fit.scale` when observation `i` has variance weight `v_i`.
These shapes are implied by the Pearson scale estimate and weights rather than
independently optimized parameters.

Fixed versus estimated scale is selected through each family's
`is_fixed_dispersion()` method. This keeps aliases such as `bernoulli` and new
fixed-dispersion families consistent without maintaining a separate list in
the optimizer.

For fractional responses, `binomial` supplies a binomial-style mean and
variance model; a fractional response should not be represented as a literal
one-trial Bernoulli draw.

The negative-binomial canonical link is implemented for family internals, but it
is **not** included in `NegativeBinomial.safe_links()`. In normal user-facing
fits, use the default `log` link or another listed legal link.

### Scale and dispersion terminology

Kanly uses `fit.scale` for the GLM scale `phi`:

- `binomial`, `bernoulli`, `poisson`, `zero_truncated_poisson`, and
  `negative_binomial` declare a fixed GLM scale, so `fit.scale` is exactly `1`.
- `gaussian`, `gamma`, and `inverse_gaussian` declare a non-fixed GLM scale, so
  `fit.scale` is estimated by the weighted Pearson formula shown above.
- `family.is_fixed_dispersion()` controls this behavior in the optimizer; it
  is not maintained through a separate hard-coded family list.

Several unrelated API values are sometimes also called "dispersion" or
`alpha`, but they must not be conflated:

- `fit.scale` is the GLM scale `phi`.
- `fit.family.alpha` is the fixed NB2 overdispersion parameter when using
  `negative_binomial`.
- `fit.alpha` is the elastic-net regularization strength.
- `cov_kwds["alpha"]` is the Bayesian-bootstrap Dirichlet concentration.
- Zero-Truncated Poisson has no free dispersion parameter; its observation-
  specific `lambda_i` is determined by the regression coefficients.

#### Fixed negative-binomial overdispersion

A negative-binomial GLM uses the NB2 variance function:

```text
V(mu) = mu + alpha_nb * mu**2
```

Its GLM scale is fixed at `1`. The family-specific `alpha_nb` is supplied by
the user and is held fixed:

```python
fit = glm(
    "y ~ x",
    df,
    family="negative_binomial(0.5)",
    alpha=0.1,
)

fit.scale         # 1.0: GLM scale.
fit.family.alpha  # 0.5: fixed NB2 overdispersion.
fit.alpha         # 0.1: elastic-net penalty strength.
```

The default `family="negative_binomial"` sets `fit.family.alpha` to `1.0`.
Kanly does not currently estimate NB2 overdispersion as part of `glm`.
"Fixed GLM scale" refers specifically to `fit.scale == 1`; independently,
this implementation also requires `alpha_nb` to be supplied and held fixed.
Neither statement means NB2 is equidispersed like Poisson.

Bayesian bootstrap has another, unrelated `alpha`:

```python
cov_kwds = {
    "method": "bayesian",
    "alpha": 1.0,
    "n_samples": 500,
    "seed": 123,
}
```

Here `cov_kwds["alpha"]` controls the Dirichlet concentration of the
bootstrap weights. It is neither NB2 overdispersion nor regularization
strength.

### Registered Link Names

| Link name | Link function `g(mu)` | Common role |
|-----------|------------------------|-------------|
| `logit` | `log(mu / (1 - mu))` | Canonical/default for `binomial` and `bernoulli` |
| `probit` | `Phi^{-1}(mu)` | Alternative binary-response link |
| `cloglog` | `log(-log(1 - mu))` | Alternative binary-response link |
| `cauchy` | Cauchy inverse-CDF link | Alternative binary-response link |
| `identity` | `mu` | Canonical/default for `gaussian`; legal for several families |
| `log` | `log(mu)` | Canonical/default for `poisson`; default for `negative_binomial`; legal for positive-mean families |
| `zero_truncated_poisson_link` | `log(lambda(mu))`, where `mu=lambda/(1-exp(-lambda))` | Canonical/default for `zero_truncated_poisson` |
| `sqrt` | `sqrt(mu)` | Legal for `poisson` and `negative_binomial` |
| `negative_inverse` | `-1 / mu` | Canonical/default for `gamma` |
| `inverse` | `1 / mu` | Legal for `gamma` |
| `negative_two_inverse_squared` | `-1 / (2 * mu**2)` | Canonical/default for `inverse_gaussian` |
| `inverse_squared` | `1 / mu**2` | Legal for `inverse_gaussian` |
| `exponential` | `exp(mu)` | Registered link name; not listed as safe for the registered families above |
| `negative_binomial_canonical_link` | `log(alpha_nb * mu / (1 + alpha_nb * mu))` | Canonical negative-binomial link; implemented but not user-facing safe by default |

The `Power` link class exists in `links.py`, but it is not registered in
`LINK_NAME_2_CLS` and is not listed as safe for any family.

Common examples:

```python
glm("y ~ x", df, family="binomial")   # logistic/probit-style binary models
glm("y ~ x", df, family="poisson")    # count models
glm("y ~ x", df, family="zero_truncated_poisson")  # positive counts only
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

These are **variance weights**, not general frequency or row-replication
weights. They adjust the observation-level scale as:

```text
phi_i = phi / w_i
```

A frequency-weighted likelihood instead multiplies the *complete*
observation-level log-likelihood:

```text
log L_frequency = sum_i w_i * log p(y_i | mu_i, family_parameters)
```

The two constructions are not equivalent for every family. In particular,
the current NB2 variance-weighted likelihood does not multiply every
dispersion-dependent normalization term by `w_i`. Therefore, do not use
`fit.llf` as a fully frequency-weighted NB2 likelihood for estimating or
comparing different values of `fit.family.alpha`.

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

Bootstrap covariance refits the GLM mean-model coefficients on
bootstrap-weighted samples and stores their empirical covariance on the
returned result object. The selected family and its user-specified
distribution parameters are held fixed. In particular, negative-binomial
bootstrap refits do not estimate, resample, or report uncertainty in
`fit.family.alpha`.

The current GLM bootstrap calls its resampling implementation with
`groups=None`. Do not assume that specifying group information in
`cov_kwds` provides a cluster or block bootstrap unless that implementation
has first been updated and tested.

Confidence intervals for the conditional mean are distinct from prediction
intervals for future outcomes. A prediction interval must additionally
include outcome-level realization uncertainty and, when applicable,
uncertainty in estimated distribution parameters.

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

- `alpha`: overall penalty strength, returned as `fit.alpha`; not the
  negative-binomial family parameter `fit.family.alpha`.
- `l1_ratio`: L1 share of the elastic-net penalty.
- `normalize`: whether to scale penalties by predictor standard deviation.
- `penalize_scale`: whether penalties are multiplied by the current GLM scale;
  this has no numerical effect when the family fixes `fit.scale` at `1`.

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

## Current Distributional Limitations

- Negative-binomial overdispersion is user-specified and fixed. It is not
  estimated, profiled, or re-estimated by the GLM bootstrap.
- Zero counts are valid for Poisson and negative-binomial models. However,
  the current negative-binomial deviance computes `y * log(y / mu)` directly,
  which can return `nan` when `y = 0`. A zero-safe implementation should use
  `scipy.special.xlogy` or explicitly handle zero counts.
- Generalized Poisson is not a registered family.
- The current GLM bootstrap does not perform group- or cluster-level
  resampling.
- Confidence intervals for fitted means do not automatically provide
  prediction intervals for future outcome realizations.

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
- `example_zero_truncated_poisson.py`: positive-count simulation and
  Zero-Truncated Poisson parameter recovery.
- `example_poisson_regression_instrumental_variables.py`: Poisson IV GLM with bootstrap covariance and result comparison.
- `example_gam_regression.py`: Poisson GAM with B-spline smooths and penalty tuning.
