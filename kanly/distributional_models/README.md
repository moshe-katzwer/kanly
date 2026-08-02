# Distributional Models User Guide

**See also:** [kanly README](../../README.md) ·
[formula guide](../formula/README.md) ·
[generalized linear models](../regression/generalized_linear_models/README.md)

> **Experimental status and authorship:** Almost all code in the other kanly
> modules was written by Moshe Katzwer. The code in this module was written
> largely by Codex on a scaffold defined by Moshe Katzwer. This module is
> experimental and should be validated carefully before use in production or
> consequential statistical analysis.

This package provides compact likelihood and estimating-equation regression
models for counts and positive continuous outcomes. Unlike the IRLS
generalized-linear-model implementation, direct models jointly optimize all
parameters in their likelihood, including Gamma and negative-binomial
dispersion parameters. Hurdle models deliberately retain separable component
estimation. Gaussian and lognormal positive components use OLS; Poisson,
Gamma, and Inverse Gaussian use GLMs; negative-binomial-P uses an exact
zero-truncated likelihood so its dispersion can be estimated.

The public base class is `DistributionalModel`. Source files are organized by
response structure:

- [`api.py`](api.py) contains the one-shot formula and array entry points plus
  the normalized model-name and alias registry.
- [`base.py`](base.py) contains the common `DistributionalModel` estimation,
  validation, bootstrap, and inference machinery.
- [`two_part.py`](two_part.py) contains shared zero-process data and formula
  handling for zero-inflated and hurdle models.
- [`count_models.py`](count_models.py) contains discrete count,
  zero-inflated, and negative-binomial likelihoods.
- [`continuous_models.py`](continuous_models.py) contains the continuous
  `Gamma` likelihood model.
- [`hurdle_models.py`](hurdle_models.py) contains Gaussian, lognormal,
  Poisson, Gamma, Inverse Gaussian, and negative-binomial-P hurdle models.
- [`results.py`](results.py) contains the shared fitted-results object.

All public model classes can be imported directly from
`kanly.distributional_models`.

## Public API and Available Models

```python
from kanly.distributional_models import (
    DISTRIBUTIONAL_MODEL,
    DISTRIBUTIONAL_MODEL_ALIASES,
    DistributionalModel,
    DistributionalModelResults,
    TwoPartModel,
    GaussianHurdle,
    Gamma,
    GammaHurdle,
    GeneralizedPoisson,
    InverseGaussianHurdle,
    LognormalHurdle,
    NegativeBinomial1,
    NegativeBinomial2,
    NegativeBinomialPHurdle,
    Poisson,
    PoissonHurdle,
    ZeroInflatedNegativeBinomial,
    ZeroInflatedPoisson,
    distributional_model,
)
```

`DistributionalModel` supplies formula construction, importance weighting of
likelihood observations, optimization, observation scores, covariance
estimation, bootstrap refits, and common prediction behavior. Concrete
subclasses provide the distribution likelihood and parameter names.

## One-Shot Formula and Array APIs

The lower-case entry point parses a formula and immediately fits the selected
model:

```python
from kanly.distributional_models import distributional_model

fit = distributional_model(
    "y ~ x $ weight",
    data=df,
    model_name="negative binomial p hurdle",
    exog_infl="z + C(group)",
    p=2,
    cov_type="SANDWICH",
)
```

The upper-case entry point performs the same one-shot operation from arrays:

```python
from kanly.distributional_models import DISTRIBUTIONAL_MODEL

fit = DISTRIBUTIONAL_MODEL(
    endog=y,
    exog=X,
    model_name="nb2_hurdle",
    exog_infl=Z,
    weights=w,
    exog_names=["Intercept", "x"],
    exog_infl_names=["Intercept", "z"],
    cov_type="NONROBUST",
)
```

Both functions return `DistributionalModelResults`. They are also exported by
`kanly.api`, paralleling `lm`/`LM` and `glm`/`GLM`.

The two functions intentionally have broad, explicit signatures. An argument
that is irrelevant to the selected model is ignored: for example,
`exog_infl` does nothing for `Poisson`, `p` does nothing for `Gamma`, and
`positive_link` does nothing outside `GammaHurdle` and
`InverseGaussianHurdle`. The important exceptions are malformed arguments
that actually apply to the selected model, which are validated normally.
Formula weights continue to use the `$` extension; array weights use the
explicit `weights` argument.

### Model-name lookup

Model-name matching is case-insensitive and removes all non-alphanumeric
separators. Consequently, `"ZeroInflatedPoisson"`,
`"zero inflated poisson"`, `"ZERO_INFLATED_POISSON"`, and
`"zero-inflated-poisson"` are equivalent. The complete alias table is:

| Selected model | Accepted names and aliases | Implied option |
|---|---|---|
| `Poisson` | `Poisson`, `pois` | — |
| `ZeroInflatedPoisson` | `ZeroInflatedPoisson`, `zero inflated poisson`, `zip` | — |
| `ZeroInflatedNegativeBinomial` | `ZeroInflatedNegativeBinomial`, `zero inflated negative binomial`, `zero inflated nb`, `zinb`, `zinb2` | NB2 count component |
| `NegativeBinomial1` | `NegativeBinomial1`, `negative binomial 1`, `negative binomial one`, `nb1` | NB1 |
| `NegativeBinomial2` | `NegativeBinomial2`, `negative binomial 2`, `negative binomial two`, `negative binomial`, `nb2`, `nb` | NB2 |
| `GeneralizedPoisson` | `GeneralizedPoisson`, `generalized poisson`, `generalized pois`, `gen poisson`, `gp` | Class default `p=1` |
| `GeneralizedPoisson` | `generalized poisson 1`, `gp1` | `p=1` |
| `GeneralizedPoisson` | `generalized poisson 2`, `gp2` | `p=2` |
| `PoissonHurdle` | `PoissonHurdle`, `poisson hurdle`, `hurdle poisson` | — |
| `NegativeBinomialPHurdle` | `NegativeBinomialPHurdle`, `negative binomial p hurdle`, `negative binomial hurdle`, `nbp hurdle`, `nb hurdle`, `hurdle negative binomial` | Class default `p=2` |
| `NegativeBinomialPHurdle` | `negative binomial 1 hurdle`, `nb1 hurdle`, `hurdle nb1` | `p=1` |
| `NegativeBinomialPHurdle` | `negative binomial 2 hurdle`, `nb2 hurdle`, `hurdle nb2` | `p=2` |
| `GammaHurdle` | `GammaHurdle`, `gamma hurdle`, `hurdle gamma` | — |
| `GaussianHurdle` | `GaussianHurdle`, `gaussian hurdle`, `normal hurdle`, `hurdle gaussian`, `hurdle normal`, `gaussian`, `normal` | — |
| `LognormalHurdle` | `LognormalHurdle`, `lognormal hurdle`, `log normal hurdle`, `hurdle lognormal`, `lognormal`, `log normal` | — |
| `InverseGaussianHurdle` | `InverseGaussianHurdle`, `inverse gaussian hurdle`, `inverse gaussian`, `ig hurdle`, `igh` | — |
| `Gamma` | `Gamma` | — |

An explicit `p` may be supplied with the general `GeneralizedPoisson` and
`NegativeBinomialPHurdle` names. If a parameterization-specific alias such as
`gp2` or `nb1_hurdle` conflicts with an explicit `p`, the API raises an error
instead of silently choosing one.

Direct models and count components use a log link for the conditional mean:

```text
eta_i = x_i' beta
mu_i = exp(eta_i)
```

| Class | Response model | Conditional variance | Additional parameters |
|---|---|---|---|
| `Poisson` | Poisson | `mu` | None |
| `Gamma` | Gamma | `alpha * mu**2` | `log_alpha` |
| `GeneralizedPoisson(p=1)` | GP-1 | `mu * (1 + alpha)**2` | Raw `alpha` |
| `GeneralizedPoisson(p=2)` | GP-2 | `mu * (1 + alpha * mu)**2` | Raw `alpha` |
| `NegativeBinomial1` | NB-1 | `mu * (1 + alpha)` | `log_alpha` |
| `NegativeBinomial2` | NB-2 | `mu + alpha * mu**2` | `log_alpha` |
| `ZeroInflatedPoisson` | Structural-zero mixture with Poisson counts | Mixture variance | Inflation coefficients |
| `ZeroInflatedNegativeBinomial` | Structural-zero mixture with NB-2 counts | Mixture variance | Inflation coefficients and `log_alpha` |
| `PoissonHurdle` | Logit zero hurdle plus zero-truncated Poisson GLM | Two-part model | Hurdle coefficients |
| `GammaHurdle` | Logit zero hurdle plus Gamma/log GLM quasi-likelihood | Two-part model | Hurdle coefficients; Pearson GLM scale |
| `GaussianHurdle` | Logit zero hurdle plus working Gaussian OLS on positive `Y` | Positive variance `scale` | Estimated `log_scale`; hurdle coefficients |
| `LognormalHurdle` | Logit zero hurdle plus OLS on positive `log(Y)` | Positive log-variance `scale` | Estimated `log_scale`; hurdle coefficients |
| `InverseGaussianHurdle` | Logit zero hurdle plus Inverse Gaussian/log GLM quasi-likelihood | Positive variance `scale * mu**3` | Hurdle coefficients; Pearson GLM scale |
| `NegativeBinomialPHurdle(p=1 or 2)` | Logit zero hurdle plus exact zero-truncated NB-P likelihood | `mu + alpha * mu**p` before truncation | Estimated `log_alpha`; hurdle coefficients |

For models reporting `log_alpha`, the positive dispersion is
`alpha = exp(log_alpha)`. Generalized Poisson estimates raw `alpha` because
valid negative values permit underdispersion. In `NegativeBinomialPHurdle`,
`p` is a fixed model choice rather than an estimated parameter: `p=1` selects
NB1 and `p=2` selects NB2 (the default).

`Gamma`, imported from `continuous_models.py`, is a continuous distribution;
its outcome must be finite and strictly positive. Zero-inflation and hurdle
probabilities use a logit. `GammaHurdle` and `InverseGaussianHurdle` use a log
positive-response link by default and permit another supported GLM link
through `positive_link`. `GaussianHurdle` regresses positive `Y` directly;
`LognormalHurdle` regresses `log(Y)`.

## Basic Formula API

Build a model and then call `fit`:

```python
import numpy as np

from kanly.distributional_models import NegativeBinomial2

model = NegativeBinomial2.build_model_from_formula(
    "y ~ x + I(z**2) $ weight",
    data=df,
)

fit = model.fit(
    cov_type="SANDWICH",
)

print(fit.summary_df())
```

The `$` formula extension supplies optional observation likelihood weights.
Instrumental-variable and absorbed-effect formulas are rejected.

Formula construction records column names and row-alignment metadata on the
model. For example, `fit.param_names` uses formula names rather than generated
names such as `x0` and `x1`.

Every concrete direct-likelihood model supplies automatic starting values, so
`model.fit()` does not require `start_params`. Inspect them with
`model.get_start_params()` or `model.default_start_params`. Passing an explicit
vector still takes precedence:

```python
fit = model.fit(start_params=np.zeros(len(model.param_names)))
```

Set `debug=True` during either construction or fitting for diagnostics similar
to `kanly.api.lm`:

```python
model = NegativeBinomial2.build_model_from_formula(
    "y ~ x $ weight",
    data=df,
    debug=True,
)
fit = model.fit(cov_type="SANDWICH", debug=True)
```

Formula debugging is forwarded to the sparse Patsy-style data builder and
then reports the aligned model dimensions, names, weights, and construction
time. Fit debugging is forwarded to the point estimator (`BFGS-PQN` for
direct models and both component GLMs for hurdle models), followed by
first-order and covariance diagnostics. Bootstrap debugging displays its
settings and a progress bar while suppressing the otherwise very noisy
per-draw optimizer transcripts.

## Zero-Inflated Models

Zero-inflated models combine a structural-zero process with a count process.
The structural-zero probability uses a logit link:

```text
pi_i = expit(z_i' gamma)
```

For zero-inflated Poisson:

```text
P(Y_i = 0) = pi_i + (1 - pi_i) * exp(-mu_i)

P(Y_i = y) = (1 - pi_i)
             * exp(-mu_i) * mu_i**y / y!,  y > 0
```

Thus an observed zero can originate from either the structural-zero component
or the ordinary count component.

The count and inflation equations are supplied separately:

```python
from kanly.distributional_models import ZeroInflatedPoisson

model = ZeroInflatedPoisson.build_model_from_formula(
    "orders ~ price + promotion $ weight",
    data=df,
    exog_infl="customer_age + C(region)",
)

fit = model.fit(
    cov_type="NONROBUST",
)
```

`exog_infl` is a Patsy-style right-hand-side formula. It includes an intercept
unless `-1` is specified. If `exog_infl=None`, the inflation equation is
intercept-only.

The shared formula builder is available on both:

```python
ZeroInflatedPoisson.build_model_from_formula(...)
ZeroInflatedNegativeBinomial.build_model_from_formula(...)
```

Missing rows are aligned across the outcome, count regressors, inflation
regressors, and weights. Formula-derived inflation coefficients receive names
such as `inflate_Intercept`, `inflate_customer_age`, and
`inflate_C(region)[...]`.

A complete parameter-recovery simulation is available in
[`example_zero_inflated_poisson.py`](../../examples/distributional_models/example_zero_inflated_poisson.py).

## Hurdle Models

Hurdle models assign every zero to a Bernoulli/logit zero process and model
the positive observations conditionally:

```text
pi_i = P(Y_i = 0) = expit(z_i' gamma)

P(Y_i = 0) = pi_i
P(Y_i = y), y > 0 = (1 - pi_i) * f_positive(y | x_i)
```

Unlike a zero-inflated model, the positive distribution cannot generate a
zero. `PoissonHurdle` uses the `ZeroTruncatedPoisson` GLM family for the
positive counts. `GammaHurdle` and `InverseGaussianHurdle` use their ordinary
GLM families because both already have support on `(0, infinity)`; each uses
a log positive-mean link by default and is estimated by IRLS. The Inverse
Gaussian positive component has `Var(Y | Y>0, X) = scale * mu**3`.
`GaussianHurdle` and `LognormalHurdle` use kanly's weighted OLS solver. The
former regresses positive `Y` directly. The latter regresses `log(Y)`, giving

```text
median(Y | Y>0, X) = exp(X beta)
mean(Y | Y>0, X) = exp(X beta + scale / 2)
scale = exp(log_scale)
```

The ordinary Gaussian density is not restricted to positive values, so
`GaussianHurdle` is a working quasi-likelihood model and does not report AIC
or BIC. `LognormalHurdle` is a normalized positive-response likelihood and
reports them for unweighted fits. Both estimate `log_scale` and include it in
the parameter score, covariance, and bootstrap draws.
`NegativeBinomialPHurdle` maximizes an exact zero-truncated NB-P likelihood for
the positive counts. It does not route that component through the GLM family
API because `alpha` is estimated rather than fixed.

```python
from kanly.distributional_models import PoissonHurdle

model = PoissonHurdle.build_model_from_formula(
    "orders ~ price + promotion $ weight",
    data=df,
    exog_infl="customer_age + C(region)",
)
fit = model.fit(cov_type="SANDWICH")

print(fit.summary_df())
print(fit.positive_fit)  # positive-response GLM, OLS, or direct fit
print(fit.hurdle_fit)    # Bernoulli/logit GLM for P(Y=0)
```

`exog_infl=None` creates an intercept-only zero hurdle. Instruments remain
unsupported and are rejected by formula construction. Parameters are ordered
as all positive-response parameters followed by coefficients named
`hurdle_...`.

The objective separates, so `fit` estimates the two components rather
than jointly optimizing the combined objective. `NONROBUST` returns the exact
block-diagonal component covariance. `SANDWICH` and `HC1` use that block-
diagonal bread with the full combined observation-score meat, retaining
cross-component covariance; `HC1` adds its finite-sample correction.
`COMPONENT_HC1` preserves the block-diagonal robust component estimator.
`BOOTSTRAP` uses each full-sample Bayesian weight draw for both components and
therefore retains coherent joint parameter draws and cross covariance.

The combined object provides `loglike`, `loglike_obs`, `score`, `score_obs`,
and predictions for the overall mean, conditional positive mean, underlying
positive-distribution mean, linear predictor, zero probability, or positive
probability. Use `which="positive_mean"`, `"underlying_mean"`,
`"linear_predictor"`, `"zero_probability"`, or `"positive_probability"`.
The combined distributional-model API keeps `loglike_obs` and `score_obs`
unweighted and applies observation weights only when aggregating them. During
component estimation those importance weights multiply the corresponding
component likelihood or estimating-equation contributions. `GammaHurdle` and
`InverseGaussianHurdle` retain Pearson-estimated scale, so they are explicitly
reported as quasi-likelihood and do not report AIC or BIC. `GaussianHurdle`
is also explicitly quasi-likelihood because its untruncated normal working
density is evaluated only over positive observations.

### Negative-binomial-P hurdle parameters

`NegativeBinomialPHurdle(endog, exog, exog_infl=Z, p=2)` has three distinct
sets of quantities. They should not be conflated:

```text
eta_i = x_i' beta
mu_i = exp(eta_i)
alpha = exp(log_alpha) > 0
r_i = mu_i**(2-p) / alpha
q_i = P_NBP(Y_i=0 | x_i) = (r_i / (r_i + mu_i))**r_i

h_i = z_i' gamma
pi_i = P(Y_i=0 | z_i) = expit(h_i)
```

- `beta` contains one coefficient for every column of `exog`. It controls
  `log(mu_i)`, where `mu_i` is the mean of the hypothetical *untruncated*
  NB-P count distribution. A one-unit increase in regressor `x_j` multiplies
  this underlying mean by `exp(beta_j)`, holding other regressors fixed.
  Because zeros are subsequently truncated away, `beta_j` is not generally a
  log effect on the positive conditional mean or on the overall hurdle mean.
- `log_alpha` is the optimized dispersion parameter on an unconstrained
  scale. The reported `fit.dispersion` is
  `alpha = exp(log_alpha)`. Larger `alpha` means more overdispersion. For
  `p=1`, the underlying variance is `mu_i * (1 + alpha)`; for `p=2`, it is
  `mu_i + alpha * mu_i**2`.
- `gamma` contains one coefficient for every column of `exog_infl`. The
  reported names are prefixed with `hurdle_`. It controls the log odds of an
  observed zero: `log(pi_i / (1-pi_i)) = z_i' gamma`. Thus a positive
  `gamma_j` increases the probability of a zero and decreases the probability
  of crossing the hurdle.
- `p` is fixed when the model is constructed and is not present in
  `fit.params`. `p=1` means NB1 and `p=2` means NB2; the default is `2`.

The NB-P probability of zero simplifies to

```text
p=1: q_i = (1 + alpha)**(-mu_i / alpha)
p=2: q_i = (1 + alpha * mu_i)**(-1 / alpha)
```

The positive component uses
`P(Y=y | Y>0,x) = P_NBP(Y=y | x) / (1-q_i)` for integer `y>=1`. Therefore,

```text
E[Y | Y>0, x_i] = mu_i / (1-q_i)
E[Y | x_i, z_i] = (1-pi_i) * mu_i / (1-q_i)
```

These correspond to `predict(which="positive_mean")` and the default
`predict(which="mean")`, respectively. `predict(which="underlying_mean")`
returns `mu_i`, not either of those observed-response means.

For example, if the positive design columns are `Intercept` and `x`, and the
hurdle design columns are `Intercept` and `z`, the exact parameter order and
names are:

```text
Intercept, x, log_alpha, hurdle_Intercept, hurdle_z
```

Complete parameter-recovery examples are available for both
[`PoissonHurdle`](../../examples/distributional_models/example_poisson_hurdle.py)
and
[`GammaHurdle`](../../examples/distributional_models/example_gamma_hurdle.py),
with an NB2 example in
[`example_negative_binomial_p_hurdle.py`](../../examples/distributional_models/example_negative_binomial_p_hurdle.py).

## Array API

Construct models directly when matrices have already been prepared:

```python
from kanly.distributional_models import ZeroInflatedNegativeBinomial

model = ZeroInflatedNegativeBinomial(
    endog=y,
    exog=X,
    endog_name="claims",
    exog_names=["constant", "exposure"],
    exog_infl=Z,
    exog_infl_names=["constant", "prior_zero"],
    weights=w,
    weights_name="importance_weight",
)

fit = model.fit(
    start_params=np.zeros(X.shape[1] + Z.shape[1] + 1),
    cov_type="SANDWICH",
)
```

For the array API, `exog_infl` is a numeric matrix. If omitted, the model
creates a column of ones. In formula construction, by contrast, `exog_infl`
must be a formula string or `None`.

Names are constructor inputs rather than attributes that need to be patched
after construction. Every model builds `param_names` immediately from
`exog_names` plus its distribution-specific parameters. Zero-inflated models
also append the prefixed `exog_infl_names`. When regression names are omitted,
the fallback names are `x0`, `x1`, and so on.

## Parameter Order and Starting Values

Automatic starting values contain exactly one entry for every name returned by
`model.get_param_names()` or `model.param_names`:

| Model | Automatic starting values |
|---|---|
| `Poisson` | Log of the weighted response mean in a detected constant; other coefficients zero |
| `Gamma` | Log-mean regression start plus moment-based `log_alpha` |
| `GeneralizedPoisson` | Log-mean regression start plus near-Poisson `alpha=0.05` |
| `NegativeBinomial1` | Log-mean regression start plus NB-1 moment `log_alpha` |
| `NegativeBinomial2` | Log-mean regression start plus NB-2 moment `log_alpha` |
| `ZeroInflatedPoisson` | Excess-zero logit and zero-adjusted count mean |
| `ZeroInflatedNegativeBinomial` | ZIP starts plus an NB-2 moment `log_alpha` |
| `PoissonHurdle` | Zero-truncated Poisson positive start plus empirical zero-logit start |
| `GammaHurdle` | Gamma positive start plus empirical zero-logit start |
| `GaussianHurdle` | Positive-response mean and variance starts plus empirical zero-logit start |
| `LognormalHurdle` | Log-positive-response mean and variance starts plus empirical zero-logit start |
| `InverseGaussianHurdle` | Inverse Gaussian positive start plus empirical zero-logit start |
| `NegativeBinomialPHurdle` | Positive-count log-mean and moment `log_alpha` starts plus empirical zero-logit start |

An explicit `start_params` vector remains supported and must follow this
parameter order:

| Model | Parameter order |
|---|---|
| `Poisson` | `beta` |
| `Gamma` | `beta`, `log_alpha` |
| `GeneralizedPoisson` | `beta`, raw `alpha` |
| `NegativeBinomial1` | `beta`, `log_alpha` |
| `NegativeBinomial2` | `beta`, `log_alpha` |
| `ZeroInflatedPoisson` | count `beta`, inflation `gamma` |
| `ZeroInflatedNegativeBinomial` | count `beta`, inflation `gamma`, `log_alpha` |
| `PoissonHurdle` | positive `beta`, hurdle `gamma` |
| `GammaHurdle` | positive `beta`, hurdle `gamma` |
| `GaussianHurdle` | positive `beta`, `log_scale`, hurdle `gamma` |
| `LognormalHurdle` | log-positive `beta`, `log_scale`, hurdle `gamma` |
| `InverseGaussianHurdle` | positive `beta`, hurdle `gamma` |
| `NegativeBinomialPHurdle` | positive `beta`, `log_alpha`, hurdle `gamma` |

For a design matrix with `k` columns, a common initial value for a positive
dispersion is `log_alpha=-1`. A simple NB-2 start is therefore:

```python
start_params = np.r_[np.zeros(k), -1.0]
```

For Generalized Poisson, `alpha=0` is the Poisson limit. A small nonzero value
can help avoid numerical ambiguity at that limit:

```python
start_params = np.r_[np.zeros(k), 0.05]
```

Zero-inflated models often benefit from a negative initial inflation intercept,
which starts with fewer than 50 percent structural zeros:

```python
start_params = np.r_[np.zeros(k_count), -1.0, np.zeros(k_inflate - 1)]
```

## Likelihood Weights

Weights multiply complete observation-level log likelihoods during estimation:

```text
log L_weighted(theta) = sum_i w_i * log L_i(theta)
```

They are not baked into observation-level methods:

```python
ll_i = model.loglike_obs(params)  # always unweighted
score_i = model.score_obs(params) # always unweighted

ll = model.loglike(params)        # weights applied during aggregation
score = model.score(params)       # weights applied during aggregation
```

These are importance weights on the complete objective, not frequency weights
or a general variance-weight API. Consequently the sandwich meat uses squared
weighted scores, `sum_i (w_i score_i) (w_i score_i)'`.

Weights must be finite and non-negative and have positive total mass. With
formula construction, `sum_to_n=True` rescales them to sum to the final
aligned number of observations.

## Covariance Estimation

Direct-model `fit` accepts three covariance types:

```python
fit_nonrobust = model.fit(start, cov_type="NONROBUST")
fit_sandwich = model.fit(start, cov_type="SANDWICH")
fit_bootstrap = model.fit(
    start,
    cov_type="BOOTSTRAP",
    cov_kwds={
        "n_samples": 250,
        "seed": 123,
        "alpha": 1.0,
        "use_correction": True,
    },
)
```

### Nonrobust

`NONROBUST` uses the inverse observed information:

```text
cov(theta_hat) = inverse(-H(theta_hat))
```

where `H` is the Hessian of the weighted log likelihood.

### Sandwich

`SANDWICH` uses observation scores:

```text
bread = inverse(-H(theta_hat))
meat = sum_i weighted_score_i * weighted_score_i'
cov(theta_hat) = bread * meat * bread'
```

For hurdle models this is the full combined score meat. Although the hurdle
bread is block diagonal, robust covariance need not have zero cross blocks.

### Bayesian bootstrap

`BOOTSTRAP` draws Dirichlet observation weights, multiplies them by any
existing model weights, and refits the model. Every refit starts from the
original point estimate. Failed or non-finite repetitions are excluded.
`min_success_rate` defaults to `0.8`; covariance estimation fails rather than
silently accepting too few successful draws. `use_correction=True` computes
the ordinary sample covariance correction exactly once.

Bootstrap results expose:

```python
fit.bootstrapped_params
fit.cov_params()
fit.bse
fit.cov_kwds["n_successful"]
fit.cov_kwds["n_failed"]
```

Direct-model bootstrap objectives receive draw-specific weights explicitly;
the model's stored weights are never mutated. Hurdle bootstrap repetitions use
the same full-sample draw for both component GLMs.

## Results

Every count, zero-inflated, Gamma, and hurdle fit returns a
`DistributionalModelResults` object derived from the package-wide
`RegressionResultsBase`. The raw BFGS result remains available as
`fit.optimization_result`; hurdle component GLM results remain available as
`fit.positive_fit` and `fit.hurdle_fit`.

| Attribute | Meaning |
|---|---|
| `params` | Estimated parameters |
| `param_names` | Names in estimation order |
| `cov_params()` | Named estimated parameter covariance DataFrame |
| `bse` / `standard_errors` | Standard errors |
| `cov_type` | Selected covariance estimator |
| `cov_kwds` | Covariance/bootstrap options and diagnostics |
| `optimizer_converged` / `first_order_valid` | Raw stopping status and post-fit score validation |
| `inference_valid` / `inference_issues` | Whether covariance inference passed diagnostics and any failure reasons |
| `information_rank` / `information_condition` | Observed-information identification diagnostics |
| `summary_df()` | Coefficient, standard error, z statistic, p-value, and CI table |
| `predict(data=...)` | Re-evaluate stored main and zero-process formulas on new DataFrame or dict-like data |
| `llf` / `average_loglike` | Total and per-observation fitted log likelihood |
| `aic` / `bic` | Information criteria for unweighted normalized-likelihood fits; otherwise `None` |
| `fittedvalues` / `resid_response` | Unconditional fitted means and response residuals |
| `resid_zero` / `resid_positive` | Hurdle-component residuals for the zero indicator and positive response |
| `dispersion` | Transformed dispersion for Gamma, NB-1, NB-2, and ZINB, or raw GP alpha |
| `zero_probability` / `positive_probability` | Fitted outcome probabilities for two-part models |
| `inflation_probability` / `count_mean` | ZIP/ZINB structural-zero probability and count-component mean |
| `positive_fit` / `hurdle_fit` | Original component results for hurdle models; NB-P uses direct MLE for `positive_fit` |
| `optimization_result` | Original BFGS-PQN result for direct-likelihood models |
| `bread` | Inverse observed information for non-bootstrap covariance |
| `meat` | Score cross-product for sandwich covariance |
| `bootstrapped_params` | Retained bootstrap draws, if requested |

Example:

```python
print(fit.summary())
print(fit.summary_df())
print(fit.params)
print(fit.bse)
```

## Response Support and Interpretation

- `Gamma` requires finite, strictly positive outcomes.
- Poisson, generalized-Poisson, negative-binomial, and zero-inflated models
  require finite, non-negative outcomes.
- `PoissonHurdle` and `NegativeBinomialPHurdle` additionally require integer
  outcomes. `GammaHurdle`, `InverseGaussianHurdle`, `GaussianHurdle`, and
  `LognormalHurdle` accept continuous positive outcomes alongside zeros.
  Every hurdle sample needs a zero and a strictly positive outcome with
  positive weight.
- An all-zero zero-inflated sample is rejected as unidentified. A sample with
  no observed zeros is flagged as a boundary fit and inference is suppressed.
- Count distributions have a literal probability-mass interpretation for
  non-negative integer outcomes.
- The likelihood expressions use Gamma-function extensions and can be
  numerically evaluated for continuous non-negative outcomes, but this does
  not turn a count PMF into a proper continuous density. In that setting,
  coefficient estimates should be interpreted as likelihood-like or
  quasi-likelihood estimates rather than a correctly normalized count model.
- Negative generalized-Poisson dispersion has parameter-dependent support.
  Invalid trial parameters receive `-inf` likelihood so the optimizer rejects
  them. Estimates near that boundary fail inference validation.

An optimizer stopping flag alone is not treated as sufficient. The fitted
likelihood and score must be finite and the scaled first-order condition must
hold. Observed information must also be finite, full rank, sufficiently
conditioned, and positive definite; otherwise covariance and standard errors
are withheld rather than silently using a pseudoinverse.

## Distributional Models Versus GLMs

Use this package when you need joint maximum-likelihood estimation of
dispersion, zero inflation, or the complete weighted observation likelihood.

Use the [GLM package](../regression/generalized_linear_models/README.md) when
you need IRLS, its broader link/family system, sparse high-dimensional design
support, fixed NB-2 overdispersion, regularization, instrumental-variable
features, or GLM marginal effects.

Important differences include:

| Feature | `distributional_models` | GLM package |
|---|---|---|
| Optimization | BFGS likelihood optimization | IRLS |
| NB dispersion | Estimated jointly | Fixed by the user |
| Gamma dispersion | Estimated jointly | Pearson scale estimate |
| Weights | Complete likelihood weights | Variance weights |
| Zero inflation | ZIP and ZINB | Not a registered GLM family |
| Instruments | Rejected | Supported for selected workflows |
