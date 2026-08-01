# Distributional Models User Guide

**See also:** [kanly README](../../README.md) ·
[formula guide](../formula/README.md) ·
[generalized linear models](../regression/generalized_linear_models/README.md)

This package provides compact, likelihood-based regression models for counts
and positive continuous outcomes. Unlike the IRLS generalized-linear-model
implementation, these classes jointly optimize all parameters in their
likelihood, including Gamma and negative-binomial dispersion parameters.

The public base class is `DistributionalModel`. Source files are organized by
response structure:

- [`count_models.py`](count_models.py) contains the base class and discrete
  count, zero-inflated, and negative-binomial likelihoods.
- [`continuous_models.py`](continuous_models.py) contains the continuous
  `Gamma` likelihood model.
- [`hurdle_models.py`](hurdle_models.py) contains GLM-composed Poisson and
  Gamma hurdle models.
- [`results.py`](results.py) contains the shared fitted-results object.

All public model classes can be imported directly from
`kanly.distributional_models`.

## Public API and Available Models

```python
from kanly.distributional_models import (
    DistributionalModel,
    DistributionalModelResults,
    Gamma,
    GammaHurdle,
    GeneralizedPoisson,
    NegativeBinomial1,
    NegativeBinomial2,
    Poisson,
    PoissonHurdle,
    ZeroInflatedNegativeBinomial,
    ZeroInflatedPoisson,
)
```

`DistributionalModel` supplies formula construction, likelihood weighting,
optimization, observation scores, covariance estimation, bootstrap refits,
and common prediction behavior. Concrete subclasses provide the distribution
likelihood and parameter names.

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
| `GammaHurdle` | Logit zero hurdle plus Gamma/log GLM | Two-part model | Hurdle coefficients; Pearson GLM scale |

For models reporting `log_alpha`, the positive dispersion is
`alpha = exp(log_alpha)`. Generalized Poisson estimates raw `alpha` because
valid negative values permit underdispersion.

`Gamma`, imported from `continuous_models.py`, is a continuous distribution;
its outcome must be finite and strictly positive. Zero-inflation and hurdle
probabilities use a logit. `GammaHurdle` uses a log positive-response link by
default and permits another supported GLM link through `positive_link`.

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
    start_params=np.array([0.0, 0.0, 0.0, -1.0]),
    cov_type="SANDWICH",
)

print(fit.summary_df())
```

The `$` formula extension supplies optional observation likelihood weights.
Instrumental-variable and absorbed-effect formulas are rejected.

Formula construction records column names and row-alignment metadata on the
model. For example, `fit.param_names` uses formula names rather than generated
names such as `x0` and `x1`.

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
    start_params=np.zeros(len(model.param_names)),
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
positive counts. `GammaHurdle` uses the ordinary Gamma family because Gamma
already has support on `(0, infinity)`; its positive conditional mean uses a
log link by default.

```python
from kanly.distributional_models import PoissonHurdle

model = PoissonHurdle.build_model_from_formula(
    "orders ~ price + promotion $ weight",
    data=df,
    exog_infl="customer_age + C(region)",
)
fit = model.fit(cov_type="SANDWICH")

print(fit.summary_df())
print(fit.positive_fit)  # positive-response GLM
print(fit.hurdle_fit)    # Bernoulli/logit GLM for P(Y=0)
```

`exog_infl=None` creates an intercept-only zero hurdle. Instruments remain
unsupported and are rejected by formula construction. Parameters are ordered
as the positive-response coefficients followed by coefficients named
`hurdle_...`.

The likelihood is separable, so `fit` estimates the two component GLMs rather
than jointly optimizing the combined likelihood. The returned covariance is
block diagonal and preserves the covariance block from each component fit.
`cov_type="SANDWICH"` maps to the GLM `HC1` estimator; `NONROBUST`, `HC1`, and
`BOOTSTRAP` are also accepted.

The combined object provides `loglike`, `loglike_obs`, `score`, `score_obs`,
and predictions for the overall mean, conditional positive mean, zero
probability, or positive probability. The combined distributional-model API
keeps `loglike_obs` and `score_obs` unweighted and applies observation weights
only when aggregating them. During component estimation those weights are
supplied through the GLM variance-weight interface. This is identical to
likelihood weighting for fixed-scale Bernoulli and zero-truncated Poisson; the
Gamma component's fitted scale and native covariance retain the usual GLM
precision-weight interpretation.

Complete parameter-recovery examples are available for both
[`PoissonHurdle`](../../examples/distributional_models/example_poisson_hurdle.py)
and
[`GammaHurdle`](../../examples/distributional_models/example_gamma_hurdle.py).

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
    weights_name="frequency_weight",
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

Starting values must contain exactly one entry for every name returned by
`model.get_param_names()` or `model.param_names`.

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

This differs from GLM variance weights. It is appropriate for likelihood,
frequency, or importance weighting when multiplying the entire observation
contribution is the intended estimand.

Weights must be finite and non-negative. With formula construction,
`sum_to_n=True` rescales them to sum to the final aligned number of
observations.

## Covariance Estimation

`fit` accepts three covariance types:

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

### Bayesian bootstrap

`BOOTSTRAP` draws Dirichlet observation weights, multiplies them by any
existing model weights, and refits the model. Every refit starts from the
original point estimate. Failed or non-finite repetitions are excluded.

Bootstrap results expose:

```python
fit.bootstrapped_params
fit.cov_params()
fit.bse
fit.cov_kwds["n_successful"]
fit.cov_kwds["n_failed"]
```

The model's original likelihood weights are restored after every refit,
including when optimization fails.

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
| `summary_df()` | Coefficient, standard error, z statistic, p-value, and CI table |
| `llf` / `average_loglike` | Total and per-observation fitted log likelihood |
| `aic` / `bic` | Information criteria using every estimated parameter |
| `fittedvalues` / `resid_response` | Unconditional fitted means and response residuals |
| `dispersion` | Transformed dispersion for Gamma, NB-1, NB-2, and ZINB, or raw GP alpha |
| `zero_probability` / `positive_probability` | Fitted outcome probabilities for two-part models |
| `inflation_probability` / `count_mean` | ZIP/ZINB structural-zero probability and count-component mean |
| `positive_fit` / `hurdle_fit` | Original component GLM results for hurdle models |
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
- Zero-inflated models require finite, non-negative outcomes.
- Count distributions have a literal probability-mass interpretation for
  non-negative integer outcomes.
- The likelihood expressions use Gamma-function extensions and can be
  numerically evaluated for continuous non-negative outcomes, but this does
  not turn a count PMF into a proper continuous density. In that setting,
  coefficient estimates should be interpreted as likelihood-like or
  quasi-likelihood estimates rather than a correctly normalized count model.
- Negative generalized-Poisson dispersion has parameter-dependent support.
  Invalid trial parameters receive `-inf` likelihood so the optimizer rejects
  them.

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
