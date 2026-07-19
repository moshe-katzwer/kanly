# kanly Linear Models

**See also:** [kanly README](../../../README.md) · [regression overview](../README.md) · [formula](../../formula/README.md) · [bootstrap](../../bootstrap/README.md)

**Related READMEs:** [robust](robust/README.md) · [quantile regression](quantile_regression/README.md) · [penalized](penalized/README.md)

`kanly.regression.linear_models` provides a unified, sparse-first linear
regression framework supporting:

- **OLS / WLS** — ordinary and weighted least squares
- **IV (2SLS / W2SLS)** — instrumental variables estimation
- **Absorbed fixed effects** — within-group demeaning (Frisch-Waugh)
- **FGLS** — feasible generalised least squares with iterative re-weighting
- **GLSAR** — feasible GLS with AR(`p`) errors; Prais–Winsten (`full_information=True`) or Cochrane–Orcutt (`full_information=False`)
- **Ridge regression** — L2-penalised OLS/WLS
- **SURE** — Seemingly Unrelated Regressions (block-diagonal joint estimation)
- **Multiple outcomes** — fit all outcomes in a single sparse pass
- **Fast path** — LSMR iterative solver (no matrix inverse, no inference)
- **Cluster / HC / HAC / Bootstrap SEs** — full sandwich estimator suite
- **Two-way clustering** — Cameron–Gelbach–Miller combination
- **Shapley R² decomposition** — Owen-style attribution by formula term (exact or permutation sampling; quadratic-form fast path)
- **Permutation tests** — non-parametric randomisation inference

---

## Quick Start

```python
from kanly.api import lm

fit = lm('y ~ x + C(grp)', df)
print(fit.summary())
print(fit.summary_df())
```

---

## Formula Syntax

| Syntax | Effect |
|---|---|
| `y ~ x + z` | OLS |
| `y ~ x + z $ w` | WLS (`w` = weight column) |
| `y ~ x + z \| z1 + z2` | IV (`z1`, `z2` = excluded instruments) |
| `y ~ x + z \| z1 $ w` | IV + WLS |
| `C(grp)` | Categorical dummies (one-hot) |
| `I(x**2)` | Inline transformation |
| `poly(x, 3)` | Degree-3 polynomial |
| `x*z` | Main effects + interaction |
| `x:z` | Interaction only |

Pass `absorb='grp'` (or `absorb=('grp', 'period')`) to absorb a variable
as a fixed effect instead of including it as dummies.

---

## OLS

```python
import numpy as np
import pandas as pd
from kanly.api import lm

n = 100
np.random.seed(0)
df = pd.DataFrame({
    'x': np.random.randn(n),
    'grp': np.random.randint(0, 12, n),
})
df['y'] = 1.2 - 0.3 * df['x'] + .2 * np.random.randn(n)

fit = lm('y ~ x + C(grp)', df, use_t=True)
print(fit.summary())       # formatted table
print(fit.summary_df())    # pandas DataFrame with coef / se / t / p / CI
```

Key result attributes:

```python
fit.params          # pd.Series of coefficients
fit.bse             # standard errors
fit.pvalues         # p-values
fit.rsquared        # R²
fit.rsquared_adj    # adjusted R²
fit.fvalue          # F-statistic
fit.llf             # log-likelihood
fit.aic, fit.bic    # information criteria
fit.resid           # residuals
fit.fittedvalues    # ŷ
```

---

## WLS (Weighted Least Squares)

Use `$` in the formula to specify a weight column:

```python
fit = lm('y ~ x + C(grp) $ obs', df)
```

Or pass the `weights` array directly via the matrix API:

```python
from kanly.regression.linear_models.model import SparseLinearModel
fit = SparseLinearModel.LM(y, X, weights=w, has_constant=True)
```

---

## Polynomial / Interaction Terms

```python
fit = lm('y ~ poly(x, 3) + z*C(grp)', df)
```

`poly(x, k)` produces an orthogonal degree-k polynomial basis.
`x*C(grp)` includes both `x`, `C(grp)`, and their interaction.

---

## Absorbed Fixed Effects

Pass `absorb=` to demean within each group without including dummy columns.
This is equivalent to including a full set of group dummies but is much
faster for high-cardinality categorical variables.

```python
# absorb a single variable
fit = lm('y ~ x', df, absorb='grp')

# absorb multiple variables (joint cross-group demeaning)
fit = lm('y ~ x', df, absorb=('grp', 'period'))
```

The result includes within R² and between R²:

```python
print(fit.absorb_info.rsquared_within)
print(fit.absorb_info.rsquared_between)
```

---

## Instrumental Variables

Use `|` to separate excluded instruments from exogenous regressors:

```python
# OLS with group dummies, IV on x (z is the excluded instrument)
fit_iv = lm('y ~ x + C(grp) | z + C(grp)', df)

# WLS-IV + absorbed FE
fit_iv_absorb = lm('y ~ x | z $ obs', df, absorb='grp')
```

Print the first-stage power table:

```python
print(fit_iv.summary_iv())   # R² and F-stat for each endogenous regressor
```

**Comparing WLS, IV, and absorbed IV** ([`examples/regression/linear_models/instrumental_variables/example_instrumental_variables.py`](../../../examples/regression/linear_models/instrumental_variables/example_instrumental_variables.py)):

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

Example output — **WLS (biased)** on endogenous `x`:

```text
Linear Model Results
WLS
Method:                         WLS    R-squared:                   0.9123
...
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

---

## Cluster-Robust Standard Errors

```python
# One-way clustering
fit = lm('y ~ x', df, cov_type='cluster', cov_kwds={'groups': 'firm_id'})

# Two-way clustering (Cameron–Gelbach–Miller) — tuple of grouping columns
fit = lm('y ~ x', df, cov_type='cluster',
         cov_kwds={'groups': ('firm_id', 'year')})

# Alternative: dedicated helper (not on kanly.api)
from kanly.api import two_way_cluster
fit = two_way_cluster('y ~ x', df, clusters=['firm_id', 'year'])
```

A helper also exists in [`two_way_cluster.py`](two_way_cluster.py) for the same estimator; prefer the tuple-`groups` pattern on `lm` when you want the standard result object API.

---

## HC Heteroscedasticity-Robust SEs

```python
fit_hc1 = lm('y ~ x', df, cov_type='HC1')
fit_hc3 = lm('y ~ x', df, cov_type='HC3')  # jackknife correction
```

Supported types: `'OLS'`, `'OLS_SMALL'` (default), `'HC0'`, `'HC1'`,
`'HC2'`, `'HC3'`, `'HAC'`, `'HAC_PANEL'`, `'CLUSTER'`, `'BOOTSTRAP'`.

---

## HAC (Newey-West) Standard Errors

```python
# Auto bandwidth (Bartlett kernel)
fit = lm('y ~ x', df, cov_type='hac')

# Custom lags and kernel
fit = lm('y ~ x', df, cov_type='hac', cov_kwds={'maxlags': 4, 'kernel': 'bartlett'})

# HAC within panels (data must be sorted by panel then time)
fit = lm('y ~ x', df_sorted, cov_type='hac_panel',
         cov_kwds={'groups': 'firm_id', 'maxlags': 3})
```

---

## Bootstrap Standard Errors

```python
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

**Bootstrap methods** (see [`kanly/bootstrap/README.md`](../../bootstrap/README.md)):

| `cov_kwds['method']` | Weights | Typical use |
| -------------------- | ------- | ----------- |
| `'bayesian'` (default) | Dirichlet(`alpha`, …, `alpha`), scaled to `nobs` or `#` clusters | Smooth, strictly positive weights; default for many kanly workflows |
| `'classical'` | Integer replication counts (rows or clusters) | Traditional with-replacement bootstrap |

With `groups` in `cov_kwds`, both methods **resample clusters** (block bootstrap): one weight vector per cluster, broadcast to member rows. Fits store `bootstrapped_params` and a bootstrap covariance; use `get_joint_bootstrapped_distribution([fit_a, fit_b])` when the same bootstrap scheme was applied to several models and you need a **joint** parameter covariance.

---

## FGLS (Feasible Generalised Least Squares)

FGLS iteratively re-weights observations to correct for heteroscedasticity.
At each iteration, log(û²) is regressed on X to estimate the
heteroscedasticity pattern, and new weights are computed from the
predicted variance.

```python
fit = lm('y ~ x + z', df, do_fgls=True,
         fgls_kwds={'maxiter': 20, 'tol': 1e-8})

print(fit.fgls_info)   # {'maxiter': 20, 'tol': 1e-08, 'err': ..., 'n_iter': ...}
```

---

## GLSAR (AR errors)

**GLSAR** (Generalized Least Squares with Autoregressive errors) fits `y ~ …` when residuals follow AR(`p`). The estimator iterates:

1. Regress `y` on `X` (OLS/GLS on whitened data after the first pass).
2. Estimate AR(`p`) on residuals (`estimate_ar`, default `ar_method='yw'`).
3. **Whiten** `y` and `X` with a sparse matrix `W` built from the AR coefficients.
4. Refit on `Wy`, `WX`; stop when AR coefficients change by less than `tol`.

Whitening is implemented in [`kanly/time_series/glsar_helper/glsar_helper.py`](../../time_series/regression/glsar_helper.py) via `make_ar_full_information_W`.

| `full_information` | Whitening | Name |
|----------------------|-----------|------|
| `True` (default) | Top block uses stationary initial covariance + innovation filter | **Prais–Winsten** |
| `False` | Innovation filter only; `nobs` and `df_resid` reduced by `nlags` | **Cochrane–Orcutt** |

**API**

```python
from kanly.api import glsar, GLSAR

fit = glsar('y ~ x + C(grp)', df, nlags=2)
fit = glsar('y ~ x', df, nlags=1, full_information=False)  # Cochrane-Orcutt

fit.glsar_info.ar_params
fit.glsar_info.full_information
```

Array API: `GLSAR(endog, exog, nlags, ...)`.

**Comparison to statsmodels**

[`statsmodels.regression.linear_model.GLSAR`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.GLSAR.html) is experimental; `whiten()` implements **Cochrane–Orcutt** only (drops the first `p` observations). kanly defaults to **Prais–Winsten**; use `full_information=False` for a Cochrane–Orcutt analogue. **FGLS** (`do_fgls=True` on `lm`) corrects heteroskedasticity, not AR serial correlation — it is a different tool.

**Restrictions:** no `absorb`, IV, WLS (`$ w`), explicit GLS `sigma`, or SURE.

### Example: OLS vs GLSAR with AR(2) errors

[`examples/regression/linear_models/example_glsar.py`](../../../examples/regression/linear_models/example_glsar.py)
simulates `y = 1.6 + x1 + e` with AR(2) innovations (`ar=[0.6, 0.15]`), then
compares OLS, `glsar(..., nlags=1)`, and `glsar(..., nlags=2)`:

```python
from kanly.api import simulate_sarima, lm, glsar, compare_results
import numpy as np
import pandas as pd

n = 500
e = simulate_sarima(n=n, ar=[.6, .15], sigma2=3.0, seed=0, burnin=500)
X = np.random.randn(n, 2)
df = pd.DataFrame(X, columns=['x0', 'x1'])
df['y'] = 1.6 + X[:, 1] + e

fits = (
    lm('y ~ x0 + x1', df),
    glsar('y ~ x0 + x1', df, nlags=1),
    glsar('y ~ x0 + x1', df, nlags=2),
)
print(compare_results(fits, ref_param_values={'Intercept': 1.6, 'x0': 0, 'x1': 1.0}))
```

Captured output (abbreviated):

| | OLS | GLSAR[1] | GLSAR[2] | True |
|---|-----|----------|----------|------|
| Intercept | 1.59 | 1.60 | 1.59 | 1.6 |
| x0 | −0.23 | −0.04 | −0.04 | 0 |
| x1 | 1.09 | 1.19 | 1.14 | 1.0 |
| R² | 0.17 | 0.40 | 0.37 | — |

OLS mis-estimates both slopes because it ignores serial correlation. GLSAR
recovers the coefficients; inspect `fit.glsar_info.ar_params` for the estimated
AR coefficients and iteration count.

---

## Ridge Regression

```python
# Uniform penalty on all variables
fit = lm('y ~ x + z', df, ridge_kwds={'alpha': 0.5})

# Per-variable penalty (dict)
fit = lm('y ~ x + z', df, ridge_kwds={'alpha': {'x': 1.0, 'z': 0.0}})

# Options
fit = lm('y ~ x + z', df, ridge_kwds={
    'alpha': 0.5,
    'normalize': True,          # scale alpha by column L2 norm (default)
    'penalize_intercept': False  # do not penalise intercept (default)
})
```

> **Note**: Inference (standard errors, t-tests) is not reliable for ridge
> regression since it is a biased estimator.  A warning is shown in the
> summary footer.

---

## SURE (Seemingly Unrelated Regression)

SURE fits multiple equations jointly with a shared block-diagonal design
matrix, enabling cross-equation Wald tests and F-tests.

```python
from kanly.api import sure

fit = sure(
    [
        {'formula': 'y1 ~ x $ wts1', 'data': df},
        {'formula': 'y2 ~ x',         'data': df},
    ],
    cov_type='cluster',
    cov_kwds={'groups': 'user_id'},
)
print(fit)
```

Cross-equation lift ratio test (e.g. comparing treatment effects across
two outcomes):

```python
result = fit.test_lift_ratio(
    treatment_index_numerator=('y1', 'x'),
    treatment_index_denominator=('y2', 'x'),
)
```

---

## Multiple Outcomes

When the formula's left-hand side is a list (or `endog` has multiple columns),
`fit` returns a `{outcome_name: result}` dict:

```python
from kanly.regression.linear_models.model import SparseLinearModel
fits = SparseLinearModel.LM(
    endog=Y,  # shape (n, k) for k outcomes
    exog=X,
    endog_name=['y1', 'y2'],
    has_constant=True,
)
print(fits['y1'])
print(fits['y2'])
```

---

## Prediction

```python
# In-sample (returns stored fittedvalues)
y_hat = fit.predict()

# Out-of-sample (re-evaluates the formula on new_df)
y_hat_oos = fit.predict(data=new_df)

# Custom parameter vector
y_hat_custom = fit.predict(params=np.zeros(len(fit.params)))
```

Prediction is not yet supported for models with absorbed fixed effects or
instrumental variables when out-of-sample data is provided.

---

## Large Sparse Regressions / Fast Path

For very large design matrices (millions of rows, thousands of columns)
where computing the matrix inverse is the bottleneck, use the LSMR fast
path.  No standard errors or covariance matrix are computed.

```python
from kanly.api import lm, lm_fast

# Full inference (slow for very large problems)
fit = lm('y ~ x*C(g) + I(x**2) + poly(z, 3)', df)

# Fast path — coefficients only, no inference
fit_fast = lm_fast('y ~ x*C(g) + I(x**2) + poly(z, 3)', df)

print(fit.fit_elapsed, fit_fast.fit_elapsed)
```

## Performance vs statsmodels

For large linear models, especially with high-cardinality fixed effects, kanly
is often **orders of magnitude faster** than
[`statsmodels`](https://www.statsmodels.org/) `smf.ols` while matching the
same core inference targets (`params`, `bse`, `tvalues`) for the slope terms.

Runnable benchmark:
[`examples/regression/linear_models/example_kanly_vs_statsmodels_time.py`](../../../examples/regression/linear_models/example_kanly_vs_statsmodels_time.py).
It fits 3M observations with 20 slopes and 180 group levels (201 regressors
with dummies), all with `cov_type='HC1'`.

| Fit | Elapsed (approx.) |
|-----|-------------------|
| kanly `lm('y ~ x1 + ... + C(g)', ...)` | ~16 s |
| kanly `lm(..., absorb='g')` | ~13 s |
| statsmodels `smf.ols(...).fit(cov_type='HC1')` | ~18 min |

Two implementation differences explain most of the gap:

1. **Sparse designs** — kanly builds sparse `X` and forms `X'X` / `X'y`
   without densifying the full `n × k` design matrix.
2. **Normal equations** — kanly solves `beta = inv(X'X) @ (X'y)` (`k × k` and
   `k × 1`). A common statsmodels path uses `(inv(X'X) @ X') @ y`, where
   `inv(X'X) @ X'` is `k × n` and dense even when `X` is sparse.

You do not trade inference for speed: `lm` still supports robust and clustered
SEs, confidence intervals, prediction, Wald/F tests, and additional linear-model
capabilities (`absorb=`, IV, FGLS, `glsar`, ridge, SURE, Shapley R², permutation tests).

---

## Shapley R² Decomposition

Owen (2000)-style Shapley decomposition of R² across **formula terms**
(e.g. `'x'`, `'C(grp)'`), not individual dummy columns. Each term gets the
average marginal contribution to R² over all entry orderings (exact mode) or
a Monte Carlo average along random permutations (`sample=k`).

After one full-model fit, subset R² is evaluated from the weighted normal
equations (`XtX`, `Xty`) via closed-form β — **no refit per subset**. The
intercept is always in subset models but is not a Shapley player; the baseline
before the first term is R² = 0.

| Mode | When to use | Cost |
|------|-------------|------|
| **Exact** (`sample=False`, default) | Moderate number of terms `p` (roughly ≲ 15) | `2^p − 1` subset R² evaluations |
| **Permutation** (`sample=k`) | Large `p` | `O(k · p)` evaluations; exact is used automatically when `k·p ≥ 2^p − 1` |

Output columns: `shapley_value` (additive R² share) and `pct`
(`shapley_value / full_model_R²`). In exact mode, `shapley_value` sums to the
fitted model's R².

**Requirements:** explicit intercept in column 0; **IV models are not supported**
(no quadratic form).

### From a fitted model

```python
from kanly.api import lm

fit = lm('y ~ x + C(grp)', df)
tab = fit.shapley_value()
meta = fit.shapley_value(return_full=True)  # dict: shapley_values, num_regressions, ...
print(meta['shapley_values'])
```

### Standalone API

```python
from kanly.api import shapley_value

# Formula string (fits once, then decomposes)
tab = shapley_value('y ~ x + C(grp)', df)

# Tuple: (endog_name, exog_term_names[, weights_column])
tab = shapley_value(('y', ['x', 'C(grp)']), df)
tab = shapley_value(('y', ['x', 'C(grp)'], 'w'), df)  # WLS

# Many sparse_terms: approximate with random permutations
tab = shapley_value('y ~ x1 + x2 + x3 + C(grp)', df, sample=200, seed=1)
```

Example output:

```text
               shapley_value       pct
C(grp)             0.4231    0.6591
x                  0.2190    0.3409
```

See [`example_shapley_value.py`](../../../examples/regression/linear_models/example_shapley_value.py) for exact vs sampled timing on a 13-term formula model.

---

## Permutation Test

Computes the null distribution of a treatment parameter by repeatedly
shuffling the treatment label and re-fitting the model.

```python
from kanly.regression.linear_models.permutation_test import permutation_test

null_dist = permutation_test(
    'y ~ treatment + x', 'treatment', df,
    num_permutations=1000, seed=42,
)

# Empirical two-sided p-value
obs = fit.params['treatment']
p_val = (null_dist['treatment'].abs() >= abs(obs)).mean()
```

Within-group permutation (preserves cluster structure):

```python
null_dist = permutation_test(
    'y ~ treatment + x', 'treatment', df,
    groups='firm_id', num_permutations=1000,
)
```

---

## Matrix API

For pre-built arrays, use the matrix entry points directly:

```python
from kanly.regression.linear_models.model import SparseLinearModel

# Full inference
fit = SparseLinearModel.LM(y, X, weights=w, instruments=Z,
                           exog_names=['Intercept', 'x1', 'x2'],
                           has_constant=True)

# Fast path (no matrix inverse)
result = SparseLinearModel.LM_fast(y, X, weights=w,
                                   exog_names=['Intercept', 'x1', 'x2'])
print(result['params'])
```

---

## Reading Results

```python
print(fit.summary())          # full formatted text summary
print(fit.summary_df())       # pandas DataFrame: coef / se / t / p / CI
fit.plot_diagnostics()        # 6-panel residual diagnostic plot

# Access individual fields
fit.params                    # pd.Series of coefficients
fit.bse                       # standard errors
fit.conf_int()                # confidence intervals (DataFrame)
fit.pvalues                   # p-values
fit.tvalues                   # t-statistics
fit.rsquared                  # R²
fit.rsquared_adj              # adjusted R²
fit.fvalue, fit.f_pvalue      # F-statistic and p-value
fit.llf, fit.aic, fit.bic     # log-likelihood, AIC, BIC
fit.condition_number          # condition number of X'X
fit.eigenvals                 # eigenvalues of X'X

# Wald / F-tests
fit.wald_test(['x1', 'x2'])   # joint significance
fit.F_test()                  # overall F-test using current cov

# Lift and ratio tests
fit.test_lift('treatment')         # ATE / control baseline
fit.test_lift_interacted(
    numer_dict={'treatment': 1, 'I(x*treatment)': 3},
    denom_dict={'x': 3})           # local ATE at x=3
```

---

## Examples in this repo

- [`examples/regression/linear_models/example_ordinary_least_squares.py`](../../../examples/regression/linear_models/example_ordinary_least_squares.py), [`example_weighted_least_squares.py`](../../../examples/regression/linear_models/example_weighted_least_squares.py) — basic OLS / WLS.
- [`examples/regression/linear_models/example_absorbed_fixed_effects.py`](../../../examples/regression/linear_models/example_absorbed_fixed_effects.py) — `absorb=`.
- [`instrumental_variables/example_instrumental_variables.py`](../../../examples/regression/linear_models/instrumental_variables/example_instrumental_variables.py) — 2SLS formulas.
- [`example_clustered_ses.py`](../../../examples/regression/linear_models/example_clustered_ses.py), [`example_clustered_ses_2way.py`](../../../examples/regression/linear_models/example_clustered_ses_2way.py) — one- vs two-way cluster SEs (with `compare_results`).
- [`example_bootstrap.py`](../../../examples/regression/linear_models/example_bootstrap.py), [`example_block_bootstrap.py`](../../../examples/regression/linear_models/example_block_bootstrap.py) — bootstrap covariances.
- [`example_fast_lm.py`](../../../examples/regression/linear_models/example_fast_lm.py) — `lm` vs `lm_fast` timing on millions of rows.
- [`example_large_sparse_regression.py`](../../../examples/regression/linear_models/example_large_sparse_regression.py) — huge sparse `SparseDataFrame` + `lm`.
- [`example_feasible_generalized_least_squares.py`](../../../examples/regression/linear_models/example_feasible_generalized_least_squares.py), [`example_ridge.py`](../../../examples/regression/linear_models/example_ridge.py).
- [`example_glsar.py`](../../../examples/regression/linear_models/example_glsar.py) — OLS vs GLSAR(1)/GLSAR(2) with AR errors (`compare_results`).
- [`example_sure.py`](../../../examples/regression/linear_models/example_sure.py), [`example_seemingly_unrelated_regression.py`](../../../examples/regression/linear_models/example_seemingly_unrelated_regression.py).
- [`example_multiple_outcomes.py`](../../../examples/regression/linear_models/example_multiple_outcomes.py), [`example_linear_model_prediction.py`](../../../examples/regression/linear_models/example_linear_model_prediction.py).
- [`example_shapley_value.py`](../../../examples/regression/linear_models/example_shapley_value.py), [`example_polynomial_regression.py`](../../../examples/regression/linear_models/example_polynomial_regression.py).
- [`example_ordinary_least_squares_with_indexing.py`](../../../examples/regression/linear_models/example_ordinary_least_squares_with_indexing.py) — row `index` subsets.

Also: [`examples/regression/example_ridge_different_ways.py`](../../../examples/regression/example_ridge_different_ways.py), [`examples/regression/example_lasso_different_ways.py`](../../../examples/regression/example_lasso_different_ways.py) — ridge / LASSO comparisons across APIs.
