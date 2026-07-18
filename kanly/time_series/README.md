# `kanly.time_series`

**See also:** [kanly README](../../README.md) · [linear models GLSAR](../regression/linear_models/README.md#glsar-ar-errors)

Univariate and regression-based time-series tools:

- **Exploratory:** sample ACF / PACF
- **AR(p) estimation:** conditional/exact likelihood and moment methods via [`estimate_ar`](autoregression/estimate_ar.py)
- **OLS-based dynamics:** [`autoreg`](autoregression/autoreg.py) / `AUTOREG` / `ARDL` (lagged regressors + trends + seasonals)
- **GLSAR:** feasible GLS for linear models with AR errors ([`glsar`](../regression/linear_models/model.py), whitening in [`glsar_helper/`](glsar_helper/))
- **SARIMAX:** seasonal ARIMA with exogenous regressors ([`sarimax/`](sarimax/))

Much of the ARMA/SARIMA state-space and simulation notation follows Brockwell & Davis (2016); see [**References**](#references).

Implementation lives under `kanly/time_series/`. Folders such as `to_delete2/` and scratch notebooks are legacy and not public API.

---

## Comparison: SciPy / statsmodels / scikit-learn

| | SciPy | statsmodels | scikit-learn | kanly |
| --- | ----- | ----------- | ------------ | ----- |
| **Time-series modelling** | [`signal`](https://docs.scipy.org/doc/scipy/reference/signal.html), [`fft`](https://docs.scipy.org/doc/scipy/reference/fft.html) — useful primitives, **no** unified SARIMAX likelihood / Kalman stack | [`tsa`](https://www.statsmodels.org/stable/tsa.html): `SARIMAX`, `AutoReg`, `ARDL`, `GLSAR`, `acf`, `pacf` | No first-class SARIMAX; not the usual tool for ARIMA | `SARIMAX` / `sarimax`, `simulate_sarima`, `estimate_ar`, `AUTOREG` / `autoreg`, `ARDL`, `glsar`, `acf` / `pacf` |
| **Practical note** | You compose filters and fits yourself | Mature reference; **kanly** aims for similar *spirit*, not guaranteed numerical parity (see [Notes and caveats](#notes-and-caveats)) | Often paired with **statsmodels**, **pmdarima**, or **Prophet** for this workload | Formula path for exogenous regressors matches regression-style APIs elsewhere in kanly |

### SARIMAX quick start (array API)

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

---

## Public entry points (`kanly.api`)

| Symbol | Module | Role |
|--------|--------|------|
| `acf`, `pacf` | [`auto_correlation_function.py`](auto_correlation_function.py) | Sample ACF (FFT) and PACF (Durbin–Levinson) |
| `estimate_ar`, `css`, `yule_walker`, `burg`, `ar_mle` | [`autoregression/estimate_ar.py`](autoregression/estimate_ar.py) | Standalone AR(`p`) coefficient estimation (multiple methods) |
| `autoreg`, `AUTOREG` | [`autoregression/autoreg.py`](autoregression/autoreg.py) | AR regression via OLS — formula or array API |
| `ARDL` | [`autoregression/autoreg.py`](autoregression/autoreg.py) | Autoregressive distributed lag (lagged `y` and lagged `X`) via OLS |
| `SARIMAX` / `ARIMA` / `arima` / `sarimax` | [`sarimax/model.py`](sarimax/model.py) | SARIMAX-style models (Kalman likelihood) |
| `simulate_sarima` | [`sarimax/simulate.py`](sarimax/simulate.py) | Simulate ARMA / SARIMA paths |
| `glsar`, `GLSAR` | [`regression/linear_models/model.py`](../regression/linear_models/model.py) | Feasible GLS when **regression residuals** follow AR(`p`); whitening in [`glsar_helper/`](glsar_helper/) |

```python
from kanly.api import (
    acf, pacf, estimate_ar, autoreg, AUTOREG, ARDL,
    SARIMAX, sarimax, simulate_sarima, glsar,
)
```

---

## Autocorrelation (`acf`, `pacf`)

[`auto_correlation_function.py`](auto_correlation_function.py) provides exploratory correlation summaries for a 1-D series. The API follows the spirit of [`statsmodels.tsa.stattools.acf`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html) / [`pacf`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.pacf.html) (FFT ACF; Durbin–Levinson PACF).

### `acf`

```python
import numpy as np
from kanly.api import acf

x = ...  # 1-D array
vals = acf(x, nlags=40, adjusted=True)
vals, se = acf(x, nlags=40, bartlett_std_err=True)  # optional Bartlett SEs
```

- `adjusted=True` (default): divide autocovariances by lag-specific sample counts.
- `bartlett_std_err=True`: return `(acf, std_err)` for confidence bands (`1.96 * se` is a common 95% rule-of-thumb).
- Constant or near-constant series return a degenerate ACF (`1` at lag 0, zeros elsewhere).

`autocovariance_function(...)` scales the ACF by the sample variance.

### `pacf`

```python
from kanly.api import pacf

pacf_vals = pacf(x, nlags=15)
```

PACF is computed from the sample ACF via the Durbin–Levinson recursion (Brockwell & Davis, 2016, §2.5.1; see [References](#references)). For theory-only ACFs from AR/MA coefficients, `kanly.time_series.sarimax.arma_innovation_functions.get_acf` is available (used in [`examples/time_series/example_acf.py`](../../examples/time_series/example_acf.py)).

### Example

[`examples/time_series/example_acf.py`](../../examples/time_series/example_acf.py) simulates an ARMA series, plots `acf` / `pacf`, and compares to `get_acf` from the fitted polynomial.

---

## AR(`p`) estimation (`estimate_ar`)

[`autoregression/estimate_ar.py`](autoregression/estimate_ar.py) fits a **univariate** AR process on a 1-D series `y`. The assumed mean model is

```text
y[t] = const + φ1·y[t-1] + … + φL·y[t-L] + e[t],   e[t] ~ N(0, σ²)
```

This is the **level/intercept parameterization** (regression of `y` on its lags). It differs from SARIMAX, where the AR polynomial applies to the **innovation** `e[t]` inside the state-space error; the AR coefficients `φ` match, but intercepts relate by `const (here) = (SARIMAX intercept) × (1 − Σ φ_i)`. See the module docstring in `estimate_ar.py` for details.

### Methods (`estimate_ar(..., method=...)`)

| `method` | Name | Idea | Intercept | Covariance |
|----------|------|------|-----------|------------|
| `'css'` | Conditional sum of squares | OLS of `y[t]` on intercept + lags (uses `t ≥ L`) | Yes | OLS `(X'X)⁻¹` |
| `'yw'`, `'yule-walker'` | Yule–Walker | Durbin–Levinson on sample autocovariances | No (demean first) | Toeplitz large-sample approx. |
| `'burg'` | Burg | Minimize forward/backward one-step prediction errors | Sample mean | Not intrinsic (use bootstrap / OLS approx.) |
| `'mle_conditional'` | Conditional Gaussian MLE | Same spirit as CSS; likelihood via AR filter | Yes | From likelihood / optimizer |
| `'mle_exact'` | Exact Gaussian MLE | Full likelihood incl. initial `p` obs (Toeplitz block) | Yes | Likelihood-based |

All methods return a **dict** with keys such as `params`, `param_names`, `arparams`, `const`, `sigma2`, `cov_params`, and often `llf` for MLE variants.

```python
from kanly.api import simulate_sarima, estimate_ar

y = simulate_sarima(n=200, ar=[0.4, -0.15], sigma2=3.0, seed=0, burnin=500)

res_css = estimate_ar(y, lags=2, method='css')
res_yw  = estimate_ar(y, lags=2, method='yw')
res_burg = estimate_ar(y, lags=2, method='burg')
res_exact = estimate_ar(y, lags=2, method='mle_exact')

print(res_css['params'], res_css['param_names'])
```

Lower-level functions are also exported: `css`, `yule_walker`, `burg`, `ar_mle`.

**Example:** [`examples/time_series/autoregression/example_estimate_AR.py`](../../examples/time_series/autoregression/example_estimate_AR.py) simulates AR(2) data and compares CSS, Yule–Walker, Burg, conditional MLE, exact MLE, `AUTOREG`, and `SARIMAX` against known coefficients.

---

## AUTOREG and ARDL (`autoreg`, `AUTOREG`, `ARDL`)

[`autoregression/autoreg.py`](autoregression/autoreg.py) provides **OLS-based** dynamic regression — not a separate likelihood engine. Lagged `y`, trends, seasonals, and exogenous columns are assembled into a design matrix, then [`SparseLinearModel.lm` / `LM`](../regression/linear_models/model.py) fits the model. Conceptually similar to statsmodels [`AutoReg`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.ar_model.AutoReg.html) and [`ARDL`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.ardl.ARDL.html), but integrated with kanly formulas and sparse matrices.

### `autoreg` (formula) and `AUTOREG` (arrays)

| API | Use when |
|-----|----------|
| `autoreg(formula, data, lags=..., trend=..., seasonal_periods=...)` | Patsy-like formula; adds `L(y, k)`, `trend(...)`, `seasonal(...)` |
| `AUTOREG(endog, exog=..., lags=..., trend=..., seasonal_periods=...)` | NumPy / sparse arrays; lag columns `L1[y]`, … |

- **`lags`:** int (1…L) or iterable of specific lags.
- **`trend`:** `'n'`, `'c'`, `'t'`, `'ct'`, `'ctt'`, or a list of polynomial powers (`0` = intercept).
- **`seasonal_periods`:** e.g. `12` for monthly dummies, or `[7, 365]`.
- First `lags` rows are dropped so all lag regressors are observed.
- Inherits **`lm`** options: `cov_type`, bootstrap, cluster SEs, etc.

```python
from kanly.api import autoreg, AUTOREG

# Formula: AR(2) + exog + quadratic trend + monthly seasonals
fit = autoreg('y ~ x0 + x1', df, lags=2, trend='ctt', seasonal_periods=12)

# Array API — equivalent structure
fit = AUTOREG(y, exog=X, lags=2, trend='ctt', seasonal_periods=12)
```

**Example:** [`examples/time_series/autoregression/example_autoreg.py`](../../examples/time_series/autoregression/example_autoreg.py) — formula vs array `AUTOREG` with trends and seasonals.

### `ARDL` (autoregressive distributed lag)

`ARDL(endog, exog, order=..., lags=..., ...)` extends `AUTOREG` with **lags of exogenous regressors**:

- **`lags`:** AR order on `endog` (same as `AUTOREG`).
- **`order`:** lag structure on each exogenous column — int `q` → lags `0,1,…,q`; dict for per-column specs; `causal=True` drops contemporaneous `exog[t]`.
- Delegates to the same OLS stack after building the combined lag design.

```python
from kanly.api import ARDL
import numpy as np

y = ...
X = np.column_stack([x1, x2])
fit = ARDL(y, X, lags=2, order=1, trend='c')  # AR(2), one lag of each exog
```

Use **`estimate_ar` / MLE / SARIMAX`** when you want likelihood-based AR on a single series; use **`AUTOREG` / `ARDL`** when the model is a **regression with lags, trends, and seasonals** and OLS inference is enough.

---

## GLSAR (regression with AR errors)

**GLSAR** is different from `AUTOREG` and `estimate_ar`: it fits **`y ~ X`** (a linear regression) when the **regression residuals** follow AR(`p`), not when `y` itself is modeled as an autoregression.

Implementation: [`SparseLinearModel.glsar` / `GLSAR`](../regression/linear_models/model.py) iterates AR estimation on residuals → **whiten** `(y, X)` → GLS refit until AR coefficients stabilize. Whitening matrices live in [`glsar_helper/glsar_helper.py`](glsar_helper/glsar_helper.py) (`make_ar_full_information_W`, `fit_glsar_internal`).

| `full_information` | Whitening | Name |
|--------------------|-----------|------|
| `True` (default) | Stationary initial covariance + innovation filter | **Prais–Winsten** |
| `False` | Innovation rows only; `nobs` reduced by `p` | **Cochrane–Orcutt** (closer to statsmodels [`GLSAR`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.GLSAR.html)) |

```python
from kanly.api import glsar

fit = glsar('y ~ x0 + x1', df, nlags=2)
fit.glsar_info.ar_params   # final AR coefficients on residuals
```

**Not supported** with absorb, IV, WLS weights, or known GLS `sigma`. Full discussion: [linear models README — GLSAR](../regression/linear_models/README.md#glsar-ar-errors).

**Examples:**

- [`examples/time_series/autoregression/example_glsar.py`](../../examples/time_series/autoregression/example_glsar.py)
- [`examples/regression/linear_models/example_glsar.py`](../../examples/regression/linear_models/example_glsar.py) — OLS vs GLSAR(1)/GLSAR(2) with `compare_results`.

---

## Simulation (`simulate_sarima`)

[`simulate_sarima`](sarimax/simulate.py) draws a **univariate Gaussian SARIMA** path. It is useful on its own (power studies, teaching, checking `acf` / `pacf`, generating data before a `sarimax` fit) and shares the same lag polynomial conventions as the SARIMAX fitter.

### How it works

1. **Expand lags** — nonseasonal `ar` / `ma` and seasonal `sar` / `sma` are merged into single AR and MA polynomials (seasonal coefficient `sar[i]` attaches to lag `(i + 1) * s`). Nonseasonal and seasonal differencing (`d`, `D`) enter the AR polynomial via `get_combined_differencing_coefs`.
2. **Simulate** — Gaussian innovations with variance `sigma2` drive a recursive ARMA update (Numba-accelerated internal loop).
3. **Burn-in** — simulate `int(n * (1 + burnin))` points and return the **last** `n` observations so transients from initial conditions fade.
4. **Demean** — when `demean=True` (default), subtract the sample mean of the returned segment.

Regular and seasonal lag lists are checked for **overlap** (`check_intersection`); conflicting specs raise `ValueError`. If `sar` or `sma` is non-empty, `s` must be at least 2.

### Coefficient indexing

| Argument | Meaning | Lag for index `i` |
|----------|---------|-------------------|
| `ar` | Nonseasonal AR | `i + 1` |
| `ma` | Nonseasonal MA | `i + 1` |
| `sar` | Seasonal AR | `(i + 1) * s` |
| `sma` | Seasonal MA | `(i + 1) * s` |
| `d` | Nonseasonal differencing order | (embedded in AR polynomial) |
| `D` | Seasonal differencing order | (embedded in AR polynomial) |
| `s` | Seasonal period | default `2` |

Omit a term with `[]` or `None` (defaults to empty). This matches the list form accepted in `order` / `seasonal_order` when you pass explicit lag tuples to `SARIMAX`.

**Rough mapping to `SARIMAX` orders:** `order=(p, d, q)` with integer `p`, `q` corresponds to `ar` / `ma` of length `p` / `q`; `seasonal_order=(P, D, Q, s)` corresponds to `sar` / `sma` of length `P` / `Q` and period `s`. Simulation does **not** include exogenous regressors, intercepts, or trends—add those manually to the simulated series if needed (as in [`example_arma.py`](../../examples/time_series/sarimax/example_sarimax_with_ar_and_x.py)).

### Parameters

| Parameter | Default | Role |
|-----------|---------|------|
| `n` | (required) | Length of series returned **after** burn-in |
| `sigma2` | `1.0` | Innovation variance |
| `burnin` | `1.0` | Extra initial fraction of `n` to simulate and discard (`burnin=1` → `2n` draws, keep last `n`) |
| `seed` | `0` | NumPy RNG seed for innovations |
| `demean` | `True` | Subtract mean of returned segment |

Use a larger `burnin` (e.g. `2.0` or `10`) when AR roots are close to the unit circle or when differencing leaves a long transient.

### Examples

**AR(2):**

```python
from kanly.api import simulate_sarima, acf

y = simulate_sarima(n=500, ar=[0.5, 0.1], seed=0, burnin=1.0)
acf(y, nlags=5)
```

**ARMA with a specific MA lag** (index `4` is lag 5):

```python
y = simulate_sarima(n=20_000, ar=[0.8, -0.3], ma=[0, 0, 0, 0, 0.5], seed=0, sigma2=1.3)
```

**Seasonal ARMA** — e.g. AR(1) with seasonal MA(1) at period 12:

```python
y = simulate_sarima(n=400, ar=[0.4], sma=[0.6], s=12, seed=1, burnin=1.0)
```

**Integrated / seasonal integration** — differencing enters the AR polynomial:

```python
y = simulate_sarima(
    n=500,
    ar=[0.5],
    ma=[-0.2],
    d=1,
    sar=[0.3],
    D=0,
    s=12,
    sigma2=1.0,
    seed=0,
    demean=True,
)
```

### Relation to fitting

A common workflow is **simulate → explore → fit**:

```python
import numpy as np
import pandas as pd
from kanly.api import simulate_sarima, sarimax

u = simulate_sarima(n=3_000, ar=[0.4, 0.1], seed=0, sigma2=1.3)
x = np.random.randn(3_000)
y = u + 1.5 * x
df = pd.DataFrame({"y": y, "x": x})

fit = sarimax("y ~ x", df, order=(2, 0, 0), trend=(1, 1, 1))
```

See [`examples/time_series/sarimax/example_arma.py`](../../examples/time_series/sarimax/example_sarimax_with_ar_and_x.py) and [`examples/time_series/example_acf.py`](../../examples/time_series/example_acf.py).

**Limitations:** Gaussian innovations only; no built-in non-Gaussian or stochastic-volatility extensions. For fully general state-space simulation with measurement error and time-varying matrices, use **statsmodels** or custom code.

---

## SARIMAX / ARIMA (`sarimax/`)

[`kanly.time_series.sarimax`](sarimax/) estimates seasonal ARIMA models with optional exogenous regressors and deterministic trend terms. `SARIMAX`, `ARIMA`, and `arima` are the same class; `sarimax(formula, data, ...)` matches the regression-style formula workflow used elsewhere in kanly.

### Quick start

```python
import numpy as np
from kanly.api import SARIMAX

y = np.asarray(...)
x = np.asarray(...).reshape(-1, 1)

res = SARIMAX(
    y,
    exog=x,
    order=(1, 1, 1),
    seasonal_order=(1, 0, 1, 12),
    trend="c",
)

print(res)
forecast = res.get_forecast(steps=12, exog=np.zeros((12, 1)))
```

Formula interface:

```python
from kanly.api import sarimax

res = sarimax(
    "sales ~ price + promo",
    data=df,
    order=(1, 0, 1),
    seasonal_order=(0, 1, 1, 52),
)
```

### Model specification

- `order=(p, d, q)` — nonseasonal ARIMA. `p` and `q` may be integers (all lags up to that order) or explicit lag iterables such as `(1, 3, 7)`. `d` is ordinary differencing.
- `seasonal_order=(P, D, Q, s)` — seasonal AR, seasonal differencing, seasonal MA, and period. Seasonal lags are multiples of `s`; overlapping nonseasonal and seasonal lag terms are rejected.
- `trend` — `None`, `"c"`, `"t"`, `"ct"`, `"n"`, or an indicator list. Constant and low-order trends are restricted when differencing would leave them unidentified.
- `exog` must have the same number of rows as `endog`. Forecasts with exogenous regressors require future `exog` with one row per step.

### Fitting options

The fit path validates orders, builds Hannan–Rissanen starts when requested, optimizes a Gaussian Kalman likelihood with bounded projected quasi-Newton ([`kanly.optimize`](../optimize/README.md)), then computes covariance estimates.

Notable options:

- `simple_differencing=True` — difference data before filtering; when `False`, integration is in the state vector and initial observations are burned from likelihood summaries.
- `concentrate_scale=True` — optimize all parameters except `sigma2`, then set `sigma2` from forecast errors.
- `standardize_endog=True` — fit on a standardized scale and map parameters back.
- `enforce_stationarity` / `enforce_invertibility` — Monahan-style transforms on AR/MA starts.
- `cov_type` — `"opg"`, `"approx"`, `"robust_approx"`, `"none"`.
- `multiplicative=True` — not implemented in the current fit path.

### Results and forecasting

`SarimaxResults` includes parameters, standard errors, information criteria, residuals, fitted values, per-observation log likelihood, roots, stationarity/invertibility flags, and summary output.

`get_forecast(steps, exog=None, return_prediction_variance=False, signal_only=False)` forecasts the ARMA state, adds differencing lags back to the original scale, then adds future exogenous and trend terms unless `signal_only=True`.

`loglikelihood_burn` is `d + D * s`. With simple differencing, leading `NaN` padding aligns residuals and fitted values to the original index; with state-space integration, the burn applies to likelihood summaries only.

### SARIMAX examples

- [`examples/time_series/sarimax/example_arma.py`](../../examples/time_series/sarimax/example_sarimax_with_ar_and_x.py) — `simulate_sarima` + formula `sarimax` with trend.
- [`examples/time_series/example_acf.py`](../../examples/time_series/example_acf.py) — ACF/PACF on simulated data.

### Autoregression and GLSAR examples

- [`examples/time_series/autoregression/example_estimate_AR.py`](../../examples/time_series/autoregression/example_estimate_AR.py) — `estimate_ar` methods vs `AUTOREG` / `SARIMAX`.
- [`examples/time_series/autoregression/example_autoreg.py`](../../examples/time_series/autoregression/example_autoreg.py) — `autoreg` and `AUTOREG` with trends and seasonals.
- [`examples/time_series/autoregression/example_glsar.py`](../../examples/time_series/autoregression/example_glsar.py) — GLSAR vs OLS.

---

## Notes and caveats

- **Three AR-related tools serve different jobs:** `estimate_ar` — standalone AR(`p`) on one series; `AUTOREG` / `ARDL` — OLS regression with lags (and exog lags); `glsar` — linear `y ~ X` with AR errors on residuals.
- SARIMAX uses Brockwell–Davis-style ARMA/ARIMA state-space recursions and kanly’s own optimizer. The API is similar in spirit to [statsmodels `SARIMAX`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html), but option names and numerical details are not guaranteed to match.
- Intercept parameterizations differ between `estimate_ar` / `AUTOREG` (level + lags) and SARIMAX (constant + AR on innovations); compare AR coefficients and likelihoods, not raw intercepts, when cross-checking.
- Kalman and distribution helpers skip many domain checks for speed; invalid specs may yield non-finite likelihoods or optimizer restarts instead of early errors.
- For general constraints, global search, or multivariate models, prefer **statsmodels**, **pmdarima**, or other libraries; kanly targets univariate SARIMAX plus lightweight ACF/PACF and OLS-based dynamic regression inside the package ecosystem.

## Key Conceptual Difference: SARIMAX vs. ARX / ARDL
| Model | Underlying Statistical Model | Treatment of Exogenous Regressors | Initial Observations |
|---------|---------|---------|---------|
| **SARIMAX** | Typically written as `y_t = μ + x_t'β + u_t` where the disturbance follows an ARIMA process: `φ(L)(1-L)^d u_t = θ(L)ε_t` | Regressors enter as a contemporaneous linear mean effect. Their impact is estimated jointly with the ARIMA error process. The AR structure is on the residuals after removing the regression component. | Uses a state-space representation and Kalman filter. No observations are dropped. Exact likelihood is evaluated using the full sample (subject to diffuse or stationary initialization assumptions). |
| **ARX / AutoReg / CSS** | `y_t = μ + φ(L)y_t + x_t'β + ε_t` where `ε_t` is white noise. This is a dynamic regression with lagged `y` directly in the equation. | Regressors enter directly into the structural equation alongside lagged `y`. Estimation is typically OLS or conditional MLE. | Drops the first `p` observations (or conditions on them). Likelihood is conditional on the initial lag values. |
| **ARDL** | `y_t = μ + Σ(i=1..p) φ_i y_(t-i) + Σ(j=0..q) β_j x_(t-j) + ε_t` | Allows distributed lags of the regressors. Both current and lagged values of `x` can affect `y`. | Usually estimated conditionally via OLS. Drops enough observations to accommodate the largest lag among `y` and `x`. |
| **AR_MLE (exact)** | Same structural model as ARX: `y_t = μ + φ(L)y_t + x_t'β + ε_t`, with `ε_t ~ N(0,σ²)`. Parameters are estimated by maximizing the exact Gaussian likelihood. | Regressors enter directly into the structural equation and are estimated jointly with the AR parameters and innovation variance. | Uses the stationary distribution of the initial AR state and includes all observations in the likelihood. No observations are dropped. |
| **AR_MLE (conditional)** | Same structural model as ARX: `y_t = μ + φ(L)y_t + x_t'β + ε_t`, with `ε_t ~ N(0,σ²)`. Parameters are estimated by maximizing the conditional Gaussian likelihood. | Regressors enter directly into the structural equation and are estimated jointly with the AR parameters and innovation variance. | Conditions on the first `p` observations (equivalent to CSS). The likelihood is computed only from observations `p+1, ..., T`. |

| ARX / ARDL | SARIMAX |
|------------|----------|
| AR structure is on **y_t** itself. | AR structure is on the **error process** after accounting for regressors. |
| `y_t = μ + φ(L)y_t + x_t'β + ε_t` | `y_t = μ + x_t'β + u_t`, where `u_t` follows an ARIMA process. |
| Lagged `y` appears directly in the regression equation. | Lagged `y` enters only indirectly through the ARIMA error dynamics. |
| Usually estimated via OLS, CSS, or exact Gaussian MLE. | Usually estimated through a state-space representation and the Kalman filter. |
| Coefficients have the interpretation of a dynamic regression model. | Coefficients describe the conditional mean, while serial dependence is captured in the residual process. |


- **CSS (Conditional Sum of Squares)** and **conditional Gaussian MLE** use the same objective up to a constant scaling factor when the innovation variance is estimated.
- For a pure AR model with no exogenous regressors, **AR_MLE (exact)** and **SARIMAX(p,0,0)** represent essentially the same Gaussian process.
- The main practical difference is computational: AR_MLE evaluates the likelihood directly from the AR covariance structure, while SARIMAX evaluates the likelihood using a state-space model and Kalman filtering.
---

## References

### Primary text (notation and algorithms)

- Brockwell, P. J., & Davis, R. A. (2016). [*Introduction to Time Series and Forecasting*](https://doi.org/10.1007/978-3-319-29854-2) (3rd ed.). Springer Texts in Statistics. Springer. ISBN 978-3-319-29852-8.  
  State-space ARMA/ARIMA likelihoods, the Durbin–Levinson PACF algorithm (§2.5.1), and stationary-process background used throughout this package align with this edition.

### Software comparison

- [statsmodels `SARIMAX`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html) — broader `tsa` ecosystem; kanly’s SARIMAX API is similar in spirit but not numerically identical.
- [statsmodels `AutoReg`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.ar_model.AutoReg.html) / [`ARDL`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.ardl.ARDL.html) — kanly’s `AUTOREG` / `ARDL` are OLS wrappers with the same regression spirit.
- [statsmodels `GLSAR`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.GLSAR.html) — Cochrane–Orcutt only; kanly `glsar` defaults to Prais–Winsten (`full_information=True`).
- [statsmodels `acf` / `pacf`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html) — sample ACF/PACF references for `kanly.api.acf` and `pacf`.
