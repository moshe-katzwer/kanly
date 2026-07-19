# kanly Robust Linear Regression

**See also:** [kanly README](../../../../README.md) · [regression overview](../../README.md) · [linear models](../README.md)

`kanly.regression.linear_models.robust` provides sparse-first
[robust linear regression](https://en.wikipedia.org/wiki/Robust_regression) via
**M-estimation** ([M-estimators](https://en.wikipedia.org/wiki/M-estimator))
solved with **IRLS** ([Iteratively Reweighted Least Squares](https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares)).

Key features:
- Six norm (influence) functions: **HuberT** (default), **LeastSquares**, **TukeyBiweight**, **TrimmedMean**, **AndrewWave**, **RamsayE**
- MAD-based scale estimation ([Median Absolute Deviation](https://en.wikipedia.org/wiki/Median_absolute_deviation))
- Four analytic covariance types: **H1** (default), **H2**, **H3**, **SANDWICH**
- **Bootstrap** covariance for robust inference
- Formula API (`rlm`) and matrix API (`RLM`)
- Handles millions of rows via sparse-CSC design matrices
- Drop-in comparison with OLS via `compare_results`

---

## Quick Start

```python
from kanly.api import rlm

fit = rlm('y ~ x', df)
print(fit)
```

---

## Basic Usage

```python
import numpy as np
import pandas as pd
from kanly.api import rlm

np.random.seed(0)
n = 50
df = pd.DataFrame({'x': np.random.rand(n), 'e': np.random.randn(n)})
# Inject 5 outliers
df.loc[np.random.choice(df.index, 5, replace=False), 'e'] += 15
df['y'] = 1.2 - 0.8 * df['x'] + df['e']

fit = rlm('y ~ x', df, debug=True)
print(fit)
```

`debug=True` prints IRLS diagnostics at each iteration:

```
========================================
Robust Regression (M-Estimation)
----------------------------------------
loss ("M"):     HuberT
nobs:           50
params:         2
max_iter:       50
x_tol:          1.0e-06
========================================

  iter          cost     % dCost        |dx|        scale     time
------------------------------------------------------------------
     0    1.5120e+02         nan         inf   2.2892e+00    0.00s
     1    4.4587e+01    -7.1e-01    5.05e-01   1.0057e+00    0.01s
...
```

---

## Reading Results

```python
print(fit.params)          # coefficient array
print(fit.bse)             # standard errors
print(fit.scale)           # MAD-based scale estimate σ̂
print(fit.pseudo_rsquared) # robust pseudo-R²
print(fit.iteration_info)  # convergence diagnostics dict

# Access individual coefficients by name (dict-style)
intercept = fit['Intercept']
slope     = fit['x']
```

---

## Norm (M) Functions

The `M=` parameter controls the [influence function](https://en.wikipedia.org/wiki/Robust_statistics#Influence_function_and_sensitivity_curve) used for downweighting outliers.

| `M=` string      | Description                             | Default `c` | Notes                                      |
|------------------|-----------------------------------------|-------------|--------------------------------------------|
| `'HuberT'`       | Quadratic inside, linear outside        | 1.345       | **Default.** Best all-around choice.       |
| `'LeastSquares'` | OLS (all weights = 1)                   | —           | No robustness; use as baseline.            |
| `'TukeyBiweight'`| Hard-redescending (zero beyond c)       | 4.685       | High [breakdown point](https://en.wikipedia.org/wiki/Breakdown_point); non-convex. |
| `'TrimmedMean'`  | Quadratic inside, zero outside          | 2.0         | Simpler variant of Tukey.                  |
| `'AndrewWave'`   | Sinusoidal, smooth redescending         | 1.339       | Smooth zero-influence beyond c·π.          |
| `'RamsayE'`      | Exponential soft-redescending           | 0.3         | Soft downweighting; never zero weight.     |

```python
# Use Tukey's bisquare norm
fit_tukey = rlm('y ~ x', df, M='TukeyBiweight')

# Pass a custom threshold c
from kanly.regression.linear_models.robust.robust_norm_functions import HuberT
fit_huber_tight = rlm('y ~ x', df, M=HuberT(c=1.0))
```

---

## Covariance Types

| `cov_type` | Description                              | When to use                          |
|------------|------------------------------------------|--------------------------------------|
| `'H1'`     | Huber (1981) variant 1 — uses `(XᵀX)⁻¹` | **Default.** Most common in practice.|
| `'H2'`     | Huber variant 2 — uses `(XᵀΨ′X)⁻¹`      | Slightly different small-sample adj. |
| `'H3'`     | Huber variant 3 — sandwich with H2      | Conservative; similar to sandwich.   |
| `'SANDWICH'`| Classic A⁻¹BA⁻¹ heteroscedastic SE    | Heteroscedasticity-robust SEs.       |
| `'BOOTSTRAP'`| Bootstrap covariance                  | Most reliable; slowest.              |

All analytic types follow Huber (1981) and apply a small-sample correction
factor `k` that vanishes as n → ∞.

```python
fit_h1       = rlm('y ~ x', df, cov_type='H1')
fit_sandwich = rlm('y ~ x', df, cov_type='SANDWICH')
```

---

## Bootstrap Standard Errors

Pass `cov_type='BOOTSTRAP'` to estimate covariance via bootstrap resampling.
Use `cov_kwds` to control the bootstrap:

```python
fit = rlm('y ~ x', df,
          cov_type='BOOTSTRAP',
          cov_kwds={
              'n_samples': 500,    # number of bootstrap draws (default 100)
              'method':   'bayesian',  # 'bayesian' (default) or 'block'
              'alpha':     0.05,   # significance level for CIs
              'seed':      42,     # random seed for reproducibility
          })
print(fit)
```

Output header shows `covariance type: BOOTSTRAP` and confidence intervals
are derived from the bootstrap distribution.

> **Note:** Bootstrap with an integer `index` is not supported.

---

## Comparing RLM with OLS

```python
from kanly.api import rlm, lm, compare_results

fit_ols = lm( 'y ~ x', df, cov_type='BOOTSTRAP')
fit_rlm = rlm('y ~ x', df, cov_type='BOOTSTRAP', debug=True)

print(compare_results(
    [fit_ols, fit_rlm],
    fit_titles=['OLS', 'RLM'],
    ref_param_values={'Intercept': 1.2, 'x': -0.8},
))
```

Example output:

```
============================================================
Regression Summary Table
============================================================
                           ols        rlm   |  Reference
------------------------------------------------------------
Intercept                 3.47       1.37   |        1.2
                        (1.59)    (0.442)   |

x                        -2.55      -1.29   |       -0.8
                        (2.64)    (0.647)   |
============================================================
```

The RLM estimate is substantially closer to the true parameter values because
the 5 injected outliers inflate the OLS coefficients.

---

## IRLS Convergence Controls

```python
fit = rlm('y ~ x', df,
          x_tol=1e-8,    # tighter tolerance (default 1e-6)
          max_iter=100,  # more iterations (default 50)
)
print(fit.iteration_info)
# {'num_iters': 9, 'max_iter': 100, 'tol': 1e-8,
#  'error': 1.46e-10, 'converged': True, 'force_scale': None}
```

Fix the scale estimate (bypass MAD) with `force_scale`:

```python
fit = rlm('y ~ x', df, force_scale=1.0)
```

---

## WLS Robust Regression

Pass weighted observations using the `$` weights syntax in the formula:

```python
df['w'] = np.random.rand(n) + 0.5  # observation weights

fit_wls_rlm = rlm('y ~ x $ w', df)
```

Or use the matrix API with an explicit `weights` array:

```python
from kanly.regression.linear_models.robust.model import SparseRobustLinearModel
import numpy as np

fit = SparseRobustLinearModel.RLM(endog=df['y'].values,
                                   exog=np.c_[np.ones(n), df['x'].values],
                                   has_constant=True,
                                   weights=df['w'].values)
```

---

## Matrix API

Use `RLM()` when you already have NumPy arrays or sparse matrices:

```python
from kanly.regression.linear_models.robust.model import SparseRobustLinearModel
import numpy as np

X = np.c_[np.ones(n), df['x'].values]   # shape (n, 2)
y = df['y'].values

fit = SparseRobustLinearModel.RLM(
    endog=y,
    exog=X,
    has_constant=True,
    cov_type='H1',
    endog_name='y',
    exog_names=['Intercept', 'x'],
)
print(fit)
```

---

## Examples in this repo

- [`examples/regression/linear_models/robust/example_robust_regression.py`](../../../../examples/regression/linear_models/robust/example_robust_regression.py)
- [`examples/regression/linear_models/robust/example_robust_regression_instrumental_variables.py`](../../../../examples/regression/linear_models/robust/example_robust_regression_instrumental_variables.py) — illustrates limitation / error path for IV with RLM (read script header).

---

## Limitations

- **Single outcome only** — `accepts_multi_outcome()` returns `False`.
- **IV not supported** — instrumental variables (`~` with `|`) raise an error.
- **Fixed-effects absorption not supported** — use explicit dummy variables or
  the `lm` absorb syntax instead, then pass residuals to `rlm`.
- **Bootstrap + integer index** — not supported; raises `Exception`.

---

## References

- [Robust regression (Wikipedia)](https://en.wikipedia.org/wiki/Robust_regression)
- [M-estimator (Wikipedia)](https://en.wikipedia.org/wiki/M-estimator)
- [Huber loss (Wikipedia)](https://en.wikipedia.org/wiki/Huber_loss)
- [Iteratively reweighted least squares (Wikipedia)](https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares)
- [Breakdown point (Wikipedia)](https://en.wikipedia.org/wiki/Breakdown_point)
- [Median absolute deviation (Wikipedia)](https://en.wikipedia.org/wiki/Median_absolute_deviation)
- [Influence function (Wikipedia)](https://en.wikipedia.org/wiki/Robust_statistics#Influence_function_and_sensitivity_curve)
- Huber, P. J. (1981). *Robust Statistics*. Wiley.
