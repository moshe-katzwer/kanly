# kanly Quantile Regression

**See also:** [kanly README](../../../../README.md) · [regression overview](../../README.md) · [linear models](../README.md)

`kanly.regression.linear_models.quantile_regression` provides sparse-first
[quantile regression](https://en.wikipedia.org/wiki/Quantile_regression) via
**IRLS** ([Iteratively Reweighted Least Squares](https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares)).

Key features:
- Three smooth surrogate loss functions: **Huber** (default), **SoftL1**, **SmoothCup**
- **Grid line search** after each IRLS step (default `line_search=True`) — see below
- Analytical covariance estimators: **IID** and heteroscedastic-**ROBUST** (both KDE-based)
- **Bootstrap** covariance (Bayesian / block) for reliable inference
- **IV** support via the [control-function approach](https://en.wikipedia.org/wiki/Control_function_(econometrics)) (experimental)
- Multi-quantile fitting with automatic **warm-starting**
- Handles millions of rows via sparse-CSC design matrices

### IRLS and line search

Each iteration solves one **weighted least-squares** subproblem from the smooth
surrogate loss — that WLS step is the expensive part.  With `line_search=True`
(the default), the solver then searches along the segment from the **previous**
coefficients to the IRLS proposal: it evaluates only the **surrogate quantile
objective** on interpolated residuals (cheap for a linear model) over a small
grid of step sizes and keeps the best point.  It stops when no grid point beats
the current iterate.

Those line-search evaluations are much cheaper than another full IRLS solve and
often cut total iterations substantially compared with taking every WLS update
at face value (e.g. **statsmodels** quantile regression when line search is off
or limited).

---

## Quick Start

```python
from kanly.api import qr

fit = qr('y ~ x + z', df, tau=0.9)
print(fit.summary())
```

---

## Single Quantile

```python
import numpy as np
import pandas as pd
from kanly.api import qr

n = 150_000
np.random.seed(0)
df = pd.DataFrame({
    'x': np.abs(np.random.randn(n)),
    'z': np.abs(np.random.randn(n)),
})
df['y'] = 2.0 + 0.5 * df.x - 0.2 * df.z + np.random.randn(n) * np.exp(df.x)

fit = qr('y ~ x + z', df, tau=0.9, debug=True)
print(fit.summary())
```

Key result attributes:

```python
fit.params           # pd.Series of coefficients
fit.bse              # standard errors
fit.pvalues          # p-values
fit.conf_int()       # confidence interval DataFrame
fit.pseudo_rsquared  # Koenker-Machado pseudo-R²
fit.tau              # 0.9
fit.fittedvalues     # Xβ̂
fit.resid            # y − Xβ̂
fit.converged        # True / False
fit.iterations       # number of IRLS steps taken
fit.message          # convergence message string
```

---

## Multiple Quantiles

Pass ``tau`` as any iterable.  Results are returned as a `dict` mapping
each quantile level to its ``SparseQuantileRegressionResults``.  Internally,
each quantile is warm-started from the previous solution (sorted ascending).

```python
taus = (0.1, 0.5, 0.9)
fits = qr('y ~ poly(x, 3)', df, taus,
          cov_type='bootstrap',
          cov_kwds={'seed': 1, 'n_samples': 500, 'max_processes': 6},
          line_search=True)

for tau, fit in fits.items():
    print(f'tau={tau}:  Intercept={fit.params["Intercept"]:.4f}')
```

---

## Formula Syntax

| Syntax | Effect |
|---|---|
| `y ~ x + z` | Standard quantile regression |
| `y ~ poly(x, 3)` | Degree-3 polynomial basis |
| `y ~ x + C(grp)` | Categorical dummies |
| `y ~ x + I(x**2)` | Inline transformation |
| `y ~ x + z \| z1 + z2` | IV — `z1`, `z2` are excluded instruments (experimental) |

---

## Loss Functions

These are smooth approximations of the asymmetric absolute ([check](https://en.wikipedia.org/wiki/Quantile_regression#Quantile_regression_in_practice)) loss
ρ_τ(z) = z·(τ − 𝟙{z<0}) used to keep [IRLS](https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares) weights finite.

| Name | ``loss=`` | Shape | Notes |
|---|---|---|---|
| [Huber](https://en.wikipedia.org/wiki/Huber_loss) (default) | `'huber'` | Quadratic in (−k, k), linear outside | Best all-round choice |
| [Soft-L1](https://en.wikipedia.org/wiki/Huber_loss#Pseudo-Huber_loss_function) | `'softl1'` | Smooth everywhere (Charbonnier / pseudo-Huber) | Softer weight transitions |
| Smooth Cup | `'smoothcup'` | Shifted quadratic, centred at tau-quantile | Better for extreme quantiles |

```python
fit_huber    = qr('y ~ x', df, tau=0.9, loss='huber')
fit_softl1   = qr('y ~ x', df, tau=0.9, loss='softl1')
fit_smoothcup = qr('y ~ x', df, tau=0.9, loss='smoothcup')
```

The smoothing bandwidth `k` (``smoothing_k``) controls how closely each
loss approximates the true check function.  Smaller values are more
accurate but can slow convergence; the default `1e-5` works well in
practice.

---

## Convergence Controls

```python
fit = qr('y ~ x + z', df, tau=0.9,
         xtol=1e-6,      # stop when max |Δβ| < xtol
         ftol=1e-8,      # stop when |ΔCost|/Cost < ftol
         max_iter=200,   # hard iteration cap
         line_search=True,   # default; see "IRLS and line search" above
         smoothing_k=1e-5,   # loss smoothing bandwidth
         debug=True)         # print iteration table
```

Set `line_search=False` only if you need to match a solver that skips this step;
convergence is usually slower.  When ``debug=True`` the solver prints an iteration table showing cost,
relative cost change, β-error, and the fraction of negative residuals
(which should converge to `1 − tau`).

---

## Covariance Types

The IID and ROBUST estimators are based on a [kernel density estimate](https://en.wikipedia.org/wiki/Kernel_density_estimation)
of the residual sparsity at zero (f̂₀), following
[Powell (1991)](https://en.wikipedia.org/wiki/Quantile_regression#Inference).

| ``cov_type`` | Description | Speed | When to use |
|---|---|---|---|
| `'IID'` (default) | KDE-based: τ(1−τ)/f̂₀² · (XᵀX)⁻¹ | Fast | Homoscedastic errors |
| `'ROBUST'` | KDE-based [heteroscedastic sandwich](https://en.wikipedia.org/wiki/Heteroscedasticity-consistent_standard_errors) | Fast | Heteroscedastic errors |
| `'BOOTSTRAP'` | [Bayesian / block bootstrap](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)) | Slow | Most reliable; use for inference |

```python
fit_iid     = qr('y ~ x', df, tau=0.9, cov_type='iid')
fit_robust  = qr('y ~ x', df, tau=0.9, cov_type='robust')
fit_boot    = qr('y ~ x', df, tau=0.9, cov_type='bootstrap')
```

---

## Bootstrap Standard Errors

```python
fit = qr('y ~ poly(x, 3)', df, tau=0.9,
         cov_type='bootstrap',
         cov_kwds={
             'n_samples': 500,      # number of bootstrap draws
             'method': 'bayesian',  # 'bayesian' (Poisson weights) or 'block'
             'seed': 42,
             'max_processes': 4,    # parallel workers
             'alpha': 1.0,          # Dirichlet concentration (bayesian only)
         })
print(fit.summary())
```

For block / cluster bootstrap, pass a group array as `'groups'`:

```python
fit = qr('y ~ x', df, tau=0.9,
         cov_type='bootstrap',
         cov_kwds={'groups': df['firm_id'].values, 'n_samples': 500})
```

---

## Instrumental Variables (Experimental)

Use `|` to separate excluded instruments in the formula.  IV for quantile
regression uses a [**control-function**](https://en.wikipedia.org/wiki/Control_function_(econometrics)) approach: endogenous regressors are
projected onto [instruments](https://en.wikipedia.org/wiki/Instrumental_variables_estimation) in a first stage, and the first-stage residuals
(optionally raised to polynomial powers) are appended to the design matrix.

```python
fit_iv = qr('y ~ x + C(grp) | z + C(grp)', df, tau=0.9,
            residual_inclusion=True, residual_inclusion_order=1)
```

> **Warning**: IV quantile regression is experimental.  For reliable
> inference, use bootstrap covariance (`cov_type='bootstrap'`).

---

## Matrix API

For pre-built arrays, use `SparseQuantileRegressionModel.QR` directly:

```python
from kanly.regression.linear_models.quantile_regression.model import (
    SparseQuantileRegressionModel
)

fit = SparseQuantileRegressionModel.QR(
    endog=y,         # ndarray or sparse, shape (n,)
    exog=X,          # ndarray or sparse, shape (n, p)
    tau=0.9,
    has_constant=True,
    endog_name='y',
    exog_names=['Intercept', 'x1', 'x2'],
    cov_type='iid',
)
print(fit.summary())
```

---

## Reading Results

```python
print(fit.summary())          # formatted text summary
print(fit.summary_df())       # pandas DataFrame: coef / se / t / p / CI

# Coefficient attributes
fit.params                    # pd.Series of estimated coefficients
fit.bse                       # standard errors
fit.tvalues                   # t-statistics
fit.pvalues                   # p-values
fit.conf_int()                # confidence interval DataFrame

# Goodness-of-fit  (see https://en.wikipedia.org/wiki/Quantile_regression#Goodness_of_fit)
fit.pseudo_rsquared           # Koenker-Machado pseudo-R²
fit.tau                       # quantile level
fit.cost                      # smoothed objective_function at convergence
fit.true_cost                 # exact check-function cost at convergence

# Convergence info
fit.converged                 # bool
fit.iterations                # number of IRLS steps
fit.error                     # final convergence error
fit.message                   # human-readable convergence string

# Timing
fit.fit_elapsed               # seconds for IRLS fit
fit.cov_elapsed               # seconds for covariance estimation

# Prediction
fit.predict()                 # in-sample fitted values (copy)
fit.predict(data=new_df)      # out-of-sample predictions
```

---

## Comparison with statsmodels

```python
# statsmodels
from statsmodels.formula.api import quantreg
fit_sm = quantreg('y ~ x + z', df).fit(0.9)

# kanly (equivalent, but faster for large n)
from kanly.api import qr
fit_kn = qr('y ~ x + z', df, 0.9)

# Both produce standard summary tables
print(fit_sm.summary())
print(fit_kn.summary())
```

Differences:
- kanly uses a **sparse-CSC** design matrix and IRLS with a smooth surrogate loss; statsmodels uses an interior-point LP solver.
- kanly supports **multiple quantiles** in one call with warm-starting.
- kanly's `BOOTSTRAP` covariance is generally more reliable than the analytical estimators for heteroscedastic data.
- statsmodels' `quantreg` is the reference for correctness validation (see `__var_covar.py`).

---

## Examples in this repo

- [`examples/regression/linear_models/quantile_regression/example_quantile_regression.py`](../../../../examples/regression/linear_models/quantile_regression/example_quantile_regression.py) — large-scale `qr` with bootstrap SEs; printed summary is stored in the file’s trailing docstring.
- Sketch / WIP: [`examples/work_in_progress/example_quantile_regression_instrumental_variables.py`](../../../../examples/work_in_progress/example_quantile_regression_instrumental_variables.py) — IV quantile workflow.

---

## References

- [Quantile regression — Wikipedia](https://en.wikipedia.org/wiki/Quantile_regression)
- [Iteratively reweighted least squares — Wikipedia](https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares)
- [Huber loss — Wikipedia](https://en.wikipedia.org/wiki/Huber_loss)
- [Pseudo-Huber loss — Wikipedia](https://en.wikipedia.org/wiki/Huber_loss#Pseudo-Huber_loss_function)
- [Kernel density estimation — Wikipedia](https://en.wikipedia.org/wiki/Kernel_density_estimation)
- [Heteroscedasticity-consistent standard errors — Wikipedia](https://en.wikipedia.org/wiki/Heteroscedasticity-consistent_standard_errors)
- [Bootstrapping (statistics) — Wikipedia](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))
- [Instrumental variables estimation — Wikipedia](https://en.wikipedia.org/wiki/Instrumental_variables_estimation)
- [Control function (econometrics) — Wikipedia](https://en.wikipedia.org/wiki/Control_function_(econometrics))
