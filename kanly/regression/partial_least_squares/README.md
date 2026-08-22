# Partial least squares (`kanly.regression.partial_least_squares`)

**See also:** [kanly README](../../../README.md) · [regression overview](../README.md) · [formula](../../formula/README.md)

**Partial least squares** (PLS) finds latent components that maximise the covariance between predictors and response(s). It is useful when predictors are highly collinear or when dimensionality reduction before regression is desired. `PLS1` handles a single response; `PLS2` handles multiple responses simultaneously.

`pls1`, `PLS1`, and `PLS2` are exported from `kanly.api` and implemented in [`pls.py`](pls.py).

---

## What it does

- **PLS1** — single-response PLS via NIPALS; sparse-aware internal path when `X` is CSC.
- **PLS2** — multi-response PLS; latent components maximise covariance between `X` and all columns of `Y`.
- **Formula API** — `pls1('y ~ x1 + x2 + ...', df, l=...)` builds sparse designs via [`kanly.formula`](../../formula/README.md).

Model equations (PLS1):

```text
X = T * P.T + E_X
y = T * q + E_y
```

where `T` are scores (latent variables), `P` are X-loadings, `q` are y-loadings, and `E_X`, `E_y` are residuals.

---

## Entry points (`kanly.api`)

| Symbol | Role |
|--------|------|
| `pls1` | Formula interface → `PlsRegressionResults` |
| `PLS1` | Array interface: `y` (n,), `X` (n × k), `l` components |
| `PLS2` | Array interface: `Y` (n × m), `X` (n × k), `l` components |

---

## Formula API (`pls1`)

```python
from kanly.api import pls1

fit = pls1('y ~ x1 + x2 + x3', df, l=3)
print(fit.summary())
```

- Parses a Patsy-style formula with `SparseDataGetter` (same grammar as `lm` for the RHS).
- **No intercept in the design** — PLS1 fits its own centred intercept; if the formula does not end in ` -1`, centring is applied automatically.
- **Not supported:** IV (`|` in formula raises); **weights** (`$`) are not implemented yet.

---

## Array API (`PLS1`, `PLS2`)

```python
import numpy as np
from kanly.api import PLS1, PLS2

# Single response, three components
fit = PLS1(y, X, l=3, center=True,
           exog_names=[f'x{j}' for j in range(X.shape[1])])
print(fit.coef, fit.intercept)

# Multiple responses; returns a PlsRegressionResults fit object
fit = PLS2(Y, X, l=5, center=True, max_iter=100, tol=1e-6)
print(fit.coef, fit.rsquared)
Y_pred = fit.predict(X)

# X and Y may independently be dense or SciPy sparse matrices
from scipy.sparse import csr_matrix
fit_sparse = PLS2(csr_matrix(Y), csr_matrix(X), l=5)
Y_pred_sparse = fit_sparse.predict(csr_matrix(X))
```

### Parameters (common)

| Argument | Meaning |
|----------|---------|
| `l` | Number of latent components (1 ≤ `l` ≤ number of predictors) |
| `center` | Centre `X` and `y`/`Y` before fitting (default `True`); implicit intercept when centred |
| `compute_cov` | Approximate covariance for `PLS1` summaries (default `True`) |
| `specification_name`, `test_level` | Printed summary labels |

`PLS2` additionally accepts `max_iter`, `tol` for NIPALS convergence.

---

## Results (`PlsRegressionResults`)

Both `PLS1` and `PLS2` return `PlsRegressionResults`. Key fields include `coef`, `intercept`, `fittedvalues`, `resid`, `rsquared`, latent components (`T`, `P`, `Q`, `W`), named `params`, `predict()`, and `summary()`. PLS2 stores one statistics value per response and exposes its parameter matrix as `params_by_response`. Covariance is currently available only for PLS1 when `compute_cov=True`.

---

## Limitations

- No instrumental-variables formulas.
- No frequency weights in the formula path.
- See the runnable PLS1 and PLS2 scripts linked below for array, formula, dense, and sparse usage.

---

## Examples in this repo

- [`example_pls1.py`](../../../examples/regression/partial_least_squares/example_pls1.py) — single-response PLS with sparse array and formula APIs.
- [`example_pls2.py`](../../../examples/regression/partial_least_squares/example_pls2.py) — multi-response PLS with dense/CSR parity and held-out prediction.
- See also the [root user guide](../../../README.md#partial_least_squares) PLS subsection.
