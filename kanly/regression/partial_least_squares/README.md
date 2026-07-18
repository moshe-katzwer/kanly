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

# Multiple responses
out = PLS2(Y, X, l=5, center=True, max_iter=100, tol=1e-6)
# out contains T, P, Q, W, coef, intercept, etc.
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

Key fields include `coef`, `intercept`, `fittedvalues`, `resid`, `rsquared`, `params`-style summaries via `summary()`, and optional `cov_params` when `compute_cov=True`.

---

## Limitations

- No instrumental-variables formulas.
- No frequency weights in the formula path.
- No dedicated scripts under `examples/` at present (use docstring examples in [`pls.py`](pls.py) or fit from notebooks).

---

## Examples in this repo

- None under `examples/regression/partial_least_squares/` yet.
- See also the [root user guide](../../../README.md#partial_least_squares) PLS subsection.
