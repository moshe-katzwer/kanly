# Penalized Linear Models

**See also:** [kanly README](../../../../README.md) · [regression overview](../../README.md) · [linear models](../README.md)

This package fits sparse linear models with elastic-net regularization.  The public API is available through `kanly.api.elastic_net` for formula-based models and `SparsePenalizedLinearModel.ELASTIC_NET` for array inputs.

## Objective

For response `y`, design matrix `X`, intercept `b0`, coefficients `b`, and optional weights, the solver minimizes a least-squares objective with elastic-net penalties:

```text
minimize over b0, b:

    SSR(b0, b) / (2 * n)
    + sum_j alpha_j * l1_ratio_j * |b_j - target_j|
    + sum_j 0.5 * alpha_j * (1 - l1_ratio_j) * (b_j - target_j)^2
```

`alpha` controls total penalty strength.  Larger values shrink coefficients more aggressively.  `l1_ratio` controls the penalty mix:

- `l1_ratio=1.0` gives LASSO and can set coefficients exactly to zero.
- `l1_ratio=0.0` gives ridge-style shrinkage.
- Values between `0` and `1` give elastic net, combining sparsity and ridge stabilization.

The intercept is handled separately and is not penalized.  When `regularize_to_values` is supplied, the penalty shrinks coefficients toward those target values instead of toward zero.

## Comparison: `elastic_net` vs sklearn `ElasticNet`

```python
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from kanly.api import elastic_net

# sklearn — dense design matrix; scale features explicitly (recommended workflow)
# X_scaled = StandardScaler().fit_transform(X)
# fit_sk = ElasticNet(alpha=1e-4, l1_ratio=1.0).fit(X_scaled, y)

# kanly — formula on DataFrame; sparse design; per-coefficient alpha dict
# normalize=True (default) scales alpha by column std dev — same scale-invariance
# goal as StandardScaler when fit_intercept=True
fit = elastic_net('y ~ x0 + x1 + x2 + ...', df, alpha=1e-4, l1_ratio=1.0)
fit = elastic_net('y ~ x + z $ w', df, alpha={'x': 3.0}, l1_ratio=1.0, refit=True)
```

## Formula API

Use `kanly.api.elastic_net` with a formula and a `pandas.DataFrame` or compatible data source.

Many predictors (from the [root user guide](../../../../README.md#linear_modelspenalized)):

```python
from kanly.api import elastic_net

fit = elastic_net(
    "y ~ " + " + ".join([f"x{i}" for i in range(k)]),
    df,
    alpha=0.0001,
    l1_ratio=1.0,
    selection="random",
    active_set=True,
)
print(fit.summary(show_only_non_zero=True, show_formula=False))
```

High-dimensional LASSO on 1,000 predictors ([`examples/regression/linear_models/penalized/example_elastic_net.py`](../../../../examples/regression/linear_models/penalized/example_elastic_net.py)):

```python
import numpy as np
import pandas as pd
from kanly.api import elastic_net

n = 10_000
k = 1000
np.random.seed(0)

df = pd.DataFrame(index=range(n))
X = np.hstack([np.random.randn(n, 1) for i in range(k)])
X = np.dot(X, np.random.randint(0, 5, (1000, 1000)))

coefs = np.zeros(k)
coefs[[2, 6, 700, 30]] = 1.5

y = -2.6 + X.dot(coefs) + .3 * np.random.randn(n)

df.loc[:, ['x%d' % i for i in range(k)]] = X
df['y'] = y

formula = 'y ~ ' + ' + '.join(['x%d' % i for i in range(k)])

fit = elastic_net(
    formula, df, alpha=.001, l1_ratio=1.0, debug=False,
    specification_name='example elastic net', selection='random',
    xtol=1e-6, active_set=True, max_iter=2000,
)

print(fit.summary(show_only_non_zero=True, show_formula=False))
```

**Output:**

```text
════════════════════════════════════════════════════════════════
Penalized Linear Model Results
example elastic net
════════════════════════════════════════════════════════════════

Dep. Variable: y

Date:             May 17, 2026    |dx|:                 1.64e-08
Time:                 07:00:04    |dF/F|                5.18e-14
Method:                  LASSO    max|subgrad|:         8.07e-05
Nobs:                    10000    alpha:                1.00e-03
Params:                   1001    l1_ratio:             1.00e+00
Score:                  1.0000    fit_intercept:            True
SSR:                1.0421e+03    normalize:                True
Penalty:            4.6862e+01    positive:                False
Objective:          4.6915e+01    scaled:                  False
Weights:                     -    relaxation:                   
Converged:                True    active_set:               True
Iters:                     139    selection:              random
Max Iter:                 2000    Tolerance:            1.00e-06
                                  Model Time:             34.01s
                                  Fit Time:                1.72s

═════════════════
             coef
─────────────────
Intercept  -2.602
x2            1.5
x6            1.5
x30           1.5
x700          1.5
═════════════════

996 parameter estimates suppressed in output that are zero.

Converged: x_error = 1.6e-08 < 1.0e-06 = xtol, f_error = 5.2e-14 < 1.0e-10 = ftol, 
           g_error = 8.1e-05 < 1.0e-04 = gtol.

                                             [kanly v=0.0.1020]
```

Formula parsing always removes the formula intercept from the design matrix and lets the solver fit an unpenalized intercept through `fit_intercept=True`.

## Array API

Use `SparsePenalizedLinearModel.ELASTIC_NET` when design matrices are already built.

```python
from kanly.regression.linear_models.penalized.model import SparsePenalizedLinearModel

fit = SparsePenalizedLinearModel.ELASTIC_NET(
    endog=y,
    exog=X,
    alpha=0.01,
    l1_ratio=0.5,
    fit_intercept=True,
    normalize=True,
    selection="cyclic",
)
```

The array API returns the same `SparsePenalizedLinearRegressionResults` object as the formula API.

## Penalty scaling (`normalize`)

`normalize=True` is the default on `elastic_net`, `SparsePenalizedLinearModel.fit`, and `ELASTIC_NET`.

When enabled (and `fit_intercept=True`), each coordinate's penalty strength is scaled by the **standard deviation** of the corresponding predictor column:

- L1 term: `alpha_j * l1_ratio_j * std(x_j)`
- L2 term: `0.5 * alpha_j * (1 - l1_ratio_j) * std(x_j)^2`

With observation weights, weighted population standard deviations are used (`ddof=0`). The design matrix is **not** centered or divided column-wise before fitting; the unpenalized intercept handles location, which matches the workflow sklearn recommends with [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) (scale features, fit with intercept).

Set `normalize=False` to apply raw `alpha` on the native scale of each column.

> **Note — change from older kanly / legacy sklearn:** Previous versions of kanly (and older sklearn `ElasticNet(normalize=True)`) scaled penalties by the **L2 norm of demeaned columns**, not the standard deviation. That legacy behavior is retained internally for testing via `OLD_SKLEARN_NORMALIZATION` in `sparse_elastic_net_internal.py` (default `False`). Public APIs use std-dev scaling.

## Common Settings

- `alpha`: scalar, iterable, or formula-name dict of penalty strengths.  Dict values are aligned to parsed coefficient names.
- `l1_ratio`: scalar, iterable, or dict of L1 shares.
- `positive`: boolean, iterable, or dict controlling non-negativity constraints.
- `weights`: optional observation weights.  Weighted models use weighted SSR and weighted R-squared.
- `normalize`: when True (default), scale each `alpha_j` by the standard deviation of column `j` so penalties are comparable across predictors on different scales. Ignored when `fit_intercept=False`. See [Penalty scaling](#penalty-scaling-normalize).
- `selection`: coordinate update order, one of `random`, `cyclic`, or `greedy`.
- `active_set`: after a full pass, update only coordinates that recently moved materially until the active set stabilizes.
- `xtol`, `ftol`, `gtol`: coefficient-change, objective-change, and subgradient convergence tolerances.
- `one_dim_search_cadence`: periodically tries a one-dimensional search along the full coordinate-descent update direction.
- `penalty_intensities`: optional sequence of penalty scales for warm-starting a path from stronger to weaker penalization.
- `relaxation_parameter`: runs a second pass with lower penalties on selected coefficients and very large penalties on unselected coefficients.

## Post-Selection Refit

Set `refit=True` to run an ordinary least-squares refit on the nonzero variables selected by the penalized fit:

```python
from kanly.api import elastic_net

fit_en, fit_ols = elastic_net(
    "y ~ poly(x,4) + np.log(1+w) + C(geo)",
    df,
    alpha=0.01,
    l1_ratio=1.0,
    refit=True,
)
```

The first result is the penalized fit; the second is a `SparseLinearModel` OLS result using the selected support.  Refit is only meaningful when `l1_ratio > 0`, because ridge does not perform variable selection.

## Results

The result object stores:

- `params`, `coef_`, and `intercept_`
- `fittedvalues`, `resid`, and `rsquared`
- convergence diagnostics: `converged`, `iters`, `x_error`, `f_error`, `g_error`, and `message`
- objective pieces: `ssr`, `penalty`, `objective`, and `objective_function`
- `predict(data=None, params=None)` for in-sample or new-data prediction

Classical standard errors, p-values, and t-statistics are not computed for penalized coefficients in this result object because the estimates are shrinkage-biased.  Use `refit=True` when post-selection OLS inference is desired.

## External References

- [Elastic net regularization](https://en.wikipedia.org/wiki/Elastic_net_regularization)
- [Regularization in mathematics and statistics](https://en.wikipedia.org/wiki/Regularization_(mathematics))
- [Lasso regression](https://en.wikipedia.org/wiki/Lasso_(statistics))
- [Ridge regression](https://en.wikipedia.org/wiki/Ridge_regression)
