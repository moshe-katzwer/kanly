# Nonlinear Least Squares User Guide

**See also:** [kanly README](../../../README.md) · [regression overview](../README.md)

This package fits nonlinear least-squares models with formula-generated
prediction functions, optional weights, bounds, robust losses, bootstrap or
cluster covariance estimates, and elastic-net regularisation.

## How fitting works

[Non-linear least squares](https://en.wikipedia.org/wiki/Non-linear_least_squares)
(NLLS) fits a model that is nonlinear in its parameters. With observations
`(x_i, y_i)`, a prediction function `y_hat_i = f(x_i, beta)`, and residuals
`r_i = y_i - f(x_i, beta)`, the usual goal is to minimize the (possibly weighted)
sum of squares `S = sum_i w_ii * r_i^2`. Unlike linear least squares, setting the
gradient of `S` to zero generally has no closed-form solution, so software refines
`beta` by successive updates `beta^{k+1} = beta^k + delta_beta`.

At each iterate, the standard approach—described in the Wikipedia article—is to
**linearize** the model around the current parameters: approximate `f` by a
first-order Taylor expansion using the **Jacobian** `J` of partial derivatives.
That yields a weighted linear least-squares subproblem for the step
`delta_beta`, often written `(J' W J) delta_beta = J' W delta_y`, which
underlies the [Gauss–Newton method](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm).

Near the minimum, `S` is approximately quadratic in `beta`; far away it is not,
so **good starting values** (`start_params`) matter and multiple local minima are
possible (see the Wikipedia sections on initial estimates and multiple minima).

**What `kanly` does.** A formula such as `"[y] ~ {a} + {b} * np.exp({c} * [x])"`
builds a `PredictionFunction` and residual vector. The default entry point
`nlls` / `NLLS` calls a **trust-region reflective** solver
(`nlls_minimize_internal`) rather than a bare Gauss–Newton loop:

1. Evaluate residuals and a Jacobian (`jac_method`: analytic from the formula when
   possible, otherwise finite differences).
2. Form a local quadratic model of the weighted sum of squares (Gauss–Newton
   structure) and compute a proposed step by solving a **trust-region**
   subproblem (Steihaug / Cauchy / optional Newton variants), with radius
   `Delta` expanded or shrunk according to how well the quadratic model predicts
   the actual decrease in `S`.
3. **Reflect** proposed steps into box bounds when `bounds` are supplied.
4. Stop when parameter, objective, and gradient tolerances are met (`xtol`,
   `ftol`, `gtol`).

Optional formula weights (`$ [w]`; see [Formula syntax](#formula-syntax)) enter the
weighted normal equations in the same spirit as on Wikipedia. Robust fits apply a **root loss** to
residuals before squaring; `nlls_en` is a separate **elastic-net coordinate
descent** path for penalized models, not the trust-region Gauss–Newton loop.

After optimization, standard errors and tests use the Jacobian at the fitted
`beta_hat` (asymptotic linearization), with optional heteroskedasticity-robust,
cluster, or bootstrap covariance.

**Relation to SciPy.** The solver design was heavily informed by Nikolay
Mayorov’s blog series on NLLS algorithms, especially
[Basic Algorithms for Nonlinear Least Squares](https://nmayorov.wordpress.com/2015/06/18/basic-algorithms-for-nonlinear-least-squares/)
(Gauss–Newton, Levenberg–Marquardt as trust-region methods, and the move toward
trust-region reflective steps for bounds). Mayorov implemented SciPy’s nonlinear
least squares (`scipy.optimize.leastsq` and related MINPACK/TRF machinery), so
the overall approach here is similar: Jacobian-based quadratic models, trust-region
step control, and reflective handling of box constraints. **Unlike SciPy**, `kanly`
adds a **patsy-like** formula language (`[data]`, `{parameters}`, `C(group)`, weights
in the formula) for building prediction functions, plus regression-style **inference**
after the fit (`summary()`, `cov_type`, cluster/bootstrap covariance, and related
helpers)—not just a bare residual callable and parameter vector.

## Formula syntax

NLLS formulas are **not** the same as linear-model syntax (`y ~ x1 + x2 + C(g)`).
You specify a **nonlinear mean function**: data appear in **`[...]`**, parameters
to estimate in **`{...}`**, and the expression on the right of `~` may be nonlinear
in those parameters.

### Template

```text
[y] ~ <prediction>  $  [weights]
```

| Part | Syntax | Role |
|------|--------|------|
| Response | `[y]` on the left of `~` | Outcome column or expression |
| Prediction | right of `~` | `y_hat_i = f(data_i, beta)` |
| Weights | optional `$ [w]` suffix | Non-negative observation weights |

Example:

```python
"[y] ~ {Intercept} + {beta} * np.exp({gamma} * [x]) $ [w]"
```

### `[...]` — data from the frame

- **Column:** `[x]`, `[y]`
- **Expression:** patsy-style terms the design machinery can evaluate, e.g.
  `I(x**2)`, splines `bs(...)`, `cc(...)`, `cr(...)`
- Data tokens are **operands inside** the prediction formula, not a separate
  “add this regressor” list like in `lm`.

**Inside vs outside the brackets.** Transformations can sit inside or outside
`[...]`; the fitted model is the same, but **when** work happens differs:

| Form | When the transform runs |
|------|-------------------------|
| `np.exp([x])` | `[x]` is loaded once into a cached array; `np.exp` runs on **every** prediction / residual / Jacobian evaluation. |
| `[np.exp(x)]` | `np.exp(x)` is evaluated **once** when the model is built and stored; later calls only read the cached vector. |

Use `[...]` around expensive, parameter-free transforms you do not want to repeat
each optimizer iteration (logs, splines, patsy expressions). Keep transforms
outside when they must be recomputed with parameters (e.g. `{beta} * np.exp([x])`).

**Special bracket forms** (the full token inside `[...]` must match):

| Token | Meaning |
|-------|---------|
| `[poly(x, 3)]` | Univariate powers of `x` through degree 3 |
| `[poly(x, 3, -1)]` | Same, but omit the constant (intercept) power (drops one dummy.) |
| `[poly(x, [1, 2, 5])]` | Explicit exponent list |
| `[polym(x, z, 2)]` | Multivariate monomials in `x`, `z` up to total degree 2 |
| `[cheb(x, 4)]` | Chebyshev basis on `x` through degree 4 |
| `[C(group)]` | Categorical fixed effect (one parameter per level) |
| `[C(group, -1)]` | Drop the first category |
| `[C(g1, g2, -1)]` | Several categorical columns, combined |
| `[C(g; prefix)]` | `; prefix` overrides auto-generated coefficient names |

### `{...}` — parameters to estimate

Each `{name}` becomes one coefficient in the optimization (or many, after `poly` /
`C` expansion). Supply `start_params` and `bounds` as a dict keyed by those names.
After expansion, internal names may look like `_A`, `C(g)[2]` — inspect
`fit.param_names` or build the model with `debug=True` when unsure.

### NumPy and custom callables

Use ordinary arithmetic and NumPy functions (`np.exp`, `exp`, etc.). Pass extra
symbols via `custom_functions={"my_fn": callable}` on `nlls` / `nlls_en`.

### Prediction-only formulas

`build_prediction_function_from_formula` (also on `kanly.api`) parses the
**right-hand side only** — no `[y] ~`:

```python
from kanly.api import build_prediction_function_from_formula

pred_fn, info, valid = build_prediction_function_from_formula(
    "{Intercept} + {beta} * exp({gamma} * [x])", df)
yhat = pred_fn(np.array([1.0, 3.0, -0.5]))  # vector in sorted {name} order
```

### Performance and memory

Unlike `lm` / `glm`, this package is **not sparse-first**: prediction code works
on dense `float64` arrays built from your data frame, not on a sparse design
matrix accumulated term by term.

Two layout choices still matter at scale:

- **Categoricals / fixed effects** — `[C(g)]` does **not** materialize a wide
  matrix of ones and zeros. Each row gets an integer level index; the generated
  code indexes `params[start:end]` with that index (see `int_dict` in the
  parser). Memory is `O(n)` for the index vector plus `O(n_levels)` for
  coefficients, not `O(n * n_levels)` for a full dummy matrix.
- **Jacobians** — residual Jacobians with respect to parameters are often
  **sparse** (especially with categoricals and local basis terms). The trust-region
  solver can keep them sparse when they exceed `dense_threshold_mb`; use
  `jac_method="analytic"` when the formula supports it to avoid repeated
  finite-difference probes.

For very large row counts, prefer caching heavy transforms inside `[...]`, use
`subsample` for starting values, and set `dense_threshold_mb` when Jacobians or
covariance helpers would otherwise densify.

### Compared with `lm` / `glm`

| Linear / GLM | NLLS (`nlls`, `nlls_en`) |
|--------------|---------------------------|
| `y ~ x1 + C(g)` | `[y] ~ {a} + {b} * [x1] + ...` |
| Coefficients implicit in terms | Parameters explicit in `{...}` |
| Linear in betas | Nonlinear in `{...}` allowed |
| Sparse design matrix by default | Dense cached data arrays; sparse Jacobians |
| Weights often via `weights=` kwarg | Often `$ [w]` in the formula string |

**Notes:** rows with missing/invalid `y`, predictors, or weights are dropped when
the model is built. Lag terms (`L(...)`) are **not** supported in NLLS formulas.

## Basic Usage

Use `kanly.api.nlls` for the standard trust-region solver (see [Formula syntax](#formula-syntax)):

```python
from kanly.api import nlls

fit = nlls(
    "[y] ~ {alpha} + {beta} * np.exp({gamma} * [x])",
    data=df,
    start_params={"alpha": 0.0, "beta": 1.0, "gamma": -0.1},
)

print(fit.summary())
pred = fit.model.predict(fit.params)
```

## Starting Values and Bounds

Starting values can be a vector or a dict keyed by parameter name:

```python
fit = nlls(
    "[y] ~ {a} + {b} * [x]",
    df,
    start_params={"a": 0.0, "b": 1.0},
    bounds={"b": (0.0, np.inf)},
)
```

Bounds can be supplied as a `(num_params, 2)` array/list or as a dict.  Dict
bounds are aligned to generated parameter names, which is often clearer for
models with categorical or polynomial expansions.

For large models, use `subsample=<n>` to estimate a starting point on a random
subset before fitting the full data set.

## Jacobians

The trust-region solver supports three Jacobian modes:

- `jac_method="analytic"`: build analytic derivatives from the generated formula.
- `jac_method="mid"`: central finite differences.
- `jac_method="fwd"`: forward finite differences.

Analytic Jacobians are fastest when the formula can be differentiated by the
automatic-differentiation utilities.  Use `do_analytic_jac_jit=True` to JIT
compile generated analytic Jacobian code.

## Weights and Covariance

Observation weights use the `$ [w]` suffix (see [Formula syntax](#formula-syntax)):

```python
fit = nlls("[y] ~ {a} + {b} * np.exp([x]) $ [w]", df)
```

Covariance options (independent of formula syntax):

```python
fit_hc1 = nlls(formula, df, cov_type="hc1")

fit_cluster = nlls(
    formula,
    df,
    cov_type="cluster",
    cov_kwds={"groups": "cluster_id"},
)

fit_boot = nlls(
    formula,
    df,
    cov_type="bootstrap",
    cov_kwds={"n_samples": 250, "seed": 123},
)
```

Block bootstrap is enabled by providing `groups` in `cov_kwds`.

## Robust and Quantile-Style Losses

The trust-region solver can transform residuals through a root-loss function:

```python
from kanly.regression.nonlinear_least_squares.function_callables.loss_functions import QuantileHuberLoss

fit = nlls(
    "[y] ~ {a} + {b} * np.exp({c} * [x])",
    df,
    root_loss_function=QuantileHuberLoss(tau=0.75),
)
```

Built-in root losses include `HuberLoss`, `LeastSquares`, `QuantileHuberLoss`,
`QuantileSmooth`, and `QuantilePseudoHuberLoss`.

## Elastic-Net NLLS

Use `kanly.api.nlls_en` for elastic-net regularised coordinate descent:

```python
from kanly.api import nlls_en

fit = nlls_en(
    "[y] ~ {a} + {b} * [x] + [C(group, -1)]",
    df,
    alpha={"b": 0.1},
    l1_ratio=0.5,
    start_params={"a": 0.0, "b": 1.0},
    selection="cyclic",
)
```

Key options:

- `alpha`: scalar, vector, or dict of total regularisation strength.
- `l1_ratio`: fraction of `alpha` used for L1 shrinkage.
- `regularize_to_values`: shrink toward values other than zero.
- `positive`: enforce non-negative actual parameters.
- `bounds`: enforce box constraints.
- `active_set`: cycle over recently improving parameters.
- `selection`: `cyclic`, `greedy`, or `random`.

Bootstrap covariance for `nlls_en` is supported with `cov_type="bootstrap"`.

## Large-Scale Patterns

For many observations or high-cardinality categorical terms:

- Use `[C(geo, -1)]` and `[C(day, -1)]` in the prediction expression.
- Use `subsample` to get a good starting point cheaply.
- Prefer `jac_method="analytic"` when the formula supports it.
- Consider `dense_threshold_mb` to control when Jacobians/covariance helpers use
  sparse representations.

## Examples

Full scripts are in `examples/regression/nonlinear_least_squares/` (each file also
stores captured `print(fit)` output in a trailing string). Below, **code** matches
those examples; **output** excerpts are representative (timings vary by machine).

### `example_nonlinear_least_squares_exponential.py`

Weighted exponential decay; bootstrap covariance.

```python
import numpy as np
import pandas as pd
from kanly.api import nlls

n = 250
np.random.seed(0)
df = pd.DataFrame({'x': np.random.randn(n), 'w': np.exp(np.random.randn(n))})
df['y'] = 1 + 3 * np.exp(-.5 * df.x) + .4 * np.random.randn(n)
df.loc[3, 'y'] = np.nan

fit = nlls(
    '[y] ~ {Intercept} + {beta} * exp({gamma} * [x]) $ [w]',
    df,
    max_iter=100,
    cov_type='bootstrap',
    cov_kwds={'n_samples': 5_000, 'max_processes': 5},
    specification_name='Example Exponential',
)
print(fit)
```

**Output:**

```text
══════════════════════════════════════════════════════════════════════════
Nonlinear Least Squares Results
Example Exponential
══════════════════════════════════════════════════════════════════════════
...
Nobs:                           249    Fit Time:                     0.17s
Df Model:                         3    Iterations:                       7
Cost:                    3.2170e+01    Converged:                     True
...
Covariance Type:          BOOTSTRAP
...
              coef        std err      t   p>|t| [0.025,    0.975]
Intercept   0.7854  ****   0.1923   4.08  <0.001   0.4066    1.164
beta         3.131  ****   0.2004  15.63  <0.001    2.737    3.526
gamma      -0.4918  ****  0.02419 -20.33  <0.001  -0.5394  -0.4441

Did 5000 Bayesian bootstrap repetitions, alpha=1.000.
message: Converged: |dF| < ftol * max(1, |F|)
```

### `example_nonlinear_least_squares_logistic.py`

Logistic mean with analytic Jacobian.

```python
import numpy as np
import pandas as pd
from kanly.api import nlls

n = 10_000
np.random.seed(0)
df = pd.DataFrame({'x': np.random.randn(n)})
df['p'] = 1 / (1 + np.exp(-(.4 + .9 * df.x)))
df['y'] = (np.random.rand(n) < df['p']).astype(float)

fit = nlls(
    '[y] ~ 1.0 / (1.0 + exp({alpha} + {beta} * [x]))',
    df,
    jac_method='analytic',
    do_analytic_jac_jit=True,
    x_scale='jac',
    specification_name='logistic regression',
)
print(fit)
```

**Output:**

```text
Nonlinear Least Squares Results
logistic regression
...
Nobs:                         10000    Fit Time:                     0.21s
Df Model:                         2    Iterations:                       4
Covariance Type:                HC1
...
alpha  -0.3955  ****  0.02238 -17.67  <0.001  -0.4394  -0.3516
beta   -0.8832  ****   0.0264 -33.45  <0.001  -0.9349  -0.8314

formula:  [y] ~ 1.0 / (1.0 + exp({alpha} + {beta} * [x]))
```

### `example_nonlinear_least_squares_exponential_quantile_regression.py`

Robust / quantile-style fit via a custom root loss.

```python
import numpy as np
import pandas as pd
from kanly.api import nlls
from kanly.regression.nonlinear_least_squares.function_callables.loss_functions import (
    QuantileHuberLoss,
)

n = 500
np.random.seed(0)
df = pd.DataFrame({'x': np.random.randn(n), 'w': np.exp(np.random.randn(n))})
df['y'] = 1 + 3 * np.exp(-.5 * df.x) + .4 * np.random.randn(n) * (1 + np.abs(df.x) / 3)

fit = nlls(
    '[y] ~ {Intercept} + {beta} * np.exp({gamma} * [x])',
    df,
    root_loss_function=QuantileHuberLoss(tau=.15, k=.001),
    specification_name='Example NLLS Quantile Regression',
)
print(fit)
```

**Output:**

```text
Nonlinear Least Squares Results
Example NLLS Quantile Regression
...
Iterations:                      20
Covariance Type:                HC1
...
Intercept   0.4495  ****   0.03059   14.69  <0.001   0.3894   0.5096
beta         3.324  ****   0.03318  100.17  <0.001    3.259    3.389
gamma      -0.4645  ****  0.002868 -161.94  <0.001  -0.4701  -0.4588

Loss Function: QuantileHuberLoss(tau=0.3000, k=1.00e-03)
message: |dx| < xtol * max(1, |x|)
```

### `example_nonlinear_least_squares_exponential_block_cluster.py`

Cluster-robust covariance.

```python
fit = nlls(
    '[y] ~ {Intercept} + {beta} * np.exp({gamma} * [x]) $ [w]',
    df,
    cov_type='cluster',
    cov_kwds={'groups': 'grp'},
    specification_name='Example Exponential',
)
print(fit)
```

**Output:**

```text
Nonlinear Least Squares Results
Example Exponential
...
Nobs:                             250    Iterations:                         7
...
Intercept  0.7879  *       0.346   2.28   0.024   0.1065    1.469
beta        3.129  ****   0.3425   9.14  <0.001    2.454    3.804
gamma      -0.492  ****  0.03831 -12.84  <0.001  -0.5675  -0.4166
```

### `example_nonlinear_least_squares_exponential_block_bootstrap.py`

Grouped Bayesian bootstrap.

```python
fit = nlls(
    '[y] ~ {Intercept} + {beta} * np.exp({gamma} * [x]) $ [w]',
    df,
    cov_type='bootstrap',
    cov_kwds={'groups': 'grp', 'n_samples': 100},
    specification_name='Example Exponential',
)
print(fit)
```

**Output:**

```text
...
Covariance Type:          BOOTSTRAP
...
Intercept    1.109  **     0.3352   3.31   0.009   0.3506    1.867
beta         2.914  ****   0.3456   8.43  <0.001    2.132    3.695
gamma      -0.5172  ****  0.04219 -12.26  <0.001  -0.6126  -0.4217

Did 100 Bayesian bootstrap repetitions, alpha=1.000, blocked on 'grp'.
```

### `example_bounded_least_squares.py`

25 nonnegative slope coefficients.

```python
import numpy as np
import pandas as pd
from kanly.api import nlls

n, k = 5_000, 25
np.random.seed(0)
df = pd.DataFrame({f'x{j:02}': np.random.randn(n) for j in range(k)})
beta = np.random.randn(k)
df['y'] = 1.6 + df.dot(beta) + np.random.randn(n)

fit = nlls(
    '[y] ~ {Intercept} + ' + ' + '.join([f'{{alpha{j:02}}}*[x{j:02}]' for j in range(k)]),
    df,
    bounds=[(-np.inf, np.inf)] + [(0, np.inf)] * k,
    specification_name='bounded least squares example',
)
print(fit)
```

**Output:**

```text
bounded least squares example
...
Nobs:                          5000    Iterations:                       8
Active Constraints:              10
Covariance Type:               None
...
Intercept  1.56661
alpha00    0.00000
alpha01    1.96261
...
alpha24    2.21391
```

### `example_nonlinear_least_squares_elastic_net.py`

Large penalized model via `nlls_en` (coordinate descent).

```python
from kanly.api import nlls_en

# ... build df with x, w, g, z0..z9 ...

alpha = {f'q{j}': 100000 * j for j in range(n_z2)}
alpha.update({f'C(g)[{j}]': 100000 * j for j in range(n_g)})

fit = nlls_en(
    '[y] ~ {a} + [poly(x,3,-1)] + np.exp({zeta}*[x]) * [w]**(1+{beta})'
    + ' + np.exp(-1+{phi}*[x]+{psi}*[z0])'
    + '+' + ' + '.join(f'{{q{j}}}*[z{j}]' for j in range(n_z2))
    + '+ [C(g,-1)] $ [w]',
    df,
    alpha=alpha,
    max_iter=1_500,
    active_set=True,
    selection='cyclic',
    seed=0,
)
print(fit)
```

**Output:**

```text
Nonlinear Least Squares Results
...
Nobs:                         25000    Fit Time:                    10.71s
Df Model:                        47    Iterations:                     424
Method:                          CD
Covariance Type:       NOT COMPUTED
...
a             -0.162
beta         -0.8475
zeta           0.984
_A[x]         0.5468
C(g)[1]   -2.976e-06
...
message: converged, relative change in objective < ftol
```

### `example_nonlinear_least_squares_large_scale.py`

~600 parameters from `C(geo,-1)` and `C(day,-1)`; `subsample` warm start.

```python
fit = nlls(
    '[y] ~ ({alpha} + {beta} * np.exp([x])) * (1 + [C(geo,-1)]) * (1 + [C(day,-1)])',
    df,  # millions of rows; geo/day fixed effects
    cov_type='hc1',
    subsample=25_000,
    max_iter=100,
    debug=True,
)
print(fit)
```

**Output** (trust-region log on subsample, then summary on full data):

```text
Nobs:         50000
Num Params:   607
...
  iter       F=cost       dF/F     pred/F          F/n       rho     Delta  ...
     0   3.7976e+06  -6.95e-01  ...                           steihaug
...
    24   6.4298e+05  -2.20e-11  ...                           steihaug
	Converged: |dx| < xtol * max(1, |x|)

Nobs:                       3000000
Df Model:                       607    Iterations:                      28
R-squared:                   0.9092    Covariance Type:                HC1
alpha        1.00646  0.01449     69.45  <0.001   0.97806  1.03486
beta         2.00104  0.02660     75.24  <0.001   1.94891  2.05317
C(geo)[1]    3.36767  ...
... hundreds of C(geo)[*] and C(day)[*] coefficients ...
```

### `example_nonlinear_least_squares_large_scale_bounded.py`

5M rows with box bounds on geo fixed effects.

```python
fit = nlls(
    '[y] ~ ({alpha} + {beta} * np.exp([x])) * (1 + [C(geo,-1)]) * (1 + [C(day,-1)])',
    df,
    subsample=10_000,
    max_iter=100,
    bounds=np.array([(-np.inf, np.inf)] * 7 + [(-.4, 2)] * num_geo),
    debug=True,
)
print(fit)
```

**Output:**

```text
Nobs:         5000000
Num Params:   407
Bounded:      True
Estimating a starting point on 10000/5000000 random observations...
...
Nobs:                       5000000    Fit Time:                   585.45s
Df Model:                       407    Iterations:                      24
Active Constraints:             105
Covariance Type:               None
alpha        1.82490
beta         3.64381
C(geo)[1]   -0.98312
...
C(day)[6]    0.02722

Cannot compute variance covariance with active constraints!
```
