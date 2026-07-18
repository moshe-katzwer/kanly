# `kanly.utils`

**See also:** [kanly README](../../README.md) · [plot](../plot/README.md)

Miscellaneous helpers used across **regression**, **Bayesian** models, **optimization**, and **nonparametric** code. Most modules are imported internally; only a few symbols are re-exported on **`kanly.api`**.

**Public API (`kanly.api`):** `compare_results`, `latex_table`, `get_highest_density_interval`, `timer`

For plotting utilities, see [`kanly/plot/README.md`](../plot/README.md) (`plot`, `scatter`, `hist`).

---

## Module map

| File | Role |
|------|------|
| [`compare_results.py`](compare_results.py) | Side-by-side regression / NLLS results tables (esttab-style) |
| [`latex_table.py`](latex_table.py) | LaTeX tabular output for multiple fits |
| [`highest_density_interval.py`](highest_density_interval.py) | Shortest empirical interval at a coverage level (HDI) |
| [`timer.py`](timer.py) | Lightweight start/stop timing by name |
| [`linalg_utils.py`](linalg_utils.py) | Gram matrices, inverses, sandwich pieces, sparse↔dense policy |
| [`util.py`](util.py) | DataFrame coercion, dense conversion, iteration logging |
| [`dict_2_array.py`](dict_2_array.py) | Ordered parameter dict → `numpy` vector |
| [`overwrite_parameter.py`](overwrite_parameter.py) | Patch parameter vectors by name |
| [`dataframe_utils.py`](dataframe_utils.py) | Merge DataFrames; iterate slices by key |
| [`parse_code_string.py`](parse_code_string.py) | Extract `{param}` / `$param$` tokens from strings |
| [`parse_string_2_tuple.py`](parse_string_2_tuple.py) | Comma-split respecting nested `()`, `[]`, `{}` |
| [`function_str_to_callable.py`](function_str_to_callable.py) | Compile formula strings to callables (`FunctionCallable`) |
| [`logit_functions.py`](logit_functions.py) | `logit`, `expit`, derivatives for transforms / Numba |
| [`stats_functions.py`](stats_functions.py) | Numba-friendly normal PDF/CDF pieces for string-compiled code |
| [`fast_histogram.py`](fast_histogram.py) | Numba histogram on a fixed grid |
| [`fast_interp1d.py`](fast_interp1d.py) | Numba binary-search linear interpolation |
| [`plot_confidence_intervals.py`](plot_confidence_intervals.py) | Matplotlib normal-approx CI plots from fits |
| [`print_options.py`](print_options.py) | Pretty-print option dicts to the console |
| [`user_prompt_for_more_iters.py`](user_prompt_for_more_iters.py) | Interactive “more iterations?” prompt for optimizers |
| [`kde_clipped.py`](kde_clipped.py) | Deprecated KDE clipping experiment (do not use) |

---

## Results tables and export

### `compare_results`

Build a **multi-column summary** of coefficient estimates from several fitted models (`RegressionResultsBase` subclasses: `lm`, `glm`, `nlls`, etc.). Optional standard errors, *t*-stats, *p*-values, significance stars, formulas, and a reference parameter column.

```python
from kanly.api import lm, compare_results

fit_ols = lm('y ~ x + C(grp) $ w', df)
fit_iv = lm('y ~ x + C(grp) | z + C(grp) $ w', df)

print(compare_results(
    [fit_iv, fit_ols],
    parameter_subset=['x'],
    ref_param_values={'x': -0.3},
    show_bse=True,
    show_formulas=True,
))
```

See [`examples/regression/linear_models/instrumental_variables/example_instrumental_variables.py`](../../examples/regression/linear_models/instrumental_variables/example_instrumental_variables.py) for a full IV vs WLS comparison.

### `latex_table`

Same idea as `compare_results`, but emits a **LaTeX `tabular`** fragment for papers or slides (`show_bse`, stars, optional *t* / *p*).

```python
from kanly.api import latex_table

print(latex_table([fit_a, fit_b], sigfigs=3, show_stars=True))
```

### `plot_confidence_intervals`

**Matplotlib** helper: error-bar style plot of normal-approximation CIs from `fit.params` / `fit.bse` (not on `kanly.api`). Used by MCMC diagnostic code paths.

```python
from kanly.utils.plot_confidence_intervals import plot_normal_conf_intervals_from_fit

fig = plot_normal_conf_intervals_from_fit(fit, params=['x', 'Intercept'], level=0.95, show=True)
```

---

## Intervals and timing

### `get_highest_density_interval`

Finds the **shortest interval** `[lb, ub]` on the empirical CDF of a 1-D sample with coverage `level` (same spirit as a highest posterior density / credible interval on a histogram).

```python
import numpy as np
from kanly.api import get_highest_density_interval

samples = np.random.randn(10_000)
lb, ub = get_highest_density_interval(samples, level=0.95)
```

### `timer`

Toggle timing without boilerplate: first call starts the clock, second call with the same `name` prints elapsed seconds.

```python
from kanly.api import timer

timer('fit')
fit = lm('y ~ x', df)
timer('fit')   # prints name='fit', elapsed=...
```

---

## Linear algebra (`linalg_utils`)

Internal workhorse for **OLS / IV / sandwich** covariance code. Highlights:

| Symbol | Purpose |
|--------|---------|
| `DenseThreshold` | Decide when a sparse design is small/dense enough to materialize as `ndarray` |
| `gram_matrix` | Weighted \(X'X\) (optional GLS `sigma` / `sigma_inv`) |
| `get_matrix_inverse_internal` | \((X'X)^{-1}\) with normalization and sparse/dense routing |
| `get_normalized_cov_params` | Covariance of \(\hat\beta\) from design and weights |
| `sandwich_diagonal` | Diagonal of \(X' W X\) for robust covariances |
| `csc_matrix_by_column_array_broadcast` | Multiply sparse columns by a weight vector |
| `get_eigenvals_and_condition_number_internal` | Eigenvalues / condition number for diagnostics |
| `flexible_mat_dot_vec` | Sparse or dense matrix–vector product |

`DEFAULT_DENSE_THRESHOLD_MB` (default 1024 MB) controls the sparse→dense crossover used by large `lm` / NLLS Jacobians.

---

## Parameters and data plumbing

### `dict_2_array`

Map a **`{name: value}`** dict to a vector in **`param_names`** order; pass through non-dicts unchanged. Used by optimizers and NLLS `start_params`.

```python
from kanly.utils.dict_2_array import dict_2_array

x0 = dict_2_array({'Intercept': 1.0, 'beta': 3.0}, ['Intercept', 'beta', 'gamma'])
```

### `overwrite_parameter_index`

Replace selected entries in a parameter vector by **name** (optional copy).

### `util.py`

- `dict_2_dataframe` — accept `DataFrame`, `SparseDataFrame`, or dict-like `data`
- `to_dense_helper` — sparse CSC → dense `ndarray`
- `str_to_args` — parse comma-separated argument strings (respects nesting when used with `parse_str_2_tuple`)
- `get_eval_env_depth` — how many caller frames up for formula `eval` environments
- `print_iter_info` — aligned iteration logs for IRLS / optimizers

### `dataframe_utils`

- `merge_dataframes` — fold a list of frames on a key (`how='outer'` by default)
- `iterate_through_sub_frames` — yield `(key_value, boolean mask)` slices for grouped work

---

## Parsing and compiled callables

### `parse_code_str` (`parse_code_string.py`)

Pull delimited tokens out of a code string, e.g. `{beta}` from an NLLS formula or `$sigma$` from a `DataModel` block. Strips comments and collapses whitespace.

### `parse_str_2_tuple`

Split on commas **without breaking** nested `(...)`, `[...]`, or `{...}` — used for formula argument lists.

### `function_str_to_callable`

Compile a **Python expression string** into a callable plus metadata (`FunctionCallable`). Injects stats / transform helpers via `exec` and supports Numba `jit` in generated code. Used by legacy string objectives and tests; prefer **`kanly.api.func_str_to_callable`** (autodiff package) for new work.

```python
from kanly.utils.function_str_to_callable import get_callable_from_func_str

fc = get_callable_from_func_str('({x}-1)**2 + ({y}+2)**2')
fc.param_names   # ['x', 'y']
fc(np.array([0.0, 2.0]))
```

### `logit_functions` / `stats_functions`

**Numba-overloadable** `logit`, `expit`, and normal distribution fragments consumed by compiled model strings and Bayesian transforms.

---

## Fast numeric kernels

### `fast_histogram`

Numba `@njit` histogram on `[lower, upper]` with fixed `nbins`; optional density normalization. Used by [`kanly.nonparametric.kde`](../nonparametric/kde.py).

### `fast_interp1d`

`fast_linear_interp1d(xval, x, y)` — assumes sorted `x` and `xval` in range; binary search + linear segment.

---

## Developer ergonomics

### `print_options`

Print a titled key–value table for debugging solver / sampler options.

### `user_prompt_for_more_iters_method`

When an optimizer has not converged, optionally **prompt stdin** for extra iterations (used by `bfgs_pqn`, NLLS, etc.).

---

## Deprecated

**`kde_clipped.py`** — old statsmodels-KDE clipping experiment; marked for removal. Use [`kanly.api.kde`](../nonparametric/kde.py) instead.

---

## Import cheat sheet

| Need | Import |
|------|--------|
| Compare fits | `from kanly.api import compare_results` |
| LaTeX table | `from kanly.api import latex_table` |
| HDI on samples | `from kanly.api import get_highest_density_interval` |
| Timing | `from kanly.api import timer` |
| Param dict → vector | `from kanly.utils.dict_2_array import dict_2_array` |
| Gram / inverse internals | `from kanly.utils.linalg_utils import gram_matrix, get_matrix_inverse_internal` |

Most other symbols should be treated as **internal** unless you are extending kanly itself.

---

## Related modules (not under `utils/`)

These live elsewhere under `kanly/` and are re-exported from `kanly.api` for convenience (see the [root user guide](../../README.md#other-modules-supporting)):

| Path | Role | Notable `kanly.api` symbols |
| ---- | ---- | --------------------------- |
| `kanly/general_models/` | Generic callable fitting | `fit_general_model_callable` |
| `kanly/dill_object.py` | Serialization helpers | `read`, `save` |

**Distributions / likelihood fragments:** see [`kanly/stats/README.md`](../stats/README.md) (`kanly.api` also exposes plotting-oriented helpers such as `get_mle_x_y` from `fit_distributions_mle` — documented in source, not the stats readme).
