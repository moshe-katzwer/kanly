# `kanly.bootstrap`

**See also:** [kanly README](../../README.md) · [linear models](../regression/linear_models/README.md)

**Resampling-based covariance** for `kanly` regression fits: **Bayesian bootstrap** (Dirichlet weights) and **classical bootstrap** (multinomial / with-replacement counts), with optional **cluster (block) bootstrap** when observations are correlated within groups.

**Public API on `kanly.api`:** [`get_joint_bootstrapped_distribution`](bootstrap.py) — joint covariance of stacked parameters across several fits that were already bootstrapped with the **same** scheme.

The heavy lifting lives in [`bootstrap.py`](bootstrap.py): weight generators, `do_bootstrap2` (mutates fits, optional **Ray** parallelism), `bootstrap_entire_procedure` (generic sequential loop), and helpers.

---

## Concepts

| Mechanism | Weights | Blocked (`groups`) |
|-----------|---------|---------------------|
| **Bayesian** (`BAYESIAN`) | Dirichlet(`α`,…,`α`), scaled to sum to `nobs` (or `#` clusters) | One weight per cluster, broadcast to rows |
| **Classical** (`CLASSICAL`) | Integer counts from resampling rows (or clusters) with replacement | Resample cluster IDs, map to row weights |

Bayesian weights stay strictly positive for `α > 0`; classical weights can be zero on some rows.

---

## Main functions (`bootstrap.py`)

| Function | Role |
|----------|------|
| [`blocks_to_ind`](bootstrap.py) | Map arbitrary group labels → `0 … K-1` integer codes (`pandas.Series`) |
| [`get_bayesian_bootstrap_weights`](bootstrap.py) | Generator of weight vectors + optional `num_unique` |
| [`get_classical_bootstrap_weights2`](bootstrap.py) | Same for classical weights |
| [`get_bootstrap_weights2`](bootstrap.py) | Multiply bootstrap weights by optional user **frequency** weights |
| [`_get_fits`](bootstrap.py) | Normalize single fit / list / dict → dict of fits |
| [`do_bootstrap2`](bootstrap.py) | Full loop: draws → `param_estimation_func` → `np.cov` → `fit.set_cov_params` / `set_bootstrapped_params`; `max_processes > 1` uses **Ray** |
| [`get_bootstrapped_param_draws2`](bootstrap.py) | Consume a weight iterator, stack parameter draws (handles multi-outcome 3-D arrays, `pandas.Series` alignment via `exog_names`) |
| [`get_joint_bootstrapped_distribution`](bootstrap.py) | `np.cov` of **horizontally stacked** `bootstrapped_params` across fits; optional `n/(n-1)` correction; labeled `DataFrame` or raw `ndarray` |
| [`bootstrap_entire_procedure`](bootstrap.py) | Bayesian weights only; call arbitrary `func(weights)`; returns dict with `result`, `nobs`, `blocks`, `options` |

---

## `do_bootstrap2` workflow

1. Normalize `fits` to a dict (`_get_fits`).
2. Split `n_samples` across workers; distinct RNG seeds per worker.
3. Each worker builds a weight generator (`get_*_bootstrap_weights`) and runs `get_bootstrapped_param_draws2` with `return_var_covar=False`.
4. Driver concatenates draws, computes `np.cov` per outcome, optional **small-sample** factor `n_samples / (n_samples - 1)`.
5. **`set_cov_params`** with `cov_type=BOOTSTRAP`, `df_t_dist` = `#clusters - 1` if clustered else residual df.
6. **`set_bootstrapped_params`** plus human-readable summary string; **`cov_elapsed`** on each fit.

---

## `get_joint_bootstrapped_distribution`

Use when you have **multiple outcomes** (or models) bootstrapped **with identical** `cov_kwds` and **`nobs`**, so each bootstrap row index lines up across fits. Validates `cov_type == 'bootstrap'` and matching metadata, then forms the joint covariance of all parameters stacked side-by-side.

---

## Dependencies

- **NumPy**, **pandas**, **SciPy** (`issparse`), **tqdm**, **Ray** (for `max_processes > 1` in `do_bootstrap2`), **`ray.experimental.tqdm_ray`** for worker progress when `debug=True`.

---

## Using bootstrap via model APIs (`lm`, `glm`, `nlls`, …)

Pass `cov_type='bootstrap'` and `cov_kwds` on formula fits. Examples match the [root user guide](../../README.md#cluster-robust-hc-hac-and-bootstrap) and [linear models README](../regression/linear_models/README.md#bootstrap-standard-errors):

```python
from kanly.api import lm

fit = lm('y ~ x', df, cov_type='cluster', cov_kwds={'groups': 'firm_id'})

# Two-way clustering: pass a tuple of grouping columns
fit = lm('y ~ x', df, cov_type='cluster',
         cov_kwds={'groups': ('grp1', 'grp2')})

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

| `cov_kwds['method']` | Weights | Typical use |
| -------------------- | ------- | ----------- |
| `'bayesian'` (default) | Dirichlet(`alpha`, …, `alpha`), scaled to `nobs` or `#` clusters | Smooth, strictly positive weights; default for many kanly workflows |
| `'classical'` | Integer replication counts (rows or clusters) | Traditional with-replacement bootstrap |

With `groups` in `cov_kwds`, both methods **resample clusters** (block bootstrap): one weight vector per cluster, broadcast to member rows.

A helper `two_way_cluster` exists in [`kanly/regression/linear_models/two_way_cluster.py`](../regression/linear_models/two_way_cluster.py) but is **not** exported from `kanly.api`; use the tuple-`groups` pattern on `lm` or import that module directly.

Supported `cov_type` values on linear models include `'OLS'`, `'OLS_SMALL'`, `'HC0'`–`'HC3'`, `'HAC'`, `'HAC_PANEL'`, `'CLUSTER'`, `'BOOTSTRAP'`, etc.

---

## Minimal pattern (joint covariance)

After each fit has been through the same bootstrap (e.g. via model `cov_kwds` that call into this module):

```python
from kanly.api import get_joint_bootstrapped_distribution

V = get_joint_bootstrapped_distribution([fit_a, fit_b], return_dataframe=True)
```

See source docstrings in [`bootstrap.py`](bootstrap.py) for `do_bootstrap2` and `bootstrap_entire_procedure` argument lists.

---

## References

**Classical (resampling) bootstrap**

- Efron, B. (1979). [Bootstrap Methods: Another Look at the Jackknife](https://doi.org/10.1214/aos/1176344552). *The Annals of Statistics*, 7(1), 1–26.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC (Monographs on Statistics and Applied Probability, 57). [Publisher page](https://www.routledge.com/An-Introduction-to-the-Bootstrap/Efron-Tibshirani/p/book/9780412042317)
- Wikipedia: [Bootstrapping (statistics)](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))

**Bayesian bootstrap**

- Rubin, D. B. (1981). [The Bayesian Bootstrap](https://doi.org/10.1214/aos/1176345338). *The Annals of Statistics*, 9(1), 130–134.
- Wikipedia (context within resampling): [Bootstrapping (statistics) — Bayesian bootstrap](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Bayesian_bootstrap)

**Cluster / block resampling** (when `groups` is used)

- Wikipedia: [Bootstrapping (statistics) — Block bootstrap](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Block_bootstrap)
- Künsch, H. R. (1989). [The Jackknife and the Bootstrap for General Stationary Observations](https://doi.org/10.1214/aos/1176347265). *The Annals of Statistics*, 17(3), 1217–1241. (Moving block bootstrap for dependent data; conceptually related to resampling *blocks* rather than rows.)
