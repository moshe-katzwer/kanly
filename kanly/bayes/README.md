# kanly Bayes Package

**See also:** [kanly README](../../README.md)

`kanly.bayes` is the **Bayesian** layer: arbitrary log-posteriors (`BayesianModel`), **numba-compiled / string-based** `DataModel` for more complicated models, MCMC samplers (**Adaptive Metropolis**, coordinate **MALA**, combined workflows) with **multi-chain parallelism via [Ray](https://docs.ray.io/)**, **MAP / MLE**, and shortcuts that wrap frequentist formula modules with priors (`bayes_lm_model`, `bayes_glm_model`, `bayes_nlls_model`). It also includes **conjugate** normal–inverse-gamma linear regression (`blm`).

**Useful background:** [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem), [Markov chain Monte Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo), [Metropolis–Hastings algorithm](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm).

`kanly.bayes` provides Bayesian modeling tools for custom log-likelihoods,
formula-based regression models, flexible code-block data models, MCMC
sampling, MAP/MLE optimization, posterior diagnostics, and closed-form
conjugate Bayesian linear regression.

The package root does not re-export these APIs directly. Prefer importing from
`kanly.api` for common workflows, or from the concrete submodules documented
below.

## What It Does

- Build generic Bayesian models from a log-likelihood function and optional
priors.
- Apply parameter bounds through bounded-to-unbounded transformations with
Jacobian-aware posterior evaluation.
- Fit Bayesian linear, generalized-linear, and nonlinear least-squares models
from formulas.
- Define flexible statistical models with a small code-block DSL for data and
parameter references.
- Run Adaptive Metropolis-Hastings, coordinate-wise MALA, and a combined
sampler workflow; run independent MCMC chains in parallel with **Ray** (see
[Parallel chains (Ray)](#parallel-chains-ray)).
- Compute MAP and MLE estimates with bounded optimization support.
- Store MCMC output with summaries, convergence diagnostics, credible
intervals, and plotting helpers.
- Fit Bayesian linear regression analytically under a Normal-Inverse-Gamma
conjugate prior.

## Core Entry Points

Common public entry points are available through `kanly.api`:

- `bmodel`: alias for `kanly.bayes.bayesian_model.BayesianModel`
- `DataModel`: code-block model builder
- `build_data_model`: alias for `DataModel.build_data_model`
- `amha`: functional Adaptive Metropolis-Hastings sampler
- `mala`: functional coordinate-wise MALA sampler
- `blm` / `BLM`: conjugate-prior Bayesian linear regression fitters
- `bayes_lm_model`: build a Bayesian linear model from a formula
- `bayes_glm_model`: build a Bayesian generalized-linear model from a formula
- `bayes_nlls_model`: build a Bayesian nonlinear least-squares model from a
formula

### Prefer imports from `kanly.api`

Examples: `bmodel`, `DataModel`, `build_data_model`, `amha`, `mala`, `bayes_lm_model`, `bayes_glm_model`, `bayes_nlls_model`, `blm`.

## Comparison: SciPy / statsmodels / scikit-learn

| Tool | Overlap | Difference |
| ---- | ------- | ---------- |
| **SciPy** | [`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html) priors / likelihood pieces; [`scipy.optimize`](https://docs.scipy.org/doc/scipy/reference/optimize.html) for **MAP**-style point estimates | **No** adaptive MCMC, `DataModel`, or sampler orchestration — you wire `logpost` + `minimize` or roll your own chain |
| **statsmodels** | Mostly **frequentist** inference; some generic likelihood with `GenericLikelihoodModel` | No built-in adaptive MCMC stack comparable to `BayesianModel.sample`; priors are not first-class like kanly’s string/`scipy` priors |
| **scikit-learn** | `BayesianRidge` (only a narrow conjugate-like ridge) | Not a general-purpose MCMC / `DataModel` system — use **PyMC**, **Stan**, or kanly for full custom posteriors |
| **kanly** | Custom log-density for parameters given data, plus samplers and formula-wrapped likelihoods | |

You can also import directly from submodules when you need implementation-level
classes:

```python
from kanly.bayes.bayesian_model import BayesianModel
from kanly.bayes.data_model import DataModel
from kanly.bayes.bayesian_regression_model import (
    BayesianGeneralizedLinearModel,
    BayesianLinearModel,
    BayesianNonlinearLeastSquaresModel,
)
from kanly.bayes.mcmc.mcmc_results import MCMCResults
```

## BayesianModel

`BayesianModel` is the base class for any model you want to optimize or sample.
Formula regression wrappers and `DataModel.to_bayesian_model()` all produce a
`BayesianModel` instance with a built log-likelihood, priors, and bounds.

### Purpose

At a high level the class wires:

`log_posterior = log_likelihood + log_prior + bound handling`

When `do_bounded_transform=True` and bounds are present, MCMC runs in an
unbounded internal space with a Jacobian correction so stored draws on the
original scale target the correct posterior. See
[Sampling Functionality](#sampling-functionality) for **AMHA**, **MALA**, and
`sample()`.

### Constructor arguments


| Argument                     | Role                                                                                                                                                                                 |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `log_likelihood_function`    | Callable `params -> float` for the log-likelihood (or full log-density if you pass no priors).                                                                                       |
| `param_names` / `num_params` | Exactly one is required; names define vector order for dict starts and summaries.                                                                                                    |
| `priors`                     | Optional dict: per-name, tuple of names, parameter-group key, or `''` for a joint prior. Values: scipy frozen RVs, strings like `'norm(0, 1)'`, callables, or lists to stack priors. |
| `bounds`                     | `dict` of `(lower, upper)`; merged with scipy prior support when applicable.                                                                                                         |
| `do_bounded_transform`       | `True`: smooth reparameterization to ℝ for MCMC; `False`: hard `-inf` outside bounds.                                                                                                |
| `parameter_groupings`        | Block names for multivariate priors (e.g. all coefficients in a group).                                                                                                              |
| `specification_name`         | Optional label on fits and `MCMCResults`.                                                                                                                                            |


Example:

```python
from kanly.api import bmodel
import numpy as np

model = bmodel(
    log_likelihood,
    param_names=["mu", "log_sigma"],
    priors={"mu": "norm(0, 10)", "log_sigma": "norm(0, 1)"},
    bounds={"log_sigma": (-20, 20)},  # optional tightening
)
```

### What gets built internally

1. `set_transformations` — `transform` / `inv_transform`, Jacobian, bound indicators.
2. `set_priors` — parsed prior callables; may refresh bounds from distribution support.
3. `log_posterior` on the original scale; `log_posterior_transformed` (+ Jacobian) for samplers.

### Typical workflows


| Goal                           | Method                                                            |
| ------------------------------ | ----------------------------------------------------------------- |
| Posterior mode                 | `model.map(start)`                                                |
| MLE (ignore priors)            | `model.mle(start)`                                                |
| Default MCMC                   | `model.sample(start, n_burnin=..., n_samples=...)`                |
| AMHA only                      | `model.amha(start, ...)` when you already have a good `step_cov`  |
| Coordinate warmup / standalone | `model.mala(start, ...)` or `sample(..., do_mala_cd_warmup=True)` |


See [Minimal Usage](#minimal-usage) for end-to-end examples.

## Main Modules

### `bayesian_model.py`

Implementation of `BayesianModel`. See [BayesianModel](#bayesianmodel) for
construction and [Sampling Functionality](#sampling-functionality) for MCMC
entry points (`sample`, `amha`, `mala`, `map`, `mle`).

### `bayesian_regression_model.py`

This module wraps existing sparse regression builders as Bayesian models:

- `BayesianLinearModel`
- `BayesianGeneralizedLinearModel`
- `BayesianNonlinearLeastSquaresModel`

Each class builds a likelihood from a formula/data pair and then delegates
posterior evaluation, sampling, and optimization to `BayesianModel`.

Regression likelihoods include an explicit residual scale parameter named
`__sigma2`. That parameter is automatically bounded to be positive unless a
compatible bound is supplied.

`BayesianNonlinearLeastSquaresModel` uses the same formula syntax and Gaussian
error model as `kanly.regression.nonlinear_least_squares` (see
[Non-linear least squares](https://en.wikipedia.org/wiki/Non-linear_least_squares)
and the [NLLS user guide](../regression/nonlinear_least_squares/README.md#how-fitting-works)
for how classical point estimation minimizes squared residuals). In the Bayesian
wrapper, `map(...)` / `mle(...)` optimize that likelihood; `sample(...)` /
`amha(...)` explore the posterior with priors on top.

The shared base class also provides `get_elastic_net_log_prior(...)`, which can
build ridge, LASSO, or elastic-net style log-priors over regression
coefficients.

### `data_model.py`

`DataModel` builds flexible statistical models from two code blocks:

- A data block that stores variables from a DataFrame or dictionary.
- A model block that returns a log-likelihood or log-posterior value.

The data block is Python code that assigns attributes on `self`. Data columns
or formula-like data expressions are wrapped in backticks:

```python
self.x = `x`
self.x2 = `I(x**2)`
self.g = `C(g)`
self.root_weights = np.sqrt(`wts`)
```

Backtick expressions are parsed from the supplied DataFrame or dictionary. The
supported data expressions are column names, categorical expressions such as
`C(g)`, and arithmetic expressions such as `I(x**2)`. Categorical expressions
are stored as integer level codes, so they can later index parameter vectors in
the model block. Interactions such as `C(g1,g2>0)` are supported in the
package's categorical syntax.

The data block and model block can use NumPy functions and symbols imported by
the package statistics import string. You can also pass `other_variables` to
`DataModel.build_data_model(...)` to make custom arrays, constants, or
functions available in both generated code blocks. Expensive reusable
quantities, such as frozen log-pdf callables, can be created once in the data
block and reused inside the model block.

By default, model-block parameters use `$...$` syntax:

- `$name$`: scalar parameter named `name`.
- `$name<lb, ub>$`: scalar parameter bounded between `lb` and `ub`.
- `$name[p]$`: vector parameter named `name` with length `p`.
- `$name[p]<lb, ub>$`: bounded vector parameter.

For vector parameters, declare the vector once and access elements through the
vector expression. For example, if the model contains `$beta[4]$`, use
`$beta$[2]` to reference its third element elsewhere in the model block. Bounds
also only need to be declared once; later `$beta$` references reuse the parsed
metadata.

The model block should end with a `return` statement. If it returns a tuple,
calling the `DataModel` returns the first element by default. Use
`return_first=False` to inspect the full tuple, for example
`(log_posterior, log_likelihood, log_prior)`.

Convenience parameter expansions include:

- `$_dummy[x]$`: categorical fixed effects for a categorical data variable
`x`.
- `$_dummy[x,-1]$`: fixed effects with the first level dropped.
- `$_dummy[x,-1;name]$`: fixed effects with an override name to disambiguate
repeated dummy terms.
- `$_bs[args]$`, `$_cc[args]$`, and `$_cr[args]$`: Patsy spline bases for
B-splines, cyclic cubic splines, and natural cubic splines. Add `;name` to
override generated parameter names.
- `$_poly[(x,y),2]$`: polynomial expansion up to the requested degree, including
interactions such as `x*y`.
- `$_par[...]$`: reference the raw parameter vector created by an expansion
rather than the model expression generated by that expansion.

Bounds can be applied to these expanded terms too, for example
`$_dummy[g]<0, 1>$` bounds each generated dummy coefficient.

`DataModel.build_data_model(...)` parses the blocks, extracts parameter
metadata, records bounds and parameter groupings, and compiles an internal
callable. The resulting object can be evaluated directly with arrays or
parameter dictionaries, printed to inspect the generated model function, or
converted with `data_model.to_bayesian_model(...)`.

`to_bayesian_model(...)` wraps the `DataModel` callable as a `BayesianModel`.
This assumes the first returned value is the log-density to sample from, usually
the full log-posterior. Additional priors and bounds passed to
`to_bayesian_model(...)` are merged with the bounds parsed from the model block.

### `mcmc/`

The `mcmc` subpackage contains samplers and diagnostics. **AMHA** and **MALA**
both use **[Ray](https://docs.ray.io/)** to run one worker process per MCMC
chain: the log-posterior (and Jacobian adjustment when bounds are
reparameterized) is placed in the Ray object store once, then each chain runs
in a `@ray.remote` task. Ray is listed in the repo `requirements.in`; install
it with the rest of the kanly stack before calling `sample`, `amha`, or `mala`.

- `adaptive_metropolis/adaptive_metropolis_mcmc.py`: functional `amha(...)`
sampler and per-chain Adaptive Metropolis internals (`run_adaptive_metropolis_chain_remote`).
- `coordinate_mala/coordinate_mala_mcmc.py`: functional `mala(...)` sampler and
coordinate-wise MALA internals (`run_mala_chain_remote`).
- `mcmc_results.py`: `MCMCResults`, the main result object.
- `diagnostics/`: split R-hat, ESS, batched means, and Geweke helpers.
- `check_starting_point.py`: validation for initial points.
- `format_starting_point_for_mcmc.py`: utilities for per-chain starting points
and transformed parameter spaces.
- `aggregate_covariances.py`: covariance aggregation helpers for adaptive
proposal tuning.

`MCMCResults` stores samples, per-draw metadata, posterior summaries,
covariance estimates, MAP draw information, acceptance rates, and diagnostics.
It also supports plotting helpers, credible intervals, HPDI computation, and
serialization through `DillObject`.

## Sampling Functionality

**AMHA** (Adaptive Metropolis–Hastings with optional differential evolution) and
**MALA** (coordinate-wise Metropolis-adjusted Langevin algorithm) are kanly’s
custom MCMC implementations. They are related to textbook adaptive Metropolis
and MALA but add differential-evolution (DE) proposals and block-wise adaptation
tuned for this codebase.

### Overview

Three related entry points on [BayesianModel](#bayesianmodel):


| Method              | When to use                                                                |
| ------------------- | -------------------------------------------------------------------------- |
| `model.sample(...)` | Default: optional MALA warmup, then AMHA main phase.                       |
| `model.amha(...)`   | AMHA only — you already have a reasonable `step_cov` or want full control. |
| `model.mala(...)`   | Coordinate sampler alone, or to understand warmup behavior.                |


`kanly.api.amha` and `kanly.api.mala` call the same core routines on arbitrary
log-densities. Model methods add bounds, transforms, Jacobian adjustment,
parameter names, and `MCMCResults` metadata.

Defaults (overridable) live in `kanly.bayes.mcmc.adaptive_metropolis.constants`
and `kanly.bayes.mcmc.coordinate_mala.constants` — e.g. `target_acceptance_rate`
≈ 0.234 for AMHA, ≈ 0.57 for MALA; `do_diff_evolution_mc=True` for AMHA;
`do_parallel=True` and `max_processes=8` for AMHA chain dispatch.

See [References](#references) for Metropolis–Hastings, Haario et al., ter Braak,
and MALA background.

### Parallel chains (Ray)

Multi-chain MCMC is the main use of Ray in `kanly.bayes`:

| Sampler | Parallelism | Controls |
| ------- | ----------- | -------- |
| **AMHA** (`amha`, `BayesianModel.amha`, AMHA phase of `sample`) | One Ray task per chain when `do_parallel=True` (default) | `do_parallel`, `max_processes` (capped at `n_chains`), `n_chains` |
| **MALA** (`mala`, MALA warmup in `sample`) | Chains always dispatched via Ray | `max_processes` (default 12), `n_chains` |

**What Ray does:** `ray.init(num_cpus=min(max_processes, n_chains))` starts a
local cluster; `ray.put(log_posterior)` (and related callables / DE pools)
avoids re-pickling the posterior on every chain. Each chain executes
`run_*_chain_remote.remote(...)`; the driver collects draws with `ray.get(...)`.
Long runs are split into **sub-chains** (`max_subchain_draws_*`) so workers stay
responsive and proposal tuning (covariance / `tau` / DE pools) can update between
blocks. Ray is shut down when sampling finishes (or on interrupt).

**Serial fallback (AMHA only):** pass `do_parallel=False` to `amha` or
`BayesianModel.sample(...)` / `amha(...)` to run chains in-process on one CPU.
MALA does not expose a serial mode; use fewer chains or lower `max_processes`
if you need to limit parallelism.

**Example:**

```python
fit = model.amha(
    start,
    n_chains=6,
    n_samples=20_000,
    n_burnin=5_000,
    do_parallel=True,   # default
    max_processes=6,    # use up to 6 Ray workers
)
```

On machines with fewer cores than `n_chains`, set `max_processes` to the number
of CPUs you want Ray to reserve. If Ray was already initialized elsewhere in the
session, kanly may re-`init` when available CPUs are insufficient.

### 1. AMHA (Adaptive Metropolis–Hastings + DE)

`BayesianModel.amha` runs multichain Metropolis–Hastings on
`log_posterior_transformed` (plus Jacobian when bounds are reparameterized).

Kanly’s AMHA is **not** adaptive Metropolis alone:

- **Adaptive covariance** (`do_adaptive=True`): proposal matrix `step_cov` is
updated from pooled chain history between blocks (`max_subchain_draws_burnin`
/ `max_subchain_draws_sample` control block size).
- **Global scale** `scaler`: tuned toward `target_acceptance_rate` (default
~0.234) using Robbins–Monro-style updates.
- **DE mix** (`do_diff_evolution_mc=True`, default): proposals move along
`ν = x_{j1} - x_{j2}` from past draws plus Gaussian noise, with weight
`diff_evolution_weight` (default ~0.95). History starts after
`diff_evolution_frac_burnin` of burn-in; cap `diff_evolution_max_draws`.

**Phases:** fixed counts `n_burnin` then `n_samples` per chain. Burn-in draws
are retained but labeled; you can relabel via `MCMCResults.set_n_burnin`.
`stop_adaptation_after_burnin=True` freezes adaptation after burn-in (default
`False`).

**When to call `amha` directly:** you have a good `step_cov` (e.g. from a pilot
run or external estimate) and do not need coordinate warmup.

Key arguments: `start_params`, `step_cov`, `n_chains`, `n_burnin`, `n_samples`,
`do_parallel`, `max_processes`, `fix_params`, `proposal_df` (Student-t vs Gaussian).

**Ray:** with `do_parallel=True` (default), each of the `n_chains` runs in its
own Ray worker; set `do_parallel=False` for debugging or single-core runs.

### 2. MALA (coordinate-wise MALA + optional DE)

`BayesianModel.mala` updates **one random coordinate** per micro-step using
finite-difference ∂/∂θ_k of the log-posterior.

- `**do_mala=True` (default):** Langevin drift along the chosen coordinate plus
Gaussian noise; Hastings correction for asymmetric proposals.
- `**do_mala=False`:** coordinate random-walk Metropolis.
- `**tau`:** per-coordinate step sizes adapted toward `target_acceptance_rate`
(default ~0.57).
- **DE cadence:** every `diff_evolution_step_cadence` iterations, a full
dimensional DE step can replace a coordinate update.

**Burn-in layout:** `frac_burnin` × `n_samples` (fraction), unlike AMHA’s
fixed `n_burnin` count.

**Ray:** `mala` always parallelizes chains with Ray (`run_mala_chain_remote`);
control load with `n_chains` and `max_processes` (default 12). The optional
MALA warmup inside `sample()` uses the same Ray-backed `mala` path.

Warmup output used by `sample()`: `other_info['cov_params_unbounded_space']` for
initial `step_cov`; last-chain draws for restarts; optional DE history via
`get_inv_transform_draws` when DE is enabled.

### 3. `BayesianModel.sample()` — combined workflow

`sample()` is the recommended high-level path for difficult posteriors.

**Defaults:** `do_mala_cd_warmup=False` (AMHA only). Enable warmup when
parameters have very different scales or strong correlation.

#### Phase A — optional MALA warmup (`do_mala_cd_warmup=True`)

**Why:** a blind identity `step_cov` for AMHA often yields near-zero acceptance
when scales and correlations are mismatched.

**What happens:**

1. Coordinate MALA runs (`n_samples_mala`, `frac_burnin_mala`, `n_chains_mala`).
2. Each step picks a random coordinate k, estimates ∂ log π/∂θ_k by finite
  differences, proposes along k with adapted `tau`, accepts with Hastings
   correction.
3. Blocks regroup chains; `tau` and optional DE pools update between blocks.

#### Handoff to AMHA (when warmup ran)

- **Starts:** last MALA draw per chain (`get_last_sample`).
- `**step_cov`:** sample covariance in unbounded space from MALA
(`cov_params_unbounded_space`).
- **DE history:** `diff_evolution_past_samples` from `get_inv_transform_draws`
when `do_diff_evolution_mc=True`.

#### Phase B — AMHA main run

Multivariate proposals: mix DE direction ν with scaled Gaussian noise; accept on
log-posterior ratio; adapt `scaler` and `step_cov` between blocks. Pseudocode
and references are in `BayesianModel.sample` docstring. Both warmup (if enabled)
and this phase inherit `do_parallel` / `max_processes` for Ray chain dispatch.

**Warmup retention:** `keep_mala_warmup=True` (default) stores warmup
`MCMCResults` in `fit.other_info['mala_cd_warmup_fit']`. If `False`, warmup time
is added to `mcmc_time` but warmup draws are discarded.

### Bounds and Transformed Space

When a model has bounds and `do_bounded_transform=True`, samplers operate in an
unbounded internal space. The result object transforms stored samples back to
the original parameter scale and uses the Jacobian adjustment in the
Metropolis-Hastings acceptance calculation.

Most callers should leave `start_params_is_original_scale=True`, which means
starting values are supplied on the same scale as the model parameters. Only set
it to `False` if you are intentionally supplying starts in the transformed
unbounded space.

### Sampling Results and Diagnostics

All model samplers return `MCMCResults`. Important attributes and methods
include:

- `sample_df`: post-transform draws for all chains.
- `sample_info_df`: chain, burn-in, log-posterior, process, and block metadata.
- `summary_df`: posterior means, standard deviations, Monte Carlo standard
errors, quantiles, credible intervals, ESS, and R-hat.
- `map_params`: the sampled draw with the highest log-posterior value.
- `acceptance_rate`: average chain acceptance rate.
- `summary(...)`: formatted text summary.
- `diagnostic_plot(...)`, `hist(...)`, `scatter(...)`, and `rank_plot(...)`:
visual checks of individual parameters and chain mixing.
- `plot_credible_intervals(...)`: interval plots for selected parameters or
transformations.
- `set_n_burnin(...)`: update the burn-in cutoff and recompute summaries.

Use R-hat, ESS, acceptance rates, rank plots, trace behavior, and domain
knowledge together. No single diagnostic proves convergence.

### `map/`

`map/maximum_a_posteriori.py` contains the optimization helpers used by
`BayesianModel.map(...)` and `BayesianModel.mle(...)`, including
`BayesianModelMaximizationResult`.

### `bayesian_linear_regression_conjugate_prior/`

This subpackage implements closed-form Bayesian linear regression with a
Normal-Inverse-Gamma prior:

- `NormalInverseGamma`: prior/posterior kernel over coefficients and `sigma2`.
- `BayesianLinearRegressionConjugatePriorModel`: formula/data model class.
- `BayesianLinearRegressionResults`: result object for posterior summaries.

Use this route when the model is Gaussian linear regression and the conjugate
prior is appropriate. It avoids MCMC by updating posterior hyperparameters
analytically.

### `parameter.py` and `parameter_transformations.py`

`parameter.py` defines parsed parameter metadata used by `DataModel`.

`parameter_transformations.py` defines bounded-to-unbounded transformations,
inverse transformations, log-Jacobian adjustments, and helpers for converting
samples between transformed and original parameter spaces.

### `utils/`

The `utils` folder contains supporting functions for prior conversion,
compiled log-pdf creation, observed-information approximations, and bound-aware
numerical helpers.

## Minimal Usage

### Custom Bayesian Model

```python
import numpy as np
from kanly.api import bmodel


def log_likelihood(params):
    mu, sigma = params
    resid = y - mu
    return -0.5 * np.sum((resid / sigma) ** 2) - len(y) * np.log(sigma)


model = bmodel(
    log_likelihood,
    param_names=["mu", "sigma"],
    bounds={"sigma": (0.0, np.inf)},
    priors={"mu": "norm(0, 10)"},
)

map_fit = model.map([0.0, 1.0])
mcmc_fit = model.sample([0.0, 1.0], n_samples=10_000, n_burnin=2_000)
print(mcmc_fit.summary_df)
```

The likelihood is a plain Gaussian model for `y` given mean `mu` and scale `sigma`.
`bmodel` attaches a Normal prior on `mu`, enforces `sigma > 0` through the bounds
machinery, and evaluates the log-posterior as log-likelihood plus log-prior (with
Jacobian terms when sampling in transformed space). `map` finds a posterior mode;
`sample` explores the full posterior with the default combined sampler workflow.
`summary_df` tabulates posterior means, intervals, ESS, and R-hat for each parameter.

### Formula-Based Bayesian Linear Model

```python
from kanly.api import bayes_lm_model


model = bayes_lm_model(
    "y ~ x $ wts",
    data,
    priors={"x": "truncnorm(1, 4, 2, 12)"},
    bounds={"x": [1, 12]},
)

fit = model.amha(
    [0, 15, 1],
    thinning=2,
    n_samples=10_000,
    max_subchain_draws_sample=50_000,
)
print(fit)
```

`bayes_lm_model` parses `y ~ x $ wts` into weighted least-squares structure, then
exposes a `BayesianModel` whose log-density is the regression likelihood (including
`__sigma2`) plus priors such as the truncated Normal on `x`. `amha` runs adaptive
Metropolis–Hastings directly on that posterior; `thinning=2` keeps every second
post-burn-in draw, and `max_subchain_draws_sample` caps how many raw subchain steps
are retained per chain during tuning.

### Code-Block Data Model

```python
from kanly.api import DataModel


data_string = """
self.x = `x`
self.y = `y`
self.weights = `wts`
self.root_weights = np.sqrt(self.weights)
"""

model_string = """
pred = $Intercept$ + $x$ * x
resid = y - pred

return logpdf_norm(
    resid,
    loc=0.0,
    scale=$sigma<0, np.inf>$ / root_weights,
).sum()
"""

data_model = DataModel.build_data_model(
    data_string,
    model_string,
    data,
    nopython=False,
)

model = data_model.to_bayesian_model(
    priors={"x": "norm(0, 10)"},
    bounds={"x": [0, 10]},
)

fit = model.sample([1.0, 1.0, 1.0], n_samples=10_000, n_burnin=3_000)
```

**WLS regression + MCMC** ([`examples/bayes/example_data_model.py`](../../examples/bayes/example_data_model.py)) — richer `DataModel` with categoricals and polynomials:

```python
from kanly.api import DataModel
import numpy as np

data = {'x': x, 'y': y, 'wts': wts, 'g': g, 'z': z}

data_string = """
self.x = `x`
self.z = `z`
self.y = `y`
self.weights = `wts`
self.g = `C(g)`
self.root_weights = np.sqrt(self.weights)
"""

model_string = """
pred = $Intercept$ + $x$ * x + $_dummy[g,-1]$ + $_poly[z,2]$
resid = y - pred
return logpdf_norm(resid, loc=0.0, scale=$sigma$ / root_weights).sum()
"""

dataobj = DataModel.build_data_model(data_string, model_string, data, nopython=False)
model = dataobj.to_bayesian_model(priors={'x': 'norm(-23.6, 1.33)'}, bounds={'x': [0, 10]})
fit = model.sample(1.2 * np.ones(8), n_samples=10_000, n_burnin=3_000, n_chains=6, ...)
print(fit)
```

Example output (from the same file):

```text
MCMC Results
Method:             AMH
No. Chains:                     6    Samples:                10000
Maximum Log Posterior:     -5.7470e+02

              mean     std      MCSE    median   [.05,    .95]     ESS   R_hat
Intercept    1.506   1.996   0.02649     1.432   -1.728  5.117  5678.0  1.0019
x            6.343   4.287     0.206     9.184  0.03976  9.704   433.0  1.1810
C(g)[1]       2.07   2.663   0.03669     1.848   -1.878  7.248  5267.0  1.0035
... (other fixed effects and sigma) ...
Some R_hat are above 1.01, the MCMC chains have not converged!
```

**Custom likelihood (beta)** ([`examples/bayes/example_data_model_fit_beta.py`](../../examples/bayes/example_data_model_fit_beta.py)):

```python
from kanly.api import DataModel

data_string = """
self.x = `x`
self.y = `y`
"""

model_string = """
return nopython_logpdf_beta(x, a=$a$, b=$b$).sum()
"""

dataobj = DataModel.build_data_model(data_string, model_string, data, nopython=True)
model = dataobj.to_bayesian_model(bounds={'a': [0, np.inf], 'b': [0, np.inf]})
fit = model.amha([1., 1], n_samples=20_000, n_burnin=10_000, do_parallel=True, ...)
print(fit)
```

Example output (from the same file):

```text
MCMC Results
Method:             AMH
No. Chains:                     4    Samples:                20000
Maximum Log Posterior:     7.1045e+02

    mean      std       MCSE median [.05,    .95]     ESS   R_hat
a  4.961   0.1807   0.001934  4.958  4.673  5.265  8733.0  1.0002
b  2.051  0.06951  0.0007354  2.049  1.938  2.168  8935.0  1.0002
```

This is an example of using `DataModel` to do linear regression where our coefficient on `x` has a prior `normal(0,1)` and bounds `[0,10]`.  The data block loads `x`, `y`, and `wts`, and forms `root_weights` for weighted
residuals. The model block predicts `$Intercept$ + $x$ * x`, forms weighted Normal
log-likelihood residuals, and declares `sigma` with a positivity bound. Compiled
`DataModel` code is wrapped by `to_bayesian_model` into a sampler-ready
`BayesianModel`; the start vector `[1.0, 1.0, 1.0]` seeds intercept, slope, and
scale, and `sample` draws from the posterior over all three.

### Conjugate Bayesian Linear Regression

```python
import numpy as np
from kanly.api import blm


fit = blm(
    "y ~ x + z $ wts",
    data,
    Lambda0=np.diag([0, 3.3 * np.sum(data["wts"]), 0]),
    mu0=np.zeros(3),
    a0=0,
    b0=0,
)

print(fit)
```

`blm` parses `y ~ x + z $ wts`, places a conjugate prior on coefficients and
`sigma2` via `mu0`, `Lambda0`, `a0`, and `b0`, and returns analytical posterior
summaries (means, variances, credible intervals) without running a Markov chain.
This is the fast route when the conjugate assumptions match the problem; use
`bayes_lm_model` or `DataModel` when you need non-Gaussian likelihoods, custom
code blocks, or MCMC for non-conjugate priors.

## Important Behaviors / Pitfalls

- `kanly.bayes.__init__` is empty. Use `kanly.api` aliases or concrete
submodule imports.
- If bounds are supplied and `do_bounded_transform=True`, MCMC and optimization
operate in an unbounded transformed space. Posterior evaluation uses the
Jacobian adjustment so samples transformed back to the original scale target
the intended density.
- If `do_bounded_transform=False`, bounds are enforced as hard log-density
barriers.
- Regression wrappers reserve `__sigma2` for residual variance and constrain it
to be positive.
- MCMC starting values are assumed to be on the original parameter scale by
default. Use the `start_params_is_original_scale` arguments when working
directly with transformed coordinates.
- `MCMCResults.summary_df`, R-hat, ESS, Geweke diagnostics, and trace/interval
plots are post-sampling checks; they do not guarantee convergence by
themselves.
- Folders such as `___to_delete/`, `mcmc/__hmc_WIP/`, and files marked
`to_delete` or `WIP` are legacy or experimental code and should not be
treated as stable public API.
- Some old notebooks and scratch files may reference stale paths such as
`kanly.bayes_wip` or Bayesian HDI utilities under `kanly.bayes.utils`. The
current HDI helper is exposed from `kanly.utils.highest_density_interval` and
through `kanly.api.get_highest_density_interval`.

## References

### Metropolis–Hastings

Core accept/reject mechanism used by `amha(...)` and by coordinate-wise updates
when `do_mala=False`.

- Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., & Teller, E. (1953). [Equation of State Calculations by Fast Computing Machines](https://doi.org/10.1063/1.1699114). *The Journal of Chemical Physics*, 21(6), 1087–1092.
- Hastings, W. K. (1970). [Monte Carlo Sampling Methods Using Markov Chains and Their Applications](https://doi.org/10.1093/biomet/57.1.97). *Biometrika*, 57(1), 97–109.
- Wikipedia: [Metropolis–Hastings algorithm](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm)

### Metropolis-adjusted Langevin algorithm (MALA)

Coordinate-wise Langevin proposals with Metropolis correction in `mala(...)`;
optional warmup in `sample(...)`.

- Wikipedia: [Metropolis-adjusted Langevin algorithm](https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm)
- Besag, J. (1994). Comments on “Representations of knowledge in complex systems” by U. Grenander and M. I. Miller. *Journal of the Royal Statistical Society, Series B*, 56, 591–592. (Original MALA proposal; see also Wikipedia references there.)
- Roberts, G. O., & Tweedie, R. L. (1996). [Exponential Convergence of Langevin Distributions and Their Discrete Approximations](https://doi.org/10.2307/3318418). *Bernoulli*, 2(4), 341–363.
- Roberts, G. O., & Rosenthal, J. S. (1998). [Optimal Scaling of Discrete Approximations to Langevin Diffusions](https://doi.org/10.1111/1467-9868.00123). *Journal of the Royal Statistical Society, Series B*, 60(1), 255–268.

### Adaptive Metropolis–Hastings

Proposal-covariance adaptation in `amha(...)` and `sample(...)` (`do_adaptive`).

- Haario, H., Saksman, E., & Tamminen, J. (2001). [An Adaptive Metropolis Algorithm](https://doi.org/10.2307/3318737). *Bernoulli*, 7(2), 223–242. ([Project Euclid](https://projecteuclid.org/journals/bernoulli/volume-7/issue-2/An-adaptive-Metropolis-algorithm/bj/1080222083.full))
- Wikipedia (general MH context; no dedicated adaptive-Metropolis article): [Metropolis–Hastings algorithm](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm)

### Differential-evolution MCMC

Mixed into `amha(...)` and `sample(...)` via `do_diff_evolution_mc`; periodic
full-dimensional jumps in `mala(...)` via `diff_evolution_step_cadence`.

- ter Braak, C. J. F. (2006). [A Markov Chain Monte Carlo Version of the Genetic Algorithm Differential Evolution: Easy Bayesian Computing for Real Parameter Spaces](https://doi.org/10.1007/s11222-006-8769-1). *Statistics and Computing*, 16, 239–249.
- Wikipedia (genetic proposal mechanism): [Differential evolution](https://en.wikipedia.org/wiki/Differential_evolution)

## Where To See Examples

- [`examples/bayes/example_data_model.py`](../../examples/bayes/example_data_model.py), [`example_data_model_fit_beta.py`](../../examples/bayes/example_data_model_fit_beta.py)
- [`example_bayesian_linear_regression.py`](../../examples/bayes/example_bayesian_linear_regression.py), [`example_bayesian_linear_regression_non_conjugate.py`](../../examples/bayes/example_bayesian_linear_regression_non_conjugate.py), [`example_bayesian_linear_regression_mcmc.py`](../../examples/bayes/example_bayesian_linear_regression_mcmc.py)
- [`example_bayesian_linear_model_4_ways.py`](../../examples/bayes/example_bayesian_linear_model_4_ways.py), [`example_bayesian_linear_regression_model_conjugate.py`](../../examples/bayes/example_bayesian_linear_regression_model_conjugate.py)
- [`example_bayesian_logistic_model.py`](../../examples/bayes/example_bayesian_logistic_model.py), [`example_bayesian_elastic_net.py`](../../examples/bayes/example_bayesian_elastic_net.py)
- [`example_mala.py`](../../examples/bayes/example_mala.py), [`example_bayes_ols_mcmc.py`](../../examples/bayes/example_bayes_ols_mcmc.py)
- `examples/regression/example_ridge_different_ways.py`, `examples/regression/example_lasso_different_ways.py`
- `testing/bayes/testing_script_banana.py`

