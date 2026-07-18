# Generalized Method Of Moments

**See also:** [kanly README](../../../README.md) · [regression overview](../README.md)

`kanly.regression.generalized_method_of_moments` estimates models by choosing
parameters that make sample moment conditions close to zero. It supports raw
moment callables, formula-defined moments, linear IV-GMM, nonlinear IV-GMM, and
MLE score equations expressed as GMM moments.

For background, see Wikipedia on the
[Generalized method of moments](https://en.wikipedia.org/wiki/Generalized_method_of_moments),
[Instrumental variables estimation](https://en.wikipedia.org/wiki/Instrumental_variables_estimation),
and [Maximum likelihood estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation).

## Core Idea

GMM starts from moment restrictions:

```text
E[g_i(theta)] = 0
```

The estimator minimizes the weighted quadratic form:

```text
gbar(theta)' W gbar(theta)
```

where `gbar(theta)` is the sample average of observation-level moments and `W`
is a weighting matrix. When there are more moments than parameters, the system
is over-identified and `kanly` can update `W` using the estimated covariance of
the moments.

## Entry Points

The common APIs are exported through `kanly.api`:

- `gmm`: fit formula-defined moment conditions.
- `GMM`: fit from a raw Python moment function.
- `gmm_iv_linear`: fit linear instrumental-variable GMM from `y ~ x | z`.
- `gmm_iv_nonlinear`: fit nonlinear IV-GMM from a residual formula and
  instrument formula.
- `gmm_mle`: fit likelihood score equations as GMM moments.

The implementation lives in:

- `model.py`: `SparseGeneralizedMethodOfMomentsModel` and public builders.
- `gmm_internal.py`: trust-region optimization and one-step/two-step/iterative
  weighting updates.
- `gmm_variance_covariance.py`: moment covariance and sandwich covariance.
- `regression_results.py`: `GMMRegressionResults` summary object.
- `constants.py`: defaults for methods, covariance types, and tolerances.

## Moment Formula Syntax

Formula GMM uses the **same string grammar as nonlinear least squares (NLLS)
prediction formulas**: data in **`[...]`**, parameters in **`{...}`**, optional
`$ [w]` weights, and the same patsy-style tokens inside brackets (`poly`,
`C(...)`, `I(...)`, etc.). Each fragment is compiled with
`build_prediction_function_from_formula` (see
[`kanly/regression/nonlinear_least_squares/README.md`](../nonlinear_least_squares/README.md#formula-syntax)).
There is no `y ~ ...` response line—each string is a scalar expression evaluated
per observation.

Pass `gmm` a **list of moment specifications** (one GMM moment condition per
entry). GMM targets `E[g_k(theta)] = 0` for each `k`.

| List entry | Meaning |
| ---------- | ------- |
| **String** | One formula fragment; treated as a single factor (equivalent to a one-element tuple). |
| **Tuple of strings** | **Element-wise product** of the compiled fragments, observation by observation. |

For a tuple `(f1, f2, …, fK)`, the `k`-th moment column is
`f1(theta) * f2(theta) * … * fK(theta)` at each row (the same multiplication
pattern used when building moments in `build_model_from_formulas`).

### Example: residual × instrument (IV orthogonality)

```python
("[y] - {a} - {b}*[x]", "[x]")
```

defines one moment whose sample average should be zero:

```text
(1/n) * sum_i (y_i - a - b*x_i) * x_i  =  0
```

i.e. **`x` is orthogonal to the residual** `y - (a + b*x)` in the GMM sense.
That is the standard moment for exogeneity of `x` when the mean equation is
linear in `(a, b)`.

A **string alone** is the same as a one-factor tuple, e.g.
`"[y] - {a} - {b}*[x]"` imposes `E[y - a - b*x] = 0` (mean residual zero).

### Example: linear IV with multiple instruments

```python
from kanly.api import gmm

fit = gmm(
    [
        "[y] - ({Intercept} + {x}*[x] + {z1}*[z1])",
        ("[y] - ({Intercept} + {x}*[x] + {z1}*[z1])", "[z1]"),
        ("[y] - ({Intercept} + {x}*[x] + {z1}*[z1])", "[z2]"),
    ],
    data,
    specification_name="GMM IV",
)

print(fit)
```

Here the first entry is the mean-residual moment; the tuple entries multiply the
same residual formula by each instrument column, giving residual–instrument
orthogonality conditions `E[z_j * (y - y_hat(theta))] = 0`.

## Linear IV-GMM

Use `gmm_iv_linear` for the common linear instrumental variables case:

```python
from kanly.api import gmm_iv_linear, lm, compare_results


fit_gmm = gmm_iv_linear(
    "y ~ x | z",
    data,
    specification_name="GMM IV",
    cov_type="SANDWICH",
)

fit_iv = lm("y ~ x | z", data, cov_type="NONROBUST")
print(compare_results([fit_gmm, fit_iv]))
```

The moment conditions are:

```text
E[z_i * (y_i - x_i' beta)] = 0
```

Set `do_2sls=True` to use the 2SLS weighting matrix and force one-step GMM.

## Nonlinear IV-GMM

Use `gmm_iv_nonlinear` when the residual is nonlinear in parameters but moments
still take the instrument-residual form:

```python
from kanly.api import gmm_iv_nonlinear


fit = gmm_iv_nonlinear(
    "[y] - np.exp({Intercept} + {x}*[x] + {z1}*[z1])",
    "z1 + z2",
    data,
    method="ITERATIVE",
    cov_type="BOOTSTRAP",
    cov_kwds={"n_samples": 100},
)
```

The residual formula is parsed through the nonlinear least-squares formula
machinery, and the instrument formula is converted to a sparse design matrix.

## MLE As GMM

`gmm_mle` treats score equations as moment conditions. Pass a likelihood or
log-likelihood contribution formula; when `is_log_llf=True`, the formula is
already interpreted as log-likelihood.

```python
from kanly.api import gmm_mle, glm, compare_results


fit_gmm = gmm_mle(
    "[y]*np.log({Intercept}+{x}*[x]) + "
    "(1-[y])*np.log(1-{Intercept}-{x}*[x])",
    data,
    is_log_llf=True,
    start_params=[0.5, 0.0],
)

fit_glm = glm("y ~ x", data, family="binomial", link="identity", start_params=[0.5, 0.0])
print(compare_results([fit_gmm, fit_glm]))
```

Internally, the model differentiates the log-likelihood contribution with
respect to each parameter and uses those score contributions as moments.

## Raw Moment Function API

Use `GMM` when you already have a callable that returns observation-level
moments:

```python
import numpy as np
from kanly.api import GMM


def moment_func(theta):
    resid = y - theta[0] - theta[1] * x
    return np.column_stack([resid, resid * z])


fit = GMM(
    moment_func,
    nobs=len(y),
    num_moments=2,
    num_params=2,
    param_names=["Intercept", "x"],
)
```

The moment function should return one row per observation and one column per
moment.

## Methods And Covariance

Supported GMM methods:

- `ONE_STEP`: fit once using the supplied or identity weighting matrix.
- `TWO_STEP`: fit once, estimate the moment covariance, update `W`, and refit.
- `ITERATIVE`: repeatedly update `W` and refit until outer-loop parameter
  changes are small or the maximum number of outer iterations is reached.

Supported covariance types:

- `SANDWICH`: standard GMM sandwich covariance.
- `CLUSTER`: cluster-robust covariance using `cov_kwds` group information.
- `BOOTSTRAP`: Bayesian bootstrap refits. Common keywords include `n_samples`,
  `seed`, `method`, and `max_processes`.

Useful fit options include:

- `start_params`: starting values for optimization.
- `W`: initial weighting matrix.
- `max_iter`, `xtol`, `ftol`, `gtol`, `Delta`: trust-region optimizer controls.
- `iterative_gmm_max_iter`, `iterative_gmm_x_tol`: outer-loop controls for
  iterative GMM.
- `index`: optional row subset for formula-based models.
- `debug=True`: print model-building and optimizer progress.

## Results

All public fit helpers return `GMMRegressionResults`. The result object stores:

- `params`: estimated parameters.
- `cov_params`: parameter covariance matrix when available.
- `avg_moment_vals`: final average moment values.
- `W`: final weighting matrix.
- `Omega`: estimated moment covariance matrix.
- `method`, `cov_type`, `converged`, `message`, and `n_iters`.
- Numerical diagnostics for `G'WG`, including eigenvalues and condition number.

Printed summaries include parameter inference, moment formulas, final moment
values, convergence messages, and conditioning warnings.

## Examples

See `examples/generalized_method_of_moments/`:

- `example_gmm_linear.py`: linear IV-GMM via `gmm_iv_linear`.
- `example_gmm_linear_instrumental_variables.py`: explicit moment-list IV-GMM.
- `example_gmm_logit.py`: nonlinear probability moments compared with GLM.
- `example_gmm_nonlinear.py`: over-identified nonlinear GMM with iterative
  weighting and bootstrap covariance.
- `example_gmm_mle.py`: MLE score equations as GMM moments.
- `example_gmm_mle_chi_squared.py`: chi-squared likelihood example.
