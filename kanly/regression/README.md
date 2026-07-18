**See also:** [kanly README](../../README.md) · [formula](../formula/README.md)

# kanly.regression

## What this subpackage does

`kanly.regression` hosts almost all **parametric frequentist** estimators: linear, GLM, GMM, nonlinear least squares, robust and quantile linear models, elastic net, and partial least squares. They share [`kanly.formula`](../formula/README.md)-based sparse matrices, so adding high-cardinality factors or long formulas stays memory-efficient compared to always densifying.

**Useful background:** [`linear regression`](https://en.wikipedia.org/wiki/Linear_regression), [`generalized linear model`](https://en.wikipedia.org/wiki/Generalized_linear_model), [`instrumental variables`](https://en.wikipedia.org/wiki/Instrumental_variables_estimation), [`generalized method of moments`](https://en.wikipedia.org/wiki/Generalized_method_of_moments), [`nonlinear least squares`](https://en.wikipedia.org/wiki/Non-linear_least_squares).

## Comparison: SciPy / statsmodels / scikit-learn


| Area                       | SciPy                                                                                                                                                                                                                                                                     | statsmodels                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | scikit-learn                                                                | kanly                                                                                                                                                                                            |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **OLS / WLS / GLS / FGLS / GLSAR** | [`linalg.lstsq`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html), [`sparse.linalg.lsmr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsmr.html) — numeric `y`, `X` only; **no** formula DSL or inference layer | [`OLS`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html) / [`WLS`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.WLS.html) / [`GLS`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.GLS.html) (and `smf.`*); [`GLSAR`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.GLSAR.html) (experimental, **Cochrane–Orcutt** only); FGLS-style heteroskedasticity via iterative workflows | `LinearRegression` — prediction-focused; no classical inference summaries   | `lm` — WLS, GLS (`sigma`), `do_fgls=True` (FGLS); **`glsar` / `GLSAR`** for AR(`p`) errors (`full_information` for Prais–Winsten vs Cochrane–Orcutt); HC / cluster / HAC / bootstrap; **absorb**; sparse formulas |
| **IV (2SLS / W2SLS)**      | —                                                                                                                                                                                                                                                                         | Sandbox [`IV2SLS`](https://www.statsmodels.org/stable/generated/statsmodels.sandbox.regression.gmm.IV2SLS.html) — array API (`endog`, `exog`, `instrument`); not unified formula IV syntax                                                                                                                                                                                                                                                                                                                                                         | —                                                                           | `lm` with `y ~ x | z` formula syntax; 2SLS / W2SLS with HC / cluster / bootstrap SEs; `absorb=` for Frisch-Waugh IV                                                                              |
| **GLM**                    | — (assemble log-likelihood + [`optimize`](https://docs.scipy.org/doc/scipy/reference/optimize.html) yourself)                                                                                                                                                             | [`smf.glm`](https://www.statsmodels.org/stable/glm.html) — families, links, `summary()`; [`GLMGam`](https://www.statsmodels.org/stable/generated/statsmodels.gam.generalized_additive_model.GLMGam.html) for smooths                                                                                                                                                                                                                                                                                                                                                                                              | `LogisticRegression` etc. — **no** full GLM family objects (e.g. Gamma, NB) | `glm` — same covariance / IV / absorb stack as `lm` on **sparse** formula designs; **`gam`** for penalized B-spline GAMs ([README](generalized_linear_models/README.md#generalized-additive-models-gam)) |
| **Quantile**               | —                                                                                                                                                                                                                                                                         | [`QuantReg`](https://www.statsmodels.org/stable/generated/statsmodels.regression.quantile_regression.QuantReg.html) / [`smf.quantreg`](https://www.statsmodels.org/stable/examples/notebooks/generated/quantile_regression.html) (IRLS; formula or arrays)                                                                                                                                                                                                                                                                                         | —                                                                           | `qr` — sparse IRLS, smooth check loss; control-function IV path ([`kanly/regression/linear_models/quantile_regression/README.md`](kanly/regression/linear_models/quantile_regression/README.md)) |
| **Robust**                 | —                                                                                                                                                                                                                                                                         | [`RLM`](https://www.statsmodels.org/stable/generated/statsmodels.robust.robust_linear_model.RLM.html)                                                                                                                                                                                                                                                                                                                                                                                                                                              | `HuberRegressor` (different algorithmic focus)                              | `rlm` sparse IRLS                                                                                                                                                                                |
| **Penalized**              | — (use third-party or `optimize` with penalty by hand)                                                                                                                                                                                                                    | Limited built-ins                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | `ElasticNet`, `Lasso`, `Ridge` on dense `ndarray`                           | `elastic_net` on **sparse** formula designs, optional OLS refit                                                                                                                                  |
| **NLLS**                   | [`optimize.least_squares`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html) — you code residuals/Jacobians                                                                                                                         | [`bayesNonlinearLS`](https://www.statsmodels.org/stable/generated/statsmodels.miscmodels.nonlinls.NonlinearLS.html) in `statsmodels.miscmodels` (subclass `_predict`; array-oriented — **not** kanly-style formula NLLS)                                                                                                                                                                                                                                                                                                                           | —                                                                           | `nlls` / `nlls_en` with formula parsers, bounds, robust/quantile root losses                                                                                                                     |
| **GMM**                    | [`optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/optimize.html) only — **no** moment API                                                                                                                                                                  | [`sandwich`](https://www.statsmodels.org/stable/sandwich.html) ecosystem; sandbox [`IV2SLS`](https://www.statsmodels.org/stable/generated/statsmodels.sandbox.regression.gmm.IV2SLS.html) and related GMM helpers                                                                                                                                                                                                                                                                                                                                  | —                                                                           | `gmm`, `gmm_iv_linear`, `gmm_iv_nonlinear`, `gmm_mle`                                                                                                                                            |


## Syntax comparisons

The snippets below assume a `pandas` `DataFrame` `df` with columns such as `y`, `x`, `z`, `grp`, and `w` (weights). Kanly, statsmodels, sklearn, and SciPy differ mainly in **how you specify the model** (formula vs arrays vs residual callable) and in **built-in inference** (SE types, IV, sparse designs).

**OLS / WLS — `lm` vs statsmodels `ols` / `wls`**

```python
import statsmodels.formula.api as smf
from kanly.api import lm

# statsmodels — formula API
fit_sm = smf.ols('y ~ x + C(grp)', data=df).fit()
fit_sm_w = smf.wls('y ~ x + C(grp)', data=df, weights=df['w']).fit()

# kanly — same formula strings; optional robust / cluster / bootstrap SEs
fit = lm('y ~ x + C(grp)', df)
fit_w = lm('y ~ x + C(grp) $ w', df, cov_type='HC1')
```

**IV (2SLS) — `lm` vs sandbox `IV2SLS`**

```python
from patsy import dmatrices
import numpy as np
from statsmodels.sandbox.regression.gmm import IV2SLS
from kanly.api import lm

# statsmodels — IV2SLS takes numeric arrays only. Categorical `grp` must be expanded
# to dummies (e.g. with patsy) before fitting; exogenous controls belong in BOTH
# exog and instrument.
y, X = dmatrices('y ~ x + C(grp)', df, return_type='matrix')
_, Z = dmatrices('y ~ z + C(grp)', df, return_type='matrix')
y = np.squeeze(np.asarray(y))
X = np.asarray(X)
Z = np.asarray(Z)
fit_sm = IV2SLS(y, X, Z).fit()

# kanly — same model syntax as OLS; C(grp) handled in the formula parser
fit_iv = lm('y ~ x + C(grp) | z + C(grp)', df)
fit_iv_w = lm('y ~ x + C(grp) | z + C(grp) $ w', df)
fit_iv_absorb = lm('y ~ x | z', df, absorb='grp')
```

**GLM — `glm` vs statsmodels `glm`**

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf
from kanly.api import glm

# statsmodels
fit_sm = smf.glm(
    'y ~ poly(x, 2)',
    data=df,
    family=sm.families.Binomial(),
).fit()

# kanly — family as string; same formula grammar as lm (IV, weights, absorb, cov_type)
fit = glm('y ~ poly(x, 2)', df, family='binomial', cov_type='HC1')
fit_iv = glm('y ~ x | z', df, family='poisson')
```

**Penalized linear — `elastic_net` vs sklearn `ElasticNet`**

```python
import numpy as np
from sklearn.linear_model import ElasticNet
from kanly.api import elastic_net

# sklearn — dense design matrix X and response y (no formula DSL)
# X = np.column_stack([np.ones(n), df['x'].values, ...])  # build manually
fit_sk = ElasticNet(alpha=1e-4, l1_ratio=1.0, fit_intercept=False).fit(X, y)

# kanly — formula on DataFrame; sparse design; per-coefficient alpha dict
fit = elastic_net('y ~ x0 + x1 + x2 + ...', df, alpha=1e-4, l1_ratio=1.0)
fit = elastic_net('y ~ x + z $ w', df, alpha={'x': 3.0}, l1_ratio=1.0, refit=True)
```

**Nonlinear least squares — `nlls` vs `scipy.optimize.least_squares`**

```python
import numpy as np
from scipy.optimize import least_squares
from kanly.api import nlls

# scipy — residual vector; x0 is required
def residuals(params):
    intercept, beta, gamma = params
    return df['y'].to_numpy() - (intercept + beta * np.exp(gamma * df['x'].to_numpy()))

res = least_squares(residuals, x0=[1.0, 3.0, -0.5])

# kanly — {braces} name parameters; [brackets] name data columns; start_params is the
# analogue of x0 (dict keyed by name or vector in parameter order). If omitted, the
# solver starts from 0 and may run a subsample pre-fit (subsample=...) first.
fit = nlls(
    '[y] ~ {Intercept} + {beta} * exp({gamma} * [x])',
    df,
    start_params={'Intercept': 1.0, 'beta': 3.0, 'gamma': -0.5},
)
fit = nlls(
    '[y] ~ {Intercept} + {beta} * exp({gamma} * [x]) $ [w]',
    df,
    start_params=[1.0, 3.0, -0.5],
    max_iter=100,
)
```

The subsections below follow the `kanly/regression/` directory layout and include example code with printed summaries from the `examples/` tree.
## Submodule guides

| Path | Guide |
|------|--------|
| `linear_models/` | [README](linear_models/README.md) — OLS, WLS, IV, absorb, FGLS, ridge, SURE, Shapley R² (formula terms), permutation tests |
| `linear_models/robust/` | [README](linear_models/robust/README.md) — M-estimation (`rlm`) |
| `linear_models/quantile_regression/` | [README](linear_models/quantile_regression/README.md) — quantile regression (`qr`) |
| `linear_models/penalized/` | [README](linear_models/penalized/README.md) — elastic net / LASSO / ridge |
| `generalized_linear_models/` | [README](generalized_linear_models/README.md) — GLM (`glm`), GAM (`gam`) |
| `generalized_method_of_moments/` | [README](generalized_method_of_moments/README.md) — GMM |
| `nonlinear_least_squares/` | [README](nonlinear_least_squares/README.md) — NLLS (`nlls`, `nlls_en`) |
| `partial_least_squares/` | [README](partial_least_squares/README.md) — PLS (`pls1`, `PLS1`, `PLS2`) |

The [root user guide](../../README.md#kanlyregression) contains the same regression-wide material plus detailed subsections for each module (quick starts, example output, and example script lists).
