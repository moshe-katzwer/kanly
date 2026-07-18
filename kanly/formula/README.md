# kanly Formula Package

`kanly.formula` provides sparse, Patsy-like formula parsing and matrix construction.
It is the low-level engine behind high-level APIs such as `kanly.api.lm`,
`kanly.api.glm`, `kanly.api.qr`, and helpers like `kanly.api.sparse_dmatrix`.

## What It Does

- Parse formula strings with familiar separators:
  - `~` for left/right sides (`y ~ x1 + x2`)
  - `|` for IV instruments (`y ~ x | z`)
  - `$` for weights (`y ~ x $ w`)
- Build sparse design matrices with term expansion.
- Track and remove invalid/null rows consistently across endog/exog/instruments/weights.
- Support categorical interactions and numerical transforms in a sparse workflow.

## Core Entry Points

- `kanly.api.sparse_dmatrix(formula_rhs, data)`
- `kanly.api.sparse_dmatrices(full_formula, data)`
- `kanly.formula.data_getter.SparseDataGetter.get_data(data, formula, ...)`

## Formula DSL Notes

Supported/common constructs include:

- Intercept control: `-1` (drop intercept)
- Categorical coding: `C(col)`
- Quoted column names: `Q("column with spaces")`
- Arithmetic wrapper: `I(...)`
- Polynomial shorthand: `poly(x, 3)`, `poly(x, [1,3,5])`, `poly(x, 3, -1)`
- Lags/leads: `L(x, 1)`, `L(x, [1,2,4])`
  * grouped lags `L(x, 1, grp)`
- [B-Splines](https://en.wikipedia.org/wiki/B-spline): `bs(x,degree=..,df=..,include_intercept=..)` 
  * (see [patsy](https://patsy.readthedocs.io/en/latest/spline-regression.html))
- Demeaning: `center(x)` or weighted `center(x, w)`
- Standardizing: `standardize(x)` or `standardize(x, w)` (for weights `w`, demeans and divides by standard deviation).
- Trends: `trend([1,3])` for `t + t**3`
  * (assumes ordered data)
- Seasonal fixed effects: `seasonality(12)` will generate 12 fixed effects
  * (assumes ordered data)

These transforms are "stateful" and are safe for predicting on other data sets - that is if we demean, the model remembers the mean.

## Minimal Usage

### Build RHS matrix only

```python
from kanly.api import sparse_dmatrix

X_obj = sparse_dmatrix('x*C(g) + L(x) + I(x**2) + poly(z, 3) + DM(w)', df, debug=True)
X = X_obj.values
cols = X_obj.column_names
null_rows = X_obj.null_rows
```

### Build both y and X

```python
from kanly.api import sparse_dmatrices

y_obj, X_obj = sparse_dmatrices('y ~ x1 + C(city) + poly(x2, 2)', df)
```

### Full data bundle (IV/weights/absorb aware)

```python
from kanly.formula.data_getter import SparseDataGetter

bundle = SparseDataGetter.get_data(
    df,
    'y ~ x + C(grp) | z + C(grp) $ w',
    absorb=None,
    check_constant_cols=True,
)
```

Returned `bundle` contains keys such as `ENDOG`, `EXOG`, `INSTRUMENTS`,
`WEIGHTS`, `VALID_OBS_ROWS`, and `NULL_ROWS_INFO_DICT`.

## Important Behaviors / Pitfalls

- At most one `~`, one `|`, and one `$` are allowed in a formula.
- Null/invalid rows are unioned across all required blocks, then all blocks are row-sliced consistently.
- If using absorb/fixed effects, specifying `-1` in exog is not allowed.
- `sparse_dmatrix` is RHS-oriented; for full `y ~ X`, use `sparse_dmatrices` or model APIs.
- Expanded lag syntax like `L(x, range(1,3))` is rewritten to explicit lag terms.

## Where To See Examples

- `examples/formula/example_sparse_dmatrix.py`
- `examples/example_formula.py`
- Formula usage through model APIs:
  - `examples/regression/linear_models/...`
  - `examples/regression/generalized_linear_models/...`
  - `examples/regression/linear_models/quantile_regression/...`

