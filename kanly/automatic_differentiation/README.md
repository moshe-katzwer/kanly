# `kanly.automatic_differentiation`

**See also:** [kanly README](../../README.md) · [optimize](../optimize/README.md)

**String-defined objectives** and **symbolic automatic differentiation** on a restricted grammar: parse a formula into an expression tree, apply chain/product rules, then **`exec`** generated Numba-annotated Python for Jacobians, gradients, Hessians, and partial derivatives.

**Public API on `kanly.api`:** [`func_str_to_callable`](function_callable.py) — the main entry point for users.

**Useful background:** [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation), [computational graph](https://en.wikipedia.org/wiki/Dataflow_programming).

### Quick usage

```python
from kanly.api import func_str_to_callable

f = func_str_to_callable("({a}-1)**2 + ({b}+2)**2")
partials, info = f.get_analytical_partial_derivatives(return_info=True)
```

For optimization with named dict initial values, pair with `dict_2_array` (see [`examples/optimize/example_bfgs.py`](../../examples/optimize/example_bfgs.py)). For delimiter customization and `return_info` inspection, see [`examples/autodiff/example_autodiff.py`](../../examples/autodiff/example_autodiff.py).

---

## Modules (this folder)

| File | Role |
|------|------|
| [`function_callable.py`](function_callable.py) | `func_str_to_callable`, [`FunctionCallable`](function_callable.py) wrapper, `get_param_names` |
| [`graph.py`](graph.py) | [`AutoDiffGraphNode`](graph.py) expression tree; `build_jacobian_from_string`, `build_partial_derivative_from_string` |
| [`elementary_functions.py`](elementary_functions.py) | Numba `expit` / `logit`; [`DERIV_FUNC_NAME_DICT`](elementary_functions.py) mapping function names → derivative source strings |
| [`function_wrapper.py`](function_wrapper.py) | [`FunctionWrapper`](function_wrapper.py) — combine callables with `+`, `*`, etc. (runtime composition, not string AD) |

`__init__.py` is intentionally minimal; import from `kanly.api` or submodule paths as needed.

---

## `func_str_to_callable`

1. **Parameter placeholders** in the string use delimiter pairs (default `{` `}`); [`get_param_names`](function_callable.py) rewrites them to `params[0]`, `params[1]`, … and records ordered `param_names`.
2. A Python function body is generated (`params = np.asarray(params); return <expr>`), optionally decorated with `@jit` when `nopython=True`.
3. The result is a **`FunctionCallable`** (`DillObject`): callable like any objective, plus `get_analytical_*` methods that delegate to [`AutoDiffGraphNode`](graph.py).

**Extra arguments:** pass names in `other_args` (comma-separated) for symbols that are **not** parameters (e.g. covariates `x`, `y`); they appear in the generated signature after `params`.

**Custom delimiters:** e.g. `param_delimiters=('$', '$')` for `$alpha$`-style names (see [`examples/autodiff/example_autodiff.py`](../../examples/autodiff/example_autodiff.py)).

---

## `FunctionCallable` — analytical derivatives

When `func_str` is set, the lazy **`auto_diff_node`** parses the expression. Useful methods:

- **`get_analytical_partial_derivative(arg_num, ...)`** — ∂f/∂param for one index.
- **`get_analytical_partial_derivatives(...)`** — all partials; optional `return_info` with expression strings and generated code.
- **`get_analytical_gradient` / `get_analytical_hessian`** — aggregated derivatives when `nobs > 1` require `agg_func` in `('mean', 'mean_squared')` (scalar summaries over observations).
- **Arithmetic on `FunctionCallable`** (`+`, `*`, …) rebuilds a composed string and calls `func_str_to_callable` again (unified parameter naming required).

**Frozen helpers:** `get_frozen_*` fix extra `*args` / `**kwargs` so the returned callable is only a function of `params`.

---

## `AutoDiffGraphNode` and low-level builders

- Parses a **single** expression string into a tree: operators (`+`, `-`, `*`, `/`, `^` tokens internally), `params[k]` identifiers, literals, and a fixed set of **functions** whose derivatives exist in `DERIV_FUNC_NAME_DICT` (e.g. `np.log`, `exp`, `sin`, …).
- **`get_analytical_jacobian`** builds a function mapping `params` → dense `(..., num_params)` or **CSC sparse** Jacobian when below a memory threshold / many zeros expected.
- **`build_jacobian_from_string`** / **`build_partial_derivative_from_string`** are thin wrappers constructing a node and calling the corresponding `get_*` method.

**Limitations:** Only operations the parser and derivative table understand are supported; unsupported syntax or functions raise during parse or differentiation. Vector-valued objectives with `nobs > 1` need care for Hessian/gradient aggregation.

---

## `FunctionWrapper`

Wraps a `callable` or scalar so **`+`, `-`, `*`, `/`, `**`** produce new callables without touching the string-based AD pipeline. Useful for composing objectives at runtime when you do not need symbolic derivatives from this package.

---

## Comparison: JAX / PyTorch / SciPy

| | JAX / PyTorch | SciPy | kanly |
|--|----------------|-------|--------|
| **Mechanism** | Trace / autograd on composed ops | Finite differences or user-supplied `jac` | **Symbolic** string → Python source → `exec` (+ optional Numba) |
| **Scope** | Broad differentiable ops | General `minimize` interfaces | **Curated** unary/binary numpy-like ops + grammar |
| **Workflow** | Define function in Python | Closed-form or black-box | **Formula strings** with `{param}` placeholders |

---

## Examples in this repo

- [`examples/autodiff/example_autodiff.py`](../../examples/autodiff/example_autodiff.py) — partial derivatives, custom `param_delimiters`.
- [`examples/optimize/example_bfgs.py`](../../examples/optimize/example_bfgs.py) — `func_str_to_callable` + `dict_2_array` + `bfgs_pqn`.
