# `kanly.optimize`

**See also:** [kanly README](../../README.md) · [automatic differentiation](../automatic_differentiation/README.md)

Bounded **numerical optimization** helpers used elsewhere in **kanly** (e.g. MAP steps, nonlinear solvers). This document covers only Python modules in the **`kanly/optimize/`** root (not files under `__TO_DELETE/`).

**Scope.** Functionality is intentionally **narrow** compared to
[`scipy.optimize`](https://docs.scipy.org/doc/scipy/reference/optimize.html):
constraints are **box bounds only** (`lb ≤ x ≤ ub`). There is no support for
general linear inequality constraints, nonlinear constraints, equality
constraints, or penalty/augmented-Lagrangian formulations. For those problems,
use SciPy (e.g. `minimize` with `method='SLSQP'` or `'trust-constr'`, or
`minimize_scalar`, global routines, root-finding, etc.).

There is also only **one** general-purpose smooth solver here—**`bfgs_pqn`**
(projected quasi-Newton). SciPy’s `minimize` exposes many algorithms (Nelder–Mead,
Powell, CG, BFGS, L-BFGS-B, TNC, COBYLA, SLSQP, trust-region, …). **`cdb`**
is a separate bounded **coordinate-descent** helper for simple black-box cases;
it is not a full alternative solver stack.

Background reading and citations are in [**References**](#references) at the end of this page.

---

## Public entry points (`kanly.api`)

| Symbol | Module | Role |
|--------|--------|------|
| `bfgs_pqn` | [`bfgs_bounded_quasi_newton.py`](bfgs_bounded_quasi_newton.py) | **Projected quasi-Newton** with BFGS Hessian updates (or optional analytic Hessian), box constraints, line search |
| `cdb` | [`coordinate_descent_bounded.py`](coordinate_descent_bounded.py) | **Bounded coordinate descent** on black-box objectives |

`func_str_to_callable` is re-exported on **`kanly.api`** but implemented under [`kanly/automatic_differentiation/`](../automatic_differentiation/); it is often paired with **`dict_2_array`** ([`kanly.utils.dict_2_array`](../utils/dict_2_array.py)) to turn named-parameter dicts into the `numpy` vector **`bfgs_pqn`** expects.

Import: `from kanly.api import bfgs_pqn, cdb` (and `func_str_to_callable` when using string objectives) or submodule paths under `kanly.optimize`.

---

## Example (in this repo)

[`examples/optimize/example_bfgs.py`](../../examples/optimize/example_bfgs.py) minimizes \((x-1)^2 + (y+2)^2\) by turning a short string into a callable, aligning initial values with **`param_names`**, then calling **`bfgs_pqn`**:

```python
from kanly.api import bfgs_pqn, func_str_to_callable
from kanly.utils.dict_2_array import dict_2_array

func = func_str_to_callable("({x}-1)**2 + ({y}+2)**2")
x0 = dict_2_array({"x": 0, "y": 2}, func.param_names)
result = bfgs_pqn(
    func,
    x0=x0,
    maxiter=10,
    maximize=False,
    ftol=1e-12,
    gtol=1e-12,
)
print(result.x, result.ferr, result.gnorm)
```

**Output** (from [`example_bfgs.py`](../../examples/optimize/example_bfgs.py)):

```text
result.x=array([ 1.00000001, -2.00000006])
result.ferr=8.248613298197706e-13
result.gnorm=1.198790486853639e-07
```

**Arbitrary callables and box constraints** (from the [root user guide](../../README.md#kanlyoptimize)):

```python
import numpy as np
from kanly.api import bfgs_pqn, cdb

def rosen(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

x0 = np.array([-1.0, 1.0])
bounds = np.array([[-2.0, -2.0], [2.0, 2.0]])

res_qn = bfgs_pqn(rosen, x0, bounds=bounds, maxiter=500)
res_cd = cdb(rosen, x0, bounds=bounds, maxiter=2000)
```

---

## `bfgs_pqn` — projected quasi-Newton

**Minimize or maximize** a scalar objective `fun(x)` over `x ∈ ℝ^p` with optional **box bounds** `lb ≤ x ≤ ub` (passed as a **2 × p** array: row 0 = lower, row 1 = upper).

The method follows the **projected quasi-Newton** paradigm for box-constrained smooth optimization: maintain a BFGS Hessian approximation, form a quasi-Newton step, restrict the search direction to coordinates that can move within the bounds, project trial steps back into the feasible box, and stop when the **projected gradient** is small. Conceptually, `bfgs_pqn` sits between projected-gradient methods and full projected-Newton methods in the sense of Kim et al. (2010), with Bertsekas (1982) as the classical projected-Newton reference on simple constraints.

- Builds a **BFGS** approximation to the Hessian (or uses `hessian_callable` if provided).
- Uses **finite-difference gradients** unless `gradient_callable` is passed.
- **Projects** line-search steps into the feasible box; tracks **binding** lower/upper constraints on the result (`lb_binding`, `ub_binding` on `BFGSPQNResults`).
- Supports `maximize=True` (internally minimizes `-fun`).

**Returns** [`BFGSPQNResults`](bfgs_bounded_quasi_newton.py) (extends [`OptimizationResult`](optimization_results.py)): `x`, `fun`, `grad`, `grad_projected`, `hess`, `converged`, `message`, `iter`, diagnostics (`ferr`, `xerr`, `gnorm`), etc.

**Notable options:** `xtol`, `ftol`, `gtol`, `maxiter`, `c1_wolfe`, `momentum`, optional `save_optimization_path`, `debug`.

**Helpers** in the same file (for reuse / testing): `grad_finite_diff`, `project`, `is_valid`, `get_direction`, etc.

---

## `cdb` — bounded coordinate descent

**Minimize or maximize** `func(x)` by **cycling coordinates**: for each index, approximate ∂f/∂xᵢ with **finite differences** clipped to the feasible interval, then take expanding steps along that coordinate until no improvement.

**Bounds** optional: `bounds` is **2 × p** (row 0 = `lb`, row 1 = `ub`), same convention as `bfgs_pqn`.

**Returns** [`CDBResults`](coordinate_descent_bounded.py), a subclass of `OptimizationResult` with coordinate-descent-specific metadata.

**Defaults:** `maxiter`, `xtol`, `ftol`, `gtol`, `debug`, progress bar cadence — see [`coordinate_descent_bounded.py`](coordinate_descent_bounded.py).

---

## Shared result type

[`optimization_results.py`](optimization_results.py) defines **`OptimizationResult`**: base container for final `x`, objective `fun`, gradients, projected gradient, convergence flag, message, timings, original callable, bounds, options. Subclasses add solver-specific fields (e.g. Hessian, binding masks).

---

## Utilities

[`utilities.py`](utilities.py):

- **`update_bfgs_hessian_approx`** — standard **BFGS** curvature update for a Hessian approximation.
- **`get_gradient_function`** — return a **finite-difference gradient** callable for an objective.

---

## Comparison: SciPy

| | SciPy (`scipy.optimize`) | kanly (`kanly.optimize`) |
|---|--------------------------|---------------------------|
| **Solver count** | Many methods via `minimize`, plus specialized entry points (`root`, `least_squares`, global optimizers, …) | **One** main smooth solver: **`bfgs_pqn`**. Plus **`cdb`** (coordinate descent), not a multi-method `minimize` API |
| **Constraints** | Box bounds, linear/nonlinear (in)equalities (method-dependent), etc. | **Box bounds only**—no `constraints=` list, no `LinearConstraint` / `NonlinearConstraint` |
| **Unconstrained** | `BFGS`, `CG`, `Nelder-Mead`, … | Use `bfgs_pqn` with `bounds=None` (still the same algorithm) |
| **Box-constrained smooth** | `L-BFGS-B`, `TNC`, `SLSQP`, `trust-constr`, … | **`bfgs_pqn`** only |
| **Coordinate descent** | No first-class `minimize` method | **`cdb`** |
| **Gradients** | `jac=` per method; finite differences in some cases | **`gradient_callable`** optional; else **`grad_finite_diff`** |
| **API shape** | `minimize(fun, x0, method=..., bounds=..., constraints=...)` | Callable-centric **`bfgs_pqn(fun, x0, bounds=...)`**—no `method=` dispatch |

**When to use kanly:** MAP/MLE steps inside kanly models, string/dict parameter wiring via `func_str_to_callable`, and box-constrained problems where you want this package’s projected quasi-Newton implementation.

**When to use SciPy:** Anything outside box constraints, algorithm choice, global optimization, least-squares with robust loss, sparse Jacobians, or production-grade breadth.

**kanly** ships a small, integrated subset of SciPy’s optimization surface—not a replacement for `scipy.optimize` as a whole.

---

## References

### Projected Newton and projected quasi-Newton with box constraints (`bfgs_pqn`)

- Bertsekas, D. P. (1982). [Projected Newton Methods for Optimization Problems with Simple Constraints](https://doi.org/10.1137/0320018). *SIAM Journal on Control and Optimization*, 20(2), 221–246. Introduces projected-Newton iterations \(x_{k+1} = [x_k - \alpha_k D_k \nabla f(x_k)]^+\) on the orthant (and extensions to linear constraints), with \(D_k\) built from second-order information; the reference **projected-Newton** anchor cited by Kim et al. (2010). (Title says *Newton*, not *quasi-Newton*, though \(D_k\) may be chosen in a quasi-Newton spirit.)
- Kim, D., Sra, S., & Dhillon, I. S. (2010). [Tackling Box-Constrained Optimization via a New Projected Quasi-Newton Approach](https://doi.org/10.1137/08073812X). *SIAM Journal on Scientific Computing*, 32(6), 3548–3563. Derives **projected quasi-Newton** (BFGS and L-BFGS variants) for \(\min f(x)\) s.t. \(l \le x \le u\), explicitly positioned between projected-gradient and Bertsekas’s projected-Newton methods; Armijo line search and free/active variable partitioning. [Preprint PDF](https://www.cs.utexas.edu/~inderjit/public_papers/pqnj_sisc10.pdf).

### Related quasi-Newton / box-constraint references

- Nocedal, J., & Wright, S. J. (2006). [*Numerical Optimization*](https://doi.org/10.1007/978-0-387-40065-5) (2nd ed.). Springer. Ch. 6 (quasi-Newton), Ch. 12 (constrained optimization), Ch. 3 (line search).
- Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995). [A Limited Memory Algorithm for Bound Constrained Optimization](https://doi.org/10.1007/BF01581334). *Mathematical Programming*, 66(1–3), 29–60. (L-BFGS-B; related bound-constrained quasi-Newton line.)

### General background

- [BFGS algorithm](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm)
- [Quasi-Newton method](https://en.wikipedia.org/wiki/Quasi-Newton_method)
- [Wolfe conditions](https://en.wikipedia.org/wiki/Wolfe_conditions)
- [Coordinate descent](https://en.wikipedia.org/wiki/Coordinate_descent) (`cdb`)
