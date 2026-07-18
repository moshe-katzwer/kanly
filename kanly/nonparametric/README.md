# kanly Nonparametric Package

**See also:** [kanly README](../../README.md)

`kanly.nonparametric` implements **smoothing and density estimation** without a full parametric likelihood: [kernel density estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation) (FFT or direct), [LOWESS / LOESS](https://en.wikipedia.org/wiki/Local_regression), STL / MSTL-style seasonal smoothing, Gaussian kernel regression, and **piecewise** splines / interpolators with analytic derivatives.

`kanly.nonparametric` provides a set of nonparametric smoothing and
density-estimation tools for one-dimensional data.  All public entry points
are re-exported through `kanly.api`.

---

## Comparison: SciPy / statsmodels / scikit-learn

| Method | SciPy | statsmodels | scikit-learn | kanly |
| ------ | ----- | ----------- | ------------ | ----- |
| KDE | [`gaussian_kde`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html) (Gaussian kernel; Silverman’s rule default) | `statsmodels.nonparametric.KDEUnivariate`, `KDEMultivariate` | `KernelDensity` (multiple kernels via `kernel=`) | `kde` with FFT path, explicit kernel name strings, power-of-two grid constraints |
| LOWESS | — (not in SciPy) | `statsmodels.nonparametric.lowess` | — | `LOWESS` / `lowess` with Numba path |
| Splines / interpolation | [`interpolate`](https://docs.scipy.org/doc/scipy/reference/interpolate.html) (`CubicSpline`, `UnivariateSpline`, `interp1d`, …) | Some formula / smoothing support | Limited vs general `scipy.interpolate` | `cubic_spline`, `linear_spline`, `interp` with **derivative** hooks on returned objects |
| STL / seasonal split | — (use `signal` or third-party for ad hoc filters) | Seasonal decomposition utilities in `tsa` (API differs from kanly STL) | — | `stl`, `mstl`; LOESS-driven STL workflow |

---

## What it does

| Module | Functionality |
|--------|---------------|
| `lowess.py` | Locally-Weighted Scatterplot Smoothing (LOWESS / LOESS) with adaptive Metropolis-Hastings and Numba acceleration |
| `kde.py` | Kernel Density Estimation — FFT and direct backends, eight kernels, optional boundary clipping |
| `stl.py` | Single-seasonality STL (trend / seasonality / residual) via LOESS backfitting; `plot_stl` |
| `mstl.py` | Multiple-seasonality MSTL (`period` as int or list); `plot_mstl` |
| `gaussian_kernel_smooth.py` | Gaussian kernel regression — FFT and direct variants, returns arrays or a fitted interpolator |
| `interpolate.py` | Piecewise polynomial interpolation (cubic, quadratic, linear, nearest/previous/next step) with derivative access |

---

## Core entry points via `kanly.api`

```python
from kanly.api import (
    # LOWESS
    lowess,          # functional interface
    LOWESS,          # object-oriented / functional (returns x, y arrays)

    # KDE
    kde,

    # STL / MSTL decomposition
    stl,
    mstl,
    plot_stl,
    plot_mstl,

    # Gaussian kernel smoothing
    gaussian_kernel_smooth,         # unified entry point (FFT or direct)
    gaussian_kernel_smooth_fft,     # FFT backend directly
    gaussian_kernel_smooth_direct,  # direct backend directly

    # Interpolation
    interp,           # generic factory (alias: interp1d)
    cubic_spline,
    quadratic_spline,
    linear_spline,
)
```

---

## Kernel Density Estimation

```python
import numpy as np
from kanly.api import kde

rng = np.random.default_rng(0)
data = rng.normal(size=500)

# Default: Gaussian kernel, normal-reference bandwidth, FFT path
support, density = kde(data)

# Epanechnikov kernel, Scott's rule
support, density = kde(data, kernel='epa', bw='scott')

# Return a callable KDEObject instead of arrays
f = kde(data, return_arrays=False)
print(f(0.0))   # evaluate density at x = 0

# Clip density to [−2, 2] (reflects tail mass back inside support)
support, density = kde(data, clip=(-2.0, 2.0))
```

Available kernel names: `'gau'`, `'epa'`, `'uni'`, `'tri'`, `'biw'`,
`'triw'`, `'cos'`, `'tric'`.

Available bandwidth rules: `'normal_reference'` (default), `'scott'`,
`'silverman'`.  A positive `float` can also be passed directly.

**Important**: when `fft=True` (default), `gridsize` must be a power of
two (default 256).

---

## LOWESS

```python
import numpy as np
from kanly.api import LOWESS

rng = np.random.default_rng(1)
x = np.linspace(0, 10, 200)
y = np.sin(x) + 0.3 * rng.standard_normal(200)

# frac controls the smoothing window (number of neighbours, not fraction)
x_smooth, y_smooth = LOWESS(y, x, frac=0.3, it=1, degree=1)
```

`lowess` is a functional alias; `LOWESS` is the primary entry point.

See `lowess_example.py` at the project root and
`testing/testing_script_lowess.py` for worked examples.

---

## STL and MSTL decomposition

Both **`stl`** and **`mstl`** decompose a regularly spaced series into **trend**, **seasonality**, and **residual** using alternating LOESS/LOWESS backfitting with optional robust bisquare re-weighting. **`mstl`** generalizes **`stl`**: a scalar `period` behaves like classical STL; a **list of periods** fits **multiple seasonal components** (e.g. weekly and yearly cycles on daily data).

Companion plotters **`plot_stl`** and **`plot_mstl`** build a stacked matplotlib figure (observed vs fitted, trend, each seasonality panel, residuals) and return the `Figure` (pass `show=True` to display).

### Single seasonality (`stl`)

```python
import numpy as np
from kanly.api import stl, plot_stl

rng = np.random.default_rng(2)
n = 120
t = np.arange(n)
y = 0.05 * t + np.sin(2 * np.pi * t / 12) + 0.2 * rng.standard_normal(n)

trend, seasonality, resid = stl(y, period=12)
# y ≈ trend + seasonality + resid

plot_stl(y, trend, seasonality, resid, show=True, title='Monthly STL')
```

### Multiple seasonalities (`mstl`)

```python
import numpy as np
from kanly.api import mstl, plot_mstl

n = 365 * 2
t = np.arange(n)
y = (np.sin(2 * np.pi * t / 7)
     + 0.5 * np.sin(2 * np.pi * t / 365)
     + 0.01 * t
     + 0.2 * np.random.randn(n))

trend, seasonalities, resid = mstl(y, period=[7, 365])
weekly, yearly = seasonalities  # list order matches your input periods

plot_mstl(y, trend, seasonalities, resid, show=True,
          period_labels=['Weekly (7)', 'Yearly (365)'])
```

With a single int, **`mstl(y, period=12)`** returns one seasonality **array** (same shape as **`stl`**). With a list, seasonality is a **list of arrays** in the order you passed `period` (internally sorted smallest→largest for estimation, then restored).

### Key parameters (`stl` and `mstl`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `period` | — | **`stl`:** int (e.g. 12 for monthly). **`mstl`:** int or list of ints (e.g. `[7, 365]`) |
| `swindow` | `np.inf` | Seasons on each side when smoothing each seasonal sub-series (`np.inf` = global median per phase). **`mstl`:** scalar or list (one per period) |
| `twindow` | `int(1.5 * period)` (odd) | LOWESS window length for trend; capped at `len(endog)`. **`mstl`:** uses `max(period)` when `period` is a list |
| `robust` | `True` | Outer bisquare re-weighting pass on residuals |
| `l_iterations` | `3` | Inner backfitting iterations (trend ↔ seasonality) per seasonal pass |
| `mstl_iterations` | `2` | (**`mstl` only**) Outer passes over the period list when `len(period) > 1` |

Plotting kwargs (both `plot_stl` and `plot_mstl`): `show`, `figsize`, `dpi`, `title`; **`plot_mstl`** also accepts `period_labels` for seasonality panel titles.

---

## Gaussian Kernel Smoothing

```python
import numpy as np
from kanly.api import gaussian_kernel_smooth

rng = np.random.default_rng(3)
x = rng.uniform(0, 10, 300)
y = np.sin(x) + 0.4 * rng.standard_normal(300)

# FFT path (default) — n_grid must be a power of two
x_grid, y_smooth = gaussian_kernel_smooth(x, y, n_grid=128, bandwidth=0.5)

# Direct path — slower but works for any n_grid
x_grid, y_smooth = gaussian_kernel_smooth(x, y, do_fft=False, n_grid=100)

# Return a fitted interpolator rather than arrays
f = gaussian_kernel_smooth(x, y, return_arrays=False, kind='cubic')
print(f(5.0))
```

The `adjust` parameter multiplies the bandwidth:
```python
# More smoothing
x_grid, y_smooth = gaussian_kernel_smooth(x, y, adjust=2.0)
```

---

## Interpolation

```python
import numpy as np
from kanly.api import cubic_spline, linear_spline, interp

x = np.array([0.0, 1.0, 2.5, 4.0, 5.0])
y = np.array([0.0, 1.0, 0.5, 2.0, 1.5])

# Cubic spline (default not-a-knot boundary conditions)
f = cubic_spline(x, y)
print(f(np.linspace(0, 5, 50)))   # evaluate
print(f.derivative(2.0))          # first derivative at x=2
print(f.derivative2(2.0))         # second derivative at x=2

# Natural or clamped boundary conditions
f_nat = cubic_spline(x, y, bc_type='natural')
f_clamp = cubic_spline(x, y, bc_type='clamped', clamped_slopes=(0.0, 0.0))

# Linear spline
g = linear_spline(x, y)

# Generic factory — 'nearest', 'previous', 'next' step interpolants
h = interp(x, y, kind='nearest')

# Print a symbolic summary of all segments
print(f)
```

The returned `Interpolator1d` object exposes:

- `f(x)` — evaluate
- `f.derivative(x)` — first derivative
- `f.derivative2(x)` — second derivative
- `f.derivative3(x)` — third derivative (always zero for quadratic/linear/step)

---

## Important behaviors and pitfalls

1. **FFT grid must be a power of two.**  Both `kde` (when `fft=True`) and
   `gaussian_kernel_smooth` (when `do_fft=True`) require `gridsize` / `n_grid`
   to be a power of two.  The default values (256 and 128 respectively) satisfy
   this.  Passing an arbitrary integer will raise an exception.

2. **Bandwidth adjustment.**  The `adjust` (or `adjust`) parameter in `kde`
   and `gaussian_kernel_smooth` is a scalar multiplier on top of the
   automatically selected bandwidth.  Values above 1 produce more smoothing;
   values below 1 produce less.

3. **LOWESS `do_njit` flag.**  The `lowess` / `LOWESS` functions use Numba
   JIT compilation by default.  The first call will trigger JIT compilation
   (one-time cost).  Pass `do_njit=False` to fall back to pure Python if
   Numba is unavailable.

4. **`stl` / `mstl` assume no missing observations.**  Both require a
   regularly spaced, complete time series.  Gaps will distort seasonal
   estimates.  Plotting requires **matplotlib** (imported inside `plot_stl` /
   `plot_mstl`).

5. **`interp` sorts by default.**  If `x` is not already sorted,
   `interp` / `cubic_spline` etc. sort it internally.  Pass
   `assume_sorted=True` to skip the sort for a small speed gain.

---

## KDE quick reference

```python
import numpy as np
from kanly.api import kde

support, density = kde(data)
support, density = kde(data, kernel='epa', bw='scott')
f = kde(data, return_arrays=False)
print(f(0.0))
```

Kernels include `'gau'`, `'epa'`, `'uni'`, `'tri'`, `'biw'`, `'triw'`, `'cos'`, `'tric'`. The default FFT path requires `gridsize` to be a power of two.

## Where to see examples

- No dedicated `examples/nonparametric/` tree for all modules; see also [`lowess_example.py`](../../lowess_example.py) and [`testing/testing_script_lowess.py`](../../testing/testing_script_lowess.py) (referenced in the [root user guide](../../README.md#kanlynonparametric)).
- **LOWESS**: `lowess_example.py` (project root), `testing/testing_script_lowess.py`
- **STL**: [`examples/nonparametric/example_stl.py`](../../examples/nonparametric/example_stl.py)
- **MSTL**: [`examples/nonparametric/example_mstl.py`](../../examples/nonparametric/example_mstl.py)
- All entry points are exercised by the test suite in `tests/`.
