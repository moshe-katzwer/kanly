# `kanly.plot`

**See also:** [kanly README](../../README.md) · [bayes](../bayes/README.md) (MCMC chain mixing plots)

**Terminal-friendly ASCII graphics** for quick inspection of `numpy` data: **line plots** (bucketed means), **scatter / density** grids, and **histograms** (single or multi-panel). No matplotlib dependency; functions return multi-line strings and optionally `print` them.  

(This code is used by the `bayes` package MCMC routine to show chain mixing.)

**Public API (`kanly.api`):** `plot`, `scatter`, `hist` — implemented in [`ascii_plotlib.py`](ascii_plotlib.py).


## `plot(y, …)`

- Drops non-finite values, truncates `y` so `len(y)` is a multiple of `ncols`, reshapes to `(ncols, n)` buckets.
- Each **column** shows the **mean** of that index bucket; vertical position is discretized to `nrows` text rows with `●`.
- Optional **`coverage`** in `(0, 1]`: per-bucket quantile band; band edges drawn with `┬` / `┴`.
- Optional **`xlabel`**, **`ylabel`**: short captions under the frame.
- Returns a `str`; set **`do_print=True`** to emit to stdout.

---

## `scatter(x, y, …)`

- Maps `(x, y)` to a `nrows × ncols` count raster (linear scaling to min/max per axis after optional quantile censoring).
- **`shade=True`** (default): density ramp `█▓▒░○·` by count quantiles.
- **`shade=False`**: single **`marker`** character per occupied cell (default `♦`).
- **`left_censor`**, **`right_censor`**: tail fractions in `[0, 0.5)` trimmed on **both** `x` and `y` using separate quantiles.
- **`xscale`**, **`yscale`**: must be `None` or `'log'` (asserted; mapping remains linear in current implementation).

---

## `hist(*xs, …)`

- One or more 1-D arrays; each panel uses **`bins`** bins, **`ncols`** wide (default `ncols == bins`; `ncols` must be divisible by `bins`).
- **`histtype`**: `'bar'` (Unicode block column heights with partial blocks ` ▁▂▃▄▅▆▇█`) or `'step'` (`♦` at bin height).
- **`sharex`**: common value range for all series when `True`; **`density`**: scale counts for comparability when appropriate.
- **`gridcols`**: number of histograms per row; multiple rows of panels as needed.
- **`cumulative`**: cumulative counts before scaling.
- **`labels`**: titles centered above each panel.

---

## Minimal examples

```python
import numpy as np
from kanly.api import plot, scatter, hist

y = np.cumsum(np.random.randn(500))
print(plot(y, nrows=12, ncols=60, do_print=False))

x = np.random.randn(300)
z = x + 0.3 * np.random.randn(300)
print(scatter(x, z, nrows=15, ncols=70, shade=True))

a = np.random.randn(2000)
b = np.random.randn(2000) * 1.5 + 1
print(hist(a, b, nrows=10, ncols=40, bins=20, labels=["a", "b"]))
```

---

## Limitations

- Fixed-width monospace font assumed; wide Unicode may misalign in some terminals.
- Log scaling flags are not fully wired through to the raster math.
- Very large `ncols` / `nrows` strings can be heavy to build; keep dimensions modest for logs.
