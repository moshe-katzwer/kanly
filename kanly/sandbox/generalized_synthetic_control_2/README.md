# Generalized synthetic control, second sandbox

This module refactors `../generalized_synthetic_control` into separate panel,
model, fitting, and placebo components. It preserves the original joint
elastic-net model and the two-stage residual regression, with explicit fixes
listed below. The original sandbox is unchanged.

This is experimental estimation code, not a replication package for the Powell
papers. The reference comparison explains the relationship and the differences.

## Run the example

From the repository root, using an environment with kanly's dependencies:

```bash
python -m kanly.sandbox.generalized_synthetic_control_2.example
python -m kanly.sandbox.generalized_synthetic_control_2.example --placebos 6
python -m kanly.sandbox.generalized_synthetic_control_2.example --plot /tmp/gsc2.png
```

The example simulates two controls and one treated unit with a known treatment
effect of 3. It prints estimates from both estimators, donor weights, optimizer
convergence, and the error in the estimated untreated trajectory. The optional
plot is saved without opening a GUI; placebo results are diagnostic estimates.

## Joint estimator: `gsc`

```python
from kanly.sandbox.generalized_synthetic_control_2 import gsc, permutation_placebos

result = gsc(
    data,
    outcome_col="outcome",
    unit_col="unit",
    time_col="time",
    treatment_cols=["treatment"],
    covariate_cols=["trend"],       # omit if no additional covariates are needed
    pooled_treatment=True,
    weight_control_only=True,
    positive=True,
    alpha=0.01,
)

print(result.treatment_effects)
print(result.donor_weights)        # rows = donors; columns = targets
print(result.optimization_result["message"])
untreated_prediction = result.counterfactual

placebos = permutation_placebos(result, num_draws=10, shifts=0, seed=4)
print(placebos.params[["treatment"]].join(placebos.diagnostics))
```

Input is a long-form DataFrame with exactly one row per time/unit combination.
The panel must be complete and contain at least two units and two periods.
Outcome, covariate, treatment, and observation-weight values must be numeric and
finite. Missing observations, duplicate keys, and all-zero treatment columns
raise errors. Unit/time labels are sorted once, and every variable is aligned
to those axes. A numeric time column can itself be a covariate. The input
DataFrame is never modified.

Each unit receives an intercept and its own covariate coefficients. Treatment
columns may describe overlapping policies, fractional exposure, or continuous
doses; each coefficient is constant over time within its unit/treatment pair.

### Equations and objective

Let `Y[t, i]` be the outcome, `D[t, i, q]` treatment q, and `X[t, i, k]`
covariate k, including the intercept. Let `delta[i, q]` and `beta[i, k]` be
coefficients. The adjusted outcome is

```text
Z[t, i] = Y[t, i] - sum_q D[t, i, q] * delta[i, q]
                   - sum_k X[t, i, k] * beta[i, k]
```

The donor matrix is oriented as `W[donor, target]`, with zero diagonal:

```text
synthetic = Z @ W
R = Z - synthetic
```

Equivalently, a target's outcome is explained by its own treatment and
covariates, plus a weighted combination of other units' adjusted outcomes.
Treated donors are adjusted for their estimated treatment effects before their
outcomes are used. The fit uses **all periods**, including treated periods.

With observation weights `v[t, i]`, the minimized objective is

```text
mean(v * R**2) / 2
    + alpha * (l1_ratio * sum(abs(W)) + (1 - l1_ratio) * sum(W**2) / 2)
```

Only donor weights are penalized. There is no automatic choice of `alpha` or
cross-validation. Observation weights are not normalized, so multiplying them
by a constant changes the fit/penalty tradeoff. A zero weight omits that target
cell's error from the loss; its observed outcome can still be a donor elsewhere.

The optimizer receives `sqrt(v) * R`, since it squares the supplied residuals.
`weight_mode="legacy"` instead supplies `v * R`, reproducing the old sandbox's
effective `v**2` weighting. With no observation weights, these modes coincide.

Weights multiply unknown treatment/covariate coefficients, so the joint
objective is generally nonconvex. The existing kanly coordinate-descent solver
is used with a zero starting vector unless `x0` is provided. Check
`result.converged` and `result.optimization_result`; convergence is not evidence
of a global optimum or of causal identification. Constant/collinear exposures,
weak adjusted-exposure variation, or redundant donor relationships can make
coefficients unidentified. This sandbox does not diagnose every such case.

### Model switches

| Argument | Behavior |
| --- | --- |
| `pooled_treatment=False` | One coefficient for each exposed unit and treatment. |
| `pooled_treatment=True` | One coefficient per treatment, shared across exposed units. |
| `fit_only_treatment_units=True` | Synthetic predictions only for ever-treated targets. **All units still enter the residual loss.** |
| `fit_only_treatment_units=False` | Synthetic predictions for every target. |
| `weight_control_only=True` | Only units never exposed to any treatment can be donors. |
| `weight_control_only=False` | Other treated units may be donors after adjustment. |
| `positive=True` | Donor weights must be nonnegative; structural coefficients remain unrestricted. |

Donor weights do **not** sum to one by construction, even with `positive=True`.
This is a consequential modeling choice, not just a numerical implementation
detail. The full off-diagonal parameter vector is retained for comparison with
the original code; disabled weights are fixed to zero by bounds.

### Results and intermediate quantities

| Attribute | Meaning |
| --- | --- |
| `params` | Flat named coefficients; `omega_A_B` means donor B contributes to target A. |
| `treatment_effects` | Unit-by-treatment coefficient table; unexposed pairs are `NaN`. |
| `covariate_coefficients` | Unit-by-covariate coefficients, including `Intercept`. |
| `donor_weights` | Labeled matrix with donors in rows and targets in columns. |
| `fittedvalues` | Full in-sample predictions, including treatment contributions. |
| `counterfactual` | Fitted predictions with the target's treatment contributions removed. |
| `treatment_contribution` | Exposure multiplied by fitted coefficients, summed over treatments. |
| `resid` | Raw residuals: `outcomes == fittedvalues + resid`. |
| `weighted_resid` | Residuals on the scale used in the objective. |
| `components` | NumPy arrays for the separate treatment, covariate, synthetic, and error terms. |
| `model`, `fit_options` | Aligned panel, structural choices, and saved optimizer settings. |

`counterfactual == fittedvalues - treatment_contribution`. Donor outcomes
already have their estimated treatment contributions removed inside the model.
These are in-sample reconstructions using contemporaneous donor outcomes, not
forecasts or independent predictions on held-out data. Untreated outcomes are
interpretable causally only under appropriate assumptions about the exposure,
covariates, donor relationships, and errors.

`build_model(...)` constructs the equations without optimization. Inspect
`model.param_names`, then pass a vector in that order to `model.evaluate(params)`
or `model.residual_vector(params)` for debugging or alternative optimization.

## Two-stage estimator: `isc`

```python
from kanly.sandbox.generalized_synthetic_control_2 import isc

two_stage = isc(data, "outcome", "unit", "time", "treatment", do_sc=True)
print(two_stage.unit_effects)
print(two_stage.pooled_effect)
```

The historical name is retained, but this routine implements the original
sandbox's two-stage algorithm:

1. For each ever-treated target, fit its outcome on donor outcomes and its own
   covariates using pretreatment elastic net. Donor coefficients are penalized;
   intercept/covariate coefficients are not.
2. Apply those coefficients to all periods to form outcome predictions. Apply
   only the donor coefficients to donor treatment histories to form predicted
   treatment exposure.
3. Compute `gap_y = outcome - predicted_outcome` and
   `gap_d = treatment - predicted_treatment`. Regress `gap_y` on `gap_d` with no
   additional intercept, separately by target and after pooling observations.

`do_sc=False` uses all other units as donors and ends every training window at
the first treatment anywhere. `do_sc=True` restricts donors to never-treated
units and trains until each target's own first treatment. At least two
pretreatment periods are required. Stage two includes ever-treated targets
from the first treatment date anywhere, following the original specification.

When donors are treated, each separate unit slope assumes that the same slope
also describes its donors' treatment effects. It is not a simultaneous estimate
of heterogeneous donor effects; use the joint model for that specification.
`predicted_outcomes` is a first-stage prediction and is not automatically an
untreated counterfactual when donors are treated. Untreated targets have missing
predictions because this routine does not fit their equations.

The result retains first-stage regression objects, donor weights, training
periods, adjusted outcomes/exposures, and second-stage point estimates. There
are no automatically calibrated causal standard errors. The optional `shift`
rotates outcomes and covariates while holding treatment fixed, preserving the
old `isc` convention; the joint-model placebo functions instead rotate treatment.

## Placebo diagnostics

`time_placebos(result, stride=1)` refits at circular shifts `1, ..., T-1`.
`permutation_placebos(result, num_draws=50, seed=0, shifts=None)` reassigns entire
treatment histories across units, using the same permutation for all treatment
columns. `shifts=None` adds random circular shifts, `shifts=0` only reassigns
units, and an integer or sequence can specify shifts explicitly.

Every refit retains the original observation weights, covariates, pooling,
donor/target rules, penalty, and solver settings. Each starts from zero rather
than reusing treatment parameters associated with the original assignment.
The assignment RNG is separate from the optimizer's coordinate-order RNG.

Results contain `params`, `diagnostics`, `shifts`, and `assignments`. Assignment
columns are recipient units; each value is the source of that recipient's
treatment history. Parameters retain the actual recipient labels in each draw.
A missing unit-specific coefficient is `NaN`, not a coefficient attributed to
the original treated unit. Failed-to-converge fits are retained and marked.

These helpers do not impose a sharp null, calculate calibrated p-values, or
invert tests into intervals. In particular, the old name `conformal_inference`
overstated what the routine computed. Circular shifts and cross-unit assignments
require their own stationarity/exchangeability justification; the parameter
histograms alone do not supply it.

## Relationship to the references

**Powell (2022), “Synthetic Control Estimation Beyond Comparative Case Studies:
Does the Minimum Wage Reduce Employment?”, JBES 40(3), 1302–1314.** The published
abstract describes joint estimation of synthetic controls and coefficients on
multiple discrete or continuous explanatory variables. That is the closest
conceptual match to `gsc`: treatment coefficients and donor weights are learned
together. The sandbox additionally makes explicit choices about elastic-net
regularization, unit-specific intercepts/covariates, donor eligibility, and
optional pooling. Similarity of this architecture does not establish numerical
or inferential equivalence to the published estimator.
[Publisher record and abstract](https://www.tandfonline.com/doi/abs/10.1080/07350015.2021.1927743).

**The supplied RAND WR-1246 is dated May 2018**, titled “Imperfect Synthetic
Controls: Did the Massachusetts Health Care Reform Save Lives?” Its Section 3.1
uses outcome and treatment gaps for all units, including untreated units whose
synthetic controls contain the treated unit. Sections 3.2 and 4 first predict
outcomes from unit-specific time functions, then learn synthetic weights from
those predictions; equation (18) estimates the effect using weighted gaps.
The sandbox's `isc` shares the gap-regression idea, but fits only treated targets,
learns weights from raw outcomes, and omits that preliminary denoising and
fit-quality equation weighting. Two stages alone therefore do not make it the
paper's estimator.
[RAND working paper, Sections 3–4](https://www.rand.org/content/dam/rand/pubs/working_papers/WR1200/WR1246/RAND_WR1246.pdf).

**Powell (2026), “Imperfect Synthetic Controls,” JAE 41(3), 253–264**, has a
different treatment of transitory shocks. Sections 3.2–3.4 construct predicted-
outcome moment conditions, jointly estimate synthetic weights under nonnegative,
sum-to-one restrictions, and update equation weights to reduce the influence
of units lacking appropriate synthetic controls. Section 5 uses multiple effect
estimates for a t-based inference procedure. Neither sandbox estimator implements
these moment conditions, adaptive equation weights, or that inference procedure.
Observation weights in `gsc` are user-supplied loss weights and are not the
paper's estimated equation weights. Allowing every target with
`fit_only_treatment_units=False` recovers an all-unit fitting scope, but not the
2026 estimator.
[Published article, Sections 3–5](https://onlinelibrary.wiley.com/doi/full/10.1002/jae.70035).

The 2018 working paper and 2026 article should not be treated as interchangeable
implementations: the former uses preliminary time-function predictions, while
the latter develops joint moment conditions. The full RAND paper and 2026
article were accessible for this comparison. Direct access to the 2022 full
article was blocked by the publisher; the 2022 comparison is limited to the
published abstract and does not claim a full equation-by-equation verification.

## Changes from the first sandbox

| Original behavior | New behavior |
| --- | --- |
| One closure constructs matrices, parameters, and errors | Explicit panel snapshot, parameter layout, and component evaluation. |
| Temporary intercept and `isc` prediction columns modify input | Input remains unchanged; outputs are separate labeled objects. |
| Unchecked duplicate/missing panel cells | Clear validation errors before optimization. |
| `weight_col` multiplies errors directly | Standard observation weighting by default; `weight_mode="legacy"` reproduces the old loss. |
| `resid` contains weighted errors | Separate `resid` and `weighted_resid`. |
| `log_transform=True` silently does nothing | Explicit unsupported-operation error; transform the input yourself. |
| Notebook adds treatment when plotting a counterfactual | `counterfactual` subtracts the target's treatment contribution. |
| Placebo fits drop some model options and reuse old labels | One shared fitting path and recipient-aware parameter alignment. |
| Placebo fits reuse the original parameter vector | Deterministic zero starts for each assignment. |
| `isc` relies on long-frame row positions and formula-safe unit labels | Panel time positions and internal formula-safe names. |
| `isc` excludes donors with enormous penalties | Ineligible donors are omitted explicitly. |
| `isc` uses a removed `tol` solver argument | Current `xtol`, `ftol`, and `gtol` arguments. |

This is a separate API, not a drop-in dictionary wrapper. Access `result.params`
instead of `result["params"]`; use `time_placebos` and `permutation_placebos`
explicitly instead of `do_conformal`/`do_permutation` flags. Two-stage results
expose slopes directly instead of wrapping the final regressions in `lm`.
Interactive requests for more optimizer iterations are disabled.

## Code map and verification

| File | Responsibility |
| --- | --- |
| `panel.py` | Input validation, axis alignment, and treatment reassignment. |
| `model.py` | Parameter layout, donor restrictions, and residual equations. |
| `gsc.py` | Optimizer configuration, joint fit, and labeled results. |
| `isc.py` | Pretreatment donor fits and second-stage gap regressions. |
| `inference.py` | Reproducible placebo refits and diagnostics. |
| `example.py` | Simulation, estimator comparison, optional placebos and plot. |

Run the focused tests from the repository root:

```bash
python -m unittest discover -s tests -p 'test_generalized_synthetic_control_2.py' -v
```

Tests compare the joint residual equations with the original across pooling and
donor/target settings, verify known-effect recovery, check observation weighting
and counterfactuals, reject malformed panels, and compare placebo results with
explicitly reassigned-data fits. They also cover shuffled rows and treatment
windows in `isc`. These checks establish software behavior, not the statistical
properties of the estimator under every data-generating process.
