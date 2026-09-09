"""The original sandbox's two-stage residual regression, with explicit stages.

Despite the historical function name, this is not Powell's full imperfect
synthetic-control estimator: it does not denoise outcomes, construct the 2026
moment conditions, or weight equations by synthetic-control fit quality.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from kanly.regression.linear_models.penalized.model import SparsePenalizedLinearModel

from .gsc import FitOptions
from .panel import PanelData


@dataclass(frozen=True)
class ISCResult:
    """First-stage predictions and second-stage descriptive coefficient estimates.

    unit_effects are separate residual-regression slopes, as in the old sandbox.
    When donors are treated, each such slope implicitly uses the same effect
    for the target and its donors. It is not a joint heterogeneous-effects fit.
    Use gsc(pooled_treatment=False) for that different specification.
    """

    first_stage_fits: dict
    predicted_outcomes: pd.DataFrame
    predicted_treatments: pd.DataFrame
    adjusted_outcomes: pd.DataFrame
    adjusted_treatments: pd.DataFrame
    donor_weights: pd.DataFrame
    treatment_start: pd.Series
    training_periods: dict
    unit_effects: pd.Series
    pooled_effect: float
    second_stage_periods: pd.Index


def _slope(outcome, exposure):
    """The no-intercept least-squares slope; NaN denotes no identifying variation."""
    denominator = np.dot(exposure, exposure)
    return float(np.dot(exposure, outcome) / denominator) if denominator > 0 else np.nan


def isc(data, outcome, unit_col, time_col, treatment_col, covariate_cols=None, *,
        max_iter=200, alpha=0.0001, l1_ratio=0.9, positive=True, tol=1e-6,
        shift=0, do_sc=False, debug=False):
    """Learn pretreatment donor weights, then regress outcome gaps on exposure gaps.

    do_sc=False uses all other units as donors and trains before the first
    treatment anywhere. do_sc=True uses never-treated donors and trains before
    each target's own treatment. Only ever-treated targets enter stage two,
    starting at the first treatment anywhere, matching the original equations.

    shift rotates outcomes AND covariates earlier while keeping treatments fixed
    (the original isc convention, distinct from time_placebos for gsc).
    Returned slopes have no automatically calibrated causal standard errors.
    """
    FitOptions(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, ftol=tol, xtol=tol)
    if not isinstance(shift, (int, np.integer)):
        raise ValueError("shift must be an integer number of periods.")
    panel = PanelData.from_frame(data, outcome, unit_col, time_col, treatment_col, covariate_cols)
    if len(panel.treatment_names) != 1:
        raise ValueError("isc supports exactly one treatment column.")
    # Work on the aligned snapshot. The long DataFrame's row index and ordering
    # never determine treatment dates, training slices, or output assignment.
    outcomes = np.roll(panel.outcomes, -int(shift), axis=0)
    covariates = np.roll(panel.covariates, -int(shift), axis=0)
    exposures = panel.treatments[:, :, 0]
    treated = np.any(exposures != 0, axis=0)
    targets = np.flatnonzero(treated)
    starts = {i: np.flatnonzero(exposures[:, i] != 0)[0] for i in targets}
    common_start = min(starts.values())

    # Untreated targets are not fitted by this historical two-stage estimator;
    # mark their predictions missing instead of manufacturing zero predictions.
    predicted_y = np.full_like(outcomes, np.nan)
    predicted_d = np.full_like(exposures, np.nan)
    weights = np.zeros((len(panel.units), len(panel.units)))
    fits, training_periods = {}, {}
    for target in targets:
        donors = np.array([j for j in range(len(panel.units))
                           if j != target and (not do_sc or not treated[j])], dtype=int)
        if len(donors) == 0:
            raise ValueError(f"No eligible donors for unit {panel.units[target]!r}.")
        # With treated donors we need a period untreated for EVERY donor. When
        # donors are all never treated, each target can use its longer preperiod.
        stop = starts[target] if do_sc else common_start
        if stop < 2:
            raise ValueError("At least two pretreatment periods are required for each first-stage fit.")

        # Formula-safe internal names support numeric, spaced, or punctuated unit
        # labels. Results get the user's original labels after prediction.
        donor_names = [f"donor_{j}" for j in donors]
        covariate_names = [f"covariate_{k}" for k in range(1, covariates.shape[2])]
        features = pd.DataFrame(outcomes[:, donors], columns=donor_names)
        features["outcome"] = outcomes[:, target]
        for k, name in enumerate(covariate_names, start=1):
            features[name] = covariates[:, target, k]
        formula = "outcome ~ " + " + ".join(donor_names + covariate_names)
        # Keep the original first-stage normalization and donor-only penalties.
        # Exclude disallowed donors explicitly, instead of approximating exclusion
        # with a huge penalty. No covariance calculation is needed for stage two.
        fit = SparsePenalizedLinearModel.elastic_net(
            formula, features, index=np.arange(len(panel.periods)) < stop,
            alpha={name: alpha for name in donor_names}, l1_ratio=l1_ratio,
            positive={name: True for name in donor_names} if positive else False,
            normalize=True, fit_intercept=True, debug=debug, max_iter=max_iter,
            xtol=tol, ftol=tol, gtol=tol, apply_scaling=False, refit=False,
            compute_cov=False, prompt_user_for_more_iters=False,
        )
        predicted_y[:, target] = np.asarray(fit.predict(features)).ravel()

        # Apply ONLY donor coefficients to treatment histories. Intercepts and
        # covariates explain outcomes, not exposure to treatment.
        donor_coefficients = fit.params.reindex(donor_names).to_numpy(dtype=float)
        weights[donors, target] = donor_coefficients
        predicted_d[:, target] = exposures[:, donors] @ donor_coefficients
        fits[panel.units[target]] = fit
        training_periods[panel.units[target]] = panel.periods[:stop].copy()

    # Stage two estimates gap_y = delta * gap_d without an additional intercept.
    # With a shared treatment effect this accounts for exposure among donors:
    # gap_d[t,i] = D[t,i] - sum_j W[j,i] * D[t,j]. Separate slopes reproduce
    # the old per-target regressions; they do not estimate donors' effects jointly.
    adjusted_y = outcomes - predicted_y
    adjusted_d = exposures - predicted_d
    stage_y = adjusted_y[common_start:, targets]
    stage_d = adjusted_d[common_start:, targets]
    # Pool by stacking observations, not by averaging the unit-specific slopes:
    # units with more adjusted-exposure variation contribute more to pooled OLS.
    effects = [_slope(stage_y[:, j], stage_d[:, j]) for j in range(len(targets))]
    pooled = _slope(stage_y.ravel(), stage_d.ravel())
    if not np.isfinite(pooled):
        raise ValueError("The adjusted treatment has no variation; the pooled effect is unidentified.")
    return ISCResult(
        first_stage_fits=fits,
        predicted_outcomes=panel.frame(predicted_y), predicted_treatments=panel.frame(predicted_d),
        adjusted_outcomes=panel.frame(adjusted_y), adjusted_treatments=panel.frame(adjusted_d),
        donor_weights=pd.DataFrame(weights, index=panel.units.rename("donor"), columns=panel.units.rename("target")),
        treatment_start=pd.Series(
            {panel.units[i]: panel.periods[start] for i, start in starts.items()}, name="first_treatment"
        ),
        training_periods=training_periods,
        unit_effects=pd.Series(effects, index=panel.units[targets], name="estimate"),
        pooled_effect=pooled, second_stage_periods=panel.periods[common_start:].copy(),
    )
