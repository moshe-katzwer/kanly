"""The joint synthetic-control equations and their parameter layout.

This is the elastic-net model from the original sandbox, not a replication of
Powell's 2026 moment-condition estimator. See README.md for the comparison.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .panel import PanelData


@dataclass(frozen=True)
class ModelOptions:
    """Structural choices, kept separate from numerical optimizer settings.

    Keeping these together prevents a placebo fit from silently reverting to a
    different donor pool, set of target equations, or weighting convention.
    """

    pooled_treatment: bool = False
    fit_only_treatment_units: bool = True
    weight_control_only: bool = False
    positive: bool = False
    weight_mode: str = "observation"

    def __post_init__(self):
        if self.weight_mode not in ("observation", "legacy"):
            raise ValueError("weight_mode must be 'observation' or 'legacy'.")


@dataclass(frozen=True)
class ModelComponents:
    """Separate structural contributions from residuals used by the optimizer."""

    donor_weights: np.ndarray
    treatment_coefficients: np.ndarray
    covariate_coefficients: np.ndarray
    treatment: np.ndarray
    covariates: np.ndarray
    synthetic: np.ndarray
    residuals: np.ndarray
    weighted_residuals: np.ndarray

    @property
    def fitted(self):
        return self.covariates + self.treatment + self.synthetic

    @property
    def counterfactual(self):
        # Donor outcomes were already adjusted for their treatment effects.
        # Remove only the target's own treatment contribution from its fit.
        return self.covariates + self.synthetic


class JointModel:
    """Evaluate z - z @ W, with z = outcome - treatment - covariates.

    W[donor, target] is a contribution TO target FROM donor. Its diagonal is
    zero. Parameter names keep the original target-first convention:
    omega_A_B is the weight on donor B when predicting target A.

    The full off-diagonal parameter layout is retained for comparisons with
    the old sandbox. Disabled donor weights are fixed at zero by bounds.
    """

    def __init__(self, panel: PanelData, options: ModelOptions):
        self.panel = panel
        self.options = options
        n_units = len(panel.units)
        self.ever_treated = np.any(panel.treatments != 0, axis=0)  # (N, Q)
        self.treated_units = self.ever_treated.any(axis=1)
        # np.where visits rows first, yielding the historical parameter order:
        # all donors for target 0, then all donors for target 1, and so on.
        self.targets, self.donors = np.where(~np.eye(n_units, dtype=bool))
        self.active_weights = np.ones(len(self.targets), dtype=bool)
        if options.fit_only_treatment_units:
            # This flag controls synthetic predictions, not inclusion in the
            # loss: control equations still estimate their covariate effects.
            self.active_weights &= self.treated_units[self.targets]
        if options.weight_control_only:
            self.active_weights &= ~self.treated_units[self.donors]

        self.n_weights = len(self.targets)
        # Each entry tells unpack() which unit/treatment cells share one scalar
        # coefficient. This avoids constructing large diagonal design matrices.
        self.treatment_positions = []
        names = [f"omega_{panel.units[i]}_{panel.units[j]}" for i, j in zip(self.targets, self.donors)]
        for q, treatment in enumerate(panel.treatment_names):
            exposed = np.flatnonzero(self.ever_treated[:, q])
            if options.pooled_treatment:
                self.treatment_positions.append((exposed, q))
                names.append(str(treatment))
            else:
                for unit in exposed:
                    self.treatment_positions.append((np.array([unit]), q))
                    names.append(f"{treatment}_{panel.units[unit]}")
        # Parameter blocks: [off-diagonal weights | exposed-unit treatment
        # coefficients | intercept and covariate coefficients for each unit].
        self.treatment_slice = slice(self.n_weights, len(names))
        self.covariate_slice = slice(len(names), None)
        names.extend(f"{c}_{unit}" for unit in panel.units for c in panel.covariate_names)
        if len(set(names)) != len(names):
            raise ValueError("Unit/column labels produce ambiguous parameter names; rename the labels.")
        self.param_names = pd.Index(names, name="parameter")

    @property
    def n_params(self):
        return len(self.param_names)

    def unpack(self, params):
        """Expand the flat optimizer vector into small, directly usable arrays."""
        params = np.asarray(params, dtype=float)
        if params.shape != (self.n_params,) or not np.isfinite(params).all():
            raise ValueError(f"Expected {self.n_params} finite parameters in a one-dimensional array.")
        n_units = len(self.panel.units)
        weights = np.zeros((n_units, n_units))
        weights[self.donors, self.targets] = params[:self.n_weights] * self.active_weights
        coefficients = np.zeros((n_units, len(self.panel.treatment_names)))
        for value, (units, treatment) in zip(params[self.treatment_slice], self.treatment_positions):
            coefficients[units, treatment] = value
        covariates = params[self.covariate_slice].reshape(n_units, len(self.panel.covariate_names))
        return weights, coefficients, covariates

    def evaluate(self, params):
        """Evaluate the model with no data mutation or optimizer side effects.

        Einstein sums contract the last axis (treatments or covariates), keeping
        time and unit. For example, treatment[t, i] = sum_q D[t, i, q]*delta[i,q].
        """
        weights, treatment_coefficients, covariate_coefficients = self.unpack(params)
        treatment = np.einsum("tnq,nq->tn", self.panel.treatments, treatment_coefficients)
        covariates = np.einsum("tnk,nk->tn", self.panel.covariates, covariate_coefficients)
        # Adjustment happens to ALL units before applying donor weights. Thus
        # a treated donor contributes its estimated untreated residual outcome,
        # and treatment coefficients interact with W inside the joint objective.
        adjusted_outcomes = self.panel.outcomes - treatment - covariates
        synthetic = adjusted_outcomes @ weights
        residuals = adjusted_outcomes - synthetic
        # The optimizer squares its residual vector. WLS therefore needs sqrt(w).
        # 'legacy' deliberately reproduces the old code's effective w**2 loss.
        scale = self.panel.observation_weights
        if self.options.weight_mode == "observation":
            scale = np.sqrt(scale)
        return ModelComponents(weights, treatment_coefficients, covariate_coefficients,
                               treatment, covariates, synthetic, residuals, residuals * scale)

    def residual_vector(self, params):
        """Flatten in time-major order; the solver minimizes mean(r**2)/2."""
        return self.evaluate(params).weighted_residuals.ravel()

    def penalty_weights(self, alpha):
        """Regularize donor weights only; treatment and covariates stay unpenalized."""
        penalties = np.zeros(self.n_params)
        penalties[:self.n_weights] = alpha
        return penalties

    def bounds(self):
        """Restrict only donor weights, leaving structural coefficients free.

        There is no sum-to-one restriction. Even positive weights can therefore
        extrapolate beyond a convex combination of the donor outcomes.
        """
        bounds = np.tile([-np.inf, np.inf], (self.n_params, 1))
        if self.options.positive:
            bounds[:self.n_weights, 0] = 0
        bounds[np.flatnonzero(~self.active_weights)] = 0
        return bounds


def build_model(data, outcome_col, unit_col, time_col, treatment_cols, covariate_cols=None,
                weight_col=None, *, pooled_treatment=False, fit_only_treatment_units=True,
                weight_control_only=False, positive=False, weight_mode="observation"):
    """Build a model without fitting it, for inspecting or checking the equations.

    model.param_names describes the flat vector expected by model.evaluate().
    This boundary keeps pandas alignment and validation out of the optimization
    loop, which only needs the already aligned NumPy arrays.
    """
    panel = PanelData.from_frame(data, outcome_col, unit_col, time_col, treatment_cols,
                                covariate_cols, weight_col)
    options = ModelOptions(pooled_treatment, fit_only_treatment_units, weight_control_only,
                           positive, weight_mode)
    return JointModel(panel, options)
