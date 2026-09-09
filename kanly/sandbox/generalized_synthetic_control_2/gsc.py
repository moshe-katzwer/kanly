"""Fit the joint model using kanly's existing nonlinear elastic-net solver."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from kanly.regression.nonlinear_least_squares.optimize.nlls_coordinate_descent_minimize_internal import (
    nlls_elastic_net_minimize_internal_coordinate_descent,
)

from .model import JointModel, ModelComponents, build_model


@dataclass(frozen=True)
class FitOptions:
    """Numerical settings saved verbatim for subsequent placebo fits.

    alpha is the overall donor-weight penalty; l1_ratio allocates it between
    absolute-value shrinkage (1) and squared-weight shrinkage (0). seed controls
    coordinate order when selection='random', separately from assignment seeds.
    """

    alpha: float = 1.0
    l1_ratio: float = 0.9
    max_iter: int = 500
    ftol: float = 1e-6
    xtol: float = 1e-6
    selection: str = "random"
    seed: int = 0

    def __post_init__(self):
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be finite and nonnegative.")
        if not np.isfinite(self.l1_ratio) or not 0 <= self.l1_ratio <= 1:
            raise ValueError("l1_ratio must be between zero and one.")
        if not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")
        if any(not np.isfinite(tol) or tol <= 0 for tol in (self.ftol, self.xtol)):
            raise ValueError("ftol and xtol must be finite and positive.")
        if self.selection not in ("random", "cyclic", "greedy"):
            raise ValueError("selection must be 'random', 'cyclic', or 'greedy'.")


@dataclass(frozen=True)
class GSCResult:
    """Fit, labeled predictions, and the complete specification needed to refit.

    Predictions are in-sample reconstructions using contemporaneous donor
    outcomes; they are not forecasts. ``converged`` is an optimizer diagnostic,
    not a guarantee of global optimality or causal identification.
    """

    model: JointModel
    fit_options: FitOptions
    params: pd.Series
    components: ModelComponents
    optimization_result: dict

    @property
    def converged(self):
        return bool(self.optimization_result["converged"])

    @property
    def fittedvalues(self):
        """Full in-sample predictions, including the target's treatment effect."""
        return self.model.panel.frame(self.components.fitted)

    @property
    def counterfactual(self):
        """Fitted outcome with all of the target's treatment contributions removed."""
        return self.model.panel.frame(self.components.counterfactual)

    @property
    def resid(self):
        """Raw errors, so outcomes == fittedvalues + resid even with weights."""
        return self.model.panel.frame(self.components.residuals)

    @property
    def weighted_resid(self):
        """Errors on the scale actually squared by the optimization objective."""
        return self.model.panel.frame(self.components.weighted_residuals)

    @property
    def treatment_contribution(self):
        """Sum of exposure times its coefficient, separately for every panel cell."""
        return self.model.panel.frame(self.components.treatment)

    @property
    def donor_weights(self):
        """A labeled W matrix: select a column to inspect one target's donors."""
        units = self.model.panel.units
        return pd.DataFrame(self.components.donor_weights,
                            index=units.rename("donor"), columns=units.rename("target"))

    @property
    def treatment_effects(self):
        # Zero exposure means no unit-specific coefficient was estimated.
        # Show NaN rather than suggesting an estimated zero treatment effect.
        coefficients = np.where(self.model.ever_treated, self.components.treatment_coefficients, np.nan)
        return pd.DataFrame(coefficients, index=self.model.panel.units,
                            columns=self.model.panel.treatment_names)

    @property
    def covariate_coefficients(self):
        """Each unit's intercept and slopes, before the donor adjustment."""
        return pd.DataFrame(self.components.covariate_coefficients, index=self.model.panel.units,
                            columns=self.model.panel.covariate_names)


def fit_model(model: JointModel, options: FitOptions, *, x0=None, debug=False):
    """One fitting path shared by the main estimate and every placebo refit.

    A labeled starting Series is reordered by name. An unlabeled vector must
    already follow model.param_names. Inactive donor coordinates are clipped
    to zero by the solver's bounds, including for user-provided starting values.
    """
    if isinstance(x0, pd.Series):
        if not x0.index.is_unique or set(x0.index) != set(model.param_names):
            raise ValueError("Starting parameter labels must match the model exactly.")
        x0 = x0.reindex(model.param_names).to_numpy()
    if x0 is not None:
        # The coordinate solver updates in place; integer inputs must not cause
        # fractional updates to be truncated by an integer parameter array.
        x0 = np.asarray(x0, dtype=float)
        model.unpack(x0)  # Fail before optimization on invalid starting values.
    # The existing solver already implements the elastic-net penalty and bounds.
    # Pin its scaling options explicitly so defaults elsewhere cannot change this
    # module's objective. A zero start is the solver's default when x0 is None.
    optimization = nlls_elastic_net_minimize_internal_coordinate_descent(
        model.residual_vector, x0=x0, num_params=model.n_params,
        bounds=model.bounds(), alpha=model.penalty_weights(options.alpha),
        l1_ratio=options.l1_ratio, max_iter=options.max_iter,
        ftol=options.ftol, xtol=options.xtol, selection=options.selection,
        seed=options.seed, debug=debug, prompt_user_for_more_iters=False,
        normalize=False, scale_penalties=True,
    )
    # Keep unsuccessful fits inspectable. Callers and placebo diagnostics can
    # check converged/message instead of silently treating every iterate as final.
    params = pd.Series(optimization["params"], index=model.param_names, name="estimate")
    return GSCResult(model, options, params, model.evaluate(params), optimization)


def gsc(data, outcome_col, unit_col, time_col, treatment_cols, covariate_cols=None,
        weight_col=None, *, pooled_treatment=False, positive=False, alpha=1.0,
        l1_ratio=0.9, max_iter=500, ftol=1e-6, xtol=1e-6,
        weight_control_only=False, selection="random", fit_only_treatment_units=True,
        weight_mode="observation", seed=0, x0=None, debug=False, log_transform=False):
    """Jointly fit treatment effects, covariates, and synthetic-control weights.

    The loss is mean(w * residual**2) / 2 plus an elastic-net penalty on
    donor weights. Observation weights are not normalized: rescaling them
    changes the strength of the penalty relative to fit. ``weight_mode='legacy'``
    uses w**2 to reproduce the original sandbox's weighting convention.

    ``pooled_treatment`` shares each treatment coefficient across exposed units.
    Otherwise each exposed unit gets its own coefficient. ``positive`` restricts
    donor weights only; weights never have a sum-to-one constraint.

    ``fit_only_treatment_units`` enables synthetic predictions only for ever-
    treated targets, while retaining every unit's residual in the loss.
    ``weight_control_only`` restricts donors to never-treated units.

    Use the separate time_placebos() and permutation_placebos() functions for
    diagnostic refits. They preserve these options and report convergence.
    """
    if log_transform:
        raise NotImplementedError(
            "Transform the outcome column explicitly; automatic log transformation is unsupported."
        )
    model = build_model(data, outcome_col, unit_col, time_col, treatment_cols, covariate_cols,
                        weight_col, pooled_treatment=pooled_treatment, positive=positive,
                        fit_only_treatment_units=fit_only_treatment_units,
                        weight_control_only=weight_control_only, weight_mode=weight_mode)
    options = FitOptions(alpha, l1_ratio, max_iter, ftol, xtol, selection, seed)
    return fit_model(model, options, x0=x0, debug=debug)
