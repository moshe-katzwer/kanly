import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from kanly.sandbox.generalized_synthetic_control_2 import gsc, isc, permutation_placebos


def simulate_panel(seed=12, num_controls=10):
    """controls explain the untreated outcome of one treated unit."""
    random = np.random.RandomState(seed)
    periods = np.arange(60)
    control = random.normal(size=(len(periods), num_controls))
    control = control @ random.rand(*(num_controls, num_controls))
    treatment = (periods >= 35).astype(float)
    effect = 0.75
    # This outcome is generated before treatment is added, so we can compare
    # the model's estimated untreated trajectory to something actually known.
    beta = random.rand(num_controls) / num_controls
    untreated = 2.0 + control.dot(beta) + 0.15 * random.normal(size=len(periods))
    data = pd.DataFrame({
        "unit": np.repeat([f"control_{j}" for j in range(num_controls)] + ["treated"], len(periods)),
        "time": np.tile(periods, num_controls + 1),
        "outcome": np.concatenate([control.T.flatten(), untreated + effect * treatment]),
        "treatment": np.concatenate([np.zeros(num_controls * 60), treatment]),
    })
    return data, untreated, effect


data, untreated, truth = simulate_panel()
result = gsc(
    data, outcome_col="outcome", unit_col="unit", time_col="time",
    treatment_cols="treatment", pooled_treatment=True,
    weight_control_only=True, positive=True,
    alpha=.005, max_iter=500, ftol=1e-10, xtol=1e-10,
)
print(f"True treatment effect: {truth:.6f}")
print(f"Joint estimate:        {result.params['treatment']:.6f}")
print(f"Optimizer converged:   {result.converged}")
print("Donor weights for the treated unit:")
print(result.donor_weights["treated"].to_string())
error = result.counterfactual["treated"].to_numpy() - untreated
print(f"Untreated trajectory RMSE: {np.sqrt(np.mean(error ** 2)):.6f}")

# The two-stage estimator learns these weights using pretreatment data only.
# With never-treated donors, its exposure adjustment is simply treatment.
two_stage = isc(data, "outcome", "unit", "time", "treatment", do_sc=True,
                alpha=1e-6, tol=1e-9)
print(f"Two-stage estimate:    {two_stage.pooled_effect:.6f}")

print(result.donor_weights)

# Import plotting only when requested, and save without opening a GUI.
##import matplotlib
##matplotlib.use("Agg")

figure, axis = plt.subplots(figsize=(9, 4))
axis.plot(result.fittedvalues.index, data.loc[data.unit == "treated", "outcome"], label="Observed")
axis.plot(result.counterfactual["treated"], label="Estimated untreated outcome")
axis.plot(result.counterfactual.index, untreated, linestyle=":", label="True untreated outcome")
axis.axvline(35, color="gray", linestyle="--", label="Treatment starts")
axis.set(xlabel="Time", ylabel="Outcome", title="Simulated synthetic-control example")
axis.legend()
figure.tight_layout()
plt.show()
