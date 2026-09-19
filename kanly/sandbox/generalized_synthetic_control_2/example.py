import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from kanly.sandbox.generalized_synthetic_control_2 import gsc, isc, permutation_placebos
from kanly.api import simulate_sarima
from tqdm import tqdm


def simulate_panel(n=60, seed=12, num_controls=10, effect=0.75):
    """controls explain the untreated outcome of one treated unit."""
    random = np.random.RandomState(seed)
    periods = np.arange(n)
    control = random.normal(size=(len(periods), num_controls//2))
    control = control @ random.rand(*(num_controls//2, num_controls))
    for j in range(num_controls):
        control[:, j] += simulate_sarima(n=n, ar=[.6, .15], seed=random.randint(0, 10000))
        control[:, j] += np.arange(n) / n * 2 * (random.rand() - .5)
    treatment = (periods >= 4 / 5 * n).astype(float)
    # This outcome is generated before treatment is added, so we can compare
    # the model's estimated untreated trajectory to something actually known.
    beta = random.rand(num_controls) / num_controls
    beta[np.random.choice(np.arange(num_controls), int(1 / 2 * num_controls), replace=False)] = 0
    untreated = 2.0 + control.dot(beta) + 0.15 * random.normal(size=len(periods))
    data = pd.DataFrame({
        "unit": np.repeat([f"control_{j}" for j in range(num_controls)] + ["treated"], len(periods)),
        "time": np.tile(periods, num_controls + 1),
        "outcome": np.concatenate([control.T.flatten(), untreated + effect * treatment]),
        "treatment": np.concatenate([np.zeros(num_controls * n), treatment]),
    })
    return data, untreated, effect


# data, untreated, truth = simulate_panel()
# result = gsc(
#     data, outcome_col="outcome", unit_col="unit", time_col="time",
#     treatment_cols="treatment", pooled_treatment=True,
#     weight_control_only=True, positive=True,
#     alpha=.005, max_iter=500, ftol=1e-10, xtol=1e-10,
# )
# print(f"True treatment effect: {truth:.6f}")
# print(f"Joint estimate:        {result.params['treatment']:.6f}")
# print(f"Optimizer converged:   {result.converged}")
# print("Donor weights for the treated unit:")
# print(result.donor_weights["treated"].to_string())
# error = result.counterfactual["treated"].to_numpy() - untreated
# print(f"Untreated trajectory RMSE: {np.sqrt(np.mean(error ** 2)):.6f}")
# 
# # The two-stage estimator learns these weights using pretreatment data only.
# # With never-treated donors, its exposure adjustment is simply treatment.
# two_stage = isc(data, "outcome", "unit", "time", "treatment", do_sc=True,
#                 alpha=1e-6, tol=1e-9)
# print(f"Two-stage estimate:    {two_stage.pooled_effect:.6f}")
# 
# print(result.donor_weights)

# Import plotting only when requested, and save without opening a GUI.
##import matplotlib
##matplotlib.use("Agg")

# figure, axis = plt.subplots(figsize=(9, 4))
# axis.plot(result.fittedvalues.index, data.loc[data.unit == "treated", "outcome"], label="Observed")
# axis.plot(result.counterfactual["treated"], label="Estimated untreated outcome")
# axis.plot(result.counterfactual.index, untreated, linestyle=":", label="True untreated outcome")
# axis.axvline(35, color="gray", linestyle="--", label="Treatment starts")
# axis.set(xlabel="Time", ylabel="Outcome", title="Simulated synthetic-control example")
# axis.legend()
# figure.tight_layout()
# plt.show()

effect = 0.5
num_controls = 40

counter = 0
for alpha in [1e-6, 1e-4, .01, .01, .05, .4]:

    res = []
    for t in tqdm(range(150)):
        counter += 1
        data, untreated, truth = simulate_panel(n=30, effect=effect, num_controls=num_controls, seed=counter)
        result = gsc(
            data, outcome_col="outcome", unit_col="unit", time_col="time",
            treatment_cols="treatment", pooled_treatment=True,
            weight_control_only=True, positive=True,
            alpha=alpha, l1_ratio=0, max_iter=500, ftol=1e-10, xtol=1e-10,
        )
        res.append(result.params['treatment'])

    plt.hist(res, label=alpha, density=True, alpha=.5)
    print(f'\n{alpha:.5f}, {np.sqrt(np.mean((np.array(res) - effect) ** 2)):.4f}, '
          f'{np.abs(np.mean(np.array(res) - effect)):.4f}, {np.std(res):.4f}')

plt.legend(loc='best')
plt.axvline(effect, color="k")
plt.show()

# print(f"True treatment effect: {truth:.6f}")
# print(f"Joint estimate:        {result.params['treatment']:.6f}")
# print(f"Optimizer converged:   {result.converged}")
# print("Donor weights for the treated unit:")
# print(result.donor_weights["treated"].to_string())
# error = result.counterfactual["treated"].to_numpy() - untreated
# print(f"Untreated trajectory RMSE: {np.sqrt(np.mean(error ** 2)):.6f}")
#
# # The two-stage estimator learns these weights using pretreatment data only.
# # With never-treated donors, its exposure adjustment is simply treatment.
# two_stage = isc(data, "outcome", "unit", "time", "treatment", do_sc=True,
#                 alpha=1e-6, tol=1e-9)
# print(f"Two-stage estimate:    {two_stage.pooled_effect:.6f}")
#
# print(result.donor_weights)
#
# # Import plotting only when requested, and save without opening a GUI.
# ##import matplotlib
# ##matplotlib.use("Agg")
#
# figure, axis = plt.subplots(ncols=2, figsize=(9, 4))
# axis[0].plot(result.fittedvalues.index, data.loc[data.unit == "treated", "outcome"], label="Observed")
# axis[0].plot(result.counterfactual["treated"], label="Estimated untreated outcome")
# axis[0].plot(result.counterfactual.index, untreated, linestyle=":", label="True untreated outcome")
# axis[0].axvline(35, color="gray", linestyle="--", label="Treatment starts")
# axis[0].set(xlabel="Time", ylabel="Outcome", title="Simulated synthetic-control example")
# axis[0].legend()
# axis[1].plot(result.fittedvalues.index, data.loc[data.unit == "treated", "outcome"].values - result.counterfactual["treated"],
#              label="Difference")
# axis[1].axhline(0, color="black")
# axis[1].axhline(effect, color="green", ls=':', label="Effect")
# axis[1].axvline(35, color="gray", linestyle="--", label="Treatment starts")
# axis[1].set(xlabel="Time", ylabel="Outcome", title="Simulated synthetic-control example")
# axis[1].legend()
# figure.tight_layout()
# plt.show()