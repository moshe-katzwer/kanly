"""Run a small reproducible example from the repository root.

    python -m kanly.sandbox.generalized_synthetic_control_2.example
    python -m kanly.sandbox.generalized_synthetic_control_2.example --placebos 6
    python -m kanly.sandbox.generalized_synthetic_control_2.example --plot /tmp/gsc2.png

The data deliberately satisfy a simple donor relationship. Recovering the known
effect checks the mechanics; it does not establish validity on arbitrary data.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from kanly.sandbox.generalized_synthetic_control_2 import gsc, isc, permutation_placebos


def simulate_panel(seed=12):
    """Two controls explain the untreated outcome of one treated unit."""
    random = np.random.RandomState(seed)
    periods = np.arange(60)
    control_a = random.normal(size=len(periods))
    control_b = random.normal(size=len(periods))
    treatment = (periods >= 35).astype(float)
    effect = 3.0
    # This outcome is generated before treatment is added, so we can compare
    # the model's estimated untreated trajectory to something actually known.
    untreated = 2.0 + 0.8 * control_a + 0.4 * control_b
    data = pd.DataFrame({
        "unit": np.repeat(["control_a", "control_b", "treated"], len(periods)),
        "time": np.tile(periods, 3),
        "outcome": np.concatenate([control_a, control_b, untreated + effect * treatment]),
        "treatment": np.concatenate([np.zeros(len(periods) * 2), treatment]),
    })
    return data, untreated, effect


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placebos", type=int, default=0, help="Number of optional unit-permutation refits")
    parser.add_argument("--plot", type=Path, help="Optional destination for a saved plot")
    args = parser.parse_args(argv)
    if args.placebos < 0:
        parser.error("--placebos must be nonnegative")

    data, untreated, truth = simulate_panel()
    result = gsc(
        data, outcome_col="outcome", unit_col="unit", time_col="time",
        treatment_cols="treatment", pooled_treatment=True,
        weight_control_only=True, positive=True,
        alpha=1e-6, max_iter=500, ftol=1e-10, xtol=1e-10,
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

    if args.placebos:
        # Pooled coefficients have the same meaning/name after reassignment.
        # These estimates are a diagnostic comparison, not calibrated p-values.
        placebos = permutation_placebos(result, num_draws=args.placebos, seed=4, shifts=0)
        print("Placebo treatment estimates and convergence:")
        print(placebos.params[["treatment"]].join(placebos.diagnostics[["converged"]]).to_string())

    if args.plot:
        # Import plotting only when requested, and save without opening a GUI.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(9, 4))
        axis.plot(result.fittedvalues.index, data.loc[data.unit == "treated", "outcome"], label="Observed")
        axis.plot(result.counterfactual["treated"], label="Estimated untreated outcome")
        axis.plot(result.counterfactual.index, untreated, linestyle=":", label="True untreated outcome")
        axis.axvline(35, color="gray", linestyle="--", label="Treatment starts")
        axis.set(xlabel="Time", ylabel="Outcome", title="Simulated synthetic-control example")
        axis.legend()
        figure.tight_layout()
        figure.savefig(args.plot)
        plt.close(figure)
        print(f"Saved plot to {args.plot}")


if __name__ == "__main__":
    main()
