"""Regression checks for model algebra, alignment, weighting, and placebo refits."""

from itertools import product
import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal

from kanly.sandbox.generalized_synthetic_control.gsc import get_gsc_pred_func
from kanly.sandbox.generalized_synthetic_control_2 import (
    FitOptions, PanelData, build_model, gsc, isc, permutation_placebos, time_placebos,
)
from kanly.sandbox.generalized_synthetic_control_2.example import simulate_panel


def varied_panel():
    """An intentionally unsorted panel with multiple overlapping treatments."""
    random = np.random.RandomState(7)
    data = pd.DataFrame({
        "unit": np.tile([10, 20, 30], 8),
        "time": np.repeat(np.arange(8), 3),
        "y": random.normal(size=24),
        "x": random.normal(size=24),
        "weight": random.uniform(0.2, 2, size=24),
    })
    data["d"] = ((data.unit != 30) & (data.time >= 4)).astype(float)
    data["e"] = np.where((data.unit == 20) & (data.time >= 2), data.time / 7, 0.0)
    return data.sample(frac=1, random_state=4)


class TestPanelAndModel(unittest.TestCase):
    def test_alignment_snapshot_and_time_as_covariate(self):
        data = varied_panel()
        data["__Intercept__"] = 17.0
        original = data.copy(deep=True)
        panel = PanelData.from_frame(data, "y", "unit", "time", "d", ["time", "x"])
        assert_array_equal(panel.periods, np.arange(8))
        assert_array_equal(panel.units, [10, 20, 30])
        assert_allclose(panel.covariates[:, :, 1], np.tile(np.arange(8)[:, None], (1, 3)))
        assert_frame_equal(data, original)
        data["y"] = 0
        self.assertFalse(np.all(panel.outcomes == 0))
        self.assertFalse(panel.outcomes.flags.writeable)

    def test_invalid_panels_fail_before_optimization(self):
        data = varied_panel()
        cases = [
            (data.iloc[:-1], "balanced"),
            (pd.concat([data, data.iloc[[0]]]), "exactly one"),
            (data.assign(y=np.nan), "nonfinite"),
            (data.assign(y=np.inf), "nonfinite"),
            (data.assign(d=0), "nonzero"),
            (data.assign(unit=None), "labels"),
            (data.assign(weight=-1), "nonnegative"),
            (data.assign(weight=0), "positive"),
        ]
        for invalid, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                build_model(invalid, "y", "unit", "time", "d", weight_col="weight")
        with self.assertRaisesRegex(ValueError, "Missing columns"):
            build_model(data, "absent", "unit", "time", "d")
        with self.assertRaisesRegex(ValueError, "duplicates"):
            build_model(data, "y", "unit", "time", ["d", "d"])

    def test_model_matches_original_equations_across_options(self):
        # Fixed parameter vectors isolate the algebra from local optimization.
        # Exercise pooled/unit-specific effects and both donor/target masks.
        data = varied_panel()
        for pooled, targets_only, controls_only in product([False, True], repeat=3):
            with self.subTest(pooled=pooled, targets_only=targets_only, controls_only=controls_only):
                options = dict(pooled_treatment=pooled, fit_only_treatment_units=targets_only,
                               weight_control_only=controls_only)
                model = build_model(data, "y", "unit", "time", ["d", "e"], ["x"], "weight",
                                    weight_mode="legacy", **options)
                old_residual, old_names, *_ = get_gsc_pred_func(
                    data.copy(), "y", "unit", "time", ["d", "e"], ["x"], weight_col="weight", **options)
                self.assertEqual(model.param_names.tolist(), old_names)
                params = np.random.RandomState(19).normal(size=model.n_params)
                expected, old = old_residual(params)
                actual = model.evaluate(params)
                assert_allclose(actual.weighted_residuals, expected, atol=1e-12)
                assert_allclose(actual.donor_weights, old["sc_weight_param_mat"], atol=1e-12)
                assert_allclose(actual.treatment, old["trd_pred"], atol=1e-12)
                assert_allclose(actual.covariates, old["covar_pred"], atol=1e-12)

    def test_observation_weights_and_target_mask_have_explicit_meanings(self):
        data = varied_panel()
        model = build_model(data, "y", "unit", "time", "d", weight_col="weight")
        params = np.ones(model.n_params)
        components = model.evaluate(params)
        assert_allclose(components.weighted_residuals ** 2,
                        model.panel.observation_weights * components.residuals ** 2)
        # Untreated target 30 has no synthetic prediction, but retains its error.
        assert_allclose(components.synthetic[:, 2], 0)
        assert_allclose(components.residuals[:, 2], model.panel.outcomes[:, 2] - 1)

    def test_donor_orientation_and_counterfactual(self):
        data, untreated, _ = simulate_panel()
        model = build_model(data, "outcome", "unit", "time", "treatment", pooled_treatment=True,
                            weight_control_only=True)
        params = pd.Series(0.0, index=model.param_names)
        params["omega_treated_control_a"] = 0.8
        params["omega_treated_control_b"] = 0.4
        params["treatment"] = 3
        params["Intercept_treated"] = 2
        components = model.evaluate(params)
        assert_allclose(components.donor_weights[:, 2], [0.8, 0.4, 0])
        assert_allclose(components.counterfactual[:, 2], untreated, atol=1e-12)
        assert_allclose(components.residuals[:, 2], 0, atol=1e-12)

    def test_treatment_reassignment_preserves_joint_histories(self):
        panel = PanelData.from_frame(varied_panel(), "y", "unit", "time", ["d", "e"], "x", "weight")
        new = panel.reassign_treatments(shift=2, permutation=[2, 0, 1])
        assert_allclose(new.treatments[:6], panel.treatments[2:, [2, 0, 1], :])
        assert_allclose(new.treatments[6:], panel.treatments[:2, [2, 0, 1], :])
        assert_allclose(new.outcomes, panel.outcomes)
        assert_allclose(new.observation_weights, panel.observation_weights)
        for permutation in ([0, 0, 1], [0, 1], [0.0, 1.0, 2.0]):
            with self.assertRaisesRegex(ValueError, "permutation"):
                panel.reassign_treatments(permutation=permutation)


class TestFitsAndPlacebos(unittest.TestCase):
    def test_integer_and_reordered_labeled_starting_values(self):
        data, _, truth = simulate_panel()
        options = dict(positive=True, weight_control_only=True, pooled_treatment=True,
                       alpha=1e-6, ftol=1e-10, xtol=1e-10)
        model = build_model(data, "outcome", "unit", "time", "treatment", pooled_treatment=True)
        integer_start = np.zeros(model.n_params, dtype=int)
        fit = gsc(data, "outcome", "unit", "time", "treatment", x0=integer_start, **options)
        self.assertAlmostEqual(fit.params["treatment"], truth, places=3)
        # A scrambled Series must be aligned by name, not interpreted positionally.
        start = fit.params.sample(frac=1, random_state=7)
        refit = gsc(data, "outcome", "unit", "time", "treatment", x0=start, **options)
        assert_allclose(refit.fittedvalues, fit.fittedvalues, atol=1e-4)
        with self.assertRaisesRegex(ValueError, "labels"):
            gsc(data, "outcome", "unit", "time", "treatment", x0=start.iloc[:-1], **options)

    def test_recovers_known_effect_and_reports_raw_residuals(self):
        data, untreated, truth = simulate_panel()
        data["weight"] = 2.0
        result = gsc(data, "outcome", "unit", "time", "treatment", weight_col="weight",
                     positive=True, weight_control_only=True, pooled_treatment=True,
                     alpha=1e-6, ftol=1e-10, xtol=1e-10)
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.params["treatment"], truth, places=3)
        assert_allclose(result.counterfactual["treated"], untreated, atol=1e-3)
        assert_allclose(result.fittedvalues + result.resid, result.model.panel.outcomes)
        assert_allclose(result.weighted_resid, np.sqrt(2) * result.resid)
        assert_allclose(result.fittedvalues - result.counterfactual, result.treatment_contribution)
        self.assertTrue(result.treatment_effects.loc["control_a"].isna().all())
        self.assertTrue(np.all(result.donor_weights.to_numpy() >= 0))
        alpha = result.model.penalty_weights(result.fit_options.alpha)
        params = result.params.to_numpy()
        expected_penalty = np.sum(alpha * (0.9 * np.abs(params) + 0.1 / 2 * params ** 2))
        expected_loss = np.mean(result.weighted_resid.to_numpy() ** 2) / 2
        self.assertAlmostEqual(result.optimization_result["objective_function"], expected_loss + expected_penalty)

    def test_nonconvergence_is_exposed(self):
        data, _, _ = simulate_panel()
        result = gsc(data, "outcome", "unit", "time", "treatment", alpha=0.01, max_iter=1)
        self.assertFalse(result.converged)

    def test_log_transform_and_invalid_solver_settings_are_rejected(self):
        data = varied_panel()
        with self.assertRaises(NotImplementedError):
            gsc(data, "y", "unit", "time", "d", log_transform=True)
        for kwargs in ({"alpha": -1}, {"l1_ratio": 2}, {"max_iter": 0}, {"ftol": 0}, {"selection": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FitOptions(**kwargs)

    def test_placebo_refit_matches_explicitly_reassigned_data(self):
        # A full manual refit catches dropped observation weights, donor masks,
        # covariates, pooling, or regularization options in the placebo path.
        data = varied_panel()
        options = dict(covariate_cols="x", weight_col="weight", pooled_treatment=False,
                       fit_only_treatment_units=False, weight_control_only=True,
                       positive=True, alpha=0.05, l1_ratio=0.4, max_iter=12,
                       selection="cyclic", weight_mode="legacy")
        result = gsc(data, "y", "unit", "time", ["d", "e"], **options)
        draw = permutation_placebos(result, num_draws=1, seed=3, shifts=[2])
        explicit = data.copy()
        for recipient, source in draw.assignments.iloc[0].items():
            for treatment in ("d", "e"):
                source_values = data.loc[data.unit == source].sort_values("time")[treatment].to_numpy()
                by_time = pd.Series(np.roll(source_values, -2), index=np.arange(8))
                mask = explicit.unit == recipient
                explicit.loc[mask, treatment] = explicit.loc[mask, "time"].map(by_time)
        expected = gsc(explicit, "y", "unit", "time", ["d", "e"], **options)
        assert_allclose(draw.params.iloc[0].reindex(expected.params.index), expected.params, atol=1e-10)
        self.assertEqual(draw.diagnostics.iloc[0].converged, expected.converged)
        assert_frame_equal(data, varied_panel())

    def test_placebos_are_reproducible_and_keep_recipient_labels(self):
        data, _, _ = simulate_panel()
        result = gsc(data, "outcome", "unit", "time", "treatment", max_iter=2)
        first = permutation_placebos(result, num_draws=4, seed=3, shifts=0)
        second = permutation_placebos(result, num_draws=4, seed=3, shifts=0)
        assert_frame_equal(first.params, second.params)
        assert_frame_equal(first.assignments, second.assignments)
        for draw, assignment in first.assignments.iterrows():
            recipient = assignment.index[assignment == "treated"][0]
            self.assertTrue(np.isfinite(first.params.loc[draw, f"treatment_{recipient}"]))
            for column in first.params.columns:
                if column.startswith("treatment_") and column != f"treatment_{recipient}":
                    self.assertTrue(np.isnan(first.params.loc[draw, column]))
        self.assertIn(False, first.diagnostics.converged.tolist())
        shifted = time_placebos(result, stride=25)
        assert_array_equal(shifted.shifts, [1, 26, 51])

    def test_invalid_placebo_requests_fail(self):
        data = varied_panel()
        result = gsc(data, "y", "unit", "time", "d", max_iter=1)
        for kwargs in ({"num_draws": 0}, {"num_draws": 2, "shifts": [0]}, {"shifts": 0.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                permutation_placebos(result, **kwargs)
        with self.assertRaises(ValueError):
            time_placebos(result, stride=0)


class TestTwoStage(unittest.TestCase):
    def test_two_stage_recovers_effect_with_numeric_labels_and_shuffled_rows(self):
        data, _, truth = simulate_panel()
        data["unit"] = data.unit.map({"control_a": 101, "control_b": 202, "treated": 303})
        data["time"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(data.time, unit="D")
        data = data.sample(frac=1, random_state=7)
        original = data.copy(deep=True)
        result = isc(data, "outcome", "unit", "time", "treatment", do_sc=True,
                     alpha=1e-6, tol=1e-9)
        self.assertAlmostEqual(result.pooled_effect, truth, places=3)
        self.assertAlmostEqual(result.unit_effects[303], truth, places=3)
        self.assertEqual(result.second_stage_periods[0], pd.Timestamp("2020-02-05"))
        self.assertEqual(len(result.training_periods[303]), 35)
        assert_allclose(result.predicted_treatments[303], 0)
        assert_frame_equal(data, original)

    def test_two_stage_uses_own_or_common_pretreatment_window(self):
        data = varied_panel().drop(columns="e")
        data["d"] = (((data.unit == 10) & (data.time >= 3))
                     | ((data.unit == 20) & (data.time >= 5))).astype(float)
        for do_sc in (False, True):
            result = isc(data, "y", "unit", "time", "d", covariate_cols="x",
                         do_sc=do_sc, positive=False, alpha=0.1)
            self.assertEqual(len(result.training_periods[10]), 3)
            self.assertEqual(len(result.training_periods[20]), 5 if do_sc else 3)
            if do_sc:
                assert_allclose(result.donor_weights.loc[[10, 20]], 0)
            reconstructed = result.adjusted_treatments + result.predicted_treatments
            expected = data.pivot(index="time", columns="unit", values="d")
            assert_allclose(reconstructed[[10, 20]], expected[[10, 20]])

    def test_two_stage_requires_pretreatment_and_eligible_donors(self):
        data = varied_panel()
        for do_sc, modified, message in (
            (False, data.assign(d=1.0), "pretreatment"),
            (True, data.assign(d=(data.time >= 4).astype(float)), "No eligible donors"),
        ):
            with self.subTest(do_sc=do_sc), self.assertRaisesRegex(ValueError, message):
                isc(modified, "y", "unit", "time", "d", do_sc=do_sc)


if __name__ == "__main__":
    unittest.main()
