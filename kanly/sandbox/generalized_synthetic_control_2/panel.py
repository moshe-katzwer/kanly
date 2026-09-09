"""Validate long-form data once and give every model array the same axes."""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


def column_names(columns, *, required=False):
    """Accept a column name or a sequence, without splitting strings into letters."""
    names = () if columns is None else ((columns,) if isinstance(columns, str) else tuple(columns))
    if required and not names:
        raise ValueError("At least one treatment column is required.")
    if len(set(names)) != len(names):
        raise ValueError("Column lists must not contain duplicates.")
    return names


@dataclass(frozen=True)
class PanelData:
    """A snapshot of a complete panel, sorted by time and unit.

    Outcomes and observation weights have shape (T, N); treatments and
    covariates have shapes (T, N, Q) and (T, N, K). The first covariate is
    always an intercept. No temporary columns are added to the caller's data.
    """

    outcomes: np.ndarray
    treatments: np.ndarray
    covariates: np.ndarray
    observation_weights: np.ndarray
    units: pd.Index
    periods: pd.Index
    treatment_names: tuple
    covariate_names: tuple

    @classmethod
    def from_frame(cls, data, outcome_col, unit_col, time_col, treatment_cols,
                   covariate_cols=None, weight_col=None):
        """Build numeric arrays from a complete long-form DataFrame.

        Missing cells are rejected rather than imputed or dropped: each unit's
        contemporaneous outcome may be a predictor for every other unit. A
        missing donor value would otherwise contaminate an entire time period.
        Treatments may be binary, fractional, signed, or continuous exposures.
        """
        treatment_names = column_names(treatment_cols, required=True)
        covariate_names = column_names(covariate_cols)
        if "Intercept" in covariate_names:
            raise ValueError("'Intercept' is reserved for the automatic intercept.")
        if unit_col == time_col:
            raise ValueError("unit_col and time_col must be different columns.")
        if not data.columns.is_unique:
            raise ValueError("Data column names must be unique.")
        columns = [outcome_col, unit_col, time_col, *treatment_names, *covariate_names]
        if weight_col is not None:
            columns.append(weight_col)
        missing = [c for c in columns if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        if data.empty or data[[unit_col, time_col]].isna().any().any():
            raise ValueError("The panel must be nonempty and unit/time labels cannot be missing.")
        if data.duplicated([time_col, unit_col]).any():
            raise ValueError("Expected exactly one observation per (time, unit).")

        units = pd.Index(data[unit_col].unique(), name=unit_col).sort_values()
        periods = pd.Index(data[time_col].unique(), name=time_col).sort_values()
        if len(units) < 2 or len(periods) < 2:
            raise ValueError("At least two units and two periods are required.")
        # After duplicate keys have been ruled out, this count is sufficient to
        # check that every possible time/unit pair is present exactly once.
        if len(data) != len(units) * len(periods):
            raise ValueError("A balanced panel is required; some (time, unit) observations are missing.")

        # Reindex every variable to the same Cartesian product. This also lets
        # a time column be used as a covariate, without duplicate pivot keys.
        index = pd.MultiIndex.from_product([periods, units], names=[time_col, unit_col])
        aligned = data.set_index([time_col, unit_col], drop=False).reindex(index)

        def matrix(column):
            # Copy before reshaping so edits to the original DataFrame cannot
            # change fitted results or the data used by later placebo refits.
            try:
                values = aligned[column].to_numpy(dtype=float, copy=True)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Column {column!r} must contain numeric values.") from exc
            if not np.isfinite(values).all():
                raise ValueError(f"Column {column!r} contains missing or nonfinite values.")
            return values.reshape(len(periods), len(units))

        outcomes = matrix(outcome_col)
        treatments = np.stack([matrix(c) for c in treatment_names], axis=2)
        # Detect exposure by nonzero entries, not a sum: positive and negative
        # doses could sum to zero even though a unit did receive treatment.
        if not np.any(treatments != 0, axis=(0, 1)).all():
            raise ValueError("Every treatment column must have at least one nonzero exposure.")
        covariates = np.stack([np.ones_like(outcomes), *[matrix(c) for c in covariate_names]], axis=2)
        weights = np.ones_like(outcomes) if weight_col is None else matrix(weight_col)
        if np.any(weights < 0) or not np.any(weights > 0):
            raise ValueError("Observation weights must be nonnegative with at least one positive value.")
        # Reassignment makes a new treatment array but can safely share these
        # read-only outcome/covariate arrays instead of copying the entire panel.
        for values in (outcomes, treatments, covariates, weights):
            values.setflags(write=False)
        return cls(outcomes, treatments, covariates, weights, units, periods,
                   treatment_names, ("Intercept", *covariate_names))

    def frame(self, values):
        """Label a (T, N) output with the original time and unit labels."""
        return pd.DataFrame(values, index=self.periods.copy(), columns=self.units.copy())

    def reassign_treatments(self, *, shift=0, permutation=None):
        """Move complete treatment histories, leaving outcomes and covariates fixed.

        permutation[j] is the source unit whose history is assigned to target j.
        A positive shift moves histories earlier with wraparound, matching the
        original sandbox. All treatment columns move together.
        """
        if not isinstance(shift, (int, np.integer)):
            raise ValueError("shift must be an integer number of periods.")
        n_units = len(self.units)
        if permutation is None:
            permutation = np.arange(n_units)
        permutation = np.asarray(permutation)
        if (permutation.shape != (n_units,) or not np.issubdtype(permutation.dtype, np.integer)
                or not np.array_equal(np.sort(permutation), np.arange(n_units))):
            raise ValueError("permutation must contain every unit position exactly once.")
        treatments = np.roll(self.treatments[:, permutation, :], -int(shift), axis=0)
        treatments.setflags(write=False)
        return replace(self, treatments=treatments)
