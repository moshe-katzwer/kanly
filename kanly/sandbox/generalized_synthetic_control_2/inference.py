"""Treatment-reassignment diagnostics, without claims of calibrated inference.

These reproduce the sandbox's circular-shift and unit-permutation experiments.
They do not impose a sharp null, invert a test, or automatically yield valid
conformal intervals. Exchangeability/stationarity assumptions require separate
justification for the application. See the README's reference comparison.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .gsc import GSCResult, fit_model
from .model import JointModel


@dataclass(frozen=True)
class PlaceboResult:
    """Labeled estimates and a record of every assignment, including failed fits.

    Columns in params name the ACTUAL recipient units in each draw. A missing
    unit-specific treatment parameter is NaN, rather than an estimate for the
    original treated unit. Shared treatment parameters can be compared by name.
    """

    params: pd.DataFrame
    diagnostics: pd.DataFrame
    shifts: pd.Series
    assignments: pd.DataFrame
    method: str
    seed: int | None


def _refit_assignments(result, shifts, permutations, *, method, seed=None, debug=False):
    """Fit each assignment and retain the information needed to audit it.

    Every returned diagnostics row corresponds to the same row in params,
    shifts, and assignments. Assignments records source labels under recipient
    columns, which makes a shuffled unit-specific coefficient interpretable.
    """
    panel = result.model.panel
    draws, diagnostics = [], []
    for draw, (shift, permutation) in enumerate(zip(shifts, permutations)):
        reassigned = panel.reassign_treatments(shift=int(shift), permutation=permutation)
        model = JointModel(reassigned, result.model.options)
        # Reuse the entire model and solver specification, including observation
        # weights and the target mask. Start from zero for each assignment: old
        # treatment coefficients can refer to different recipients after a shuffle.
        fit = fit_model(model, result.fit_options, debug=debug)
        draws.append(fit.params)
        diagnostics.append({
            "converged": fit.converged,
            "iterations": fit.optimization_result["iterations"],
            "objective": fit.optimization_result["objective_function"],
            "message": fit.optimization_result["message"],
        })
    index = pd.RangeIndex(len(draws), name="draw")
    # pandas aligns Series by parameter NAME here. Aligning by position would
    # incorrectly attach the original unit's name to a reassigned coefficient.
    return PlaceboResult(
        params=pd.DataFrame(draws, index=index),
        diagnostics=pd.DataFrame(diagnostics, index=index),
        shifts=pd.Series(shifts, index=index, name="shift", dtype=int),
        assignments=pd.DataFrame([panel.units.take(p).tolist() for p in permutations],
                                 index=index, columns=panel.units.rename("recipient")),
        method=method, seed=seed,
    )


def time_placebos(result: GSCResult, *, stride=1, debug=False):
    """Refit at nonzero circular time shifts; the observed assignment is excluded.

    With T periods, stride=1 produces T-1 refits. The shift is applied to the
    exposure schedule, including its duration and overlaps between treatments.
    """
    if not isinstance(stride, (int, np.integer)) or stride < 1:
        raise ValueError("stride must be a positive integer.")
    shifts = np.arange(1, len(result.model.panel.periods), stride)
    permutations = np.tile(np.arange(len(result.model.panel.units)), (len(shifts), 1))
    return _refit_assignments(result, shifts, permutations, method="time", debug=debug)


def permutation_placebos(result: GSCResult, *, num_draws=50, seed=0, shifts=None, debug=False):
    """Reassign units' complete treatment histories using a reproducible RNG.

    shifts=None draws circular shifts uniformly, shifts=0 only permutes units.
    A scalar integer applies the same shift to all draws; a sequence specifies
    each draw explicitly. Draws can include the original assignment or repeats.
    """
    if not isinstance(num_draws, (int, np.integer)) or num_draws < 1:
        raise ValueError("num_draws must be a positive integer.")
    # A local RNG makes this reproducible without resetting NumPy's global state.
    # RandomState preserves the old sandbox's sequence of assignment draws.
    random = np.random.RandomState(seed)
    if shifts is None:
        shifts = random.randint(len(result.model.panel.periods), size=num_draws)
    elif isinstance(shifts, (int, np.integer)):
        shifts = np.full(num_draws, shifts, dtype=int)
    else:
        shifts = np.asarray(shifts)
        if shifts.shape != (num_draws,) or not np.issubdtype(shifts.dtype, np.integer):
            raise ValueError("shifts must be an integer or a sequence of num_draws integers.")
    shifts = shifts % len(result.model.panel.periods)
    permutations = np.array([random.permutation(len(result.model.panel.units)) for _ in range(num_draws)])
    return _refit_assignments(result, shifts, permutations, method="permutation", seed=seed, debug=debug)
