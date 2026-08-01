"""Shared infrastructure for zero-inflated and hurdle models."""

from __future__ import absolute_import, print_function

import warnings

import numpy as np

from kanly.distributional_models.base import _NonnegativeDistributionalModel
from kanly.formula.data_getter import SparseDataGetter
from kanly.formula.exceptions import MissingDataException
from kanly.utils.util import dict_2_dataframe


class TwoPartModel(_NonnegativeDistributionalModel):
    """Shared data, formula, and identification handling for two-part models.

    ``exog_infl`` controls the probability assigned to the zero process. Its
    coefficients use a logit link in the current zero-inflated and hurdle
    subclasses. When omitted, a single intercept column is used.
    """

    _requires_both_outcome_parts = False

    def __init__(
            self, endog, exog, weights=None, exog_infl=None,
            exog_infl_names=None, endog_name=None, exog_names=None,
            weights_name=None, exog_infl_formula=None,
            exog_infl_term_names=None, exog_infl_term_to_indices=None,
            **model_metadata):
        """Initialize and validate count and zero-inflation design data.

        Args:
            endog: Finite non-negative response observations.
            exog: Design matrix for the conditional count mean.
            weights: Optional observation likelihood weights.
            exog_infl: Optional design matrix for the structural-zero logit.
                Defaults to an intercept-only matrix.
            exog_infl_names: Optional names for columns of ``exog_infl``.
            endog_name: Optional response name.
            exog_names: Optional names for count-equation columns.
            weights_name: Optional likelihood-weight variable name.
            exog_infl_formula: Formula used to build the inflation matrix.
            exog_infl_term_names: Terms represented by ``exog_infl``.
            exog_infl_term_to_indices: Mapping from inflation terms to columns.
            **model_metadata: Remaining metadata accepted by
                :class:`DistributionalModel`.
        """
        endog = np.asarray(endog, dtype=float)
        if endog.ndim != 1:
            raise ValueError(
                "Two-part outcomes must be one-dimensional"
            )
        if np.any(~np.isfinite(endog)) or np.any(endog < 0.0):
            raise ValueError(
                "Two-part outcomes must be finite and non-negative"
            )

        has_default_inflation = exog_infl is None
        if has_default_inflation:
            exog_infl = np.ones((len(endog), 1), dtype=float)
        else:
            exog_infl = np.asarray(exog_infl, dtype=float)
            if exog_infl.ndim == 1:
                exog_infl = exog_infl[:, None]
        if exog_infl.ndim != 2 or exog_infl.shape[0] != len(endog):
            raise ValueError(
                "exog_infl must be two-dimensional with one row per outcome"
            )
        if np.any(~np.isfinite(exog_infl)):
            raise ValueError("exog_infl must contain only finite values")

        if exog_infl_names is not None:
            exog_infl_names = [str(name) for name in exog_infl_names]
            if len(exog_infl_names) != exog_infl.shape[1]:
                raise ValueError(
                    "exog_infl_names must match the columns of exog_infl"
                )

        self.exog_infl = exog_infl
        self.exog_infl_names = exog_infl_names
        self.k_inflate = exog_infl.shape[1]
        self.exog_infl_formula = exog_infl_formula
        self.exog_infl_term_names = (
            ['Intercept']
            if exog_infl_term_names is None and has_default_inflation
            else (
                None
                if exog_infl_term_names is None
                else list(exog_infl_term_names)
            )
        )
        self.exog_infl_term_to_indices = (
            {'Intercept': np.array([0])}
            if exog_infl_term_to_indices is None and has_default_inflation
            else exog_infl_term_to_indices
        )
        super().__init__(
            endog,
            exog,
            weights=weights,
            endog_name=endog_name,
            exog_names=exog_names,
            weights_name=weights_name,
            **model_metadata,
        )
        self.identification_issues = []
        self._validate_two_part_sample()

    def _validate_two_part_sample(self):
        """Validate identification from positive-weight outcome categories."""
        active = (
            np.ones(self.nobs, dtype=bool)
            if self.weights is None
            else self.weights > 0.0
        )
        has_zero = bool(np.any(active & (self.endog == 0.0)))
        has_positive = bool(np.any(active & (self.endog > 0.0)))

        if self._requires_both_outcome_parts:
            if not has_zero or not has_positive:
                raise ValueError(
                    'Hurdle models require at least one zero and one strictly '
                    'positive outcome with positive weight'
                )
            return

        if not has_positive:
            raise ValueError(
                'Zero-inflated models are unidentified when every '
                'positive-weight outcome is zero'
            )
        if not has_zero:
            issue = (
                'No positive-weight zero outcomes were observed; the '
                'inflation probability is a boundary parameter.'
            )
            self.identification_issues.append(issue)
            warnings.warn(issue, RuntimeWarning, stacklevel=3)

    def _inference_issues(self, params):
        """Return mixture-identification issues relevant to inference."""
        del params
        return list(self.identification_issues)

    def _get_formula_prediction_designs(
            self, data, index=None, debug=False):
        """Build aligned main and zero-process prediction matrices."""
        data_frame = dict_2_dataframe(data)
        if index is not None:
            data_frame = data_frame.iloc[index]
        exog = self._get_formula_prediction_exog(
            data_frame, debug=debug
        )
        if self.exog_infl_formula is None:
            exog_infl = np.ones((len(exog), 1), dtype=float)
        else:
            inflation = SparseDataGetter.sparse_dmatrix(
                self.exog_infl_formula,
                data_frame,
                debug=debug,
                check_constant_cols=False,
                cache_intermediate=True,
                drop_1_for_FE=True,
                name='EXOG_INFL_PREDICT',
            )
            if inflation.null_rows:
                missing_rows = sorted(int(row) for row in inflation.null_rows)
                raise MissingDataException(
                    'Inflation prediction exog has missing data in rows '
                    f'{missing_rows}!'
                )
            values = inflation.values
            if hasattr(values, 'toarray'):
                values = values.toarray()
            exog_infl = np.asarray(values, dtype=float)
            names = list(inflation.column_names)
            if (self.exog_infl_names is not None
                    and names != list(self.exog_infl_names)):
                raise ValueError(
                    'Prediction inflation formula produced columns that do '
                    'not match the fitted design: expected '
                    f'{self.exog_infl_names}, got {names}'
                )
        return exog, exog_infl

    @classmethod
    def build_model_from_formula(
            cls, formula, data, index=None, exog_infl=None, debug=False,
            check_constant_cols=False, fail_on_missing=False,
            cache_intermediate=True, sum_to_n=False,
            test_formula_on_dummy=True, drop_1_for_FE=True, **model_kwargs):
        """Build count and inflation equations from Patsy-style formulas.

        Args:
            formula: Full count-model formula, including the outcome and
                optional ``$`` likelihood-weight expression.
            data: DataFrame or dict-like object containing formula variables.
            index: Optional boolean or integer row selector.
            exog_infl: Patsy-style right-hand-side formula for the
                structural-zero logit, such as ``'z1 + C(group)'``.  ``None``
                creates a single constant column.
            debug: Whether to print formula-construction diagnostics.
            check_constant_cols: Whether to remove redundant constant columns.
            fail_on_missing: Raise instead of dropping rows with missing data.
            cache_intermediate: Formula-term cache configuration.
            sum_to_n: Normalize likelihood weights to sum to retained rows.
            test_formula_on_dummy: Validate the count formula on dummy data.
            drop_1_for_FE: Apply the formula engine's drop-one categorical rule.
            **model_kwargs: Additional constructor arguments.

        Returns:
            An unfitted zero-inflated model of type ``cls`` with aligned count,
            inflation, response, and weight arrays.
        """
        if exog_infl is not None and not isinstance(exog_infl, str):
            raise TypeError(
                "exog_infl must be a Patsy-style string or None when using "
                "build_model_from_formula"
            )

        # Formula-derived inflation names replace matrix-API names.
        if exog_infl is not None:
            model_kwargs.pop('exog_infl_names', None)

        constructor_kwargs = super()._get_formula_constructor_kwargs(
            formula=formula,
            data=data,
            index=index,
            debug=debug,
            check_constant_cols=check_constant_cols,
            fail_on_missing=fail_on_missing,
            cache_intermediate=cache_intermediate,
            # Normalize only after inflation-specific missing rows are removed;
            # the final retained sample size is the appropriate target.
            sum_to_n=False,
            test_formula_on_dummy=test_formula_on_dummy,
            drop_1_for_FE=drop_1_for_FE,
        )
        inflation_kwargs = {}

        if exog_infl is not None:
            data_frame = dict_2_dataframe(data)
            inflation_obj = SparseDataGetter.sparse_dmatrix(
                exog_infl,
                data_frame,
                debug=debug,
                check_constant_cols=check_constant_cols,
                cache_intermediate=cache_intermediate,
                drop_1_for_FE=drop_1_for_FE,
                name='EXOG_INFL',
                index=index,
            )
            inflation_null_rows = inflation_obj.null_rows.copy()
            if fail_on_missing and inflation_null_rows:
                missing_rows = sorted(int(row) for row in inflation_null_rows)
                raise MissingDataException(
                    "Inflation exog has missing data in rows "
                    f"{missing_rows}!"
                )

            main_valid_rows = np.asarray(
                constructor_kwargs['valid_obs_rows'], dtype=int
            )
            keep_main_rows = ~np.isin(
                main_valid_rows, list(inflation_null_rows)
            )
            final_valid_rows = main_valid_rows[keep_main_rows]
            if len(final_valid_rows) == 0:
                raise ValueError(
                    "No valid observations remain after aligning exog_infl"
                )

            constructor_kwargs['endog'] = (
                constructor_kwargs['endog'][keep_main_rows]
            )
            constructor_kwargs['exog'] = (
                constructor_kwargs['exog'][keep_main_rows]
            )
            if constructor_kwargs['weights'] is not None:
                constructor_kwargs['weights'] = (
                    constructor_kwargs['weights'][keep_main_rows]
                )

            # Inflation values still refer to the original selected-row space.
            inflation_obj.slice_null_rows(final_valid_rows)
            exog_infl_values = inflation_obj.values
            if hasattr(exog_infl_values, 'toarray'):
                exog_infl_values = exog_infl_values.toarray()
            exog_infl_values = np.asarray(exog_infl_values, dtype=float)
            if exog_infl_values.ndim == 1:
                exog_infl_values = exog_infl_values[:, None]

            null_rows_info = constructor_kwargs['null_rows_info_dict'].copy()
            null_rows_info['EXOG_INFL'] = inflation_null_rows
            constructor_kwargs['null_rows_info_dict'] = null_rows_info
            constructor_kwargs['valid_obs_rows'] = final_valid_rows
            inflation_kwargs = {
                'exog_infl': exog_infl_values,
                'exog_infl_names': list(inflation_obj.column_names),
                'exog_infl_formula': exog_infl,
                'exog_infl_term_names': list(inflation_obj.term_names),
                'exog_infl_term_to_indices': (
                    inflation_obj.var_2_col_indices
                ),
            }

        if sum_to_n and constructor_kwargs['weights'] is not None:
            weights = constructor_kwargs['weights']
            weights_sum = weights.sum()
            if weights_sum <= 0.0:
                raise ValueError(
                    "weights must have a positive sum when sum_to_n=True"
                )
            constructor_kwargs['weights'] = (
                weights * len(constructor_kwargs['endog']) / weights_sum
            )

        supplied_kwargs = set(constructor_kwargs) | set(inflation_kwargs)
        duplicate_kwargs = supplied_kwargs.intersection(model_kwargs)
        if duplicate_kwargs:
            duplicates = ', '.join(sorted(duplicate_kwargs))
            raise TypeError(
                f'Formula construction supplies these arguments: {duplicates}'
            )

        return cls(
            **constructor_kwargs, **inflation_kwargs, **model_kwargs
        )


__all__ = ['TwoPartModel']
