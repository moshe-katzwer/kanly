from __future__ import absolute_import, print_function

import numpy as np
import pandas as pd

from kanly.regression.linear_models.penalized.dual_gap import elastic_net_duality_gap
from kanly.regression.regression_results_base import RegressionResultsBase


class SparsePenalizedLinearRegressionResults(RegressionResultsBase):
    """Results container for sparse elastic-net linear model fits.

    Stores coefficients, convergence diagnostics, objective_function components, selected
    features, and prediction helpers.  Inference fields are intentionally filled
    with NaN because penalized coefficients are biased estimators."""

    def __init__(self, nobs, params, model, method, fittedvalues, resid, positive, normalize, fit_intercept, l1_ratio, alpha,
                 l1_penalties, l2_penalties,
                 apply_scaling, fit_elapsed, rsquared, coef_, intercept_, iters, converged, x_error, f_error, g_error, message,
                 cost, ssr, penalty, objective_function_, objective_function,
                 solver_settings, relaxation_parameter, specification_name=None, keep_model=True):
        """Initialize elastic-net fit results.

        Args:
            nobs: Number of observations.
            params: Full parameter vector including intercept when present.
            model: Source ``SparsePenalizedLinearModel``.
            method: Human-readable method label.
            fittedvalues: Fitted response vector.
            resid: Residual vector.
            positive: Non-negativity constraints used in fitting.
            normalize: Whether std-dev penalty scaling was used (``normalize`` arg).
            fit_intercept: Whether an intercept was fitted.
            l1_ratio: Elastic-net mixing value(s).
            alpha: Penalty strength(s).
            apply_scaling: Whether coefficient scaling correction was applied.
            fit_elapsed: Solver elapsed time.
            rsquared: In-sample score/R-squared.
            coef_: Slope coefficient vector.
            intercept_: Fitted intercept.
            iters: Number of coordinate-descent iterations.
            converged: Whether convergence criteria were met.
            x_error: Final coefficient-change error.
            f_error: Final objective_function-change error.
            g_error: Final subgradient error.
            message: Solver termination message.
            cost: Raw least-squares cost.
            ssr: Final sum of squared residuals.
            penalty: Final elastic-net penalty.
            objective_function_: Final penalized objective_function value.
            objective_function: Objective function callable
            solver_settings: Dict of solver settings.
            relaxation_parameter: Optional relaxed-fit penalty factor.
            specification_name: Optional result label.
            keep_model: Whether to retain the model on the result."""

        exog_names = np.array(model.exog_names)
        if fit_intercept:
            exog_names = np.hstack((['Intercept'], exog_names))

        super().__init__(nobs, params, None, df_model=np.nan, df_resid=np.nan, df_t_dist=np.nan, exog_names=exog_names,
                         endog_name=model.endog_name, cov_type='N/A', cov_kwds=None, test_level=None, use_t=True,
                         alpha=alpha, l1_ratio=l1_ratio, specification_name=specification_name)

        self.l1_penalties = l1_penalties
        self.l2_penalties = l2_penalties

        self.set_properties_from_model(model, keep_model)
        self.exog_names = exog_names  # overwritten in model copy

        self.method = method
        self.apply_scaling = apply_scaling
        self.fittedvalues = fittedvalues
        self.resid = resid
        self.weights_name = self.model.weights_name
        self.rsquared = rsquared
        self.rsquared_adj = np.nan
        self.positive = positive
        self.normalize = normalize
        self.fit_intercept = fit_intercept
        self.fit_elapsed = fit_elapsed
        self.solver_settings = solver_settings

        # Penalized estimates do not get classical standard errors here; fill
        # inference fields with NaN so table formatting still works.
        for attr in ('pvalues', 'tvalues', 'bse'):
            setattr(self, attr, pd.Series([np.nan] * self.params.shape[0], index=self.params.index))

        self.coef_ = coef_.copy()
        self.intercept_ = intercept_

        self.iters = iters
        self.converged = converged
        self.x_error = x_error
        self.f_error = f_error
        self.g_error = g_error
        self.message = message

        self.ssr = ssr
        self.cost = cost
        self.penalty = penalty

        self.objective_function_ = objective_function_
        self.objective_function = objective_function

        self.relaxation_parameter = relaxation_parameter
        self.dual_gap_ = None

    def dual_gap(self, intercept_=None, coef_=None):
        has_args = coef_ is not None or intercept_ is not None
        if has_args or self.dual_gap_ is None:
            coef_ = self.coef_ if coef_ is None else coef_
            intercept_ = self.intercept_ if intercept_ is None else intercept_
            dual_gap_ = elastic_net_duality_gap(
                self.model.exog,
                self.model.endog,
                intercept_,
                coef_,
                self.l1_penalties,
                self.l2_penalties,
                self.fit_intercept,
                self.model.weights,
                self.positive,
            )
            if not has_args and self.dual_gap_ is None:
                self.dual_gap_ = dual_gap_
        else:
            dual_gap_ = self.dual_gap_
        return dual_gap_

    @staticmethod
    def get_result_type():
        """Return the short result type label used in comparison tables.

        Returns:
            String ``'EN_sk'``."""
        return 'EN_sk'

    def get_header_info_array(self):
        """Build key-value rows for the summary header.

        Returns:
            Two-column NumPy array of display labels and formatted values."""
        return np.array(
            [
                ['Date:', self.date],
                ['Time:', self.timestamp],
                ['Method:', self.method],
                ['Nobs:', self.nobs],
                ['Params:', len(self.params)],
                ['Score:', "%.4f" % self.rsquared],
                ['SSR:', "%.4e" % self.ssr],
                ['Penalty:', "%.4e" % self.penalty],
                ['Objective:', "%.4e" % self.objective_function_],
                ['Weights:', self.weights_name],
                ['Converged:', self.converged],
                ['Iters:', self.iters],
                ['Max Iter:', self.solver_settings['max_iter']],
                ['', ''],
                ['', ''],
                ['|dx|:', "%.2e" % self.x_error],
                ['|dF/F|', "%.2e" % self.f_error],
                ['max|subgrad|:', "%.2e" % self.g_error],
                ['alpha:', self.dict_args_to_str(self.alpha)],
                ['l1_ratio:', self.dict_args_to_str(self.l1_ratio)],
                ['fit_intercept:', self.fit_intercept],
                ['normalize:', self.normalize],
                ['positive:', self.dict_args_to_str(self.positive)],
                ['scaled:', self.apply_scaling],
                ['relaxation:', '' if self.relaxation_parameter is None else
                    '%.3e' % self.relaxation_parameter],
                ['active_set:', self.solver_settings['active_set']],
                ['selection:', self.solver_settings['selection']],
                ['Tolerance:', "%.2e" % self.solver_settings['xtol']],
                ['Model Time:', "%.2fs" % self.model_elapsed],
                ['Fit Time:', "%.2fs" % self.fit_elapsed],
            ]
        )

    @staticmethod
    def dict_args_to_str(arg):
        """Format scalar or heterogeneous penalty/configuration arguments for summaries.

        Args:
            arg: Scalar, dict, list, array, or Series argument used in the fit.

        Returns:
            Compact string for summary display, or ``'<heterogeneous>'`` when values vary."""
        if isinstance(arg, (float, int, bool, str)):
            if isinstance(arg, float):
                return "%.2e" % arg
            return str(arg)
        elif isinstance(arg, (dict, list, np.ndarray, pd.Series)):
            if isinstance(arg, dict):
                arg = arg.values()
            arg = list(set(arg))
            if len(arg) == 1:
                return str(arg[0])
            else:
                return '<heterogeneous>'

    def get_result_name(self):
        """Return the full result name printed by summaries.

        Returns:
            ``'Penalized Linear Model Results'``."""
        return 'Penalized Linear Model Results'

    def get_footer_info(self, *args, **kwargs):
        """Return footer text describing solver termination.

        Args:
            *args: Ignored compatibility arguments.
            **kwargs: Ignored compatibility keyword arguments.

        Returns:
            Solver message string."""
        return self.message

    def __str__(self):
        """Return the formatted result summary string.

        Returns:
            Multi-line summary string."""
        return self.summary()

    def __repr__(self):
        """Return the formatted result summary for interactive display.

        Returns:
            Multi-line summary string."""
        return self.summary()

    def predict(self, data=None, params=None, index=None, debug=False, ignore_column_mismatch=False):
        """Predict fitted values using stored or supplied parameters.

        Args:
            data: Optional new data for formula-based prediction.
            params: Optional parameter vector, Series, list, ndarray, or dict.  Missing
                coefficients default to zero.
            index: Optional row subset for new data.
            debug: Whether to print prediction parsing diagnostics.
            ignore_column_mismatch (bool): When ``True``, allow prediction when
                the out-of-sample design has fewer columns than the training
                model (e.g. missing fixed-effect levels). Forwarded to the
                penalized model's ``predict``.

        Returns:
            Fitted values for stored data or predictions for new data."""

        if params is None and data is None:
            return self.fittedvalues.copy()

        if params is None:
            params = self.params.copy()

        if isinstance(params, (pd.Series, np.ndarray, list)):
            if isinstance(params, (np.ndarray, list)):
                params = pd.Series(index=self.params.index, data=params)

            # Build a full coefficient vector aligned to the training design;
            # omitted coefficients are treated as zeros for support-restricted predictions.
            if 'Intercept' in params.index:
                intercept = params['Intercept']
            else:
                intercept = 0.0
            coef = pd.Series(index=self.params.index[int(self.fit_intercept):], data=0.0)
            for i in coef.index:
                if i in params.keys() and i != 'Intercept':
                    coef[i] = params[i]

        elif isinstance(params, dict):
            intercept = params.get('Intercept', 0.0)
            coef = pd.Series(index=self.params.index[int(self.fit_intercept):], data=0.0)
            for i in coef.index:
                if i in params.keys():
                    coef[i] = params[i]
        else:
            raise Exception(f"Unsupported params type {type(params)}!")

        return self.model.predict(intercept, coef, data=data, index=index, debug=debug,
                                  ignore_column_mismatch=ignore_column_mismatch)
