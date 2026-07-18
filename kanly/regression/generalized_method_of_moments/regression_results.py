"""Result container for generalized method of moments fits."""

from __future__ import absolute_import, print_function

import numpy as np

from kanly.regression.regression_results_base import RegressionResultsBase


class GMMRegressionResults(RegressionResultsBase):
    """Regression-style summary object for fitted GMM models.

    The object stores estimated parameters, covariance, fitted moment values,
    optimizer diagnostics, the final weighting matrix, and display metadata.
    It inherits coefficient summaries and inference helpers from
    ``RegressionResultsBase``.
    """

    def __init__(
            self, model, nobs, params, cov_params, df_model, df_resid, df_t_dist, converged, fval, avg_moment_vals,
            n_iters, message, optimization_result, W, Omega, method, num_params, num_moments,
            param_names, over_identified, eigenvals, condition_number, cov_type, cov_kwds, test_level, use_t,
            specification_name=None, fit_elapsed=0, model_elapsed=0, formula=None, moment_vals=None):
        """Initialize GMM result metadata and base regression summary state.

        Args:
            model: Source ``SparseGeneralizedMethodOfMomentsModel``.
            nobs: Number of observations.
            params: Estimated parameter vector.
            cov_params: Parameter covariance matrix, or ``None`` when not
                computed yet.
            df_model: Number of estimated parameters.
            df_resid: Residual degrees of freedom.
            df_t_dist: Degrees of freedom used for t-based inference.
            converged: Whether the optimizer reported convergence.
            fval: Final GMM objective_function_ value.
            avg_moment_vals: Final sample-average moment values.
            n_iters: Total optimizer iterations.
            message: Optimizer convergence message.
            optimization_result: Raw optimizer result dictionary.
            W: Final GMM weighting matrix.
            Omega: Estimated moment covariance matrix.
            method: GMM method used.
            num_params: Number of parameters.
            num_moments: Number of moment conditions.
            param_names: Ordered parameter names.
            over_identified: Whether there are more moments than parameters.
            eigenvals: Eigenvalues of ``G'WG``.
            condition_number: Condition number of ``G'WG``.
            cov_type: Covariance type.
            cov_kwds: Covariance keyword arguments.
            test_level: Test size for confidence intervals and p-values.
            use_t: Whether inference uses t distributions.
            specification_name: Optional result label.
            fit_elapsed: Seconds spent fitting.
            model_elapsed: Seconds spent building the model.
            formula: Moment formula metadata for display.
            moment_vals: Optional target moment values.
        """

        super().__init__(nobs, params, cov_params, df_model, df_resid, df_t_dist, exog_names=param_names,
                         endog_name='{Not Applicable in GMM}', cov_type=cov_type, cov_kwds=cov_kwds,
                         test_level=test_level, use_t=use_t, alpha=0.0, l1_ratio=0.0,
                         specification_name=specification_name)

        self.model = model
        self.model_type = model.model_type
        self.converged = converged
        self.message = message
        self.n_iters = n_iters
        self.fval = fval
        self.avg_moment_vals = avg_moment_vals
        self.optimization_result = optimization_result
        self.W = W
        self.Omega = Omega
        self.method = method
        self.num_params = num_params
        self.num_moments = num_moments
        self.fit_elapsed = fit_elapsed
        self.model_elapsed = model_elapsed
        self.formula = formula
        self.moment_vals = moment_vals
        self.over_identified = over_identified
        self.eigenvals = eigenvals
        self.condition_number = condition_number

    def get_result_name(self):
        """Return the title used in printed summaries.

        Returns:
            Result name string.
        """
        return 'Generalized Method of Moments Results'

    @staticmethod
    def get_result_type():
        """Return the compact result type identifier.

        Returns:
            The string ``'GMM'``.
        """
        return 'GMM'

    def get_header_info_array(self):
        """Build header metadata for the standard regression summary.

        Returns:
            List of ``(label, value)`` pairs displayed above coefficient tables.
        """
        return [
            ('Date:', self.date),
            ('Time:  ', self.timestamp),
            ('Nobs:', self.nobs),
            ('No. Moments:', self.num_moments),
            ('No. Params:', self.num_params),
            ('Over Identified:', self.over_identified),
            ('Method:', self.method),
            ('Model:', self.model_type),
            ('Converged:', self.converged),
            ('No. Iters:', self.n_iters),
            ('Objective:', "%.4e" % self.fval),
            ('Cov Type:', self.cov_type),
            ('Model Elapsed:', "%.2fs" % self.model_elapsed),
            ('Fit Elapsed:', "%.2fs" % self.fit_elapsed),
            ('Condition No.:', "%.2e" % self.condition_number),
        ]

    def get_formula_str(self):
        """Render moment formulas for the summary footer.

        Returns:
            Multi-line formula string when formula metadata is available,
            otherwise ``None``.
        """
        ret = ''
        if self.formula is not None:
            for i, moment_formula in enumerate(self.formula):
                ret += f'\nm({i}):   '
                if isinstance(moment_formula, str) or len(moment_formula) == 1:
                    if len(moment_formula) == 1:
                        moment_formula = moment_formula[0]
                    ret += 'E[ ' + moment_formula + ' ]'
                else:
                    # Tuple formulas represent products of formula fragments,
                    # such as residual times instrument.
                    ret += 'E[ ' + ' * '.join([f'({stub})' for stub in moment_formula]) + ' ]'
                if self.moment_vals is not None:
                    ret += ' == %.2e' % self.moment_vals[i]
                else:
                    ret += " == 0"
            return ret
        else:
            return None

    def get_footer_info(self, **kwargs):
        """Build footer diagnostics for printed summaries.

        Args:
            **kwargs: Optional display arguments. ``test_level`` is passed to
                inherited inference-text rendering.

        Returns:
            String containing convergence text, inference notes, moment values,
            and any numerical conditioning warning.
        """
        ret = self.message
        ret += self.get_inference_string(test_level=kwargs.get('test_level', np.nan))
        ret += '\n\nMoments Values:\n' \
               + ', '.join([f'm({j}) = {"%.2e" % m}' for j, m in enumerate(self.avg_moment_vals)])

        # Poor conditioning of G'WG often points to weak identification or
        # nearly redundant moments, so surface it in the summary footer.
        if self.eigenvals[-1] < 1e-10 or self.condition_number > 1e3:
            numerical_warning_str = (
                '\n\nThe smallest eigenvalue of G\'WG is %.2e and the condition number of G is %.2e;'
                '\n  this may indicate numerical issues with the specification.'
            ) % (self.eigenvals[-1], self.condition_number)
        else:
            numerical_warning_str = ''

        return ret + numerical_warning_str

    def predict(self):
        """Prediction is not implemented for generic GMM results.

        Raises:
            NotImplementedError: Always, because a generic moment model does not
                necessarily define a response prediction function.
        """
        raise NotImplementedError
