from __future__ import absolute_import, print_function

import numpy as np

from kanly.regression.regression_results_base import RegressionResultsBase


class SparseNonlinearLeastSquaresRegressionResults(RegressionResultsBase):
    """Regression-results container for nonlinear least-squares fits.

    Extends the generic regression-results base with NLLS-specific diagnostics:
    trust-region status, active bounds, objective_function components, Jacobians,
    optimisation path, robust loss metadata, and elastic-net penalty details.
    """

    def __init__(self, model, params, cov_params, cov_type, cov_kwds, resid, wresid, fittedvalues, nobs, df_resid, df_model,
                 df_t_dist, rsquared, rsquared_adj, wssr, wsst, method, converged, iterations, optimality,
                 cost, scale, scale_mle, llf, penalty, objective, jac, grad, subgrad, v,
                 status, message, active_mask, is_bounded, solver_options, test_level=.05, use_t=True,
                 fit_elapsed=None, cov_elapsed=None, specification_name=None, keep_model=True, root_loss_function=None,
                 optimization_result=None, is_penalized=False, optimization_path=None, jacobian_function_callable=None,
                 l1_penalty=None, l2_penalty=None, regularize_to_values=None, alpha=None, l1_ratio=None, positive=None
                 ):
        """Initialise NLLS fit results and summary diagnostics.

        Args:
            model: Fitted ``SparseNonlinearLeastSquaresModel``.
            params: Estimated parameter vector.
            cov_params: Covariance matrix or ``None`` when not computed.
            cov_type: Covariance estimator label.
            cov_kwds: Covariance estimator options.
            resid: Raw residual vector.
            wresid: Weighted residual vector.
            fittedvalues: Fitted response values.
            nobs: Number of observations.
            df_resid: Residual degrees of freedom.
            df_model: Model degrees of freedom / number of parameters.
            df_t_dist: Degrees of freedom for t-based inference.
            rsquared: R-squared statistic.
            rsquared_adj: Adjusted R-squared statistic.
            wssr: Weighted sum of squared residuals.
            wsst: Weighted total sum of squares.
            method: Optimisation method label (e.g. ``'TR'`` or ``'CD'``).
            converged: Whether optimisation converged.
            iterations: Number of iterations performed.
            optimality: Final optimality/subgradient measure.
            cost: Final unpenalised cost.
            scale: Residual scale estimate.
            scale_mle: Maximum-likelihood residual variance estimate.
            llf: Log-likelihood under spherical normal errors.
            penalty: Final regularisation penalty.
            objective: Final penalised objective_function.
            jac: Final Jacobian, if available.
            grad: Final gradient.
            subgrad: Final subgradient values for penalised fits.
            v: Trust-region scaling vector.
            status: Integer optimisation status code.
            message: Human-readable optimiser termination message.
            active_mask: Bound-activity mask.
            is_bounded: Whether bounds were supplied and active.
            solver_options: Dict of solver settings used for the fit.
            test_level: Significance level for confidence intervals/tests.
            use_t: Whether to use t-distribution inference where available.
            fit_elapsed: Fitting time in seconds.
            cov_elapsed: Covariance-estimation time in seconds.
            specification_name: Optional fit label.
            keep_model: Whether the model is retained on the result.
            root_loss_function: Optional robust root-loss object/name.
            optimization_result: Raw optimiser result object/dict.
            is_penalized: Whether the fit used elastic-net or L2 penalties.
            optimization_path: Optional stored iteration path.
            jacobian_function_callable: Callable used for Jacobian evaluation.
            l1_penalty: L1 penalties for penalised fits.
            l2_penalty: L2 penalties for penalised fits.
            regularize_to_values: Penalty target values.
            alpha: Elastic-net alpha values.
            l1_ratio: Elastic-net L1 ratio.
            positive: Positivity mask for penalised fits.
        """

        super().__init__(nobs, params, cov_params, df_model, df_resid, df_t_dist, exog_names=model.param_names,
                         endog_name=model.endog_name, cov_type=cov_type, cov_kwds=cov_kwds, test_level=test_level,
                         use_t=use_t, alpha=alpha, l1_ratio=l1_ratio, specification_name=specification_name)

        self.model = model
        self.resid = resid
        self.wresid = wresid
        self.fittedvalues = fittedvalues
        self.rsquared = rsquared
        self.rsquared_adj = rsquared_adj
        self.wssr = wssr
        self.wsst = wsst

        self.endog_name = model.endog_name
        self.param_names = model.param_names
        self.weights_name = model.weights_name
        self.fit_elapsed = fit_elapsed
        self.cov_elapsed = cov_elapsed

        self.converged = converged
        self.iterations = iterations
        self.optimality = optimality
        self.cost = cost
        self.penalty = penalty
        self.objective = objective
        self.scale = scale
        self.scale_mle = scale_mle
        self.llf = llf

        self.jac = jac
        self.grad = grad
        self.subgrad = subgrad
        self.v = v
        self.status = status
        self.message = message
        self.solver_options = solver_options
        self.active_mask = active_mask.copy()
        self.is_bounded = is_bounded
        self.num_active_constraints = sum(active_mask != 0)

        self.root_loss_function = root_loss_function

        self.optimization_result = optimization_result
        self.is_penalized = is_penalized

        self.optimization_path = optimization_path
        self.method = method
        self.jacobian_function_callable = jacobian_function_callable

        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.regularize_to_values = regularize_to_values
        self.positive = positive

    @staticmethod
    def get_result_type():
        """Return the short result-type label used by comparison tables."""
        return 'NLLS'

    def get_header_info_array(self):
        """Build the key-value header rows for ``summary()`` output.

        Returns:
            Two-column NumPy array of display labels and values.
        """
        return np.array(
            [
                ['Date: ', self.date],
                ['Time:', self.timestamp],
                ['Weights:', self.weights_name],
                ['Nobs:', self.nobs],
                ['Df Residuals:', self.df_resid],
                ['Df Model:', self.df_model],
                ['Cost:', "%.4e" % self.cost],
                ['Scale:', "%.4e" % self.scale],
                ['LLF:', "%.4e" % self.llf],
                ['Penalty:', "%.4e" % self.penalty],
                ['Objective:', "%.4e" % self.objective],
                ['Optimality:', "%.2e" % self.optimality],
                ['R-squared:', "%.4f" % self.rsquared],
                ['Adj. R-squared:', "%.4f" % self.rsquared_adj],
                ['Model Time:', "%.2fs" % self.model.model_elapsed],
                ['Fit Time:', "%.2fs" % self.fit_elapsed],
                ['Cov Time:', "%.2fs" % self.cov_elapsed],
                ['Iterations:', self.iterations],
                ['Converged:', self.converged],
                ['Status:', self.status],
                ['Covariance Type:', self.cov_type],
                ['Active Constraints:', self.num_active_constraints],
                ['Method:', self.method],
                ['', ''],
            ]
        )

    def get_result_name(self):
        """Return the full result name printed in summaries."""
        return 'Nonlinear Least Squares Results'

    def get_footer_info(self, *args, **kwargs):
        """Build footer text for summaries, including loss and inference notes.

        Args:
            *args: Ignored; present for compatibility with base-class hooks.
            **kwargs: Forwarded to ``get_inference_string``.

        Returns:
            Footer string appended below the coefficient table.
        """

        footer_str = ''

        if self.root_loss_function is not None:
            footer_str += "\n\nLoss Function: " + str(self.root_loss_function)

        footer_str += self.get_inference_string(**kwargs)

        footer_str += "\n\nmessage: " + (self.message if self.message is not None else '')
        return footer_str

    def __str__(self):
        """Return the model summary string."""
        return self.summary()

    def __repr__(self):
        """Return the model summary string for interactive display."""
        return self.summary()