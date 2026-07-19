"""Result container for fitted quantile regression models.

``SparseQuantileRegressionResults`` extends ``RegressionResultsBase`` and
stores all fit metadata — quantile level, convergence state, cost values,
IRLS weights, and timing — together with the standard result API
(``summary()``, ``summary_df()``, ``predict()``, ``params``, ``bse``, etc.)
inherited from the base class.
"""
from __future__ import absolute_import, print_function

import numpy as np

from kanly.regression.linear_models.quantile_regression.constants import QR_COV_TYPE_BOOTSTRAP
from kanly.regression.regression_results_base import RegressionResultsBase


class SparseQuantileRegressionResults(RegressionResultsBase):
    """Fitted quantile regression result container.

    Stores the IRLS solution together with fit metadata (tau, cost,
    convergence, timing) and formats the results summary table and footer.
    Covariance parameters (and therefore standard errors, p-values, and
    confidence intervals) are set after the fit via ``set_cov_params`` or the
    bootstrap routine in ``model.py``.

    Key attributes (beyond those from ``RegressionResultsBase``):
        tau (float): Quantile level at which the model was estimated.
        resid (ndarray): Final residuals y − Xβ̂, shape (n,).
        fittedvalues (ndarray): Final fitted values Xβ̂, shape (n,).
        weights (ndarray): Final IRLS weights ψ(r)/r, shape (n,).
        cost (float): Smoothed objective_function value at convergence (halved).
        true_cost (float): Exact check-function cost at convergence (halved).
        pseudo_rsquared (float): Koenker-Machado pseudo-R².
        fit_elapsed (float): Wall time in seconds for the IRLS fit.
        cov_elapsed (float): Wall time in seconds for covariance estimation.
        converged (bool): Whether the IRLS loop met a convergence criterion.
        iterations (int): Number of IRLS iterations performed.
        error (float): Final convergence error value.
        line_search (bool): Whether grid line search was enabled.
        compute_cov (bool): Whether covariance was requested and computed.
        message (str): Human-readable convergence message.
        residual_inclusion (bool): Whether IV residual powers were included.
        method (str): Estimation method label (``'IRLS'``).
    """

    def __init__(self, model, tau, exog_names, params, cov_params, cov_type, cov_kwds, resid, fittedvalues, weights,
                 pseudo_rsquared, cost, true_cost, nobs, df_resid,
                 df_model, df_t_dist, converged, iterations, error, line_search, compute_cov, message, test_level=.05,
                 use_t=True, fit_elapsed=None, residual_inclusion=False,
                 cov_elapsed=0.0, specification_name=None, keep_model=True, method=None):
        """Initialise the quantile regression result container.

        Args:
            model (SparseQuantileRegressionModel): The fitted model object.
            tau (float): Quantile level.
            exog_names (list of str): Column names of the (possibly
                instrumented) design matrix.
            params (ndarray): Estimated coefficient vector, shape (p,).
            cov_params (ndarray or None): Covariance matrix of the
                coefficients, shape (p, p).  May be None initially; set
                later via ``set_cov_params``.
            cov_type (str): Covariance type string (e.g. ``'IID'``,
                ``'ROBUST'``, ``'BOOTSTRAP'``).
            cov_kwds (dict): Keyword arguments used for covariance estimation.
            resid (ndarray): Residuals y − Xβ̂, shape (n,).
            fittedvalues (ndarray): Fitted values Xβ̂, shape (n,).
            weights (ndarray): Final IRLS weights, shape (n,).
            pseudo_rsquared (float): Koenker-Machado pseudo-R².
            cost (float): Smoothed objective_function value at convergence.
            true_cost (float): Exact check-function cost at convergence.
            nobs (int): Number of observations.
            df_resid (int): Residual degrees of freedom (n − p).
            df_model (int): Model degrees of freedom (p − has_intercept).
            df_t_dist (int or float): Degrees of freedom for the t-distribution
                used in inference.
            converged (bool): Whether the IRLS loop converged.
            iterations (int): Number of IRLS iterations.
            error (float): Final convergence error.
            line_search (bool): Whether line search was enabled.
            compute_cov (bool): Whether covariance was successfully computed.
            message (str): Convergence message.
            test_level (float): Significance level for CIs and p-values.
                Defaults to 0.05.
            use_t (bool): Use t-distribution for inference.  Defaults to
                True.
            fit_elapsed (float or None): Wall time in seconds for the fit.
            residual_inclusion (bool): Whether IV residual powers were used.
                Defaults to False.
            cov_elapsed (float): Wall time for covariance estimation.
                Defaults to 0.0.
            specification_name (str or None): Label for the summary footer.
            keep_model (bool): If True, store a reference to ``model``.
                Defaults to True.
            method (str or None): Estimation method label (``'IRLS'``).
        """
        super().__init__(nobs, params, cov_params, df_model, df_resid, df_t_dist, exog_names=exog_names,
                         endog_name=model.endog_name, cov_type=cov_type, cov_kwds=cov_kwds, test_level=test_level,
                         use_t=use_t, alpha=0.0, l1_ratio=0.0, specification_name=specification_name)

        self.tau = tau
        self.resid = resid
        self.fittedvalues = fittedvalues
        self.weights = weights
        self.cost = cost
        self.true_cost = true_cost
        self.pseudo_rsquared = pseudo_rsquared
        self.fit_elapsed = fit_elapsed
        self.cov_elapsed = cov_elapsed
        self.converged = converged
        self.iterations = iterations
        self.error = error
        self.line_search = line_search
        self.compute_cov = compute_cov
        self.message = message

        self.set_properties_from_model(model, keep_model)
        self.exog_names = exog_names  # overwrite
        self.residual_inclusion = residual_inclusion
        self.method = method

    # # Dead-code stub — Python uses the second definition of predict below.
    # # This NotImplementedError version is an artefact and is never reached.
    # def predict(self, data=None, params=None, index=None, ignore_column *args, **kwargs):
    #     """Dead-code stub — overridden by the concrete predict definition immediately below.
    #
    #     This definition is never called; Python resolves the name ``predict``
    #     to the second definition in the class body.  It is retained as an
    #     artefact of the class's development history.
    #     """
    #     raise NotImplementedError

    def predict(self, data=None, params=None, index=None, debug=False, ignore_column_mismatch=False):
        """Return in-sample fitted values or compute out-of-sample predictions.

        When both ``data`` and ``params`` are None, returns a copy of the
        stored ``fittedvalues`` array (fast path, no re-evaluation).  For
        out-of-sample prediction, the model's linear predictor is re-evaluated
        on ``data`` with the given ``params`` (or the estimated parameters if
        ``params`` is None).

        Note: The first ``predict`` stub above raises ``NotImplementedError``
        and is never called — Python resolves the name to this concrete
        definition.  It is retained as a documentation artefact.

        Args:
            data (DataFrame or None): Out-of-sample data.  If None, in-sample
                fitted values are returned.
            params (ndarray or None): Coefficient vector to use for prediction.
                Defaults to ``self.params``.
            index (array-like or None): Row index for out-of-sample alignment.
            debug (bool): If True, pass debug flag to the model predictor.
                Defaults to False.
            ignore_column_mismatch (bool): When ``True``, allow prediction when
                the out-of-sample design has fewer columns than the fitted
                model (e.g. missing fixed-effect levels). Forwarded to the
                model's ``predict``.

        Returns:
            ndarray: Predicted values, shape (n,).
        """
        if params is None and data is None:
            return self.fittedvalues.copy()

        if params is None:
            params = self.params.copy()

        return self.model.predict(params, data=data, index=index, debug=debug, ignore_column_mismatch=ignore_column_mismatch)

    def get_result_name(self):
        """Return the display name for the summary table header.

        Returns:
            str: ``'Quantile Regression Results'``.
        """
        return 'Quantile Regression Results'

    def get_result_type(self):
        """Return the short result type identifier.

        Returns:
            str: ``'QR'``.
        """
        return 'QR'

    def get_header_info_array(self):
        """Build the 2-column label/value array for the summary header block.

        Each row is a ``[label, value]`` pair displayed at the top of
        ``summary()``.  Includes date/time, elapsed times, quantile level,
        pseudo-R², convergence state, cost values, and covariance type.

        Returns:
            ndarray: Shape (n_rows, 2) array of string-formatted metadata.
        """
        return np.array([
            ['Date:', self.date],
            ['Time:  ', self.timestamp],
            ['Model Elapsed:', '%.2f s' % self.model_elapsed],
            ['Fit Elapsed:', '%.2f s' % self.fit_elapsed],
            ['Cov Elapsed:', '%.2f s' % self.cov_elapsed],
            ['Quantile:', np.round(self.tau, 4)],
            ['Pseudo-rsquared:', np.round(self.pseudo_rsquared, 4)],
            ['Method:', 'IRLS'],
            # ['Intercept:', self.has_intercept],
            # ['Implicit Intercept:', self.has_implicit_constant],
            ['Covariance Type:', self.cov_type],
            ['No. Obs.', self.nobs],
            ['Df Residuals:', self.df_resid],
            ['Df Model:', self.df_model],
            # [f'R-squared{uncentered}:', np.round(self.rsquared, 3)],
            # [f'Adj. R-squared{uncentered}:', np.round(self.rsquared_adj, 3)],
            # ['F-statistic:', ('%.3e' % self.fvalue) if self.fvalue > 100_000 else ("%.2f" % self.fvalue)],
            # ['Prob (F-statistic):', ('%.3f' % self.f_pvalue) if self.f_pvalue > .001 else '<.001'],
            # ['Log-Likelihood:', "-" if self.is_iv else "%.4f" % self.llf],
            # ['AIC:', "-" if self.is_iv else "%.2f" % self.aic],
            # ['BIC:', "-" if self.is_iv else "%.2f" % self.bic],
            ['Converged:', self.converged],
            ['Iterations:', self.iterations],
            ['Error:', "%.2e" % self.error],
            ['Cost:', "%.4e" % self.cost],
            ['True Cost:', "%.4e" % self.true_cost],
            ['Line Search:', self.line_search],
        ])

    def get_footer_info(self, test_level=None):
        """Build the footer string appended below the coefficient table.

        Includes the IRLS convergence message, inference distribution string,
        and — when the model was estimated with IV — warnings about the
        experimental nature of IV quantile regression and about the
        unreliability of non-bootstrap inference for IV.

        Args:
            test_level (float or None): Significance level for the inference
                string.  If None, uses the level stored at fit time.

        Returns:
            str: Multi-line footer string.
        """
        ret = self.message

        if self.is_iv and self.did_compute_var_covar():
            ret += f"\n\nIV residual_inclusion={self.residual_inclusion}"
            ret += '\n\n*** Note: IV is complicated in non-linear settings,' \
                   '\n    experts only! ***\n'
            if QR_COV_TYPE_BOOTSTRAP not in self.cov_type.upper():
                ret += "\n*** Note: NON-BOOTSTRAP INFERENCE MAY BE " \
                       "\n    UNRELIABLE FOR INSTRUMENTAL VARIABLES!! ***"

        ret += self.get_inference_string(test_level=test_level)
        return ret
