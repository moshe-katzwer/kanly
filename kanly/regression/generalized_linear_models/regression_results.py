from __future__ import absolute_import, print_function

import numpy as np
import pandas as pd

from kanly.regression.generalized_linear_models.constants import BOOTSTRAP
from kanly.regression.generalized_linear_models.families import _get_family_and_link
from kanly.regression.generalized_linear_models.marginal_effects import _get_marginal_effects
from kanly.regression.regression_results_base import RegressionResultsBase


class SparseGLMRegressionResults(RegressionResultsBase):
    """Results container for sparse generalized linear model fits."""

    def __init__(self, _params,  _cov_params, model, exog_names, endog_name,
                 nobs, df_model, df_resid, family, link, alpha, l1_ratio,
                 penalize_scale, fit_intercept, normalize, converged,
                 num_iter, abs_error, rel_error, llf, scale, irls_weights, normalized_cov_params, edf,
                 endog_predicted,
                 test_level, use_t, llnull, deviance, pearson_chi2, convergence_path, start_params,
                 max_iter, _instrument_params, residual_inclusion, resid,
                 g_prime, lin_pred, cov_type, cov_kwds, var_weights_name, fit_time, cov_time,
                 method, keep_model=True, specification_name=None):
        """Initialize GLM regression results and summary diagnostics.

        Args:
            _params: Estimated coefficient vector.
            _cov_params: Estimated covariance matrix, or ``None`` when omitted.
            model: Fitted ``SparseGeneralizedLinearModel``.
            exog_names: Coefficient names after intercept/IV expansion.
            endog_name: Response variable name.
            nobs: Number of observations.
            df_model: Model degrees of freedom.
            df_resid: Residual degrees of freedom.
            family: Fitted GLM family instance.
            link: Fitted link instance.
            alpha: Elastic-net regularization strength.
            l1_ratio: L1 share of regularization.
            penalize_scale: Whether penalties were scaled by dispersion.
            fit_intercept: Whether an intercept was fit separately.
            normalize: Whether predictors were normalized for penalization.
            converged: Whether fitting converged.
            num_iter: Number of optimizer iterations.
            abs_error: Final absolute convergence error.
            rel_error: Final relative convergence error.
            llf: Fitted log-likelihood.
            scale: Estimated scale/dispersion.
            irls_weights: Final IRLS weights.
            endog_predicted: Fitted mean response values.
            test_level: Test significance level.
            use_t: Whether inference uses t critical values.
            llnull: Null-model log-likelihood.
            deviance: Fitted model deviance.
            pearson_chi2: Pearson chi-squared statistic.
            convergence_path: Optional per-iteration diagnostics.
            start_params: Starting coefficient vector.
            max_iter: Maximum iterations requested.
            _instrument_params: First-stage IV coefficients, if any.
            residual_inclusion: Whether residual inclusion was used for IV GLM.
            resid: Response residuals.
            g_prime: Link derivative at fitted means.
            lin_pred: Final linear predictor.
            cov_type: Covariance estimator type.
            cov_kwds: Covariance estimator options.
            var_weights_name: Optional variance-weight variable name.
            fit_time: Fit elapsed time in seconds.
            cov_time: Covariance elapsed time in seconds.
            method: Optimization method label.
            keep_model: Whether to retain the model on the result.
            specification_name: Optional display label for the fit.
        """

        super().__init__(nobs, _params, _cov_params, df_resid=df_resid, df_model=df_model, df_t_dist=df_resid,
                         exog_names=exog_names, endog_name=endog_name,
                         cov_type=cov_type, cov_kwds=cov_kwds, test_level=test_level, use_t=use_t, alpha=alpha,
                         l1_ratio=l1_ratio, specification_name=specification_name)

        self.is_gam = model.is_gam
        self.set_properties_from_model(model, keep_model)
        self.var_weights_name = var_weights_name

        self.family = family
        self.link = link

        self.scale = scale
        self.llf = llf

        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.penalize_scale = penalize_scale
        self.method = method
        self.alpha = alpha
        self.l1_ratio = l1_ratio

        self.converged = converged
        self.num_iter = num_iter
        self.abs_error = abs_error
        self.rel_error = rel_error

        self.endog_predicted = endog_predicted.copy()
        self.fittedvalues = self.endog_predicted
        self.irls_weights = irls_weights
        self.normalized_cov_params = normalized_cov_params
        self.edf = edf
        self.resid = resid

        self.llnull = llnull
        self.pseudo_rsquared = 1.0 - self.llf / self.llnull
        self.deviance = deviance
        self.pearson_chi2 = pearson_chi2

        self.convergence_path = convergence_path
        self.start_params = start_params
        self.max_iter = max_iter

        self._instrument_params = _instrument_params
        if model.instruments is not None:
            self.instrument_params = pd.DataFrame(index=list(model.instrument_names),
                                                  columns=list(model.exog_names[:_instrument_params.shape[1]]),
                                                  data=_instrument_params)

        self.is_iv = model.instruments is not None
        self.residual_inclusion = residual_inclusion

        self.g_prime = g_prime
        self.lin_pred = lin_pred

        self.fit_elapsed = fit_time
        self.cov_elapsed = cov_time

        self.exog_names = exog_names

        self.loglike = self.model.get_log_likelihood_function(self.family, self.link)
        self.loglike_obs = self.model.get_log_likelihood_function_obs(self.family, self.link)

    @staticmethod
    def get_result_type():
        """Return the short result type used in comparison tables."""
        return 'GLM'

    def get_header_info_array(self):
        """Build the summary header rows for GLM results.

        Returns:
            Two-column NumPy array of display labels and values.
        """
        return np.array(
            [
                ['Date:', self.date],
                ['Time:', self.timestamp],
                ['Family:', self.family.name()],
                ['Link:', self.link.name()],
                ['Var Weights:', self.var_weights_name],
                ['Method:', self.method],
                ['Nobs:', self.nobs],
                ['Df Residuals:', self.df_resid],
                ['Df Model:', f'{self.edf.sum():.2f}' if self.is_gam else self.df_model],
                ['Log-Likelihood:', "%.4e" % self.llf],
                ['Pseudo Rsq:', "%.4f" % np.round(self.pseudo_rsquared, 4)],
                ['Deviance:', "%.4e" % self.deviance],
                ['Pearson chi2:', np.round(self.pearson_chi2, 4)],
                ['Scale:', "%.4e" % self.scale],
                ['Converged:', self.converged],
                ['Iterations:', self.num_iter],
                ['Rel. Err.:', '%.2e' % self.rel_error],
                ['Abs. Err.:', '%.2e' % self.abs_error],
                ['Cov. Type:', self.cov_type],
                ['Model Elapsed:', '%.2fs' % self.model.model_elapsed],
                ['Fit Elapsed:', '%.2fs' % self.fit_elapsed],
                ['Cov Elapsed:', '%.2fs' % self.cov_elapsed],
            ]
        )

    def get_result_name(self):
        """Return the full result name printed in summaries."""
        return f'Generalized {"Additive" if self.is_gam else "Linear"} Model Results'

    def get_footer_info(self, *args, **kwargs):
        """Build footer text for summaries, including link and inference notes.

        Args:
            *args: Ignored; present for base-class compatibility.
            **kwargs: Forwarded to ``get_inference_string``.

        Returns:
            Footer string appended below coefficient tables.
        """
        ret = 'fit_intercept = %s' % self.fit_intercept
        ret += '\nLink Function: ' + self.link.function_str()

        if np.any(self.alpha) > 0:
            ret += "\nPenalization: alpha = %5s, l1_ratio = %5s" \
                   "\n              normalize = %s, penalize_scale = %s" % (
                       str(self.alpha)[:5], str(self.l1_ratio)[:5], self.normalize, self.penalize_scale)
            if np.any(self.l1_ratio > 0):
                ret += '\n{Note: Inference is *not* available\n since penalized regression is a biased estimator!}'
        if self.is_iv and self.did_compute_var_covar():
            ret += f"\n\nIV residual_inclusion={self.residual_inclusion}"
            ret += '\n\n*** Note: IV is complicated in non-linear settings,' \
                   '\n    experts only! ***\n'
            if BOOTSTRAP not in self.cov_type.upper():
                ret += "\n*** Note: NON-BOOTSTRAP INFERENCE MAY BE " \
                       "\n    UNRELIABLE FOR INSTRUMENTAL VARIABLES!! ***"
        ret += self.get_inference_string(**kwargs)
        return ret

    def predict(self, data=None, params=None, index=None, debug=False, *args, **kwargs):
        """Predict fitted GLM means using the stored model and link.

        Args:
            data: Optional new data for prediction.
            params: Optional coefficient vector; defaults to fitted params.
            index: Optional row subset for new data.
            debug: Whether to print data parsing diagnostics.
            *args: Ignored; retained for API compatibility.
            **kwargs: Extra prediction options; ``link`` defaults to this
                result's fitted link.

        Returns:
            Predicted response-scale means.
        """

        if 'link' not in kwargs.keys():
            kwargs['link'] = self.link

        return super().predict(data=data, params=params, index=index, debug=debug, **kwargs)

    def plot_diagnostics(self, figsize=(6, 5), dpi=130, show=True, maxlags=15):
        """Placeholder for GLM diagnostic plotting.

        Args:
            figsize: Matplotlib figure size.
            dpi: Figure DPI.
            show: Whether to display the plot.
            maxlags: Maximum lags for autocorrelation-style diagnostics.

        Raises:
            NotImplementedError: Always, because GLM diagnostics are not yet implemented.
        """
        raise NotImplementedError("Not Implemented for GLM")

    def get_log_likelihood_function(self, family=None, link=None):
        if family is None:
            family = self.family
        family, link = _get_family_and_link(family, link)
        return self.model.get_log_likelihood_function(family, link)

    def get_marginal_effects(self, at='overall', dummy=True, test_level=.05):
        """Marginal effects of regressors on the fitted response mean.

        Delegates to :func:`~kanly.regression.generalized_linear_models.marginal_effects._get_marginal_effects`.
        For nonlinear links (logit, probit, log, …) coefficient ``beta_k`` is
        not itself ``d mu / d x_k``; this method applies the chain rule through
        ``g^{-1}`` and optionally treats 0/1 columns as discrete shifts.

        Parameters
        ----------
        at : {'overall', 'mean', 'median', 'all'}, optional
            Evaluation point, as in statsmodels ``GLMResults.get_margeff``:

            - ``'overall'`` (default) — average partial effect over observations.
            - ``'mean'`` — effect at ``x* = column means``.
            - ``'median'`` — effect at ``x* = column medians``.
            - ``'all'`` — ``(nobs, nparams)`` matrix of observation-level effects;
              no standard errors returned.

        dummy : bool, optional
            If True (default), auto-detect 0/1 columns and report discrete
            ``P(y|x_k=1) - P(y|x_k=0)``-style effects on the mean scale.

        test_level : float, optional
            Significance level for two-sided normal intervals (default 0.05).

        Returns
        -------
        dict
            Marginal effects, delta-method covariance, and ``summary_df``.  Print
            ``result['summary_df']`` for a statsmodels-style table.

        Examples
        --------
        >>> fit = glm('y ~ x1 + x2', df, family='poisson')       # doctest: +SKIP
        >>> me = fit.get_marginal_effects(at='overall')
        >>> print(me['summary_df'])

        See Also
        --------
        kanly.regression.generalized_linear_models.marginal_effects._get_marginal_effects
        """
        return _get_marginal_effects(self, at=at, dummy=dummy, test_level=test_level)
