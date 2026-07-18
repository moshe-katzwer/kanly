from __future__ import absolute_import, print_function

from copy import copy

import numpy as np
from scipy.stats import skew, kurtosis, chi2

from kanly.regression.constants import DEFAULT_TEST_LEVEL
from kanly.regression.regression_results_base import RegressionResultsBase
from kanly.time_series.sarimax.arma_innovation_functions import get_autocovariance_function, get_causal_representation
from kanly.time_series.sarimax.polynomial import get_combined_differencing_coefs, combine_lag_coefs
from kanly.time_series.sarimax.sarimax_internal_helper_functions import combine_seasonal_lag_params_into_one_vector
from kanly.time_series.sarimax.state_space_arma import get_arma_statespace_matrices
from kanly.time_series.sarimax.validate_sarima_orders import validate_orders

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kanly.time_series.sarimax.model import SarimaxModel


class SarimaxResults(RegressionResultsBase):

    """
    Results container for fitted SARIMAX models, diagnostics, summaries, and forecasts.

    This object participates in the tracked SARIMAX implementation under ``kanly.time_series``.
    """
    def __init__(self, model: "SarimaxModel", order, seasonal_order, nobs, params, cov_params, df_model, df_resid,
                 param_names, endog_name,
                 specification_name, converged, optimization_result, llf, llf_obs,
                 fittedvalues, fittedinnovations, resid,
                 forecasts_error_cov, loglike, loglike_obs, aic, aicc, bic, hqic, loglikelihood_burn,
                 arima_options, cov_type, trend, trend_offset, grad_norm, iter,
                 arparams, maparams, seasonalarparams, seasonalmaparams,
                 exogparams, trendparams,
                 model_elapsed, fit_elapsed, cov_elapsed, k_trend, k_exog,
                 state_vec, simple_differencing, multiplicative
                 ):

        """
        Initialize the object and store parsed SARIMAX state.

        Args:
            model: Fitted SARIMAX model instance.
            order: Nonseasonal ARIMA order specification ``(p, d, q)``.
            seasonal_order: Optional seasonal order specification ``(P, D, Q, s)``.
            nobs: Number of observations.
            params: Packed parameter vector in model parameter order.
            cov_params: Estimated parameter covariance matrix.
            df_model: Model degrees of freedom.
            df_resid: Residual degrees of freedom.
            param_names: Names associated with packed model parameters.
            endog_name: Name of the endogenous variable.
            specification_name: Optional display name for the model specification.
            converged: Whether optimization converged.
            optimization_result: Optimizer result object.
            llf: Total log-likelihood value.
            llf_obs: Per-observation log-likelihood values.
            fittedvalues: In-sample fitted values.
            fittedinnovations: One-step-ahead fitted innovations.
            resid: Model residuals.
            forecasts_error_cov: One-step-ahead forecast error variances.
            loglike: Log-likelihood callable.
            loglike_obs: Per-observation log-likelihood callable.
            aic: Akaike information criterion.
            aicc: Small-sample corrected AIC.
            bic: Bayesian information criterion.
            hqic: Hannan-Quinn information criterion.
            loglikelihood_burn: Number of leading observations excluded from likelihood summaries.
            arima_options: Dictionary of fit options and parsed order metadata.
            cov_type: Covariance estimator name.
            trend: Trend specification, such as None, ``c``, ``t``, ``ct``, ``n``, or an indicator list.
            trend_offset: Starting offset for deterministic trend time indices.
            grad_norm: Final optimizer gradient norm.
            iter: Number of optimizer iterations.
            arparams: Autoregressive coefficients ordered by active AR lags.
            maparams: Moving-average coefficients ordered by active MA lags.
            seasonalarparams: Seasonal autoregressive coefficient vector.
            seasonalmaparams: Seasonal moving-average coefficient vector.
            exogparams: Coefficient vector for exogenous regressors.
            trendparams: Coefficient vector for deterministic trend terms.
            model_elapsed: Time spent constructing the model.
            fit_elapsed: Time spent optimizing the likelihood.
            cov_elapsed: Time spent computing covariance estimates.
            k_trend: Number of deterministic trend parameters.
            k_exog: Number of exogenous-regressor parameters.
            state_vec: Stored Kalman state sequence.
            simple_differencing: Whether to difference data before the state-space likelihood instead of integrating in the state.
            multiplicative: Whether to request multiplicative seasonal dynamics.

        Returns:
            None.  All fitted quantities are stored as instance attributes,
            including ``arparams``, ``maparams``, ``sigma2``, ``llf``,
            ``aic``/``bic``/``hqic``, ``resid``, ``fittedvalues``,
            AR/MA roots and their moduli, the autocovariance function,
            and the ARMA state-space matrices used for forecasting.
        """
        super().__init__(nobs, params, cov_params, df_model, df_resid, df_t_dist=df_resid, exog_names=param_names,
                         endog_name=endog_name,
                         cov_type=cov_type, cov_kwds=None, test_level=DEFAULT_TEST_LEVEL, use_t=False,
                         alpha=0.0, l1_ratio=0.0, specification_name=specification_name

                         )

        self.model: SarimaxModel = model
        ((self.order, self.seasonal_order),
         self.is_seasonal,
         (self.has_ar_terms, self.has_ma_terms),
         (((self.p, self.ar_lags, self.k_ar), self.d, (self.q, self.ma_lags, self.k_ma)),
          ((self.P, self.sar_lags, self.k_sar), self.D, (self.Q, self.sma_lags, self.k_sma), self.s),
          )) = validate_orders(order, seasonal_order)

        self.k_trend = k_trend
        self.k_exog = k_exog

        self.trend = copy(trend)
        self.trend_offset = trend_offset

        self.converged = converged
        self.optimization_result = optimization_result
        self.llf = llf
        self.llf_obs = llf_obs
        self.fittedvalues = fittedvalues
        self.fittedinnovations = fittedinnovations
        self.resid = resid
        self.forecasts_error_cov = forecasts_error_cov
        self.loglike = loglike
        self.loglike_obs = loglike_obs
        self.aic = aic
        self.aicc = aicc
        self.bic = bic
        self.hqic = hqic
        self.loglikelihood_burn = loglikelihood_burn
        self.arima_options = arima_options

        self.scale = self.params.iloc[-1]
        self.sigma2 = self.scale
        self.grad_norm = grad_norm
        self.iter = iter

        self.arparams = arparams
        self.maparams = maparams
        self.seasonalarparams = seasonalarparams
        self.seasonalmaparams = seasonalmaparams
        self.exog_params = exogparams
        self.trend_params = trendparams

        self.model_elapsed, self.fit_elapsed, self.cov_elapsed = model_elapsed, fit_elapsed, cov_elapsed

        self.state_vec = state_vec
        self.num_params = len(params)

        self.arparams_expanded = combine_seasonal_lag_params_into_one_vector(
            self.ar_lags, self.arparams, self.sar_lags, self.seasonalarparams, self.s)
        self.maparams_expanded = combine_seasonal_lag_params_into_one_vector(
            self.ma_lags, self.maparams, self.sma_lags, self.seasonalmaparams, self.s)

        self.autocovariance_function = get_autocovariance_function(
            self.arparams_expanded, self.maparams_expanded, scale=self.sigma2
        )

        self.ssr = np.sum(self.resid[self.loglikelihood_burn:] ** 2)
        self.tss = np.sum(
            (self.model.endog[self.loglikelihood_burn:] - self.model.endog[self.loglikelihood_burn:].mean()) ** 2)
        self.rsquared = 1.0 - self.ssr / self.tss
        self.rsquared_adj = 1.0 - (1.0 - self.rsquared) * (self.nobs - 1) / (self.nobs - self.num_params)

        self.transition_arma, self.observation_arma, self.var_innovation, self.unconditional_state_variance \
            = get_arma_statespace_matrices(self.arparams_expanded, self.maparams_expanded, self.sigma2)
        self.final_state = self.state_vec[-len(self.observation_arma):]

        self.arroots = np.roots(np.flip(np.hstack([1., -self.arparams_expanded])))
        self.maroots = np.roots(np.flip(np.hstack([1., -self.maparams_expanded])))
        self.abs_arroots = np.abs(self.arroots)
        self.abs_maroots = np.abs(self.maroots)
        self.is_stationary = np.all(np.abs(self.arroots) > 1)
        self.is_invertible = np.all(np.abs(self.maroots) > 1)

        self.diff_coefs = get_combined_differencing_coefs(d=self.d, D=self.D, s=self.s)
        self.causal_representation = get_causal_representation(
            combine_lag_coefs((self.arparams_expanded, 1), (self.diff_coefs, 1)),
            self.maparams_expanded
        )

        self.simple_differencing = simple_differencing

        self.multiplicative = multiplicative

    def get_diagnostic_stats(self):
        """
        Compute residual diagnostics for summary output.

        Evaluates the following statistics on the likelihood-burned residual
        series: Durbin-Watson (serial autocorrelation), Jarque-Bera (normality),
        Ljung-Box at lag 1 (residual autocorrelation), skewness, and excess
        kurtosis.  All float values are formatted as 3-decimal strings.

        Returns:
            Dictionary mapping diagnostic label strings (e.g.
            ``'Ljung-Box(L1)(Q):'``, ``'Jarque-Bera(JB):'``,
            ``'Durbin-Watson:'``) to formatted string values.
        """
        wresid = np.asarray(self.resid[self.loglikelihood_burn:])
        n = len(wresid)

        wresid = np.asarray(wresid)
        s = skew(wresid)
        k = 3 + kurtosis(wresid)

        dw = np.sum((wresid[1:] - wresid[:-1]) ** 2) / np.sum(wresid ** 2)
        jb = n / 6 * (s ** 2 + (k - 3) ** 2 / 4)
        prob_jb = chi2.sf(jb, df=2)

        # Ljung-Box uses the likelihood-burned residual series; keep this slice
        # aligned with the summary's effective sample.
        v = (np.cov(wresid[self.loglikelihood_burn + 1:], wresid[self.loglikelihood_burn:-1])[0, 1]
             / np.var(wresid[self.loglikelihood_burn:]))
        lb = n * (n + 2) * v ** 2 / (n - 1)
        prob_lb = chi2.sf(lb, 1)

        ret = {
            'Ljung-Box(L1)(Q):': lb,
            'Prob(Q):': prob_lb,
            'Jarque-Bera(JB):': jb, 'Prob(JB)': prob_jb,
            'Durbin-Watson:': dw,
            'Skew:': s,
            'Kurtosis:': k,
        }

        for k, v in ret.items():
            if isinstance(v, float):
                ret[k] = '%.3f' % v

        return ret

    def get_result_name(self):
        """
        Return the display name for SARIMAX summaries.

        Returns:
            The string ``'SARIMAX Model Results'``.
        """
        return 'SARIMAX Model Results'

    def get_result_type(self):
        """
        Return the estimation type label for summaries.

        Returns:
            The string ``'MLE'``, indicating maximum-likelihood estimation.
        """
        return 'MLE'

    def get_footer_info(self, *args, **kwargs):
        """
        Return explanatory footer text for model summaries.

        The footer describes the estimation method and, depending on whether
        ``simple_differencing`` was used, either the Brockwell & Davis ARMA
        state-space representation on pre-differenced data or the full ARIMA
        state-space representation.

        Returns:
            Multi-line string describing the estimation method and
            state-space representation used.
        """
        return ('Parameters estimated via maximum-likelihood\n'
                + ('Uses Brockwell & Davis ARMA state space representation on\npre-differenced data.'
                    if self.simple_differencing else
                   'Uses Brockwell & Davis ARIMA state space representation.')
                )

    def get_time_series_model_name(self):
        """
        Build a compact time-series model name from active components.

        Assembles abbreviation tokens in the order S (seasonal), AR, I
        (integrated), MA, X (exogenous/trend), producing names such as
        ``'ARMA'``, ``'ARIMA'``, ``'SARIMA'``, ``'ARIMAX'``, ``'SARIMAX'``,
        etc.

        Returns:
            String model-type abbreviation derived from the active components
            of the fitted model.
        """
        has_integ = 'I' if self.D or self.d else ''
        seasonal = 'S' if self.is_seasonal else ''
        has_exog = 'X' if self.k_exog or self.k_trend else ''
        has_ar = 'AR' if self.has_ar_terms else ''
        has_ma = 'MA' if self.has_ma_terms else ''
        return f'{seasonal}{has_ar}{has_integ}{has_ma}{has_exog}'

    def get_header_info_array(self):
        """
        Build summary header metadata for fitted SARIMAX results.

        Constructs an array of ``(label, value)`` pairs covering the model
        specification, fit timing, covariance type, sample size, likelihood-
        burn count, information criteria (AIC/AICc/BIC/HQIC), R-squared,
        convergence status, final gradient norm, and iteration count.

        Returns:
            2-D NumPy object array of shape ``(n_rows, 2)`` containing
            ``(label, value)`` string pairs for rendering the summary header.
        """
        header_info = np.array(
            [
                ('Model:', (f'{"SARIMAX"}({self.p},{self.d},{self.q})'))
            ]
            + [
                ('',
                 f'x({self.P},{self.D},{self.Q},{self.s})'
                 if self.is_seasonal
                 else ''
                 )
            ]
            + [
                ('Date:', self.date),
                ('Time:', self.timestamp),
                ('model time:', '%.2fs' % self.model_elapsed),
                ('fit time:', '%.2fs' % self.fit_elapsed),
                ('cov time:', '%.2fs' % self.cov_elapsed),
                ('Covariance Type:', self.cov_type),
                ('No. Observations:', self.nobs),
                ('Likelihood Burn:', self.loglikelihood_burn),
                ('df Model:', self.df_model),
                ('Log Likelihood:', "%.4f" % self.llf),
                ('Avg. LL:', "%.4f" % (self.llf / (self.nobs - self.loglikelihood_burn))),
                ('AIC:', '%.3f' % self.aic),
                ('AICc:', '%.3f' % self.aicc),
                ('BIC:', '%.3f' % self.bic),
                ('HQIC:', '%.3f' % self.hqic),
                ('R-squared:', '%.4f' % self.rsquared),
                ('Adj. R-squared:', '%.4f' % self.rsquared_adj),
                ('converged:', self.converged),
                ('max|grad|:', '%.2e' % self.grad_norm),
                ('iter:', self.iter),
            ])

        return header_info

    def forecast(self, steps=1, exog=None, return_prediction_variance=False, signal_only=False):
        """alias for ``get_forecast``"""
        return self.get_forecast(steps, exog, return_prediction_variance, signal_only)

    def get_forecast(self, steps=1, exog=None, return_prediction_variance=False, signal_only=False):
        """
        SARIMAX helper function ``get_forecast``.

        Args:
            steps: Number of forecast steps.
            exog: Optional exogenous regressor matrix aligned to ``endog``.
            return_prediction_variance: Whether to return forecast variances with means.
            signal_only: Whether to omit trend and exogenous forecast contributions.

        Returns:
            Forecast mean array, optionally with prediction variances.
        """
        r = len(self.observation_arma)
        state = self.state_vec[-r:].copy()
        y_prev = list(self.model.endog[-max((r + 1), self.d + self.s * self.D):])

        predicted_mean = []
        for i in range(steps):
            pred_state = self.transition_arma.dot(state)
            pred = self.observation_arma.dot(pred_state)
            # Forecast on the integrated/original scale by adding lagged
            # differencing terms back to the ARMA signal forecast.
            for l, c in enumerate(self.diff_coefs):
                pred += c * y_prev[-(l + 1)]
            state = pred_state
            predicted_mean.append(pred)
            y_prev.append(pred)

        predicted_mean = np.array(predicted_mean)
        if not signal_only:
            if self.k_exog:
                # Future exog must be supplied with one row per forecast step.
                predicted_mean += np.dot(exog, self.exog_params)
            if self.k_trend:
                tr = np.arange(self.trend_offset + self.nobs, self.trend_offset + self.nobs + steps) / self.model.trend_scale
                i = 0
                for power, c in enumerate(self.trend):
                    if c:
                        predicted_mean += self.trend_params[i] * tr ** power
                        i += 1

        var_pred_mean = np.cumsum(self.causal_representation[:steps] ** 2) * self.sigma2

        if return_prediction_variance:
            return predicted_mean, var_pred_mean
        else:
            return predicted_mean

    def get_differenced(self):
        """
        Return the differenced endogenous data used by the model.

        Delegates to ``self.model.get_differenced()``, which applies the
        combined ``(1-L)^d * (1-L^s)^D`` filter to the stored endogenous
        series.

        Returns:
            1-D NumPy array of the differenced endogenous series, with length
            ``nobs - d - s*D``.
        """
        return self.model.get_differenced()
