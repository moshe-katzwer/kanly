"""Result classes for kanly linear regression models.

This module defines three result containers and the top-level result class:

- :class:`PredictionDimensionException` — raised when ``predict`` is called
  with incorrectly shaped exog data.
- :class:`InstrumentInfo` — stores IV first-stage outputs and produces the
  IV power / diagnostic table (``summary_iv``).
- :class:`AbsorbInfo2` — stores absorbed-FE outputs (group baselines, within
  and between R²).
- :class:`SparseLinearRegressionResults` — inherits from
  :class:`~kanly.regression.regression_results_base.RegressionResultsBase`
  and extends it with:

  - F-statistics (including robust F via Wald test and non-robust fallback).
  - AIC / BIC.
  - IV power tables and ``summary_iv``.
  - Lift and ratio tests (``test_lift``, ``test_lift_interacted``,
    ``test_lift_ratio``).
  - Eigenvalue / condition number diagnostics.
  - FGLS metadata.
  - Ridge penalty reporting.
"""

from __future__ import absolute_import, print_function

import time
from textwrap import wrap

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist, chi2, normaltest, skew, kurtosis

from kanly import __version__
from kanly.dill_object import DillObject
from kanly.regression.linear_models.constants import DEFAULT_LM_TEST_LEVEL, DEFAULT_LM_COV_TYPE, DEFAULT_LM_USE_T
from kanly.regression.linear_models.shapley import Shapley
from kanly.regression.regression_results_base import RegressionResultsBase
from kanly.utils.util import none_copy, to_dense_helper
from kanly.utils.linalg_utils import get_eigenvals_and_condition_number_internal

MAX_DF_MODEL_FOR_F_STAT = 250


class PredictionDimensionException(Exception):
    """Raised when the prediction exog has a dimension mismatch with the model.

    This occurs when ``predict`` is called with a matrix or DataFrame whose
    number of columns does not match the number of model parameters (or
    instrument columns for IV prediction).
    """
    pass


class InstrumentInfo(DillObject):
    """Container for IV first-stage results and diagnostic information.

    Stores the mapping between original regressors and their instrument
    projections, and provides a formatted string for regression summaries.

    Inherits ``DillObject`` for pickle-compatible serialisation.

    Attributes:
        endog_regressors: Names of endogenous regressors (columns of ``exog``
            not present in the instrument set).
        instrument_regressors: Names of included instruments (columns of ``Z``
            also in ``X``).
        instrument_params: First-stage coefficient matrix π̂, shape
            ``(q, p)`` or list of per-equation matrices for SURE.
        normalized_cov_params_instruments: ``(Z'WZ)^{-1}``, used for IV
            power diagnostics.
        endog_reg_cols: Boolean mask of length ``p`` identifying endogenous
            regressor columns.
        excluded_regressors: Names of excluded instruments (in ``Z`` but not
            in ``X``).
    """

    def __init__(self, instrument_params, endog_regressors, instrument_regressors, excluded_regressors,
                 endog_reg_cols, normalized_cov_params_instruments):
        """Initialise the instrument-info result container.

        Args:
            instrument_params: First-stage coefficient matrix π̂.
            endog_regressors: Names of endogenous regressors.
            instrument_regressors: Names of included instruments.
            excluded_regressors: Names of excluded instruments.
            endog_reg_cols: Boolean mask of endogenous columns.
            normalized_cov_params_instruments: ``(Z'WZ)^{-1}``.
        """
        self.endog_regressors = none_copy(endog_regressors)
        self.instrument_regressors = none_copy(instrument_regressors)

        self.instrument_params = none_copy(instrument_params)
        self.normalized_cov_params_instruments = none_copy(normalized_cov_params_instruments)

        self.instrument_regressors = none_copy(instrument_regressors)
        self.endog_reg_cols = none_copy(endog_reg_cols)
        self.excluded_regressors = none_copy(excluded_regressors)

    def get_iv_string(self):
        """Return a formatted multi-line string listing IV endogenous and excluded regressors.

        Returns:
            str: Two-paragraph string with ``Endogenous Regressors`` and
                ``Excluded Regressors`` sections, word-wrapped to 70 chars.
        """
        iv_string = '\nEndogenous Regressors: ' + '\n'.join(
            wrap(', '.join([str(x) for x in sorted(set(self.endog_regressors) - {'Intercept'})]), width=70 - 24,
                 subsequent_indent=' ' * 24))
        iv_string += '\n\nExcluded Regressors:   ' + '\n'.join(
            wrap(', '.join(sorted(self.excluded_regressors)), width=70 - 24, subsequent_indent=' ' * 24))
        return iv_string


# TODO delete, _____deprecated
# class AbsorbInfo(object):
#
#     def __init__(self, absorb_term_name, num_absorbed, absorbed_y_baselines, absorb_group_mean_dict, wendog_group_means,
#                  within_rsquared, within_rsquared_adj):
#         self.absorbed_y_baselines = none_copy(absorbed_y_baselines)
#         self.absorb_group_mean_dict = none_copy(absorb_group_mean_dict)
#         self.num_absorbed = num_absorbed
#         self.absorb_term_name = absorb_term_name
#         self.wendog_group_means = none_copy(wendog_group_means)
#         self.within_rsquared = within_rsquared
#         self.within_rsquared_adj = within_rsquared_adj
#
#     def get_absorb_string(self):
#         return "\nAbsorbed: '" + str(self.absorb_term_name) + "', num=%d" % self.num_absorbed \
#                + "\n\tWithin R**2=%.3f" % self.within_rsquared + ", Adj. Within R**2=%.3f" % self.within_rsquared_adj

class GLSARInfo(DillObject):
    """
    Diagnostics from a :meth:`~kanly.regression.linear_models.model.SparseLinearModel.fit_glsar` fit.

    Attached to regression results as ``fit.glsar_info`` after calling
    :func:`~kanly.api.glsar` or :func:`~kanly.api.GLSAR`.

    Attributes
    ----------
    nlags : int
        AR order ``p`` used in the model (``GLSAR[p]`` in ``fit.method``).
    ar_params : ndarray
        Final estimated AR coefficients on residuals.
    scale : float
        Innovation variance scale from the last :func:`~kanly.time_series.autoregression.estimate_ar` call.
    numiter : int
        Number of AR/GLS iterations performed (including the initial OLS step in the count).
    error : float
        Max absolute change in AR coefficients at termination (convergence metric).
    ar_method : str
        AR estimation method (e.g. ``'yw'``, ``'css'``).
    full_information : bool
        If True, Prais-Winsten whitening was used; if False, Cochrane-Orcutt.

    Notes
    -----
    ``str(fit.glsar_info)`` and ``repr(fit.glsar_info)`` print the attribute dict
    for quick inspection after fitting.
    """
    def __init__(self, nlags, ar_params, scale, numiter, error, ar_method, full_information):
        self.nlags = nlags
        self.ar_params = ar_params
        self.scale = scale
        self.numiter = numiter
        self.error = error
        self.ar_method = ar_method
        self.full_information = full_information

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self)


class AbsorbInfo2(DillObject):
    """Container for fixed-effects absorption results.

    Stores all outputs from ``AbsorbTools2`` that are needed to produce the
    absorbed-FE section of the regression summary (within / between R²,
    group-level fitted value baselines, etc.).

    Inherits ``DillObject`` for pickle-compatible serialisation.

    Attributes:
        absorbed_y_baselines (ndarray or None): Per-observation group-mean
            baselines added back to fitted values, shape ``(n,)``.
        absorb_group_mean_dict (dict or None): Mapping from group key to
            within-group endog mean (used for prediction).
        num_absorbed (int): Number of absorbed fixed-effect levels.
        absorb_name (str or list of str): Name(s) of the absorbed variable(s).
        wendog_group_means: Group-level weighted endog means (used internally).
        rsquared_within (float): Fraction of within-group endog variance
            explained by the regressors.
        rsquared_between (float): Fraction of between-group endog variance
            attributable to group membership.
    """

    def __init__(self, absorb_name, num_absorbed, absorbed_y_baselines, absorb_group_mean_dict, wendog_group_means,
                 rsquared_within, rsquared_between):
        """Initialise the absorbed-FE result container.

        Args:
            absorb_name (str or list of str): Name(s) of absorbed variable(s).
            num_absorbed (int): Number of absorbed FE levels.
            absorbed_y_baselines (ndarray or None): Group-mean offsets.
            absorb_group_mean_dict (dict or None): Group → mean mapping.
            wendog_group_means: Weighted within-group endog means.
            rsquared_within (float): Within-group R².
            rsquared_between (float): Between-group R².
        """
        self.absorbed_y_baselines = none_copy(absorbed_y_baselines)
        self.absorb_group_mean_dict = none_copy(absorb_group_mean_dict)
        self.num_absorbed = num_absorbed
        self.absorb_name = absorb_name
        self.wendog_group_means = none_copy(wendog_group_means)
        self.rsquared_within = rsquared_within
        self.rsquared_between = rsquared_between

    def get_absorb_string(self):
        """Return a formatted multi-line string summarising the absorbed FE.

        Returns:
            str: Summary showing absorbed variable name, count, and within /
                between R² values.
        """
        return "\nAbsorbed: '" + str(self.absorb_name) + "', num=%d" % self.num_absorbed \
            + "\n\tWithin  R\u00B2 = %.4f" % self.rsquared_within \
            + "\n\tBetween R\u00B2 = %.4f" % self.rsquared_between


class SparseLinearRegressionResults(RegressionResultsBase):
    """Regression result object for kanly linear models.

    Extends :class:`~kanly.regression.regression_results_base.RegressionResultsBase`
    with linear-model-specific attributes and methods:

    - F-statistic (robust via Wald test, or non-robust OLS fallback).
    - AIC and BIC (NA for IV / SURE models).
    - IV power diagnostics and ``summary_iv``.
    - Absorbed-FE summary and within/between R².
    - Lift / ratio tests (``test_lift``, ``test_lift_interacted``,
      ``test_lift_ratio``).
    - FGLS metadata (``do_fgls``, ``fgls_weights``, ``fgls_info``).
    - Ridge penalty reporting.
    - Eigenvalues and condition number of X'X.
    - Shapley R² decomposition across formula sparse_terms (:meth:`shapley_value`).

    Attributes mirror those of the base class plus the linear-model-specific
    fields listed in :meth:`__init__`.
    """

    def __init__(self, use_t, model, nobs, df_resid, df_model, df_t_dist, params, cov_params, rsquared, rsquared_adj,
                 ssr, wssr, sst, wsst, uncentered_tss, endog_name, exog_names, exog_term_names, llf, fit_elapsed, cov_elapsed,
                 cov_type, cov_kwds, has_const, has_implicit_constant, method, resid, wresid, fittedvalues,
                 eigenvals, condition_number, normalized_cov_params,
                 absorb_info=None, instrument_info=None,
                 weights_name=None, cov_string=None, test_level=DEFAULT_LM_TEST_LEVEL,
                 small_samp_correct=None, wexog_instrumented_means=None, keep_model=True,
                 null_rows_info_dict=None, valid_obs_rows=None, specification_name=None,
                 do_fgls=False, fgls_weights=None, fgls_info=dict(), ridge_kwds=None, glsar_info=None,
                 ):
        """Initialise the linear regression result container.

        Args:
            use_t (bool): Use t-distribution for inference.
            model (SparseLinearModel or None): Fitted model object.
            nobs (int): Number of observations.
            df_resid (int): Residual degrees of freedom.
            df_model (int): Model degrees of freedom.
            df_t_dist (int or None): DF for t-tests (e.g. G−1 for cluster).
            params (array-like): Coefficient vector, shape ``(p,)``.
            cov_params (sparse or ndarray or None): Parameter covariance,
                shape ``(p, p)``.
            rsquared (float): R².
            rsquared_adj (float): Adjusted R².
            ssr (float): Unweighted SSR.
            wssr (float): Weighted SSR.
            sst (float): Unweighted SST.
            wsst (float): Weighted SST.
            uncentered_tss (float): Weighted uncentred TSS.
            endog_name (str or list of str): Dependent variable name(s).
            exog_names (list of str): Regressor names.
            exog_term_names (list of str): Patsy term names.
            llf (float or None): Log-likelihood (``None`` for IV/SURE).
            fit_elapsed (float): Wall-clock seconds for estimation.
            cov_elapsed (float): Wall-clock seconds for covariance.
            cov_type (str): Covariance type label.
            cov_kwds (dict): Covariance keyword arguments.
            has_const (bool): True if model has any constant term.
            has_implicit_constant (bool): True if constant is implicit.
            method (str): Method label (``'OLS'``, ``'WLS'``, etc.).
            resid (array-like): Residuals.
            wresid (array-like): Weighted residuals.
            fittedvalues (array-like): Fitted values.
            eigenvals (array-like or None): Eigenvalues of X'WX.
            condition_number (float or None): Condition number.
            normalized_cov_params: ``(X'WX)^{-1}``.
            absorb_info (AbsorbInfo2, optional): Absorbed-FE metadata.
            instrument_info (InstrumentInfo, optional): IV metadata.
            weights_name (str, optional): Weight variable name.
            cov_string (str, optional): Human-readable covariance description.
            test_level (float): Significance level.
            small_samp_correct: Small-sample correction factor or tuple.
            wexog_instrumented_means (ndarray, optional): Weighted means of
                the instrumented design matrix columns.
            keep_model (bool): Attach model to results.
            null_rows_info_dict (dict): Metadata about dropped null rows.
            valid_obs_rows (array-like): Boolean mask of valid observations.
            specification_name (str, optional): Label for output tables.
            do_fgls (bool): True if FGLS was performed.
            fgls_weights (ndarray or None): Final FGLS weights.
            fgls_info (dict): FGLS convergence metadata.
            ridge_kwds (dict or None): Ridge penalty options used.
        """

        self.has_implicit_constant = has_implicit_constant
        self.has_intercept = model.has_intercept
        self.is_sure = model.is_sure

        self.instrument_info = instrument_info
        self.is_iv = instrument_info is not None

        self.has_const = has_const

        self.ssr = ssr
        self.wssr = wssr
        self.sst = sst
        self.wsst = wsst
        self.uncentered_tss = uncentered_tss

        self.cost = wssr / 2

        super().__init__(nobs, params, cov_params, df_resid=df_resid, df_model=df_model, df_t_dist=df_t_dist,
                         exog_names=exog_names, endog_name=endog_name, cov_type=cov_type, cov_kwds=cov_kwds,
                         test_level=test_level, use_t=use_t, l1_ratio=0,
                         alpha=0.0 if ridge_kwds is None else ridge_kwds['alpha'],
                         specification_name=specification_name)

        self.num_params = len(self.params)
        if absorb_info is not None:
            self.num_params += absorb_info.num_absorbed
        self.scale = (self.wssr / (self.nobs - self.num_params)) if self.nobs > self.num_params else np.inf
        self.scale_mle = self.wssr / self.nobs

        self.set_properties_from_model(model, keep_model)

        self.exog_term_names = none_copy(exog_term_names)
        self.instrument_names = (
            model.instrument_names.copy()
            if hasattr(self.model, 'instrument_names')
            else [])

        self.rsquared = rsquared
        self.rsquared_adj = rsquared_adj

        self.fit_elapsed = fit_elapsed
        self.model_elapsed = model.model_elapsed
        self.cov_elapsed = cov_elapsed

        self.weights_name = weights_name
        self.cov_string = cov_string

        if method is None:
            method = 'LM'
        self.method = method + (" (SURE)" if self.is_sure else '')

        self.resid = to_dense_helper(resid, flatten=True)
        self.wresid = to_dense_helper(wresid, flatten=True)
        self.fittedvalues = to_dense_helper(fittedvalues, flatten=True)

        self.test_level = test_level

        self.bootstrapped_params = None

        self.absorb_info = absorb_info
        self.is_absorb = self.absorb_info is not None

        self.exog_names = none_copy(model.exog_names)

        self.formula = none_copy(model.formula)

        self.wexog_instrumented_means = None
        self.wexog_instrumented_means_df = None

        self.wexog_instrumented_means = wexog_instrumented_means
        if wexog_instrumented_means is not None:
            self.wexog_instrumented_means_df = pd.DataFrame(
                index=self.params.index, columns=['mean'],
                data=self.wexog_instrumented_means
            )
        else:
            self.wexog_instrumented_means_df = None

        self.instrument_rsquared_df = None

        self.small_samp_correct = small_samp_correct

        self.valid_obs_rows = valid_obs_rows
        self.null_rows_info_dict = null_rows_info_dict

        self.set_F_stat()

        self.do_fgls = do_fgls
        self.fgls_weights = fgls_weights
        self.fgls_info = fgls_info

        if self.is_iv or self.is_sure:
            self.llf, self.aic, self.bic = None, None, None
        else:
            self.llf = llf
            self.aic, self.bic = self.get_aic_bic(
                self.llf, self.nobs, self.df_model + (self.has_implicit_constant or self.has_intercept))

        self.condition_number = condition_number  # condition number of X
        self.eigenvals = np.array(eigenvals)  # eigenvalues of X'X = inv(normalized_cov_params)

        self.normalized_cov_params = (
                normalized_cov_params.copy() if normalized_cov_params is not None else None)

        self.ridge_kwds = ridge_kwds

        self.glsar_info = glsar_info

    @staticmethod
    def get_result_type():
        """Return the short type code for this result class.

        Returns:
            str: ``'LLS'`` (Linear Least Squares).
        """
        return 'LLS'

    @staticmethod
    def get_aic_bic(llf, nobs, num_params):
        """Compute the Akaike (AIC) and Bayesian (BIC) information criteria.

        Args:
            llf (float): Log-likelihood of the model.
            nobs (int): Number of observations.
            num_params (int): Number of free parameters.

        Returns:
            tuple:
                - **aic** (float): 2k − 2 log L.
                - **bic** (float): k log(n) − 2 log L.
        """
        aic = 2 * num_params - 2 * llf
        bic = np.log(nobs) * num_params - 2 * llf
        return aic, bic

    def set_F_stat(self):
        """Compute and cache the F-statistic, p-value, and non-robust flag.

        Called at the end of ``__init__`` and after :meth:`set_cov_params`
        to keep the cached F-stat consistent with the current covariance.
        """
        self.fvalue, self.f_pvalue, self.nonrobust_fvalue = self.get_F_stat()

    def get_F_stat(self):
        """Compute the F-statistic for the overall model.

        Uses a robust Wald F-test when the covariance is available and the
        model has few enough parameters (≤ ``MAX_DF_MODEL_FOR_F_STAT`` = 250).
        Falls back to the classic SSR-based non-robust F-statistic for large
        models with intercepts.

        Returns:
            tuple:
                - **fvalue** (float): F-statistic (NaN for IV / SURE / no covariance).
                - **f_pvalue** (float): P-value from F-distribution.
                - **nonrobust_fvalue** (bool): True when the non-robust
                  fallback was used (a warning is shown in the summary footer).
        """

        if self.cov_type.upper() == 'NOT COMPUTED' or not self.did_compute_var_covar():
            return np.nan, np.nan, False

        if self.is_sure:
            return np.nan, np.nan, False

        #if self.has_implicit_constant:
        #    return np.nan, np.nan, False

        else:
            nonrobust_fvalue = False
            if self.df_model == 0:
                fvalue = 0
                f_pvalue = 1.0
            elif (self.df_model <= MAX_DF_MODEL_FOR_F_STAT or self.is_iv) and not self.has_implicit_constant:
                result = self.F_test()
                fvalue, f_pvalue = result["test_stat"], result["pvalue"]
            elif self.cov_type.upper() == 'OLS' and (self.has_intercept or self.has_implicit_constant):
                fvalue = ((self.wsst - self.wssr) / self.df_model) / (self.wssr / self.nobs)
                f_pvalue = f_dist.sf(fvalue, self.nobs, self.df_model)
            elif self.cov_type.upper() == 'OLS_SMALL' and (self.has_intercept or self.has_implicit_constant):
                fvalue = ((self.wsst - self.wssr) / self.df_model) / (self.wssr / self.df_resid)
                f_pvalue = f_dist.sf(fvalue, self.df_resid, self.df_model)
            elif self.has_intercept or self.has_implicit_constant:
                nonrobust_fvalue = True
                fvalue = ((self.wsst - self.wssr) / self.df_model) / (self.wssr / self.df_resid)
                f_pvalue = f_dist.sf(fvalue, self.df_resid, self.df_model)
            else:
                fvalue, f_pvalue, nonrobust_fvalue = np.nan, np.nan, False
            return fvalue, f_pvalue, nonrobust_fvalue

    def get_result_name(self):
        """Return the display name for this result type.

        Returns:
            str: ``'Linear Model Results'``.
        """
        return 'Linear Model Results'

    def get_header_info_array(self):
        """Build the header metadata array for the regression summary table.

        Returns a 2-D array of ``['label', 'value']`` pairs covering:
        date/time, elapsed times, method, L2 penalty, weights, intercept
        flags, covariance type, observation counts, R², F-stat, LLF, AIC,
        BIC, and scale.

        Returns:
            np.ndarray, shape (n, 2): Header key-value pairs.
        """
        uncentered = '' if (self.has_intercept or self.has_implicit_constant) else ' (uncentered)'
        if isinstance(self.weights_name, str):
            weights_name = self.weights_name
        else:
            try:
                weights_name = self.weights_name[0]
            except:
                weights_name = self.weights_name

        return np.array([
            ['Date:', self.date],
            ['Time:  ', self.timestamp],
            ['Model Elapsed:', '%.2f s' % self.model_elapsed],
            ['Fit Elapsed:', '%.2f s' % self.fit_elapsed],
            ['Cov Elapsed:', '%.2f s' % self.cov_elapsed],
            ['Method:', self.method],
            ['L2 Penalty:', None if self.ridge_kwds is None else
            ("%.3e" % self.alpha if isinstance(self.alpha, (float, int)) else '<custom>')],
            ['Weights:', weights_name],
            ['Intercept:', self.has_intercept],
            ['Implicit Intercept:', self.has_implicit_constant],
            ['Covariance Type:', self.cov_type],
            ["", ""],
            ['No. Obs.', self.nobs],
            ['Df Residuals:', self.df_resid],
            ['Df Model:', self.df_model],
            [f'R-squared{uncentered}:', np.round(self.rsquared, 4)],
            [f'Adj. R-squared{uncentered}:', np.round(self.rsquared_adj, 4)],
            ['F-statistic:', '-' if (self.is_iv or self.is_sure) else
            ('%.3e' % self.fvalue) if self.fvalue > 100_000 else ("%.2f" % self.fvalue)],
            ['Prob (F-statistic):', '-' if (self.is_iv or self.is_sure) else
            self.f_pvalue if np.isnan(self.f_pvalue) else
            ('%.3f' % self.f_pvalue) if self.f_pvalue > .001 else '<.001'],
            ['Log-Likelihood:', "-" if (self.is_iv or self.is_sure) else "%.4f" % self.llf],
            ['AIC:', "-" if (self.is_iv or self.is_sure) else "%.2f" % self.aic],
            ['BIC:', "-" if (self.is_iv or self.is_sure) else "%.2f" % self.bic],
            ['scale:', "%.4e" % self.scale],
        ])

    def get_diagnostic_stats(self):
        """Compute residual diagnostic statistics for the summary footer.

        Computes: Omnibus test, Jarque-Bera test, Durbin-Watson statistic,
        skewness, kurtosis, and condition number.

        Returns:
            dict: Mapping of statistic name to formatted string value.
        """
        wresid = np.asarray(self.wresid)
        n = len(wresid)

        wresid = np.asarray(wresid)
        s = skew(wresid)
        k = 3 + kurtosis(wresid)

        dw = np.sum((wresid[1:] - wresid[:-1]) ** 2) / np.sum(wresid ** 2)
        jb = n / 6 * (s ** 2 + (k - 3) ** 2 / 4)
        prob_jb = chi2.sf(jb, df=2)

        try:
            omni, pr_omni = tuple(normaltest(wresid))
        except:
            omni, pr_omni = np.nan, np.nan

        ret = {
            'Omnibus': omni, 'Prob(Omnibus):': pr_omni,
            'Jarque-Bera(JB):': jb, 'Prob(JB)': prob_jb, 'Durbin-Watson:': dw,
            'Skew:': s,
            'Kurtosis:': k,
            'Cond. No.:': self.condition_number if self.condition_number is not None else 'not computed'
        }

        for k, v in ret.items():
            if isinstance(v, float):
                ret[k] = '%.3f' % v

        return ret

    def get_footer_info(self, *args, **kwargs):
        """Build the footer text for the regression summary.

        Assembles IV info, absorbed-FE info, inference string, ridge
        warning, non-robust F-stat note, and numerical stability warnings
        based on eigenvalues and condition number.

        Args:
            **kwargs: Must include ``'test_level'`` (float).

        Returns:
            str: Multi-line footer string.
        """

        test_level = kwargs['test_level']

        iv_string = ''
        if self.is_iv:
            iv_string = self.instrument_info.get_iv_string() + "\n"

        absorbed_string = ''
        if self.is_absorb:
            absorbed_string = self.absorb_info.get_absorb_string() + "\n"

        var_suppressed_str = ''
        # if show_only_stat_sig: TODO
        #     var_suppressed_str = '\nParameter estimates suppressed for %d variables' % (
        #             len(stars) - len(coef_table))

        inference_str = self.get_inference_string(**kwargs)

        ridge_str = ''
        if hasattr(self, 'ridge_kwds') and self.ridge_kwds is not None:
            if (isinstance(self.ridge_kwds['alpha'], dict) and np.any(np.array(list(self.ridge_kwds['alpha'].values())) > 0)) \
                or ((not isinstance(self.ridge_kwds['alpha'], dict)) and np.any(self.ridge_kwds['alpha'] > 0)):
                ridge_str = '\nRidge regression with penalty %s' % self.ridge_kwds['alpha']
                ridge_str += f" (normalize={self.ridge_kwds.get('normalize', True)}, " \
                             f"penalize_intercept={self.ridge_kwds.get('penalize_intercept', False)})"

        try:
            if self.condition_number is None and (self.eigenvals is None or np.ndim(self.eigenvals) == 0):
                numerical_warning_str = '\n\nEigenvalues and condition number not computed.'
            elif self.condition_number is not None:
                if self.condition_number > 1e3:
                    numerical_warning_str = (
                                                '\n\nThe condition number is %.2e;'
                                                '\n  this may indicate strong multicollinearity or other numerical issues'
                                                '\n  with the specified design matrix. (Eigenvalues not computed.)'
                                            ) % (self.condition_number)
                else:
                    numerical_warning_str = ''  #\n\nEigenvalues not computed.'
            elif self.eigenvals[-1] < 1e-10 or self.condition_number > 1e3:
                numerical_warning_str = (
                                            '\n\nThe smallest eigenvalue is %.2e and the condition number is %.2e;'
                                            '\n  this may indicate strong multicollinearity or other numerical issues'
                                            '\n  with the specified design matrix.'
                                        ) % (self.eigenvals[-1], self.condition_number)
            else:
                numerical_warning_str = ''

        except:
            numerical_warning_str = (f'ERROR with eigenvalues:\n\t'
                                     f'type(eigenvalues) = {type(self.eigenvals)}, '
                                     f'type(condition_number) = {type(self.condition_number)}'
                                     )

        return (
                absorbed_string
                + iv_string
                + var_suppressed_str
                + inference_str
                # + cluster_string
                + ridge_str
                + ('\nNote: Inference is *not* reliable since Ridge is a biased estimator!'
                   if hasattr(self.model, 'ridge_kwds') and self.ridge_kwds else '')
                + (("\n\n(*) F-statistic computed assuming spherical errors."
                    + "\n    For robust F-statistic, use `fit.F_test()`\n") if self.nonrobust_fvalue and
                                                                               self.did_compute_var_covar() else '')
                + numerical_warning_str
        )

    def get_dep_variable_str(self):
        """Return a formatted string listing dependent variable name(s).

        For single-outcome models returns a single line.  For multi-outcome
        models, each variable is listed with a numeric prefix.

        Returns:
            str: One or more lines describing the dependent variable(s).
        """
        if not isinstance(self.endog_name, list):
            endog_table_strs = ['Dep. Variable:   ' + self.endog_name]
        else:
            endog_table_strs = \
                ['Dep. Variable:   ' + '0) ' + self.endog_name[0]] \
                + ['                 ' + str(j + 1) + ") " + y for j, y in enumerate(self.endog_name[1:])]
        return '\n'.join(endog_table_strs)

    def get_formula_str(self):
        """Return the formula string(s) for the summary header.

        For single-outcome models delegates to the base class.  For SURE
        / multi-formula models, formats each formula with an index prefix.

        Returns:
            str: Formula description, or ``''`` when no formula is set.
        """
        if self.formula is not None:
            formulas = self.formula
            if isinstance(formulas, str):
                return super().get_formula_str()

            formula_str = 'formulas:'
            for i, f in enumerate(formulas):
                if f is not None:
                    formula_str += ('\n   [%d] ' % i) + '\n\t'.join(wrap(f, width=50, subsequent_indent=' ' * 8))
            return formula_str
        else:
            return ''

    # # TODO TODO model removal
    # def predict_iv(self, exog=None, debug=False):
    #     """
    #     Skeels, Christopher L. & Taylor, Larry W., 2014. "Prediction after IV estimation,"
    #     Economics Letters, Elsevier, vol. 122(3), pages 420-422.
    #     :param exog: a dataframe or numpy ndarray
    #     """
    #
    #     if self.is_sure:
    #         raise Exception("prediction not yet supported in case of SURE.")
    #     if not self.is_iv:
    #         raise Exception("Only call 'predict_iv' on IV models!")
    #     if self.is_absorb:
    #         raise Exception("Cannot do `predict_iv` with absorbed fixed effects!")
    #
    #     params = np.dot(self.instrument_params, self.params)
    #
    #     if exog is None:
    #         exog = self.model.instruments
    #
    #     if isinstance(exog, np.ndarray) or isinstance(exog, np.matrix) or isspmatrix(exog):
    #         if exog.shape[1] != self.instrument_params.shape[0]:
    #             raise PredictionDimensionException(
    #                 "Supplied `exog` has %d columns but the model has "
    #                 "%d parameters!" % (exog.shape[1], self.instrument_params.shape[0]))
    #
    #         if isspmatrix(exog):
    #             return exog.dot(params.reshape((-1, 1))).reshape((-1, 1)).flatten()
    #         else:
    #             return exog.dot(params).reshape((-1, 1)).flatten()
    #
    #     elif isinstance(exog, DataFrame) or isinstance(exog, SparseDataFrame):
    #
    #         try:
    #             exog_obj = SparseDataGetter.dmatrix(self.instrument_term_names, exog,
    #                                                 check_zero_cols=False, debug=debug)
    #         except:
    #             raise SparseDataGetterException("SparseFormula failed! (data columns=%s, instrument names=%s"
    #                                             % (str(exog.columns), str(self.model.instrument_term_names)))
    #
    #         exog2, nan_rows, Z_col_names = exog_obj.values, exog_obj.null_rows, exog_obj.column_names
    #
    #         if Z_col_names != list(self.instrument_params.index):
    #             raise PredictionDimensionException(
    #                 "Supplied `exog` has columns %s but the model has "
    #                 "parameters %s!" % (str(Z_col_names), str(self.instrument_params.index)))
    #
    #         # if 'Intercept' not in self.exog_names:  # TODO REMOVE AS WRONG?
    #         #     exog2 = exog2[:, 1:]
    #         pred = exog2.dot(params.reshape((-1, 1)))[:, 0]
    #         pred[list(nan_rows)] = np.nan
    #
    #         return pred
    #
    #     raise Exception(
    #         "`exog` must be one of `None`, `pd.DataFrame` "
    #         "or `numpy.ndarray` "
    #         "or `scipy.sparse.spmatrix`!\n`exog` type %s not supported"
    #         % str(type(exog)))
    #
    # def predict(self, exog=None, absorb=None, debug=False, fail_on_column_difference=False, params=None,
    #             override_iv_error=False):
    #
    #     if exog is None:
    #         return self.fittedvalues.copy()
    #
    #     if self.is_sure:
    #         raise Exception("prediction with `exog` arg not yet supported in case of SURE.")
    #
    #     fitted_values = self._predict_with_data_arg(exog, params=params, override_iv_error=override_iv_error,
    #                                                 fail_on_column_difference=fail_on_column_difference)
    #
    #     if absorb is not None:
    #         fitted_values += self._get_absorb_predicted_baselines(exog, absorb)
    #
    #     return fitted_values

    # def _get_absorb_predicted_baselines(self, exog, absorb):
    #
    #     if not self.is_absorb:
    #         raise Exception("!")  # TODO
    #
    #     if isinstance(exog, DataFrame):
    #
    #         if isinstance(absorb, bool) and absorb:
    #
    #             absorb_data_obj = get_categorical_control_data(
    #                 self.absorb_info.absorb_name, exog)
    #             groups_to_enum_dict, group_to_row_lists_dict = AbsorbTools.get_absorb_mappings(absorb_data_obj)
    #             absorb_baselines = np.zeros(exog.shape[0])
    #             for group in groups_to_enum_dict.keys():
    #                 rows = group_to_row_lists_dict[group]
    #                 absorb_baselines[rows] = self.absorb_info.absorb_group_mean_dict[group]
    #             return absorb_baselines
    #
    #     # sparse matrix or ndarray
    #     elif isinstance(absorb, np.ndarray):
    #
    #         exog_absorb = pd.DataFrame(data=absorb)
    #         return np.array([
    #             self.absorb_info.absorb_group_mean_dict[v]
    #             for v in exog_absorb.apply(tuple, axis=1)
    #         ])
    #
    #     else:
    #         raise Exception("!!! TODO") # TODO

    # TODO FIX this!
    # def predict(self, exog=None, absorb=None, debug=False):
    #
    #     if exog is None:
    #         return self.fittedvalues
    #
    #     if self.model == 'NLLS' or self.model == 'WNLLS':
    #         if exog is None:
    #             exog = self.model.exog
    #         return self.model.pred(self._params, exog)
    #
    #     if self.model.is_sure:
    #         raise Exception("prediction not yet supported in case of SURE.")
    #     # if self.model.is_iv:
    #     #     raise Exception("'predict' function not supported in case of IV, use 'predict_iv'.")
    #
    #     if isinstance(exog, np.ndarray) or isinstance(exog, np.matrix) or isspmatrix(exog):
    #         if exog.shape[1] != len(self.exog_names):
    #             raise PredictionDimensionException(
    #                 "Supplied `exog` has %d columns but the model has "
    #                 "%d parameters!" % (exog.shape[1], len(self.exog_names)))
    #
    #         if isspmatrix(exog):
    #             pred = exog.dot(self._params.reshape((-1, 1))).flatten()
    #         else:
    #             pred = exog.dot(self._params)
    #
    #         if self.absorb_group_mean_dict is not None:
    #             if absorb is not None:
    #
    #                 exog_absorb = pd.DataFrame(data=absorb)
    #                 to_add = np.array([
    #                     self.absorb_group_mean_dict[v]
    #                     for v in exog_absorb.apply(tuple, axis=1)
    #                 ])
    #                 pred += to_add
    #
    #             else:
    #
    #                 warnings.warn("Supplied np.ndarray `exog` without absorb, so prediction does not include "
    #                               "absorbed fixed effects.")
    #         return pred
    #
    #     elif isinstance(exog, DataFrame) or isinstance(exog, SparseDataframe):
    #
    #         try:
    #             exog_obj = SparseFormula.dmatrix(self.exog_term_names, exog, check_zero_cols=False, debug=debug,
    #                                              do_absorb=self.absorb_group_mean_dict is not None)
    #
    #         except:
    #             raise SparseFormulaException("SparseFormula failed! (data columns=%s, exog_term_names=%s)"
    #                                          % (str(exog.columns), str(self.exog_term_names)))
    #
    #         exog2, nan_rows, X_col_names = exog_obj.values, exog_obj.null_rows, exog_obj.column_names
    #
    #         params_local = self._params.reshape((-1, 1))
    #
    #         if X_col_names != list(self.exog_names):
    #
    #             if set(X_col_names) < set(self.exog_names):
    #
    #                 missing_vars = set(self.exog_names) - set(X_col_names)
    #                 all_categorical = np.all([m[:2] == 'C(' for m in missing_vars])
    #
    #                 if all_categorical:
    #                     if debug:
    #                         warnings.warn("Predicting on data frame missing values for columns %s!" % str(missing_vars))
    #
    #                     keep = [m in X_col_names for m in self.exog_names]
    #                     params_local = params_local[keep]
    #
    #                 else:
    #                     raise PredictionDimensionException(
    #                         "Supplied `exog` has columns %s but the model has "
    #                         "parameters %s!" % (str(X_col_names), str(self.exog_names)))
    #
    #             else:
    #                 raise PredictionDimensionException(
    #                     "Supplied `exog` has columns %s but the model has "
    #                     "parameters %s!" % (str(X_col_names), str(self.exog_names)))
    #
    #         # if 'Intercept' not in self.exog_names:  # TODO REMOVE AS WRONG?
    #         #     exog2 = exog2[:, 1:]
    #         pred = exog2.dot(params_local)[:, 0]
    #         pred[list(nan_rows)] = np.nan
    #
    #         if self.absorb_group_mean_dict is not None:
    #             if isinstance(self.model.absorb_name, str):
    #                 absorb_val = Series(exog[self.model.absorb_name])
    #                 pred += absorb_val.map(self.absorb_group_mean_dict)
    #             else:
    #                 exog_absorb = exog[self.model.absorb_name]
    #                 to_add = np.array([
    #                     self.absorb_group_mean_dict[v]
    #                     for v in exog_absorb.apply(tuple, axis=1)
    #                 ])
    #                 pred += to_add
    #
    #         return pred
    #
    #     raise Exception(
    #         "`exog` must be one of `None`, `pd.DataFrame` "
    #         "or `numpy.ndarray` "
    #         "or `scipy.sparse.spmatrix`!\n`exog` type %s not supported"
    #         % str(type(exog)))

    def summary_iv(self, debug=False, only_endog_cols=True):
        """Print and return the IV first-stage power table.

        Displays R² and approximate F-statistic for each regressor's
        projection onto the instruments.  For strong instruments, the R² of
        each endogenous regressor on the instrument set should be high.

        Args:
            debug (bool): Verbose output during R² computation.
            only_endog_cols (bool): If ``True``, restrict the table to
                endogenous regressor columns.  If ``False``, show all
                regressor columns.

        Returns:
            str: Formatted IV power table.

        Raises:
            Exception: If called on a non-IV model.
        """
        if not self.is_iv:
            raise Exception("Can only call `summary_iv` on an IV regression!")

        recompute_df = False
        if self.instrument_rsquared_df is None:
            recompute_df = True
        else:
            if only_endog_cols:
                if self.instrument_rsquared_df.shape[0] > np.sum(self.endog_reg_cols):
                    recompute_df = True
            else:
                if self.instrument_rsquared_df.shape[0] < len(self.exog_names):
                    recompute_df = True

        if recompute_df:
            self._set_instrument_rsquared_df(debug=debug, only_endog_cols=only_endog_cols)

        iv_string = self._get_iv_string()
        rsqstrs = self.instrument_rsquared_df.to_string().split('\n')
        width = max(50, len(rsqstrs[0]))
        bar = '-' * width
        dblbar = '=' * width

        return (
                dblbar
                + "\nIV Power"
                + "\n" + ((self.specification_name) if (self.specification_name is not None) else "")
                + "\n" + dblbar
                + "\n" + rsqstrs[0] + "\n" + bar + "\n" + "\n".join(rsqstrs[1:])
                + "\n" + dblbar
                + iv_string
                + "\n" + dblbar
                + "\n(F stats only approximate)"
                + "\n" + (" " * max(width - 31, 0)) + "[kanly package, v=%s]\n" % __version__
                + "\n\n")

    # TODO TODO model removal — needs wexog / winstruments; currently requires keep_model=True
    def _set_instrument_rsquared_df(self, debug=False, only_endog_cols=True):
        """Compute and cache the IV R² and F-statistic DataFrame.

        Populates ``self.instrument_rsquared_df`` with a DataFrame of shape
        ``(p_endog, 3)`` (or ``(p, 3)`` when ``only_endog_cols=False``) with
        columns ``['rsquared', 'fvalue', 'is_endog']``.

        Args:
            debug (bool): Verbose output.
            only_endog_cols (bool): Restrict to endogenous columns.
        """

        if only_endog_cols:
            columns = self.endog_reg_cols
            endog_col_df = np.array(self.endog_reg_cols)[columns]
            index = np.array(self.exog_names)[columns]
        else:
            columns = None
            index = self.exog_names
            endog_col_df = self.endog_reg_cols

        Rsq_temp, F_temp = self.instrument_rsquared_df = self._get_rsquared_instruments(
            self.model.wexog, self.model.wexog_instrumented, self.model.winstruments.shape[1],
            debug=debug, columns=columns)

        self.instrument_rsquared_df = pd.DataFrame(
            index=index,
            columns=['rsquared', 'fvalue', 'is_endog'],
            data=np.array([Rsq_temp, F_temp, endog_col_df]).T
        )

    def _get_param_index(self, variable):
        """Resolve a variable name or (dep_var, indep_var) tuple to an integer column index.

        Args:
            variable (str, tuple, or int): A regressor name, a (dep_var, indep_var) tuple
                for multi-outcome models (where column names are prefixed ``{i}_{name}``),
                or a raw integer index.

        Returns:
            int: The position of *variable* in ``self.params``.

        Raises:
            Exception: If the variable is not found, or if the dep_var component of a
                tuple matches multiple regressands.
        """
        if isinstance(variable, tuple):
            dep_var, indep_var_name = variable
            dep_var_idx = np.where(np.array(self.endog_name) == dep_var)[0]
            if len(dep_var_idx) == 0:
                raise Exception("'%s' not found as a regressand!" % dep_var)
            elif len(dep_var_idx) > 1:
                raise Exception("'%s' found as multiple regressands!" % dep_var)
            else:
                variable = '%d_%s' % (int(dep_var_idx[0]), indep_var_name)
        if isinstance(variable, str):
            idx = np.where(self.params.index == variable)[0]
            if len(idx) != 1:
                raise Exception("'%s' not found as column!" % variable)
            idx = int(idx.item())
        else:
            idx = variable
        if isinstance(idx, int):
            return idx
        else:
            raise Exception("`treatment_index` must be either a string mapping to regressor,"
                            "or the integer index of that column")

    def test_lift_interacted(self, numer_dict, denom_dict=None, null_hypothesis=0):
        """
        Fieller ratio test with the following procedure:

            - `top` assumed to be all zero unless variable specified in `numer_dict` keys,
              in which case it uses that coefficient
            - `bottom` assumed to be the (weighted in case of WLS) average of the design matrix
              columns, unless
                 * any key in `numer_dict` is set to 0 in denominator coefficients
                 * any key in `denom_dict` has it's value set by `denom_dict` values

            `numer_dict` and `denom_dict` cannot have overlapping keys!

            For example, if we ran

                `endog ~ Intercept + treatment + x + I(x*treatment) + w`

            then the local treatment effect at x=3 is given by

                test_lift_interacted(
                    numer_dict={'treatment': 1, 'I(x*treatment)': 3},
                    denom_dict={'x': 3})
        """

        top = np.zeros(len(self.params))
        bottom = self.wexog_instrumented_means.copy()

        if denom_dict is None:
            denom_dict = dict()

        if len(set(numer_dict.keys()).intersection(denom_dict.keys())):
            raise Exception("`numer_dict` and `denom_dict` must have exclusive keys!")

        for i, key in enumerate(numer_dict.keys()):
            idx = self._get_param_index(key)
            top[idx] = numer_dict[key]
            bottom[idx] = 0

        for i, key in enumerate(denom_dict.keys()):
            idx = self._get_param_index(key)
            bottom[idx] = denom_dict[key]

        to_return = self.test_ratio_fieller(top, bottom, null_hypothesis=null_hypothesis,
                                            top_constant=0, bottom_constant=0)

        to_return['control_baseline'] = np.dot(self.params, bottom)
        to_return['treatment_baseline'] = to_return['control_baseline'] + np.dot(top, self.params)

        return to_return

    def test_lift(self, treatment_index, null_hypothesis=0, test_level=.05):
        """Test the marginal treatment lift as a fraction of the control mean.

        Computes the ratio:

            treatment effect / E[y | X=x̄, treatment=0]

        using the Fieller delta method, where the denominator is the
        predicted control mean at the weighted-average covariate values.

        Args:
            treatment_index (int or str): Column index or name of the
                treatment variable in ``params``.
            null_hypothesis (float): Null hypothesis value for the ratio.
            test_level (float): Significance level.

        Returns:
            dict: Ratio test result including ``'test_stat'``, ``'pvalue'``,
                ``'ci_lo'``, ``'ci_hi'``, ``'control_baseline'``,
                ``'treatment_baseline'``.
        """

        treatment_index = self._get_param_index(treatment_index)

        top = np.zeros(len(self.params))
        top[treatment_index] = 1.0

        bottom = self.wexog_instrumented_means.copy()
        bottom[treatment_index] = 0.0

        to_return = self.test_ratio_fieller(top, bottom, null_hypothesis=null_hypothesis, test_level=test_level)
        to_return['control_baseline'] = np.dot(self.params, bottom)
        to_return['treatment_baseline'] = to_return['control_baseline'] + self.params.iloc[treatment_index]

        return to_return

    def test_lift_ratio(self, treatment_index_numerator, treatment_index_denominator, null_hypothesis=0,
                        test_level=.05):
        """Test the ratio of two treatment effects (SURE models only).

        Performs a Fieller test of the ratio:

            β_numerator / β_denominator

        where both coefficients are from a SURE model's joint parameter vector.
        Useful for comparing relative treatment effects across two outcomes.

        Args:
            treatment_index_numerator (int or str): Numerator coefficient.
            treatment_index_denominator (int or str): Denominator coefficient.
            null_hypothesis (float): Null hypothesis value for the ratio.
            test_level (float): Significance level.

        Returns:
            dict: Ratio test result from ``test_ratio_fieller``.

        Raises:
            Exception: If the model is not a SURE model.
        """
        if not self.is_sure:
            raise Exception('`test_lift_ratio` only applicable to SURE regressions')

        treatment_index_numerator = self._get_param_index(treatment_index_numerator)
        treatment_index_denominator = self._get_param_index(treatment_index_denominator)

        top = np.zeros(len(self.params))
        top[treatment_index_numerator] = 1

        bottom = np.zeros(len(self.params))
        bottom[treatment_index_denominator] = 1

        return self.test_ratio_fieller(top, bottom, null_hypothesis=null_hypothesis, test_level=test_level)

    @staticmethod
    def _get_rsquared_instruments(wexog, wexog_instrumented, num_instrument_cols, debug=False,
                                  columns=None):
        """Compute R² of each endogenous regressor's first-stage fit.

        Measures how well the excluded instruments explain each endogenous
        column by comparing the weighted residual variance of the instrumented
        design matrix against the original column norms.

        Args:
            wexog (ndarray): Weighted original design matrix (n × k).
            wexog_instrumented (ndarray): Weighted instrumented design matrix,
                same shape as ``wexog``.
            num_instrument_cols (int): Number of endogenous columns being
                instrumented (rightmost columns of the matrices).
            debug (bool): If True, print timing information. Default False.
            columns (array-like or None): Integer column indices to evaluate;
                if None, all columns are used.

        Returns:
            ndarray: 1-D array of first-stage R² values, one per column.
        """
        if wexog.shape != wexog_instrumented.shape:
            raise Exception("wexog and wexog_instrumented must have same shape")

        if debug:
            print("Computing Rsquared for regressors on instruments...", end='')

        if columns is not None:
            temp = np.arange(wexog.shape[1])
            wexog = wexog[:, temp[columns]]
            wexog_instrumented = wexog_instrumented[:, temp[columns]]

        _t = time.time()

        _n_temp = wexog_instrumented.shape[0]
        Rsq_temp = []
        F_temp = []

        E_wexog = np.array(wexog.sum(axis=0)).flatten() / _n_temp
        E_wexog_instrumented = np.array(wexog_instrumented.sum(axis=0)).flatten() / _n_temp

        E_wexog_sq = np.array(wexog.power(2).sum(axis=0)).flatten() / _n_temp
        E_wexog_instrumented_sq = np.array(wexog_instrumented.power(2).sum(axis=0)).flatten() / _n_temp

        E_wexog_instrumented_x_wexog = np.array(wexog_instrumented.multiply(wexog).sum(axis=0)).flatten() / _n_temp

        for i in range(wexog_instrumented.shape[1]):
            E_xy = E_wexog_instrumented_x_wexog[i]
            E_x_E_y = E_wexog[i] * E_wexog_instrumented[i]
            _cov_xy = E_xy - E_x_E_y

            s_x = np.sqrt(E_wexog_instrumented_sq[i] - E_wexog_instrumented[i] ** 2)
            s_y = np.sqrt(E_wexog_sq[i] - E_wexog[i] ** 2)

            if s_x == 0 or s_y == 0:
                if s_x == 0 and s_y == 0:
                    _rsq = 1
                else:
                    _rsq = np.nan
            else:
                _rsq = (_cov_xy / (s_x * s_y)) ** 2

            Rsq_temp.append(_rsq)
            if _rsq == 1:
                F_temp.append(np.inf)
            else:
                F_temp.append(
                    (_rsq / max(1, num_instrument_cols - 1)) / ((1.0 - _rsq) / (_n_temp - num_instrument_cols)))

        if debug:
            print("%.3f s" % (time.time() - _t))

        return Rsq_temp, F_temp

    def weak_iv_test(self):
        """Test for weak instruments (Cragg-Donald or similar).

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError()

    def recompute_cov(self, cov_type=DEFAULT_LM_COV_TYPE, cov_kwds=None, use_t=DEFAULT_LM_USE_T, debug=False,
                      save_cov_params=False, test_level=DEFAULT_LM_TEST_LEVEL, return_dense=True):
        """Recompute the variance-covariance matrix with a different estimator.

        Raises:
            NotImplementedError: Not yet implemented (requires ``keep_model=True``
                and model re-attachment logic).
        """
        raise NotImplementedError
        # if self.model is None:
        #     raise Exception("Must set `keep_model=True` to recompute covariance matrix after regression fit!")
        #
        # var_covar, num_groups, df_t_dist, small_samp_correct, cluster_name \
        #     = self.model.compute_cov_params(cov_type, cov_kwds, use_t=use_t, debug=debug, _time=None)
        #
        # if save_cov_params:
        #     self.set_cov_params(var_covar, cov_type, df_t_dist=df_t_dist, cluster_name=cluster_name,
        #                         test_level=test_level, debug=debug)
        #
        # if return_dense:
        #     return DataFrame(index=self.exog_names, columns=self.exog_names, data=var_covar.toarray())
        # else:
        #     return var_covar.copy()

    def set_cov_params(self, cov_params, cov_type=None, cov_kwds=None, ci_lo=None, ci_hi=None, test_level=None,
                       df_t_dist=None, cluster_name=None, debug=False):
        """Update the variance-covariance matrix and recompute the F-statistic.

        Delegates to the base class ``set_cov_params``, then calls
        :meth:`set_F_stat` to keep the cached F-statistic consistent with
        the new covariance.

        Args:
            cov_params: New covariance matrix.
            cov_type (str, optional): Covariance type label.
            cov_kwds (dict, optional): Covariance keyword arguments.
            ci_lo (array-like, optional): Pre-computed lower CI bounds.
            ci_hi (array-like, optional): Pre-computed upper CI bounds.
            test_level (float, optional): Significance level.
            df_t_dist (int or None): Degrees of freedom for t-tests.
            cluster_name (str, optional): Cluster variable name.
            debug (bool): Verbose output.
        """

        super().set_cov_params(
            cov_params, cov_type, cov_kwds,
            ci_lo, ci_hi, test_level, df_t_dist, cluster_name, debug)
        self.set_F_stat()

    def loglike(self, params=None):
        """Return the log-likelihood, optionally at arbitrary parameters.

        Args:
            params (array-like, optional): Parameter vector to evaluate.  If
                ``None``, returns the cached ``self.llf`` from estimation.

        Returns:
            float: Log-likelihood value.

        Raises:
            Exception: If ``params`` is not ``None`` and ``self.model`` is
                ``None`` (requires ``keep_model=True``).
        """
        if params is None:
            return self.llf
        elif self.model is None:
            raise Exception("Must set `keep_model=True` to recompute log-likelihood!")
        return self.model.loglike(params)

    def predict(self, data=None, params=None, index=None, debug=False,
                override_absorb_error=False, override_iv_error=False,
                ignore_column_mismatch=False,
                **kwargs):
        """Generate predictions from the fitted model.

        For in-sample predictions (``data=None`` and ``params=None``),
        returns ``self.fittedvalues`` without re-evaluation.  For out-of-
        sample predictions, delegates to the base-class ``predict`` which
        re-evaluates the formula on new data.

        .. note::
            Out-of-sample prediction is not yet supported for models with
            absorbed fixed effects or instrumental variables.  Set
            ``override_absorb_error=True`` or ``override_iv_error=True`` to
            bypass the guard (results may not be meaningful).

        Args:
            data (pd.DataFrame, optional): Out-of-sample data.
            params (array-like, optional): Alternative parameters to use.
            index (array-like, optional): Row index for output alignment.
            debug (bool): Verbose output.
            override_absorb_error (bool): Skip the absorb-prediction guard.
            override_iv_error (bool): Skip the IV-prediction guard.
            ignore_column_mismatch (bool): When ``True``, allow prediction when
                the out-of-sample design has fewer columns than the fitted
                model (e.g. missing fixed-effect levels). Forwarded to
                :meth:`~kanly.regression.regression_results_base.RegressionResultsBase.predict`.
            **kwargs: Forwarded to base class ``predict``.

        Returns:
            ndarray: Predicted values.

        Raises:
            NotImplementedError: If the model has absorbed FEs or instruments
                and the corresponding override flag is ``False``.
        """

        if data is None and params is None:
            return self.fittedvalues.copy()

        if self.absorb_info is not None and not override_absorb_error:
            raise NotImplementedError("Can't do prediction for models with absorbed FEs yet!")

        if self.instrument_info is not None and not override_iv_error:
            raise NotImplementedError("Can't do prediction for instrumental variables models yet!")

        return super().predict(data=data, params=params, index=index, debug=debug, check_constant_cols=False,
                               ignore_column_mismatch=ignore_column_mismatch)

    def get_eigenvals_and_condition_number(self):
        """Returns eigenvalues and condition number of X^T X"""
        self.eigenvals, self.condition_number = \
            get_eigenvals_and_condition_number_internal(self.normalized_cov_params, is_inverse=True)
        return self.eigenvals.copy(), self.condition_number

    def shapley_value(self, debug=False, sample=False, seed=0, return_full=False):
        """Decompose R² into Shapley values over formula sparse_terms (Owen-style).

        Allocates the fitted model's R² among ``exog_term_names`` (Patsy RHS
        sparse_terms such as ``'x'`` or ``'C(grp)'``, not individual dummy columns).
        Uses the already-fitted normal equations via
        :meth:`~kanly.regression.linear_models.shapley.Shapley._shapley_value_internal`
        — no refit per subset.

        **Exact mode** (``sample`` false): enumerate all non-empty term subsets
        (feasible for moderate ``p``).  **Permutation mode** (``sample=k``):
        approximate with ``k`` random term orderings.

        The intercept is always in subset models but is not a Shapley player;
        centered R² for an intercept-only model is 0, so the first marginal
        step uses baseline ``rsquared_last = 0``.

        For a new specification without an existing fit, use
        :func:`~kanly.api.shapley_value` (``Shapley.shapley_value``).

        Args:
            debug (bool): If ``True``, wrap the subset/permutation loop in
                ``tqdm`` for progress display.
            sample (bool or int, optional): Falsy for exact enumeration; a
                positive integer gives the number of random permutations.
            seed (int): RNG seed when ``sample`` is used.
            return_full (bool): If ``False`` (default), return a
                ``DataFrame`` with columns ``shapley_value`` and ``pct``
                (share of full-model R²).  If ``True``, return a dict with
                that frame plus ``fit``, ``full_model_rsquared``, and run
                metadata.

        Returns:
            pd.DataFrame or dict: See ``return_full``.

        Raises:
            Exception: If the model uses instrumental variables (no quadratic
                form for IV).

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from kanly.api import lm
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({'x': rng.normal(100), 'grp': rng.integers(0, 5, 100)})
        >>> df['y'] = 1 - 0.3 * df.x + rng.normal(100)
        >>> fit = lm('y ~ x + C(grp)', df)
        >>> fit.shapley_value()  # doctest: +SKIP
        """
        return Shapley._shapley_value_internal(
            self, debug=debug, sample=sample, seed=seed, return_full=return_full)

#
# if __name__ == '__main__':
#     import pandas as pd
#     from kanly.api import lm, LOWESS
#
#     df = pd.concat([pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [1, 4, 2, 2.05, -1]})] * 10)
#     print(lm('y~x', df, compute_eigenvalues=True))
#
#     print(LOWESS(df.y, df.x))
