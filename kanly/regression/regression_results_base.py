"""Base class for all fitted kanly regression results.

``RegressionResultsBase`` collects the parameter estimates, variance-covariance
matrix, and model metadata produced by ``LinearModelBase.fit``, and provides
a rich set of inference, summary, and diagnostic methods:

- **Inference**: confidence intervals (``conf_int``), Wald test (``wald_test``),
  F-test (``F_test``), bootstrap test (``test_from_bootstrap``),
  Benjamini-Hochberg FDR correction (``benjamini_hochberg_fdr``).
- **Summaries**: parameter table string (``get_param_table_str``), full text
  summary (``summary``), tidy summary DataFrame (``summary_df``,
  ``summary_table``).
- **Plots**: residual diagnostics (``plot_diagnostics``), fitted-vs-actual
  scatter (``plot_fitted_values``), confidence interval plot
  (``plot_confidence_intervals``).
- **Prediction**: in-sample or out-of-sample via the stored ``model``
  attribute (``predict``).

Sub-classes must implement the abstract methods ``get_result_type``,
``get_header_info_array``, ``get_result_name``, and ``get_footer_info``.
"""

from __future__ import absolute_import, print_function

import datetime
import textwrap
import warnings
from abc import abstractmethod

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from scipy.sparse import isspmatrix
from scipy.stats import t as t_dist, norm as norm_dist, multivariate_normal, multivariate_t

from kanly.regression.constants import DEFAULT_TEST_LEVEL
from kanly.regression.covariance_cluster_groups_model_base import CovarianceClusterGroupsModelBase
from kanly.regression.linear_model_base import LinearModelBase
from kanly.regression.plot_diagnostics import plot_diagnostics
from kanly.stats.statistical_tests import StatisticalTests
from kanly.testing_results_base import TestingResultsBase
from kanly.utils.function_str_to_callable import _check_func_for_test, get_key
from kanly.utils.plot_confidence_intervals import plot_normal_conf_intervals, plot_confidence_intervals
from kanly.utils.util import round_list, none_copy
import matplotlib.pyplot as plt

# Significance-star thresholds (p < threshold → add one more star).
ONE_STAR_LEVEL = .1
TWO_STAR_LEVEL = .05
THREE_STAR_LEVEL = .01
FOUR_STAR_LEVEL = .001


class RegressionResultsBase(TestingResultsBase):
    """Abstract base class for all kanly regression result objects.

    Stores estimated parameters, the variance-covariance matrix, degrees of
    freedom, and associated metadata.  Provides a complete inference and
    reporting API that concrete subclasses inherit.

    Attributes:
        nobs: Number of observations used to fit the model.
        params: ``pd.Series`` of estimated coefficients indexed by
            ``exog_names``.
        bse: ``pd.Series`` of standard errors.
        tvalues: ``pd.Series`` of t-statistics (or z-statistics).
        pvalues: ``pd.Series`` of two-sided p-values.
        crit_value: Critical value used for confidence intervals.
        ci_lo / ci_hi: Lower/upper confidence-interval bounds as numpy arrays.
        cov_type: String label for the variance estimator used.
        df_model: Model degrees of freedom (number of regressors).
        df_resid: Residual degrees of freedom.
        df_t_dist: Degrees of freedom for the t-distribution reference; may
            differ from ``df_resid`` for cluster-robust inference.
        use_t: If ``True``, t-distribution is used for inference; normal
            otherwise.
        alpha: Penalty strength(s) for penalised models; ``0`` for OLS.
        l1_ratio: L1/L2 mixing parameter for elastic net models.
    """

    def __init__(self, nobs, params, cov_params, df_model, df_resid, df_t_dist, exog_names=None, endog_name=None,
                 cov_type=None, cov_kwds=None, test_level=DEFAULT_TEST_LEVEL, use_t=True,
                 alpha=0.0, l1_ratio=0.0, specification_name=None):
        """Initialise regression results and compute inference quantities.

        If ``cov_params`` is not ``None``, immediately calls
        ``set_cov_params`` to derive standard errors, t-statistics, p-values,
        and confidence-interval bounds.

        Args:
            nobs: Number of observations.
            params: Coefficient array or list (length p).
            cov_params: (p × p) variance-covariance matrix, or ``None`` to
                defer computation.
            df_model: Degrees of freedom for the model (number of regressors,
                excluding the intercept).
            df_resid: Residual degrees of freedom (nobs - df_model - 1).
            df_t_dist: Degrees of freedom for the t-distribution; may equal
                ``df_resid`` or a smaller cluster-adjusted value.
            exog_names: List of regressor names; defaults to
                ``['x0', 'x1', …]``.
            endog_name: Response variable name; defaults to ``'y'``.
            cov_type: String label for the variance estimator; defaults to
                ``'NOT COMPUTED'`` when ``None``.
            cov_kwds: Optional dict of covariance keyword arguments.
            test_level: Significance level for confidence intervals and
                critical values; defaults to ``DEFAULT_TEST_LEVEL``.
            use_t: If ``True``, use the t-distribution for inference.
            alpha: Regularisation penalty strength (scalar or array);
                ``0`` for unpenalised models.
            l1_ratio: Elastic-net mixing parameter (scalar or array).
            specification_name: Optional human-readable model label.
        """

        if cov_kwds is None:
            cov_kwds = dict()

        self.specification_name = str(specification_name) if specification_name is not None else None
        self.date = datetime.datetime.today().strftime('%b %d, %Y')
        self.timestamp = datetime.datetime.today().strftime('%H:%M:%S')

        self.nobs = nobs
        self.alpha = alpha if isinstance(alpha, (float, int)) else np.array(alpha).flatten()
        self.l1_ratio = l1_ratio if isinstance(l1_ratio, (float, int)) else np.array(l1_ratio).flatten()

        self.use_t = use_t
        self.df_t_dist = df_t_dist

        self.df_model = df_model
        self.df_resid = df_resid

        if endog_name is None:
            endog_name = 'y'
        if exog_names is None:
            exog_names = ['x%d' % d for d in range(len(params))]
        self.endog_name = endog_name
        self.exog_names = exog_names
        self.use_t = use_t

        params = np.array(params).flatten()
        self._params = params
        self.params = Series(index=exog_names, data=none_copy(self._params))
        self.param_names = exog_names

        if cov_type is None:
            cov_type = 'NOT COMPUTED'
        self.cov_type = cov_type

        if cov_params is not None:
            self.set_cov_params(cov_params, cov_type=cov_type, test_level=test_level, df_t_dist=df_t_dist)
            self.cov_kwds = cov_kwds
        else:
            self._cov_params = None

    @staticmethod
    @abstractmethod
    def get_result_type():
        """Return a short string identifying the result type (e.g. ``'OLS'``).

        Sub-classes must implement this method; the value is used in
        summaries and for dispatch logic.
        """
        raise NotImplementedError

    def set_cov_params(self, cov_params, cov_type=None, cov_kwds=None, ci_lo=None, ci_hi=None, test_level=None,
                       df_t_dist=None, cluster_name=None, debug=False):
        """Store the variance-covariance matrix and recompute all inference quantities.

        Derives ``bse``, ``tvalues``, ``pvalues``, ``crit_value``, ``ci_lo``,
        and ``ci_hi`` from the supplied ``cov_params``.  Inference uses the
        t-distribution when ``self.use_t`` is ``True``; otherwise the standard
        normal.  All quantities are stored as both raw numpy arrays (prefixed
        with ``_``) and named ``pd.Series``.

        This method is a no-op when ``cov_params`` is ``None``.

        Args:
            cov_params: (p × p) variance-covariance matrix (dense or sparse).
            cov_type: Optional string label for the variance estimator;
                updates ``self.cov_type`` when supplied.
            cov_kwds: Optional dict of covariance keyword arguments.
            ci_lo: Optional pre-computed lower confidence-interval bound;
                when ``None`` the bound is derived from ``crit_value``.
            ci_hi: Optional pre-computed upper confidence-interval bound.
            test_level: Significance level for critical values and CI
                computation; falls back to ``DEFAULT_TEST_LEVEL`` when
                ``None``.
            df_t_dist: Updated degrees of freedom for the t-distribution;
                updates ``self.df_t_dist`` when truthy.
            cluster_name: Optional cluster variable label stored as
                ``self.cluster_name``.
            debug: Reserved for future diagnostic output.
        """
        self.cov_kwds = dict() if cov_kwds is None else cov_kwds.copy()

        if cov_params is None:
            self._cov_params = None
            return

        if test_level is None:
            test_level = DEFAULT_TEST_LEVEL
        self.test_level = test_level

        if cluster_name:
            self.cluster_name = cluster_name

        if df_t_dist:
            self.df_t_dist = df_t_dist

        if isspmatrix(cov_params):
            cov_params = cov_params.toarray()

        if cov_type is not None:
            self.cov_type = cov_type

        if (np.all(self.alpha == 0) or np.all(self.l1_ratio == 0)
                or np.all(self.alpha == None) or np.all(self.l1_ratio == None)):

            self._cov_params = none_copy(cov_params)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
                self._bse = np.sqrt(self._cov_params.diagonal())

            self._tvalues = self._params / self._bse

            if cov_type == 'NOT COMPUTED':
                self.crit_value = np.nan
                self._pvalues = self._params * np.nan

            else:

                if self.use_t:
                    try:
                        self.crit_value = t_dist.ppf(1.0 - test_level / 2, self.df_t_dist)
                        self._pvalues = 2.0 * t_dist.sf(np.abs(self._tvalues), self.df_t_dist)
                    except:
                        self.crit_value = np.nan
                        self._pvalues = np.array([np.nan] * len(self._params))
                else:
                    self.crit_value = norm_dist.ppf(1.0 - test_level / 2)
                    self._pvalues = 2.0 * norm_dist.sf(np.abs(self._tvalues))

            self.bse = Series(index=self.exog_names, data=self._bse)
            self.tvalues = Series(index=self.exog_names, data=self._tvalues)
            self.pvalues = Series(index=self.exog_names, data=self._pvalues)

            self.ci_lo = ci_lo if ci_lo else self._params - self.crit_value * self._bse
            self.ci_hi = ci_hi if ci_hi else self._params + self.crit_value * self._bse

    def cov_params(self):
        """Return the parameter variance-covariance matrix as a named DataFrame.

        Returns:
            ``pd.DataFrame`` of shape (p × p) with row and column labels from
            ``self.exog_names``.

        Raises:
            Exception: If the variance-covariance matrix has not been
                computed (i.e. ``cov_params`` was ``None`` at construction).
        """
        if hasattr(self, '_cov_params') and self._cov_params is not None:
            return DataFrame(
                index=self.exog_names.copy(),
                columns=self.exog_names.copy(),
                data=self._cov_params.copy(),
            )
        else:
            raise Exception("Model fit does not have variance-covariance computed!")

    def corr_params(self):
        """Return the parameter correlation matrix as a named DataFrame.

        Scales the variance-covariance matrix by dividing each row and
        column by the corresponding standard error, yielding a correlation
        matrix with ones on the diagonal.

        Returns:
            ``pd.DataFrame`` of shape (p × p).

        Raises:
            Exception: If the variance-covariance matrix has not been
                computed.
        """
        if hasattr(self, '_cov_params') and self._cov_params is not None:
            var_covar = self.cov_params()
            var_covar /= self.bse.values.reshape((-1, 1))
            var_covar /= self.bse.values.reshape((1, -1))
            return var_covar
        else:
            raise Exception("Model fit does not have variance-covariance computed!")

    def get_asymptotic_distribution_rv(self, parameters=None):
        """Return a scipy random variable for the (joint) asymptotic distribution of parameters.

        When ``parameters`` is a single string the marginal t- or normal
        distribution for that parameter is returned.  When ``parameters`` is
        a list (or ``None`` for all parameters) the multivariate t or
        multivariate normal is returned.

        Args:
            parameters: A parameter name string for the marginal distribution,
                a list of parameter names for the joint distribution, or
                ``None`` to use all parameters.

        Returns:
            A scipy ``rv_continuous`` or multivariate distribution object
            centred at the estimated parameters with scale/covariance given
            by the fitted variance-covariance matrix.

        Raises:
            Exception: If the variance-covariance matrix has not been
                computed.
        """
        if hasattr(self, '_cov_params') and self._cov_params is not None:
            if isinstance(parameters, str):
                p, b = self[parameters], self.bse[parameters]
                if self.use_t:
                    return t_dist(self.df_t_dist, loc=p, scale=b)
                else:
                    return norm_dist(mean=p, loc=b)
            else:
                if parameters is None:
                    parameters = self.params.index
                p = self.params.loc[parameters]
                cov = self.cov_params().loc[parameters, parameters]
                if self.use_t:
                    return multivariate_t(loc=p, shape=cov, df=self.df_t_dist, allow_singular=True)
                else:
                    return multivariate_normal(mean=p, cov=cov, allow_singular=True)
        else:
            raise Exception("Model fit does not have variance-covariance computed!")

    # NOTE: The two definitions of _conf_int and conf_int below are the FIRST
    # pair and are DEAD CODE – they are silently overridden by the second pair
    # defined later in this class.  The first `_conf_int` raises for penalised
    # models (``alpha > 0``); the second pair (the active ones) silently omit
    # that guard.  These first definitions are retained to document intent.
    def _conf_int(self, test_level=None):
        """(Dead code – overridden below) Internal CI computation with penalisation guard."""
        if self.alpha > 0:
            raise NotImplementedError("No confidence intervals for penalized regressions!")
        if not test_level:
            test_level = self.test_level

        if self.use_t:
            crit_value = t_dist.ppf(1.0 - test_level / 2, self.df_t_dist)
        else:
            crit_value = norm_dist.ppf(1.0 - test_level / 2)

        ci_lo = self._params - crit_value * self._bse
        ci_hi = self._params + crit_value * self._bse
        return ci_lo, ci_hi

    def conf_int(self, test_level=None):
        """(Dead code – overridden below) Return CI DataFrame."""
        if not test_level:
            test_level = self.test_level

        ci_lo, ci_hi = self._conf_int(test_level)
        col_lo, col_hi = '[%.4f,' % (test_level / 2), ' %.4f]' % (1 - test_level / 2)
        return pd.DataFrame(
            index=self.exog_names,
            data={col_lo: ci_lo, col_hi: ci_hi}
        )[[col_lo, col_hi]].copy()

    def wald_test(self, r_matrix=None, q=None, use_f=False):
        """Perform a Wald (linear hypothesis) test on the estimated parameters.

        Tests the null hypothesis ``R @ params == q`` using the score form
        of the Wald statistic.  Delegates to
        ``StatisticalTests.wald_test``.

        Args:
            r_matrix: (m × p) contrast/restriction matrix defining the linear
                hypotheses; ``None`` uses an identity (tests all parameters).
            q: (m,) right-hand-side vector; ``None`` defaults to a zero
                vector (testing equality to zero).
            use_f: If ``True``, divide by ``m`` and compare to the
                F-distribution; otherwise compare to χ².

        Returns:
            Test result object from ``StatisticalTests.wald_test``.
        """
        return StatisticalTests.wald_test(
            self._params, self._cov_params,
            self.df_t_dist if self.use_t else self.df_resid,
            r_matrix=r_matrix, q=q, use_f=use_f)

    def F_test(self):
        """Compute the overall F-statistic for the regression.

        Tests the joint null hypothesis that all slope coefficients are zero
        (excluding the intercept when present).  Constructs the appropriate
        contrast matrix and delegates to ``wald_test``.

        Returns:
            Test result object from ``wald_test``.

        Raises:
            Exception: For models with an implicit constant (saturated dummies)
                where the F-statistic is not well-defined.
        """
        if self.has_implicit_constant:
            raise Exception("F statistic not supported for models with implicit constant!")
        ind_to_start = self.has_intercept
        return self.wald_test(
            r_matrix=np.eye(len(self._params))[ind_to_start:, :],
            q=np.zeros(len(self._params) - ind_to_start),
            use_f=True)

    def benjamini_hochberg_fdr(self, fdr=.2):
        """Apply the Benjamini-Hochberg FDR procedure to the parameter p-values.

        Sorts parameters by p-value and flags those that pass the BH step-up
        criterion ``p_i < fdr * i / m`` where ``i`` is the p-value rank and
        ``m`` is the total number of parameters.

        Args:
            fdr: Target false-discovery rate (between 0 and 1); default 0.2.

        Returns:
            ``pd.DataFrame`` with columns ``'param'``, ``'p'``, and
            ``'stat_sig(q=<fdr>)'``, sorted by ascending p-value.
        """
        result = DataFrame(
            index=self.exog_names,
            data={'param': self.params,
                  'p': self.pvalues}
        )
        result.sort_values(by='p', inplace=True)
        result['stat_sig(q=%.3f)' % fdr] = result['p'] < fdr * np.arange(1, len(result) + 1) / len(result)
        return result

    def _conf_int(self, test_level=None):
        """Compute confidence-interval bounds as raw arrays.

        This is the active definition (the earlier ``_conf_int`` above is dead
        code).  Uses ``t_dist`` or ``norm_dist`` depending on ``self.use_t``;
        falls back to ``nan`` critical value if the t-distribution call fails.

        Args:
            test_level: Significance level; defaults to ``self.test_level``.

        Returns:
            Tuple of (ci_lo, ci_hi) numpy arrays of length p.
        """
        if not test_level:
            test_level = self.test_level

        if self.use_t:
            try:
                crit_value = t_dist.ppf(1.0 - test_level / 2, self.df_t_dist)
            except:
                crit_value = np.nan
        else:
            crit_value = norm_dist.ppf(1.0 - test_level / 2)
        ci_lo = self._params - crit_value * self._bse
        ci_hi = self._params + crit_value * self._bse
        return ci_lo, ci_hi

    def conf_int(self, test_level=None):
        """Return confidence intervals for all parameters as a DataFrame.

        This is the active definition (the earlier ``conf_int`` above is dead
        code).  Column labels reflect the tail probabilities, e.g.
        ``'[0.0250,'`` and ``' 0.9750]'`` for a 95 % interval.

        Args:
            test_level: Significance level; defaults to ``self.test_level``.

        Returns:
            ``pd.DataFrame`` with two columns (lower and upper CI bounds) and
            row labels from ``self.exog_names``.
        """
        if not test_level:
            test_level = self.test_level

        ci_lo, ci_hi = self._conf_int(test_level)
        col_lo, col_hi = '[%.4f,' % (test_level / 2), ' %.4f]' % (1 - test_level / 2)
        return DataFrame(
            index=self.exog_names,
            data={col_lo: ci_lo, col_hi: ci_hi}
        )[[col_lo, col_hi]]

    def test_from_bootstrap(self, func):
        """Test a (possibly non-linear) hypothesis using bootstrapped parameter draws.

        Applies ``func`` to the bootstrapped parameter samples to obtain a
        sampling distribution for the quantity of interest, then computes a
        two-sided bootstrap p-value relative to a null hypothesis of zero.

        Args:
            func: Callable mapping a parameter vector (or matrix of vectors)
                to a scalar test statistic.  May also be a string key
                accepted by ``_check_func_for_test``.

        Returns:
            Bootstrap test result from ``StatisticalTests._test_from_samples``.

        Raises:
            Exception: If the model was not fitted with ``cov_type='BOOTSTRAP'``
                (i.e. ``bootstrapped_params`` is ``None``).
        """
        if self.bootstrapped_params is None:
            raise Exception("Can only use this function if you used `covtype='BOOTSTRAP`!")

        func = _check_func_for_test(func, self.param_names)
        try:
            func_samples = func(self.bootstrapped_params)
            assert len(func_samples) == len(self.bootstrapped_params)
        except:
            func_samples = np.array([func(x) for x in self.bootstrapped_params])
        return StatisticalTests._test_from_samples(func(self._params), func_samples, null_hypothesis=0,
                                                   test_level=.05, tail='two')

    def summary_df(self, test_level=DEFAULT_TEST_LEVEL):
        """Return a tidy DataFrame of coefficients, standard errors, and inference.

        Builds a ``pd.DataFrame`` with columns ``'coef'``, ``'std err'``,
        ``'t'``, ``'p>|t|'``, lower and upper CI bounds, and a significance
        ``'stars'`` column.  Columns other than ``'coef'`` are omitted when
        the variance-covariance matrix has not been computed.

        Args:
            test_level: Significance level for CI bounds; defaults to
                ``DEFAULT_TEST_LEVEL``.

        Returns:
            ``pd.DataFrame`` indexed by ``self.exog_names``.
        """
        result = pd.DataFrame(
            index=self.params.index,
            data={'coef': self.params},
        )

        if self._cov_params is not None:
            result['std err'] = self._bse

            result['t'] = self._tvalues
            result['p>|t|'] = 2 * self._pvalues

            ci = self._conf_int(test_level)
            result['[%.3f, ' % (test_level / 2)] = ci[0]
            result['%.3f]' % (1.0 - test_level / 2)] = ci[1]
            star = '*'
            result['stars'] = [
                star * 4 if p < FOUR_STAR_LEVEL
                else (star * 3 + ' ' if p < THREE_STAR_LEVEL
                      else (star * 2 + '  ' if p < TWO_STAR_LEVEL
                            else (star + '   ' if p < ONE_STAR_LEVEL else '')))
                for p in self._pvalues]

        return result.copy()

    def set_bootstrapped_params(self, bootstrapped_params, cov_string=None):
        """Store bootstrapped parameter samples for bootstrap inference.

        Args:
            bootstrapped_params: Array of shape (n_samples × p) containing
                the bootstrap draws of the coefficient vector.
            cov_string: Optional descriptive string for the bootstrap
                procedure (stored and displayed in summaries).
        """
        self.bootstrapped_params = bootstrapped_params
        self.cov_string = cov_string

    def set_properties_from_model(self, model, keep_model=True):
        """Copy metadata attributes from a fitted ``LinearModelBase`` onto this results object.

        Transfers all instance attributes from ``model`` except the large
        data arrays (``exog``, ``endog``, ``instruments``, ``absorb``,
        ``weights``, ``data``) and the ``specification_name`` / ``method``
        labels (which are set independently on the results object).
        ``valid_obs_rows`` and ``index`` are always copied explicitly.

        Args:
            model: A fitted ``LinearModelBase`` instance, or ``None`` to
                indicate that no model reference should be stored.
            keep_model: If ``True`` (default), store a reference to ``model``
                as ``self.model`` (enabling out-of-sample ``predict`` calls).
                Set to ``False`` to save memory.

        Raises:
            Exception: If ``model`` is not a ``LinearModelBase`` instance.
        """
        if model is None:
            self.model = None
            self.keep_model = False
            return

        if not isinstance(model, LinearModelBase):
            raise Exception

        setattr(self, 'valid_obs_rows', np.array(model.valid_obs_rows))
        setattr(self, 'index', np.array(model.index) if model.index is not None else None)

        # Copy all model attributes except the large array data members that
        # would waste memory, and the fields that are already set on the
        # results object directly.
        for attr in model.__dict__.keys():
            if attr not in ['exog', 'endog', 'instruments', 'absorb', 'weights', 'data', 'specification_name', 'method']:
                v = getattr(model, attr)
                try:
                    v = v.copy()
                except:
                    pass
                setattr(self, attr, v)

        self.model: CovarianceClusterGroupsModelBase = model if keep_model else None
        self.keep_model = keep_model

    @staticmethod
    def get_header_info_str(header_info_arr):
        """Format a two-column label/value array into a two-column header string.

        Splits the array in half and pairs each entry from the first half with
        the corresponding entry from the second half to produce a compact
        two-column layout similar to statsmodels summary headers.

        Args:
            header_info_arr: Array-like of shape (n, 2) where column 0 holds
                labels and column 1 holds values.

        Returns:
            Tuple of (header_string, width) where ``header_string`` is the
            multi-line formatted text and ``width`` is the character width of
            the first row.
        """
        header_info_arr = np.asarray(header_info_arr)
        header_info_series = pd.Series(header_info_arr[:, 1], index=header_info_arr[:, 0])
        header_info_strs = header_info_series.to_string(dtype=False).split('\n')
        if len(header_info_strs) % 2:
            header_info_strs.append('')
        num_info = len(header_info_strs) // 2
        head_info_str = ''
        for j in range(num_info):
            head_info_str += header_info_strs[j] + '    ' + header_info_strs[num_info + j]
            if j == 0:
                length = len(head_info_str)
            if j < num_info - 1:
                head_info_str += '\n'

        return head_info_str, length

    def get_param_table_str(self, test_level=None, param_sigfigs=5, show_stars=True, show_t=True, show_p_values=True,
                            show_std_err=True, show_CI=True, ci_sigfigs=5, t_decimals=2, std_sigfigs=5,
                            bh_correction_fdr=None,
                            uniform_decimal_output=False, show_only_stat_sig=False, show_only_non_zero=False,
                            parameter_subset=None):
        """Format the parameter table as a printable string with optional filtering.

        Constructs a column-formatted table from ``summary_df`` with optional
        standard errors, t-statistics, p-values, confidence intervals,
        significance stars, and Benjamini-Hochberg FDR indicators.  Rows with
        zero coefficients or non-significant p-values can be suppressed.

        Args:
            test_level: Significance level; falls back to ``self.test_level``
                or ``DEFAULT_TEST_LEVEL``.
            param_sigfigs: Significant figures for coefficient values.
            show_stars: Include the ``'stars'`` significance column.
            show_t: Include the t-statistic column.
            show_p_values: Include the ``'p>|t|'`` column.
            show_std_err: Include the standard-error column.
            show_CI: Include lower and upper CI columns.
            ci_sigfigs: Significant figures for CI bounds.
            t_decimals: Decimal places for t-statistics.
            std_sigfigs: Significant figures for standard errors.
            bh_correction_fdr: If not ``None``, add a BH column at this FDR
                level; ``show_only_stat_sig`` then filters by this column.
            uniform_decimal_output: If ``True``, use uniform decimal places
                rather than significant figures.
            show_only_stat_sig: Suppress rows that are not statistically
                significant.
            show_only_non_zero: Suppress rows where the coefficient is
                effectively zero (``|coef| < 1e-10``).
            parameter_subset: Optional iterable of parameter names to retain;
                all others are suppressed.

        Returns:
            Tuple of (table_string, table_width, suppress_string) where
            ``table_string`` is the formatted ASCII table, ``table_width``
            is its character width, and ``suppress_string`` is a message
            describing suppressed rows (or ``None``).
        """

        if test_level is None:
            if hasattr(self, 'test_level'):
                test_level = self.test_level
            else:
                test_level = None

        tbl = self.summary_df(test_level)
        tbl_new = tbl.loc[:, []]
        tbl_new['coef'] = round_list(tbl['coef'], param_sigfigs, uniform_decimal_output)

        if self.did_compute_var_covar():
            if show_stars:
                tbl_new[' '] = tbl['stars']
            if show_std_err:
                tbl_new['std err'] = round_list(tbl['std err'], std_sigfigs, uniform_decimal_output)
            if show_t:
                tbl_new['t'] = tbl['t'].round(t_decimals).values
            if show_p_values:
                tbl_new['p>|t|'] = [z if not np.isfinite(z) or z > .001 else "<0.001" for z in self._pvalues.round(3)]
            if show_CI:
                tbl_new[f'[{"%.3f" % (test_level / 2)}, '] \
                    = round_list(tbl[f'[{"%.3f" % (test_level / 2)}, '], ci_sigfigs, uniform_decimal_output)
                tbl_new[f'{"%.3f" % (1 - test_level / 2)}]'] \
                    = round_list(tbl[f'{"%.3f" % (1 - test_level / 2)}]'], ci_sigfigs, uniform_decimal_output)
            if bh_correction_fdr is not None:
                if bh_correction_fdr > 1 or bh_correction_fdr < 0:
                    raise Exception("`bh_correction_fdr` must be in [0,1]")
                fdr_table = self.benjamini_hochberg_fdr(bh_correction_fdr)
                tbl_new['Benj.-Hoch.'] = fdr_table.iloc[:, 2]
                if show_only_stat_sig:
                    tbl_new = tbl_new.loc[tbl_new['Benj.-Hoch.']]
            else:
                if show_only_stat_sig:
                    tbl_new = tbl_new.loc[self.pvalues < test_level]

        if show_only_non_zero:
            tbl_new = tbl_new[np.abs(tbl.coef) > 1e-10]

        suppress_string = None
        if len(tbl_new) < len(tbl):
            suppress_string = f'{len(tbl) - len(tbl_new)} parameter estimates suppressed in output '
            if show_only_non_zero and show_only_stat_sig:
                suppress_string += 'that are either zero or not stat sig.'
            elif show_only_non_zero:
                suppress_string += 'that are zero.'
            else:
                suppress_string += 'that are not stat sig.'

        if parameter_subset is not None:
            tbl_new2 = tbl_new.loc[tbl_new.index.isin(parameter_subset)].copy()
        else:
            tbl_new2 = tbl_new

        if len(tbl_new2) < len(tbl_new):
            if suppress_string is None:
                suppress_string = ''
            else:
                suppress_string += '\n'
            suppress_string += f'suppressed output for {len(tbl_new) - len(parameter_subset)} many parameters\n'

        param_strs = tbl_new2.to_string().split('\n')
        width = len(param_strs[0])

        return (
            ('═' * width + "\n" +
             param_strs[0] + '\n' +
             '─' * width + '\n' +
             '\n'.join(param_strs[1:]) + '\n' +
             '═' * width),
            width,
            suppress_string
        )

    # @abstractmethod
    # def predict(self, data=None, params=None, integer_index=None, *args, **kwargs):
    #     raise NotImplementedError

    @abstractmethod
    def get_header_info_array(self):
        """Return the header metadata array for this result type.

        Sub-classes must implement this method and return an (n × 2)
        array/list of (label, value) pairs that will be formatted by
        ``get_header_info_str`` into the summary header block (e.g. nobs,
        R-squared, F-statistic, log-likelihood, etc.).

        NOTE: The body of this method currently reads ``raise not
        NotImplementedError``, which evaluates to ``raise True`` and raises
        a ``TypeError`` at runtime rather than the intended
        ``NotImplementedError``.  This is a latent bug in sub-classes that
        forget to override this method.
        """
        raise not NotImplementedError  # BUG: should be `raise NotImplementedError`

    def get_title(self):
        """Construct the summary title string.

        Concatenates the result name (from ``get_result_name``) with the
        ``specification_name`` on a new line when a specification name is
        present.

        Returns:
            str: Title string for the summary output.
        """
        title = self.get_result_name()
        if self.specification_name is not None:
            title += "\n" + self.specification_name
        return title

    @abstractmethod
    def get_result_name(self):
        """Return a short human-readable name for this result type.

        Used as the first line of the summary title block.
        Sub-classes must implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def get_footer_info(self, *args, **kwargs):
        """Return the footer string for the summary output.

        Sub-classes must implement this method and return a string with
        additional model-specific statistics or notes (e.g. R-squared,
        log-likelihood, covariance-type note).
        """
        raise NotImplementedError

    def get_dep_variable_str(self):
        """Return a 'Dep. Variable: <name>' label string for the summary.

        Returns:
            str: Formatted dependent variable label.
        """
        return 'Dep. Variable: ' + self.endog_name

    def get_formula_str(self):
        """Return a wrapped formula string for the summary footer.

        Looks up the formula first on ``self``, then on ``self.model``.
        Long formulas are truncated to 100 leading + 100 trailing characters
        with an ellipsis in the middle, then word-wrapped to 60 characters.

        Returns:
            str: Formatted formula line, or an empty string when no formula
            is available.
        """
        if hasattr(self, 'formula') and self.formula is not None:
            formula = self.formula
        elif hasattr(self, 'model') and self.model is not None and hasattr(self.model,
                                                                           'formula') and self.model.formula is not None:
            formula = self.model.formula
        else:
            return ''

        if len(formula) > 300:
            formula = formula[:100] + "  ...  " + formula[-100:]

        return 'formula:  ' + '\n'.join(textwrap.wrap(formula, width=60, subsequent_indent=' ' * 10))

    def summary(self, show_only_stat_sig=False, show_std_err=True, show_stars=True, show_t=True, show_CI=True,
                show_p_values=True, test_level=None,
                t_decimals=2, std_sigfigs=4, param_sigfigs=4, ci_sigfigs=4,
                bh_correction_fdr=None, show_formula=True, uniform_decimal_output=False, show_only_non_zero=False,
                parameter_subset=None
                ):
        """Build and return the full text summary of the regression results.

        Assembles a multi-section summary string containing the title, header
        metadata, the parameter table, optional diagnostic statistics, the
        formula, footer inference notes, and a kanly version string.

        Args:
            show_only_stat_sig: If ``True``, suppress rows with p-value
                above ``test_level`` (or BH threshold when
                ``bh_correction_fdr`` is set).
            show_std_err: Include the standard-error column.
            show_stars: Include the significance-star column.
            show_t: Include the t-statistic column.
            show_CI: Include confidence-interval columns.
            show_p_values: Include the p-value column.
            test_level: Significance level; falls back to ``self.test_level``
                or ``DEFAULT_TEST_LEVEL``.
            t_decimals: Decimal places for t-statistics.
            std_sigfigs: Significant figures for standard errors.
            param_sigfigs: Significant figures for coefficients.
            ci_sigfigs: Significant figures for CI bounds.
            bh_correction_fdr: If not ``None``, add a Benjamini-Hochberg FDR
                column at this level.
            show_formula: If ``True``, include the formula string in the
                footer.
            uniform_decimal_output: Use uniform decimal places instead of
                significant figures.
            show_only_non_zero: Suppress rows where the coefficient is
                effectively zero.
            parameter_subset: Optional list of parameter names to retain.

        Returns:
            str: Fully formatted summary text.
        """

        old_pd_col_width = pd.options.display.max_colwidth
        pd.options.display.max_colwidth = 100

        if test_level is None:
            if hasattr(self, 'test_level'):
                test_level = self.test_level
            else:
                test_level = DEFAULT_TEST_LEVEL

        title = self.get_title()
        dep_variable_str = self.get_dep_variable_str()

        param_table_str, param_width, suppress_str = self.get_param_table_str(
            test_level=test_level, param_sigfigs=param_sigfigs, show_stars=show_stars, show_t=show_t,
            show_std_err=show_std_err,
            show_p_values=show_p_values, show_CI=show_CI, ci_sigfigs=ci_sigfigs, t_decimals=t_decimals,
            show_only_stat_sig=show_only_stat_sig,
            std_sigfigs=std_sigfigs, bh_correction_fdr=bh_correction_fdr, uniform_decimal_output=uniform_decimal_output,
            show_only_non_zero=show_only_non_zero, parameter_subset=parameter_subset)

        head_info_str, header_width = self.get_header_info_str(self.get_header_info_array())
        width = max(max(header_width, param_width), 60)

        foot_info = self.get_footer_info(test_level=test_level)

        version_str = self.get_version_str(width)

        fdr_str = ''
        if bh_correction_fdr is not None and self.did_compute_var_covar():
            fdr_str = '\nBenjamini-Hochberg correction at FDR=%.3f' % bh_correction_fdr

        if show_formula:
            formula_str = self.get_formula_str()
            if formula_str is None:
                formula_str = ''
        else:
            formula_str = ''

        dbl_bar = '═' * width
        pd.options.display.max_colwidth = old_pd_col_width

        if hasattr(self, 'get_diagnostic_stats'):
            statss = self.get_diagnostic_stats()
            diagnostic_str, _ = self.get_header_info_str(np.array([(k, v) for k, v in statss.items()]))
            diagnostic_str += '\n' + '═' * param_width + '\n'
        else:
            diagnostic_str = ''

        to_return = (
                dbl_bar + "\n" +
                title + "\n" +
                dbl_bar + "\n" +
                "\n" +
                dep_variable_str + "\n" +
                "\n" +
                head_info_str + "\n" +
                "\n" +
                param_table_str + "\n" +
                diagnostic_str +
                "\n" +
                ((suppress_str + "\n") if suppress_str is not None else '') +
                formula_str + "\n" +
                "\n" +
                foot_info + "\n" +
                fdr_str + "\n" +
                version_str
        ).replace('\n\n\n', '\n\n')

        return to_return

    def did_compute_var_covar(self):
        """Return ``True`` if the variance-covariance matrix is available.

        Returns:
            bool: Whether ``_cov_params`` has been set to a non-``None``
            value.
        """
        return self._cov_params is not None

    def get_inference_string(self, **kwargs):
        """Build a human-readable description of the inference method used.

        Describes whether a t- or normal distribution was used, the degrees
        of freedom, the significance level, and (when available) the
        covariance estimator description string.

        Args:
            **kwargs: Accepts ``test_level`` (float) to include in the
                description string.

        Returns:
            str: Multi-line inference description, or an empty string when
            the variance-covariance matrix has not been computed.
        """
        to_ret = ''
        if self.did_compute_var_covar():
            if self.use_t and self.df_t_dist is not None:
                to_ret += "\nUsed t distribution with %d df at test level %.4f." \
                          % (self.df_t_dist, kwargs.get('test_level', np.nan))
            else:
                to_ret += "\nUsed Normal distribution with %d df at test level %.4f." \
                          % (self.df_resid, kwargs.get('test_level', np.nan))

            if hasattr(self, 'cov_string') and self.cov_string is not None:
                to_ret += '\n' + self.cov_string

            # if BOOTSTRAP in self.cov_type:
            #     if hasattr(self, 'cov_string') and self.cov_string is not None:
            #         to_ret += '\n' + self.cov_string
            #     else:
            # #         to_ret += "\nConverged on %d Bootstrap repetitions" % len(self.bootstrapped_params)
            # elif self.cov_type == CLUSTER:
            #     to_ret += "\nVariance clustered on %s" % (
            #         "'%s'" % self.cluster_name if self.cluster_name is not None else "<unknown>")

        return to_ret

    def predict(self, data=None, params=None, debug=False, index=None,
                ignore_column_mismatch=False,
                *args, **kwargs):
        """Generate predictions from the fitted model.

        Returns in-sample fitted values when both ``data`` and ``params`` are
        ``None``.  Otherwise delegates to ``self.model.predict`` using the
        stored or supplied parameter vector.

        Args:
            data: New data for out-of-sample prediction; ``None`` for
                in-sample predictions.
            params: Optional alternative parameter vector; defaults to
                ``self.params``.
            debug: If ``True``, pass debug flag to the underlying model.
            index: Row-selector applied to ``data`` (DataFrame only).
            ignore_column_mismatch (bool): Passed to the model's
                ``get_linear_predictor`` / ``predict``. When ``True``, allows
                out-of-sample prediction when the new design has fewer columns
                than the fitted model (e.g. missing fixed-effect levels).
                See :meth:`~kanly.regression.linear_model_base.LinearModelBase.get_linear_predictor`.
            *args: Forwarded to the model's ``predict`` method.
            **kwargs: Forwarded to the model's ``predict`` method.

        Returns:
            1-D numpy array of predicted values.
        """
        if params is None and data is None:
            return self.fittedvalues.copy()

        if params is None:
            params = self.params.copy()

        return self.model.predict(params, data=data, index=index, debug=debug,
                                  ignore_column_mismatch=ignore_column_mismatch,
                                  *args, **kwargs)

    def summary_table(self, test_level=DEFAULT_TEST_LEVEL):
        """Return a compact tidy DataFrame with renamed columns.

        Equivalent to ``summary_df`` but with the ``'stars'`` column removed
        and columns renamed to short programmatic labels (``'param'``,
        ``'bse'``, ``'t'``, ``'p'``, ``'ci_lo'``, ``'ci_hi'``).

        Args:
            test_level: Significance level for CI bounds.

        Returns:
            ``pd.DataFrame`` indexed by ``self.exog_names`` with 6 columns.
        """
        tbl = self.summary_df(test_level=test_level)
        del tbl['stars']
        tbl.columns = ['param', 'bse', 't', 'p', 'ci_lo', 'ci_hi']
        return tbl

    # def get_results_indices_to_original_integer_indices(self):
    #     """Integer indexes of results mapped to original rows of data"""
    #     if not hasattr(self, 'model') or self.model is None:
    #         raise Exception("Must have `model` attribute to access original row indices!")
    #     n = len(self.model.valid_obs_rows)
    #     print(self.model.valid_obs_rows)
    #     return np.arange(n)[self.model.valid_obs_rows]

    def plot_diagnostics(self, figsize=(6, 5), dpi=130, show=True, maxlags=15):
        """Render the six-panel regression diagnostic plot for this result.

        Delegates to ``kanly.regression.plot_diagnostics.plot_diagnostics``
        after extracting residuals, fitted values, and the residual variance
        estimate from this result object.

        Args:
            figsize: Matplotlib figure size tuple; default ``(6, 5)``.
            dpi: Figure resolution; default 130.
            show: If ``True``, call ``plt.show()`` immediately.
            maxlags: Maximum autocorrelation lag displayed in the correlogram.

        Returns:
            ``matplotlib.figure.Figure``: The diagnostic figure.
        """
        llb = getattr(self, 'loglikelihood_burn', 0)
        scale = getattr(self, 'scale', 1.0)
        resid = self.resid[llb:]
        fittedvalues = self.fittedvalues[llb:]
        return plot_diagnostics(resid, fittedvalues, scale, figsize, dpi, show, maxlags)

    def get_key(self, key, debug=False):
        """Resolve a parameter key or callable to a canonical parameter name.

        Delegates to ``kanly.utils.function_str_to_callable.get_key``.
        Useful for looking up a parameter when the caller may supply either
        a raw name string or a transformation function.

        Args:
            key: Parameter name string or callable.
            debug: If ``True``, emit debug output.

        Returns:
            Resolved parameter name string.
        """
        return get_key(key, self.param_names, debug)

    def plot_confidence_intervals(self, params=None, labels=None, title=None, figsize=(10, 5),
                                  dpi=130, show=False, level=.95, plot_horizontal_line=False,
                                  midpoint=None, sample_size=10_000):
        """Symmetric (Equitail) Confidence Intervals
        appeals to CLT on asymptotics
        midpoint in ('mean', 'median')"""

        if labels is None:
            labels = params.copy()

        if midpoint is None:
            midpoint = 'mean'
        else:
            midpoint = midpoint.lower()
            assert midpoint in ('mean', 'median')

        if params is None:
            params = list(self.params.index)
        params = [self.get_key(p) for p in params]

        if np.any([p not in self.param_names for p in params]):
            sample = np.random.multivariate_normal(mean=self.params, cov=self.cov_params(), size=sample_size)

        conf_ints = []
        point_estims = []
        cv = -norm_dist.ppf((1 - level) / 2)
        for p in params:
            if p in self.param_names:
                b, s = self.params[p], self.bse[p]
                conf_ints.append((b - cv * s, b + cv * s))
                point_estims.append(b)
            else:
                x = np.array([p(s) for s in sample])
                conf_ints.append(np.quantile(x, [(1 - level) / 2, 1 - (1 - level) / 2]))
                point_estims.append(x.mean() if midpoint == 'mean' else np.median(x))

        return plot_confidence_intervals(conf_ints, point_estims, labels=labels, title=title, figsize=figsize,
                                         dpi=dpi, show=show, level=level, plot_horizontal_line=plot_horizontal_line)

    def plot_fitted_values(self, dpi=130, figsize=(5, 5), title='fitted vs actual', alpha=.5, show=False,
                           xlabel='Actuals', ylabel='Fitted', fontsize=14):
        """Plot fitted values against actual observed values.

        Creates a scatter plot of ``fittedvalues`` (y-axis) vs actual
        ``fittedvalues + resid`` (x-axis) with a 45-degree reference line.
        A well-fitting model will cluster tightly along the diagonal.

        Args:
            dpi: Figure resolution; default 130.
            figsize: Figure size tuple; default ``(5, 5)``.
            title: Plot title string.
            alpha: Scatter point transparency; default 0.5.
            show: If ``True``, call ``plt.show()`` immediately.
            xlabel: Label for the x-axis (actuals).
            ylabel: Label for the y-axis (fitted values).
            fontsize: Font size for axis labels and title.

        Returns:
            ``matplotlib.figure.Figure``: The scatter figure.

        Raises:
            Exception: If the result object does not have ``fittedvalues``
                and ``resid`` attributes.
        """
        fig = plt.figure(dpi=dpi, figsize=figsize)
        if hasattr(self, 'fittedvalues') and hasattr(self, 'resid'):
            y = self.fittedvalues + self.resid
            plt.scatter(y, self.fittedvalues, alpha=alpha)
            l,u = min(y), max(y)
            plt.plot([l,u], [l,u], lw=1.5, c='k')
            plt.xlabel(xlabel, fontsize=fontsize)
            plt.ylabel(ylabel, fontsize=fontsize)
            plt.title(title, fontsize=fontsize)
        else:
            raise Exception(f'Object of type {type(self)} does not have `fittedvalues` or `resid` attributes!')
        if show:
            plt.show()
        return fig
