"""Public API for sparse quantile regression via IRLS.

``SparseQuantileRegressionModel`` is the primary entry point.  It extends
``LinearModelBase`` and exposes two static convenience wrappers:

- ``qr(formula, data, tau, ...)``  — patsy formula API (mirrors ``kanly.api.qr``)
- ``QR(endog, exog, tau, ...)``    — raw array API for pre-built matrices

The ``fit`` method handles:
- Single-tau and multi-tau (iterable) estimation with warm-starts
- Bootstrap covariance estimation (Bayesian / block)
- Analytical IID / robust covariance via ``get_var_covar``
- IV estimation via ``qr_internal``'s control-function path (experimental)
"""
from __future__ import absolute_import, print_function

import time
import warnings

from pandas import DataFrame

from kanly.bootstrap.bootstrap import (do_bootstrap2, DEFAULT_BB_METHOD, DEFAULT_BB_ALPHA, DEFAULT_BB_MAX_PROCESSES)
from kanly.formula.data_getter import SparseDataGetter
from kanly.formula.keys import (EXOG_KEY, ENDOG_KEY, VALID_OBS_ROWS_KEY, NULL_ROWS_INFO_DICT_KEY, HAS_INTERCEPT_KEY,
                                HAS_IMPLICIT_CONSTANT_KEY, INSTRUMENTS_KEY,
                                FORMULA_DESIGN_INFO_KEY)  # , COV_GROUPS_KEY, COV_GROUPS_NAME_KEY)
from kanly.regression.linear_model_base import LinearModelBase
from kanly.regression.linear_models.quantile_regression.constants import (DEFAULT_QR_MIN_RESID_CLIP, DEFAULT_QR_SMOOTHING_K, DEFAULT_QR_X_TOL,
                                                               DEFAULT_QR_F_TOL, DEFAULT_QR_MAX_ITER, DEFAULT_QR_COV_TYPE,
                                                               DEFAULT_QR_LINE_SEARCH, DEFAULT_QR_USE_T, DEFAULT_QR_TEST_LEVEL,
                                                               DEFAULT_QR_LOSS_FUNCTION, DEFAULT_QR_BOOTSTRAP_N_SAMPLES,
                                                               QR_COV_TYPES, QR_COV_TYPE_BOOTSTRAP, DEFAULT_QR_RESIDUAL_INCLUSION,
                                                               DEFAULT_QR_RESIDUAL_INCLUSION_ORDER, DEFAULT_QR_DENSE_THRESHOLD_MB)
from kanly.regression.linear_models.quantile_regression.qr_internal import qr_internal, get_var_covar
from kanly.regression.linear_models.quantile_regression.regression_results import SparseQuantileRegressionResults
from kanly.regression.linear_models.sparse_iv_first_stage2 import convert_exog_col_map_to_col_names2
from kanly.regression.cov_types import check_cov_kwds


class SparseQuantileRegressionModel(LinearModelBase):
    """Sparse quantile regression model using Iteratively Reweighted Least Squares.

    Extends ``LinearModelBase`` and supports formula-based and matrix-based
    estimation of quantile regression models.  Supports WLS-quantile
    (observation weights), IV via the control-function approach (experimental),
    and multiple covariance estimators (IID, robust, bootstrap).

    Use the static entry points ``qr`` (formula) or ``QR`` (matrix) rather
    than constructing this class directly.

    Examples
    --------
    Median regression (``tau=0.5``):

    >>> import numpy as np, pandas as pd
    >>> from kanly.api import qr
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({'x': rng.normal(size=500)})
    >>> df['y'] = 1.0 + 2.0 * df['x'] + rng.standard_cauchy(size=500) * 0.1
    >>> fit_med = qr('y ~ x', df, tau=0.5)                 # doctest: +SKIP

    Multiple quantiles in one call returns a dict ``{tau: results}``:

    >>> fits = qr('y ~ x', df, tau=[0.1, 0.5, 0.9])        # doctest: +SKIP
    >>> fits[0.9].params                                    # doctest: +SKIP

    Bootstrap covariance is recommended for QR (IID asymptotic SEs assume
    the conditional density at zero is correctly estimated):

    >>> fit_bs = qr('y ~ x', df, tau=0.5,                  # doctest: +SKIP
    ...             cov_type='bootstrap',
    ...             cov_kwds={'n_samples': 500, 'method': 'bayesian'})

    Aliases: ``quantreg`` (lowercase) and ``QUANTREG`` (matrix). See
    ``examples/regression/linear_models/quantile_regression/`` for
    runnable scripts.
    """

    def __init__(self, endog, exog, add_intercept, formula_design_info, has_intercept, has_implicit_constant, endog_name, exog_names,
                 index, valid_obs_rows, null_rows_info_dict, specification_name,
                 model_elapsed, instruments=None, instrument_names=None, cov_groups=None, cov_groups_name=None):
        """Initialise the quantile regression model from pre-parsed matrices.

        All arguments are forwarded to ``LinearModelBase.__init__``.  The
        estimation method is hardcoded to ``'IRLS'``.

        Args:
            endog (sparse or ndarray): Response vector, shape (n, 1).
            exog (sparse): Design matrix, shape (n, p).
            add_intercept (bool): Whether an intercept was added by patsy.
            formula (str or None): Original patsy formula string, or None for
                the matrix API.
            from_formula (bool): True if the model was built from a formula.
            has_intercept (bool): Whether the design matrix contains an explicit
                intercept column.
            has_implicit_constant (bool): True when an implicit constant is
                present (e.g. all-ones absorbed group).
            endog_name (str): Name of the response variable.
            exog_names (list of str): Names of all regressor columns.
            index (array-like or None): Row index for alignment.
            valid_obs_rows (array-like): Boolean or integer mask of valid rows
                after dropping NaN observations.
            null_rows_info_dict (dict): Metadata about dropped rows (for
                summary reporting).
            specification_name (str or None): Optional label for the model
                specification (shown in the summary footer).
            model_elapsed (float): Time in seconds spent building the model
                object (formula parsing / data preparation).
            instruments (sparse or None): Instrument matrix for IV, shape
                (n, q).
            instrument_names (list of str or None): Column names of the
                instrument matrix.
            cov_groups (array-like or None): Cluster group labels for clustered
                SEs (reserved; not yet wired into QR).
            cov_groups_name (str or None): Name of the cluster-group variable.
        """
        super().__init__(endog, exog, add_intercept, has_intercept, has_implicit_constant, formula_design_info,
                         endog_name=endog_name, exog_names=exog_names,
                         instruments=instruments, instrument_names=instrument_names,
                         index=index, valid_obs_rows=valid_obs_rows,
                         null_rows_info_dict=null_rows_info_dict, method='IRLS',
                         specification_name=specification_name,
                         model_elapsed=model_elapsed, cov_groups=cov_groups, cov_groups_name=cov_groups_name)

    @staticmethod
    def build_model_from_formula(formula, data, index=None, debug=False, specification_name=None,
                                 drop_1_for_FE=True, check_constant_cols=True,
                                 cov_groups=None):
        """Parse a patsy formula and construct a ``SparseQuantileRegressionModel``.

        Handles ``dict``-to-DataFrame conversion, instrument extraction from
        the ``|`` syntax (e.g. ``'y ~ x | z'``), and IV warning.

        Args:
            formula (str): Patsy formula string.  Use ``$`` for weights,
                ``|`` for excluded instruments, and standard patsy transforms
                (``C()``, ``I()``, ``poly()``).
            data (DataFrame or dict): Data source.  A dict is converted to a
                DataFrame automatically.
            index (array-like or None): Optional row index for result
                alignment.
            debug (bool): If True, print the constructed model object after
                building.  Defaults to False.
            specification_name (str or None): Optional label stored in the
                results footer.
            drop_1_for_FE (bool): Whether to drop one column per categorical
                group to avoid perfect multicollinearity.  Defaults to True.
            check_constant_cols (bool): Whether to raise if any column of the
                design matrix is constant.  Defaults to True.
            cov_groups (array-like or None): Reserved for cluster-group
                variable (not yet wired).

        Returns:
            SparseQuantileRegressionModel: Model object ready to call
                ``.fit(tau, ...)``.
        """
        _t = time.time()

        if isinstance(data, dict):
            data = DataFrame(data, copy=False)

        data_result = SparseDataGetter.get_data(data, formula, index=index, debug=debug,
                                                drop_1_for_FE=drop_1_for_FE, check_constant_cols=check_constant_cols)
        exog, exog_names = data_result[EXOG_KEY].values, data_result[EXOG_KEY].column_names
        endog, endog_name = data_result[ENDOG_KEY].values, data_result[ENDOG_KEY].column_names[0]

        if data_result[INSTRUMENTS_KEY] is not None:
            instruments, instrument_names = data_result[INSTRUMENTS_KEY].values, \
                                            data_result[INSTRUMENTS_KEY].column_names
        else:
            instruments, instrument_names = None, None

        # if data_result[COV_GROUPS_KEY] is not None:
        #     cov_groups, cov_groups_name = data_result[COV_GROUPS_KEY], data_result[COV_GROUPS_NAME_KEY]
        # else:
        #     cov_groups, cov_groups_name = None, None
        cov_groups, cov_groups_name = None, None

        valid_obs_rows = data_result[VALID_OBS_ROWS_KEY]
        null_rows_info_dict = data_result[NULL_ROWS_INFO_DICT_KEY]
        has_intercept, has_implicit_constant = data_result[HAS_INTERCEPT_KEY], data_result[HAS_IMPLICIT_CONSTANT_KEY]
        formula_design_info = data_result[FORMULA_DESIGN_INFO_KEY]

        model = SparseQuantileRegressionModel(
            endog, exog, False, formula_design_info, has_intercept, has_implicit_constant, endog_name, exog_names,
            index=index, valid_obs_rows=valid_obs_rows,
            null_rows_info_dict=null_rows_info_dict, specification_name=specification_name,
            model_elapsed=time.time() - _t, instruments=instruments, instrument_names=instrument_names,
            cov_groups=cov_groups, cov_groups_name=cov_groups_name)

        if debug:
            print(model)

        if model.is_iv:
            warnings.warn("IV for quantile regression still experimental!!!!!!!!")

        return model

    def fit(self, tau, compute_cov=True, cov_type=DEFAULT_QR_COV_TYPE, cov_kwds=None, debug=False, start_params=None,
            specification_name=None, test_level=DEFAULT_QR_TEST_LEVEL, keep_model=True, use_t=DEFAULT_QR_USE_T,
            smoothing_k=DEFAULT_QR_SMOOTHING_K, min_resid_clip=DEFAULT_QR_MIN_RESID_CLIP, xtol=DEFAULT_QR_X_TOL,
            ftol=DEFAULT_QR_F_TOL, max_iter=DEFAULT_QR_MAX_ITER, line_search=DEFAULT_QR_LINE_SEARCH,
            loss=DEFAULT_QR_LOSS_FUNCTION, residual_inclusion=DEFAULT_QR_RESIDUAL_INCLUSION,
            residual_inclusion_order=DEFAULT_QR_RESIDUAL_INCLUSION_ORDER,
            dense_threshold_mb=DEFAULT_QR_DENSE_THRESHOLD_MB):
        """Fit the quantile regression model at one or more quantile levels.

        When ``tau`` is iterable (e.g. ``(0.1, 0.5, 0.9)``), the quantiles are
        sorted and fitted in order; each fit warm-starts from the previous
        solution, which can significantly speed up convergence.

        Covariance estimation is controlled by ``cov_type``:
        - ``'IID'`` — KDE-based homoscedastic sandwich (fast, assumes i.i.d. errors).
        - ``'ROBUST'`` — KDE-based heteroscedastic sandwich.
        - ``'BOOTSTRAP'`` — Bayesian / block bootstrap (most reliable, slowest).

        Args:
            tau (float or iterable of float): Quantile level(s) in (0, 1).
            compute_cov (bool): Whether to compute the covariance matrix.
                Defaults to True.
            cov_type (str): Covariance estimator; one of ``QR_COV_TYPES``.
                Defaults to ``DEFAULT_QR_COV_TYPE``.
            cov_kwds (dict or None): Keyword arguments for the chosen
                covariance estimator.  For bootstrap: ``n_samples``,
                ``method`` (``'bayesian'``/``'block'``), ``seed``,
                ``max_processes``, ``alpha``, ``use_correction``.
            debug (bool): If True, print iteration table and convergence
                message.  Defaults to False.
            start_params (array-like or None): Initial β vector of length p.
                Defaults to the OLS solution.
            specification_name (str or None): Label for the model
                specification (shown in the summary footer).
            test_level (float): Significance level for confidence intervals
                and p-values.  Defaults to ``DEFAULT_QR_TEST_LEVEL``.
            keep_model (bool): If True, store a reference to the model object
                inside the result.  Defaults to True.
            use_t (bool): Use the t-distribution for inference.  Defaults to
                ``DEFAULT_QR_USE_T``.
            smoothing_k (float): Surrogate-loss smoothing bandwidth.
                Defaults to ``DEFAULT_QR_SMOOTHING_K``.
            min_resid_clip (float): Floor on |residual| in the IRLS weight
                denominator.  Defaults to ``DEFAULT_QR_MIN_RESID_CLIP``.
            xtol (float): β-change convergence threshold.  Defaults to
                ``DEFAULT_QR_X_TOL``.
            ftol (float): Relative cost-change convergence threshold.
                Defaults to ``DEFAULT_QR_F_TOL``.
            max_iter (int): Maximum IRLS iterations.  Defaults to
                ``DEFAULT_QR_MAX_ITER``.
            line_search (bool): Run a grid line search at each step.
                Defaults to ``DEFAULT_QR_LINE_SEARCH``.
            loss (str or type): Surrogate loss function name or class.
                Defaults to ``DEFAULT_QR_LOSS_FUNCTION``.
            residual_inclusion (bool): Include IV residual powers (control
                function).  Defaults to ``DEFAULT_QR_RESIDUAL_INCLUSION``.
            residual_inclusion_order (int): Polynomial order for residual
                inclusion.  Defaults to
                ``DEFAULT_QR_RESIDUAL_INCLUSION_ORDER``.
            dense_threshold_mb (float or None): Memory threshold for
                converting sparse X to dense.  Defaults to
                ``DEFAULT_QR_DENSE_THRESHOLD_MB``.

        Returns:
            SparseQuantileRegressionResults or dict:
                - A single ``SparseQuantileRegressionResults`` when ``tau`` is
                  a scalar.
                - A ``dict`` mapping each tau value to its
                  ``SparseQuantileRegressionResults`` when ``tau`` is
                  iterable.
        """
        if hasattr(tau, '__iter__'):
            to_return = dict()
            for t in sorted(tau):
                fit = self.fit(
                    tau=t, compute_cov=compute_cov, cov_type=cov_type, cov_kwds=cov_kwds, debug=debug,
                    start_params=start_params, specification_name=specification_name, test_level=test_level,
                    keep_model=keep_model, use_t=use_t, smoothing_k=smoothing_k, min_resid_clip=min_resid_clip,
                    xtol=xtol, ftol=ftol, max_iter=max_iter, line_search=line_search, loss=loss,
                    residual_inclusion=residual_inclusion, residual_inclusion_order=residual_inclusion_order,
                    dense_threshold_mb=dense_threshold_mb)
                # Warm-start next quantile from the previous solution.
                start_params = fit._params
                to_return[t] = fit
            return to_return

        _t = time.time()

        if cov_type is None:
            cov_type = DEFAULT_QR_COV_TYPE
        cov_type = cov_type.upper()
        if cov_type not in QR_COV_TYPES:
            if QR_COV_TYPE_BOOTSTRAP not in cov_type:
                raise Exception(f"`cov_type` must be one of {str(QR_COV_TYPES)}, you gave '{cov_type}'!")
        if cov_kwds is None:
            cov_kwds = dict()
        if QR_COV_TYPE_BOOTSTRAP in cov_type:
            check_cov_kwds(cov_type, cov_kwds)

        # # TODO - move IV into qr_internal
        # if self.is_iv:
        #     iv_result = iv_first_stage(self.exog, self.instruments, exog_col_names=self.exog_names,
        #                                instrument_col_names=self.instrument_names,
        #                                param_return_type_dataframe=False, debug=debug, _time=_t,
        #                                residual_inclusion=residual_inclusion,
        #                                residual_inclusion_order=residual_inclusion_order)
        #     exog = iv_result[EXOG_INSTRUMENTED]
        #     exog_names = iv_result[EXOG_COL_NAMES]
        # else:
        #     exog = self.exog
        #     exog_names = self.exog_names

        results = qr_internal(
            self.endog, self.exog, tau, debug=debug, start_params=start_params, line_search=line_search,
            loss=loss, max_iter=max_iter, smoothing_k=smoothing_k, min_resid_clip=min_resid_clip,
            xtol=xtol, ftol=ftol, Z=self.instruments, is_endog_regressor=self.is_endog_regressor,
            residual_inclusion=residual_inclusion, residual_inclusion_order=residual_inclusion_order,
            dense_threshold_mb=dense_threshold_mb)

        exog_names = convert_exog_col_map_to_col_names2(results['exog_col_map'], self.exog_names)
        exog_instrumented = results['exog_instrumented']

        df_model = self.exog.shape[1] - (self.has_implicit_constant or self.has_intercept)
        df_resid = self.exog.shape[0] - self.exog.shape[1]
        df_t_dist = df_resid

        fit_time = time.time() - _t
        fit = SparseQuantileRegressionResults(
            self, tau, exog_names, results['params'], None, cov_type, cov_kwds, results['resid'],
            results['fittedvalues'], results['weights'], results['pseudo_rsquared'], results['cost'],
            results['true_cost'], self.nobs, df_resid, df_model, df_t_dist,
            results['converged'], results['iterations'], results['error'],
            line_search, message=results['message'],
            test_level=test_level, use_t=use_t, residual_inclusion=residual_inclusion,
            fit_elapsed=fit_time,
            cov_elapsed=0,  # TODO
            specification_name=specification_name, keep_model=keep_model,
            compute_cov=compute_cov,
            method='IRLS',
        )

        _t = time.time()
        if compute_cov and results['converged']:

            if QR_COV_TYPE_BOOTSTRAP not in cov_type:

                print(cov_type)
                try:
                    var_covar = get_var_covar(
                        cov_type, exog_instrumented, results['resid'], tau, small_sample_correct=self.nobs / df_resid)

                    if var_covar is not None:
                        fit.set_cov_params(var_covar, cov_type=cov_type, debug=debug, test_level=test_level)
                    else:
                        fit.compute_cov = False
                        fit._cov_params = None
                        if debug:
                            print("Computing Variance-Covariance failed!!")

                except Exception as e:
                    fit.compute_cov = False
                    fit._cov_params = None
                    warnings.warn("Computing qr variance-covariance failed")

            else:
                # Bootstrap path: re-fit the model with each set of resampling weights
                # and collect the resulting coefficient vectors for variance estimation.
                def param_estimation_func2(bootstrap_weights):
                    """Re-fit the quantile model with bootstrap weights and return params if converged.

                    Args:
                        bootstrap_weights (ndarray): Non-negative observation weights drawn
                            by the bootstrap sampler, shape (n,).

                    Returns:
                        ndarray or None: Coefficient vector of shape (p,) if the IRLS loop
                            converged for this bootstrap draw, otherwise None (draw is
                            discarded by ``do_bootstrap2``).
                    """
                    result_temp = qr_internal(
                        self.endog, self.exog, tau, debug=False, start_params=results['params'],
                        line_search=line_search, loss=loss, max_iter=max_iter, smoothing_k=smoothing_k,
                        min_resid_clip=min_resid_clip, xtol=xtol, ftol=ftol, weights=bootstrap_weights,
                        Z=self.instruments, is_endog_regressor=self.is_endog_regressor,
                        residual_inclusion=residual_inclusion, residual_inclusion_order=residual_inclusion_order,
                        dense_threshold_mb=dense_threshold_mb)
                    if result_temp['converged']:
                        return result_temp['params']
                    else:
                        return None

                do_bootstrap2(self.nobs, fit, param_estimation_func2, groups=cov_kwds.get('groups', None),
                              n_samples=cov_kwds.get('n_samples', DEFAULT_QR_BOOTSTRAP_N_SAMPLES),
                              method=cov_kwds.get('method', DEFAULT_BB_METHOD),
                              alpha=cov_kwds.get('alpha', DEFAULT_BB_ALPHA),
                              seed=cov_kwds.get('seed', 0), debug=debug,
                              max_processes=cov_kwds.get('max_processes', DEFAULT_BB_MAX_PROCESSES),
                              use_correction=cov_kwds.get('use_correction', True), test_level=test_level)

        else:
            cov_type = 'NOT_COMPUTED'
            fit.cov_type = cov_type
            fit.compute_cov = False

        fit.cov_elapsed = time.time() - _t

        return fit

    @staticmethod
    def qr(formula, data, tau, index=None, cov_type=DEFAULT_QR_COV_TYPE, cov_kwds=None, debug=False,
           start_params=None, compute_cov=True, specification_name=None, test_level=DEFAULT_QR_TEST_LEVEL,
           keep_model=True, use_t=DEFAULT_QR_USE_T, line_search=DEFAULT_QR_LINE_SEARCH, ftol=DEFAULT_QR_F_TOL,
           smoothing_k=DEFAULT_QR_SMOOTHING_K, min_resid_clip=DEFAULT_QR_MIN_RESID_CLIP, xtol=DEFAULT_QR_X_TOL,
           max_iter=DEFAULT_QR_MAX_ITER, loss=DEFAULT_QR_LOSS_FUNCTION,
           residual_inclusion=DEFAULT_QR_RESIDUAL_INCLUSION, dense_threshold_mb=DEFAULT_QR_DENSE_THRESHOLD_MB,
           residual_inclusion_order=DEFAULT_QR_RESIDUAL_INCLUSION_ORDER):
        """Fit quantile regression from a patsy formula (formula API entry point).

        Parses the formula, builds the model, and calls ``fit``.  This is the
        method exposed via ``kanly.api.qr``.

        Args:
            formula (str): Patsy formula.  Supports ``$`` for weights and
                ``|`` for excluded instruments.
            data (DataFrame or dict): Data containing all variables in the
                formula.
            tau (float or iterable of float): Quantile level(s) in (0, 1).
            index (array-like or None): Row index for result alignment.
            cov_type (str): Covariance estimator.  Defaults to
                ``DEFAULT_QR_COV_TYPE``.
            cov_kwds (dict or None): Keyword arguments for the covariance
                estimator.
            debug (bool): Print iteration table.  Defaults to False.
            start_params (array-like or None): Initial β vector.
            compute_cov (bool): Whether to compute covariance.  Defaults to
                True.
            specification_name (str or None): Label for the summary footer.
            test_level (float): Significance level.  Defaults to
                ``DEFAULT_QR_TEST_LEVEL``.
            keep_model (bool): Store model reference in results.  Defaults to
                True.
            use_t (bool): Use t-distribution for inference.  Defaults to
                ``DEFAULT_QR_USE_T``.
            line_search (bool): Run grid line search.  Defaults to
                ``DEFAULT_QR_LINE_SEARCH``.
            ftol (float): Relative cost-change convergence threshold.
            smoothing_k (float): Surrogate-loss smoothing bandwidth.
            min_resid_clip (float): Floor on |residual| in IRLS weight denominator.
            xtol (float): β-change convergence threshold.
            max_iter (int): Maximum IRLS iterations.
            loss (str or type): Surrogate loss.  Defaults to
                ``DEFAULT_QR_LOSS_FUNCTION``.
            residual_inclusion (bool): Include IV residual powers.  Defaults
                to ``DEFAULT_QR_RESIDUAL_INCLUSION``.
            dense_threshold_mb (float or None): Dense-conversion threshold.
            residual_inclusion_order (int): Polynomial order for residual
                inclusion.

        Returns:
            SparseQuantileRegressionResults or dict: Same as ``fit``.

        Examples
        --------
        Median (``tau=0.5``) regression with bootstrap covariance:

        >>> import numpy as np, pandas as pd
        >>> from kanly.api import qr
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({'x': rng.normal(size=500)})
        >>> df['y'] = 1.0 + 2.0 * df['x'] + 0.5 * rng.normal(size=500)
        >>> fit = qr('y ~ x', df, tau=0.5,                  # doctest: +SKIP
        ...          cov_type='bootstrap',
        ...          cov_kwds={'n_samples': 500, 'method': 'bayesian'})
        >>> fit.params.round(2)                              # doctest: +SKIP
        Intercept    1.01
        x            1.98
        dtype: float64

        Several quantiles at once (returns a dict keyed by ``tau``):

        >>> fits = qr('y ~ x', df, tau=[0.1, 0.5, 0.9])     # doctest: +SKIP
        >>> {t: round(float(fits[t].params['x']), 2)         # doctest: +SKIP
        ...  for t in fits}
        {0.1: 1.97, 0.5: 1.98, 0.9: 1.99}

        Weighted quantile regression via the ``$`` weight syntax:

        >>> df['w'] = np.abs(rng.normal(size=500)) + 0.1
        >>> fit_w = qr('y ~ x $ w', df, tau=0.5)            # doctest: +SKIP

        Alias on ``kanly.api``: ``quantreg``.
        """

        # # TODO
        # groups = cov_kwds.get('groups', None)
        # if not isinstance(groups, str):
        #    groups = None

        groups = None
        model = SparseQuantileRegressionModel.build_model_from_formula(
            formula, data, index=index, debug=debug, specification_name=specification_name,
            cov_groups=groups)

        # TODO
        # if 'groups' in cov_kwds.keys() and groups is not None:
        #     cov_kwds['groups'] = model.cov_groups

        return model.fit(tau, debug=debug, start_params=start_params, line_search=line_search,
                         smoothing_k=smoothing_k, min_resid_clip=min_resid_clip, xtol=xtol, ftol=ftol, max_iter=max_iter,
                         cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t, keep_model=keep_model,
                         test_level=test_level, compute_cov=compute_cov, loss=loss, specification_name=specification_name,
                         residual_inclusion=residual_inclusion, residual_inclusion_order=residual_inclusion_order,
                         dense_threshold_mb=dense_threshold_mb)

    @staticmethod
    def QR(endog, exog, tau, add_constant=False, has_constant=True, endog_name=None, exog_names=None, cov_type=DEFAULT_QR_COV_TYPE, cov_kwds=None,
           debug=False, start_params=None, compute_cov=True, specification_name=None, test_level=DEFAULT_QR_TEST_LEVEL,
           keep_model=True, use_t=DEFAULT_QR_USE_T, line_search=DEFAULT_QR_LINE_SEARCH, ftol=DEFAULT_QR_F_TOL,
           smoothing_k=DEFAULT_QR_SMOOTHING_K, min_resid_clip=DEFAULT_QR_MIN_RESID_CLIP, xtol=DEFAULT_QR_X_TOL,
           max_iter=DEFAULT_QR_MAX_ITER, loss=DEFAULT_QR_LOSS_FUNCTION, dense_threshold_mb=DEFAULT_QR_DENSE_THRESHOLD_MB,
           residual_inclusion=DEFAULT_QR_RESIDUAL_INCLUSION, residual_inclusion_order=DEFAULT_QR_RESIDUAL_INCLUSION_ORDER,
           instruments=None, instrument_names=None):
        """Fit quantile regression from pre-built arrays (matrix API entry point).

        Constructs a ``SparseQuantileRegressionModel`` directly from arrays
        (bypassing patsy) and calls ``fit``.

        Args:
            endog (ndarray or sparse): Response vector, shape (n,) or (n, 1).
            exog (ndarray or sparse): Design matrix, shape (n, p).  Must
                already include an intercept column if desired.
            tau (float or iterable of float): Quantile level(s) in (0, 1).
            add_constant (bool): If True, prepend a column of ones to
                ``exog``.  Defaults to False.
            has_constant (bool): Whether ``exog`` already contains a constant
                column.  Defaults to True.
            endog_name (str or None): Name of the response variable for
                display purposes.
            exog_names (list of str or None): Column names for ``exog``.
            cov_type (str): Covariance estimator.  Defaults to
                ``DEFAULT_QR_COV_TYPE``.
            cov_kwds (dict or None): Keyword arguments for the covariance
                estimator.
            debug (bool): Print iteration table.  Defaults to False.
            start_params (array-like or None): Initial β vector.
            compute_cov (bool): Whether to compute covariance.  Defaults to
                True.
            specification_name (str or None): Label for the summary footer.
            test_level (float): Significance level.  Defaults to
                ``DEFAULT_QR_TEST_LEVEL``.
            keep_model (bool): Store model reference in results.  Defaults to
                True.
            use_t (bool): Use t-distribution.  Defaults to
                ``DEFAULT_QR_USE_T``.
            line_search (bool): Run grid line search.  Defaults to
                ``DEFAULT_QR_LINE_SEARCH``.
            ftol (float): Relative cost-change convergence threshold.
            smoothing_k (float): Surrogate-loss smoothing bandwidth.
            min_resid_clip (float): Floor on |residual| in IRLS weight denominator.
            xtol (float): β-change convergence threshold.
            max_iter (int): Maximum IRLS iterations.
            loss (str or type): Surrogate loss.
            dense_threshold_mb (float or None): Dense-conversion threshold.
            residual_inclusion (bool): Include IV residual powers.
            residual_inclusion_order (int): Polynomial order for residual
                inclusion.
            instruments (ndarray or sparse or None): Instrument matrix for
                IV, shape (n, q).
            instrument_names (list of str or None): Column names for
                ``instruments``.

        Returns:
            SparseQuantileRegressionResults or dict: Same as ``fit``.

        Examples
        --------
        Median regression on pre-built arrays:

        >>> import numpy as np
        >>> from kanly.api import QR
        >>> rng = np.random.default_rng(0)
        >>> n = 500
        >>> X = np.column_stack([np.ones(n), rng.normal(size=n)])
        >>> y = 1.0 + 2.0 * X[:, 1] + 0.5 * rng.normal(size=n)
        >>> fit = QR(y, X, tau=0.5,                        # doctest: +SKIP
        ...           exog_names=['Intercept', 'x'])

        Several quantiles in one call:

        >>> fits = QR(y, X, tau=[0.1, 0.5, 0.9],          # doctest: +SKIP
        ...            exog_names=['Intercept', 'x'])

        Alias on ``kanly.api``: ``QUANTREG``.
        """

        model = SparseQuantileRegressionModel(
            endog, exog, add_constant, None, has_constant, False, endog_name, exog_names, None,
            valid_obs_rows=range(exog.shape[0]),
            null_rows_info_dict=dict(), specification_name=specification_name, model_elapsed=0,
            instruments=instruments, instrument_names=instrument_names)

        return model.fit(tau, debug=debug, start_params=start_params, line_search=line_search,
                         smoothing_k=smoothing_k, min_resid_clip=min_resid_clip, xtol=xtol, ftol=ftol,
                         max_iter=max_iter, residual_inclusion=residual_inclusion,
                         residual_inclusion_order=residual_inclusion_order,
                         cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t, keep_model=keep_model,
                         test_level=test_level, compute_cov=compute_cov, loss=loss,
                         specification_name=specification_name, dense_threshold_mb=dense_threshold_mb)

    def predict(self, params, data=None, index=None, debug=False, ignore_column_mismatch=False, *args, **kwargs):
        """Compute the linear predictor Xβ for a given coefficient vector.

        Args:
            params (ndarray): Coefficient vector of length p.
            data (DataFrame or None): Out-of-sample data.  If None, the
                in-sample design matrix is used.
            index (array-like or None): Row index for the out-of-sample data.
            debug (bool): If True, print diagnostic information.  Defaults
                to False.
            ignore_column_mismatch (bool): When ``True``, allow prediction when
                the new design has fewer columns than ``params`` (e.g. missing
                fixed-effect levels). See
                :meth:`~kanly.regression.linear_model_base.LinearModelBase.get_linear_predictor`.

        Returns:
            ndarray: Predicted values Xβ, shape (n,).
        """
        return self.get_linear_predictor(params, data=data, index=index, debug=debug,
                                         ignore_column_mismatch=ignore_column_mismatch)

    def accepts_multi_outcome(self):
        """Return False — quantile regression supports only a single response variable.

        Returns:
            bool: Always False.
        """
        return False
