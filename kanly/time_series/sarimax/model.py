from __future__ import absolute_import, print_function

import time

import numpy as np
from typing_extensions import Self

from kanly.bayes.bayesian_model import BayesianModel
from kanly.formula.data_getter import (SparseDataGetter, ENDOG_KEY, EXOG_KEY, WEIGHTS_KEY, INSTRUMENTS_KEY, ABSORB_KEY,
                                       HAS_INTERCEPT_KEY, VALID_OBS_ROWS_KEY)
from kanly.formula.formula_design_info import FormulaDesignInfo
from kanly.formula.keys import FORMULA_DESIGN_INFO_KEY
from kanly.regression.model_base import ModelBase
from kanly.time_series.sarimax.constants import (
    DEFAULT_SARIMAX_COV_TYPE, DEFAULT_SARIMAX_DO_HANNAN_RISSANEN, DEFAULT_SARIMAX_MAXITER,
    DEFAULT_SARIMAX_GTOL, DEFAULT_SARIMAX_FTOL, DEFAULT_SARIMAX_XTOL, DEFAULT_SARIMAX_STEADY_STATE_TOL,
    DEFAULT_SARIMAX_BFGS_B0,
    DEFAULT_SARIMAX_ENFORCE_STATIONARITY, DEFAULT_SARIMAX_ENFORCE_INVERTIBILITY,
    DEFAULT_SARIMAX_INITIAL_DIFFUSE_VARIANCE,
    DEFAULT_SARIMAX_CONCENTRATE_SCALE, DEFAULT_SARIMAX_SIMPLE_DIFFERENCING, DEFAULT_SARIMAX_MULTIPLICATIVE,
    DEFAULT_SARIMAX_DIFFUSE, DEFAULT_SARIMAX_STANDARDIZE_ENDOG)
from kanly.time_series.sarimax.results import SarimaxResults
from kanly.time_series.sarimax.sarimax_internal import get_loglike_function, sarimax_internal
from kanly.time_series.sarimax.sarimax_internal_helper_functions import format_trend_scale
from kanly.time_series.sarimax.validate_sarima_orders import validate_orders
from kanly.time_series.sarimax.var_covar import get_cov_params
from kanly.utils.util import dict_2_dataframe


def get_trend_names(trend, trend_scale, trend_offset):
    """
    SARIMAX helper function ``get_trend_names``.

    Args:
        trend: Trend specification, such as None, ``c``, ``t``, ``ct``, ``n``, or an indicator list.
        trend_scale: Scale used to normalize deterministic trend time indices.
        trend_offset: Starting offset for deterministic trend time indices.

    Returns:
        List of trend parameter names.
    """
    if trend_scale is None or not trend_scale or trend_scale == 1:
        varname = 't'
    else:
        varname = f'({f"(t+{trend_offset-1})" if trend_offset != 1 else "t"}/{trend_scale})'
    return [] if trend is None else [('const' if j == 0 else f'{varname}**{j}') for j, t in enumerate(trend) if t > 0]


def get_arma_param_names(ar_lags, ma_lags, sar_lags, sma_lags,
                         s, k_exog=0, trend=None, exog_names=None,
                         trend_scale=1, trend_offset=1):
    """
    SARIMAX helper function ``get_arma_param_names``.

    Args:
        ar_lags: Active nonseasonal autoregressive lags.
        ma_lags: Active nonseasonal moving-average lags.
        sar_lags: Active seasonal autoregressive lags.
        sma_lags: Active seasonal moving-average lags.
        s: Seasonal period length.
        k_exog: Number of exogenous-regressor parameters.
        trend: Trend specification, such as None, ``c``, ``t``, ``ct``, ``n``, or an indicator list.
        exog_names: Names of exogenous regressors.
        trend_scale: Scale used to normalize deterministic trend time indices.
        trend_offset: Starting offset for deterministic trend time indices.

    Returns:
        List of parameter names in packed SARIMAX order.
    """
    return (
            get_trend_names(trend, trend_scale, trend_offset)
            + (([] if k_exog is None else [f'x{j}' for j in range(k_exog)])
               if exog_names is None else
               exog_names
               )
            + [f'ar.L{j}' for j in ar_lags]
            + [f'ma.L{j}' for j in ma_lags]
            + [f'ar.S.L{s * j}' for j in sar_lags]
            + [f'ma.S.L{s * j}' for j in sma_lags]
            + ['sigma2']
    )


def parse_trend(trend, d):
    """
    SARIMAX helper function ``parse_trend``.

    Args:
        trend: Trend specification, such as None, ``c``, ``t``, ``ct``, ``n``, or an indicator list.
        d: Nonseasonal differencing order.

    Returns:
        Normalized trend indicator list.
    """
    if trend is None:
        trend = 'c' if d == 0 else []

    else:
        if d == 1:
            assert trend not in ['c', 'ct']
        elif d > 1:
            assert trend not in ['c', 't', 'ct']

    if trend == 'c':
        trend = [1]
    elif trend == 'ct':
        trend = [1, 1]
    elif trend == 't':
        trend = [0, 1]
    elif trend == 'n':
        trend = []

    if sum(trend) == 0:
        trend = []

    assert sum(trend[:d]) == 0
    assert set(trend) <= {0, 1}

    return trend


class SarimaxModel(ModelBase):
    """[S]easonal [A]uto[R]egressive [I]ntegrated [M]oving [A]verage with e[X]ogenous regressors.

    This class is the user-facing wrapper around the tracked SARIMAX MLE
    implementation. It parses orders and trend sparse_terms, builds packed parameter
    names, exposes likelihood methods, and delegates fitting to
    ``sarimax_internal``.

    Examples
    --------
    Fit a SARIMAX(2,0,0) on simulated data using the array-form entry point:

    >>> import numpy as np
    >>> from kanly.api import SarimaxModel, simulate_sarima
    >>> y = simulate_sarima(n=500, ar=[0.5, 0.1], seed=0, burnin=1000)
    >>> fit = SarimaxModel.SARIMAX(y, order=(2, 0, 0))
    >>> print(fit.summary())                          # doctest: +SKIP
    SARIMAX Results
    ...

    See Also
    --------
    `SARIMAX` : array entry point ``(endog, exog, order, seasonal_order, ...)``.
    `sarimax` (alias `arima`) : formula entry point.
    `bsarimax` : Bayesian SARIMAX wrapper.
    """

    def __init__(self, endog, exog, order, seasonal_order, trend, trend_offset, trend_scale,
                 formula_design_info: FormulaDesignInfo,
                 endog_name=None, exog_names=None, exog_term_names=None,
                 data=None, specification_name=None,
                 enforce_stationarity=DEFAULT_SARIMAX_ENFORCE_STATIONARITY,
                 enforce_invertibility=DEFAULT_SARIMAX_ENFORCE_INVERTIBILITY,
                 simple_differencing=DEFAULT_SARIMAX_SIMPLE_DIFFERENCING,
                 initial_diffuse_variance=DEFAULT_SARIMAX_INITIAL_DIFFUSE_VARIANCE,
                 multiplicative=DEFAULT_SARIMAX_MULTIPLICATIVE,
                 diffuse=DEFAULT_SARIMAX_DIFFUSE,
                 index=None, model_elapsed=0, debug=False):

        """
        Initialize the object and store parsed SARIMAX state.

        Args:
            endog: Observed endogenous time-series values.
            exog: Optional exogenous regressor matrix aligned to ``endog``.
            order: Nonseasonal ARIMA order specification ``(p, d, q)``.
            seasonal_order: Optional seasonal order specification ``(P, D, Q, s)``.
            trend: Trend specification, such as None, ``c``, ``t``, ``ct``, ``n``, or an indicator list.
            trend_offset: Starting offset for deterministic trend time indices.
            trend_scale: Scale used to normalize deterministic trend time indices.
            formula: Formula string describing endogenous and exogenous variables.
            from_formula: Whether the model was built from a formula interface.
            endog_name: Name of the endogenous variable.
            exog_names: Names of exogenous regressors.
            exog_term_names: Formula term names associated with exogenous regressors.
            data: Data source used by formula parsing.
            specification_name: Optional display name for the model specification.
            enforce_stationarity: Whether to transform AR parameters into the stationary region.
            enforce_invertibility: Whether to transform MA parameters into the invertible region.
            simple_differencing: Whether to difference data before the state-space likelihood instead of integrating in the state.
            initial_diffuse_variance: Initial variance assigned to diffuse state components.
            multiplicative: Whether to request multiplicative seasonal dynamics.
            diffuse: Whether to use diffuse initialization for integrated state components.
            index: Optional observation index.
            model_elapsed: Time spent constructing the model.
            debug: Whether to print fitting diagnostics.

        Returns:
            Function result for the SARIMAX estimation workflow.
        """
        super().__init__(
            nobs=len(endog), endog=endog, index=index, valid_obs_rows=None,
            specification_name=specification_name,
            formula_design_info=formula_design_info, model_elapsed=model_elapsed,
            is_sure=False, parent_model=None)

        ((self.order, self.seasonal_order),
         self.is_seasonal,
         (self.has_ar_terms, self.has_ma_terms),
         (((self.p, self.ar_lags, self.k_ar), self.d, (self.q, self.ma_lags, self.k_ma)),
          ((self.P, self.sar_lags, self.k_sar), self.D, (self.Q, self.sma_lags, self.k_sma), self.s),
          )) = validate_orders(order, seasonal_order)

        self.trend = parse_trend(trend, self.d + self.D)
        self.trend_offset = trend_offset

        self.trend_scale = format_trend_scale(trend_scale, self.nobs)
        self.trend_names = get_trend_names(self.trend, self.trend_scale, self.trend_offset)

        if np.ndim(exog) == 1:
            exog = exog.reshape((-1,1))
        self.k_exog = exog.shape[1] if exog is not None else 0
        self.k_trend = np.count_nonzero(self.trend)

        assert set(self.trend) <= {0, 1}

        if endog_name is None:
            endog_name = 'y'
        if exog_names is None:
            exog_names = [f'x{j + 1}' for j in range(self.k_exog)]

        self.endog_name, self.exog_names = endog_name, exog_names
        self.exog_term_names = exog_term_names

        self.param_names = get_arma_param_names(
            self.ar_lags, self.ma_lags, self.sar_lags, self.sma_lags, self.s,
            exog_names=self.exog_names, trend=self.trend, trend_scale=self.trend_scale,
            trend_offset=self.trend_offset
        )

        if debug:
            print(f'Param Names are {self.param_names}')

        self.exog = exog

        if self.k_exog:
            assert len(exog) == self.nobs

        self.df_model = self.k_exog + self.k_trend + self.k_ar + self.k_ma + self.k_sar + self.k_sma + 1
        self.df_resid = self.nobs - self.df_model
        self.num_params = self.df_model

        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.simple_differencing = simple_differencing
        self.initial_diffuse_variance = initial_diffuse_variance
        self.diffuse = diffuse

        self.loglike = get_loglike_function(
            self.endog, self.exog, self.trend, self.trend_offset, self.trend_scale, self.d, self.D, self.s,
            self.k_trend, self.k_exog,
            self.k_ar, self.k_ma, self.k_sar, self.k_sma,
            self.ar_lags, self.ma_lags, self.sar_lags, self.sma_lags,
            self.simple_differencing, self.initial_diffuse_variance,
            self.diffuse
        )

        self.multiplicative = multiplicative

    def loglike(self, params, return_type='llf', steady_state_tol=DEFAULT_SARIMAX_STEADY_STATE_TOL):
        """
        Evaluate the model log-likelihood for packed parameters.

        Args:
            params: Packed parameter vector in model parameter order.
            return_type: Amount of likelihood output to return, such as ``llf`` or ``all``.
            steady_state_tol: Tolerance for detecting steady-state Kalman forecast error variance.

        Returns:
            Function result for the SARIMAX estimation workflow.
        """
        return self.loglike(params, return_type, steady_state_tol)

    def loglike_obs(self, params, steady_state_tol=DEFAULT_SARIMAX_STEADY_STATE_TOL):
        """
        Evaluate per-observation model log-likelihood values.

        Args:
            params: Packed parameter vector in model parameter order.
            steady_state_tol: Tolerance for detecting steady-state Kalman forecast error variance.

        Returns:
            Function result for the SARIMAX estimation workflow.
        """
        vals = self.loglike(params, return_type='full', steady_state_tol=steady_state_tol)
        return vals['llf_obs']

    @staticmethod
    def SARIMAX(endog, exog=None, order=(0, 0, 0), seasonal_order=None, trend=None, trend_offset=1,
                trend_scale=None, debug=False,
                specification_name=None, drop_1_for_FE=True, exog_names=None, endog_name=None,
                start_params=None, nlags=None,
                maxiter=DEFAULT_SARIMAX_MAXITER, gtol=DEFAULT_SARIMAX_GTOL, ftol=DEFAULT_SARIMAX_FTOL,
                xtol=DEFAULT_SARIMAX_XTOL,
                cov_type=DEFAULT_SARIMAX_COV_TYPE,
                do_hannan_rissanen=DEFAULT_SARIMAX_DO_HANNAN_RISSANEN, B0=DEFAULT_SARIMAX_BFGS_B0, do_numba=None,
                enforce_stationarity=DEFAULT_SARIMAX_ENFORCE_STATIONARITY,
                enforce_invertibility=DEFAULT_SARIMAX_ENFORCE_INVERTIBILITY,
                concentrate_scale=DEFAULT_SARIMAX_CONCENTRATE_SCALE,
                simple_differencing=DEFAULT_SARIMAX_SIMPLE_DIFFERENCING,
                initial_diffuse_variance=DEFAULT_SARIMAX_INITIAL_DIFFUSE_VARIANCE,
                multiplicative=DEFAULT_SARIMAX_MULTIPLICATIVE,
                diffuse=DEFAULT_SARIMAX_DIFFUSE,
                standardize_endog=DEFAULT_SARIMAX_STANDARDIZE_ENDOG,
                **kwargs) -> SarimaxResults:

        """
        SARIMAX helper function ``SARIMAX``.

        Args:
            endog: Observed endogenous time-series values.
            exog: Optional exogenous regressor matrix aligned to ``endog``.
            order: Nonseasonal ARIMA order specification ``(p, d, q)``.
            seasonal_order: Optional seasonal order specification ``(P, D, Q, s)``.
            trend: Trend specification, such as None, ``c``, ``t``, ``ct``, ``n``, or an indicator list.
            trend_offset: Starting offset for deterministic trend time indices.
            trend_scale: Scale used to normalize deterministic trend time indices.
            debug: Whether to print fitting diagnostics.
            specification_name: Optional display name for the model specification.
            drop_1_for_FE: Whether formula parsing drops singleton fixed-effect indicators.
            exog_names: Names of exogenous regressors.
            endog_name: Name of the endogenous variable.
            start_params: Optional initial parameter vector.
            nlags: Number of lags to compute or use.
            maxiter: Maximum optimizer iterations.
            gtol: Optimizer projected-gradient tolerance.
            ftol: Optimizer objective_function-change tolerance.
            xtol: Optimizer parameter-change tolerance.
            cov_type: Covariance estimator name.
            do_hannan_rissanen: Whether to estimate starting ARMA parameters with Hannan-Rissanen.
            B0: Initial Hessian scale or approximation passed to BFGS/PQN.
            do_numba: Compatibility flag for numba execution paths.
            enforce_stationarity: Whether to transform AR parameters into the stationary region.
            enforce_invertibility: Whether to transform MA parameters into the invertible region.
            concentrate_scale: Whether to optimize with innovation variance concentrated out.
            simple_differencing: Whether to difference data before the state-space likelihood instead of integrating in the state.
            initial_diffuse_variance: Initial variance assigned to diffuse state components.
            multiplicative: Whether to request multiplicative seasonal dynamics.
            diffuse: Whether to use diffuse initialization for integrated state components.
            standardize_endog: Whether to standardize endogenous data during optimization.

        Returns:
            Fitted ``SarimaxResults`` object.

        Examples
        --------
        Array-form SARIMAX fit on simulated ARMA(2,1) data:

        >>> import numpy as np
        >>> from kanly.api import SARIMAX, simulate_sarima
        >>> y = simulate_sarima(n=500, ar=[0.6, 0.2], ma=[0.4],
        ...                     seed=0, burnin=1000)
        >>> fit = SARIMAX(y, order=(2, 0, 1))
        >>> print(fit.summary())                          # doctest: +SKIP

        With an exogenous regressor:

        >>> rng = np.random.default_rng(0)
        >>> x = rng.normal(size=500)
        >>> y2 = y + 1.5 * x
        >>> fit2 = SARIMAX(y2, exog=x.reshape(-1, 1), order=(2, 0, 0))   # doctest: +SKIP

        See Also
        --------
        `sarimax` (alias `arima`) : formula entry point.
        """
        endog = np.asarray(endog)
        if exog is not None:
            exog = np.asarray(exog)
            if np.ndim(exog) == 1:
                exog = exog.reshape((-1, 1))

        model = SarimaxModel(
            endog, exog, order, seasonal_order, trend, trend_offset, trend_scale, None,
            endog_name=endog_name, exog_names=exog_names, exog_term_names=None,
            data=None, specification_name=specification_name,
            index=None, model_elapsed=0, debug=debug,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
            simple_differencing=simple_differencing,
            initial_diffuse_variance=initial_diffuse_variance,
            multiplicative=multiplicative, diffuse=diffuse,
        )

        fit = model.fit(start_params=start_params, do_hannan_rissanen=do_hannan_rissanen,
                        nlags=nlags, debug=debug, cov_type=cov_type,
                        maxiter=maxiter, gtol=gtol, ftol=ftol, xtol=xtol, B0=B0, do_numba=do_numba,
                        concentrate_scale=concentrate_scale,
                        initial_diffuse_variance=initial_diffuse_variance,
                        standardize_endog=standardize_endog,
                        **kwargs)

        return fit

    @staticmethod
    def sarimax(formula, data, order=(0, 0, 0), seasonal_order=None,
                trend=None, trend_offset=1, trend_scale=None,
                index=None,
                debug=False,
                check_constant_cols=True, specification_name=None, drop_1_for_FE=True,
                start_params=None, nlags=None,
                maxiter=DEFAULT_SARIMAX_MAXITER, gtol=DEFAULT_SARIMAX_GTOL, ftol=DEFAULT_SARIMAX_FTOL,
                xtol=DEFAULT_SARIMAX_XTOL,
                cov_type=DEFAULT_SARIMAX_COV_TYPE, do_hannan_rissanen=DEFAULT_SARIMAX_DO_HANNAN_RISSANEN,
                B0=DEFAULT_SARIMAX_BFGS_B0,
                do_numba=None,
                concentrate_scale=DEFAULT_SARIMAX_CONCENTRATE_SCALE,
                enforce_stationarity=DEFAULT_SARIMAX_ENFORCE_STATIONARITY,
                enforce_invertibility=DEFAULT_SARIMAX_ENFORCE_INVERTIBILITY,
                simple_differencing=DEFAULT_SARIMAX_SIMPLE_DIFFERENCING,
                initial_diffuse_variance=DEFAULT_SARIMAX_INITIAL_DIFFUSE_VARIANCE,
                multiplicative=DEFAULT_SARIMAX_MULTIPLICATIVE,
                diffuse=DEFAULT_SARIMAX_DIFFUSE,
                standardize_endog=DEFAULT_SARIMAX_STANDARDIZE_ENDOG,
                **kwargs) -> SarimaxResults:

        """
        SARIMAX helper function ``sarimax``.

        Args:
            formula: Formula string describing endogenous and exogenous variables.
            data: Data source used by formula parsing.
            order: Nonseasonal ARIMA order specification ``(p, d, q)``.
            seasonal_order: Optional seasonal order specification ``(P, D, Q, s)``.
            trend: Trend specification, such as None, ``c``, ``t``, ``ct``, ``n``, or an indicator list.
            trend_offset: Starting offset for deterministic trend time indices.
            trend_scale: Scale used to normalize deterministic trend time indices.
            index: Optional observation index.
            debug: Whether to print fitting diagnostics.
            check_constant_cols: Whether to check for constant columns in formula-built exog.
            specification_name: Optional display name for the model specification.
            drop_1_for_FE: Whether formula parsing drops singleton fixed-effect indicators.
            start_params: Optional initial parameter vector.
            nlags: Number of lags to compute or use.
            maxiter: Maximum optimizer iterations.
            gtol: Optimizer projected-gradient tolerance.
            ftol: Optimizer objective_function-change tolerance.
            xtol: Optimizer parameter-change tolerance.
            cov_type: Covariance estimator name.
            do_hannan_rissanen: Whether to estimate starting ARMA parameters with Hannan-Rissanen.
            B0: Initial Hessian scale or approximation passed to BFGS/PQN.
            do_numba: Compatibility flag for numba execution paths.
            concentrate_scale: Whether to optimize with innovation variance concentrated out.
            enforce_stationarity: Whether to transform AR parameters into the stationary region.
            enforce_invertibility: Whether to transform MA parameters into the invertible region.
            simple_differencing: Whether to difference data before the state-space likelihood instead of integrating in the state.
            initial_diffuse_variance: Initial variance assigned to diffuse state components.
            multiplicative: Whether to request multiplicative seasonal dynamics.
            diffuse: Whether to use diffuse initialization for integrated state components.
            standardize_endog: Whether to standardize endogenous data during optimization.

        Returns:
            Fitted ``SarimaxResults`` object built from a formula.

        Examples
        --------
        Formula-form SARIMAX fit, using ``simulate_sarima`` to build data:

        >>> import numpy as np, pandas as pd
        >>> from kanly.api import sarimax, simulate_sarima
        >>> y = simulate_sarima(n=500, ar=[0.5, 0.1], seed=0, burnin=1000)
        >>> df = pd.DataFrame({'y': y})
        >>> fit = sarimax('y', df, order=(2, 0, 0))
        >>> print(fit.summary())                          # doctest: +SKIP

        With exogenous regressors via the formula RHS:

        >>> rng = np.random.default_rng(0)
        >>> df['x'] = rng.normal(size=len(df))
        >>> df['y2'] = df['y'] + 1.5 * df['x']
        >>> fit2 = sarimax('y2 ~ x', df, order=(2, 0, 0))      # doctest: +SKIP

        See Also
        --------
        Alias: ``arima``.
        `SARIMAX` : array entry point.
        `bsarimax` : Bayesian SARIMAX wrapper.
        """
        model = SarimaxModel.build_model_from_formula(
            # formula, data, order, seasonal_order, trend, trend_offset, trend_scale, index, debug, check_constant_cols,
            # specification_name, drop_1_for_FE,
            # enforce_invertibility=enforce_invertibility, enforce_stationarity=enforce_stationarity,
            # initial_diffuse_variance=initial_diffuse_variance,
            # simple_differencing=simple_differencing, multiplicative=multiplicative,
            # diffuse=diffuse,
            formula, data, order=order, seasonal_order=seasonal_order,
            trend=trend, trend_offset=trend_offset, index=index, debug=debug,
            check_constant_cols=check_constant_cols, specification_name=specification_name, drop_1_for_FE=drop_1_for_FE,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
            diffuse=diffuse,
        )

        fit = model.fit(start_params=start_params, do_hannan_rissanen=do_hannan_rissanen,
                        concentrate_scale=concentrate_scale,
                        nlags=nlags, debug=debug, cov_type=cov_type,
                        maxiter=maxiter, gtol=gtol, ftol=ftol, xtol=xtol, B0=B0, do_numba=do_numba,
                        initial_diffuse_variance=initial_diffuse_variance,
                        standardize_endog=standardize_endog,
                        **kwargs)
        return fit

    def fit(self, start_params=None, do_hannan_rissanen=DEFAULT_SARIMAX_DO_HANNAN_RISSANEN, debug=False, nlags=None,
            specification_name=None, cov_type=DEFAULT_SARIMAX_COV_TYPE, maxiter=DEFAULT_SARIMAX_MAXITER,
            gtol=DEFAULT_SARIMAX_GTOL, ftol=DEFAULT_SARIMAX_FTOL, xtol=DEFAULT_SARIMAX_XTOL, B0=DEFAULT_SARIMAX_BFGS_B0,
            do_numba=None,
            steady_state_tol=DEFAULT_SARIMAX_STEADY_STATE_TOL,
            initial_diffuse_variance=DEFAULT_SARIMAX_INITIAL_DIFFUSE_VARIANCE,
            enforce_stationarity=DEFAULT_SARIMAX_ENFORCE_STATIONARITY,
            enforce_invertibility=DEFAULT_SARIMAX_ENFORCE_INVERTIBILITY,
            concentrate_scale=DEFAULT_SARIMAX_CONCENTRATE_SCALE,
            standardize_endog=DEFAULT_SARIMAX_STANDARDIZE_ENDOG,
            **kwargs) -> SarimaxResults:

        """
        SARIMAX helper function ``fit``.

        Args:
            start_params: Optional initial parameter vector.
            do_hannan_rissanen: Whether to estimate starting ARMA parameters with Hannan-Rissanen.
            debug: Whether to print fitting diagnostics.
            nlags: Number of lags to compute or use.
            specification_name: Optional display name for the model specification.
            cov_type: Covariance estimator name.
            maxiter: Maximum optimizer iterations.
            gtol: Optimizer projected-gradient tolerance.
            ftol: Optimizer objective_function-change tolerance.
            xtol: Optimizer parameter-change tolerance.
            B0: Initial Hessian scale or approximation passed to BFGS/PQN.
            do_numba: Compatibility flag for numba execution paths.
            steady_state_tol: Tolerance for detecting steady-state Kalman forecast error variance.
            initial_diffuse_variance: Initial variance assigned to diffuse state components.
            enforce_stationarity: Whether to transform AR parameters into the stationary region.
            enforce_invertibility: Whether to transform MA parameters into the invertible region.
            concentrate_scale: Whether to optimize with innovation variance concentrated out.
            standardize_endog: Whether to standardize endogenous data during optimization.

        Returns:
            Fitted ``SarimaxResults`` object.
        """
        result = sarimax_internal(
            self.endog, order=self.order, seasonal_order=self.seasonal_order,
            exog=self.exog, trend=self.trend, trend_offset=self.trend_offset,
            trend_scale=self.trend_scale,
            debug=debug,
            maxiter=maxiter, gtol=gtol, ftol=ftol,
            xtol=xtol,
            B0=B0,
            start_params=start_params, do_hannan_rissanen=do_hannan_rissanen,
            steady_state_tol=steady_state_tol,
            initial_diffuse_variance=initial_diffuse_variance,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
            concentrate_scale=concentrate_scale,
            simple_differencing=self.simple_differencing,
            multiplicative=self.multiplicative,
            diffuse=self.diffuse,
            standardize_endog=standardize_endog,
            **kwargs)

        cov_params, cov_elapsed = get_cov_params(self.loglike_obs, result['params'], cov_type, debug=debug)

        return SarimaxResults(
            self,
            self.order, self.seasonal_order,
            self.nobs, result['params'], cov_params,
            self.df_model, self.df_resid, self.param_names, self.endog_name,
            specification_name=self.specification_name if specification_name is None else specification_name,
            converged=result["converged"],
            optimization_result=result["optimization_result"],
            llf=result["llf"],
            llf_obs=result["llf_obs"],
            fittedvalues=result["fittedvalues"],
            fittedinnovations=result['fittedinnovations'],
            resid=result["resid"],
            forecasts_error_cov=result["forecasts_error_cov"],
            loglike=result["loglike"], loglike_obs=result["loglike_obs"],
            aic=result["aic"], aicc=result["aicc"], bic=result["bic"], hqic=result["hqic"],
            loglikelihood_burn=result["loglikelihood_burn"], arima_options=result['sarimax_options'],
            cov_type=cov_type, trend=self.trend, trend_offset=self.trend_offset,
            grad_norm=result['grad_norm'], iter=result['iter'],
            arparams=result['arparams'], maparams=result['maparams'],
            seasonalarparams=result['seasonalarparams'], seasonalmaparams=result['seasonalmaparams'],
            exogparams=result['exogparams'], trendparams=result['trendparams'],
            model_elapsed=self.model_elapsed, fit_elapsed=result['fit_elapsed'],
            cov_elapsed=cov_elapsed,  # acov0=result['acov0'], #state_vec=result['state_vec'],
            k_trend=self.k_trend, k_exog=self.k_exog,
            state_vec=result['state_vec'], simple_differencing=self.simple_differencing,
            multiplicative=self.multiplicative,
        )

    def build_model(self, data, index=None, debug=False, *args, **kwargs):
        """
        SARIMAX helper function ``build_model``.

        Args:
            data: Data source used by formula parsing.
            index: Optional observation index.
            debug: Whether to print fitting diagnostics.

        Returns:
            Constructed ``SarimaxModel`` instance.
        """
        raise NotImplementedError("Not Implemented for SARIMAX")

    @staticmethod
    def build_model_from_formula(formula, data, order=(0, 0, 0), seasonal_order=None,
                                 trend=None, trend_offset=1, trend_scale=1, index=None, debug=False,
                                 check_constant_cols=True, specification_name=None, drop_1_for_FE=True,
                                 enforce_stationarity=DEFAULT_SARIMAX_ENFORCE_STATIONARITY,
                                 enforce_invertibility=DEFAULT_SARIMAX_ENFORCE_INVERTIBILITY,
                                 diffuse=DEFAULT_SARIMAX_DIFFUSE,
                                 ) -> Self:
        """
        SARIMAX helper function ``build_model_from_formula``.

        Args:
            formula: Formula string describing endogenous and exogenous variables.
            data: Data source used by formula parsing.
            order: Nonseasonal ARIMA order specification ``(p, d, q)``.
            seasonal_order: Optional seasonal order specification ``(P, D, Q, s)``.
            trend: Trend specification, such as None, ``c``, ``t``, ``ct``, ``n``, or an indicator list.
            trend_offset: Starting offset for deterministic trend time indices.
            trend_scale: Scale used to normalize deterministic trend time indices.
            index: Optional observation index.
            debug: Whether to print fitting diagnostics.
            check_constant_cols: Whether to check for constant columns in formula-built exog.
            specification_name: Optional display name for the model specification.
            drop_1_for_FE: Whether formula parsing drops singleton fixed-effect indicators.
            enforce_stationarity: Whether to transform AR parameters into the stationary region.
            enforce_invertibility: Whether to transform MA parameters into the invertible region.
            diffuse: Whether to use diffuse initialization for integrated state components.

        Returns:
            Constructed ``SarimaxModel`` instance.
        """
        _t = time.time()

        no_rhs = '~'  not in formula
        if no_rhs:
            formula = formula + " ~ " + formula

        formula += ' -1'

        data = dict_2_dataframe(data)

        result = SparseDataGetter.get_data(data=data, formula=formula, debug=debug, index=index,
                                           check_constant_cols=check_constant_cols, drop_1_for_FE=drop_1_for_FE)

        assert result[HAS_INTERCEPT_KEY] is False
        assert result[WEIGHTS_KEY] is None
        assert result[INSTRUMENTS_KEY] is None
        assert result[ABSORB_KEY] is None
        if index is None:
            assert len(result[VALID_OBS_ROWS_KEY]) == len(data)
        else:
            assert len(result[VALID_OBS_ROWS_KEY]) == len(data.index[index])

        endog = result[ENDOG_KEY].values.toarray().flatten()
        exog = result[EXOG_KEY].values.toarray()
        endog_name = result[ENDOG_KEY].column_names[0]
        exog_names = result[EXOG_KEY].column_names
        exog_term_names = result[EXOG_KEY].term_names
        formula_design_info = result[FORMULA_DESIGN_INFO_KEY]

        if no_rhs:
            exog = None
            exog_names = None
            exog_term_names = None

        # TODO valid obs, fail on missing?

        model_elapsed = time.time() - _t

        return SarimaxModel(endog, exog, order, seasonal_order, trend, trend_scale, trend_offset, formula_design_info,
                            endog_name=endog_name, exog_names=exog_names, exog_term_names=exog_term_names,
                            data=data, specification_name=specification_name,
                            index=index, model_elapsed=model_elapsed, debug=debug,
                            enforce_stationarity=enforce_stationarity,
                            enforce_invertibility=enforce_invertibility,
                            diffuse=diffuse
                            )

    def to_bayesian_model(self, priors=None, bounds=None) -> BayesianModel:
        """
        SARIMAX helper function ``to_bayesian_model``.

        Args:
            priors: Parameter ``priors`` passed to the SARIMAX helper.
            bounds: Parameter ``bounds`` passed to the SARIMAX helper.

        Returns:
            Bayesian model representation.
        """
        if bounds is None:
            bounds = dict()
        if 'sigma2' in bounds:
            bounds['sigma2'] = (0, bounds['sigma2'][1])
            assert bounds['sigma2'][1] > bounds['sigma2'][0]
        else:
            bounds['sigma2'] = (0, np.inf)

        bmodel = BayesianModel(self.loglike, param_names=self.param_names, priors=priors,
                               bounds=bounds)
        return bmodel

    @staticmethod
    def bsarimax(formula, data, order=(0, 0, 0), seasonal_order=None, trend=None, trend_offset=1, index=None, debug=False,
                 check_constant_cols=True, specification_name=None, drop_1_for_FE=True,
                 priors=None, bounds=None,
                 enforce_stationarity=DEFAULT_SARIMAX_ENFORCE_STATIONARITY,
                 enforce_invertibility=DEFAULT_SARIMAX_ENFORCE_INVERTIBILITY,
                 ) -> BayesianModel:

        """
        SARIMAX helper function ``bsarimax``.

        Args:
            formula: Formula string describing endogenous and exogenous variables.
            data: Data source used by formula parsing.
            order: Nonseasonal ARIMA order specification ``(p, d, q)``.
            seasonal_order: Optional seasonal order specification ``(P, D, Q, s)``.
            trend: Trend specification, such as None, ``c``, ``t``, ``ct``, ``n``, or an indicator list.
            trend_offset: Starting offset for deterministic trend time indices.
            index: Optional observation index.
            debug: Whether to print fitting diagnostics.
            check_constant_cols: Whether to check for constant columns in formula-built exog.
            specification_name: Optional display name for the model specification.
            drop_1_for_FE: Whether formula parsing drops singleton fixed-effect indicators.
            priors: Parameter ``priors`` passed to the SARIMAX helper.
            bounds: Parameter ``bounds`` passed to the SARIMAX helper.
            enforce_stationarity: Whether to transform AR parameters into the stationary region.
            enforce_invertibility: Whether to transform MA parameters into the invertible region.

        Returns:
            Bayesian SARIMAX model representation.

        Examples
        --------
        Build a Bayesian SARIMAX model and sample its posterior:

        >>> import numpy as np, pandas as pd
        >>> from kanly.api import bsarimax, simulate_sarima
        >>> y = simulate_sarima(n=500, ar=[0.5, 0.1], seed=0, burnin=1000)
        >>> df = pd.DataFrame({'y': y})
        >>> bmodel = bsarimax('y', df, order=(2, 0, 0),
        ...                   priors={'ar.L1': 'norm(0, 1)'})    # doctest: +SKIP
        >>> fit = bmodel.sample([0.0, 0.0, 1.0],                 # doctest: +SKIP
        ...                     n_samples=5_000, n_burnin=1_000,
        ...                     n_chains=4)
        """
        model = SarimaxModel.build_model_from_formula(
            formula, data, order, seasonal_order, trend, trend_offset, index, debug, check_constant_cols,
            specification_name, drop_1_for_FE, debug=debug,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility, )

        return model.to_bayesian_model(priors=priors, bounds=bounds)

    # def get_differenced(self):
    #     return difference(self.endog, self.d, self.D, self.s)

# #
# if __name__ == '__main__':
#
#     from kanly.api import simulate_sarima
#     e = simulate_sarima(n=500, ar=[.5, .1], seed=0, burnin=1000)
#     print(SarimaxModel.SARIMAX(e, order=(2,0,0)))
#
#     e = simulate_sarima(n=500, ar=[.5, .1], seed=0, burnin=1000)
#     print(SarimaxModel.sarimax('e', {'e': e}, order=(2,0,0)))
