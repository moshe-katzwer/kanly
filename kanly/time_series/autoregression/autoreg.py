"""
Autoregressive and distributed-lag time series model wrappers for kanly.

This module provides lightweight convenience interfaces for fitting common
time series regression specifications using the ordinary least squares
infrastructure provided by :class:`kanly.regression.linear_models.SparseLinearModel`.

The implementations here do not use specialized likelihood-based estimation
routines. Instead, they construct deterministic design matrices containing
lagged endogenous variables, lagged exogenous variables, seasonal dummy
variables, and deterministic trend terms, then delegate estimation to
`SparseLinearModel.lm` or `SparseLinearModel.LM`.

Two interface styles are provided:

Formula interfaces
------------------
- `autoreg`
    Formula-based autoregressive regression wrapper.

- `ardl`
    Formula-based autoregressive distributed lag regression wrapper.

Array / matrix interfaces
-------------------------
- `AUTOREG`
    Matrix-based autoregressive regression wrapper.

- `ARDL`
    Matrix-based autoregressive distributed lag regression wrapper.

These functions are conceptually similar to the corresponding routines in
`statsmodels.tsa`, especially:

- `statsmodels.tsa.ar_model.AutoReg`
- `statsmodels.tsa.ardl.ARDL`

but are intentionally implemented as thin OLS wrappers integrated with
kanly's formula system, sparse matrix support, and deterministic feature
generators.

Notes
-----
These models are estimated using ordinary least squares and inherit all
behavior, numerical methods, and covariance estimation options from the
underlying linear model routines.

Seasonality is represented through generated seasonal fixed-effect matrices
rather than specialized state-space or frequency-domain methods.

Formula interfaces expand lag, trend, and seasonal specifications into
kanly formula syntax before estimation. Array interfaces construct the
corresponding regression matrices directly.

Sparse exogenous matrices are supported throughout where possible.
"""
from __future__ import absolute_import, print_function

import itertools

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.sparse import isspmatrix, hstack as sp_hstack

from kanly.formula.keys import ENDOG_KEY, EXOG_KEY, WEIGHTS_KEY, INSTRUMENTS_KEY
from kanly.formula.parse_formula import parse_formula
from kanly.formula.seasonal_and_trend_matrices import generate_seasonal_matrix, generate_trend_matrix
from kanly.regression.linear_models.model import SparseLinearModel
from kanly.sparse_data_frame import SparseDataFrame


def parse_trend(trend):
    if trend is None:
        trend = 'c'
    if isinstance(trend, str):
        trend = trend.lower().replace(' ', '')
        trend_dict = {
            'n': [], 'c': [0], 't': [1], 'ct': [0, 1], 'ctt': [0, 1, 2]
        }
        if trend in trend_dict:
            trend = trend_dict[trend]
        else:
            raise ValueError(f"`trend` must be a string in {list(trend_dict.keys())} or an iterable of ints!")
    return sorted(set(trend))


def autoreg(formula, data, lags=0, trend='c', seasonal_periods=None, **kwargs):
    """
    Fit an Autoregressive (AR) model via formula manipulation using `kanly.SparseLinearModel.lm`.

    This function acts as a wrapper that translates high-level autoregressive components
    (lags, trends, and seasonal effects) into a formula string compatible with `kanly`'s
    linear model syntax, analogous to `statsmodels.tsa.ar_model.AutoReg`.

    Parameters
    ----------
    formula : str
        A patsy-like formula string defining the dependent variable (e.g., 'y ~ 1' or 'y ~ x').
        If the formula ends with '-1', the intercept omission is temporarily stripped and
        re-evaluated based on the `trend` parameter.
    data : pandas.DataFrame or dict-like
        The dataset containing the variables specified in the formula.
    lags : int or iterable of ints, default 0
        The number of lags to include.
        - If an int, includes all lags from 1 up to `lags`.
        - If an iterable, includes the specific lags requested.
    trend : str or iterable of ints, default 'c'
        The trend component to include:
        - 'n': No trend, no intercept.
        - 'c': Constant / intercept only.
        - 't': Time trend only.
        - 'ct': Constant and time trend.
        - 'ctt': Constant, linear time trend, and quadratic time trend.
        - Alternatively, an iterable of polynomial powers (e.g., `[0, 1]` for constant and linear).
    seasonal_periods : int, iterable of ints, or None, default None
        The period(s) for seasonal fixed effects. For example, 4 for quarterly data or 12
        for monthly data. Periods <= 1 are automatically filtered out.
    **kwargs
        Additional keyword arguments passed directly to `SparseLinearModel.lm`.

    Returns
    -------
    kanly.SparseLinearModel
        The fitted linear model object.

    Raises
    ------
    ValueError
        If the `trend` string provided is not one of 'n', 'c', 't', 'ct', or 'ctt'.

    See Also
    --------
    statsmodels.tsa.ar_model.AutoReg : The statsmodels implementation of autoregressive models.

    Notes
    -----
    While `statsmodels.tsa.ar_model.AutoReg` handles lags by shift-transforming the underlying
    arrays natively, this function leverages `kanly`'s specialized formula syntax sugar
    (e.g., `L(y, 1)`, `trend()`, `seasonal()`) to dynamically construct a comprehensive OLS formula.
    """
    # 1. Standardize formula and strip explicit intercept removal ('-1') if present.
    # This allows the 'trend' parameter to dynamically govern intercept presence later.
    formula = formula.strip()
    if formula[-2:] == '-1':
        formula = formula[:-2]

    # 2. Isolate the endogenous (dependent) variable name to build lag features.
    result = parse_formula(formula)
    endog_name, exog_names, weights_name, instruments_names \
        = result[ENDOG_KEY][0], result[EXOG_KEY], result[WEIGHTS_KEY], result[INSTRUMENTS_KEY]

    if instruments_names is not None:
        raise Exception("No instruments in ARDL!")

    # 3. Append lag terms to the formula.
    # Converts an integer max-lag into a sequential range (e.g., 3 -> [1, 2, 3]).
    if isinstance(lags, int):
        lags = range(1, lags + 1)
    for l in lags:
        formula += f' + L({endog_name}, {l})'

    # 4. Map the shorthand trend string to numerical polynomial degrees.
    # 0 represents intercept, 1 represents linear trend, 2 represents quadratic trend.
    trend = parse_trend(trend)

    # Extract deterministic time trends (anything greater than 0)
    trend_wout_zero = sorted(set(trend) - {0})

    # 5. Append seasonal fixed effects if requested.
    if seasonal_periods is not None:
        if isinstance(seasonal_periods, int):
            # Convert single integer period to a list for standard processing
            seasonal_periods = [seasonal_periods]

        # Deduplicate, sort, and ensure periods are structurally meaningful (> 1)
        seasonal_periods = sorted(set(seasonal_periods))
        seasonal_periods = [s for s in seasonal_periods if s > 1]

        if len(seasonal_periods):
            formula += f' + seasonal({seasonal_periods})'

    # 6. Append the deterministic time trend syntax to the formula if applicable.
    if len(trend_wout_zero):
        formula += f' + trend({trend_wout_zero})'

    # 7. Enforce intercept suppression if 0 (the intercept marker) is omitted from `trend`.
    if 0 not in trend:
        formula += ' -1'

    # 8. Delegate the fully constructed macro formula to kanly's underlying OLS engine.
    return SparseLinearModel.lm(formula, data, **kwargs)


def AUTOREG(endog, exog=None, lags=0, trend='c', seasonal_periods=None,
            has_constant=False, add_constant=False, endog_name=None,
            exog_names=None, weights=None, **kwargs):
    """
    Fit an autoregressive linear model using ordinary least squares.

    This is a convenience wrapper around :meth:`SparseLinearModel.LM`
    that constructs a regression design matrix containing optional:

    - lagged values of the endogenous variable,
    - deterministic trend terms,
    - seasonal dummy variables,
    - user-supplied exogenous regressors.

    The resulting model is estimated using OLS (or weighted OLS when
    ``weights`` are provided).

    Unlike :class:`statsmodels.tsa.ar_model.AutoReg`, seasonal structure
    is specified entirely through the ``seasonal_periods`` argument,
    which may contain one or more seasonal cycle lengths.

    Parameters
    ----------
    endog : array-like
        Endogenous response variable of shape ``(n_obs,)``.

    exog : array-like or sparse matrix, optional
        Optional exogenous regressors of shape ``(n_obs, n_exog)``.

    lags : int, default 0
        Number of autoregressive lags to include. Lagged values of
        ``endog`` are added as regressors named
        ``L1[y]``, ``L2[y]``, ..., ``Lp[y]``.

    trend : str or sequence, default 'c'
        Trend specification passed to
        :meth:`SparseLinearModel.parse_trend`.

        Common values include:

        - ``'n'`` : no deterministic trend
        - ``'c'`` : constant term
        - ``'t'`` : linear time trend
        - ``'ct'`` : constant and linear trend

    seasonal_periods : int or sequence of int, optional
        Seasonal cycle lengths used to generate seasonal dummy variables.

        Examples
        --------
        ``12`` adds monthly seasonality.

        ``[7, 365]`` adds both weekly and yearly seasonality.

        If a constant is present, one seasonal column per period is
        dropped to avoid perfect multicollinearity.

    has_constant : bool, default False
        If ``True``, indicates that ``exog`` already contains a constant
        column. Used to avoid adding duplicate intercept terms or
        redundant seasonal dummy columns.

    add_constant : bool, default False
        If ``True``, force inclusion of a constant term in the generated
        trend specification.

        Cannot be used together with ``has_constant=True``.

    endog_name : str, optional
        Name used when labeling lagged endogenous regressors.
        Defaults to ``'<y>'``.

    exog_names : sequence of str, optional
        Column names for exogenous regressors. If omitted, names are
        generated automatically as ``<x0>``, ``<x1>``, etc.

    weights : array-like, optional
        Observation weights passed through to
        :meth:`SparseLinearModel.LM`.

    **kwargs
        Additional keyword arguments forwarded to
        :meth:`SparseLinearModel.LM`.

    Returns
    -------
    LinearModelResults
        Result object returned by :meth:`SparseLinearModel.LM`.

    Notes
    -----
    This function does not estimate autoregressive parameters using a
    specialized likelihood-based routine. It simply constructs the
    appropriate lagged regression matrix and fits the model via OLS.

    When ``lags > 0``, the first ``lags`` observations are discarded so
    that all lagged regressors are fully observed.

    Sparse exogenous inputs are preserved and combined using sparse
    matrix operations where possible.
    """

    if add_constant and has_constant:
        raise Exception("Cannot both add a constant and say there is a constant!")

    n = len(endog)

    is_sparse = exog is not None and isspmatrix(exog)

    if endog_name is None:
        endog_name = '<y>'

    if exog_names is None:
        if exog is not None:
            exog_names = [f'<x{j}>' for j in range(exog.shape[1])]
        else:
            exog_names = []

    exog_to_stack = []
    if exog is not None:
        if lags:
            exog = exog[lags:]
        exog_to_stack = [exog]

    trend = parse_trend(trend)
    if add_constant and trend[0] != 0:
        trend = [0] + trend
    elif has_constant and trend[0] == 0:
        trend = trend[1:]

    if seasonal_periods is not None:
        if isinstance(seasonal_periods, int):
            seasonal_periods = [seasonal_periods]
        if len(seasonal_periods):
            mat0, colnames0 = generate_seasonal_matrix(
                seasonal_periods, n, return_array=True, return_dense=not is_sparse,
                drop1=has_constant or add_constant or 'c' in trend or 0 in trend)
            if lags:
                mat0 = mat0[lags:]
            exog_to_stack.append(mat0)
            exog_names += colnames0

    if len(trend):
        mat1, colnames1 = generate_trend_matrix(trend, n, return_array=True, return_dense=not is_sparse)
        if lags:
            mat1 = mat1[lags:]
        exog_to_stack = [mat1] + exog_to_stack
        exog_names = colnames1 + exog_names

    if lags:
        lag_matrix = np.zeros((n - lags, lags))
        lag_cols = [f'L{l}[{endog_name}]' for l in range(1, lags + 1)]
        for l in range(1, lags + 1):
            lag_matrix[:, l - 1] = endog[lags - l:n - l]
        exog_to_stack.append(lag_matrix)
        exog_names += lag_cols

    if is_sparse:
        exog = sp_hstack(exog_to_stack, format='csc')
    else:
        exog = np.hstack(exog_to_stack)

    endog = endog[lags:]

    return SparseLinearModel.LM(endog, exog, weights=weights, exog_names=exog_names, **kwargs)


def _get_order(k_exog, lags, order, causal, exog_names=None):
    if order is None:
        order = 0
    if isinstance(order, int):
        order = {i: list(range(causal, order+1)) for i in (range(k_exog) if exog_names is None else exog_names)}
    elif isinstance(order, dict):
        order = {k: list(range(causal, o + 1)) if isinstance(o, int) else o
                 for k, o in order.items()}
    if causal:
        order = {k: list(set(o) - {0}) for k, o in order.items()}

    max_order_lag = np.max([0] + [np.max(o) for o in order.values() if len(o)])
    max_lag = max(max_order_lag, lags if isinstance(lags, int) else max(lags))

    return order, max_lag


def ardl(formula, data, order=0, lags=0, trend='c', seasonal_periods=None, causal=False, **kwargs):
    """
    Fit an autoregressive distributed lag (ARDL) model using kanly's
    formula interface.

    This function acts as a high-level wrapper that converts ARDL
    specifications into an expanded regression formula compatible with
    :meth:`SparseLinearModel.lm`.

    The resulting model may include:

    - lagged values of the dependent variable,
    - contemporaneous and lagged exogenous variables,
    - deterministic trend terms,
    - seasonal fixed effects,

    and is estimated via ordinary least squares.

    Parameters
    ----------
    formula : str
        Formula defining the dependent and exogenous variables.

        Examples
        --------
        ``'y ~ x1 + x2'``

        ``'y ~ x1 + x2 -1'``

    data : pandas.DataFrame or dict-like
        Dataset containing all variables referenced in the formula.

    order : int or dict, default 0
        Lag structure for exogenous regressors.

        If an integer ``q`` is supplied, every exogenous variable receives
        lags ``0, 1, ..., q``.

        If a dictionary is supplied, keys correspond to exogenous variable
        names and values specify lag structures individually.

        Examples
        --------
        ``order=2``

            Includes lags ``0, 1, 2`` for all exogenous regressors.

        ``order={'x1': 2, 'x2': [1, 3]}``

            Applies different lag structures by regressor.

    lags : int or iterable of int, default 0
        Autoregressive lag specification for the dependent variable.

        If an integer ``p`` is supplied, lags
        ``1, 2, ..., p`` are included.

        If an iterable is supplied, only the specified lags are used.

    trend : str or iterable of int, default 'c'
        Deterministic trend specification.

        Common values include:

        - ``'n'`` : no trend
        - ``'c'`` : intercept only
        - ``'t'`` : linear trend only
        - ``'ct'`` : intercept and linear trend
        - ``'ctt'`` : intercept, linear trend, and quadratic trend

        Alternatively, an iterable of polynomial powers may be supplied.

    seasonal_periods : int, iterable of int, or None, default None
        Seasonal cycle lengths used to generate seasonal fixed effects.

        Examples
        --------
        ``12`` adds monthly seasonality.

        ``[7, 365]`` adds weekly and yearly seasonal effects.

    causal : bool, default False
        Whether to impose a causal distributed lag structure.

        If ``True``, contemporaneous exogenous terms (lag 0) are excluded
        from the distributed lag expansion.

    **kwargs
        Additional keyword arguments forwarded directly to
        :meth:`SparseLinearModel.lm`.

    Returns
    -------
    LinearModelResults
        Result object returned by :meth:`SparseLinearModel.lm`.

    Raises
    ------
    Exception
        If instrumental variables are present in the formula.

    Notes
    -----
    This function expands the supplied formula into an equivalent OLS
    regression containing lagged endogenous and exogenous variables.

    For example:

    >>> ardl('y ~ x', data, lags=2, order=1)

    expands approximately to:

    >>> y ~ x + L(y,1) + L(y,2) + L(x,1)

    together with any requested trend and seasonal terms.

    Estimation is performed entirely through ordinary least squares rather
    than specialized ARDL likelihood methods.
    """
    result = parse_formula(formula)
    endog_name, exog_names, weights_name, instruments_names \
        = result[ENDOG_KEY][0], result[EXOG_KEY], result[WEIGHTS_KEY], result[INSTRUMENTS_KEY]
    if instruments_names is not None:
        raise Exception("No instruments in ARDL!")
    k_exog = len(exog_names)

    trend = parse_trend(trend)
    trend_wout_zero = sorted(set(trend) - {0})
    if 0 in trend and '-1' in exog_names:
        raise Exception("Cannot remove constant from formula and specify constant!")
    remove_constant = '-1' in exog_names or 0 not in trend
    if '-1' in exog_names:
        exog_names = exog_names[:-1]

    if isinstance(lags, int):
        lags = range(1, lags + 1)

    # say it is causal here - we'll remove from exog if casual
    order, max_lag = _get_order(k_exog, lags, order, causal=True, exog_names=exog_names)
    order_plus_lags = dict(**{endog_name: lags}, **order)

    order_string = ' + '.join(itertools.chain.from_iterable([[f'L({xn},{l})' for l in o] for xn, o in order_plus_lags.items()]))
    if causal:
        exog_names = [e for e in exog_names if e not in order]
    formula = endog_name + " ~ " + ' + '.join(exog_names)
    if len(order_string):
        formula += ' + ' + order_string

    if seasonal_periods is not None:
        if isinstance(seasonal_periods, int):
            # Convert single integer period to a list for standard processing
            seasonal_periods = [seasonal_periods]

        # Deduplicate, sort, and ensure periods are structurally meaningful (> 1)
        seasonal_periods = sorted(set(seasonal_periods))
        seasonal_periods = [s for s in seasonal_periods if s > 1]

        if len(seasonal_periods):
            formula += f' + seasonal({seasonal_periods})'

    # 6. Append the deterministic time trend syntax to the formula if applicable.
    if len(trend_wout_zero):
        formula += f' + trend({trend_wout_zero})'

    # 7. Enforce intercept suppression if 0 (the intercept marker) is omitted from `trend`.
    if 0 not in trend:
        formula += ' -1'

    return SparseLinearModel.lm(formula, data, **kwargs)


def ARDL(endog, exog=None, fixed=None, order=0, lags=0, trend='c', seasonal_periods=None,
         causal=False, has_constant=False, add_constant=False, endog_name=None,
         exog_names=None, weights=None, **kwargs):
    """
    Fit an autoregressive distributed lag (ARDL) model using ordinary least squares.

    This function is a convenience wrapper around :meth:`AUTOREG` and
    ultimately :meth:`SparseLinearModel.LM`. It constructs a regression
    design matrix containing:

    - lagged values of the endogenous variable,
    - lagged values of exogenous regressors,
    - optional deterministic trend terms,
    - optional seasonal fixed effects,

    and estimates the resulting linear regression via OLS (or weighted OLS).

    The interface is conceptually similar to
    :class:`statsmodels.tsa.ardl.ARDL`, though this implementation is a
    direct regression-based wrapper rather than a specialized time series
    estimation routine.

    Parameters
    ----------
    endog : array-like
        Endogenous response variable of shape ``(n_obs,)``.

    exog : array-like, DataFrame, sparse matrix, optional
        Exogenous regressors of shape ``(n_obs, n_exog)``.

        If a pandas or sparse dataframe is supplied, column names are used
        automatically when constructing lag labels.

    order : int, dict, or iterable specification, default 0
        Lag structure for exogenous regressors.

        If an integer ``q`` is provided, each exogenous variable receives
        lags ``0, 1, ..., q``.

        If ``causal=True``, contemporaneous lag ``0`` is excluded.

        If a dictionary is provided, keys correspond to exogenous variable
        indices (or column labels when using dataframes), and values specify
        lag structures individually.

        Examples
        --------
        ``order=2``

            Includes lags ``0, 1, 2`` for all exogenous variables.

        ``order={0: [0, 1], 1: [1, 2, 3]}``

            Uses different lag structures for different regressors.

    lags : int, default 0
        Number of autoregressive lags of ``endog`` to include.

        Lagged endogenous regressors are labeled:

        ``L1[y]``, ``L2[y]``, ..., ``Lp[y]``.

    trend : str or sequence, default 'c'
        Deterministic trend specification.

        Common options include:

        - ``'n'`` : no deterministic trend
        - ``'c'`` : constant term
        - ``'t'`` : linear time trend
        - ``'ct'`` : constant and linear trend
        - ``'ctt'`` : constant, linear, and quadratic trend

        Alternatively, a sequence of polynomial powers may be supplied.

    seasonal_periods : int or sequence of int, optional
        Seasonal cycle lengths used to generate seasonal dummy variables.

        Examples
        --------
        ``12`` adds monthly seasonality.

        ``[7, 365]`` adds weekly and yearly seasonal effects.

    causal : bool, default False
        Whether to impose a causal distributed lag structure.

        If ``True``, lag ``0`` of exogenous regressors is excluded so that
        only strictly lagged exogenous values enter the model.

    has_constant : bool, default False
        Indicates that ``exog`` already contains a constant column.

        Used to avoid duplicate intercept terms and redundant seasonal
        dummy columns.

    add_constant : bool, default False
        Force inclusion of a constant term in the generated trend structure.

        Cannot be used simultaneously with ``has_constant=True``.

    endog_name : str, optional
        Name used when labeling lagged endogenous regressors.

        Defaults to ``'<y>'``.

    exog_names : sequence of str, optional
        Names for exogenous regressors when ``exog`` is not a dataframe.

        Defaults to ``<x0>``, ``<x1>``, etc.

    weights : array-like, optional
        Observation weights passed through to
        :meth:`SparseLinearModel.LM`.

        When lagging removes initial observations, weights are truncated
        accordingly.

    **kwargs
        Additional keyword arguments forwarded to
        :meth:`SparseLinearModel.LM`.

    Returns
    -------
    LinearModelResults
        Result object returned by :meth:`SparseLinearModel.LM`.

    Notes
    -----
    The model is estimated by explicitly constructing a lagged regression
    matrix and fitting the resulting specification via OLS.

    Let:

    - ``p`` denote the autoregressive lag order,
    - ``q_j`` denote the lag order of exogenous regressor ``j``.

    Then the fitted model has the general form:

    .. math::

        y_t =
        \\sum_{i=1}^{p} \\phi_i y_{t-i}
        +
        \\sum_{j=1}^{k}
        \\sum_{l \\in q_j}
        \\beta_{j,l} x_{j,t-l}
        +
        d_t + s_t + \\varepsilon_t

    where:

    - ``d_t`` represents deterministic trend components,
    - ``s_t`` represents seasonal fixed effects.

    Observations lost due to lagging are removed automatically using the
    maximum lag implied by either ``lags`` or ``order``.

    Sparse matrices are supported where possible through the underlying
    kanly linear model infrastructure.
    """

    n = len(endog)
    k_exog = 0 if exog is None else exog.shape[1]
    k_fixed = 0 if fixed is None else fixed.shape[1]

    exog_is_dataframe = exog is not None and isinstance(exog, (DataFrame, SparseDataFrame))
    fixed_is_dataframe = fixed is not None and isinstance(fixed, (DataFrame, SparseDataFrame))

    if endog_name is None:
        endog_name = '<y>'
    if exog_names is None:
        if exog_is_dataframe:
            exog_names = exog.columns
        else:
            exog_names = [f'<x{j}>' for j in range(k_exog)]

    order, max_lag = _get_order(k_exog, lags, order, causal)

    n_cols = sum([len(o) for o in order.values()]) + lags + k_fixed
    exog_names_expanded = []
    X_ = np.zeros((n - max_lag, n_cols))
    i = 0
    for l in range(1, lags+1):
        X_[:,i] = endog[max_lag-l:n-l]
        i += 1
        exog_names_expanded.append(f'L{l}[{endog_name}]')
    for k, o in order.items():
        v = exog[k].values if exog_is_dataframe else exog[:, k]
        for l in o:
            X_[:,i] = v[max_lag-l:n-l]
            i += 1
            exog_names_expanded.append(f'{f"L{l}[" if l else ""}{k if exog_is_dataframe else exog_names[k]}{"}" if l else ""}')
    X_[:,n_cols-k_fixed:n_cols] = fixed[max_lag:]
    exog_names_expanded += [fixed.columns[j] if fixed_is_dataframe else f'<w{j}>'
                            for j in range(k_fixed)]

    y_ = endog[max_lag:]

    if weights is not None:
        weights = weights[max_lag:]

    return AUTOREG(
        y_, exog=X_, lags=0, trend=trend, seasonal_periods=seasonal_periods,
        has_constant=has_constant, add_constant=add_constant, endog_name=endog_name,
        exog_names=exog_names_expanded,
        weights=weights, **kwargs
    )
