from __future__ import absolute_import, print_function

import numpy as np


def format_lag_order(lag_order):
    """
    Converts an int lag-length e.g. `3` to a tuple of lags (1,2,3),
    otherwise validates an inputed iterable e.g. `p=[1,2,7]` for lags
    at 1,2,7

    Args:
        lag_order: Integer maximum lag or iterable of explicit positive lags.

    Returns:
        Tuple ``(lag_list, k_lags)`` where ``lag_list`` contains active lags
        and ``k_lags`` is its length.
    """
    if isinstance(lag_order, int):
        assert lag_order >= 0
        lag_list = tuple(j for j in range(1, lag_order + 1))
    else:
        lag_order = tuple(sorted(set(lag_order)))
        assert np.all([isinstance(j, int) and j >= 1 for j in lag_order])
        lag_list = lag_order

    return lag_list, len(lag_list)


def validate_orders(order, seasonal_order=None):
    """
    Validate and parse SARIMAX order specifications into internal metadata.

    Converts the user-facing ``(p, d, q)`` and ``(P, D, Q, s)`` tuples into
    the internal representation used throughout the SARIMAX pipeline.  Integer
    lag orders are expanded to explicit lag lists (e.g. ``p=3`` becomes
    ``ar_lags=(1, 2, 3)``), and iterable lag orders are validated and sorted.
    Overlapping nonseasonal and seasonal lags (e.g. a nonseasonal AR lag at 4
    and a seasonal AR lag at 4 when ``s=4``) raise an exception.

    Args:
        order: Nonseasonal ARIMA order specification ``(p, d, q)``.
        seasonal_order: Optional seasonal order specification ``(P, D, Q, s)``.

    Returns:
        4-tuple structured as:
        ``((order, seasonal_order), is_seasonal, (has_ar_terms, has_ma_terms),
        (((p, ar_lags, k_ar), d, (q, ma_lags, k_ma)),
         ((P, sar_lags, k_sar), D, (Q, sma_lags, k_sma), s)))``

        - ``is_seasonal``: bool indicating seasonal AR/MA or seasonal differencing.
        - ``has_ar_terms`` / ``has_ma_terms``: bools indicating any AR or MA lags.
        - ``ar_lags``, ``ma_lags``, ``sar_lags``, ``sma_lags``: tuples of active lag indices.
        - ``k_ar``, ``k_ma``, ``k_sar``, ``k_sma``: lag counts.
    """
    order = tuple(order)
    assert len(order) == 3

    p, d, q = order
    assert isinstance(d, int) and d >= 0

    ar_lags, k_ar = format_lag_order(p)
    ma_lags, k_ma = format_lag_order(q)

    if seasonal_order is None:
        P = D = Q = k_sar = k_sma = 0
        s = 2
        sar_lags = sma_lags = tuple()
        seasonal_order = (P, D, Q, s)
    else:
        P, D, Q, s = seasonal_order
        assert isinstance(s, int)
        if s < 1:
            raise Exception("Seasonality parameter 's' must be greater than 1!")
        assert isinstance(d, int) and D >= 0
        sar_lags, k_sar = format_lag_order(P)
        sma_lags, k_sma = format_lag_order(Q)

    intersections = set(ar_lags) & set([s*l for l in sar_lags])
    if len(intersections):
        raise Exception(f"Invalid Specification: lag(s) {intersections} appear in both"
                        f" seasonal and non-seasonal AR terms!")

    intersections = set(ma_lags) & set([s*l for l in sma_lags])
    if len(intersections):
        raise Exception(f"Invalid Specification: lag(s) {intersections} appear in both"
                        f" seasonal and non-seasonal MA terms!")

    is_seasonal = len(sar_lags) or D or len(sma_lags)

    has_ar_terms = len(ar_lags) or len(sar_lags)
    has_ma_terms = len(ma_lags) or len(sma_lags)

    return (
        (order, seasonal_order),
        is_seasonal,
        (has_ar_terms, has_ma_terms),
        (((p, ar_lags, k_ar), d, (q, ma_lags, k_ma)),
         ((P, sar_lags, k_sar), D, (Q, sma_lags, k_sma), s))
    )
