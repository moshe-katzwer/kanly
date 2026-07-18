"""Small helper utilities for formula parsing and lag construction."""
from __future__ import absolute_import, print_function

import re
import numpy as np


def temp_replace(formula, expr, dummy_char):
    """Temporarily replace regex matches with stable placeholder tokens.

    This helper is used while parsing formula strings to protect quoted or
    otherwise sensitive substrings from subsequent text splitting/replacement.

    Args:
        formula (str): Original formula string.
        expr (str): Regular expression used to find substrings to protect.
        dummy_char (str): Prefix character used to build placeholder tokens.

    Returns:
        tuple:
            - str: Formula with protected substrings replaced by placeholders.
            - dict: Mapping from original substring to placeholder token.
    """
    dbl_qts = re.findall(expr, formula)
    dbl_qt_dict = {q: dummy_char * 10 + str(i).zfill(10) for i, q in enumerate(dbl_qts)}

    for k, v in dbl_qt_dict.items():
        formula = formula.replace(k, v)

    return formula, dbl_qt_dict


def is_float(num):
    """Check whether a value can be safely cast to ``float``.

    Args:
        num (Any): Value to test.

    Returns:
        bool: True when ``float(num)`` succeeds, else False.
    """
    try:
        float(num)
        return True
    except ValueError:
        return False


def Lag(x, l=1):
    """Shift a one-dimensional array forward/backward, filling with NaN.

    Positive lags create leading NaNs (classic lag operator). Negative values
    create trailing NaNs (a lead operator).

    Args:
        x (array-like): Input values.
        l (int): Number of periods to shift. Positive for lag, negative for
            lead, and zero for identity.

    Returns:
        ndarray: Shifted array with NaN padding.
    """
    x = np.asarray(x)
    if l == 0:
        return x
    if l > 0:
        return np.hstack(([np.nan] * l, x[:-l]))
    else:
        return np.hstack((x[l:], [np.nan] * l))
