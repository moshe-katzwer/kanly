"""Default predicates that mark invalid rows during formula matrix building.

Each helper returns a boolean mask where True means the corresponding row is
invalid and should be removed from the design/response blocks.
"""

from __future__ import absolute_import, print_function

import numpy as np
from pandas.api.types import is_numeric_dtype


def default_categorical_invalid_row_func(v):
    """
    Categorical controls must be
        (i)  not nan/inf if numeric
        (ii) not null if object

    Args:
        v (Series or array-like): Candidate categorical column.

    Returns:
        ndarray or Series[bool]: True where values are invalid.
    """
    if is_numeric_dtype(v):
        return ~np.isfinite(v)
    else:
        return v.isnull()


def default_weights_invalid_row_func(v):
    """Weights must be a numeric type, not nan/inf, and strictly positive.

    Args:
        v (Series or array-like): Weights column.

    Returns:
        ndarray[bool]: True where weights are invalid.

    Raises:
        Exception: If the column is not numeric.
    """
    if not is_numeric_dtype(v):
        raise Exception(f"Weights must be a numerical dtype; has dtype {v.dtype}")
    return default_numerical_invalid_row_func(v) | np.array(v <= 0)


def default_numerical_invalid_row_func(v):
    """A numerical variable must be numeric type, not nan/inf.

    Args:
        v (Series or array-like): Numeric candidate column.

    Returns:
        ndarray[bool]: True where values are invalid.

    Raises:
        Exception: If the column is not numeric.
    """
    if is_numeric_dtype(v):
        return ~np.isfinite(v)
    else:
        raise Exception(f"A numerical/arithmetic variable must '"
                        f"'be numeric dtype; has dtype {v.dtype}")
