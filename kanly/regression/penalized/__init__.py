from __future__ import absolute_import, print_function

"""Root helpers for sparse penalized linear-model estimation."""

import numpy as np


class SparseElasticNetException(Exception):
    """Exception raised for invalid elastic-net configuration or penalty inputs.

    Used by package-level validation helpers to distinguish penalized-model input
    errors from generic Python exceptions."""
    pass


def _check_penalties(alpha, l1_ratio):
    """Validate elastic-net penalty parameters before fitting.

    Args:
        alpha: Scalar or array-like non-negative penalty strength(s).
        l1_ratio: Scalar or array-like elastic-net mixing value(s) in ``[0, 1]``.

    Raises:
        SparseElasticNetException: If any ``alpha`` is negative or any
            ``l1_ratio`` falls outside ``[0, 1]``."""
    if np.any(np.asarray(alpha) < 0):
        raise SparseElasticNetException("alpha must be non-negative!")
    if np.any(np.asarray(l1_ratio) < 0) or np.any(np.asarray(l1_ratio) > 1):
        raise SparseElasticNetException("l1_ratio must be in [0, 1]")
