"""
Converts a dictionary to a numpy array based on list of ordered param name strings.
If not a dictionary, returns original argument, possible `None`
"""

from __future__ import absolute_import, print_function

import numpy as np


def dict_2_array(param_dict, param_names, ignore_extra_keys=True, default_value=0.0, dtype=float, return_array=None):
    """Convert a parameter dictionary to a NumPy array ordered by ``param_names``.

    If ``param_dict`` is not a ``dict`` the value is returned unchanged, which
    allows callers to pass either a dict or an already-array-like object without
    branching.

    Args:
        param_dict: ``dict`` mapping parameter names to values, or any other
            value (returned as-is when not a dict).
        param_names: Ordered sequence of parameter names that defines the
            position of each value in the output array.
        ignore_extra_keys: When ``True``, silently drop keys in ``param_dict``
            that are not present in ``param_names``.
        default_value: Value used for parameters in ``param_names`` that are
            absent from ``param_dict``.  Defaults to ``0.0``.
        dtype: NumPy dtype of the output array.  Defaults to ``float``.
        return_array: Optional pre-allocated array to fill in-place.  A new
            array is created when ``None``.

    Returns:
        NumPy array of shape ``(len(param_names),)`` when input is a dict;
        the original ``param_dict`` value otherwise.
    """
    if isinstance(param_dict, dict):

        if return_array is None:
            return_array = np.array([default_value] * len(param_names), dtype=dtype)

        if ignore_extra_keys:
            param_dict = {k: v for k, v in param_dict.items() if k in param_names}

        for i, k in enumerate(param_names):
            if k in param_dict:
                return_array[i] = param_dict[k]

        return return_array

    else:
        return param_dict


def dict_2_list(param_dict, param_names, ignore_extra_keys=True, default_value=0.0):
    """Convert a parameter dictionary to a list ordered by ``param_names``.

    If ``param_dict`` is not a ``dict`` the value is returned unchanged.

    Args:
        param_dict: ``dict`` mapping parameter names to values, or any other
            value (returned as-is when not a dict).
        param_names: Ordered sequence of parameter names defining the output
            ordering.
        ignore_extra_keys: When ``True``, silently drop keys in ``param_dict``
            that are not present in ``param_names``.
        default_value: Value used for parameters in ``param_names`` that are
            absent from ``param_dict``.  Defaults to ``0.0``.

    Returns:
        A ``list`` of values in the order of ``param_names`` when input is a
        dict; the original ``param_dict`` value otherwise.
    """
    if isinstance(param_dict, dict):
        if ignore_extra_keys:
            param_dict = {k: v for k, v in param_dict.items() if k in param_names}
        return [param_dict.get(p, default_value) for p in param_names]
    else:
        return param_dict
