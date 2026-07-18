"""Convenience code for overwriting specific parameter indices in a vector of parameters"""
from __future__ import absolute_import, print_function

from collections.abc import Iterable

from copy import copy as copy_func


def overwrite_parameter_index(x: Iterable, overwrite_vals: dict, param_names: Iterable = None,
                              param_2_idx: dict = None, copy: bool = True):
    """Overwrite specific elements of a parameter vector by name.

    Given an ordered parameter vector ``x`` and a mapping from parameter
    names to new values, replaces the elements at the corresponding positions.
    Either ``param_names`` (used to build the name→index mapping on the fly)
    or a pre-built ``param_2_idx`` dict must be provided.

    Args:
        x: Mutable sequence (e.g. list or NumPy array) of parameter values.
        overwrite_vals: Mapping from parameter name strings to replacement
            values.
        param_names: Ordered iterable of parameter names used to construct
            the name→index mapping.  Ignored when ``param_2_idx`` is provided.
        param_2_idx: Pre-computed ``{name: index}`` dict.  When ``None``,
            built from ``param_names``.
        copy: When ``True`` (default), operates on a shallow copy of ``x``
            so the original is not mutated.

    Returns:
        Modified (copy of) ``x`` with the specified values overwritten.

    Raises:
        AssertionError: If both ``param_names`` and ``param_2_idx`` are
            ``None``.
    """

    assert not (param_names is None and param_2_idx is None)
    if param_2_idx is None:
        param_2_idx = dict(zip(param_names, range(len(param_names))))

    if copy:
        x = copy_func(x)

    for k, v in overwrite_vals.items():
        x[param_2_idx[k]] = v

    return x
