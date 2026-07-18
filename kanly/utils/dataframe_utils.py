from __future__ import absolute_import, print_function

import pandas as pd


def merge_dataframes(dataframes, on, how='outer'):
    """Sequentially merge a collection of DataFrames on a shared key column.

    Reduces the iterable of DataFrames by calling ``pd.merge`` pairwise, left to
    right, accumulating into a single result.  Requires at least one DataFrame.

    Args:
        dataframes: Iterable of ``pd.DataFrame`` objects to merge.  Must have
            length >= 1.
        on: Column name (or list of names) to merge on.  Must be present in
            every DataFrame.
        how: Type of merge to perform; one of ``'inner'``, ``'outer'``,
            ``'left'``, ``'right'``.  Defaults to ``'outer'``.

    Returns:
        A single merged ``pd.DataFrame``.

    Raises:
        Exception: If ``dataframes`` is empty (length < 1).
    """
    merged = None
    if len(dataframes) < 1:
        raise Exception("len of `dataframes` iterable must exceed 1!")

    for i, d in enumerate(dataframes):
        if i == 0:
            merged = d
        else:
            merged = pd.merge(merged, d, on=on, how=how)

    return merged


def iterate_through_sub_frames(dataframe, key, key_subset=None):
    """Use this to iterate through indices along a key to slice dataframes"""
    if key_subset is None:
        key_subset = sorted(list(dataframe[key].unique()))
    return (
        (key, dataframe[key] == key) for key in key_subset
    )
