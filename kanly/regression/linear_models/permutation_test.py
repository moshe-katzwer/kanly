"""Randomisation inference for linear models via treatment-label permutation.

This module implements a non-parametric permutation test that approximates
the null distribution of any regression parameter by repeatedly shuffling
the treatment variable across observations (or within groups) and recording
the resulting parameter estimates.  The resulting distribution can be used
to compute empirical p-values without relying on asymptotic normality.

Typical usage::

    null_distribution = permutation_test('y ~ treatment + x', 'treatment', df,
                                         num_permutations=1000, seed=42)
    p_value = (null_distribution['treatment'].abs()
               >= abs(fit.params['treatment'])).mean()
"""

from __future__ import absolute_import, print_function

import numpy as np
import pandas as pd

from kanly import tqdm
from kanly.api import lm


def permutation_test(formula, treatment_key, df, num_permutations=100, seed=0,
                     estimator=None, debug=False, groups=None):
    """Run a permutation test to approximate the null distribution of regression parameters.

    The treatment column ``treatment_key`` in ``df`` is repeatedly shuffled
    (optionally within groups) and the model is re-estimated on each shuffled
    dataset.  The resulting collection of parameter vectors forms a Monte-Carlo
    approximation to the null distribution under the hypothesis that the
    treatment has no effect.

    .. note::
        This function **mutates** ``df`` in place during estimation: it adds a
        temporary column whose name encodes ``treatment_key``.  The column is
        deleted before the function returns.

    Args:
        formula (str): A patsy-style formula string, e.g. ``'y ~ treatment + x'``.
            Must reference ``treatment_key`` as one of its sparse_terms.
        treatment_key (str): The name of the treatment variable in ``df``.
            This column is permuted on each iteration.
        df (pd.DataFrame): The dataset.  Mutated temporarily (see note above).
        num_permutations (int): Number of permutation draws.  Defaults to 100.
        seed (int): Random seed for reproducibility.  Defaults to 0.
        estimator (callable, optional): Estimation function with the same
            signature as ``kanly.api.lm``.  Defaults to ``lm`` when ``None``.
        debug (bool): If ``True``, displays a tqdm progress bar.
        groups (str or list of str, optional): Column name(s) in ``df`` used
            to define permutation groups.  When supplied, the treatment label
            is permuted **within** each group rather than globally.

    Returns:
        pd.DataFrame: A DataFrame of shape ``(num_permutations, num_params)``
            where each row is the full parameter vector from one permuted fit.
            Column names match the original formula sparse_terms (``treatment_key``
            is restored to its original name).
    """

    if estimator is None:
        estimator = lm

    rand = np.random.RandomState(seed=seed)

    treatment_key_permute = f'______{treatment_key}____PERMUTATION___________'
    formula_permute = formula.replace(treatment_key, treatment_key_permute)

    if groups is None:
        group_dict = {None: df.index}
    else:
        group_dict = df.groupby(groups).groups

    params = []
    rng = range(num_permutations)
    if debug:
        rng = tqdm(range(num_permutations), position=0, leave=True)

    df[treatment_key_permute] = 0
    for _ in rng:
        for k, v in group_dict.items():
            df.loc[v, treatment_key_permute] = rand.permutation(df[treatment_key][v].values)
        params.append(estimator(formula_permute, df).params.copy())

    params = pd.DataFrame(params)
    params.columns = [c.replace(treatment_key_permute, treatment_key) for c in params.columns]

    del df[treatment_key_permute]

    return params
