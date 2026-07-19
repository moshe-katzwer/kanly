"""Shapley R² decomposition across regressors.

This module implements Owen (2000)-style Shapley value decomposition of the
coefficient of determination (R²).  Each regressor (formula **term**) receives
a Shapley value equal to its average marginal contribution to R² across all
possible orderings in which it could enter the model.

**Fast path** (:meth:`Shapley.shapley_value` → :meth:`_shapley_value_internal`):

After one full-model fit, the weighted normal equations are held in a
:class:`~kanly.regression.linear_models.linear_model_2_quadratic_form.QuadraticForm`
(``XtX``, ``Xty``, ``f0 = y'Wy``).  Each subset regression reuses sub-blocks
``XtX[indices][:, indices]`` and ``Xty[indices]``, solves for β in closed form,
and evaluates R² without refitting :class:`~kanly.regression.linear_models.model.SparseLinearModel`.
Column subsets are assembled via ``fit.model.exog_term_to_indices`` (term name
→ design-matrix column indices, e.g. ``'C(g)' → [3, 4, 5, 6]``).

Two modes are available:

* **Exact (default, ``sample=False``)** — enumerate every non-empty subset of
  the ``p`` sparse_terms, evaluate R² for each subset, and aggregate with closed-form
  Shapley weights involving factorials (``2^p − 1`` subset evaluations).
* **Permutation subsampling (``sample=k``)** — draw ``k`` random orderings of
  the sparse_terms; along each ordering, evaluate nested models that add one term at
  a time and accumulate marginal ΔR², averaged over permutations.  If
  ``k·p ≥ 2^p − 1``, exact enumeration is used instead.

Exact enumeration is feasible only for moderate ``p`` (roughly ``p ≲ 15``).
For larger ``p``, set ``sample`` to a positive integer (e.g. 100–1000).

References
----------
* Shapley, L. S. (1953). A value for n-person games. In *Contributions to the
  Theory of Games*, vol. II, 307–317. Princeton University Press.
* Owen, A. B. (2000). Multilinear regression via data matroids.
  *In* Skandinavisk Aktuarietidskrift, 83(2), 195–210. (Shapley decomposition
  of explained variance / R² in regression.)
* Grömping, U. (2007). Estimators of relative importance in linear regression
  based on variance decomposition. *The American Statistician*, 61(2), 139–147.
* Song, W.-Y., Di, C., & Karrison, T. G. (2011). On estimation of relative
  importance in linear regression. *Communications in Statistics—Theory and
  Methods*, 40(14), 2485–2502. (Permutation / order-based relative importance.)
* https://en.wikipedia.org/wiki/Shapley_value — cooperative game theory definition.
* https://en.wikipedia.org/wiki/Permutation_importance — related idea of
  marginal contributions along random feature orderings (ML literature).
"""

from __future__ import absolute_import, print_function

import time
import warnings
from collections.abc import Iterable
from itertools import combinations, chain

import numpy as np
import scipy
from pandas import DataFrame
from scipy.linalg import LinAlgWarning
from scipy.special import factorial
from tqdm import tqdm


class Shapley(object):
    """Shapley value R² decomposition across a set of regressors.

    The primary entry point is :meth:`shapley_value`, which fits the full model
    once (or use an existing :class:`~kanly.regression.linear_models.regression_results.SparseLinearRegressionResults`
    via :meth:`~kanly.regression.linear_models.regression_results.SparseLinearRegressionResults.shapley_value`)
    and evaluates subset R² via :meth:`_shapley_value_internal` and quadratic forms.

    **Exact mode** (``sample=False``): for every non-empty subset ``S`` of the
    ``p`` formula sparse_terms in ``exog_term_names``, evaluate R² for the model with
    intercept plus sparse_terms in ``S``, then weight by
    ``(|S|−1)! (p−|S|)! / p!`` (add if the term is in ``S``, subtract otherwise).
    Under exact arithmetic the per-term values sum to the full-model R²
    (efficiency property).

    **Permutation mode** (``sample=k > 0``): approximate Shapley values by
    averaging marginal ΔR² along ``k`` random permutations of term order.

    Attributes:
        FACT (list): Pre-computed factorial cache used in exact mode; extended
            when ``p`` exceeds the current cache length.

    Examples
    --------
    Allocate the R² of a 3-variable regression among its inputs:

    >>> import numpy as np, pandas as pd
    >>> from kanly.api import shapley_value
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> df = pd.DataFrame({'x1': rng.normal(size=n),
    ...                    'x2': rng.normal(size=n),
    ...                    'x3': rng.normal(size=n)})
    >>> df['y'] = (1.0 + 2.0*df['x1'] - 1.0*df['x2'] +
    ...            0.5*df['x3'] + rng.normal(size=n))
    >>> shapley_value('y ~ x1 + x2 + x3', df)               # doctest: +SKIP
    >>> shapley_value(('y', ['x1', 'x2', 'x3']), df)       # doctest: +SKIP
        shapley_value       pct
    x1       0.522     0.69...
    x2       0.158     0.21...
    x3       0.075     0.10...

    In exact mode, ``shapley_value`` sums to the model R²; ``pct`` is each
    value divided by that R².
    """

    # Pre-computed factorial cache; extended on demand in shapley_value.
    FACT = [factorial(i) for i in range(101)]

    @staticmethod
    def powerset(iterable):
        """Return the power set of *iterable*, excluding the empty set.

        Yields all non-empty subsets in order of increasing size.

        Args:
            iterable: Any finite iterable whose elements will form subsets.

        Returns:
            itertools.chain: An iterator over tuples representing every
                non-empty subset.
        """
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))

    @staticmethod
    def _fit_subset(rsquared_dict, indices, XtX, Xty, yty, wtss, num_regressions):
        """R² for an OLS/WLS subset defined by design-matrix column indices.

        Given the full-model normal equations ``XtX = X'WX`` and ``Xty = X'Wy``
        (from :class:`~kanly.regression.linear_models.linear_model_2_quadratic_form.QuadraticForm`),
        extracts the principal submatrix/subvector for the requested columns,
        solves ``β = XtX_sub^{-1} Xty_sub``, and evaluates weighted R²:

            R²_sub = 1 − SSR(β) / TSS

        where ``SSR(β) = yty + β' XtX_sub β − 2 β' Xty_sub`` with ``yty = y'Wy``
        from the **full** model (held fixed across subsets), and ``TSS`` is the
        full-model centred ``wsst`` passed as ``wtss``.

        Results are memoised in ``rsquared_dict`` keyed by the sorted index list
        so repeated subsets (across Shapley permutations or exact enumeration)
        are not recomputed.

        Args:
            rsquared_dict (dict): Cache mapping index-key strings to R² values.
            indices (array-like of int): 0-based column indices into the full
                design matrix (may span several columns per formula term).
            XtX (ndarray): Full ``p × p`` Gram matrix ``X'WX``.
            Xty (ndarray): Full length-``p`` vector ``X'Wy``.
            yty (float): Scalar ``y'Wy`` from the full-model quadratic form.
            wtss (float): Full-model weighted total sum of squares (``wsst``).

        Returns:
            float: Coefficient of determination R² for this regressor subset.
        """
        key = ','.join(sorted([str(f) for f in indices]))
        if key in rsquared_dict:
            rsquared_temp = rsquared_dict[key]
        else:
            # Restrict normal equations to the active columns for this subset.
            XtX_temp = XtX[indices][:,indices]
            Xty_temp = Xty[indices]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', LinAlgWarning)
                beta = scipy.linalg.solve(XtX_temp, Xty_temp)
            # SSR(β) = f0 + β'XtXβ − 2β'Xty (see QuadraticForm); f0 from full model.
            wssr_temp = beta.dot(XtX_temp).dot(beta) - 2*np.dot(beta, Xty_temp) + yty
            rsquared_temp = 1.0 - wssr_temp / wtss
            rsquared_dict[key] = rsquared_temp
            num_regressions += 1
        return rsquared_temp, num_regressions
    
    @staticmethod
    def _shapley_value_internal(fit, debug=False, sample=False, seed=0, return_full=False):
        """Shapley R² decomposition from an existing linear-model fit (fast path).

        Operates on **formula sparse_terms** (``fit.exog_term_names``), not individual
        design columns.  Multi-column sparse_terms (e.g. ``C(grp)``) are handled via
        ``fit.model.exog_term_to_indices``, which maps each term to the column
        indices it occupies in ``exog`` (e.g. ``'C(grp)' → [3, 4, 5, 6]``).

        Algorithm
        ---------
        1. Read ``wsst`` and build ``XtX``, ``Xty`` from the full-model
           :class:`~kanly.regression.linear_models.linear_model_2_quadratic_form.QuadraticForm`
           (one pass over data at fit time).
        2. **Exact mode**: for each non-empty subset ``S`` of sparse_terms, form column
           indices = ``Intercept`` columns + columns for sparse_terms in ``S``, evaluate
           R² via :meth:`_fit_subset`, apply Owen subset weights.
        3. **Permutation mode** (``sample = k``): ``k`` random term orderings;
           for ``j = 1, …, p``, fit intercept plus the first ``j`` sparse_terms in the
           permutation; credit term ``π(j−1)`` with ΔR² / ``k``.  The baseline
           before the first term is R² = 0 (centered R² with intercept only).

        The intercept is always included in subset models via the
        ``'Intercept'`` key in ``exog_term_to_indices``; it is not listed in
        ``exog_term_names``.

        Args:
            fit: Fitted :class:`~kanly.regression.linear_models.regression_results.SparseLinearRegressionResults`
                (or compatible) with ``wsst``, ``rsquared``, ``exog_term_names``,
                and ``model.exog_term_to_indices``.
            debug (bool): If ``True``, show a ``tqdm`` progress bar over subsets
                or permutations.
            sample (bool or int): Falsy for exact enumeration; positive int for
                permutation subsampling (auto-disabled when
                ``sample * p >= 2^p - 1``).
            seed (int): RNG seed for permutation shuffles.
            return_full (bool): If ``True``, return a dict with the results
                table and metadata; otherwise return the table only.

        Returns:
            pd.DataFrame or dict: Index = term names; columns ``shapley_value``
            and ``pct`` (``shapley_value / fit.rsquared``); sorted descending
            by ``shapley_value``.

        Raises:
            Exception: If ``fit.model`` is an IV model (no quadratic form).
            KeyError: If ``'Intercept'`` is missing from ``exog_term_to_indices``.

        See Also:
            :meth:`shapley_value`,
            :meth:`~kanly.regression.linear_models.regression_results.SparseLinearRegressionResults.shapley_value`.
        """

        t = time.time()

        # Full-model weighted TSS (denominator for every subset R²).
        wtss = fit.wsst
        full_model_rsquared = fit.rsquared

        # Normal equations for the full design; subset fits use sub-blocks only.
        _, quad_form = fit.model.get_quadratic_form_and_llf()
        XtX = quad_form.XtX()
        Xty = quad_form.Xty()
        yty = quad_form.f0

        exog_term_names = fit.exog_term_names.copy()
        term_2_indices = fit.model.exog_term_to_indices.copy()
        try:
            v = term_2_indices['Intercept']
            assert len(v) == 1 and v[0] == 0
        except:
            raise Exception("Can only do Shapley Value decomposition on linear models with"
                            " an `Intercept` in column 0 currently!")

        p = len(exog_term_names[1:])

        # Permutation subsampling only when k·p < 2^p−1 exact subset evaluations.
        use_sample = bool(sample) and sample * p < 2 ** p - 1
        if not use_sample:
            draws = Shapley.powerset(exog_term_names[1:])
        else:
            draws = range(sample)
        if debug:
            draws = tqdm(draws)

        random = np.random.RandomState(seed=seed)

        # Extend factorial table for exact-mode Shapley weights (|S|! sparse_terms).
        if p > len(Shapley.FACT):
            Shapley.FACT = [factorial(i) for i in range(p)]

        fact = Shapley.FACT

        rsquared_dict = dict()
        vals = dict.fromkeys(exog_term_names[1:], 0.0)

        num_regressions = 0
        for exog_term_subsample in draws:

            if use_sample:
                # --- Permutation / order-based Monte Carlo Shapley approximation ---
                random.shuffle(exog_names_subset := exog_term_names[1:].copy())
                rsquared_last = 0.0
                for j in range(1, p+1):
                    # Nested model: intercept + first j sparse_terms in this permutation.
                    indices = np.concatenate([term_2_indices[k] for k in ['Intercept'] + exog_names_subset[:j]])
                    rsquared_temp, num_regressions = Shapley._fit_subset(rsquared_dict, indices, XtX, Xty, yty, wtss, num_regressions)
                    vals[exog_names_subset[j-1]] += (rsquared_temp - rsquared_last) / sample
                    rsquared_last = rsquared_temp

            else:
                # --- Exact Owen / subset enumeration ---
                S = len(exog_term_subsample)
                indices = np.concatenate([term_2_indices[k] for k in ('Intercept',) + exog_term_subsample])
                rsquared_temp, num_regressions = Shapley._fit_subset(rsquared_dict, indices, XtX, Xty, yty, wtss, num_regressions)

                # Distribute R²(S) across sparse_terms via Shapley subset weights.
                for x in exog_term_names[1:]:
                    if x in exog_term_subsample:
                        vals[x] += rsquared_temp * fact[S - 1] * fact[p - S] / fact[p]
                    else:
                        vals[x] -= rsquared_temp * fact[S] * fact[p - S - 1] / fact[p]


        res3 = DataFrame(index=vals.keys(), data={'shapley_value': vals.values()})
        res3.sort_values(by='shapley_value', inplace=True, ascending=False)
        res3['pct'] = res3.shapley_value / full_model_rsquared
        
        if return_full:
            return {
                'shapley_values': res3, 
                'fit': fit,
                'full_model_rsquared': full_model_rsquared,
                'sample': sample,
                'seed': seed,
                'return_full': return_full,
                'use_sample': use_sample,
                'num_terms': p,
                'time_elapsed': time.time()-t,
                'num_regressions': num_regressions,
            }
        else:
            return res3
      
    @staticmethod
    def shapley_value(specification, data, debug=False, sample=False, seed=0, return_full=False):
        """Compute Shapley R² decomposition (quadratic-form fast path).

        Fits the full model once with :meth:`~kanly.regression.linear_models.model.SparseLinearModel.lm`,
        then decomposes R² via :meth:`_shapley_value_internal` without refitting
        each subset.

        Equivalent to fitting with ``lm`` then calling
        ``fit.shapley_value(...)``.  See
        :meth:`~kanly.regression.linear_models.regression_results.SparseLinearRegressionResults.shapley_value`
        for mode details and ``return_full``.

        Args:
            specification (str or iterable): Model specification, either:

                - A kanly **formula string** (e.g. ``'y ~ x + C(grp)'``,
                  ``'y ~ x1 + x2 $ w'`` for WLS), passed directly to ``lm``; or
                - An **iterable** of length 2 or 3 that is *not* a string,
                  unpacked as ``(endog_name, exog_term_names[, weights])`` and
                  converted to a formula.  ``exog_term_names`` must be an
                  iterable of RHS term names (e.g. ``['x', 'C(grp)']``); they
                  are joined with ``'+'`` on the RHS.  Optional third element
                  ``weights`` is the weight column name (``$`` syntax).

            data (pd.DataFrame): Dataset containing all variables referenced
                in the specification.
            debug (bool): If ``True``, show ``tqdm`` during decomposition.
            sample (bool or int, optional): Exact enumeration if falsy; else
                number of random permutations.
            seed (int): RNG seed when ``sample`` is used.
            return_full (bool): If ``True``, return a dict with the table and
                metadata (see :meth:`_shapley_value_internal`).

        Returns:
            pd.DataFrame or dict: ``shapley_value`` and ``pct`` by formula term.

        Raises:
            TypeError: If ``specification`` is not a string and not iterable.
            ValueError: If a non-string ``specification`` does not have length
                2 or 3.

        Examples
        --------
        Formula string:

        >>> import numpy as np, pandas as pd
        >>> from kanly.api import shapley_value
        >>> rng = np.random.default_rng(0)
        >>> n = 500
        >>> df = pd.DataFrame({'x1': rng.normal(size=n), 'x2': rng.normal(size=n),
        ...                    'x3': rng.normal(size=n)})
        >>> df['y'] = 1 + 2*df.x1 - df.x2 + 0.5*df.x3 + rng.normal(size=n)
        >>> shapley_value('y ~ x1 + x2 + x3', df)  # doctest: +SKIP

        Tuple specification (endog, sparse_terms[, weights]):

        >>> shapley_value(('y', ['x1', 'x2', 'x3']), df)  # doctest: +SKIP
        >>> shapley_value(('y', ['x1', 'x2'], 'w'), df)   # doctest: +SKIP
        """

        t = time.time()

        if isinstance(specification, str):
            formula = specification
        else:
            if not isinstance(specification, Iterable):
                raise TypeError(
                    "specification must be a formula string or an iterable "
                    "(endog_name, exog_term_names[, weights]); got "
                    f"{type(specification).__name__}")
            spec = tuple(specification)
            if len(spec) not in (2, 3):
                raise ValueError(
                    "specification iterable must have length 2 "
                    "(endog_name, exog_term_names) or 3 "
                    "(endog_name, exog_term_names, weights); got "
                    f"length {len(spec)}")
            endog_name, exog_term_names = spec[0], spec[1]
            weights = spec[2] if len(spec) == 3 else None
            if not isinstance(exog_term_names, Iterable) or isinstance(exog_term_names, (str, bytes)):
                raise TypeError(
                    "exog_term_names must be an iterable of term name strings; got "
                    f"{type(exog_term_names).__name__}")
            formula = f'{endog_name} ~ {"+".join(str(t) for t in exog_term_names)}'
            if weights is not None:
                formula += f'$ {weights}'

        from kanly.regression.linear_models.model import SparseLinearModel
        fit = SparseLinearModel.lm(formula, data, compute_cov=False)
        if fit.exog_names[0] != 'Intercept':
            raise Exception("Can only do Shapley Value decomposition on linear models with"
                            " an `Intercept` in column 0 currently!")

        result = Shapley._shapley_value_internal(
            fit, debug=debug, sample=sample, seed=seed, return_full=return_full)

        if return_full:
            result['time_elapsed'] = time.time() - t
            result['data'] = data
            result['formula'] = formula

        return result
    
    # OLD DEPRECATED FUNCTION
    # @staticmethod
    # def shapley_value_old(endog_name, exog_names, data, weights=None, debug=False, sample=False, seed=0):
    #     """Compute Shapley value R² decomposition for a set of regressors (legacy).

    #     **Superseded by** :meth:`shapley_value`, which evaluates subset R² from
    #     the full-model quadratic form instead of calling ``lm_fast`` per subset.
    #     Retained for reference; documents the statistical behaviour both paths
    #     aim to implement.

    #     **Exact enumeration** (``sample`` falsy, or auto-disabled when subsampling
    #     would not save work): fits ``2^p − 1`` OLS models (one per non-empty subset
    #     of ``p`` regressors), applies the subset Shapley weights, and aggregates.
    #     Values satisfy the **efficiency property**: they sum exactly to the
    #     full-model R² (checked with ``assert`` when ``sample`` is off).

    #     Subset weighting for variable *x* and subset *S* (exact mode only):

    #     - Add ``R²(S) · (|S|−1)! · (p−|S|)! / p!`` when *x* ∈ *S*.
    #     - Subtract ``R²(S) · |S|! · (p−|S|−1)! / p!`` when *x* ∉ *S*.

    #     **Permutation subsampling** (``sample`` a positive int): repeat ``sample``
    #     times:

    #     1. Draw a random permutation π of the ``p`` regressor names.
    #     2. For ``j = 0, …, p−1``, fit ``y ~`` first ``j+1`` regressors in π,
    #        record ΔR² = R²_j − R²_{j−1} (with R²_{−1} = 0).
    #     3. Credit regressor π(j) with ΔR² / ``sample``.

    #     This is the standard **random-order marginal contribution** estimator;
    #     it converges to the exact Shapley values as ``sample → ∞`` but is biased
    #     for finite ``sample``.  Cost is ``O(sample · p)`` regressions vs
    #     ``O(2^p)`` in exact mode.  If ``sample · (p − 1) ≥ 2^p − 1``, exact
    #     enumeration is used automatically.

    #     Args:
    #         endog_name (str): Name of the dependent variable column in ``data``.
    #         exog_names (str or list of str): Regressor names.  If a string is
    #             supplied it is parsed via
    #             ``SparseDataGetter.parse_to_variable_lists_helper``.
    #         data (pd.DataFrame): Dataset containing all variables.
    #         weights (str, optional): Column name in ``data`` for WLS weights.
    #             If ``None``, unweighted OLS.
    #         debug (bool): If ``True``, print progress and show a tqdm bar over
    #             subsets (exact) or permutations (sample).
    #         sample (bool or int, optional): If falsy, exact enumeration.  If a
    #             positive integer, number of random permutations for the
    #             approximate estimator.  Automatically set to exact mode when
    #             ``sample · (p − 1) ≥ 2^p − 1``.
    #         seed (int): RNG seed for permutation shuffles when ``sample`` is used.

    #     Returns:
    #         pd.DataFrame: Index = regressor names; columns:

    #         - ``shapley_value``: Shapley R² contribution (exact or approximate).
    #         - ``pct``: ``shapley_value / rsquared`` using the full-model R² from
    #           the last nested fit in the final permutation (sample mode) or the
    #           full subset (exact mode).

    #         Sorted descending by ``pct``.

    #     Raises:
    #         AssertionError: In exact mode only, if ``|sum(vals) − rsquared| > 1e-3``.

    #     Examples
    #     --------
    #     Exact decomposition (small ``p``):

    #     >>> import numpy as np, pandas as pd
    #     >>> from kanly.api import shapley_value
    #     >>> rng = np.random.default_rng(0)
    #     >>> n = 500
    #     >>> df = pd.DataFrame({'x1': rng.normal(size=n),
    #     ...                    'x2': rng.normal(size=n),
    #     ...                    'x3': rng.normal(size=n)})
    #     >>> df['y'] = (1.0 + 2.0*df['x1'] - 1.0*df['x2']
    #     ...            + 0.5*df['x3'] + rng.normal(size=n))
    #     >>> tab = shapley_value('y', ['x1', 'x2', 'x3'], df)   # doctest: +SKIP
    #     >>> tab.round(2)                                       # doctest: +SKIP

    #     Approximate decomposition with 200 random permutations (large ``p``):

    #     >>> tab = shapley_value('y', ['x1', 'x2', 'x3'], df, sample=200, seed=1)  # doctest: +SKIP

    #     Pass ``weights='w'`` for weighted regression.
    #     """
    #     from kanly.formula.data_getter import SparseDataGetter

    #     if isinstance(exog_names, str):
    #         _, exog_names = SparseDataGetter.parse_to_variable_lists_helper(exog_names)
    #     p = len(exog_names)

    #     # Extend factorial table for exact-mode Shapley weights (|S|! sparse_terms).
    #     if len(exog_names) > len(Shapley.FACT):
    #         Shapley.FACT = [factorial(i) for i in range(len(exog_names))]

    #     fact = Shapley.FACT

    #     if debug:
    #         print("Computing R^2...")

    #     num_regs = 2 ** p - 1  # exact mode: one regression per non-empty subset
    #     vals = {x: 0.0 for x in exog_names}
    #     rsquared = None  # full-model R²; used for pct column and exact-mode check
    #     rsquared_cache = dict()

    #     # Permutation subsampling only helps when k·(p−1) < 2^p−1 nested fits.
    #     use_sample = bool(sample)
    #     if use_sample and sample * (p - 1) >= num_regs:
    #         use_sample = False

    #     if use_sample:
    #         random = np.random.RandomState(seed)
    #         to_iter = enumerate(range(sample))  # k independent permutations
    #     else:
    #         to_iter = enumerate(Shapley.powerset(exog_names))  # all non-empty S

    #     if debug:
    #         to_iter = tqdm(to_iter, position=0, leave=True)

    #     for i, exog_names_subset in to_iter:

    #         if use_sample:
    #             # --- Permutation / order-based Monte Carlo Shapley approximation ---
    #             # Shuffle regressor order; nested models add one variable at a time.
    #             random.shuffle(exog_names_subset := exog_names.copy())
    #             val_last = 0.0
    #             for j in range(p):
    #                 subset = sorted(exog_names_subset[:j + 1])
    #                 temp_key = '#'.join(subset)
    #                 if temp_key in rsquared_cache:
    #                     val = rsquared_cache[temp_key]
    #                 else:
    #                     formula = (
    #                         endog_name + " ~ " + ' + '.join(subset)
    #                         + (" $ " + weights if weights is not None else '')
    #                     )
    #                     val = SparseLinearModel.lm_fast(formula, data).rsquared
    #                     rsquared_cache[temp_key] = val
    #                 # Marginal contribution of the j-th regressor in this permutation.
    #                 vals[exog_names_subset[j]] += (val - val_last) / sample
    #                 val_last = val
    #                 rsquared = val  # after j=p−1 this is the full-model R²

    #         else:
    #             # --- Exact Owen / subset enumeration ---
    #             if debug:
    #                 print('\n (%d/%d) Regressing on %s...' % (i + 1, num_regs, str(exog_names_subset)))
    #             formula = (
    #                 endog_name + " ~ " + ' + '.join(exog_names_subset)
    #                 + (" $ " + weights if weights is not None else '')
    #             )
    #             val = SparseLinearModel.lm_fast(formula, data).rsquared

    #             S = len(exog_names_subset)
    #             if S == p:
    #                 rsquared = val  # R² of the model with all regressors

    #             # Distribute R²(S) across variables via Shapley subset weights.
    #             for x in exog_names:
    #                 if x in exog_names_subset:
    #                     vals[x] += val * fact[S - 1] * fact[p - S] / fact[p]
    #                 else:
    #                     vals[x] -= val * fact[S] * fact[p - S - 1] / fact[p]

    #             if debug:
    #                 print(f'R^2 = {"%.4f" % val}')

    #     # Efficiency: exact Shapley values sum to full-model R² (not checked in sample mode).
    #     if not use_sample:
    #         assert np.abs(rsquared - sum(vals.values())) < 1e-3

    #     df_shapley = DataFrame(index=list(vals.keys()), data={
    #         'shapley_value': list(vals.values()),
    #         "pct": np.array(list(vals.values())) / rsquared,
    #     })
    #     df_shapley.sort_values(by='pct', inplace=True, ascending=False)

    #     if debug:
    #         print("\n\n")
    #         print(df_shapley.to_string())
    #         print("\n")

    #     # print(rsquared_cache)

    #     return df_shapley
