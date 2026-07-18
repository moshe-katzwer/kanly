# kanly/bootstrap/bootstrap.py — Bayesian and classical bootstrap weights, block resampling,
# Ray-parallel parameter draws for regression fits, and joint covariance across bootstrapped models.
"""
Bootstrap resampling utilities for regression covariance estimation.

This module implements two flavors of the bootstrap for use with regression
fit objects from the `kanly` package:

  * **Bayesian bootstrap** -- weights are drawn from a Dirichlet(alpha, ..., alpha)
    distribution and rescaled so they sum to `nobs`. This is a smooth alternative
    to the classical bootstrap and never produces zero-weight observations
    (when alpha > 0).
  * **Classical bootstrap** -- weights are integer counts produced by sampling
    observations with replacement (equivalently, a multinomial draw).

Both methods optionally support **block (cluster) bootstrapping** via the
`blocks` / `groups` argument, where the resampling unit is a group of
observations rather than individual rows. This is the appropriate approach
when residuals are correlated within groups (e.g. panel data, repeated
measures).

The main entry points are:

  * `do_bootstrap2` -- runs the bootstrap end-to-end and attaches the resulting
    covariance matrix and bootstrapped parameter draws to one or more regression
    fit objects. Supports parallel execution across multiple processes via Ray.
  * `bootstrap_entire_procedure` -- a more general wrapper that bootstraps an
    arbitrary user-supplied function (need not be a regression). Sequential
    only; returns the raw draws rather than mutating a fit object.
  * `get_joint_bootstrapped_distribution` -- given a collection of already-
    bootstrapped fits that share the same resampling scheme, returns the
    joint covariance of their stacked parameter vectors.

Helpers `get_bayesian_bootstrap_weights` and `get_classical_bootstrap_weights2`
expose the underlying weight generators directly, which is useful when a caller
wants to drive its own loop.

References (see also ``README.md`` in this package): Efron (1979) on the classical
bootstrap; Rubin (1981) on the Bayesian bootstrap; Efron & Tibshirani (1993) textbook.
"""

from __future__ import absolute_import, print_function

import time

import numpy as np
import pandas as pd
# import ray
# from ray.experimental import tqdm_ray
from pandas import Series, DataFrame

from scipy.sparse import isspmatrix
from tqdm import tqdm

from kanly.regression.cov_types import BOOTSTRAP

DEFAULT_BOOTSTRAP_N_SAMPLES = 100
DEFAULT_TEST_LEVEL = .05

# Method tags for ``do_bootstrap2`` (compared with ``.upper()``).
BAYESIAN = 'BAYESIAN'
CLASSICAL = 'CLASSICAL'
BOOTSTRAP_METHODS = [BAYESIAN, CLASSICAL]
DEFAULT_BB_METHOD = BAYESIAN
DEFAULT_BB_SEED = 0
DEFAULT_BB_ALPHA = 1.0
DEFAULT_BB_MAX_PROCESSES = 1


def blocks_to_ind(blocks):
    """
    Convert an array of arbitrary block (group) labels into a 0-based integer
    index per row, along with the count of unique blocks.

    Accepts dense arrays, pandas Series, and scipy sparse matrices. The label
    ordering of the returned indices follows the order in which each unique
    value first appears in `blocks`.

    Parameters
    ----------
    blocks : array-like or scipy sparse matrix
        Per-observation block / group labels of any hashable dtype.

    Returns
    -------
    blocks_indices : pandas.Series of int
        For each row of `blocks`, the integer code (0 .. num_unique - 1)
        identifying its block.
    num_unique : int
        Number of distinct blocks.
    """
    # Densify sparse input before flattening to a 1-D Series.
    if isspmatrix(blocks):
        blocks = blocks.toarray()
    blocks = Series(np.asarray(blocks).ravel())
    # Build a label -> integer-code mapping based on first-seen order.
    unique = blocks.unique()
    num_unique = len(unique)
    unique_map = dict(zip(unique, range(num_unique)))
    blocks_indices = blocks.map(unique_map)
    return blocks_indices, num_unique


def get_bayesian_bootstrap_weights(nobs, blocks=None, n_samples=DEFAULT_BOOTSTRAP_N_SAMPLES, seed=DEFAULT_BB_SEED,
                                   alpha=DEFAULT_BB_ALPHA):
    """
    Generate a lazy stream of Bayesian-bootstrap weight vectors.

    Each draw is a Dirichlet(alpha, ..., alpha) sample, rescaled so the weights
    sum to the population size (`nobs` if unblocked, `num_unique` if blocked).
    With blocks, every observation in a block receives that block's shared
    weight, which yields a cluster-level bootstrap.

    Parameters
    ----------
    nobs : int
        Number of observations (only used when `blocks is None`).
    blocks : array-like or None
        Per-observation cluster labels. If supplied, weights are drawn at the
        cluster level and broadcast to rows.
    n_samples : int
        Number of bootstrap weight vectors to produce.
    seed : int
        Seed for the internal `numpy.random.RandomState`.
    alpha : float
        Concentration parameter of the Dirichlet. `alpha=1.0` is the standard
        Bayesian bootstrap; smaller values concentrate weight on fewer
        observations, larger values smooth toward uniform.

    Returns
    -------
    weights_samples : generator of numpy.ndarray
        Lazy generator producing `n_samples` weight vectors of length `nobs`.
    num_unique : int or None
        Number of distinct blocks, or `None` when `blocks is None`.
    """
    # Use a local RandomState so callers' global numpy state is not disturbed.
    rand = np.random.RandomState(seed)

    if blocks is None:
        # Unblocked case: draw a length-`nobs` Dirichlet per sample.
        return (rand.dirichlet(alpha * np.ones(nobs), size=1)[0] * nobs for _ in range(n_samples)), None

    else:
        # Blocked case: draw weights at the cluster level, then expand to rows
        # by indexing with `blocks_indices` so every row in a block shares the
        # block's weight.
        blocks_indices, num_unique = blocks_to_ind(blocks)

        weights_samples = (rand.dirichlet(alpha * np.ones(num_unique), size=1)[0][blocks_indices] * num_unique
                           for _ in range(n_samples))
        return weights_samples, num_unique


def get_classical_bootstrap_weights2(nobs, blocks=None, n_samples=DEFAULT_BOOTSTRAP_N_SAMPLES, seed=DEFAULT_BB_SEED):
    """
    Generate a lazy stream of classical-bootstrap weight vectors.

    The classical (Efron) bootstrap draws `nobs` rows with replacement; the
    resulting weight on each row is the number of times it was selected. With
    blocks, sampling is performed at the cluster level and the chosen cluster
    indices are broadcast back to row weights.

    Parameters
    ----------
    nobs : int
        Number of observations.
    blocks : array-like or None
        Per-observation cluster labels for block bootstrap. If `None`, rows are
        resampled directly.
    n_samples : int
        Number of bootstrap weight vectors to produce.
    seed : int
        Seed for the internal `numpy.random.RandomState`.

    Returns
    -------
    weights_samples : generator of numpy.ndarray
        Lazy generator producing `n_samples` weight vectors.
    num_unique : int or None
        Number of distinct blocks, or `None` when `blocks is None`.
    """
    rand = np.random.RandomState(seed)

    if blocks is None:
        # Bincount of a with-replacement sample gives integer multiplicities
        # which are the bootstrap weights.
        return (np.bincount(rand.choice(np.arange(nobs), nobs, replace=True), minlength=nobs).astype(float)
                for _ in range(n_samples)), None

    else:
        blocks_indices, num_unique = blocks_to_ind(blocks)
        # Resample cluster IDs with replacement and broadcast to rows.
        weights_samples = (rand.choice(range(num_unique), num_unique, replace=True)[blocks_indices]
                           for _ in range(n_samples))
        return weights_samples, num_unique


def get_bootstrap_weights2(bootstrap_weights, weights=None):
    """
    Combine bootstrap weights with optional user-supplied observation weights.

    If `weights` is provided, the bootstrap weights are multiplied by it
    in-place and the (mutated) array is returned; otherwise the bootstrap
    weights are returned unchanged.

    Parameters
    ----------
    bootstrap_weights : numpy.ndarray
        Weights produced by one of the bootstrap weight generators.
    weights : numpy.ndarray or None
        Per-observation analytic / frequency weights, if any.

    Returns
    -------
    numpy.ndarray
        Element-wise product `bootstrap_weights * weights`, or
        `bootstrap_weights` itself when `weights is None`.
    """
    if weights is not None:
        bootstrap_weights *= weights
    return bootstrap_weights


def _get_fits(fits):
    """
    Normalize the `fits` argument to a `dict` of fit objects.

    Callers may pass a single fit, a list/tuple of fits, or a dict already.
    Higher-dimensional arrays of fits are rejected.

    Parameters
    ----------
    fits : fit object, sequence of fit objects, or dict

    Returns
    -------
    dict
        A dict whose values are fit objects. Keys are positional indices for
        a sequence input or the single key `'_'` for a scalar input.
    """
    if np.shape(fits) != tuple():
        # Sequence input: must be 1-D; key by positional index.
        if np.ndim(fits) > 1:
            raise Exception
        else:
            fits = {j: f for j, f in enumerate(fits)}
    elif not isinstance(fits, dict):
        # Scalar fit: wrap in a singleton dict.
        fits = {'_': fits}
    return fits


def do_bootstrap2(nobs, fits, param_estimation_func, groups=None, n_samples=DEFAULT_BOOTSTRAP_N_SAMPLES,
                  max_processes=DEFAULT_BB_MAX_PROCESSES,
                  seed=DEFAULT_BB_SEED, debug=False, use_correction=True, test_level=DEFAULT_TEST_LEVEL,
                  group_name=None, method=DEFAULT_BB_METHOD, alpha=DEFAULT_BB_ALPHA, exog_names=None):
    """
    Run the bootstrap end-to-end and attach the results to one or more fit objects.

    For each of `n_samples` bootstrap weight vectors, `param_estimation_func`
    is invoked to produce a parameter vector; the empirical covariance of the
    resulting draws is then assigned to each fit via `set_cov_params`, and the
    raw draws themselves via `set_bootstrapped_params`. When `max_processes > 1`,
    the sampling loop is split across Ray actors.

    Parameters
    ----------
    nobs : int
        Number of observations in the original sample.
    fits : fit, sequence of fits, or dict of fits
        Regression fit objects to mutate. Multi-outcome fits are supported: the
        parameter estimator may return one parameter vector per outcome, in
        which case one fit per outcome is expected and they are paired in order.
    param_estimation_func : callable
        Function mapping a single weights vector to a parameter vector (or a
        2-D array of shape (n_params, n_outcomes) for multi-outcome fits).
        May return `None` for failed draws, which are dropped.
    groups : array-like or None
        Per-observation cluster labels for block bootstrap.
    n_samples : int
        Total number of bootstrap repetitions (across all processes).
    max_processes : int
        Number of parallel Ray workers. Must be a positive int. `1` runs
        sequentially in-process.
    seed : int
        Base seed; workers use `seed + k` for `k = 0 .. max_processes - 1`.
    debug : bool
        If True, enables tqdm progress bars and verbose Ray driver logs.
    use_correction : bool
        If True, apply the small-sample correction `n / (n - 1)` to the
        bootstrapped covariance.
    test_level : float
        Significance level passed through to the fit when setting covariance.
    group_name : str or None
        Display name for the clustering variable in summary strings. Defaults
        to `'<groups>'`.
    method : str
        Either `'BAYESIAN'` or `'CLASSICAL'` (case-insensitive).
    alpha : float
        Dirichlet concentration parameter for the Bayesian method (ignored
        otherwise).
    exog_names : sequence of str or None
        If `param_estimation_func` returns a pandas Series, these names are
        used to align/reindex the parameter vector across draws, filling
        missing entries with NaN.

    Returns
    -------
    None
        Side effect: each fit in `fits` has its covariance, bootstrapped
        parameters, and `cov_elapsed` attribute set.

    Raises
    ------
    Exception
        If `method` is not one of `BAYESIAN` or `CLASSICAL`.
    """
    _t = time.time()

    fits = _get_fits(fits)

    if group_name is None:
        group_name = '<groups>'

    assert max_processes >= 1 and isinstance(max_processes, int)

    # Partition the total sample budget across workers, putting any remainder
    # on the first worker, and assign a distinct seed to each.
    n_samples_list = [n_samples // max_processes] * max_processes
    n_samples_list[0] += n_samples % max_processes
    seed_list = [seed + k for k in range(max_processes)]

    def sub_func(seed_, n_samples_, chunk_number):
        """Worker: draw ``n_samples_`` weight vectors with ``seed_``; return param draws and ``num_unique``."""

        if method.upper() == CLASSICAL:
            sample_weights, num_unique = get_classical_bootstrap_weights2(nobs, groups, n_samples_, seed_)
        elif method.upper() == BAYESIAN:
            sample_weights, num_unique = get_bayesian_bootstrap_weights(nobs, groups, n_samples_, seed_, alpha=alpha)
        else:
            raise Exception(f"Bootstrap method {method} invalid!")

        params_boot_list_ = get_bootstrapped_param_draws2(
            sample_weights, param_estimation_func, n_samples_,
            debug=debug, exog_names=exog_names, return_var_covar=False,
            chunk_number=chunk_number)

        return {'params': params_boot_list_, 'num_unique': num_unique}

    if max_processes > 1:

        import ray

        # Best-effort shutdown of any pre-existing Ray instance before we
        # start a fresh one for this bootstrap call.
        try:
            ray.shutdown()
        except:
            pass

        ray.init(num_cpus=max_processes, log_to_driver=debug)

        sub_func_remote = ray.remote(sub_func)

        # Fan out one remote task per worker and block on all of them.
        param_draw_list = ray.get(
            [sub_func_remote.remote(*arg)
             for arg in zip(seed_list, n_samples_list, range(max_processes))])

        time.sleep(.01)
        if debug:
            print('\nJoining the data from the parallel jobs...')

        ray.shutdown()
        time.sleep(.01)

    else:
        # Sequential path: same per-worker function, just called inline.
        param_draw_list = [sub_func(*_arg)
                           for _arg in zip(seed_list, n_samples_list, range(len(seed_list)))]

    # All workers should agree on the number of unique blocks (or all be None).
    # Collapse to a single value via set-pop.
    num_unique = {c['num_unique'] for c in param_draw_list}
    num_unique = num_unique.pop()
    param_draw_list = [c['params'] for c in param_draw_list]

    # Stitch the per-worker chunks together for each outcome, then compute
    # the empirical covariance across all bootstrap draws.
    n_outcomes = len(param_draw_list[0])
    params_boot_list = [np.vstack([param_draw_list[j][i] for j in range(max_processes)]) for i in range(n_outcomes)]
    var_covar_list = [np.cov(P, rowvar=False) for P in params_boot_list]
    # When there's a single parameter, np.cov returns a 0-d array; reshape to
    # 1x1 so downstream consumers always see a 2-D matrix.
    var_covar_list = [v if np.shape(v) != tuple() else v.reshape(1, 1) for v in var_covar_list]

    for fit, params_boot, var_covar in zip(fits.values(), params_boot_list, var_covar_list):

        # For clustered bootstrap the t-distribution df is (num_clusters - 1);
        # otherwise fall back to the model's residual df.
        df_t_dist = num_unique - 1 if groups is not None else fit.df_resid

        if use_correction:
            var_covar *= float(n_samples) / (n_samples - 1)

        cov_kwds = {'seed': seed, 'n_samples': len(params_boot_list),
                    'method': method, 'alpha': alpha,
                    'max_processes': max_processes}

        if groups is not None:
            cov_kwds['groups'] = group_name

        fit.set_cov_params(
            var_covar, cov_kwds=cov_kwds, cov_type=BOOTSTRAP, test_level=test_level, df_t_dist=df_t_dist)

        # Build the human-readable summary string, branching on clustered vs
        # unclustered and on bootstrap method.
        if groups is None:
            if method.upper() == CLASSICAL:
                bootstrap_string = f'Did {len(params_boot)} classical bootstrap repetitions.'
            else:
                bootstrap_string = f'Did {len(params_boot)} Bayesian bootstrap repetitions, alpha={"%.3f" % alpha}.'
        else:
            if method.upper() == CLASSICAL:
                bootstrap_string = f'Did {len(params_boot)} classical bootstrap repetitions, blocked on {group_name}.'
            else:
                bootstrap_string = f"Did {len(params_boot)} Bayesian bootstrap repetitions, alpha={'%.3f' % alpha}" \
                                   f", blocked on '{group_name}'."

        fit.set_bootstrapped_params(params_boot, bootstrap_string)

    cov_elapsed = time.time() - _t
    for fit in fits.values():
        fit.cov_elapsed = cov_elapsed


def get_bootstrapped_param_draws2(samples, func, n_samples, debug=False, exog_names=None, return_var_covar=True,
                                  chunk_number=0):
    """
    Apply a parameter-estimation function across a stream of weight vectors.

    Iterates over `samples` (typically a generator from one of the bootstrap
    weight functions), invokes `func` on each, and stacks the returned
    parameter vectors. Optionally also returns the empirical covariance of
    the stacked draws.

    Parameters
    ----------
    samples : iterable of numpy.ndarray
        Bootstrap weight vectors.
    func : callable
        Maps one weight vector to a parameter vector. May return `None` to
        signal a failed draw, in which case that draw is silently skipped.
    n_samples : int
        Expected total count; used only to set the tqdm progress-bar length
        when `debug` is True.
    debug : bool
        If True, display a Ray-aware tqdm progress bar for this chunk.
    exog_names : sequence of str or None
        When `func` returns a pandas Series, reorder/realign to this column
        list; missing names become NaN.
    return_var_covar : bool
        If True, also compute and return the empirical covariance of the
        draws (one matrix per outcome).
    chunk_number : int
        Worker / chunk identifier used to label the progress bar.

    Returns
    -------
    params_boot : list of numpy.ndarray
        One element per outcome; each is a (n_kept_samples, n_params) array.
    var_covar_list : list of numpy.ndarray
        Only returned when `return_var_covar` is True. One covariance matrix
        per outcome.
    """
    params_boot = []
    # Wrap `samples` in a progress bar only when debugging; the bar is
    # Ray-aware so it streams from workers to the driver.
    if debug:
        from ray.experimental import tqdm_ray
        pbar = tqdm_ray.tqdm(samples, total=n_samples)
        pbar.set_description(f'Bootstrap sampling {chunk_number}')
    else:
        pbar = samples
    for sample in pbar: #tqdm(samples, desc='Bootstrap', disable=not debug, total=n_samples):
        params_new = func(sample)
        if params_new is not None:
            # Series outputs are realigned to `exog_names` (NaN for missing),
            # so that all draws end up the same length even if the estimator
            # drops collinear columns inconsistently across samples.
            if isinstance(params_new, pd.Series):
                if exog_names is not None:
                    params_new = np.array([params_new[x] if x in params_new.index else np.nan
                                           for x in exog_names])
            params_boot.append(params_new)
    params_boot = np.array(params_boot)
    if debug:
        pbar.set_description(f'Bootstrap sampling {chunk_number}')

    # If the estimator returns a 2-D parameter matrix per draw (multi-outcome),
    # the stacked array is 3-D; split it into a list of 2-D per-outcome arrays.
    is_multi_outcome = np.ndim(params_boot) == 3
    if is_multi_outcome:
        params_boot = [params_boot[:, :, k] for k in range(params_boot.shape[2])]
    else:
        params_boot = [params_boot]

    if return_var_covar:

        var_covar_list = []
        for p in params_boot:
            var_covar = np.cov(p.T)
            # Single-parameter case: np.cov collapses to a scalar; restore the
            # expected 2-D shape so consumers don't need to special-case it.
            if np.product(np.shape(var_covar)) <= 1:
                var_covar = np.array(var_covar).reshape((p.shape[1], p.shape[1]))
            var_covar_list.append(var_covar)

        return params_boot, var_covar_list

    else:
        # ``do_bootstrap2`` path: covariance is assembled across workers later, not here.
        return params_boot


def get_joint_bootstrapped_distribution(fits, return_dataframe=True, small_sample_correction=True):
    """
    Build the joint covariance across the parameters of several bootstrapped fits.

    All fits must have been bootstrapped with the same scheme (same `cov_kwds`)
    and on the same data (same `nobs`); otherwise their bootstrap draws are not
    comparable row-by-row and the joint covariance is meaningless.

    Parameters
    ----------
    fits : fit, sequence of fits, or dict of fits
        Fit objects that have already been bootstrapped (i.e. have
        `cov_type == 'bootstrap'` and a `bootstrapped_params` attribute).
    return_dataframe : bool
        If True, wrap the covariance in a pandas DataFrame with labeled rows
        and columns of the form `'{position}_{endog_name}_{exog_name}'`.
    small_sample_correction : bool
        If True, apply the `n / (n - 1)` correction to the covariance.

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        Joint covariance matrix of the stacked parameter vectors across all
        fits.

    Raises
    ------
    Exception
        If any fit's `cov_type` is not `'BOOTSTRAP'`, or if `cov_kwds` /
        `nobs` differ across fits.

    Examples
    --------
    Bootstrap two OLS fits jointly so you can test linear combinations of
    their parameters via the bootstrap distribution:

    >>> import numpy as np, pandas as pd
    >>> from kanly.api import lm, get_joint_bootstrapped_distribution
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({
    ...     'x': rng.normal(size=300),
    ...     'g': rng.integers(0, 5, 300),
    ... })
    >>> df['y1'] = 1.0 + 0.5*df['x'] + rng.normal(size=300)
    >>> df['y2'] = 0.5 - 0.3*df['x'] + rng.normal(size=300)
    >>> common_kwds = dict(cov_type='bootstrap',
    ...                    cov_kwds={'n_samples': 500, 'method': 'bayesian',
    ...                              'seed': 0})
    >>> fit1 = lm('y1 ~ x', df, **common_kwds)            # doctest: +SKIP
    >>> fit2 = lm('y2 ~ x', df, **common_kwds)            # doctest: +SKIP
    >>> joint = get_joint_bootstrapped_distribution(       # doctest: +SKIP
    ...     [fit1, fit2])
    >>> joint.round(3)                                     # doctest: +SKIP
                       0_y1_Intercept  0_y1_x  1_y2_Intercept  1_y2_x
    0_y1_Intercept              0.003   0.000          -0.000   0.000
    ...
    """
    fits = _get_fits(fits)
    # Sanity checks: every fit must have been bootstrapped, with matching
    # scheme and sample size, so that the column-wise stack is meaningful.
    if np.any([f.cov_type.lower() != 'bootstrap' for f in fits.values()]):
        raise Exception("All covariance types must be 'BOOTSTRAP' in `fits`!")
    f0 = list(fits.values())[0]
    if np.any([f.cov_kwds != f0.cov_kwds for f in fits.values()]):
        raise Exception("All cov_kwds dicts must be equal across the fits in `fits`!")
    if np.any([f.nobs != f0.nobs for f in fits.values()]):
        raise Exception("All `nobs` must be equal across the fits in `fits`!")

    # Stack draws side-by-side: rows are bootstrap repetitions, columns are
    # parameters from all fits concatenated.
    bootstrapped_params = np.hstack([f.bootstrapped_params for f in fits.values()])
    var = np.cov(bootstrapped_params.T)

    if small_sample_correction:
        var *= bootstrapped_params.shape[0] / (bootstrapped_params.shape[0] - 1)

    if return_dataframe:
        # Compose informative column labels: position index, response name,
        # then the regressor name -- unique even if fits share exog names.
        cols = [f'{j}_{f.endog_name}_{x}'
                for j, f in enumerate(fits.values())
                for x in f.exog_names]
        return DataFrame(var, columns=cols, index=cols)
    else:
        return var


def bootstrap_entire_procedure(func, nobs, blocks=None,
                               n_samples=DEFAULT_BOOTSTRAP_N_SAMPLES, seed=DEFAULT_BB_SEED,
                               alpha=DEFAULT_BB_ALPHA, return_type='list', debug=False):
    """
    Bootstrap an arbitrary user-supplied procedure using Bayesian weights.

    Unlike `do_bootstrap2`, this is not tied to regression fit objects: it
    simply repeatedly applies `func` to weight vectors and returns the raw
    outputs along with the configuration used. Always sequential.

    Parameters
    ----------
    func : callable
        Maps one nonnegative weight vector (length ``nobs``) to an arbitrary
        return value recorded per draw (scalar, array, Series, etc.).
    nobs : int
        Number of observations.
    blocks : array-like or None
        Per-observation cluster labels for block bootstrap.
    n_samples : int
        Number of bootstrap repetitions.
    seed : int
        Random seed.
    alpha : float
        Dirichlet concentration parameter.
    return_type : {'list', 'dataframe'}
        Shape of the returned `result`. `'dataframe'` requires that each
        `func` call returns a mapping-like / Series-like object.
    debug : bool
        If True, wrap the iteration in a tqdm progress bar.

    Returns
    -------
    dict
        Dict with keys:
          * `'result'`     -- list (or DataFrame) of `func` outputs.
          * `'nobs'`       -- echoed input.
          * `'blocks'`     -- echoed input.
          * `'options'`    -- dict of resampling-config knobs.

    Raises
    ------
    Exception
        If `return_type` is not `'list'` or `'dataframe'`.
    """

    weights_generator, n_unique = get_bayesian_bootstrap_weights(nobs, blocks, n_samples, seed, alpha)
    if debug:
        # Wrap the lazy generator in tqdm so we can show progress without
        # materializing all weights up front.
        weights_generator = tqdm(weights_generator, total=n_samples)
        weights_generator.set_description('Bootstrapping procedure: ')
    result = [func(w) for w in weights_generator]

    if return_type.lower() == 'list':
        pass
    elif return_type.lower() == 'dataframe':
        result = DataFrame(result)
    else:
        raise Exception("`return_type` must be 'list' or 'dataframe'!")

    return {
        'result': result,
        'nobs': nobs,
        'blocks': blocks,
        'options': {
            'n_samples': n_samples,
            'seed': seed,
            'alpha': alpha,
            'return_type': return_type
        }
    }

# from __future__ import absolute_import, print_function
#
# import time
#
# import numpy as np
# import pandas as pd
# from pandas import Series, DataFrame
# from scipy.sparse import isspmatrix
# from tqdm import tqdm
#
# from kanly.regression.cov_types import BOOTSTRAP
#
# DEFAULT_BOOTSTRAP_N_SAMPLES = 100
# DEFAULT_TEST_LEVEL = .05
#
# BAYESIAN = 'BAYESIAN'
# CLASSICAL = 'CLASSICAL'
# BOOTSTRAP_METHODS = [BAYESIAN, CLASSICAL]
# DEFAULT_BB_METHOD = BAYESIAN
# DEFAULT_SEED = 0
# DEFAULT_BB_ALPHA = 1.0
# DEFAULT_BB_MAX_PROCESSES = 4
#
#
# def blocks_to_ind(blocks):
#     if isspmatrix(blocks):
#         blocks = blocks.toarray()
#     blocks = Series(np.asarray(blocks).ravel())
#     unique = blocks.unique()
#     num_unique = len(unique)
#     unique_map = dict(zip(unique, range(num_unique)))
#     blocks_indices = blocks.map(unique_map)
#     return blocks_indices, num_unique
#
#
# def blocks_to_ind(blocks):
#     if isspmatrix(blocks):
#         blocks = blocks.toarray()
#     blocks = Series(np.asarray(blocks).ravel())
#     unique = blocks.unique()
#     num_unique = len(unique)
#     unique_map = dict(zip(unique, range(num_unique)))
#     blocks_indices = blocks.map(unique_map)
#     return blocks_indices, num_unique
#
#
# def get_bayesian_bootstrap_weights(nobs, blocks=None, n_samples=DEFAULT_BOOTSTRAP_N_SAMPLES, seed=DEFAULT_SEED,
#                                    max_processes=DEFAULT_BB_MAX_PROCESSES, alpha=DEFAULT_BB_ALPHA):
#     rands = [np.random.RandomState(seed + j) for j in range(max_processes)]
#     n_samples_list = [n_samples // max_processes] * max_processes
#     n_samples_list[0] += n_samples % max_processes
#
#     if blocks is None:
#         return [(r.dirichlet(alpha * np.ones(nobs), size=1)[0] * nobs for _ in range(n))
#                 for r, n in zip(rands, n_samples_list)], None
#
#     else:
#         blocks_indices, num_unique = blocks_to_ind(blocks)
#
#         weights_samples = [(r.dirichlet(alpha * np.ones(num_unique), size=1)[0][blocks_indices] * num_unique
#                            for _ in range(n_samples)) for r, n in zip(rands, n_samples_list)]
#
#         return weights_samples, num_unique
#
#
# def get_classical_bootstrap_weights2(nobs, blocks=None, n_samples=DEFAULT_BOOTSTRAP_N_SAMPLES, seed=DEFAULT_SEED):
#
#     rand = np.random.RandomState(seed)
#
#     if blocks is None:
#         return (np.bincount(rand.choice(np.arange(nobs), nobs, replace=True), minlength=nobs).astype(float)
#                 for _ in range(n_samples)), None
#
#     else:
#         blocks_indices, num_unique = blocks_to_ind(blocks)
#         weights_samples = (rand.choice(range(num_unique), num_unique, replace=True)[blocks_indices]
#                            for _ in range(n_samples))
#         return weights_samples, num_unique
#
#
# def get_bootstrap_weights2(bootstrap_weights, weights=None):
#     if weights is not None:
#         bootstrap_weights *= weights
#     return bootstrap_weights
#
#
# def _get_fits(fits):
#     if np.shape(fits) != tuple():
#         if np.ndim(fits) > 1:
#             raise Exception
#         else:
#             fits = {j: f for j, f in enumerate(fits)}
#     elif not isinstance(fits, dict):
#         fits = {'_': fits}
#     return fits
#
#
# def do_bootstrap2(nobs, fits, param_estimation_func, groups=None, n_samples=DEFAULT_BOOTSTRAP_N_SAMPLES,
#                   seed=DEFAULT_SEED, debug=False, use_correction=True, test_level=DEFAULT_TEST_LEVEL,
#                   group_name=None, method=DEFAULT_BB_METHOD, alpha=DEFAULT_BB_ALPHA, exog_names=None,
#                   max_processes=DEFAULT_BB_MAX_PROCESSES):
#
#     _t = time.time()
#
#     fits = _get_fits(fits)
#
#     if group_name is None:
#         group_name = '<groups>'
#
#     if method.upper() == CLASSICAL:
#         sample_weights, num_unique = get_classical_bootstrap_weights2(nobs, groups, n_samples, seed, max_processes)
#     elif method.upper() == BAYESIAN:
#         sample_weights, num_unique = get_bayesian_bootstrap_weights(nobs, groups, n_samples, seed, max_processes, alpha=alpha)
#     else:
#         raise Exception(f"Bootstrap method {method} invalid!")
#
#     from multiprocessing import RLock
#     from pathos.multiprocessing import Pool
#
#     pool = Pool(processes=max_processes, initargs=(RLock(),), initializer=tqdm.set_lock)
#
#     temp_func = lambda s: get_bootstrapped_param_draws2(s, param_estimation_func, debug, exog_names)
#     jobs = [pool.apply_async(temp_func, s) for s in sample_weights]
#
#     pool.close()
#     pool.join()
#
#     chain_results = [job.get() for job in jobs]
#     print(chain_results)
#
#     # params_boot_list, var_covar_list = get_bootstrapped_param_draws2(
#     #     sample_weights, param_estimation_func, debug=debug, exog_names=exog_names)
#
#     for fit, params_boot, var_covar in zip(fits.values(), params_boot_list, var_covar_list):
#
#         df_t_dist = num_unique - 1 if groups is not None else fit.df_resid
#
#         if use_correction:
#             var_covar *= float(n_samples) / (n_samples - 1)
#
#         cov_kwds = {'seed': seed, 'n_samples': n_samples, 'method': method, 'alpha': alpha}
#         if groups is not None:
#             cov_kwds['groups'] = group_name
#
#         fit.set_cov_params(
#             var_covar, cov_kwds=cov_kwds, cov_type=BOOTSTRAP, test_level=test_level, df_t_dist=df_t_dist)
#
#         if groups is None:
#             if method == CLASSICAL:
#                 bootstrap_string = f'Did {len(params_boot)} classical bootstrap repetitions.'
#             else:
#                 bootstrap_string = f'Did {len(params_boot)} Bayesian bootstrap repetitions, alpha={"%.3f" % alpha}.'
#         else:
#             if method == CLASSICAL:
#                 bootstrap_string = f'Did {len(params_boot)} classical bootstrap repetitions, blocked on {group_name}.'
#             else:
#                 bootstrap_string = f"Did {len(params_boot)} Bayesian bootstrap repetitions, alpha={'%.3f' % alpha}"\
#                                    f", blocked on '{group_name}'."
#
#         fit.set_bootstrapped_params(params_boot, bootstrap_string)
#
#     cov_elapsed = time.time() - _t
#     for fit in fits.values():
#         fit.cov_elapsed = cov_elapsed
#
#
# def get_bootstrapped_param_draws2(samples, func, debug=False, exog_names=None):
#     params_boot = []
#     for sample in tqdm(samples, desc='Bootstrap regression', disable=not debug):
#         params_new = func(sample)
#         if params_new is not None:
#             if isinstance(params_new, pd.Series):
#                 if exog_names is not None:
#                     params_new = np.array([params_new[x] if x in params_new.index else np.nan
#                                            for x in exog_names])
#             params_boot.append(params_new)
#     params_boot = np.array(params_boot)
#
#     is_multi_outcome = np.ndim(params_boot) == 3
#     if is_multi_outcome:
#         params_boot = [params_boot[:, :, k] for k in range(params_boot.shape[2])]
#     else:
#         params_boot = [params_boot]
#
#     var_covar_list = []
#     for p in params_boot:
#         var_covar = np.cov(p.T)
#         if np.product(np.shape(var_covar)) <= 1:
#             var_covar = np.array(var_covar).reshape((p.shape[1], p.shape[1]))
#         var_covar_list.append(var_covar)
#
#     return params_boot, var_covar_list
#
#
# def get_joint_bootstrapped_distribution(fits, return_dataframe=True, small_sample_correction=True):
#
#     fits = _get_fits(fits)
#     if np.any([f.cov_type.lower() != 'bootstrap' for f in fits.values()]):
#         raise Exception("All covariance types must be 'BOOTSTRAP' in `fits`!")
#     f0 = list(fits.values())[0]
#     if np.any([f.cov_kwds != f0.cov_kwds for f in fits.values()]):
#         raise Exception("All cov_kwds dicts must be equal across the fits in `fits`!")
#     if np.any([f.nobs != f0.nobs for f in fits.values()]):
#         raise Exception("All `nobs` must be equal across the fits in `fits`!")
#
#     bootstrapped_params = np.hstack([f.bootstrapped_params for f in fits.values()])
#     var = np.cov(bootstrapped_params.T)
#
#     if small_sample_correction:
#         var *= bootstrapped_params.shape[0] / (bootstrapped_params.shape[0] - 1)
#
#     if return_dataframe:
#         cols = [f'{j}_{f.endog_name}_{x}'
#                 for j, f in enumerate(fits.values())
#                 for x in f.exog_names]
#         return DataFrame(var, columns=cols, index=cols)
#     else:
#         return var
#
#
# def bootstrap_entire_procedure(func, nobs, blocks=None,
#                                n_samples=DEFAULT_BOOTSTRAP_N_SAMPLES, seed=DEFAULT_SEED,
#                                alpha=DEFAULT_BB_ALPHA, return_type='list', debug=False):
#     """
#     :param func: Should be a function that takes in a "weights" vector for bootstrapping and returns parameters
#     """
#
#     weights_generator, n_unique = get_bayesian_bootstrap_weights(nobs, blocks, n_samples, seed, alpha)
#     if debug:
#         weights_generator = tqdm(weights_generator, total=n_samples)
#         weights_generator.set_description('Bootstrapping procedure: ')
#     result = [func(w) for w in weights_generator]
#
#     if return_type.lower() == 'list':
#         pass
#     elif return_type.lower() == 'dataframe':
#         result = DataFrame(result)
#     else:
#         raise Exception("`return_type` must be 'list' or 'dataframe'!")
#
#     return {'result': result,
#             'nobs': nobs,
#             'blocks': blocks,
#             'options': {
#                 'n_samples': n_samples,
#                 'seed': seed,
#                 'alpha': alpha,
#                 'return_type': return_type
#             }
#            }
#
#
# if __name__ == '__main__':
#
#     def go():
#         from kanly.api import lm
#         import numpy as np
#         import pandas as pd
#
#         from kanly.api import lm
#
#         n = 5_500
#         np.random.seed(0)
#         df = pd.DataFrame({
#             'x': np.random.randn(n),
#             'grp': np.random.randint(0, 30, n),
#         })
#         df['y'] = 1.2 - 0.3 * df['x'] + .2 * np.random.randn(n)
#         df['z'] = 4.2 - 0.9 * df['x'] + .2 * np.random.randn(n)
#
#         formula = 'y ~ x + C(grp)'
#
#         T = 2000
#
#         fit_par = lm(formula, df, debug=True, use_t=True, cov_type='bootstrap',
#                      cov_kwds=dict(n_samples=T, max_processes=6, seed=5,
#                                    # groups='grp'
#                                    ),
#                      dense_threshold_mb=np.inf,
#                      inverse_method=np.linalg.inv,
#                      )
#         print(fit_par)
#
#         fit_seq = lm(formula, df, debug=True, use_t=True, cov_type='bootstrap',
#                      cov_kwds=dict(n_samples=T, max_processes=1, seed=5,
#                                    # groups='grp'
#                                    ),
#                      dense_threshold_mb=np.inf,
#                      inverse_method=np.linalg.inv,
#                      )
#         print(fit_seq)
#
#         print(fit_seq.cov_elapsed / fit_par.cov_elapsed)
#
#
#     go()
