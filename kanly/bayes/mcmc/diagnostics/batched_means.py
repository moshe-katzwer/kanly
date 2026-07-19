"""
Batched-means estimators for ESS (Effective Sample Size) of MCMC chains.

The core idea is to divide the chain into ``a`` non-overlapping batches of
size ``b = n // a`` and use the variance of the batch means as an estimate
of the serially-correlated variance.  The ESS is then ``n / inflation_factor``
where ``inflation_factor = serial_correlated_var / iid_var``.

Batch size is controlled by a tuning parameter ``tau`` via ``b = n**tau``,
``a = n**(1-tau)``.  Several ``tau`` values are tried and the most conservative
(largest variance / lowest ESS) is kept.

Functions:
    _get_batch_mean_values_1d  — core 1-D batching (Numba JIT)
    _get_batch_mean_values_2d  — multi-parameter wrapper
    get_batch_mean_autocorrelation — AR(1) coefficient of batch means
    get_batch_mean_cov         — serially-correlated covariance estimate
    _get_default_taus          — default grid of tau values
    get_max_batch_mean_var_by_sample — max variance over tau grid
    get_min_self_autocorrel_batch_mean_var_by_sample — min AR(1) variance
    get_ess_batched_means      — per-chain ESS for a single sample array
    get_ess_batched_means_for_chains — ESS across all chains
    multi_ess                  — multivariate ESS (Vats et al. 2019)
"""
from __future__ import absolute_import, print_function

from numba import njit
import numpy as np

import warnings

DEFAULT_BATCHED_MEANS_SELECTION = 'coordinate'


@njit(cache=True)
def _get_batch_mean_values_1d(posterior, a=None):
    """Compute batch means for a 1-D chain.

    Divides ``posterior`` into ``a`` contiguous batches of equal size
    ``b = n // a`` and returns the mean of each batch.

    Args:
        posterior: 1-D array of MCMC draws for a single parameter.
        a: Number of batches.  Defaults to ``int(n ** 0.5)`` when ``None``.

    Returns:
        3-tuple ``(batch_means, a, b)`` where ``batch_means`` is a 1-D array
        of length ``a``, and ``b = n // a`` is the batch size.
    """
    # b is batch size, a is number of batches
    n = len(posterior)
    if a is None:
        a = int(n ** 0.5)  # size of batch
    b = n // a  # number of batches

    batch_mns = [posterior[a_ * b:(a_ + 1) * b].mean() for a_ in range(a)]
    return np.array(batch_mns), a, b


def _get_batch_mean_values_2d(posterior, a=None, return_ar1_coefs=False):
    """Compute batch means for a 2-D (multi-parameter) chain.

    Applies ``_get_batch_mean_values_1d`` independently to each column
    (parameter) of ``posterior``.

    Args:
        posterior: 2-D array of shape ``(n, p)`` where ``n`` is the number of
            draws and ``p`` is the number of parameters.
        a: Number of batches.  Defaults to ``int(n ** 0.5)`` when ``None``.
        return_ar1_coefs: When ``True``, also computes and returns the AR(1)
            coefficient of the batch means for each parameter (used to select
            the batch size that minimizes autocorrelation).

    Returns:
        When ``return_ar1_coefs=False``:
            3-tuple ``(batch_means, a, b)`` where ``batch_means`` has shape
            ``(p, a)``.

        When ``return_ar1_coefs=True``:
            4-tuple ``(batch_means, a, b, rho)`` where ``rho`` has shape
            ``(p,)`` containing the per-parameter AR(1) coefficients of the
            batch means.
    """
    result = [_get_batch_mean_values_1d(posterior[:, j], a)
              for j in range(posterior.shape[1])]
    batch_mns = np.array([r[0] for r in result])
    a, b = result[0][1], result[0][2]
    if return_ar1_coefs:
        rho = get_batch_mean_autocorrelation(batch_mns)
        return batch_mns, a, b, rho
    else:
        return batch_mns, a, b


def get_batch_mean_autocorrelation(batch_means):
    """Compute the lag-1 autocorrelation of the batch means for each parameter.

    The AR(1) coefficient is estimated as
    ``rho[j] = Cov(bm[j, 1:], bm[j, :-1]) / Var(bm[j, :])``
    using the full batch-mean series.  A value near zero indicates that the
    batch means are approximately serially uncorrelated, which is desired for
    a valid batched-means variance estimate.

    Args:
        batch_means: 2-D array of shape ``(p, a)`` where ``p`` is the number
            of parameters and ``a`` is the number of batches.

    Returns:
        1-D array of length ``p`` with the lag-1 autocorrelation coefficient
        for each parameter's batch-mean series.
    """
    xy = (batch_means[:, 1:] * batch_means[:, :-1]).mean(axis=1)
    m1, m2 = batch_means[:, 1:].mean(axis=1), batch_means[:, :-1].mean(axis=1)
    var = (batch_means ** 2).mean(axis=1) - batch_means.mean(axis=1) ** 2
    rho = (xy - m1 * m2) / var
    return rho


def get_batch_mean_cov(posterior, a=None, diag=True):
    """Estimate the (co)variance of the sample mean under serial correlation.

    Uses the batch-means estimator: split the chain into ``a`` batches and
    compute the (co)variance of the batch means, scaled by the batch size
    ``b`` to estimate the asymptotic variance of the chain mean.

    The scaling factor ``b * a / (a - 1)`` makes the estimator unbiased for
    the asymptotic variance of the sample mean.

    Args:
        posterior: 1-D (single parameter) or 2-D (``n × p``) array of draws.
        a: Number of batches; defaults to ``int(n ** 0.5)`` when ``None``.
        diag: When ``True`` (and ``posterior`` is 2-D), return only the
            diagonal (per-parameter variances) rather than the full
            covariance matrix.

    Returns:
        Scalar (1-D input), 1-D array of per-parameter variances (2-D input
        with ``diag=True``), or a ``(p, p)`` covariance matrix (``diag=False``).
    """
    # b is batch size, a is number of batches
    if np.ndim(posterior) == 2:
        bch_means, a, b = _get_batch_mean_values_2d(posterior, a)
    else:
        bch_means, a, b = _get_batch_mean_values_1d(posterior, a)
        bch_means = bch_means

    if diag and np.ndim(posterior) == 2:
        return (b * a / (a - 1)) * np.var(bch_means, axis=1)
    else:
        return (b * a / (a - 1)) * np.cov(bch_means)


def _get_default_taus(n):
    """Return the default grid of tau values used to select the batch size.

    The batch size is ``n ** tau`` and the number of batches is ``n ** (1 - tau)``.
    The grid runs from 0.3 (many batches, small batches) up to
    ``log(n/40) / log(n)`` (fewer, larger batches), covering 8 evenly spaced
    points on the log-scale.

    Args:
        n: Total number of draws in the chain.

    Returns:
        1-D NumPy array of 8 tau values in ``(0, 1)``.
    """
    return np.linspace(.3, np.log(n / 40) / np.log(n), 8)


def get_max_batch_mean_var_by_sample(posterior, taus=None):
    """Estimate the serially-correlated variance using the most conservative tau.

    Computes the batch-mean variance for each candidate tau in ``taus``
    and returns the element-wise maximum across all tau values.  Taking
    the maximum (minimum ESS) provides a conservative, robust estimate.

    Args:
        posterior: 1-D or 2-D array of draws.  When 2-D, each column is a
            separate parameter.
        taus: 1-D array of tau candidates in ``(0, 1)``; auto-selected
            from ``_get_default_taus(n)`` when ``None``.

    Returns:
        Scalar (1-D input) or 1-D array of per-parameter maximum
        batch-mean variances.
    """
    n = len(posterior)
    if taus is None:
        taus = _get_default_taus(n)

    batch_mean_vars = [get_batch_mean_cov(posterior, a=int(n ** (1 - t)), diag=True)
                       for t in taus]

    return np.max(batch_mean_vars, axis=0)


def get_min_self_autocorrel_batch_mean_var_by_sample(posterior, taus=None):
    """Estimate the serially-correlated variance using the tau with lowest batch-mean AR(1).

    For each candidate tau, computes the batch means and their AR(1)
    coefficients.  Selects the tau that minimizes the sum of squared AR(1)
    coefficients across parameters—i.e., the batch size that makes the
    batch means most nearly serially uncorrelated—then returns the
    batch-mean variance at that tau.

    Args:
        posterior: 2-D array of shape ``(n, p)`` of MCMC draws.
        taus: 1-D array of tau candidates in ``(0, 1)``; auto-selected when
            ``None``.

    Returns:
        1-D array of length ``p`` with the batch-mean variance at the
        selected tau for each parameter.
    """
    n = len(posterior)

    if taus is None:
        taus = _get_default_taus(n)

    As = [int(n ** (1 - t)) for t in taus]
    result = [_get_batch_mean_values_2d(posterior, a=a, return_ar1_coefs=True)
              for a in As]
    lowest_ar1_sqrd = np.argmin([np.sum(r[3] ** 2) for r in result])
    a = As[lowest_ar1_sqrd]
    b = n // a
    return (b * a / (a - 1)) * np.var(result[lowest_ar1_sqrd][0], axis=1)


def get_ess_batched_means(posterior, taus=None, selection=DEFAULT_BATCHED_MEANS_SELECTION):
    """Estimate the Effective Sample Size (ESS) via the batched-means method.

    ESS is computed as ``n / inflation_factor`` where
    ``inflation_factor = serial_correlated_var / iid_var``.

    Two strategies are supported:

    - ``'coordinate'``: Uses ``get_max_batch_mean_var_by_sample`` — picks
      the tau that maximises the estimated serially-correlated variance for
      each parameter independently, giving a conservative (lower) ESS.
    - ``'joint'``: Uses ``get_min_self_autocorrel_batch_mean_var_by_sample``
      — picks the batch size that minimizes the total AR(1) of batch means.

    Args:
        posterior: 1-D or 2-D array of post-burn-in draws.
        taus: Candidate tau values; auto-selected when ``None``.
        selection: Either ``'coordinate'`` or ``'joint'``.

    Returns:
        Integer array of ESS values (one per parameter for 2-D input,
        scalar for 1-D input).
    """
    n = len(posterior)
    if selection == 'coordinate':
        serial_correlated_var = get_max_batch_mean_var_by_sample(posterior, taus)
    elif selection == 'joint':
        serial_correlated_var = get_min_self_autocorrel_batch_mean_var_by_sample(posterior, taus)
    else:
        raise Exception('selection must by "joint" or "coordinate"')

    independent_var = np.var(posterior, axis=0)
    inflation_factor = serial_correlated_var / independent_var
    return (n / inflation_factor).astype(int)


def get_ess_batched_means_for_chains(chain_samples, taus=None, selection=DEFAULT_BATCHED_MEANS_SELECTION):
    """Compute per-chain ESS for all parameters using the batched-means method.

    Applies ``get_ess_batched_means`` to each chain's sample array and stacks
    the results.

    ``tau`` variables in ``(0, 1)`` control the batch size: batch size
    ``b = n ** tau``, number of batches ``a = n ** (1 - tau)``.

    When ``selection='joint'``, ``tau`` is chosen to minimize the sum of
    squares of AR(1) coefficients across all parameters in the batch means,
    making the batch means as serially uncorrelated as possible.

    When ``selection='coordinate'``, ``tau`` is chosen per parameter to
    maximise the estimate of the serially-correlated variance, yielding a
    conservative (lower) ESS.

    Args:
        chain_samples: List of 2-D arrays, one per chain, each of shape
            ``(n_samples, num_params)``.
        taus: Candidate tau values; auto-selected when ``None``.
        selection: ``'coordinate'`` or ``'joint'``; see ``get_ess_batched_means``.

    Returns:
        2-D integer array of shape ``(num_params, n_chains)`` with the ESS
        for each parameter and chain.
    """
    return np.array([
        get_ess_batched_means(C, taus, selection) for C in chain_samples
    ]).transpose()


def multi_ess(posterior, taus=None, cov_indep=None):
    """Compute the multivariate Effective Sample Size (multi-ESS).

    Implements the estimator from Vats, Flegal & Jones (2019):

        multi_ESS = n * (|Λ| / |Σ|)^(1/p)

    where ``Λ`` is the sample covariance under independence (``np.cov``),
    ``Σ`` is the asymptotic covariance in the Markov-chain CLT (estimated
    via batched means), ``n`` is the number of draws, and ``p`` is the
    number of parameters.

    The most conservative (largest ``|Σ|``) estimate across all tau values
    is used to give a lower bound on the effective information.

    Reference:
        Vats D., Flegal J.M., Jones G.L. (2019).
        "Multivariate Output Analysis for Markov chain Monte Carlo."
        Biometrika, 106(2), 321–337. https://arxiv.org/abs/1512.07713

    Args:
        posterior: 2-D array of shape ``(n, p)`` of post-burn-in draws.
        taus: Candidate batch-size tau values; auto-selected when ``None``.
        cov_indep: Optional pre-computed sample covariance matrix (e.g.,
            the posterior ``cov_params`` from ``MCMCResults``).  Computed
            from ``posterior`` when ``None``.

    Returns:
        Integer multi-ESS estimate, or ``np.nan`` if computation fails
        (e.g., singular covariance).
    """
    n, p = posterior.shape
    if taus is None:
        taus = _get_default_taus(n)

    if cov_indep is None:
        Lambda = np.cov(posterior, rowvar=False)
    else:
        Lambda = cov_indep

    slogdet_Lambda = np.linalg.slogdet(Lambda)[1]

    Sigmas = [
        get_batch_mean_cov(posterior, a=int(n ** (1 - t)), diag=False)
        for t in taus
    ]
    for i, S in enumerate(Sigmas):
        if np.ndim(S) < 2:
            Sigmas[i] = np.array([S]).reshape((1, 1))
    slogdet_Sigma = np.max([np.linalg.slogdet(S)[1] for S in Sigmas])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return int(n * np.exp((slogdet_Lambda - slogdet_Sigma) / p))
        except:
            return np.nan
