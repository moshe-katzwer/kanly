from __future__ import absolute_import, print_function

import numpy as np

from kanly.bayes.mcmc.diagnostics.batched_means import get_max_batch_mean_var_by_sample


def geweke_approx_chain(samples, frac1=.1, frac2=.5, bm_taus=None):
    """Compute approximate Geweke z-scores for a single chain.

    Tests whether the mean of the first ``frac1`` fraction of draws equals
    the mean of the last ``frac2`` fraction, which would indicate stationarity.
    The variance of each window's mean is estimated using the batched-means
    variance estimator on the full chain, inflated by the serial-correlation
    inflation factor, then divided by the window length.

    Under convergence and stationarity, the resulting z-scores should be
    approximately N(0,1).  Values outside ±1.96 suggest non-convergence.

    Reference:
        Geweke, J. (1992). Evaluating the accuracy of sampling-based approaches
        to calculating posterior moments.  In *Bayesian Statistics 4*
        (eds Bernardo, Berger, Dawid & Smith). Clarendon Press, Oxford.

    Args:
        samples: 2-D array of shape ``(n, p)`` of post-burn-in draws for one
            chain.
        frac1: Fraction of draws forming the early window (default 10%).
        frac2: Fraction of draws forming the late window (default 50%).
        bm_taus: Optional tau grid for the batched-means variance estimator.

    Returns:
        1-D array of length ``p`` with the per-parameter Geweke z-scores.
    """

    n = len(samples)
    ns = [int(f * n) for f in [frac1, frac2]]
    cuts = [samples[:ns[0]], samples[-ns[1]:]]
    means = [cut.mean(axis=0) for cut in cuts]

    serial_correlated_variance = get_max_batch_mean_var_by_sample(samples, bm_taus)
    independent_variance = np.var(samples, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        var_inflation_factor = serial_correlated_variance / independent_variance

    varcs = np.array([np.var(cut, axis=0) for cut in cuts])
    with np.errstate(divide='ignore', invalid='ignore'):
        varcs *= var_inflation_factor
        z_stat = (means[0] - means[1]) / np.sqrt(varcs[0] / ns[0] + varcs[1] / ns[1])

    return z_stat


def geweke_approx(chain_samples, frac1=.1, frac2=.5, bm_taus=None):
    """Compute approximate Geweke z-scores across all chains.

    Applies ``geweke_approx_chain`` to each chain in ``chain_samples`` and
    stacks the results into a matrix.  Each row corresponds to a parameter
    and each column to a chain.

    Under convergence the z-scores should be approximately N(0,1); the
    fraction exceeding ±1.96 should be close to 5%.

    Reference:
        Geweke, J. (1992). Evaluating the accuracy of sampling-based approaches
        to calculating posterior moments.  In *Bayesian Statistics 4*.

    Args:
        chain_samples: List of 2-D arrays, one per chain, each of shape
            ``(n_samples, num_params)``.
        frac1: Fraction of draws forming the early window.
        frac2: Fraction of draws forming the late window.
        bm_taus: Optional tau grid for the batched-means variance estimator.

    Returns:
        2-D array of shape ``(num_params, n_chains)`` with per-parameter,
        per-chain Geweke z-scores.
    """

    z_stats = np.array([geweke_approx_chain(cs, frac1, frac2, bm_taus)
                        for cs in chain_samples]).T
    return z_stats
