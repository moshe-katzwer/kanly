from __future__ import absolute_import, print_function

import time

import numpy as np


def get_rhat(chain_samples, split=2, debug=False):
    """Compute the Gelman-Rubin split R-hat convergence diagnostic.

    Each chain is divided into ``split`` equal segments.  The resulting
    ``n_chains * split`` sub-chains are treated as independent chains for
    the R-hat calculation.  This "split" variant is more sensitive to
    within-chain non-stationarity than the standard (unsplit) R-hat.

    The formula follows Vehtari et al. (2021):

        R_hat = sqrt(var_plus / W)

    where ``var_plus = (n-1)/n * W + 1/n * B`` blends the within-chain
    variance ``W`` and the between-chain variance ``B``.

    Reference:
        Gelman A. et al. (2019). "Rank-normalization, folding, and
        localization: An improved R-hat for assessing convergence."
        https://arxiv.org/abs/1903.08008

    Args:
        chain_samples: List of 2-D arrays, one per chain, each of shape
            ``(n_draws, num_params)``.  Different chains may have different
            lengths.
        split: Number of equal segments to divide each chain into before
            computing R-hat.  Defaults to 2 (standard split R-hat).
        debug: Whether to print timing information.

    Returns:
        5-tuple ``(R_hat, n_eff, r_hat_time, chain_means, between_chain_std)``
        where:

        - ``R_hat``: 1-D array of R-hat values per parameter.
        - ``n_eff``: 1-D integer array of estimated effective sample sizes.
        - ``r_hat_time``: Wall-clock time in seconds for the computation.
        - ``chain_means``: 1-D array of grand means (mean of split-chain means).
        - ``between_chain_std``: 1-D array of between-chain standard deviations.
    """

    _t = time.time()
    if debug:
        print("Computing Gelman-Rubin Split R_hat... ", end='')

    num_draws_total = sum([len(c) for c in chain_samples])
    n_chains = len(chain_samples)
    draws_per_split_chain = num_draws_total / (n_chains * split)

    chn_means = []
    chn_vars = []
    for c in chain_samples:
        ll = len(c)
        chn_means += [
            np.mean(c[(s * ll // split):((s + 1) * ll // split)], axis=0)
            for s in range(split)
        ]
        chn_vars += [
            np.var(c[(s * ll // split):((s + 1) * ll // split)], axis=0, ddof=1)
            for s in range(split)
        ]
    chn_means = np.array(chn_means)
    chn_vars = np.array(chn_vars)

    W = np.mean(chn_vars, axis=0)
    B = np.var(chn_means, axis=0, ddof=1) * draws_per_split_chain

    with np.errstate(divide='ignore', invalid='ignore'):
        R_hat = np.sqrt(
            ((draws_per_split_chain - 1) / draws_per_split_chain * W + 1 / draws_per_split_chain * B)
            / W
        )

    between_chain_std = np.sqrt(B / draws_per_split_chain)

    r_hat_time = time.time() - _t
    if debug:
        print('%.2fs' % r_hat_time)

    # R_hat = sqrt( var_plus / W )
    var_plus = R_hat ** 2 * W
    with np.errstate(divide='ignore', invalid='ignore'):
        n_eff = np.clip(num_draws_total * var_plus / B, 0, np.inf).astype(int)
        if isinstance(B, float):
            if B < 1e-8:
                n_eff = 0
        else:
            n_eff[B < 1e-8] = 0

    return R_hat, n_eff, r_hat_time, np.mean(chn_means, axis=0), between_chain_std


def acf_approx(x, max_lag=10_000):
    """Compute a fast approximate autocorrelation function at selected lags.

    Instead of computing ACF at every lag (O(n²)), evaluates it at a sparse
    geometric grid of lag values: ``[0, 1, 2, 4, 10, 50, 100, 200, ..., max_lag]``.
    The ACF values are floored at 0 (non-negative truncation) to avoid the
    sign-oscillation problem in long-run variance estimation.

    The function also returns ``n_eff_denom``, the denominator for the
    initial positive sequence ESS estimator:

        ESS_denom = 1 + 2 * Σ_{pairs} rho[lag] * Δ_lag

    where the sum is over adjacent pairs in the lag grid.

    Args:
        x: 1-D array of MCMC draws for a single parameter.
        max_lag: Maximum lag to include; automatically capped at 2n/3.

    Returns:
        3-tuple ``(lags, rhos, n_eff_denom)`` where ``lags`` is a 1-D integer
        array of evaluated lags, ``rhos`` is the corresponding ACF values
        (floored at 0), and ``n_eff_denom`` is the ESS denominator.
    """
    n = len(x)
    max_lag = min(2 * n // 3, max_lag)

    lags = np.array(
        [0, 1, 2, 4, 10, 50]
        + [int(2 ** k) for k in range(int(np.log2(100)), int(np.log2(max_lag)))]
        + [max_lag]
    )
    m, v = np.mean(x), np.var(x)

    x_dem = x - m
    rhos = np.ones(len(lags))

    for i in range(1, len(lags)):
        rhos[i] = max(np.dot(x_dem[lags[i]:], x_dem[:-lags[i]]) / ((n - lags[i]) * v), 0)

    # sum of 1 + 2 * sum_{lag=1}^{max_lag} ( rho[lag] )
    n_eff_denom = 1 + np.sum((rhos[1:-1] + rhos[2:]) * (lags[2:] - lags[1:-1]))

    return lags, rhos, n_eff_denom

#
# def fast_slope(X):
#     """
#     Given an n x k array, computes the slope of each column, against the 'index' (row index).
#
#     That is, if x[t] is the vector, this returns cov(x, t) / var(t), that is
#     n times the slope of x regressed on the index t.
#     Also returns standard error of slope coefficient.
#
#     For well mixing MCMC this should be zero.
#
#     Returns coef_slope, std_err_slope
#     """
#
#     n = X.shape[0]
#     mean_t = (n - 1) / 2
#     denom = (n - 1) * (2 * (n - 1) + 1) / 6 - (n - 1) ** 2 / 4  # variance of np.range(n)
#
#     t = np.arange(n) - mean_t
#     coef_slope = np.dot(t, X) / (n * denom)
#
#     X_mean = np.mean(X, axis=0)
#     std_err_slope = np.array([
#         np.linalg.norm(X[:, j] - X_mean[j] - coef_slope[j] * t)
#         for j in range(X.shape[1])
#     ])
#     std_err_slope /= np.sqrt(denom) * n
#
#     return coef_slope, std_err_slope


# if __name__ == '__main__':
#
#     import matplotlib.pyplot as plt
#
#     n_chains = 4
#     np.random.seed(0)
#     runs_per_chain = 1000
#     x = [np.random.randn(runs_per_chain) + np.random.randn()*.954 for _ in range(n_chains)]
#     print(get_rhat(x))
#
#     x_sub = []
#     for _x in x:
#         x_sub += [_x[:len(_x) // 2], _x[len(_x) // 2:]]
#
#     chain_means = np.array([_x.mean() for _x in x_sub])
#     grand_mean = np.mean(chain_means)
#     N = len(x_sub[0])  # draws per chain
#     M = len(x_sub)  # number of chains
#
#     B = N / (M - 1.) * np.sum((chain_means - grand_mean) ** 2)
#     s_m = np.array([np.sum((_x - chain_means[m]) ** 2) for m, _x in enumerate(x_sub)]) / (N - 1)
#     W = 1. / M * np.sum(s_m)
#
#     var_plus = (N - 1.) / N * W + 1. / N * B
#
#     print(">>> ", N)
#     print('\t > W = ', W)
#     print('\t > B = ', B)
#     print('\t > N = ', N)
#     print('\t > var_plus = ', (N - 1.) / N * W + 1. / N * B)
#
#
#     R_hat = np.sqrt(var_plus / W)
#     print(R_hat, var_plus, B, W)
