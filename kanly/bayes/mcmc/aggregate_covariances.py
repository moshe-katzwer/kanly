"""Streaming-style covariance aggregation across sample batches/chains."""
from __future__ import absolute_import, print_function
import numpy as np


def get_total_covariance_from_batches(batches, window_start=None):
    """
    Takes in a list of matrices and computes the overall variance-covariance
    and mean as if they were v-stacked together, without v-stacking them.

    See `aggregate_covs` function.

    Args:
        batches: Sequence of sample matrices, each shaped ``(n_i, p)``.
        window_start: Optional starting row to trim each batch before aggregation.
    """
    if window_start is not None:
        if not (isinstance(window_start, int) and window_start >= 0 and window_start + 2 < len(batches[0])):
            raise Exception(f"Window start {window_start} is not valid with chain sample length {len(batches[0])}")
    batches_sub = [b if window_start is None else b[window_start:] for b in batches]
    lens = [len(b) for b in batches_sub]
    covs = [np.cov(b, rowvar=False, ddof=0) for b in batches_sub]
    means = [np.mean(b, axis=0) for b in batches_sub]
    return aggregate_covs(covs, means, lens)


def aggregate_covs(covs, means, lens):
    """
    Takes in a list of covariances and means, and
    lens (number of samples in each batch) and
    computes an aggregate mean and covariance.

    Args:
        covs: Per-batch covariance matrices.
        means: Per-batch mean vectors.
        lens: Batch sample sizes.
    """
    # Accumulate first and second moments without concatenating all draws.
    M = 0.0
    L = 0.0
    C = 0.0
    m_adjust = 0.0

    for l, m, c in zip(lens, means, covs):
        L += l
        M += l * m
        C += l * c
        m_adjust += l * np.outer(m, m)

    m_adjust -= np.outer(M, M) / L
    C += m_adjust
    M /= L
    C /= L

    return C, M

#
# if __name__ == '__main__':
#     np.random.seed(0)
#     n = 500
#     X = np.array([[1, 3, 2]]) + np.random.randn(n, 3).dot(np.random.rand(3, 3))
#     print(np.cov(X, rowvar=False, ddof=0))
#     print(np.mean(X, axis=0))
#
#     print(get_total_covariance_from_batches([X[:n // 2], X[n // 2:]]))
