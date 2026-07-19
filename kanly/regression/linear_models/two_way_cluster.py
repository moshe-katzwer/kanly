"""Two-way clustered standard errors via Cameron–Gelbach–Miller combination.

This module provides a convenience wrapper that constructs two-way clustered
variance-covariance matrices using the inclusion–exclusion (V₁ + V₂ − V₁₂)
formula from:

    Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
    Robust inference with multiway clustering.
    *Journal of Business & Economic Statistics*, 29(2), 238–249.

Typical usage::

    result = two_way_cluster('y ~ x + z', df, clusters=['firm', 'year'])
"""

from kanly.api import lm


def two_way_cluster(formula, data, clusters, **kwargs):
    """Fit a linear model with two-way clustered standard errors.

    Estimates the model three times using single-way clustering on each
    cluster dimension and on the intersection, then combines the results
    using the Cameron–Gelbach–Miller identity:

        V_two_way = V(cluster_1) + V(cluster_2) − V(cluster_1 × cluster_2)

    The function modifies ``data`` in place by adding a combined cluster
    column whose name is ``clusters[0] + '_' + clusters[1]``.

    Args:
        formula (str): A patsy-style formula string, e.g. ``'y ~ x + z'``.
        data (pd.DataFrame): The dataset.  A new column encoding the
            interaction of the two cluster variables is added temporarily.
        clusters (list of str): Exactly two column names from ``data``
            identifying the two clustering dimensions, e.g.
            ``['firm_id', 'year']``.
        **kwargs: Additional keyword arguments forwarded to ``lm`` (e.g.
            ``absorb``, ``test_level``, ``compute_eigenvalues``).

    Returns:
        SparseLinearRegressionResults: Regression result from the third fit
            (clustered on the intersection), with ``_cov_params`` replaced by
            the two-way combined covariance matrix.  The ``cov_type`` attribute
            is set to ``'CLUSTER-2WAY'``.

    Raises:
        AssertionError: If ``len(clusters) != 2``.
    """
    assert len(clusters) == 2
    data['_'.join(clusters)] = ['_'.join(z) for z in zip(data[clusters[0]].astype(str), data[clusters[1]].astype(str))]
    fit1 = lm(formula, data, cov_type='cluster', cov_kwds={'groups': clusters[0]}, **kwargs)
    fit2 = lm(formula, data, cov_type='cluster', cov_kwds={'groups': clusters[1]}, **kwargs)
    fit3 = lm(formula, data, cov_type='cluster', cov_kwds={'groups': '_'.join(clusters)}, **kwargs)
    cov = fit1._cov_params + fit2._cov_params - fit3._cov_params
    # for f in [fit1, fit2, fit3]:
    #     print(f)
    fit3.set_cov_params(cov, cov_type='CLUSTER-2WAY', cov_kwds={'groups': '_'.join(clusters)})
    return fit3
