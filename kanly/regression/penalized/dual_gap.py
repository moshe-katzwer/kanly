from __future__ import absolute_import, print_function
import numpy as np
from scipy.sparse import isspmatrix


def elastic_net_duality_gap(
        X: np.ndarray,
        y: np.ndarray,
        intercept: float,
        coef: np.ndarray,
        l1_penalties: np.ndarray,  # (k,) per-coefficient L1 penalties
        l2_penalties: np.ndarray,  # (k,) per-coefficient L2 penalties
        fit_intercept: bool = True,
        sample_weight: np.ndarray | None = None,
        positive: bool = False,
) -> float:
    """
    Compute the elastic net duality gap with per-coefficient penalties.

    Primal problem
    --------------
    Let sw_i be sample weights normalized to sum to n, r_i = y_i - x_i @ b - b0.
    The primal objective_function is:

        P(b0, b) = (1/[2n]) * sum_i sw_i * r_i^2
                   + sum_j lambda1_j * |b_j|
                   + sum_j lambda2_j * b_j^2

    where lambda1_j = l1_penalties[j] and lambda2_j = l2_penalties[j].
    When all lambda1_j = alpha * l1_ratio and all lambda2_j = alpha * (1 - l1_ratio) / 2,
    this reduces to the standard sklearn ElasticNet objective_function.

    Dual feasibility
    ----------------
    The KKT stationarity condition for coefficient j is:

        -(X_j^T S r) / n  +  lambda1_j * s_j  +  lambda2_j * b_j  =  0

    where s_j in subdiff(|b_j|): s_j in [-1, 1] (or [−inf, 1] when positive=True).
    Rearranging, the dual variable theta = S r / n must satisfy, for each j:

        |(X_j^T S theta) - lambda2_j * b_j|  <=  lambda1_j

    Define XtA_j = (X.T @ R)_j - centering_j - n * lambda2_j * b_j, where
    R = S r are the weighted residuals and centering is the intercept correction
    (see below).  Feasibility then requires |XtA_j| <= n * lambda1_j for all j.

    Projection
    ----------
    The unscaled dual candidate theta = R / n may violate feasibility.  We
    project it via a single scalar: theta_feas = const_ * theta, where const_
    is the largest value in [0, 1] such that theta_feas remains feasible.

    Since XtA_j scales linearly with the residuals, the per-coordinate upper
    bound on const_ is n * lambda1_j / |XtA_j| when |XtA_j| > n * lambda1_j.
    Taking the minimum over all violating coordinates:

        const_ = min(1, min_j  n * lambda1_j / |XtA_j|)

    For uniform penalties this reduces exactly to alpha_c / ||XtA||_inf,
    recovering sklearn's scalar projection.

    Gap formula
    -----------
    Following sklearn's unified KKT expression (scaled by n to match the
    internal Cython convention of working with the un-normalized objective_function):

        n * gap = (1/2) * (1 + const_^2) * R_norm2          # loss terms
                  + n * sum_j lambda1_j * |b_j|              # L1
                  - const_ * R^T y                           # cross term
                  + (n/2) * (1 + const_^2) * sum_j lambda2_j * b_j^2  # L2

    where R_norm2 = sum_i sw_i * r_i^2 is the weighted sum of squared residuals.
    When const_ = 1 (already feasible), the first term simplifies to R_norm2.

    Dividing by n gives the per-sample gap.

    Intercept and centering
    -----------------------
    When fit_intercept=True, the intercept b0 is unpenalized.  Its KKT
    condition requires 1^T (S theta) = 0, i.e. sum_i sw_i * theta_i = 0.
    This is equivalent to working with centered X and y.  Rather than forming
    X_c = X - X_mean explicitly (which destroys sparsity), we use the identity:

        X_c^T R = X^T R - X_mean * (1^T R) = X^T R - X_mean * R_sum

    The correction X_mean * R_sum is a rank-1 subtraction that costs O(n + k)
    and never modifies X.  At convergence R_sum ~ 0 (intercept stationarity).

    Parameters
    ----------
    X             : (n, k) — feature matrix, never modified (sparsity safe)
    y             : (n,)   — response vector
    coef          : (k,)   — fitted coefficients
    intercept     : float  — fitted intercept (0.0 if fit_intercept=False)
    l1_penalties  : (k,)   — per-coefficient L1 penalty (lambda1_j >= 0)
    l2_penalties  : (k,)   — per-coefficient L2 penalty (lambda2_j >= 0)
    fit_intercept : bool   — must match the value used when fitting
    sample_weight : (n,) or None — raw sample weights, normalized to sum to n
    positive      : bool   — if True, dual feasibility is one-sided (XtA_j <= n*lambda1_j)

    Returns
    -------
    float
        Duality gap >= 0.  Equals sklearn's ``dual_gap_`` when penalties are
        uniform (all lambda1_j = alpha * l1_ratio, all lambda2_j = alpha * (1 - l1_ratio)).
    """
    n = X.shape[0]

    l2_penalties = l2_penalties * 2.0

    # Normalize weights to sum to n
    is_weighted = sample_weight is not None
    if is_weighted:
        sw = sample_weight * n / sample_weight.sum()
    else:
        sw = np.ones(n)

    if isspmatrix(y):
        y = y.toarray().flatten()

    resid = y - X.dot(coef) - intercept
    R = sw * resid  # weighted residuals: R_i = sw_i * r_i
    R_sum = R.sum()  # 1^T R; ~0 at convergence when intercept is fit

    # Rank-1 centering correction: X_c^T R = X^T R - X_mean * R_sum
    if fit_intercept:
        if isspmatrix(X):
            if is_weighted:
                X_mean = np.array([
                    (X.getcol(i) * sw).mean() for i in range(X.shape[1])
                ])
            else:
                X_mean = X.mean(axis=0)
        else:
            X_mean = np.average(X, axis=0, weights=sw if is_weighted else None)
        centering = X_mean * R_sum
    else:
        centering = 0.0

    # XtA_j = (X^T R - centering)_j - n * lambda2_j * b_j
    # Feasibility: |XtA_j| <= n * lambda1_j for all j
    XtA = X.T @ R - centering - n * l2_penalties * coef

    if np.any(positive):
        violations = XtA / (n * l1_penalties)
        worst = violations.max()
    else:
        violations = np.abs(XtA) / (n * l1_penalties)
        worst = violations.max()

    const_ = 1.0 / worst if worst > 1.0 else 1.0

    # Weighted SSR: sum_i sw_i * r_i^2
    R_norm2 = (R ** 2 / sw).sum()

    # Unified KKT gap expression, scaled by n
    gap = R_norm2 if const_ == 1.0 else 0.5 * (1 + const_ ** 2) * R_norm2
    gap += (n * (l1_penalties * np.abs(coef)).sum()
            - const_ * (R @ y)  # sum sw_i r_i y_i
            + (n / 2) * (1 + const_ ** 2) * (l2_penalties * coef ** 2).sum())

    return gap / n
