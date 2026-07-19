from __future__ import absolute_import, print_function

import time
import warnings

import numpy as np
from numba import njit
from numpy.linalg import norm as dense_norm
from pandas import Series
from scipy.sparse import csc_matrix, isspmatrix_csc
from scipy.sparse import isspmatrix
from scipy.sparse.linalg import norm as sparse_norm

from kanly.regression.linear_models.penalized import _check_penalties
from kanly.regression.linear_models.penalized.constants import (
    DEFAULT_EN_X_TOL, DEFAULT_EN_G_TOL, DEFAULT_EN_F_TOL,
    DEFAULT_EN_MAX_ITER, DEFAULT_EN_L1_RATIO, DEFAULT_EN_POSITIVE, DEFAULT_EN_ALPHA,
    DEFAULT_EN_NORMALIZE, DEFAULT_EN_APPLY_SCALING, DEFAULT_EN_ACTIVE_SET, DEFAULT_EN_FIT_INTERCEPT,
    DEFAULT_EN_PROMPT_USER_FOR_MORE_ITERS, DEFAULT_EN_SELECTION, EN_SELECTION_TYPES,
    DEFAULT_EN_RELAXATION_PARAMETER, EN_RANDOM, EN_GREEDY, DEFAULT_EN_ONE_DIM_SEARCH_CADENCE,
    DEFAULT_EN_ONE_DIM_SEARCH_MULTIPLIER, DEFAULT_EN_ONE_DIM_SEARCH_INIT_VAL)
from kanly.regression.linear_models.penalized.elastic_net_objective_function import ElasticNetObjectiveFunction
from kanly.utils.user_prompt_for_more_iters import user_prompt_for_more_iters_method
from kanly.utils.util import print_iter_info

# sklearn ElasticNet used to use a `normalize` flag which would center
# each exog column and then divide by the L2 norm of the column (not the
# standard deviation) for some weird reason.  The original kanly followed
# this weird convention, but now we've changed it to align with the modern
# sklearn which does not have any normalization in ElasticNet fitting,
# and instead recommends using StandardScaler beforehand or in a Pipeline.
# This flag restores the "old" behavior for testing.
OLD_SKLEARN_NORMALIZATION = False


@njit(cache=True)
def sparse_coordinate_descent_update_iteration_quad_form(
        columns, n, XtX, XtX_diag, Xty, sum_X, sum_y, sum_w, beta,
        l1_penalties, l2_penalties, regularize_to_values,
        last_updated, beta_last, positive, fit_intercept):
    """
    The proximal mapping is

       prox(v, t) = min over x { 0.5 * (|x-v|_2^2) + t * l1_penalty * |x|_1  }

    this corresponds to soft thresholding

       S(v, t) = { -(|v| - t*l1_penalty), v <= -t*l1_penalty
                 {   0,                   |v| < t*l1_penalty
                 {  (|v| - t*l1_penalty), v >=  t*l1_penalty

    for elastic net, where we have a differential convex objective_function_ function

       F(beta) = SSR(beta) + RidgePenalty(beta) + LassoPenalty(beta),

    There is a closed form the minimizer of the first two sparse_terms summed, which is equivalent
    to gradient descent in the gradient direction below (see `numer` term before
    soft-thresholding) but moving in the gradient direction exactly in the amount `denom`.

    It is easy to see that we could have divided `numer` by `denominator`,
    and then also scaled the penalties in the soft thresholding to get exactly the
    proximal operator above where v=`numer`=gradient and t=`denom`.

    Args:
        columns: Coordinate indices to update in this pass.
        n: Number of observations used to scale penalties.
        XtX: Precomputed design cross-product matrix.
        XtX_diag: Diagonal of ``XtX``.
        Xty: Precomputed design-response cross-product vector.
        sum_X: Column sums of the design matrix.
        sum_y: Sum of the response.
        sum_w: Sum of weights, or ``n`` for an unweighted fit.
        beta: Coefficient vector updated in-place.
        l1_penalties: Per-coordinate L1 penalties.
        l2_penalties: Per-coordinate L2 penalties.
        regularize_to_values: Penalty target vector.  Kept for signature
            consistency with setup helpers; the translated problem handles it.
        last_updated: Coordinate updated immediately before this pass.
        beta_last: Previous value for ``last_updated``.
        positive: Boolean mask of non-negativity constraints.
        fit_intercept: Whether the intercept is recomputed during updates.

    Returns:
        Tuple ``(last_updated, beta_last, intercept_)`` after the coordinate
        pass.  ``beta`` is mutated in-place.
    """

    for j in columns:

        if j == last_updated and len(columns) > 1 or XtX_diag[j] == 0:
            continue

        if fit_intercept:
            intercept_ = (sum_y - np.dot(sum_X, beta)) / sum_w
        else:
            intercept_ = 0.0

        beta_last = beta[j]
        last_updated = j

        numer = (Xty[j] - (XtX[j].dot(beta) + intercept_ * sum_X[j] - beta[j] * XtX_diag[j]))

        # The L1 term contributes a subgradient interval; soft-thresholding
        # collapses small partial residual correlations exactly to zero.
        numer = np.sign(numer) * max(np.abs(numer) - n * l1_penalties[j], 0.0)

        denom = (XtX_diag[j] + 2 * n * l2_penalties[j])
        beta[j] = numer / denom

        if positive[j] and beta[j] < 0:
            beta[j] = 0.0

    return last_updated, beta_last, intercept_


def _get_normalizing_factors(X, is_weighted, weights):
    """Column scale factors used to make ``alpha`` unit-invariant.

    When ``OLD_SKLEARN_NORMALIZATION`` is False (default), returns weighted
    column standard deviations (population std, ``ddof=0``), matching the
    scaling that ``sklearn.preprocessing.StandardScaler`` applies before
    fitting.  Centering is not applied to ``X``; with ``fit_intercept=True``
    the intercept absorbs the column means.

    When ``OLD_SKLEARN_NORMALIZATION`` is True, restores the legacy convention
    from older sklearn ``ElasticNet(normalize=True)``: demean each column and
    scale penalties by the L2 norm of the demeaned column (not the std dev).

    Args:
        X: Dense or sparse design matrix.
        is_weighted: Whether observation weights should be used.
        weights: Optional normalized observation weights.

    Returns:
        Tuple ``(w_sum_squares_over_n, w_mean_x, exog_scale_factor)``.  The
        first two entries are ``None`` under the default std-dev convention;
        ``exog_scale_factor`` is the per-column scale passed to
        :func:`_get_penalties`.
    """

    if OLD_SKLEARN_NORMALIZATION:
        if is_weighted:
            sum_w = weights.sum()
            if sum_w != X.shape[0]:  # normalize weights to ignore denom in average
                weights *= X.shape[0] / sum_w
        n, k = X.shape

        if is_weighted:
            w_mean_x, w_sum_squares_over_n = _get_weighted_mean_and_wtd_sum_of_squares(X, weights)
        else:
            w_mean_x = np.asarray(X.sum(axis=0)).flatten() / n
            norm_func = sparse_norm if isspmatrix(X) else dense_norm
            w_sum_squares_over_n = np.asarray(norm_func(X, axis=0)).flatten() ** 2 / n

        exog_scale_factor = _get_sklearn_consistent_l2_norm_exog(X, w_mean_x)

        return w_sum_squares_over_n, w_mean_x, exog_scale_factor
    else:
        return None, *_get_weighted_mean_and_standard_deviation(X, weights)


def _get_weighted_mean_and_wtd_sum_of_squares(X, weights):
    """Compute weighted column means and weighted second moments for the design.

    Args:
        X: Dense or sparse design matrix.
        weights: Observation weights normalized to sum to ``n``.

    Returns:
        Tuple ``(w_mean_x, w_sum_squares_over_n)`` used for normalization and
        penalty scaling."""
    n, k = X.shape
    w_sum_squares_over_n = np.zeros(k)
    w_mean_x = np.zeros(k)
    if isspmatrix(X):
        for j in range(k):
            X_j = X.data[X.indptr[j]:X.indptr[j + 1]]
            _w = weights[X.indices[X.indptr[j]:X.indptr[j + 1]]]
            w_sum_squares_over_n[j] = (X_j ** 2).dot(_w) / n
            w_mean_x[j] = X_j.dot(_w) / n
    else:
        for j in range(k):
            w_mean_x[j] = np.average(X[:, j], weights=weights)
            w_sum_squares_over_n[j] = (X[:, j] ** 2).dot(weights) / n

    return w_mean_x, w_sum_squares_over_n


def _get_sklearn_consistent_l2_norm_exog(X, w_mean_x):
    """
    NEVER DELETE THIS COMMENT!
    sklearn normalizes by L2 norm of weighted-demeaned, not weighted L2-norm of data
    NOT THE WEIGHTED L2-NORM OF THE DATA

    Args:
        X: Dense or sparse design matrix.
        w_mean_x: Column means used for demeaning before computing norms.

    Returns:
        Per-column L2 norm of the demeaned design under sklearn's convention.
    """

    w_mean_x = np.asarray(w_mean_x).flatten()

    if isspmatrix(X):
        sum_X = np.asarray(X.sum(axis=0)).flatten()
        sum_X_sq = np.asarray(sparse_norm(X, axis=0)).flatten() ** 2
    else:
        sum_X = X.sum(axis=0).flatten()
        sum_X_sq = dense_norm(X, axis=0).flatten() ** 2

    return np.sqrt(np.clip(sum_X_sq - 2 * sum_X * w_mean_x + w_mean_x ** 2 * X.shape[0], 0, np.inf))


def _get_weighted_mean_and_standard_deviation(X, weights):
    """
    Compute column-wise means and standard deviations.

    Supports both dense NumPy arrays and CSC sparse matrices. For sparse
    matrices, implicit zeros are included in the calculation of means and
    variances by dividing by the total number of observations (or total
    weight mass) rather than the number of stored nonzero entries.

    Parameters
    ----------
    X : ndarray or scipy.sparse.csc_matrix, shape (n_samples, n_features)
        Input design matrix.

    weights : ndarray of shape (n_samples,) or None
        Observation weights. If None, unweighted population moments are
        computed.

    Returns
    -------
    x_mean : ndarray of shape (n_features,)
        Column means.

    x_std : ndarray of shape (n_features,)
        Column standard deviations (population standard deviation,
        i.e. ddof=0).

    Notes
    -----
    For sparse matrices, the variance is computed using the identity

        Var(X) = E[X^2] - E[X]^2

    where both expectations include the contribution of implicit zeros.
    The implementation iterates over CSC columns and operates directly on
    the sparse storage arrays to avoid densifying the matrix.
    """
    if isspmatrix(X):
        assert isspmatrix_csc(X)

        n, k = X.shape
        x_mean = np.zeros(k)
        x_std = np.zeros(k)

        # Total weight mass used in all weighted moment calculations.
        if weights is not None:
            sum_w = weights.sum()

        for j in range(k):

            # Nonzero values stored in column j.
            start = X.indptr[j]
            end = X.indptr[j + 1]
            X_j = X.data[start:end]

            if weights is None:
                # Mean including implicit zeros.
                x_mean[j] = X_j.sum() / n

                # Variance via E[X^2] - E[X]^2.
                x_std[j] = (X_j ** 2).sum() / n - x_mean[j] ** 2

            else:
                # Weights corresponding to the nonzero entries in column j.
                _w = weights[X.indices[start:end]]

                # Weighted mean including implicit zeros. The denominator is
                # the total weight mass, not the weight mass of nonzeros.
                x_mean[j] = (X_j * _w).sum() / sum_w

                # Weighted variance via E[X^2] - E[X]^2.
                x_std[j] = (_w * X_j ** 2).sum() / sum_w - x_mean[j] ** 2

        # Convert variances to standard deviations.
        x_std = np.sqrt(x_std)

        return x_mean, x_std

    else:
        # Dense mean calculation.
        x_mean = np.average(X, axis=0, weights=weights)

        if weights is None:
            return x_mean, X.std(axis=0, ddof=0)

        else:
            wts_sum = weights.sum()

            # Weighted variance via E[X^2] - E[X]^2.
            x_std = np.array([
                (weights * X[:, i] ** 2).sum() / wts_sum - x_mean[i] ** 2
                for i in range(X.shape[1])
            ]) ** 0.5

            return x_mean, x_std


def _get_penalties(alpha, l1_ratio, normalize, exog_scale_factor):
    """Build per-coordinate L1 and L2 penalty weights.

    When ``normalize=True``, each ``alpha_j`` is multiplied by
    ``exog_scale_factor[j]`` for the L1 term and by
    ``exog_scale_factor[j]**2`` for the L2 term, where ``exog_scale_factor``
    is the column standard deviation (default) or legacy demeaned L2 norm
    (see :func:`_get_normalizing_factors`).

    Args:
        alpha: Per-coordinate total penalty strengths.
        l1_ratio: Per-coordinate L1 shares.
        normalize: Whether to scale penalties by ``exog_scale_factor``.
        exog_scale_factor: Per-column scale from :func:`_get_normalizing_factors`.

    Returns:
        Tuple ``(l1_penalties, l2_penalties, penalty_func)`` where
        ``penalty_func`` evaluates the unscaled elastic-net penalty.
    """

    l1_penalties = np.array([a * l if not normalize else a * l * s
                             for a, l, s in zip(alpha, l1_ratio, exog_scale_factor)])
    l2_penalties = np.array([.5 * a * (1. - l) if not normalize
                             else .5 * a * (1. - l) * s ** 2
                             for a, l, s in zip(alpha, l1_ratio, exog_scale_factor)])
    penalty_func = lambda p: np.dot(l1_penalties, np.abs(p)) + np.dot(l2_penalties, p ** 2)

    return l1_penalties, l2_penalties, penalty_func


def _get_quadratic_form_and_ssr_func(X, y, l1_penalties, l2_penalties, weights=None, debug=False,
                                     ssr_quad_form=None, regularize_to_values=None):
    """Build quadratic-form pieces for the elastic-net least-squares objective_function_.

    Args:
        X: Design matrix.
        y: Response vector, possibly translated by ``regularize_to_values``.
        l1_penalties: Per-coordinate L1 penalty weights.
        l2_penalties: Per-coordinate L2 penalty weights.
        weights: Optional observation weights.
        debug: Whether to print timing diagnostics.
        ssr_quad_form: Optional precomputed quadratic form.
        regularize_to_values: Optional penalty target values.

    Returns:
        Tuple of cross-product arrays and an ``ElasticNetObjectiveFunction``."""

    _t = time.time()
    if debug:
        print('Converting Least Squares Objective to Quadratic Form...', end='')

    en_obj_func, _ = ElasticNetObjectiveFunction.build_elastic_net_objective_function(
        X, y, l1_penalties, l2_penalties, weights, debug, ssr_quad_form=ssr_quad_form,
        regularize_to_values=None
        # don't include reg to vals here, we handle this by differently
        # specifically, we subtract X.dot(reg_2_val) from y, solve elastic net on modified y
        # then adjust coefs at the end
    )

    if debug:
        print('%.2fs' % (time.time() - _t))

    return (
        en_obj_func.XtX, np.diag(en_obj_func.XtX), en_obj_func.Xty, en_obj_func.yty, en_obj_func.sum_y,
        en_obj_func.sum_X, en_obj_func.sum_w, en_obj_func
    )


def _format_input_data(X, y, weights, debug):
    """Normalize weights and coerce response data to a flat dense vector.

    Args:
        X: Design matrix, returned unchanged.
        y: Response vector, dense/sparse/Series.
        weights: Optional observation weights.
        debug: Reserved for diagnostic output.

    Returns:
        Tuple ``(X, y, weights)`` with ``y`` flattened and weights normalized to
        sum to the number of observations."""
    n = X.shape[0]
    if weights is not None:
        if isinstance(weights, Series):
            weights = weights.values
        if isspmatrix(weights):
            weights = weights.toarray().flatten()
        weights = weights.astype(float)
        weights *= n / weights.sum()

    if isspmatrix(y):
        y = y.toarray()
    elif isinstance(y, Series):
        y = y.values
    y = np.asarray(y).flatten()

    return X, y, weights


def _process_penalty_parameters(alpha, l1_ratio, n_params, regularize_to_values, positive):
    """Validate and broadcast penalty-related inputs to per-coordinate arrays.

    Args:
        alpha: Scalar or vector penalty strength.
        l1_ratio: Scalar or vector L1 mixing share.
        n_params: Number of coefficient coordinates.
        regularize_to_values: Optional scalar/vector penalty targets.
        positive: Boolean or vector non-negativity constraints.

    Returns:
        Tuple ``(alpha, l1_ratio, regularize_to_values, positive)`` as arrays."""
    # if regularize_to_values is not None:
    #     raise NotImplementedError("still working on `regularize_to_values`!")

    _check_penalties(alpha, l1_ratio)

    if isinstance(positive, bool):
        positive = np.array([positive] * n_params)
    if isinstance(alpha, (float, int)):
        alpha = np.array([alpha] * n_params)
    if isinstance(l1_ratio, (float, int)):
        l1_ratio = np.array([l1_ratio] * n_params)
    alpha, l1_ratio = np.array(alpha).flatten(), np.array(l1_ratio).flatten()

    if regularize_to_values is None:
        regularize_to_values = np.zeros(n_params)
    elif isinstance(regularize_to_values, (float, int)):
        regularize_to_values = np.full(n_params, regularize_to_values, dtype=float)
    else:
        regularize_to_values = np.array(regularize_to_values, dtype=float, copy=True).flatten()

    assert np.shape(regularize_to_values) == (n_params,)
    assert np.shape(alpha) == (n_params,)
    assert np.shape(l1_ratio) == (n_params,)
    assert np.shape(positive) == (n_params,)

    return alpha, l1_ratio, regularize_to_values, positive


def sparse_elastic_net_coordinate_descent_quad_form_setup(
        X, y, weights=None, start_coef=None, start_intercept=None, alpha=DEFAULT_EN_ALPHA, l1_ratio=DEFAULT_EN_L1_RATIO,
        regularize_to_values=None,
        max_iter=DEFAULT_EN_MAX_ITER, xtol=DEFAULT_EN_X_TOL, ftol=DEFAULT_EN_F_TOL, gtol=DEFAULT_EN_G_TOL,
        fit_intercept=DEFAULT_EN_FIT_INTERCEPT,
        normalize=DEFAULT_EN_NORMALIZE,
        positive=DEFAULT_EN_POSITIVE, debug=False, active_set=DEFAULT_EN_ACTIVE_SET,
        apply_scaling=DEFAULT_EN_APPLY_SCALING, prompt_user_for_more_iters=DEFAULT_EN_PROMPT_USER_FOR_MORE_ITERS,
        selection=DEFAULT_EN_SELECTION, seed=0, penalty_intensities=None, ssr_quad_form=None,
        one_dim_search_cadence=DEFAULT_EN_ONE_DIM_SEARCH_CADENCE,
        one_dim_search_multiplier=DEFAULT_EN_ONE_DIM_SEARCH_MULTIPLIER,
        one_dim_search_init_value=DEFAULT_EN_ONE_DIM_SEARCH_INIT_VAL
):
    """Prepare data and run coordinate descent for sparse elastic-net least squares.

    This function normalizes inputs, translates the response when regularizing
    toward nonzero target values, constructs penalty weights and quadratic forms,
    runs one or more coordinate-descent passes, restores results to the original
    coefficient scale, and computes fitted values/residual diagnostics.

    Args:
        X: Dense or sparse design matrix.
        y: Response vector.
        weights: Optional observation weights.
        start_coef: Optional starting coefficient vector.
        start_intercept: Optional starting intercept.
        alpha: Penalty strength(s).
        l1_ratio: L1/L2 mixing value(s).
        regularize_to_values: Optional coefficient target values.
        max_iter: Maximum coordinate-descent iterations.
        xtol: Coefficient-change convergence tolerance.
        ftol: Objective-change convergence tolerance.
        gtol: Subgradient convergence tolerance.
        fit_intercept: Whether to estimate an intercept.
        normalize: When True, scale each ``alpha_j`` by the standard deviation
            of column ``j`` (weighted population std when ``weights`` is set).
            Ignored if ``fit_intercept=False``.  Matches sklearn's recommended
            ``StandardScaler`` workflow; see package README.
        positive: Non-negativity constraints.
        debug: Whether to print timing/progress diagnostics.
        active_set: Whether to update only recently changing coordinates after a full pass.
        apply_scaling: Legacy optional coefficient rescaling (unrelated to
            ``normalize``); rarely needed with the std-dev penalty scaling.
        prompt_user_for_more_iters: Whether/how to prompt after max iterations.
        selection: Coordinate selection strategy.
        seed: Random seed for random coordinate order.
        penalty_intensities: Optional descending penalty scales for warm starts.
        ssr_quad_form: Optional precomputed SSR quadratic form.
        one_dim_search_cadence: Optional cadence for full-direction line search.
        one_dim_search_multiplier: Full-direction search expansion factor.
        one_dim_search_init_value: Initial full-direction search step.

    Returns:
        Dict containing coefficients, fitted values, residuals, objective_function_ pieces,
        convergence diagnostics, and solver settings."""
    tt = time.time()

    # if regularize_to_values is not None:
    #     raise Exception("`regularize_to_values` not supported yet!")

    if penalty_intensities is None:
        penalty_intensities = 1.

    if isinstance(penalty_intensities, (float, int)):
        penalty_intensities = [penalty_intensities]
    penalty_intensities = np.array(sorted(penalty_intensities)[::-1])

    if selection is None:
        selection = 'cyclic'
    else:
        selection = selection.lower()
        if selection not in EN_SELECTION_TYPES:
            raise Exception(f'`selection` arg must be in {EN_SELECTION_TYPES}!')

    n, n_params = X.shape

    alpha, l1_ratio, regularize_to_values, positive \
        = _process_penalty_parameters(alpha, l1_ratio, n_params, regularize_to_values, positive)

    is_weighted = weights is not None

    if start_coef is None:
        start_coef = np.zeros(n_params)
    beta_ = start_coef.copy()

    if start_intercept is None:
        start_intercept = 0.0
    intercept_ = start_intercept

    X, y, weights = _format_input_data(X, y, weights, debug)
    y_orig = y
    if isspmatrix(X):
        X_dot_r2v = X.dot(csc_matrix(regularize_to_values).reshape(-1, 1)).toarray().flatten()
    else:
        X_dot_r2v = X.dot(regularize_to_values)
    # Solve in translated coefficient space: beta_delta = beta - regularize_to_values.
    # This lets the same zero-centered soft-threshold update handle nonzero targets.
    y = y - X_dot_r2v

    _t = time.time()
    if debug:
        print('Getting scales and centers of regressors and adjusting penalties...', end='')
    _, w_mean_x, exog_scale_factor = _get_normalizing_factors(X, is_weighted, weights)

    l1_penalties, l2_penalties, _ = _get_penalties(alpha, l1_ratio, normalize, exog_scale_factor)
    if debug:
        print('%.2fs' % (time.time() - _t))

    XtX, XtX_diag, Xty, yty, sum_y, sum_X, sum_w, en_obj_func \
        = _get_quadratic_form_and_ssr_func(
            X, y, l1_penalties, l2_penalties, weights, debug, ssr_quad_form, regularize_to_values)

    for penalty_scale in penalty_intensities:
        en_obj_func.scale_penalties(penalty_scale)

        result = _sparse_elastic_net_coordinate_descent_quad_form(
            intercept_, beta_, en_obj_func, n, n_params,
            XtX, XtX_diag, Xty, sum_X, sum_y, sum_w,
            l1_penalties * penalty_scale, l2_penalties * penalty_scale, regularize_to_values,
            xtol=xtol, ftol=ftol, gtol=gtol, positive=positive,
            fit_intercept=fit_intercept, max_iter=max_iter,
            selection=selection, prompt_user_for_more_iters=prompt_user_for_more_iters, active_set=active_set,
            debug=debug, seed=seed, one_dim_search_cadence=one_dim_search_cadence,
            one_dim_search_multiplier=one_dim_search_multiplier,
            one_dim_search_init_value=one_dim_search_init_value,
        )

        intercept_, beta_ = result['intercept_'], result['coef_']

    if apply_scaling:
        # Optional sklearn-style correction for the way ridge curvature is scaled
        # when elastic-net coefficients are reported on the original design scale.
        beta_ *= (1. + alpha * (1. - l1_ratio))

    # Adjust results to reflect that we solved the translated problem above.
    beta_ += regularize_to_values
    en_obj_func.regularize_to_values = regularize_to_values
    if is_weighted:
        en_obj_func.yty = np.sum(weights * y_orig ** 2)
        en_obj_func.sum_y = np.sum(weights * y_orig)
    else:
        en_obj_func.yty = np.sum(y_orig ** 2)
        en_obj_func.sum_y = np.sum(y_orig)
    en_obj_func.Xty += en_obj_func.XtX.dot(regularize_to_values)

    y = y + X_dot_r2v
    if isspmatrix(X):
        fitted_values = X.dot(csc_matrix(beta_).reshape((-1, 1))).toarray().flatten() + intercept_
    else:
        fitted_values = (X.dot(beta_) + intercept_).flatten()
    resid = y - fitted_values

    rsquared = _get_rsquared(is_weighted, weights, resid, y)

    result.update({

        'coef_': beta_.copy().flatten(),
        'intercept_': intercept_,

        'x_error': result['x_error'],
        'f_error': result['f_error'],
        'g_error': result['g_error'],

        'fit_time': time.time() - tt,
        'fittedvalues': fitted_values,
        'resid': resid,
        'positive': positive,
        'fit_intercept': fit_intercept,
        'normalize': normalize,
        'alpha': alpha,
        'l1_ratio': l1_ratio,
        'rsquared': rsquared,
        'score': rsquared,
        'apply_scaling': apply_scaling,
        'l1_penalties': l1_penalties,
        'l2_penalties': l2_penalties,
        'regularize_to_values': regularize_to_values,
        'objective_function': en_obj_func,

        'solver_settings': {
            'seed': seed,
            'xtol': xtol,
            'gtol': gtol,
            'ftol': ftol,
            'max_iter': max_iter,
            'start_coef': start_coef,
            'start_intercept': start_intercept,
            'active_set': active_set,
            'selection': selection,
            'one_dim_search_cadence': one_dim_search_cadence,
            'one_dim_search_multiplier': one_dim_search_multiplier,
            'one_dim_search_init_value': one_dim_search_init_value,
        }
    })

    return result


def _get_rsquared(is_weighted, weights, resid, y):
    """Compute weighted or unweighted in-sample R-squared.

    Args:
        is_weighted: Whether to use weighted averages.
        weights: Observation weights when weighted.
        resid: Residual vector.
        y: Response vector.

    Returns:
        Scalar R-squared/score value."""
    if is_weighted:
        wtd_mean = np.average(y, weights=weights)
        return 1.0 - np.average(resid ** 2, weights=weights) / np.average((y - wtd_mean) ** 2, weights=weights)
    else:
        return 1.0 - (resid ** 2).mean() / ((y - y.mean()) ** 2).mean()


def _obj_func_1d(ss, intercept_direction, intercept_old, beta_direction, beta_old, en_obj_func, positive):
    """Evaluate the objective_function_ along a full coordinate-descent update direction.

    Args:
        ss: Scalar step along the direction.
        intercept_direction: Intercept movement direction.
        intercept_old: Starting intercept.
        beta_direction: Coefficient movement direction.
        beta_old: Starting coefficients.
        en_obj_func: Elastic-net objective_function_ function.
        positive: Non-negativity mask.

    Returns:
        Objective value at the candidate point, or ``np.inf`` if positivity is violated."""
    b = beta_old + beta_direction * ss
    if np.any(b[positive] < 0):
        return np.inf
    return en_obj_func(
        intercept_old + intercept_direction * ss,
        b
    )


def _do_one_dimensional_search(
        objective_itr, intercept_, intercept_old, beta_, beta_old, en_obj_func, positive,
        one_dim_search_init_value, one_dim_search_multiplier):
    """
    Look on either side of the coordinate-descent iteration to see if we can do better
    by going in that direction (across all the coordinates jointly), or backtracking
    """

    s_best = 0
    s_0 = one_dim_search_init_value
    f_best = objective_itr

    beta_direction = beta_ - beta_old
    intercept_direction = intercept_ - intercept_old

    # look at going right
    while True:
        f = _obj_func_1d(s_0, intercept_direction, intercept_, beta_direction, beta_, en_obj_func, positive)
        if f < f_best:
            f_best = f
            s_best = s_0
            s_0 *= one_dim_search_multiplier
        else:
            break

    # if going right didn't help, try left
    if s_best == 0:
        s_0 = -one_dim_search_init_value
        while True:
            f = _obj_func_1d(s_0, intercept_direction, intercept_, beta_direction, beta_, en_obj_func,
                             positive)
            if f < f_best:
                f_best = f
                s_best = s_0
                s_0 *= one_dim_search_multiplier
            else:
                break

    if s_best != 0:
        intercept_ = intercept_ + intercept_direction * s_best
        beta_ = beta_ + beta_direction * s_best
        objective_itr = f_best

    return intercept_, beta_, objective_itr, s_best


def _sparse_elastic_net_coordinate_descent_quad_form(
        intercept_, beta_, en_obj_func, nobs, n_params,
        XtX, XtX_diag, Xty, sum_X, sum_y, sum_w,
        l1_penalties, l2_penalties, regularize_to_values, xtol=DEFAULT_EN_X_TOL, gtol=DEFAULT_EN_G_TOL,
        ftol=DEFAULT_EN_F_TOL,
        positive=DEFAULT_EN_POSITIVE,
        fit_intercept=DEFAULT_EN_FIT_INTERCEPT, max_iter=DEFAULT_EN_MAX_ITER,
        selection=DEFAULT_EN_SELECTION, prompt_user_for_more_iters=DEFAULT_EN_PROMPT_USER_FOR_MORE_ITERS,
        active_set=DEFAULT_EN_ACTIVE_SET,
        debug=False, seed=0,
        one_dim_search_cadence=DEFAULT_EN_ONE_DIM_SEARCH_CADENCE,
        one_dim_search_multiplier=DEFAULT_EN_ONE_DIM_SEARCH_MULTIPLIER,
        one_dim_search_init_value=DEFAULT_EN_ONE_DIM_SEARCH_INIT_VAL,
):
    """Run coordinate descent on precomputed elastic-net quadratic-form sparse_terms.

    Args:
        intercept_: Starting intercept.
        beta_: Starting coefficient vector, updated in-place.
        en_obj_func: Objective helper for value/subgradient evaluation.
        nobs: Number of observations.
        n_params: Number of coefficient coordinates.
        XtX: Design cross-product matrix.
        XtX_diag: Diagonal of ``XtX``.
        Xty: Design-response cross-product vector.
        sum_X: Column sums of ``X``.
        sum_y: Sum of response values.
        sum_w: Sum of weights or observations.
        l1_penalties: Per-coordinate L1 penalties.
        l2_penalties: Per-coordinate L2 penalties.
        regularize_to_values: Penalty target vector.
        xtol: Coefficient-change tolerance.
        gtol: Subgradient tolerance.
        ftol: Objective-change tolerance.
        positive: Non-negativity mask.
        fit_intercept: Whether to update an intercept.
        max_iter: Maximum iterations.
        selection: Coordinate order strategy.
        prompt_user_for_more_iters: Whether/how to prompt after max iterations.
        active_set: Whether to restrict later passes to changing coordinates.
        debug: Whether to print iteration diagnostics.
        seed: Random seed.
        one_dim_search_cadence: Optional cadence for line search along full update.
        one_dim_search_multiplier: Line-search expansion factor.
        one_dim_search_init_value: Initial line-search step.

    Returns:
        Dict of final coordinates, objective_function_ pieces, convergence diagnostics, and
        solver metadata."""
    last_updated = -1
    beta_last = np.inf

    selection = str(selection).lower()

    param_index_rng = np.array(range(n_params))

    if debug:
        print('Beginning coordinate descent...')

    full_update = True

    ssr_last = en_obj_func.ssr_func(intercept_, beta_)
    penalty_last = en_obj_func.penalty_func(beta_)
    objective_last = ssr_last / (2 * nobs) + penalty_last

    _t = time.time()

    converged = False
    itr = 0
    rand = np.random.RandomState(seed)

    subgrad = None

    while True:

        if selection == EN_RANDOM:
            # Shuffle only the current active coordinate set and avoid immediately
            # repeating the coordinate updated at the end of the previous pass.
            param_index_rng = rand.permutation(param_index_rng)
            if param_index_rng[0] == last_updated:
                param_index_rng = np.hstack([param_index_rng[1:], param_index_rng[:1]])

        elif selection == EN_GREEDY:
            # Greedy selection visits coordinates with the largest current
            # optimality violations first.
            if subgrad is None:
                subgrad = en_obj_func.objective_subgrad_func(intercept_, beta_)
            abs_grad = np.abs(subgrad)
            param_index_rng = param_index_rng[np.argsort(abs_grad[param_index_rng])][::-1]

        if len(param_index_rng) == 0:
            print(itr)
            print(beta_)
            print(param_index_rng)
            print(abs_grad)
            raise Exception

        beta_old = beta_.copy()

        intercept_old = intercept_
        last_updated, beta_last, intercept_ = \
            sparse_coordinate_descent_update_iteration_quad_form(
                param_index_rng, nobs, XtX, XtX_diag, Xty, sum_X, sum_y, sum_w, beta_,
                l1_penalties, l2_penalties, regularize_to_values,
                last_updated, beta_last, positive, fit_intercept,
            )

        objective_itr = en_obj_func(intercept_, beta_)

        # Periodically evaluate the joint step implied by a full coordinate pass;
        # this can accelerate convergence when coordinates move in a consistent direction.
        do_1d_search = (not (
                one_dim_search_init_value is None or one_dim_search_cadence is None or one_dim_search_multiplier is None)
                        and itr % one_dim_search_cadence == 1)

        s_best = np.nan
        if do_1d_search:
            intercept_, beta_, objective_itr, s_best = _do_one_dimensional_search(
                objective_itr, intercept_,
                intercept_old,
                beta_,
                beta_old,
                en_obj_func, positive,
                one_dim_search_init_value, one_dim_search_multiplier)

        penalty_itr = en_obj_func.penalty_func(beta_)
        ssr_itr = (objective_itr - penalty_itr) * (2. * nobs)

        objective_chg = objective_last - objective_itr
        f_error = objective_chg / max(1, objective_itr)

        with np.errstate(divide='ignore', invalid='ignore'):
            diff = np.where(np.abs(beta_old) < 1, np.abs(beta_old - beta_), np.abs(beta_ / beta_old - 1))
        x_error = np.max(diff)

        if debug:
            iter_info = [
                ({'name': 'iter', 'value': itr, 'format': '%6d', 'len': 6}),
                ({'name': 'Obj', 'value': objective_itr, 'format': '%15.2e', 'len': 15}),
                ({'name': '|dx|', 'value': x_error, 'format': '%12.2e', 'len': 12}),
                ({'name': 'SSR', 'value': ssr_itr, 'format': '%15.2e', 'len': 15}),
                ({'name': 'Penalty', 'value': penalty_itr, 'format': '%15.2e', 'len': 15}),
                ({'name': '|dObj/Obj|', 'value': objective_chg / objective_last, 'format': '%15.2e', 'len': 15}),
                ({'name': 'Full Update', 'value': full_update, 'format': '%13s', 'len': 13}),
                ({'name': '# Active', 'value': len(param_index_rng), 'format': '%10d', 'len': 10}),
                ({'name': '# Nonzero', 'value': np.count_nonzero(beta_), 'format': '%10d', 'len': 10}),
                ({'name': 's_best', 'value': s_best, 'format': '%10.2e', 'len': 10}),
                ({'name': 'Time', 'value': time.time() - _t, 'format': '%9.2fs', 'len': 10}),
            ]
            if itr == 0:
                print_iter_info(iter_info, is_header=True)
            print_iter_info(iter_info)

        message, converged, subgrad, g_error = _check_convergence(
            x_error, f_error, itr, full_update, do_1d_search, intercept_, beta_, en_obj_func, xtol, ftol, gtol, positive)
        if converged:
            break

        ssr_last = ssr_itr
        penalty_last = penalty_itr
        objective_last = objective_itr

        itr += 1
        if itr == max_iter:
            incremental_iters = user_prompt_for_more_iters_method(
                f"\n\t[iteration = {'%d' % itr}, |dx| = {'%.2e' % x_error}, "
                f"|dObjective/Objective| = {'%.2e' % (objective_chg / objective_last)}]",
                prompt_user_for_more_iters
            )
            if incremental_iters == 0:
                break
            else:
                max_iter += incremental_iters

        # Active-set mode focuses on coordinates that still moved materially, but
        # returns to a full pass once the active set has stabilized.
        full_update = (not active_set) or x_error < xtol
        if full_update:
            param_index_rng = np.arange(n_params)
        else:
            param_index_rng = np.arange(n_params)[diff > xtol]

    if debug:
        print_iter_info(iter_info, is_footer=True)
        print('\nCoordinate descent complete! {iters=%d, error=%.2e, time=%.2fs}' % (
            itr + 1, x_error, time.time() - _t))
        print(f"Converged = {converged}\n")

    if not converged:
        subgrad = en_obj_func.objective_subgrad_func(intercept_, beta_)
        g_error = np.abs(subgrad).max()

    return {
        'intercept_': intercept_,
        'coef_': beta_.copy().flatten(),
        'x_error': x_error,
        'f_error': f_error,
        'g_error': g_error,
        'xtol': xtol,
        'ftol': ftol,
        'gtol': gtol,
        'converged': converged,
        'iters': itr,
        'max_iter': max_iter,
        'positive': positive,
        'fit_intercept': fit_intercept,
        'active_set': active_set,
        'l1_penalties': l1_penalties,
        'l2_penalties': l2_penalties,
        'objective_function': en_obj_func,
        'penalty': penalty_last,
        'ssr': ssr_last,
        'objective_function_': objective_last,
        'message': message,
        'subgrad': subgrad,
    }


def _check_convergence(x_error, f_error, itr, full_update, do_1d_search, intercept_, beta_, en_obj_func,
                       xtol, ftol, gtol, positive):
    """Check elastic-net coordinate-descent convergence criteria.

    Args:
        x_error: Maximum relative/absolute coefficient change.
        f_error: Relative objective_function_ change.
        itr: Current iteration number.
        full_update: Whether the last pass covered all coordinates.
        do_1d_search: Whether line search was attempted this iteration.
        intercept_: Current intercept.
        beta_: Current coefficients.
        en_obj_func: Objective helper for subgradients.
        xtol: Coefficient-change tolerance.
        ftol: Objective-change tolerance.
        gtol: Subgradient tolerance.
        positive: Non-negativity mask.

    Returns:
        Tuple ``(message, converged, subgrad, g_error)``."""

    subgrad = en_obj_func.objective_subgrad_func(intercept_, beta_)
    subgrad = np.where(positive & (subgrad > 0), 0.0, subgrad)
    g_error = np.abs(subgrad).max()

    message = f'Did not converge! ({x_error=:.1e}, {g_error=:.1e}, {f_error=:.1e})'
    converged = False

    if itr > 1 and full_update:
        if x_error < xtol or f_error < ftol:
            if g_error < gtol:
                converged = True
                message = f'Converged in {itr} iterations: '
                if x_error < xtol:
                    message += f'\n\tx_error = {"%.1e" % x_error} < {"%.1e" % xtol} = xtol, '
                if f_error < ftol:
                    message += f'\n\tf_error = {"%.1e" % f_error} < {"%.1e" % ftol} = ftol, '
                message += '\n\t' + f'g_error = {"%.1e" % g_error} < {"%.1e" % gtol} = gtol.'
    return message, converged, subgrad, g_error


def _relax_penalty(alpha, coefs, relaxation_parameter):
    "Relax the penalties on the subset of selected coefficients to allow a more unbiased fit"
    if isinstance(alpha, (float, int)):
        alpha = np.full(len(coefs), alpha, dtype=float)
    alpha_relaxed = np.array([a * relaxation_parameter if abs(p) > 0 else max(a, 1e100)
                              for a, p in zip(alpha, coefs)])
    return alpha_relaxed


def _elastic_net_internal(
        X, y, fit_intercept=DEFAULT_EN_FIT_INTERCEPT, normalize=DEFAULT_EN_NORMALIZE, alpha=DEFAULT_EN_ALPHA,
        l1_ratio=DEFAULT_EN_L1_RATIO, max_iter=DEFAULT_EN_MAX_ITER,
        xtol=DEFAULT_EN_X_TOL, ftol=DEFAULT_EN_F_TOL, gtol=DEFAULT_EN_G_TOL,
        positive=DEFAULT_EN_POSITIVE, weights=None, debug=False, active_set=DEFAULT_EN_ACTIVE_SET,
        apply_scaling=DEFAULT_EN_APPLY_SCALING, prompt_user_for_more_iters=DEFAULT_EN_PROMPT_USER_FOR_MORE_ITERS,
        start_coef=None, start_intercept=None, selection=DEFAULT_EN_SELECTION, seed=0, penalty_intensities=None,
        ssr_quad_form=None, regularize_to_values=None, relaxation_parameter=DEFAULT_EN_RELAXATION_PARAMETER,
        one_dim_search_cadence=DEFAULT_EN_ONE_DIM_SEARCH_CADENCE,
        one_dim_search_multiplier=DEFAULT_EN_ONE_DIM_SEARCH_MULTIPLIER,
        one_dim_search_init_value=DEFAULT_EN_ONE_DIM_SEARCH_INIT_VAL,
):
    """Top-level internal elastic-net solver used by model APIs.

    Validates penalties, disables incompatible normalization, runs the coordinate
    descent setup, optionally performs a relaxed second pass on selected variables,
    and packages fitted parameters plus raw cost.

    Args:
        X: Dense or sparse design matrix.
        y: Response vector.
        fit_intercept: Whether to fit an intercept.
        normalize: When True, scale penalties by predictor standard deviations.
        alpha: Penalty strength(s).
        l1_ratio: L1/L2 mixing value(s).
        max_iter: Maximum iterations.
        xtol: Coefficient-change tolerance.
        ftol: Objective-change tolerance.
        gtol: Subgradient tolerance.
        positive: Non-negativity constraints.
        weights: Optional observation weights.
        debug: Whether to print diagnostics.
        active_set: Whether to use active-set coordinate updates.
        apply_scaling: Legacy optional coefficient rescaling after fit.
        prompt_user_for_more_iters: Whether/how to prompt after max iterations.
        start_coef: Optional starting coefficients.
        start_intercept: Optional starting intercept.
        selection: Coordinate selection strategy.
        seed: Random seed.
        penalty_intensities: Optional warm-start penalty scales.
        ssr_quad_form: Optional precomputed quadratic form.
        regularize_to_values: Optional coefficient targets.
        relaxation_parameter: Optional second-pass penalty relaxation factor.
        one_dim_search_cadence: Optional full-direction search cadence.
        one_dim_search_multiplier: Full-direction search expansion factor.
        one_dim_search_init_value: Initial full-direction search step.

    Returns:
        Tuple ``(fit_dict, params, cost)``."""
    _check_penalties(alpha, l1_ratio)
    assert relaxation_parameter is None or 0.0 <= relaxation_parameter < 1.0

    is_weighted = weights is not None

    if normalize and not fit_intercept:
        if debug:
            warnings.warn("`normalize=True` ignored because `fit_intercept=False`")
        normalize = False

    fit_dict = sparse_elastic_net_coordinate_descent_quad_form_setup(
        X, y, fit_intercept=fit_intercept, normalize=normalize, positive=positive, alpha=alpha, l1_ratio=l1_ratio,
        weights=weights, max_iter=max_iter, xtol=xtol, gtol=gtol, ftol=ftol,
        debug=debug, active_set=active_set, apply_scaling=apply_scaling,
        prompt_user_for_more_iters=prompt_user_for_more_iters, start_coef=start_coef, start_intercept=start_intercept,
        seed=seed, selection=selection, penalty_intensities=penalty_intensities, ssr_quad_form=ssr_quad_form,
        regularize_to_values=regularize_to_values,
        one_dim_search_cadence=one_dim_search_cadence,
        one_dim_search_multiplier=one_dim_search_multiplier,
        one_dim_search_init_value=one_dim_search_init_value,
    )

    if relaxation_parameter is not None:
        alpha_relaxed = _relax_penalty(alpha, fit_dict['coef_'], relaxation_parameter)

        # The relaxed pass lowers penalties on selected coefficients and assigns
        # huge penalties to zeros, approximating post-selection shrinkage reduction.
        fit_dict = sparse_elastic_net_coordinate_descent_quad_form_setup(
            X, y, fit_intercept=fit_intercept, normalize=normalize, positive=positive,
            alpha=alpha_relaxed, l1_ratio=l1_ratio,
            weights=weights, max_iter=max_iter, xtol=xtol, gtol=gtol, ftol=ftol,
            debug=debug, active_set=active_set,
            apply_scaling=apply_scaling,
            prompt_user_for_more_iters=prompt_user_for_more_iters, start_coef=fit_dict['coef_'],
            start_intercept=fit_dict['intercept_'],
            seed=seed, selection=selection, penalty_intensities=penalty_intensities, ssr_quad_form=ssr_quad_form,
            regularize_to_values=regularize_to_values,
            one_dim_search_cadence=one_dim_search_cadence,
            one_dim_search_multiplier=one_dim_search_multiplier,
            one_dim_search_init_value=one_dim_search_init_value,
        )

    params = np.array(([fit_dict['intercept_']] if fit_intercept else []) + list(fit_dict['coef_']))
    cost = np.sum(weights * fit_dict['resid'] ** 2) / 2 if is_weighted else sum(fit_dict['resid'] ** 2) / 2

    return fit_dict, params, cost


# if __name__ == '__main__':
#     from sklearn.preprocessing import StandardScaler
#
#     np.random.seed(0)
#     X = np.random.randn(10000, 3)
#     w = np.exp(np.random.randn(10_000))
#     X[:, 0] *= 3
#     X[:, 0] += 5
#
#     res = _get_normalizing_factors(X, False, None)
#     print(np.sqrt(((X[:, 0] - 5) ** 2).sum()))
#     print(res)
#
#     Xstd = StandardScaler().fit_transform(X, sample_weight=w)
#     print(np.average(Xstd, axis=0))
#     print(np.average(Xstd, weights=w, axis=0))
