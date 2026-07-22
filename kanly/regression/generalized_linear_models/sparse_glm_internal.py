from __future__ import absolute_import, print_function

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy.sparse import csc_matrix, isspmatrix, diags, SparseEfficiencyWarning

from kanly.regression.generalized_linear_models.constants import (
    METHOD_IRLS, METHOD_COORD_DESC, METHOD_COORD_DESCENT_1_ITER, DEFAULT_GLM_MAX_ITER, MAX_LINE_SEARCH,
    DEFAULT_GLM_L1_RATIO, SHRINK_INTERCEPT, DEFAULT_GLM_FAMILY, DEFAULT_GLM_TOL, DEFAULT_GLM_ALPHA,
    DEFAULT_GLM_RESIDUAL_INCLUSION, DEFAULT_GLM_FORCE_IV_PROJECTION, LINE_SEARCH_SHRINK,
    DEFAULT_GLM_RESIDUAL_INCLUSION_ORDER, DEFAULT_GLM_PROMPT_USER_FOR_MORE_ITERS)
from kanly.regression.generalized_linear_models.families import (Binomial, Poisson, _get_family_and_link, Gaussian,
                                                                 NegativeBinomial)
from kanly.regression.linear_models.sparse_iv_first_stage2 import iv_first_stage2
from kanly.utils.linalg_utils import get_matrix_inverse_internal, csc_matrix_by_column_array_broadcast
from kanly.utils.user_prompt_for_more_iters import user_prompt_for_more_iters_method

DEBUG_INTERNAL = False


class GLMRawFitData(object):
    """
    Container for raw generalized linear model fit results.

    This object stores the low-level output from ``glm_internal``: fitted
    parameters, likelihood and deviance statistics, residuals, convergence
    information, optimizer metadata, and any instrumental-variable projection
    information.

    Notes
    -----
    This class intentionally does not compute derived statistics. It is a
    simple structured result object used to move fit state to higher-level GLM
    APIs.
    """

    def __init__(self, params, exog, fit_time, pearson_chi2, deviance, llf, llnull, scale, resid, irls_weights,
                 normalized_cov_params, edf, g_prime,
                 lin_pred, endog_predicted, opt_method, converged, max_iter, num_iter, abs_error, rel_error,
                 instrument_params, exog_col_map, family, link, l2s, alpha, l1_ratio, normalize, penalize_scale,
                 df_resid, df_model, convergence_path):
        """
        Initialize raw GLM fit data.

        Parameters
        ----------
        params : numpy.ndarray
            Estimated model parameters. Includes an intercept when one is fit
            and not already represented by the first design-matrix column.
        exog : scipy.sparse.csc_matrix
            Design matrix used during fitting. This may be the instrumented
            design matrix when instruments are supplied.
        fit_time : float
            Total elapsed fit time, in seconds.
        pearson_chi2 : float
            Pearson chi-squared statistic for the fitted model.
        deviance : float
            Model deviance.
        llf : float
            Final fitted log-likelihood.
        llnull : float
            Null-model log-likelihood.
        scale : float
            Estimated scale parameter.
        resid : numpy.ndarray
            Response residuals, computed as ``endog - endog_predicted``.
        irls_weights : numpy.ndarray
            IRLS working weights from the final iteration.
        normalized_cov_params :
        edf : effective degrees of freedom
        g_prime : numpy.ndarray or float
            Link derivative evaluated at the fitted mean.
        lin_pred : numpy.ndarray
            Final linear predictor.
        endog_predicted : numpy.ndarray
            Final fitted mean response.
        opt_method : str
            Optimization method used.
        converged : bool
            Whether the optimizer met the convergence criterion.
        max_iter : int
            Final maximum number of iterations allowed.
        num_iter : int
            Number of iterations actually run.
        abs_error : float
            Final maximum absolute parameter change.
        rel_error : float
            Final maximum relative parameter change.
        instrument_params : numpy.ndarray or None
            First-stage instrument parameters, if instrumental variables were
            used.
        exog_col_map : list
            Mapping from the transformed design matrix columns back to original
            exogenous columns.
        family : object
            GLM family instance.
        link : object
            Link function instance.
        l2s : numpy.ndarray
            Per-parameter L2 penalty weights.
        alpha : float
            Overall regularization strength.
        l1_ratio : float
            Fraction of regularization assigned to the L1 penalty.
        normalize : bool
            Whether predictor scales were used to normalize penalties.
        penalize_scale : bool
            Whether penalties were multiplied by the current scale estimate.
        df_resid : int
            Residual degrees of freedom.
        df_model : int
            Model degrees of freedom.
        convergence_path : list[dict] or None
            Per-iteration diagnostics when requested; otherwise ``None``.
        """
        self.params = params
        self.exog = exog
        self.fit_time = fit_time
        self.pearson_chi2 = pearson_chi2
        self.deviance = deviance
        self.llf = llf
        self.llnull = llnull
        self.scale = scale
        self.resid = resid
        self.irls_weights = irls_weights
        self.normalized_cov_params = normalized_cov_params
        self.edf = edf
        self.g_prime = g_prime
        self.lin_pred = lin_pred
        self.endog_predicted = endog_predicted
        self.opt_method = opt_method
        self.converged = converged
        self.max_iter = max_iter
        self.num_iter = num_iter
        self.abs_error = abs_error
        self.rel_error = rel_error
        self.instrument_params = instrument_params
        self.exog_col_map = exog_col_map
        self.family = family
        self.link = link
        self.l2s = l2s
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.normalize = normalize
        self.penalize_scale = penalize_scale
        self.df_resid = df_resid
        self.df_model = df_model
        self.convergence_path = convergence_path


def wtd_std(x, weights=None):
    """
    Compute a weighted or unweighted standard deviation.

    Parameters
    ----------
    x : array-like
        Values whose standard deviation should be computed.
    weights : array-like or None, optional
        Observation weights. When omitted, this returns ``np.std(x)``.

    Returns
    -------
    float
        Weighted or unweighted standard deviation.
    """
    if weights is not None:
        # Weighted mean is computed explicitly so the variance can use the same
        # weights and denominator.
        wtd_mean = (x * weights).sum() / weights.sum()
        return np.sqrt((weights * (x - wtd_mean) ** 2).sum() / weights.sum())
    else:
        return np.std(x)


@njit(cache=True)
def soft_threshold(z, penalty):
    """
    Apply scalar soft-thresholding for an L1-penalized coordinate update.

    Parameters
    ----------
    z : float
        Unpenalized coordinate-update value.
    penalty : float
        L1 penalty threshold.

    Returns
    -------
    float
        The thresholded value. Very small values are rounded to exactly zero.
    """
    v = np.sign(z) * max(abs(z) - penalty, 0)

    # Avoid retaining tiny numerical noise as a nonzero coefficient.
    if abs(v) < 1e-5:
        v = 0
    return v


#@njit
def _update_coord_descent2(
        k, N, intercept, params, lin_pred, endog, exog_data, exog_indptr, exog_indices, fit_intercept,
        var_weights, l1s, l2s, regularize_to_values, endog_predicted, irls_weights, g_prime, scale,
        penalize_scale):
    """
    Update one coefficient during coordinate-descent optimization.

    The sparse CSC column for coordinate ``k`` is reconstructed, the
    elastic-net penalized update is applied, and the linear predictor is updated
    in place to reflect the coefficient change. The intercept is then updated
    separately when requested.

    Parameters
    ----------
    k : int
        Column index to update.
    N : int
        Number of observations.
    intercept : float
        Current intercept estimate.
    params : numpy.ndarray
        Current coefficient vector. ``params[k]`` is modified in place.
    lin_pred : numpy.ndarray
        Current linear predictor. Modified in place.
    endog : numpy.ndarray
        Observed response values.
    exog_data : numpy.ndarray
        Nonzero CSC matrix values.
    exog_indptr : numpy.ndarray
        CSC column pointer array.
    exog_indices : numpy.ndarray
        CSC row index array.
    fit_intercept : bool
        Whether to update an intercept.
    var_weights : numpy.ndarray
        Observation variance weights.
    l1s : numpy.ndarray
        Per-coordinate L1 penalties.
    l2s : numpy.ndarray
        Per-coordinate L2 penalties.
    regularize_to_values : numpy.ndarray
        Per-coordinate targets. Both L1 and L2 penalties are applied to
        ``params - regularize_to_values``.
    endog_predicted : numpy.ndarray
        Current fitted mean response.
    irls_weights : numpy.ndarray
        Current IRLS weights.
    g_prime : numpy.ndarray or float
        Link derivative evaluated at the fitted mean.
    scale : float
        Current scale estimate.
    penalize_scale : bool
        Whether to multiply penalties by ``scale``.

    Returns
    -------
    tuple
        Updated ``intercept``, ``endog_predicted``, ``irls_weights``, and
        ``g_prime``.
    """
    # Reconstruct the k-th sparse design column as a dense vector. This is done
    # one column at a time to keep the coordinate update simple.
    xk = np.zeros(N)
    xk[exog_indices[exog_indptr[k]:exog_indptr[k + 1]]] \
        = exog_data[exog_indptr[k]:exog_indptr[k + 1]]

    # Weighted squared column norm, using IRLS weights. This is the local
    # curvature term for the coordinate update.
    w_mean_sq = (var_weights * irls_weights * (xk ** 2)).mean()
    scale_factor = 1.0 if not penalize_scale else scale

    # Solve in the displacement delta = beta - target. Centering the local
    # quadratic approximation on the target lets the existing soft-threshold
    # operator handle both L1 and L2 penalties around a nonzero value.
    beta_old = params[k]
    target = regularize_to_values[k]
    unpenalized_numerator = (
            (var_weights * irls_weights * xk * (endog - endog_predicted) * g_prime).mean()
            + beta_old * w_mean_sq
    )
    delta_new = (
            soft_threshold(
                unpenalized_numerator - target * w_mean_sq,
                scale_factor * l1s[k])
            / (w_mean_sq + 2 * scale_factor * l2s[k])
    )
    params[k] = target + delta_new

    # Update the linear predictor incrementally instead of recomputing X @ beta.
    lin_pred += xk * (params[k] - beta_old)

    if fit_intercept:
        # Intercept update uses the weighted mean residual on the working scale.
        b0_incr = (var_weights * irls_weights * (endog - endog_predicted) * g_prime).sum() / (var_weights * irls_weights).sum()
        intercept += b0_incr
        lin_pred += b0_incr

    else:
        # If no intercept is being fit, gradually shrink any carried intercept
        # contribution out of the linear predictor.
        lin_pred -= intercept * (1 - SHRINK_INTERCEPT)
        intercept *= SHRINK_INTERCEPT

    return intercept, endog_predicted, irls_weights, g_prime


def _get_working_endog_predicted_and_irls_weights(lin_pred, family, link):
    """
    Compute fitted means, IRLS weights, and link derivatives.

    Parameters
    ----------
    lin_pred : numpy.ndarray
        Current linear predictor.
    family : object
        GLM family object providing a variance function and canonical link.
    link : object
        Link object providing inverse-link and derivative methods.

    Returns
    -------
    tuple
        ``endog_predicted``, ``irls_weights``, and ``g_prime``.
    """
    # Transform the linear predictor onto the response scale.
    endog_predicted = link.inverse_link(lin_pred)

    # The GLM variance is evaluated at the fitted mean.
    _var = family.variance(endog_predicted)

    if link.name() == family.canonical_link().name():
        # Canonical links simplify the IRLS weights.
        irls_weights = _var
        g_prime = 1.0 / irls_weights
    else:
        # Noncanonical links require the derivative of the link function.
        g_prime = link.deriv(endog_predicted)
        irls_weights = 1.0 / (g_prime ** 2 * _var)

    return endog_predicted, irls_weights, g_prime


def _get_penalty(nobs, params, l1s, l2s, regularize_to_values, scale, penalize_scale):
    """
    Compute the elastic-net penalty contribution to the objective_function.

    Parameters
    ----------
    nobs : int
        Number of observations.
    params : numpy.ndarray
        Current model parameters.
    l1s : numpy.ndarray
        Per-parameter L1 penalty weights.
    l2s : numpy.ndarray
        Per-parameter L2 penalty weights.
    regularize_to_values : numpy.ndarray
        Per-parameter targets for the centered elastic-net penalty.
    scale : float
        Current scale estimate.
    penalize_scale : bool
        Whether to multiply penalties by ``scale``.

    Returns
    -------
    float
        Penalty term added to the negative log-likelihood objective_function.
    """
    scale_factor = scale if penalize_scale else 1.0
    params_delta = params - regularize_to_values
    return scale_factor * nobs * (
            np.dot(l1s, np.abs(params_delta))
            + np.dot(l2s, np.power(params_delta, 2)))


def _get_matrix_penalty(params, L2_penalty_matrix, regularize_to_values):
    """Evaluate a centered quadratic penalty represented by a matrix.

    The factor of one half matches the IRLS normal equations
    ``(X'WX + P) beta = X'Wz + P r``.

    Parameters
    ----------
    params : numpy.ndarray
        Current coefficient vector.
    L2_penalty_matrix : numpy.ndarray or None
        Dense penalty matrix ``P``. ``None`` disables matrix penalization.
    regularize_to_values : numpy.ndarray
        Target vector ``r``.

    Returns
    -------
    float
        ``0.5 * (params - r)' P (params - r)``.
    """
    if L2_penalty_matrix is None:
        return 0.0
    params_delta = params - regularize_to_values
    return 0.5 * params_delta.dot(L2_penalty_matrix).dot(params_delta)


def _coerce_regularize_to_values(regularize_to_values, k_exog, original_k_exog=None):
    """Return a flat target vector aligned with the final design matrix.

    Scalar targets apply to the original user-supplied columns. Columns added
    internally for IV residual inclusion receive zero targets.

    Parameters
    ----------
    regularize_to_values : scalar, array-like, or None
        User-supplied penalty targets.
    k_exog : int
        Number of columns in the final fitting design.
    original_k_exog : int or None
        Number of user-supplied columns before internal design expansion.

    Returns
    -------
    numpy.ndarray
        Length-``k_exog`` target vector.
    """
    if original_k_exog is None:
        original_k_exog = k_exog

    if regularize_to_values is None:
        targets = np.zeros(original_k_exog, dtype=float)
    elif np.isscalar(regularize_to_values):
        targets = np.full(original_k_exog, regularize_to_values, dtype=float)
    else:
        targets = np.asarray(regularize_to_values, dtype=float).reshape(-1)

    if targets.size == original_k_exog and k_exog > original_k_exog:
        targets = np.pad(targets, (0, k_exog - original_k_exog))
    elif targets.size != k_exog:
        raise ValueError(
            "`regularize_to_values` must be a scalar or have one value per "
            f"coefficient; got {targets.size} values for {k_exog} coefficients."
        )

    return targets


def _get_starting_params(start_params, fit_intercept, first_column_constant,
                         family, link, endog, var_weights, nobs, k_exog, exog,
                         pick_default_start, debug=False):
    """
    Build initial intercept, coefficient, and linear-predictor values.

    If user-provided starting parameters are available, this function can
    compare their initial log-likelihood against an intercept-only default and
    choose the better starting point.

    Parameters
    ----------
    start_params : array-like or None
        User-supplied initial parameters.
    fit_intercept : bool
        Whether an intercept is fit.
    first_column_constant : bool
        Whether the first design-matrix column is an explicit constant.
    family : object
        GLM family object.
    link : object
        Link function object.
    endog : numpy.ndarray
        Observed response values.
    var_weights : numpy.ndarray
        Observation variance weights.
    nobs : int
        Number of observations.
    k_exog : int
        Number of exogenous columns.
    exog : scipy.sparse.csc_matrix
        Design matrix.
    pick_default_start : bool
        Whether to use the default start if it has better likelihood than the
        user-supplied start.
    debug : bool, optional
        Whether to print starting-likelihood diagnostics.

    Returns
    -------
    tuple
        Initial ``intercept``, ``params``, and ``lin_pred``.
    """
    # Family-specific intercept-only default, typically based on the weighted
    # mean response transformed through the link.
    intercept0 = family.get_starting_intercept(endog, var_weights=var_weights, link=link)

    if start_params is None:
        params = np.zeros(k_exog)
        if fit_intercept and first_column_constant:
            params[0] = intercept0
        return intercept0, np.zeros(k_exog), np.ones(nobs) * intercept0

    # Find likelihood under the default intercept-only starting point.
    endog_predicted0 = link.inverse_link(np.asarray([intercept0] * nobs))
    theta0 = family.b_deriv_inv(endog_predicted0)
    llf0 = family.log_likelihood(endog, theta0, var_weights=var_weights)

    # Find likelihood under the user-supplied starting point.
    params_user = np.array(start_params, dtype=float)

    if (fit_intercept and first_column_constant) or (not fit_intercept):
        lin_pred_user = exog.dot(csc_matrix(params_user.reshape(-1, 1))).toarray().flatten()
    else:
        # When the intercept is not represented in exog, peel it off from the
        # user parameter vector before multiplying by X.
        params_user = params_user[1:]
        lin_pred_user = params_user[0] + exog.dot(csc_matrix(params_user.reshape(-1, 1))).toarray().ravel()

    intercept_user = params_user[0] if fit_intercept else 0.0

    if not pick_default_start:
        return intercept_user, params_user, lin_pred_user

    endog_predicted_user = link.inverse_link(lin_pred_user)
    theta_user = family.b_deriv_inv(endog_predicted_user)
    llf_user = family.log_likelihood(endog, theta_user, var_weights=var_weights)

    # Prefer the starting point with the better initial likelihood.
    use_default = llf0 > llf_user
    if debug:
        print('\nllf under defaults is %.6e, llf under user choice is %.6e, reverting to default = %s' % (
            llf0 / nobs, llf_user / nobs, use_default
        ))

    if use_default:
        return intercept0, np.zeros(k_exog), np.ones(nobs) * intercept0
    else:
        return intercept_user, params_user, lin_pred_user


def _update_irls_iter(var_weights, lin_pred, family, link, endog, exog, L2_penalty_matrix, penalty_rhs):
    """
    Perform one iteratively reweighted least squares update.

    Parameters
    ----------
    var_weights : numpy.ndarray
        Observation variance weights.
    lin_pred : numpy.ndarray
        Current linear predictor.
    family : object
        GLM family object.
    link : object
        Link function object.
    endog : numpy.ndarray
        Observed response values.
    exog : array-like or scipy.sparse matrix
        Design matrix.
    L2_penalty_matrix : array-like or scipy.sparse matrix or None, optional
        Penalty matrix added to the IRLS normal equations each iteration:
        ``X'WX + L2_penalty_matrix`` before inversion. It may be a general ridge
        matrix or the block-diagonal roughness matrix built for a GAM. ``None``
        disables matrix penalization.
    penalty_rhs : numpy.ndarray or None, optional
        Right-hand-side penalty contribution ``L2_penalty_matrix @ r``, where
        ``r`` is the target coefficient vector.

    Returns
    -------
    tuple
        Updated parameters, fitted means, IRLS weights, link derivatives,
        linear predictor, and working response.
    """
    if not isspmatrix(exog):
        exog = csc_matrix(exog)

    endog_predicted, irls_weights, g_prime \
        = _get_working_endog_predicted_and_irls_weights(lin_pred, family, link)

    # Combine user variance weights with IRLS weights.
    wts = var_weights * irls_weights

    # Working response for the weighted least-squares approximation.
    working_response = csc_matrix((lin_pred + (endog - endog_predicted) * g_prime).reshape((-1, 1)))

    # Form X'WX by scaling rows of X by sqrt(weights), then multiplying.
    # W = diags(wts)
    exog_w = csc_matrix_by_column_array_broadcast(exog, wts)
    XpX = exog.transpose().dot(exog_w)

    # Add the general ridge/GAM matrix to the penalized WLS normal equations.
    if L2_penalty_matrix is not None:
        XpX_penalized = XpX + L2_penalty_matrix
    else:
        XpX_penalized = XpX

    with warnings.catch_warnings():
        # scipy may warn about sparse formats or solve efficiency; the code
        # converts through dense inversion intentionally via project utilities.
        warnings.filterwarnings("ignore", message="splu requires CSC matrix format")
        warnings.filterwarnings("ignore", message="spsolve is more efficient when sparse b ")
        warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
        ncp = get_matrix_inverse_internal(XpX_penalized.toarray())
        # ncp = csc_matrix(ncp.reshape((exog.shape[1], exog.shape[1])))

    # Weighted least-squares normal-equation update.
    rhs = exog_w.transpose().dot(working_response).toarray()
    if penalty_rhs is not None:
        rhs += penalty_rhs

    params_new = ncp.dot(rhs)
    # del exog_w

    #params_new = csc_matrix(params_new)

    #lin_pred = exog.dot(params_new)#.toarray().ravel()
    #params_new = params_new.toarray().ravel()

    lin_pred = exog.dot(params_new).flatten()
    params_new = params_new.flatten()

    return params_new, endog_predicted, irls_weights, g_prime, lin_pred, working_response, XpX, ncp


def _get_opt_method(method, alpha, l1_ratio):
    """
    Resolve and validate the optimization method.

    Parameters
    ----------
    method : str or None
        Requested optimization method. If ``None``, a default is selected based
        on whether regularization is active.
    alpha : float
        Regularization strength.
    l1_ratio : float
        Fraction of regularization assigned to the L1 penalty. A pure L2
        penalty is compatible with either IRLS or coordinate descent.

    Returns
    -------
    str
        Validated optimization method.

    Raises
    ------
    Exception
        Raised when an incompatible method is requested with penalization.
    NotImplementedError
        Raised when the method name is not recognized.
    """
    if method is None:
        if alpha == 0:
            method = METHOD_IRLS
        else:
            method = METHOD_COORD_DESC
    else:
        method = method.upper()

    # An L1 component requires coordinate descent. Pure ridge is smooth and can
    # instead be folded into the IRLS weighted least-squares normal equations.
    if np.any(alpha) > 0 and np.any(l1_ratio > 0) and method in [METHOD_IRLS, METHOD_COORD_DESCENT_1_ITER]:
        raise Exception("Cannot do method '%s' with non-ridge penalization, must do '%s" % (method, METHOD_COORD_DESC))

    if method not in [METHOD_IRLS, METHOD_COORD_DESC, METHOD_COORD_DESCENT_1_ITER]:
        raise NotImplementedError("Method '%s' not recognized!" % method)

    return method


def _get_penalty_coefficients(alpha, l1_ratio, exog, var_weights, k_exog, normalize, fit_intercept,
                              first_column_constant):

    # Penalty weights are scaled by weighted predictor standard deviations when
    # normalize=True. This makes the penalty less sensitive to column scale.
    stds = ([wtd_std(exog.getcol(k).toarray().flatten() if isspmatrix(exog) else exog[:,k]
                     , weights=var_weights)
             for k in range(k_exog)]
            if normalize else np.ones(k_exog))

    if isinstance(alpha, (int, float, np.integer)):
        alpha = [alpha] * k_exog
    if isinstance(l1_ratio, (int, float, np.integer)):
        l1_ratio = [l1_ratio] * k_exog

    l1s = np.array([a * l * s for a, l, s in zip(alpha, l1_ratio, stds)])
    l2s = np.array([.5 * a * (1.0 - l) * s ** 2 for a, l, s in zip(alpha, l1_ratio, stds)])

    # Never penalize the explicit constant column.
    if fit_intercept and first_column_constant:
        l1s[0] = 0
        l2s[0] = 0

    return l1s, l2s


def glm_internal(
        endog, exog, var_weights=None, instruments=None, start_params=None, L2_penalty_matrix=None, regularize_to_values=None,
        tol=DEFAULT_GLM_TOL, max_iter=DEFAULT_GLM_MAX_ITER, alpha=DEFAULT_GLM_ALPHA, l1_ratio=DEFAULT_GLM_L1_RATIO,
        debug=False, family=DEFAULT_GLM_FAMILY, link=None, fit_intercept=True, normalize=True, penalize_scale=False,
        store_convergence_path=False, line_search_fallback=True, pick_default_start=True, is_endog_regressor=None,
        opt_method=METHOD_IRLS, first_column_constant=False, force_iv_projection=DEFAULT_GLM_FORCE_IV_PROJECTION,
        residual_inclusion=DEFAULT_GLM_RESIDUAL_INCLUSION, residual_inclusion_order=DEFAULT_GLM_RESIDUAL_INCLUSION_ORDER,
        prompt_user_for_more_iters=DEFAULT_GLM_PROMPT_USER_FOR_MORE_ITERS) -> GLMRawFitData:
    """
    Fit a generalized linear model using IRLS or coordinate descent.

    This is the low-level GLM fitting routine. It supports sparse design
    matrices, optional variance weights, optional instrumental-variable
    first-stage projection, elastic-net style penalization, convergence
    diagnostics, and optional line-search fallback when the objective_function increases.

    Parameters
    ----------
    endog : array-like
        Response vector.
    exog : array-like or scipy.sparse matrix
        Design matrix.
    var_weights : array-like or None, optional
        Observation variance weights.
    instruments : array-like or scipy.sparse matrix or None, optional
        Instrument matrix used for first-stage projection.
    start_params : array-like or None, optional
        Initial parameter values.
    L2_penalty_matrix : array-like or scipy.sparse matrix or None, optional
        Symmetric penalty matrix added to ``X'WX`` at each IRLS iteration (see
        :func:`_update_irls_iter`). It can define a general ridge penalty or the
        assembled B-spline roughness penalty for a GAM. The matrix is used only
        by IRLS. If pure-ridge ``alpha`` is also nonzero, the diagonal matrix
        derived from ``alpha`` replaces this matrix.
    regularize_to_values : scalar or array-like or None, optional
        Center ``r`` of the elastic-net penalty. Coordinate descent applies
        both L1 and L2 penalties to ``params - r``. IRLS uses
        ``L2_penalty_matrix @ r`` on the right-hand side of its normal
        equations. ``None`` centers the penalty at zero.
    tol : float, optional
        Absolute parameter-change tolerance used for convergence.
    max_iter : int, optional
        Maximum number of optimization iterations.
    alpha : float or array-like, optional
        Overall or per-parameter regularization strength. A value of zero
        disables elastic-net penalization. Pure ridge can be optimized with
        either coordinate descent or IRLS.
    l1_ratio : float, optional
        Fraction of regularization assigned to the L1 penalty. The remainder is
        assigned to the L2 penalty.
    debug : bool, optional
        Whether to print per-iteration diagnostics and optional debug plots.
    family : str or object, optional
        GLM family specification.
    link : str or object or None, optional
        Link specification. If omitted, the family default is used.
    fit_intercept : bool, optional
        Whether to fit an intercept.
    normalize : bool, optional
        Whether to scale penalties using weighted predictor standard deviations.
    penalize_scale : bool, optional
        Whether to multiply penalties by the estimated scale.
    store_convergence_path : bool, optional
        Whether to retain per-iteration optimization diagnostics.
    line_search_fallback : bool, optional
        Whether to shrink updates when the objective_function increases.
    pick_default_start : bool, optional
        Whether to compare supplied starting parameters against the default
        intercept-only start and choose the better likelihood.
    is_endog_regressor : list[bool] or None, optional
        Flags indicating which regressors should be treated as endogenous when
        instruments are supplied.
    opt_method : str, optional
        Optimization method. Supported values are IRLS, coordinate descent, and
        one-iteration coordinate descent. Any L1 component requires coordinate
        descent; pure L2 penalization supports both main methods.
    first_column_constant : bool, optional
        Whether the first exogenous column is an explicit constant.
    force_iv_projection : bool, optional
        Whether to force instrumental-variable projection behavior.
    residual_inclusion : bool, optional
        Whether to include first-stage residual sparse_terms.
    residual_inclusion_order : int, optional
        Order of residual inclusion sparse_terms.
    prompt_user_for_more_iters : bool or callable, optional
        Controls whether the user may be prompted to extend optimization when
        ``max_iter`` is reached.

    Returns
    -------
    GLMRawFitData
        Raw fitted model data and optimization diagnostics.

    Raises
    ------
    Exception
        Raised for incompatible intercept/method settings or invalid penalized
        optimization settings.
    NotImplementedError
        Raised for unsupported optimization methods.
    """
    _time = time.time()
    opt_method = _get_opt_method(opt_method, alpha, l1_ratio)

    # The IRLS path expects the intercept to already be present as a constant
    # column if an intercept is fit.
    if opt_method == METHOD_IRLS and not first_column_constant and fit_intercept:
        raise Exception("For opt_method 'IRLS' method, cannot have "
                        "first_column_constant=False and fit_intercept=True!")

    # CSC is the preferred sparse format here because coordinate descent updates
    # need efficient column access.
    if not isinstance(exog, csc_matrix):
        exog = csc_matrix(exog)
    if instruments is not None and not isspmatrix(instruments):
        instruments = csc_matrix(instruments)
    original_k_exog = exog.shape[1]

    if is_endog_regressor is None:
        is_endog_regressor = [True] * exog.shape[1]

    if instruments is not None:
        # Replace endogenous columns with first-stage instrumented projections.
        exog_orig = exog
        iv_info = iv_first_stage2(exog, instruments, is_endog_regressor=is_endog_regressor,
                                  debug=debug, _time=_time, residual_inclusion=residual_inclusion,
                                  weights=var_weights, residual_inclusion_order=residual_inclusion_order,
                                  force_iv_projection=force_iv_projection)
        instrument_params = iv_info.instrument_params.toarray()
        exog = iv_info.exog_instrumented
        exog_col_map = iv_info.exog_col_map

        # Residual-inclusion or projection expansion can add columns. Pad any
        # supplied starting values so the shapes remain compatible.
        if start_params is not None and len(start_params) < len(exog_col_map):
            start_params = np.hstack((start_params, [0.0] * (len(exog_col_map) - len(start_params))))

    else:
        instrument_params = None
        exog_orig = exog
        exog_col_map = [(i,) for i in range(exog.shape[1])]

    nobs, k_exog = exog.shape
    penalty_targets = _coerce_regularize_to_values(
        regularize_to_values, k_exog, original_k_exog=original_k_exog)
    df_resid = nobs - (k_exog + (fit_intercept and not first_column_constant))
    df_model = exog.shape[1]

    family, link = _get_family_and_link(family, link)

    is_weighted = var_weights is not None
    if not is_weighted:
        orig_weights = np.ones(nobs).astype(float)
        var_weights = orig_weights

    # Keep a copy of the original weights for likelihood/statistic calculations,
    # then normalize the fitting weights to have mean one.
    orig_weights = var_weights.copy()
    mean_weights = var_weights.mean()
    var_weights = var_weights / mean_weights

    intercept, params, lin_pred = _get_starting_params(
        start_params, fit_intercept, first_column_constant,
        family, link, endog, var_weights, nobs, k_exog, exog,
        pick_default_start=pick_default_start, debug=debug)

    intercept_last = intercept

    l1s, l2s = _get_penalty_coefficients(
        alpha, l1_ratio, exog, var_weights, k_exog, normalize,
        fit_intercept, first_column_constant)

    L2_penalty_matrix_dense = None
    penalty_rhs = None

    if opt_method == METHOD_IRLS:
        L2_penalty_matrix_dense = (
            L2_penalty_matrix.toarray()
            if isspmatrix(L2_penalty_matrix)
            else L2_penalty_matrix
        ) if L2_penalty_matrix is not None else None

        # Pure ridge specified through alpha uses the same matrix normal-equation
        # path as a user-supplied L2 matrix.
        if np.any(l2s > 0) and np.all(np.asarray(l1_ratio) == 0):
            L2_penalty_matrix_dense = np.diag(l2s) * nobs * 2

        if L2_penalty_matrix_dense is not None:
            L2_penalty_matrix_dense = np.asarray(L2_penalty_matrix_dense, dtype=float)
            if L2_penalty_matrix_dense.shape != (k_exog, k_exog):
                raise ValueError(
                    "`L2_penalty_matrix` must have shape "
                    f"({k_exog}, {k_exog}); got {L2_penalty_matrix_dense.shape}."
                )
            L2_penalty_matrix = csc_matrix(L2_penalty_matrix_dense)
            penalty_rhs = (L2_penalty_matrix_dense @ penalty_targets).reshape((-1, 1))
        else:
            L2_penalty_matrix = None

    elif L2_penalty_matrix is not None:
        raise ValueError("`L2_penalty_matrix` is supported only with `opt_method='IRLS'`.")

    endog_predicted, irls_weights = None, None

    converged = False
    convergence_path = []
    objective_last = np.inf

    time_ = time.time()

    len_bar = 0
    if debug:

        print('\n' + '=' * 50)
        print("GLM")
        print('-' * 50)
        print('* Method: %s' % opt_method)
        print('* Endog is %d-length' % endog.shape[0])
        print('* Exog is %d * %d' % exog.shape)
        if instruments is not None:
            print('* Instruments is %d * %d' % instruments.shape)
        if is_weighted:
            print('* Variance weights supplied')
        if alpha > 0:
            al_str = ("%.4f" % alpha) if alpha >= .0001 else ("%.2e" % alpha)
            l1_str = ("%.4f" % l1_ratio) if l1_ratio >= .0001 else ("%.2e" % l1_ratio)
            print('* Penalization: alpha=%s, l1_ratio=%s' % (al_str, l1_str))
        print('* Fitting intercept: %s' % fit_intercept)
        print('* Family is %s with %s link' % (family.name(), link.name()))
        print('-' * 50 + '\n')

        s = (("%6s%10s" + "%15s" * 5 + '%11s') % ('iter', 'error', 'llf', 'penalty', 'objective_function', 'obj. diff', 'scale', 'iter time'))
        len_bar = len(s)
        print('\n' + '=' * len_bar)
        print('  ' + opt_method)
        print('-' * len_bar)
        print(s)
        print('=' * len_bar)

    iter_time = time.time()
    itr = 0
    ncp = None

    while True:

        try:

            itr += 1

            # Store prior values for convergence checks and optional line search.
            params_last = params.copy()
            intercept_last = intercept
            lin_pred_last = lin_pred.copy()

            if opt_method == METHOD_COORD_DESC or opt_method == METHOD_COORD_DESCENT_1_ITER:

                if itr == 0 and opt_method == METHOD_COORD_DESCENT_1_ITER:
                    opt_method = METHOD_IRLS  # switch after this iter

                # Sweep through all non-intercept coordinates.
                for col_idx in range(first_column_constant, exog.shape[1]):

                    if penalize_scale and itr + col_idx > 0:
                        scale = _estimate_scale(
                            endog, link.inverse_link(lin_pred), orig_weights, family, df_resid, use_correction=False)
                    else:
                        scale = 1.0

                    endog_predicted, irls_weights, g_prime \
                        = _get_working_endog_predicted_and_irls_weights(lin_pred, family, link)

                    intercept, endog_predicted, irls_weights, g_prime = _update_coord_descent2(
                        col_idx, nobs, intercept, params, lin_pred, endog, exog.data, exog.indptr, exog.indices,
                        fit_intercept, var_weights, l1s, l2s, penalty_targets,
                        endog_predicted, irls_weights, g_prime, scale, penalize_scale)

            elif opt_method == METHOD_IRLS:
                # IRLS solves a weighted least-squares approximation to the GLM
                # objective_function at the current linear predictor.
                params, endog_predicted, irls_weights, g_prime, lin_pred, working_response,  XpX, ncp \
                    = _update_irls_iter(
                        var_weights, lin_pred, family, link, endog, exog,
                        L2_penalty_matrix, penalty_rhs)
                scale = _estimate_scale(
                    endog, link.inverse_link(lin_pred), orig_weights, family, df_resid, use_correction=False)

                if fit_intercept:
                    intercept = params[0]
                else:
                    intercept = 0

            else:
                raise NotImplementedError(opt_method)

            # Use maximum parameter movement as the primary convergence metric.
            err = max(np.max(np.abs(params_last - params)), abs(intercept - intercept_last))

            if not fit_intercept and ((itr + 1) * k_exog > 20 or err < 100 * tol):
                intercept = 0
                err = max(np.max(np.abs(params_last - params)), abs(intercept - intercept_last))

            # Evaluate objective_function = negative log-likelihood + penalty.
            theta = family.b_deriv_inv(endog_predicted)
            llf = family.log_likelihood(endog, theta, scale=1, var_weights=orig_weights)

            if opt_method == METHOD_IRLS:
                penalty = _get_matrix_penalty(
                    params, L2_penalty_matrix_dense, penalty_targets)
            else:
                penalty = _get_penalty(
                    nobs, params, l1s, l2s, penalty_targets, scale,
                    penalize_scale)
            objective = -llf + penalty

            if debug:
                if objective > objective_last * (1 + 1e-3):
                    print("Warning: objective_function function did not decrease on iter %d   " % itr
                          + str((objective, objective_last)))

            # aa = 1.5
            # if False and objective < objective_last:
            #     while True:
            #         params_new = params_last + aa * (params - params_last)
            #         intercept_new = intercept_last + aa * (intercept - intercept_last)
            #         lin_pred_new = lin_pred_last + aa * (lin_pred - lin_pred_last)
            #         endog_predicted_new = link.inverse_link(lin_pred_new)
            #         theta_new = family.b_deriv_inv(endog_predicted_new)
            #         llf_new = family.log_likelihood(endog, theta_new, scale=1, var_weights=orig_weights)
            #         penalty_new = _get_penalty(
            #             nobs, params_new, l1s, l2s, penalty_targets, scale,
            #             penalize_scale)
            #         objective_new = -llf_new + penalty_new
            #         if objective_new < objective:
            #             # print(aa, objective_new-objective_function, objective_new-objective_last)
            #             aa *= 1.25
            #             objective = objective_new
            #             params = params_new
            #             penalty = penalty_new
            #             llf = llf_new
            #             intercept = intercept_new
            #             lin_pred = lin_pred_new
            #             theta = theta_new
            #             endog_predicted = endog_predicted_new
            #         else:
            #             break

            # If the objective_function got worse, shrink the step toward the previous
            # iterate until the objective_function is acceptable or the line-search cap is
            # reached.
            if line_search_fallback:
                cnt_line_search = 0
                while objective - objective_last > 1e-3 * abs(objective):
                    params = params_last + LINE_SEARCH_SHRINK * (params - params_last)
                    intercept = intercept_last + LINE_SEARCH_SHRINK * (intercept - intercept_last)
                    lin_pred = lin_pred_last + LINE_SEARCH_SHRINK * (lin_pred - lin_pred_last)
                    endog_predicted = link.inverse_link(lin_pred)
                    theta = family.b_deriv_inv(endog_predicted)
                    llf = family.log_likelihood(endog, theta, scale=1, var_weights=orig_weights)
                    if opt_method == METHOD_IRLS:
                        penalty = _get_matrix_penalty(
                            params, L2_penalty_matrix_dense, penalty_targets)
                    else:
                        penalty = _get_penalty(
                            nobs, params, l1s, l2s, penalty_targets, scale,
                            penalize_scale)
                    objective = -llf + penalty
                    if debug:
                        print("\t\t\tLine search%4d%15.2e%15.2e%15.2e" % (
                            cnt_line_search, objective, objective_last, objective - objective_last))
                    cnt_line_search += 1
                    err = np.inf  # do not converge if line search triggered
                    if cnt_line_search > MAX_LINE_SEARCH:
                        if debug:
                            print()
                        break

            # Store per-iteration diagnostics. The full path is returned only if
            # store_convergence_path=True.
            convergence_path.append({
                'iter': itr, 'params': np.array(([intercept] if fit_intercept else []) + list(params)),
                'llf': llf, 'penalty': penalty,
                'objective_function': objective,
                'error': err, 'scale': scale
            })

            if debug:
                print(("%6d%10.2e" + "%15.4e" * 3 + '%15.2e%15.4f' + '%10.1fs') % (
                    itr, err, llf, penalty, objective, objective - objective_last, scale, time.time()-iter_time))

            objective_chg = objective - objective_last
            objective_last = objective

            if DEBUG_INTERNAL and itr % 10 == 0:
                try:
                    f, ax = plt.subplots(ncols=4, figsize=(10, 4))
                    ax[0].hist(endog_predicted)
                    ax[0].axvline(0, color='k')
                    ax[0].axvline(1, color='k')
                    ax[0].set_title("%d, %10.2e" % (itr, err), fontsize=10)
                    ax[0].set_xlabel('endog_predicted')

                    sz = 1.0 / family.variance(endog_predicted)
                    sz *= 5 / max(sz)

                    ax[1].scatter(endog_predicted, endog, alpha=.1, s=sz)
                    ax[1].set_xlabel('mu_hat')
                    ax[1].set_ylabel('endog')

                    llfs = [r['llf'] for r in convergence_path]
                    ax[2].plot(llfs, marker='.')
                    ax[2].set_title("%.6f" % llf)
                    ax[2].set_ylabel('-llf')

                    ax[3].set_xlabel('params')
                    ax[3].bar(range(k_exog + 1), [intercept] + list(params))

                    plt.tight_layout()
                    plt.suptitle((itr, family.name(), link.name()))
                    plt.show()
                except Exception as e:
                    raise e

            if err < tol:
                converged = True
                break

            if itr == max_iter:
                incremental_iters = user_prompt_for_more_iters_method(
                    f"\n\t[iteration = {'%d' % itr}, |dx| = {'%.2e' % err}, "
                    f"|dObjective/Objective| = {'%.2e' % (objective_chg/objective)}]",
                    prompt_user_for_more_iters
                )
                if incremental_iters == 0:
                    break
                else:
                    max_iter += incremental_iters

        except KeyboardInterrupt:
            print("\nProcess interrupted, breaking...\n")
            break

        except Exception as e:
            raise e

    if debug:
        print('-' * len_bar + "\n")
        print("Iterations complete!  %s did %sconverge!\nTime = %.1fs" % (
            opt_method,
            '' if converged else 'not ', time.time() - time_))

    # Re-estimate scale after convergence. With instruments, use predictions
    # generated from the original exogenous matrix for the final scale estimate.
    if instruments is None:
        scale = _estimate_scale(endog, endog_predicted, orig_weights, family, df_resid)

    else:
        scale = _estimate_scale(
            endog,
            exog_orig.dot(params[:exog_orig.shape[1]]),
            #exog_orig.dot(csc_matrix(params[:exog_orig.shape[1]].reshape((-1, 1)))).toarray().ravel(),
            orig_weights, family, df_resid)

    llf = family.log_likelihood(
        endog=endog,
        theta=family.b_deriv_inv(endog_predicted),
        scale=scale * (df_resid / nobs
                       if family.name in (Gaussian.name,)
                       else 1),
        var_weights=orig_weights
    )

    resid = endog - endog_predicted

    # Null model log-likelihood uses an intercept-only mean.
    beta0_null = family.b_deriv_inv(np.average(endog, weights=var_weights))
    llnull = family.log_likelihood(
        endog, beta0_null, scale=scale * (1 + 0 * df_resid / nobs
                                          if family.name in (Gaussian.name,)
                                          else 1),
        var_weights=orig_weights
    )

    # Assemble the returned parameter vector so it consistently includes the
    # intercept when the intercept is not already the first exog column.
    if first_column_constant:
        params[0] = intercept
        params_last[0] = intercept_last
    else:
        params = np.array(([intercept] if fit_intercept else []) + list(params))
        params_last = np.array(([intercept_last] if fit_intercept else []) + list(params_last))

    abs_error, rel_error = _get_errors(params, params_last)

    deviance = family.deviance(endog, endog_predicted, var_weights=orig_weights)
    pearson_chi2 = family.pearson_chi2(endog, endog_predicted, var_weights=orig_weights)

    if ncp is not None:
        edf = np.diag(ncp @ XpX)
    else:
        edf = np.ones(k_exog)  # coordinate descent case?

    fit_time = time.time() - time_

    return GLMRawFitData(params, exog, fit_time, pearson_chi2, deviance, llf, llnull, scale, resid, irls_weights,
                         ncp, edf,
                         g_prime, lin_pred, endog_predicted, opt_method, converged, max_iter, itr, abs_error,
                         rel_error, instrument_params, exog_col_map, family, link, l2s, alpha, l1_ratio,
                         normalize, penalize_scale, df_resid, df_model,
                         convergence_path if store_convergence_path else None)


def _get_errors(params, params_last):
    """
    Compute absolute and relative parameter-change errors.

    Parameters
    ----------
    params : numpy.ndarray
        Current parameter vector.
    params_last : numpy.ndarray
        Parameter vector from the previous iteration.

    Returns
    -------
    tuple[float, float]
        Maximum absolute parameter change and maximum relative parameter change.
    """
    abs_error = np.abs(params - params_last).max()

    # Only compute relative error on nonzero current parameters; otherwise
    # relative changes are either undefined or not informative.
    idx = np.abs(params) > 0
    if np.count_nonzero(idx) == 0:
        rel_error = 0
    else:
        rel_error = np.abs(params / (1e-10 + params_last) - 1)[idx].max()
    return abs_error, rel_error


def _estimate_scale(endog, endog_predicted, var_weights, family, df_resid, use_correction=True):
    """
    Estimate the GLM scale parameter.

    Discrete count/binomial-like families use a fixed scale of one. Other
    families use a Pearson chi-squared style scale estimate.

    Parameters
    ----------
    endog : numpy.ndarray
        Observed response values.
    endog_predicted : numpy.ndarray
        Fitted mean response values.
    var_weights : numpy.ndarray
        Observation variance weights.
    family : object
        GLM family object.
    df_resid : int
        Residual degrees of freedom.
    use_correction : bool, optional
        Whether to divide by residual degrees of freedom instead of the number
        of observations.

    Returns
    -------
    float
        Estimated scale parameter.
    """
    if family.name in [Poisson.name, Binomial.name, NegativeBinomial.name]:
        return 1.0
    scale = _estimate_x2_scale(
        endog, endog_predicted, family.variance, var_weights, df_resid, use_correction=use_correction)
    return scale


def _estimate_x2_scale(endog, mu, var_func, orig_weights, df_resid, use_correction=True):
    """
    Estimate scale using a Pearson chi-squared statistic.

    Parameters
    ----------
    endog : numpy.ndarray
        Observed response values.
    mu : numpy.ndarray
        Fitted mean response values.
    var_func : callable
        Variance function evaluated at ``mu``.
    orig_weights : numpy.ndarray or None
        Original observation weights.
    df_resid : int
        Residual degrees of freedom.
    use_correction : bool, optional
        Whether to divide by residual degrees of freedom. If false, the mean
        Pearson contribution is returned.

    Returns
    -------
    float
        Pearson chi-squared scale estimate.
    """
    resid_sq = (endog - mu) ** 2
    if orig_weights is not None:
        resid_sq *= orig_weights

    # The correction gives the usual Pearson chi-squared scale estimate.
    if use_correction:
        return np.sum(resid_sq / var_func(mu)) / df_resid
    else:
        return np.mean(resid_sq / var_func(mu))


# if __name__ == '__main__':
#     n = 5_000_000
#     df = pd.DataFrame({'y': np.random.randint(0,2,n),
#                        'cat': np.random.randint(0,100,n)})
#     from kanly.api import glm
#     import time
#
#     t = time.time()
#     fit=glm('y ~ C(cat)', df, family='poisson', debug=False)
#     print(fit)
#     print('##### ', time.time()-t)
#
#     import statsmodels.api as sm
#     import statsmodels.formula.api as smf
#
#     t = time.time()
#     # Define the model with a Poisson link function
#     model = smf.glm(formula='y~C(cat)', data=df, family=sm.families.Poisson())
#     # Fit the model
#     results = model.fit()
#     print(results.summary())
#     print('##### ', time.time()-t)
#
#
#
#
