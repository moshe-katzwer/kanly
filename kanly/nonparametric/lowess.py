from __future__ import absolute_import, print_function

import numpy as np
from numba import njit
from pandas import DataFrame

from kanly.formula.data_getter import SparseDataGetter
from kanly.formula.keys import ENDOG_KEY, EXOG_KEY
from kanly.bootstrap.bootstrap import bootstrap_entire_procedure
from kanly.nonparametric.interpolate import interp, LINEAR

DEGREE = 1
IT = 1
FRAC = .33
DELTA = None
RETURN_SORTED = True


@njit(cache=True)
def tricube_weight(u, s):
    """Return LOWESS tricube kernel weights for scaled distances.

    Args:
        u: Distance(s) from the local fit location.
        s: Scale radius for the neighborhood. Distances with ``abs(u / s) >= 1``
            receive zero weight.

    Returns:
        Array of tricube weights with the same broadcasted shape as ``u``.
    """
    z = np.abs(u / s)
    return np.where(z < 1, (1. - z ** 3) ** 3, 0.)


@njit(cache=True)
def bisquare_weight(u, s):
    """Return robust bisquare weights for residual reweighting.

    Args:
        u: Residual(s) from the previous LOWESS iteration.
        s: Robust scale estimate, typically the median absolute residual.

    Returns:
        Array of bisquare weights. Residuals more than ``6 * s`` away receive
        zero weight.
    """
    z = np.abs(u / (6 * s))
    return np.where(z < 1, (1. - z ** 2) ** 2, 0.)


@njit(cache=True)
def _wls_1degree(y, x, w):
    """Fit a weighted local linear regression.

    Args:
        y: Response values in the local neighborhood.
        x: Centered predictor values in the local neighborhood.
        w: Non-negative observation weights.

    Returns:
        Tuple ``(b0, b1)`` for the weighted regression ``y = b0 + b1 * x``.
        Degenerate neighborhoods fall back to the weighted mean with zero slope.
    """

    n = len(y)
    w_sum = w.sum()
    w_x_sum = (w * x).sum()
    w_y_sum = (w * y).sum()
    y_mean = w_y_sum / w_sum

    if abs(w_x_sum) < 1e-8:
        return y_mean, 0.0

    denom = (w * x ** 2).sum() - w_x_sum ** 2 / w_sum
    if abs(denom) < 1e-6:
        return y_mean, 0.0

    b1 = (
            ((w * y * x).sum() - w_y_sum * w_x_sum / w_sum)
            / denom
    )
    b0 = (w_y_sum - b1 * w_x_sum) / w_sum
    return b0, b1


def get_xvals(xvals, exog, delta, reindex, do_quantile=True):
    """Normalize the grid of x-values where LOWESS should be evaluated.

    Args:
        xvals: Optional evaluation grid. If an integer, build that many grid
            points. If array-like, sort and use the supplied values. If ``None``,
            use the observed ``exog`` values.
        exog: Sorted explanatory variable values.
        delta: LOWESS interpolation skip distance. User-supplied grids require
            ``delta == 0``.
        reindex: Sort index for the original observed data.
        do_quantile: If ``True`` and ``xvals`` is an integer, space grid points
            by empirical quantiles of ``exog``. Otherwise space them evenly over
            the range of ``exog``.

    Returns:
        Tuple ``(xvals, reindex_xvals, user_xvals, added_left, added_right)``.
        Boundary flags indicate whether observed endpoints were temporarily
        added to support interpolation over a user grid.
    """
    user_xvals = False
    if xvals is not None:
        user_xvals = True
        if delta is None:
            delta = 0
        assert delta == 0

        if isinstance(xvals, int):
            assert xvals > 1
            reindex_xvals = np.arange(xvals)
            if do_quantile:
                xvals = np.quantile(exog, np.linspace(0, 1, xvals))
                xvals = np.unique(xvals)
            else:
                xvals = exog.min() + np.linspace(0, 1, xvals) * (exog.max() - exog.min())
            added_left = added_right = False

        else:
            xvals = np.asarray(xvals, dtype=np.float64)
            reindex_xvals = np.argsort(xvals)
            xvals = xvals[reindex_xvals]

            added_left = xvals[0] > exog[0]
            if added_left:
                xvals_new = np.zeros(len(xvals) + 1)
                xvals_new[0] = exog[0]
                xvals_new[1:] = xvals
                xvals = xvals_new
            added_right = xvals[-1] < exog[-1]
            if added_right:
                xvals_new = np.zeros(len(xvals) + 1)
                xvals_new[-1] = exog[-1]
                xvals_new[:-1] = xvals
                xvals = xvals_new

    else:
        added_left = added_right = False
        xvals = exog
        reindex_xvals = reindex

    return xvals, reindex_xvals, user_xvals, added_left, added_right


def LOWESS(endog, exog=None, xvals=None, frac=FRAC, delta=DELTA, degree=DEGREE, it=IT, weights=None,
           bootstrap_samples=None, do_xval_quantiles=True,
           debug=False, return_sorted=RETURN_SORTED, return_arrays=True, do_njit=True):
    """Fit LOWESS smoothing to arrays.

    Args:
        endog: Response values to smooth.
        exog: Optional explanatory variable values. If omitted, observation
            index positions are used.
        xvals: Optional evaluation grid. If an integer, that many evaluation
            points are generated. If array-like, those x-values are used.
        frac: Neighborhood size. Values below 1 are interpreted as a fraction
            of observations; values above 1 are interpreted as a point count and
            must be at least 3.
        delta: Distance within which fitted values are linearly interpolated
            instead of refitting. If omitted, defaults to a range-based spacing
            for observed-grid fits and zero for user grids.
        degree: Degree of the local polynomial fit.
        it: Number of robust residual-reweighting iterations after the initial
            fit.
        weights: Optional observation weights multiplied into the local kernel
            weights.
        bootstrap_samples: Optional number of Bayesian bootstrap refits to run.
        do_xval_quantiles: If ``xvals`` is an integer, use quantile-spaced grid
            points when ``True`` and evenly spaced grid points when ``False``.
        debug: Forwarded to the bootstrap procedure for progress output.
        return_sorted: If ``False`` and returning arrays, restore user-grid or
            observed-grid order after fitting.
        return_arrays: If ``True``, return ``(xgrid, ygrid)`` arrays. If
            ``False``, return an interpolation callable built from the fitted
            grid.
        do_njit: If ``True``, use the Numba-compiled internal smoother.

    Returns:
        Either ``(xgrid, ygrid)`` or an interpolation callable. If
        ``bootstrap_samples`` is supplied, the bootstrap result is appended to
        the returned tuple.

    Examples
    --------
    Fit a LOWESS smoother to noisy non-linear data:

    >>> import numpy as np
    >>> from kanly.api import LOWESS
    >>> rng = np.random.default_rng(0)
    >>> x = np.sort(rng.uniform(0, 10, size=200))
    >>> y = np.sin(x) + 0.3 * rng.normal(size=200)
    >>> x_smooth, y_smooth = LOWESS(y, exog=x, frac=0.3)

    Bayesian-bootstrap uncertainty bands:

    >>> x_smooth, y_smooth, bs = LOWESS(y, exog=x, frac=0.3,    # doctest: +SKIP
    ...                                 bootstrap_samples=200)

    See Also
    --------
    `lowess` : formula-API wrapper.
    """

    if exog is not None:
        exog = np.asarray(exog, dtype=np.float64)
    endog = np.asarray(endog, dtype=np.float64)

    frac = float(frac)
    assert frac > 0 and (frac < 1 or frac >= 3)

    if exog is None:
        reindex = np.arange(len(endog))
        exog = np.arange(len(endog))
    else:
        reindex = np.argsort(exog)
        exog = exog[reindex]
        endog = endog[reindex]

    is_weighted = weights is not None
    if is_weighted:
        weights = np.asarray(weights, dtype=np.float64)
        # Keep external weights aligned with the sorted x/y arrays.
        weights = weights[reindex]
    else:
        # A length-one placeholder keeps the njit signature simple when the fit
        # is unweighted; the flag controls whether it is actually used.
        weights = np.ones(1, dtype=np.float64)

    # xvals
    xvals, reindex_xvals, is_user_xvals, added_left, added_right = \
        get_xvals(xvals, exog, delta, reindex, do_quantile=do_xval_quantiles)

    if delta is None:
        if is_user_xvals:
            delta = 0.0
        else:
            delta = (exog.max() - exog.min()) / 40

    endog, exog = np.asarray(endog, dtype=np.float64), np.asarray(exog, dtype=np.float64)

    lowess_internal_func = _LOWESS_INTERNAL_NJIT if do_njit else _LOWESS_INTERNAL

    xgrid, ygrid = lowess_internal_func(
        endog, exog,
        xvals=xvals,
        frac=frac, delta=delta, degree=degree, it=it,
        weights=None if weights is None else np.asarray(weights, dtype=np.float64),
        is_weighted=is_weighted, user_xvals=is_user_xvals,
        added_left=added_left, added_right=added_right,
    )

    if not return_sorted and return_arrays:
        unsort_index = np.argsort(reindex_xvals)
        xgrid, ygrid = xgrid[unsort_index], ygrid[unsort_index]

    ygrid[np.isnan(ygrid)] = 0

    if return_arrays:
        retval = xgrid, ygrid
    else:
        # Return a reusable interpolator when callers want predictions at new x.
        retval = interp(xgrid, ygrid, kind=LINEAR)

    if bootstrap_samples is not None and bootstrap_samples:
        def temp_func(bayes_boot_wts):
            """Refit LOWESS under one Bayesian bootstrap draw.

            Args:
                bayes_boot_wts: Bootstrap weights generated for the full
                    observed sample.

            Returns:
                Fitted values on the LOWESS evaluation grid for this bootstrap
                replicate.
            """
            _, mu_boot = _LOWESS_INTERNAL(
                endog, exog,
                xvals=xvals,
                frac=frac, delta=delta, degree=degree, it=it,
                weights=bayes_boot_wts if weights is None else bayes_boot_wts * weights,
                is_weighted=True, user_xvals=is_user_xvals,
                added_left=added_left, added_right=added_right
            )
            if not return_sorted and not return_arrays:
                mu_boot = mu_boot[unsort_index]
            return mu_boot

        boot_result = bootstrap_entire_procedure(temp_func, len(endog), n_samples=bootstrap_samples, debug=debug)
        return *retval, boot_result
    else:
        return retval


def _LOWESS_INTERNAL(endog, exog, xvals, frac, delta, degree, it, weights=None, is_weighted=False,
                     added_left=False, added_right=False,
                     user_xvals=False):
    """Core LOWESS smoother used by both Python and Numba entry points.

    Args:
        endog: Sorted response values.
        exog: Sorted explanatory variable values.
        xvals: Sorted evaluation grid, possibly with temporary endpoint values.
        frac: Neighborhood fraction or absolute point count.
        delta: Minimum x-distance before recomputing a local fit; intervening
            values are linearly interpolated.
        degree: Degree of the local polynomial fit.
        it: Number of robust residual-reweighting iterations.
        weights: Optional sorted observation weights.
        is_weighted: Whether ``weights`` should be multiplied into the local
            kernel weights.
        added_left: Whether a temporary left endpoint was added to ``xvals``.
        added_right: Whether a temporary right endpoint was added to ``xvals``.
        user_xvals: Whether the evaluation grid was supplied by the caller.

    Returns:
        Tuple ``(xvals, mean_smoothed)`` after removing any temporary endpoints.
    """
    mean_smoothed = np.zeros(len(xvals))
    n = len(exog)

    if frac > 1:
        num_pts = int(frac)
    else:
        num_pts = max(int(frac * n), 2)

    design = np.zeros((num_pts, degree + 1))
    w = np.zeros(num_pts)
    rtw = np.zeros(num_pts)
    yvec = np.zeros(num_pts)

    resid = np.ones(n)
    resid_median = 1.

    for itr_no in range(it + 1):

        # Track the last exact local fit so skipped grid points can be filled by
        # linear interpolation when delta permits.
        xlast = -np.inf
        ilast = 0

        left = 0
        right = num_pts

        for i, xval in enumerate(xvals):
            if i > 0 and abs(xval - xvals[i-1]) < 1e-8:
                mean_smoothed[i] = mean_smoothed[i-1]
                continue

            if user_xvals or (i == 0 or i == n - 1 or xval - xlast > delta):

                # Find the right set of nearby points
                # -----------------------------------
                while True:
                    if right < n - 1:
                        if xval > (exog[left] + exog[right]) / 2.0:
                            left += 1
                            right += 1
                        else:
                            break
                    else:
                        break

                max_gap = max(abs(xval - exog[left]), abs(xval - exog[right])) * 1.005
                if max_gap == 0:
                    max_gap = 1e-3
                # Center x at the target value so the fitted intercept is the
                # smoothed estimate at xval.
                xvec = exog[left:right] - xval
                yvec[:] = endog[left:right]

                w[:] = tricube_weight(xvec, max_gap)
                if itr_no:
                    w *= bisquare_weight(resid[left:right], resid_median)
                if is_weighted:
                    w *= weights[left:right]
                w += 1e-6

                # f, ax = plt.subplots(figsize=(6,3),ncols=2)
                # ax[0].set_title((i,xval))
                # ax[0].scatter(xvec, yvec)
                # ax[0].axhline(np.average(yvec, weights=w), c='g')
                # ax[0].axvline(0, c='k', ls=':')
                # ax[0].twinx().plot(xvec, w, c='r')

                # print()
                # print(f'{max_gap=}')
                # print(f'{xvec=}')
                # print(f'{yvec=}')
                # print(f'{w=}')

                # do weighted least squares
                # -------------------------
                if degree == 0:
                    mean_smoothed[i] = np.average(yvec, weights=w)
                elif degree == 1:
                    beta0, beta1 = _wls_1degree(yvec, xvec, w)
                    # ax[0].plot(xvec, beta0 + beta1 * xvec, c='magenta')
                    mean_smoothed[i] = beta0
                else:

                    rtw[:] = np.sqrt(w)
                    # Build the weighted polynomial design matrix directly in
                    # weighted space so normal equations use ordinary dot
                    # products below.
                    design[:, 0] = rtw
                    design[:, 1] = xvec * rtw
                    for j in range(2, degree + 1):
                        design[:, j] = design[:, j - 1] * xvec

                    # if i == 6:
                    #     plt.figure()
                    #     plt.title((i,itr_no,))
                    #     plt.scatter(xvec, yvec)
                    #     plt.axvline(-max_gap, c='k')
                    #     plt.axvline(max_gap, c='k')
                    #     plt.gca().twinx().plot(xvec, w, c='r')

                    try:
                        temp_avg_y = np.average(yvec, weights=w)
                    except:
                        temp_avg_y = yvec.mean()
                    # try:
                    #     temp_avg_y = np.average(yvec, weights=w)
                    # except Exception as e:
                    #     plt.figure()
                    #     plt.title((i,itr_no,))
                    #     plt.scatter(xvec, yvec)
                    #     plt.axvline(-max_gap, c='k')
                    #     plt.axvline(max_gap, c='k')
                    #     plt.gca().twinx().plot(xvec, w, c='r')
                    #     raise e
                    if abs(rtw.dot(xvec)) < 1e-8 or np.std(xvec) < 1e-8:
                        mean_smoothed[i] = temp_avg_y
                    else:
                        yvec *= rtw
                        try:
                            beta = np.linalg.solve(design.T.dot(design), design.T.dot(yvec))
                            beta0 = beta[0]
                            # if beta0 < 0:
                            #     plt.figure()
                            #     plt.scatter(xvec, yvec/rtw)
                            #     plt.show()
                        except:
                            beta0 = temp_avg_y
                        mean_smoothed[i] = beta0

                # Linearly interpolate between computed points
                # --------------------------------------------
                if i and i - ilast > 1:
                    if xvals[i] == xvals[ilast]:
                        mean_smoothed[ilast + 1:i] = mean_smoothed[ilast]
                        continue
                    slope = (mean_smoothed[i] - mean_smoothed[ilast])
                    slope /= (xvals[i] - xvals[ilast])
                    for j in range(ilast + 1, i):
                        mean_smoothed[j] = mean_smoothed[ilast] + slope * (xvals[j] - xvals[ilast])

                xlast = xval
                ilast = i

        if itr_no < it:
            if user_xvals:
                # Do interpolation for residuals
                bucket_above = 1
                pred = np.zeros(len(endog))
                slope = ((mean_smoothed[bucket_above] - mean_smoothed[bucket_above - 1])
                         / (xvals[bucket_above] - xvals[bucket_above - 1]))
                for i, xx in enumerate(exog):
                    while xvals[bucket_above] < xx:
                        bucket_above += 1
                        slope = ((mean_smoothed[bucket_above] - mean_smoothed[bucket_above - 1])
                                 / (xvals[bucket_above] - xvals[bucket_above - 1]))
                        if bucket_above == len(xvals) - 1:
                            break

                    pred[i] = mean_smoothed[bucket_above - 1] + slope * (xx - xvals[bucket_above - 1])
                resid = endog - pred
            else:
                resid = np.abs(endog - mean_smoothed)
            # Robust iterations downweight observations with large residuals
            # through the bisquare kernel on the next pass.
            resid_median = np.median(np.abs(resid))

    if added_left:
        xvals, mean_smoothed = xvals[1:], mean_smoothed[1:]

    if added_right:
        xvals, mean_smoothed = xvals[:-1], mean_smoothed[:-1]

    # plt.figure()
    return xvals, mean_smoothed


_LOWESS_INTERNAL_NJIT = njit(_LOWESS_INTERNAL, cache=True)


def lowess(formula, data, xvals=None, frac=FRAC, delta=DELTA, degree=DEGREE, debug=False, index=None, it=IT,
           bootstrap_samples=None, return_sorted=RETURN_SORTED, return_arrays=True, do_xval_quantiles=True):
    """Formula API wrapper for LOWESS smoothing.

    Args:
        formula: Sparse formula with one response and one explanatory variable.
            A ``-1`` intercept removal is appended automatically when missing.
            Formula weights are accepted by the parser but this wrapper passes
            only endog/exog to ``LOWESS``.
        data: DataFrame-like object or dictionary containing formula variables.
        xvals: Optional evaluation grid or integer grid size.
        frac: Neighborhood size, interpreted as a share when below 1 and an
            absolute point count when above 1.
        delta: Distance within which fitted values are interpolated instead of
            refit.
        degree: Degree of the local polynomial fit.
        debug: Forwarded to formula parsing and bootstrap fitting.
        index: Optional row subset passed to ``SparseDataGetter``.
        it: Number of robust residual-reweighting iterations.
        bootstrap_samples: Optional number of Bayesian bootstrap refits.
        return_sorted: If ``False`` and returning arrays, restore the original
            x order.
        return_arrays: If ``True``, return arrays; otherwise return an
            interpolation callable.
        do_xval_quantiles: If ``xvals`` is an integer, use quantile-spaced grid
            points when ``True``.

    Returns:
        The value returned by ``LOWESS`` for the parsed response and single
        explanatory variable.

    Examples
    --------
    Use a formula to fit LOWESS on DataFrame columns:

    >>> import numpy as np, pandas as pd
    >>> from kanly.api import lowess
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({'x': np.sort(rng.uniform(0, 10, size=200))})
    >>> df['y'] = np.sin(df['x']) + 0.3 * rng.normal(size=200)
    >>> x_smooth, y_smooth = lowess('y ~ x', df, frac=0.3)
    """

    if isinstance(data, dict):
        data = DataFrame(data, copy=False)

    if formula.replace(' ', '')[-2:] != '-1':
        formula += ' -1'

    data_result = SparseDataGetter.get_data(
        data, formula, fail_on_iv=True, fail_on_absorb=True, fail_on_weights=False, debug=debug,
        test_formula_on_dummy=False, index=index)

    endog = data_result[ENDOG_KEY].values.toarray().flatten()
    exog = data_result[EXOG_KEY].values
    assert exog.shape[1] == 1
    exog = exog.toarray().flatten()

    return LOWESS(endog, exog, xvals=xvals, frac=frac, degree=degree, delta=delta, it=it,
                  bootstrap_samples=bootstrap_samples, return_sorted=return_sorted, return_arrays=return_arrays,
                  do_xval_quantiles=do_xval_quantiles)

# if __name__ == '__main__':
#     import pandas as pd
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from kanly.api import LOWESS, lm
#
#     np.random.seed(0)
#     nY = 6
#     n = nY * 12
#     df = pd.DataFrame({
#         'year': np.repeat(range(nY), 12),
#         'month': np.tile(range(12), nY)
#     })
#     df['y'] = np.sqrt(np.arange(n)) + np.sin(np.arange(n) / 5) + df.month.map(
#         dict(zip(range(12), 1 * np.random.randn(12)))) + .5 * np.random.randn(n)
#
#     x, ysmooth = LOWESS(df.y, frac=10)
#     plt.plot(df.y)
#     plt.plot(ysmooth)
#     plt.show()
#
# if __name__ == '__main__':
#     import time
#     np.random.seed(0)
#     x = np.random.rand(100)
#     y = np.sin(x * 5) + .15 * np.random.randn(100)
#
#     t = time.time()
#     LOWESS(y, x, np.linspace(0, 1, 10), .15, 0, 1, 0)
#     print(time.time() - t)
#
#
#     t = time.time()
#     LOWESS(y, x, np.linspace(0, 1, 10), .15, 0, 0, 0)
#     print(time.time() - t)