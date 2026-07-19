from __future__ import absolute_import, print_function

import numpy as np
from numba import njit
from scipy.sparse import isspmatrix, csc_matrix

from kanly.regression.linear_models.linear_model_2_quadratic_form import \
    _linear_model_components_2_quadratic_form_and_likelihood


@njit(cache=True)
def _ssr_func(intercept_, coef_, yty, sum_y, Xty, XtX, sum_X, n, sum_w):
    """Evaluate the quadratic sum of squared residuals from precomputed sparse_terms.

    Args:
        intercept_: Current intercept value.
        coef_: Current coefficient vector.
        yty: Precomputed ``y'y`` value.
        sum_y: Weighted or unweighted sum of the response.
        Xty: Precomputed ``X'y`` vector.
        XtX: Precomputed ``X'X`` matrix.
        sum_X: Weighted or unweighted column sums of ``X``.
        n: Number of observations.
        sum_w: Sum of observation weights, or ``n`` when unweighted.

    Returns:
        Scalar SSR value ``sum((y - intercept - X beta)^2)`` in the weighted or
        unweighted metric encoded by the precomputed sparse_terms."""
    val = (
            yty
            + intercept_ ** 2 * sum_w
            + coef_.dot(XtX).dot(coef_)

            + 2 * intercept_ * sum_X.dot(coef_)
            - 2 * intercept_ * sum_y
            - 2 * Xty.dot(coef_)
    )
    return val


@njit(cache=True)
def _ssr_grad_func(intercept_, coef_, Xty, sum_X, XtX):
    """Evaluate the gradient of the SSR quadratic with respect to coefficients.

    Args:
        intercept_: Current intercept value.
        coef_: Current coefficient vector.
        Xty: Precomputed ``X'y`` vector.
        sum_X: Weighted or unweighted column sums of ``X``.
        XtX: Precomputed ``X'X`` matrix.

    Returns:
        Gradient vector of the unscaled SSR with respect to ``coef_``."""
    return 2.0 * (
            -Xty
            + intercept_ * sum_X
            + XtX.dot(coef_)
    )


class ElasticNetObjectiveFunction(object):
    """Quadratic-form elastic-net objective_function used by coordinate descent.

    Stores sufficient statistics for the least-squares part of the objective_function so the
    solver can evaluate objective_function values, gradients, subgradients, and penalties
    without repeatedly multiplying by the original design matrix.  The objective_function is
    ``SSR / (2*n) + penalty_scale * (L1 + L2)``."""

    def __init__(self, n, XtX, Xty, yty, sum_y, sum_X, sum_w, l1_penalties, l2_penalties, penalty_scale=1.0,
                 regularize_to_values=None):
        """Initialize the elastic-net objective_function from quadratic-form components.

        Args:
            n: Number of observations.
            XtX: Precomputed cross-product matrix ``X'X``.
            Xty: Precomputed vector ``X'y``.
            yty: Precomputed scalar ``y'y``.
            sum_y: Weighted or unweighted sum of ``y``.
            sum_X: Weighted or unweighted column sums of ``X``.
            sum_w: Sum of weights, or ``n`` when unweighted.
            l1_penalties: Per-coordinate L1 penalty weights.
            l2_penalties: Per-coordinate L2 penalty weights.
            penalty_scale: Multiplicative scale applied to both penalty sparse_terms.
            regularize_to_values: Optional coefficient targets; penalties apply to
                ``coef - regularize_to_values``."""
        self.n = n
        self.XtX = XtX
        self.XtX_diag = np.diag(XtX)
        self.Xty = Xty
        self.yty = yty
        self.sum_y = sum_y
        self.sum_X = sum_X
        self.sum_w = sum_w
        self.l1_penalties = np.array(l1_penalties)
        self.l2_penalties = np.array(l2_penalties)
        self.penalty_scale = penalty_scale
        if regularize_to_values is None:
            self.regularize_to_values = 0.0
        else:
            self.regularize_to_values = np.array(regularize_to_values)

    # TODO generate a quadratic form with the
    # intercept as part of the coefficient vector

    def scale_penalties(self, penalty_scale):
        """Set the multiplicative scale applied to L1 and L2 penalties.

        Args:
            penalty_scale: Non-negative scalar used to scale the penalty portion of the
                objective_function."""
        assert penalty_scale >= 0
        self.penalty_scale = penalty_scale

    def __call__(self, intercept_, coef_):
        """Evaluate the full elastic-net objective_function at intercept and coefficients.

        Args:
            intercept_: Current intercept value.
            coef_: Current coefficient vector.

        Returns:
            Scalar penalized objective_function value."""
        return self.objective_func(intercept_, np.asarray(coef_))

    def ssr_func(self, intercept_, coef_):
        """Evaluate the least-squares SSR portion at intercept and coefficients.

        Args:
            intercept_: Current intercept value.
            coef_: Current coefficient vector.

        Returns:
            Scalar sum of squared residuals from the stored quadratic form."""
        return _ssr_func(intercept_, coef_, self.yty, self.sum_y, self.Xty, self.XtX, self.sum_X, self.n, self.sum_w)

    def penalty_func(self, coef_):
        """Evaluate the elastic-net penalty at a coefficient vector.

        Args:
            coef_: Coefficient vector on the original parameter scale.

        Returns:
            Scalar penalty ``penalty_scale * (sum(l1*abs(delta)) + sum(l2*delta**2))``
            where ``delta = coef_ - regularize_to_values``."""
        par = np.abs(coef_ - self.regularize_to_values)
        return (
                np.sum(par ** 2 * self.l2_penalties)
                + np.sum(par * self.l1_penalties)
        ) * self.penalty_scale

    def objective_func(self, intercept_, coef_):
        """Evaluate ``SSR/(2*n) + elastic_net_penalty``.

        Args:
            intercept_: Current intercept value.
            coef_: Current coefficient vector.

        Returns:
            Scalar objective_function minimized by the coordinate-descent solver."""
        return (
                self.ssr_func(intercept_, coef_) / (2 * self.n)
                + self.penalty_func(coef_)
        )

    def ssr_grad_func(self, intercept_, coef_):
        """Evaluate the SSR gradient with respect to coefficients.

        Args:
            intercept_: Current intercept value.
            coef_: Current coefficient vector.

        Returns:
            Gradient vector for the unscaled SSR quadratic."""
        return _ssr_grad_func(intercept_, coef_, self.Xty, self.sum_X, self.XtX)

    def objective_subgrad_func(self, intercept_, coef_):
        """Evaluate a valid subgradient of the elastic-net objective_function.

        Args:
            intercept_: Current intercept value.
            coef_: Current coefficient vector.

        Returns:
            Subgradient vector.  Coordinates at the L1 kink are set to zero when the
            smooth gradient lies within the L1 subgradient interval."""
        val = self.ssr_ridge_subgrad_func(intercept_, coef_)

        sgn = np.sign(coef_)
        val += self.l1_penalties * sgn * self.penalty_scale
        # At exactly zero, the L1 derivative is an interval.  If the smooth
        # gradient fits inside that interval, zero satisfies the KKT condition.
        val[(np.sign(coef_) == 0) & (np.abs(val) <= self.penalty_scale * self.l1_penalties)] = 0.0

        return val

    def ssr_ridge_subgrad_func(self, intercept_, coef_):
        """Evaluate the smooth part of the objective_function gradient.

        Args:
            intercept_: Current intercept value.
            coef_: Current coefficient vector.

        Returns:
            Gradient of ``SSR/(2*n)`` plus the L2 penalty gradient, excluding the L1
            subgradient."""
        _ssr_grad = self.ssr_grad_func(intercept_, coef_)
        val = (
                _ssr_grad / (2. * self.n)
                + 2. * self.l2_penalties * (coef_ - self.regularize_to_values) * self.penalty_scale
        )
        return val

    def objective_function_one_argument(self, x):
        """Evaluate the objective_function using a packed vector ``[intercept, coefficients...]``.

        Args:
            x: One-dimensional vector whose first element is the intercept and remaining
                elements are coefficients.

        Returns:
            Scalar objective_function value."""
        return self(x[0], x[1:])

    def add_intercept(self):
        """
        Transforms the objective_function function to include an unpenalized
        intercept term.  Should be called after construction
        with `intercept_` arg set to 0.

        Returns:
            New ``ElasticNetObjectiveFunction`` whose first coordinate is the
            unpenalized intercept and whose remaining coordinates are the
            original slope coefficients.
        """

        def _add_intercept_to_term(term):
            """Pad penalty/target arrays with an unpenalized intercept coordinate.

            Args:
                term: Scalar or vector penalty/target term for slope coefficients.

            Returns:
                Vector with leading zero for the intercept followed by the original term."""
            if isinstance(term, (int, float)) or (isinstance(term, np.ndarray) and np.prod(term.shape) == 1):
                if term == 0:
                    return 0.0
                else:
                    return np.hstack([[0.0], np.full(self.XtX.shape[1], term)])
            else:
                return np.hstack([[0.0], term])

        # Promote the intercept into the quadratic form so generic one-argument
        # optimizers can consume a single packed parameter vector.
        XtX = np.array(np.bmat([
            [np.array([[self.sum_w]]), self.sum_X.reshape((1, -1)).copy()],
            [self.sum_X.reshape((-1, 1)).copy(), self.XtX.copy()]
        ]))

        return ElasticNetObjectiveFunction(
            self.n,
            XtX,
            np.hstack([[self.sum_y], self.Xty.copy()]),
            self.yty,
            self.sum_y,
            np.hstack([[self.sum_w], self.sum_X.copy()]),
            self.sum_w,
            l1_penalties=_add_intercept_to_term(self.l1_penalties),
            l2_penalties=_add_intercept_to_term(self.l2_penalties),
            penalty_scale=self.penalty_scale,
            regularize_to_values=_add_intercept_to_term(self.regularize_to_values)
        )

    @staticmethod
    def build_elastic_net_objective_function(
            X, y, l1_penalties, l2_penalties, weights=None, debug=False, ssr_quad_form=None,
            regularize_to_values=None, penalty_scale=1.):
        """Build an ``ElasticNetObjectiveFunction`` from raw design/response data.

        Args:
            X: Design matrix, dense or sparse.
            y: Response vector.
            l1_penalties: Per-coordinate L1 penalty weights.
            l2_penalties: Per-coordinate L2 penalty weights.
            weights: Optional observation weights.
            debug: Whether to print lower-level conversion diagnostics.
            ssr_quad_form: Optional precomputed quadratic form to reuse.
            regularize_to_values: Optional coefficient target vector for penalties.
            penalty_scale: Multiplicative penalty scale.

        Returns:
            Tuple ``(objective_function, ssr_quad_form)``."""

        n = X.shape[0]
        if ssr_quad_form is None:
            _, ssr_quad_form = _linear_model_components_2_quadratic_form_and_likelihood(y, X, weights)

        is_weighted = weights is not None
        XtX = ssr_quad_form.d2f_db2
        Xty = -ssr_quad_form.df_db / 2
        yty = ssr_quad_form.f0

        sum_y = np.sum(y * weights) if is_weighted else y.sum()

        if is_weighted:
            if isspmatrix(X):
                sum_X = csc_matrix(weights).dot(X).toarray().flatten()
            else:
                sum_X = np.dot(weights, X)
        else:
            sum_X = np.array(X.sum(axis=0)).flatten()

        sum_w = sum(weights) if is_weighted else n

        en_obj_func = ElasticNetObjectiveFunction(
            n, XtX, Xty, yty, sum_y, sum_X, sum_w, l1_penalties, l2_penalties, penalty_scale=penalty_scale,
            regularize_to_values=regularize_to_values
        )

        return en_obj_func, ssr_quad_form

    def proximal(self, x, step):
        """
        proximal operator
        solves
            min over z {
                |z|_2^2 / (2*step)
                + dot(l1_penalties, |z|)
                + dot(l2_penalties, z**2)
            }

        Args:
            x: Input vector before applying the elastic-net proximal step.
            step: Positive step size controlling threshold strength.

        Returns:
            Vector after L1 soft-thresholding and L2 shrinkage.
        """
        abs_x = np.abs(x)
        return np.where(
            abs_x > self.l1_penalties * step,
            np.sign(x) * (abs_x - self.l1_penalties * step),
            0.0
        ) / (1 + 2 * step * self.l2_penalties)


# if __name__ == '__main__':
#     import pandas as pd
#
#     n = 2_000
#     p = 20
#     s = .01
#     np.random.seed(0)
#     X = np.random.randn(n, p).dot((s * np.eye(p) + (1 - s) * np.ones((p, p))))
#     z = np.ones(4)
#     beta = np.hstack([z, np.zeros(p - len(z))])
#     y = 3 + X.dot(beta) + np.random.randn(n)

#     xcols = [f'x{d}' for d in range(p)]
#     df = pd.DataFrame(X, columns=xcols)
#     df['y'] = y
#
#     from kanly.regression.linear_models.penalized.sparse_elastic_net_internal \
#         import sparse_elastic_net_coordinate_descent_quad_form_setup
#
#     alpha, l1_ratio = 1, .9
#
#     fit = sparse_elastic_net_coordinate_descent_quad_form_setup(
#         X, y,
#         alpha=alpha,
#         l1_ratio=l1_ratio,
#         normalize=False,
#         max_iter=150_000,
#         selection='cyclic',
#         #regularize_to_values=[10.] + [0.0] * (p - 1),
#     )
#     print(fit['converged'], fit['x_error'])
#     print(fit['objective_function'])
#     print(fit['coef_'])
#
#     from kanly.api import bfgs_pqn
#
#     res = bfgs_pqn(fit['objective_function'].objective_function_one_argument, np.zeros(p + 1), maxiter=500)
#     print(res.converged, res.xerr)
#     print(res.fun)
#     print(res.x[1:])
#
#     print(pd.DataFrame({'en': fit['coef_'], 'bfgs': res.x[1:]}).round(4))


#     def prox(self, x, step):
#         return np.where(np.abs(x) > self.l1_penalties*step,
#                         np.sign(x) * np.abs(x - self.l1_penalties*step),
#                         0.0)
#
#
# if __name__ == '__main__':
#     n = 2_000
#     p = 40
#     s = .01
#     np.random.seed(0)
#     X = np.random.randn(n, p).dot((s * np.eye(p) + (1 - s) * np.ones((p, p))))
#     z = np.ones(4)
#     beta = np.hstack([z, np.zeros(p - len(z))])
#     y = 3 + X.dot(beta) + np.random.randn(n)
#
#
#     import pandas as pd
#     xcols = [f'x{d}' for d in range(p)]
#     df = pd.DataFrame(X, columns=xcols)
#     df['y'] = y
#
#     alpha_ = .005
#     l1_ratio = 1.0
#
#     from kanly.api import elastic_net
#     fit1 = elastic_net('y ~ ' + '+'.join(xcols), df, alpha=alpha_, l1_ratio=l1_ratio, normalize=False,
#                        fit_intercept=True, max_iter=1000)
#     print(fit1.summary(show_only_non_zero=True))
#
#     func = ElasticNetObjectiveFunction(n, X.T.dot(X), X.T.dot(y), y.T.dot(y), sum(y),
#                                        X.sum(axis=0), n, np.full(p, alpha_ * l1_ratio),
#                                        np.full(p, alpha_ / 2 * (1 - l1_ratio)))
#
#     coef0 = np.zeros(X.shape[1])
#     intercept0 = y.mean()
#     mean_X = X.mean(axis=0)
#     y_mean = y.mean()
#     g0 = func.ssr_ridge_subgrad_func(intercept0, coef0)
#     fun0 = func(intercept0, coef0)
#
#     eps = 1e-6
#
#     step_size = 1
#     for i in range(1000):
#
#         coef_last = coef0.copy()
#         fun_last = fun0
#
#         updated = False
#         coef1_intermediate = coef0 - step_size * (g0 * (1 + .0 * np.random.randn(p)))
#         coef1 = func.prox(coef1_intermediate, step=step_size)
#
#         # if fun0 < .6:
#         #     print("* ", i)
#         #     print(np.abs(coef1_intermediate).sum(), np.abs(coef1).sum())
#         #     print(pd.DataFrame({'x': coef1_intermediate, 'x_prox': coef1, 'l1r': func.l1_penalties}))
#         #     raise Exception
#
#         intercept1 = y_mean - mean_X.dot(coef1)
#         fun1 = func(intercept1, coef1)
#
#         if fun1 < fun0:
#             coef0 = coef1
#             intercept0 = intercept1
#             step_size *= 1.5
#             fun0 = fun1
#             updated = True
#         else:
#             step_size = max(1e-15, step_size/1.5)
#
#         # if step_size == 1e-15:
#         #     dir = (g0 * (1 + .01 * np.random.randn(p)))
#         #     gap = 1e-4
#         #     while True:
#         #         XX = np.linspace(-gap, gap, 7)
#         #         YY = [func(intercept0, coef0 - x * dir) for x in XX]
#         #         best = np.argmin(YY)
#         #         coef0 = coef0 - XX[best] * dir
#         #         intercept0 = y_mean - mean_X.dot(coef1)
#         #         fun1 = func(intercept0, coef0)
#         #         if fun1 < fun0:
#         #             fun0 = fun1
#         #             step_size = XX[best]
#         #             updated = True
#         #             break
#         #         elif gap < 1e-15:
#         #             step_size = 0
#         #             break
#         #         else:
#         #             gap /= 100
#
#         if updated:
#             g0 = func.ssr_ridge_subgrad_func(intercept0, coef0)
#
#         if i < 300 or i % 5000 == 0 or updated:
#             err = max(np.abs(coef0 - coef_last))
#             print({'itr': i, 'fun_last': "%10.4e" % fun_last, "fun": "%10.4e" % fun0, "dF": "%10.4e" % (fun_last-fun0),
#                    "step": "%10.1e" % step_size, 'err': "%10.2e" % err, 'updated': updated})
#             if (updated and err < 1e-8):
#                 break
#
#     print(intercept0, coef0)
#     print(g0)
