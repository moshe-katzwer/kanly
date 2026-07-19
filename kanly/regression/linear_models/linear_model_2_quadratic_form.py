"""Quadratic-form representation of the SSR and Gaussian log-likelihood.

This module converts linear-model matrices (``y``, ``X``, optionally ``W``)
into a callable :class:`QuadraticForm` object that evaluates the weighted
sum of squared residuals (SSR):

    SSR(β) = y'Wy − 2 β'X'Wy + β' X'WX β  =  f0 + df_db · β + β' d2f_db2 β

together with a closed-form Gaussian log-likelihood function.

The quadratic form representation is useful for optimisers and Bayesian
samplers that need to evaluate or differentiate the objective_function without
recomputing X'X on every call.

Typical usage::

    llf, quad_form = SparseLinearModel(formula, data).get_quadratic_form_and_llf()
    ssr = quad_form(beta_candidate)
    log_p = llf(beta_candidate, sigma=0.5)
"""

from __future__ import absolute_import, print_function

import time

import numpy as np
from scipy.sparse import csc_matrix, isspmatrix

from kanly.dill_object import DillObject
from kanly.utils.linalg_utils import get_matrix_inverse_internal, gram_matrix


class QuadraticForm(DillObject):
    """Callable quadratic form representing the weighted sum of squared residuals.

    Encapsulates the three components needed to evaluate SSR(β) without
    re-accessing the raw data:

        SSR(β) = f0 + df_db · β + β' d2f_db2 β

    where ``f0 = y'Wy``, ``df_db = −2 X'Wy``, and ``d2f_db2 = X'WX``.

    The object supports batched evaluation: if ``beta`` is a 2-D array of
    shape ``(m, p)``, all ``m`` SSR values are returned simultaneously.

    Inherits ``DillObject`` for pickle-compatible serialisation.
    """

    def __init__(self, f0, df_db, d2f_db2):
        """Construct the quadratic form from its three pre-computed components.

        Args:
            f0 (float): Scalar ``y'Wy`` — the unparameterized sum of squares.
            df_db (ndarray, shape (p,)): Linear term ``-2 X'Wy``; the gradient
                of SSR at β = 0.
            d2f_db2 (ndarray or sparse matrix, shape (p, p)): Quadratic term
                ``X'WX``; the Hessian of SSR (positive semi-definite).
        """
        self.f0 = f0            # y'Wy
        self.df_db = df_db      # -2 * X'Wy
        self.d2f_db2 = d2f_db2  # X'WX

    def __call__(self, beta):
        """Evaluate the quadratic form SSR(β) at one or more parameter vectors.

        Args:
            beta (array-like, shape (p,) or (m, p)): Parameter vector(s).
                A 1-D input returns a scalar SSR; a 2-D input returns a
                1-D array of ``m`` SSR values (one per row).

        Returns:
            float or ndarray: SSR value(s).
        """
        beta = np.asarray(beta)
        if np.ndim(beta) == 1:
            return self.f0 + self.df_db.dot(beta) + np.dot(beta, self.d2f_db2).dot(beta)
        else:
            return (beta.dot(self.df_db)
                    + (beta.dot(self.d2f_db2) * beta).sum(axis=1)
                    ) + self.f0

    def minimize(self, return_ncp=False):
        """Return the OLS minimiser β* = (X'WX)^{-1} X'Wy.

        Args:
            return_ncp (bool): If ``True``, also return the normalised
                covariance matrix ``(X'WX)^{-1}`` alongside the solution.

        Returns:
            ndarray or tuple:
                - If ``return_ncp=False``: 1-D ndarray of shape ``(p,)``
                  containing the OLS parameter estimates.
                - If ``return_ncp=True``: ``(beta, ncp)`` where ``ncp`` is
                  the ``(p, p)`` normalised covariance matrix.
        """
        ncp = get_matrix_inverse_internal(self.d2f_db2)
        beta = ncp.dot(self.df_db/2)

        if return_ncp:
            return beta, ncp
        else:
            return beta

    def XtX(self):
        """Return a copy of the Gram matrix X'WX (the Hessian term).

        Returns:
            ndarray or sparse matrix, shape (p, p): Copy of ``d2f_db2``.
        """
        return self.d2f_db2.copy()

    def Xty(self):
        """Return a copy of the cross-product vector X'Wy.

        Returns:
            ndarray, shape (p,): Copy of ``-df_db / 2``.
        """
        return self.df_db.copy() / -2.0

    def yty(self):
        """Return the scalar y'Wy.

        Returns:
            float: Copy of ``f0``.
        """
        return self.f0.copy()


def _linear_model_components_2_quadratic_form_and_likelihood(y, X, weights=None):
    """Build a callable SSR quadratic form and a Gaussian log-likelihood from model data.

    Computes the three components of the SSR quadratic form (``f0``,
    ``df_db``, ``d2f_db2``) from ``y``, ``X``, and optional weights, then
    wraps them in a :class:`QuadraticForm` and a closure for the
    Gaussian log-likelihood.

    The log-likelihood function signature is ``llf(beta, sigma)`` where
    ``sigma`` is the **square root** of the model variance (i.e. the
    standard deviation σ, not the variance σ²).  This convention matches
    the docstring note in ``SparseLinearModel.get_quadratic_form_and_llf``.

    Both sparse (CSC) and dense (ndarray) ``y`` and ``X`` are supported.
    Mixed sparse/dense combinations are handled by normalising to the type
    of ``X``.

    Args:
        y (array-like or csc_matrix, shape (n,) or (n, 1)): Dependent
            variable vector.
        X (array-like or csc_matrix, shape (n, p)): Design matrix.
        weights (array-like, optional): Non-negative weight vector of
            length ``n``.  When ``None`` all observations are weighted equally.

    Returns:
        tuple:
            - **log_likelihood_func** (callable): ``llf(beta, sigma)`` →
              float or ndarray, evaluating the Gaussian log-likelihood at
              the given parameter vector(s) and error standard deviation.
            - **ssr_quad_form** (QuadraticForm): Callable quadratic form
              ``SSR(beta)`` ready for direct evaluation or minimisation.
    """

    _t = time.time()

    nobs = X.shape[0]
    is_sparse_X = isspmatrix(X)

    if is_sparse_X:
        y = csc_matrix(y).reshape((-1, 1))
    elif isspmatrix(y):
        y = y.toarray().flatten()

    if weights is None:
        if is_sparse_X:
            f0 = y.power(2).sum()
            df_db = -2.0 * (y.transpose().dot(X).toarray().flatten())
        else:
            f0 = (y ** 2).sum()
            df_db = -2.0 * (y.T.dot(X)).flatten()
        # d2f_db2 = X.transpose().dot(X).toarray()

    else:
        if isspmatrix(y):
            y = y.toarray().flatten()
        f0 = np.dot(y ** 2, weights)
        if is_sparse_X:
            df_db = -2.0 * csc_matrix(y * weights).dot(X).toarray().flatten()
        else:
            df_db = -2.0 * (y * weights).dot(X)

    d2f_db2 = gram_matrix(X, weights)

    ssr_quad_form = QuadraticForm(f0, df_db, d2f_db2)

    if weights is None:
        wt_term = 0.0
    else:
        wt_term = np.sum(np.log(weights[weights > 0])) / 2

    def log_likelihood_func(beta, sigma):
        """
        :param beta: params
        :param sigma: sqrt of model variance
        """
        sigma2 = np.asarray(sigma) ** 2
        ssr = ssr_quad_form(beta)
        val = -nobs / 2.0 * np.log(2 * np.pi * sigma2) - ssr / (2.0 * sigma2) + wt_term
        return val

    return log_likelihood_func, ssr_quad_form


# if __name__ == '__main__':
#     from scipy.sparse import csc_matrix
#     n = 2_000
#     p = 100
#     s = .3
#     np.random.seed(0)
#     X = np.random.randn(n, p).dot((s * np.eye(p) + (1 - s) * np.ones((p, p))))
#     z = np.ones(4)
#     beta = np.hstack([z, np.zeros(p - len(z))])
#     y = 3 + X.dot(beta) + np.random.randn(n)
#
#     for wts in [None, np.random.rand(n)]:
#         _linear_model_components_2_quadratic_form_and_likelihood(y, X, weights=wts)
#         _linear_model_components_2_quadratic_form_and_likelihood(y, csc_matrix(X), weights=wts)
#         _linear_model_components_2_quadratic_form_and_likelihood(csc_matrix(y), X, weights=wts)
#         _linear_model_components_2_quadratic_form_and_likelihood(csc_matrix(y), csc_matrix(X), weights=wts)



# if __name__ == '__main__':
#
#     import pandas as pd
#     from kanly.api import lm
#
#     n = 150
#     np.random.seed(0)
#     df = pd.DataFrame({
#         'x': np.random.randn(n),
#         'grp': np.random.randint(0, 12, n),
#         'obs': np.random.randint(1, 4, n),
#     })
#     df['y'] = 1.2 - 0.3 * df['x'] + np.sqrt(.2) * np.random.randn(n)
#     fit = lm('y ~ x + C(grp) $ obs', df)
#
#     print(fit)
#
#     llf, qf = _linear_model_components_2_quadratic_form_and_likelihood(
#         y := fit.model.endog, X := fit.model.exog, w := fit.model.weights)
#
#     theta = fit.params.values
#     print(qf(theta))
#     print(sum(w * (X.toarray().dot(theta) - y.toarray().flatten()) ** 2))
#     print(qf(np.vstack([theta]*3)))
#
#     print(llf(theta, fit.scale*(n-13)/n))
#     print(llf(np.vstack([theta]*3), [fit.scale*(n-13)/n]*3))

# #
# # def linear_model_2_normal_likelihood(model):
# #     f0, df_db, d2f_db2, quad_form = linear_model_2_quadratic_form(model)
# #     n = model.nobs
# #
# #     if model.weights is None:
# #         wt_term = 0.0
# #     else:
# #         wt_term = np.sum(np.log(model.weights)) / 2
# #
# #     def log_likelihood_normal(params):
# #         params = np.asarray(params)
# #         if np.ndim(params) == 1:
# #             beta, rt_scale = params[:-1], params[-1]
# #         else:
# #             beta, rt_scale = params[:, :-1], params[:, -1]
# #
# #         sst = quad_form(beta)
# #         sst /= (2.0 * rt_scale ** 2)
# #         val = -n * (np.log(2 * np.pi) / 2 + np.log(rt_scale)) - sst + wt_term
# #         return val
# #
# #     return log_likelihood_normal, quad_form, f0, df_db, d2f_db2
# #
# #
# # def linear_model_2_normal_likelihood_from_formula(formula, data, index=None, debug=False):
# #     model = SparseLinearModel.build_model_from_formula(formula, data, index=index, debug=debug)
# #     log_likelihood_normal, quad_form, f0, df_db, d2f_db2 = linear_model_2_normal_likelihood(model)
# #     return log_likelihood_normal, quad_form, f0, df_db, d2f_db2, model
#
#
# if __name__ == '__main__':
#
#     import pandas as pd
#     from kanly.api import lm
#
#     n = 150
#     np.random.seed(0)
#     df = pd.DataFrame({
#         'x': np.random.randn(n),
#         'grp': np.random.randint(0, 12, n),
#         'obs': np.random.randint(1, 6, n),
#     })
#     df['y'] = 1.2 - 0.3 * df['x'] + np.sqrt(.2) * np.random.randn(n)
#
#     fit = lm('y ~ x + C(grp) $ obs', df)
#     print(fit)
#     print('scale = ' , fit.scale)
#
#     model = fit.model
#     llf, ssr_quad_form \
#         = _linear_model_components_2_quadratic_form_and_likelihood(model.endog, model.exog, model.weights)
#     print(ssr_quad_form(fit.params), sum(fit.resid**2))
#     print(ssr_quad_form(fit.params), sum(df.obs * fit.resid**2))
#     print(llf(fit.params, fit.scale * (fit.df_resid / fit.nobs)))
#
#     print(fit.params, "*****", ssr_quad_form.minimize())