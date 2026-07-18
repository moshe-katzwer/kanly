"""
Marginal effects and (semi-)elasticities for fitted GLMs.

Computes average and point effects of regressors on the **fitted mean**
``mu = g^{-1}(eta)`` with ``eta = X beta``, in the spirit of statsmodels
`GLMResults.get_margeff
<https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLMResults.get_margeff.html>`_.

Four effect types are supported (statsmodels naming):

- ``dydx`` — marginal effect            ``d mu / d x_k``
- ``eydx`` — semi-elasticity (y in %)   ``(d mu / d x_k) / mu``
- ``eyex`` — elasticity                 ``(d mu / d x_k) * x_k / mu = d ln(mu)/d ln(x_k)``
- ``dyex`` — semi-elasticity (x in %)   ``(d mu / d x_k) * x_k = d mu / d ln(x_k)``

For continuous covariates the ``dydx`` effect is ``g'(eta) * beta_k`` (chain
rule).  For 0/1 dummy columns the ``dx`` effects can be computed as the average
**discrete** change ``E[g(eta | x_k=1) - g(eta | x_k=0)]`` (``dummy_method=
'secant'``) rather than the tangent slope; the ``ex`` effects always treat
dummies as continuous, since a percent change in a 0/1 regressor is ill-defined.

Standard errors use the delta method: ``cov(me) = J @ cov(beta) @ J'`` where ``J``
is the Jacobian of the effect vector with respect to ``beta``.

Entry point: :meth:`~kanly.regression.generalized_linear_models.regression_results.SparseGLMRegressionResults.get_marginal_effects`.
"""
from __future__ import absolute_import, print_function

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.sparse import isspmatrix, diags
from scipy.stats import norm
import datetime

from kanly.dill_object import DillObject
from kanly.regression.generalized_linear_models.links import Identity
from kanly import __version__

if TYPE_CHECKING:
    from kanly.regression.generalized_linear_models.regression_results import SparseGLMRegressionResults


class GLMMarginalEffects(DillObject):

    eff_label = {'dydx': 'dy/dx', 'eydx': 'ey/dx',
                 'eyex': 'ey/ex', 'dyex': 'dy/ex'}

    def __init__(self, margeff, margeff_cov, at, effect_type, dummy, dummy_method, dummies, fit):
        self.margeff = margeff
        self.margeff_cov = margeff_cov
        self.at = at
        self.effect_type = effect_type
        self.dummy = dummy
        self.dummy_method = dummy_method
        self.dummies = dummies
        self.fit = fit
        self.endog_name = fit.endog_name
        self.exog_names = fit.exog_names[fit.has_intercept:]

        self.has_cov = self.margeff_cov is not None

        if self.has_cov:
            self.margeff_se = np.diag(self.margeff_cov) ** .5
            self.margeff_zvalues = self.margeff / self.margeff_se
            self.margeff_pvalues = 2 * norm.sf(np.abs(self.margeff_zvalues))
        else:
            self.margeff_se, self.margeff_pvalues, self.margeff_zvalues = None, None, None

        self.date = datetime.datetime.today().strftime('%b %d, %Y')
        self.timestamp = datetime.datetime.today().strftime('%H:%M:%S')

    def summary_df(self, test_level=.05):
        df = pd.DataFrame({self.eff_label[self.effect_type]: self.margeff})
        df.index = self.exog_names
        if self.has_cov:
            df['std err'] = self.margeff_se
            df['z'] = self.margeff_zvalues
            df['p>|z|'] = self.margeff_pvalues
            cv = -norm.ppf(test_level / 2)
            df[f'[{test_level / 2:.3f}, '] = self.margeff - cv * self.margeff_se
            df[f'{1 - test_level / 2:.3f}]'] = self.margeff + cv * self.margeff_se
        return df

    def summary(self, test_level=.05):
        header = np.array([
            ("Dep. Var.", self.endog_name),
            ("Method:", self.effect_type),
            ("At:", self.at),
            ("Date:", self.date),
            ("Time:", self.timestamp),
        ])
        header = pd.Series(index=header[:,0], data=header[:,1])

        summary_df = self.summary_df(test_level)
        summary_df_strs = summary_df.to_string().split('\n')
        len_ = len(summary_df_strs[0])
        bar = '─' * len_
        dblbar = '═' * len_


        ver_str = f"[kanly v={__version__}]"
        ver_str = " " * (len_ - len(ver_str)) + ver_str

        res = (
                dblbar + "\n"
                + "GLM Marginal Effects\n"
                + ("" if self.fit.specification_name is None else (self.fit.specification_name + "\n"))
                + bar + '\n'
                + "\n"
                + "\n".join(str(header).split("\n")[:-1]) + "\n\n"
                + dblbar + "\n" + summary_df_strs[0] + "\n" + bar + "\n"
                + "\n".join(summary_df_strs[1:]) + "\n" + bar + "\n"
                + ver_str
        )

        return res

    def __str__(self):
        return self.summary()

    def __repr__(self):
        return self.summary()


def sparse_col_medians(X):
    """Column medians for a CSC sparse matrix including implicit zeros.

    Each column's median is taken over all ``n`` rows; unstored entries count as
    zero.  Used for ``at='median'`` when the design matrix is sparse.

    Parameters
    ----------
    X : scipy.sparse matrix
        Design matrix; converted to CSC if needed.

    Returns
    -------
    ndarray
        Shape ``(p,)`` column medians.
    """
    X = X.tocsc()  # efficient column access; no-op if already CSC
    n, p = X.shape
    indptr, data = X.indptr, X.data
    med = np.empty(p)

    for j in range(p):
        vals = data[indptr[j]:indptr[j + 1]]  # only the stored nonzeros of column j
        neg = np.sort(vals[vals < 0])
        pos = np.sort(vals[vals > 0])
        n_neg, n_pos = neg.size, pos.size
        n_zero = n - n_neg - n_pos  # implicit + explicit zeros

        def value_at(k):  # k = 0-based order statistic
            if k < n_neg:
                return neg[k]
            elif k < n_neg + n_zero:
                return 0.0
            else:
                return pos[k - n_neg - n_zero]

        if n & 1:
            med[j] = value_at(n // 2)
        else:
            med[j] = 0.5 * (value_at(n // 2 - 1) + value_at(n // 2))

    return med


def _col(X, k):
    """Return column ``k`` of ``X`` as a 1-D dense array."""
    if isspmatrix(X):
        return np.asarray(X[:, k].todense()).ravel()
    return np.asarray(X[:, k]).ravel()


def _Tmv(X, v):
    """Return ``X.T @ v`` as a flat ``(p,)`` ndarray (robust to sparse / np.matrix)."""
    return np.asarray(X.T @ v).ravel()


def get_dummy_cols(X):
    """Detect design columns that are (possibly sparse) 0/1 indicators.

    Parameters
    ----------
    X : ndarray or scipy.sparse matrix
        Fitted design matrix (including intercept column if present).

    Returns
    -------
    dict
        Maps column index ``i`` to ``True`` if every stored value in column ``i``
        is ``0`` or ``1`` (implicit zeros in sparse columns count as zero).
    """
    p = X.shape[1]
    is_sp = isspmatrix(X)
    dummies = {}
    for i in range(p):
        if is_sp:
            x = X.data[X.indptr[i]:X.indptr[i + 1]]
        else:
            x = X[:, i]
        dummies[i] = bool(np.isin(x, [0, 1]).all())

    return dummies


def _get_marginal_effects(fit: "SparseGLMRegressionResults",
                          at='overall', effect_type='dydx',
                          dummy=True, dummy_method='secant', test_level=.05,
                          link=None,
                          ):
    """Compute GLM marginal effects / elasticities and delta-method standard errors.

    Effects are on the **response mean** ``mu = g^{-1}(X beta)``, not on the
    linear predictor.  The intercept column is excluded from returned vectors
    and inference when the model includes one.

    Parameters
    ----------
    fit : SparseGLMRegressionResults
        A converged GLM fit with ``cov_params()`` available.
    at : {'overall', 'mean', 'median', 'all'}, optional
        Where to evaluate effects (statsmodels ``get_margeff`` naming):

        - ``'overall'`` — average of observation-level effects; default.
        - ``'mean'`` — evaluate at ``x* = mean(X)`` (column-wise).
        - ``'median'`` — evaluate at ``x* = median(X)``; not allowed with dummies.
        - ``'all'`` — return an ``(n, p)`` matrix of per-observation effects;
          no standard errors or summary table.
    effect_type : {'dydx', 'eydx', 'eyex', 'dyex'}, optional
        dydx : marginal effect           ``d mu / d x_k`` (default)
        eydx : semi-elasticity (y in %)  ``(d mu / d x_k) / mu``
        eyex : elasticity                ``(d mu / d x_k) * x_k / mu``
        dyex : semi-elasticity (x in %)  ``(d mu / d x_k) * x_k``
    dummy : bool, optional
        If True, detect columns whose entries are all ``0`` or ``1`` and (for the
        ``dx`` effects, subject to ``dummy_method``) treat them as discrete
        indicators rather than continuous regressors.
    dummy_method : {'secant', 'tangent'}, optional
        How detected dummies are handled in the ``dx`` effects (``dydx``,
        ``eydx``): ``'secant'`` uses the discrete 0→1 change with its
        finite-difference Jacobian; ``'tangent'`` uses the continuous formula.
        Ignored for the ``ex`` effects (``eyex``, ``dyex``), which always treat
        dummies as continuous.
    test_level : float, optional
        Two-sided significance level for normal critical values and confidence
        intervals (default ``0.05`` → 95% CI).

    Returns
    -------
    dict
        Keys include:

        - ``marginal_eff`` — effect vector (or ``(n, p)`` array if ``at='all'``)
        - ``marginal_eff_cov``, ``marginal_eff_se`` — delta-method covariance / SEs
        - ``marginal_eff_zvalues``, ``marginal_eff_pvalues``
        - ``marginal_eff_ci_lo``, ``marginal_eff_ci_hi``
        - ``summary_df`` — :class:`pandas.DataFrame` with effect, SEs, z, p, CI
        - ``dummies``, ``at``, ``effect_type``, ``dummy``, ``dummy_method``,
          ``test_level`` — settings used

        For ``at='all'`` the inference keys are ``None`` (no covariance is
        generated for per-observation effects).

    Raises
    ------
    ValueError
        If ``at``, ``effect_type``, or ``dummy_method`` is not recognized.
    Exception
        If ``at='median'`` with ``dummy=True`` and dummy columns are detected.

    Notes
    -----
    Link handling. ``g`` is the inverse link, ``g'`` / ``g''`` its derivatives
    w.r.t. ``eta`` (from the ``Link`` object).  ``eydx``/``eyex`` additionally
    use ``r = g'/mu`` and ``r' = (g'' mu - g'^2)/mu^2 = d r / d eta``.

    Jacobian structure. ``dydx`` and ``eydx`` have a rank-one continuous
    Jacobian ``J = a I + b beta sᵀ``; ``eyex`` and ``dyex`` carry an explicit
    ``x_k`` factor so their Jacobian is a full ``p×p`` matrix.  The covariance
    is always formed as ``J V Jᵀ``; for a pure-continuous ``dydx`` (or any ``dx``
    effect with ``dummy_method='tangent'``) this reproduces the factored
    expansion exactly.
    """
    beta = fit.params.values
    V = fit.cov_params().values
    X = fit.model.exog
    n, p = X.shape

    if link is None:
        if hasattr(fit, 'link'):
            link = fit.link
        else:
            link = Identity()

    g = link.inverse_link
    gprime = link.deriv_inverse_link
    gprime2 = link.deriv2_inverse_link

    if hasattr(fit, 'lin_pred'):
        eta = np.asarray(fit.lin_pred).ravel()
    else:
        eta = np.asarray(fit.fittedvalues).ravel()

    if at not in ('overall', 'mean', 'median', 'all'):
        raise ValueError(f"unknown at={at!r}")
    if effect_type not in ('dydx', 'eydx', 'eyex', 'dyex'):
        raise ValueError(f"unknown effect_type={effect_type!r}")
    if dummy_method not in ('secant', 'tangent'):
        raise ValueError(f"unknown dummy_method={dummy_method!r}")

    if dummy:
        dummies = get_dummy_cols(X)
        dummy_idx = [i for i, d in dummies.items() if d]
    else:
        dummies = dict()
        dummy_idx = []

    if at == 'median' and len(dummy_idx):
        raise Exception("Can't do `median` with dummy variables!")

    # secant dummy handling only applies to the dx effects; the ex effects
    # always treat dummies as continuous (percent change in 0/1 is ill-defined)
    is_dx = effect_type in ('dydx', 'eydx')
    do_secant = dummy and (dummy_method == 'secant') and is_dx

    # ------------------------------------------------------------------ #
    #  at='all' : per-observation effects, no inference / covariance
    # ------------------------------------------------------------------ #
    if at == 'all':
        if effect_type == 'dydx':
            me = np.outer(gprime(eta), beta)              # g'(η_i) β_k
        elif effect_type == 'eydx':
            r = gprime(eta) / g(eta)                       # r_i = g'/μ
            me = np.outer(r, beta)                         # r_i β_k
        elif effect_type == 'eyex':
            r = gprime(eta) / g(eta)
            me = (r[:, None] * beta) * (np.asarray(X.todense()) if isspmatrix(X) else np.asarray(X))
        else:  # dyex
            me = (gprime(eta)[:, None] * beta) * (np.asarray(X.todense()) if isspmatrix(X) else np.asarray(X))

        # overwrite dummy columns with the per-observation discrete change
        if do_secant:
            for k in dummy_idx:
                xk = _col(X, k)
                eta0 = eta - xk * beta[k]                  # x_k = 0
                eta1 = eta0 + beta[k]                      # x_k = 1
                g0, g1 = g(eta0), g(eta1)
                if effect_type == 'dydx':
                    me[:, k] = g1 - g0
                else:  # eydx : relative change
                    me[:, k] = (g1 - g0) / g0

        me = me[:, fit.has_intercept:]
        return GLMMarginalEffects(
            me, None, at, effect_type,
            dummy, dummy_method, dummies, fit
        )

    # ------------------------------------------------------------------ #
    #  evaluation point for the MEM-style cases (shared by all effects)
    # ------------------------------------------------------------------ #
    if at in ('mean', 'median'):
        if at == 'mean':
            x_star = np.asarray(X.mean(axis=0)).ravel()
        elif isspmatrix(X):
            x_star = sparse_col_medians(X)
        else:
            x_star = np.median(X, axis=0)
        x_star = np.asarray(x_star).ravel()
        eta_star = float(x_star @ beta)

    me = np.empty(p)
    J = np.empty((p, p))

    # ================================================================== #
    #  dx effects: rank-one continuous Jacobian + optional secant dummies
    # ================================================================== #
    if is_dx:
        if at == 'overall':
            if effect_type == 'dydx':
                a = gprime(eta).mean()                     # ḡ′
                s = _Tmv(X, gprime2(eta))                  # Xᵀ g″(η)
            else:  # eydx
                mu = g(eta)
                r = gprime(eta) / mu                       # d ln μ / dη
                rprime = (gprime2(eta) * mu - gprime(eta) ** 2) / mu ** 2
                a = r.mean()                               # r̄
                s = _Tmv(X, rprime)                        # Xᵀ r′
            b = 1.0 / n
        else:                                              # 'mean' / 'median'
            if effect_type == 'dydx':
                a = gprime(eta_star)                       # g′(η*)
                s = gprime2(eta_star) * x_star             # g″(η*) · x*
            else:  # eydx
                mu_s = g(eta_star)
                a = gprime(eta_star) / mu_s                # r(η*)
                rprime_s = (gprime2(eta_star) * mu_s - gprime(eta_star) ** 2) / mu_s ** 2
                s = rprime_s * x_star                      # r′(η*) · x*
            b = 1.0

        # continuous rows: rank-one J = a I + b β sᵀ ; point est = a β
        me[:] = a * beta
        J[:] = a * np.eye(p) + b * np.outer(beta, s)

        # overwrite dummy rows with the discrete-change (secant) treatment,
        # unless dummy_method='tangent' (then they stay as the rows above)
        if do_secant:
            for k in dummy_idx:
                bk = beta[k]
                if at == 'overall':
                    xk = _col(X, k)
                    eta0 = eta - xk * bk                   # x_k = 0
                    eta1 = eta0 + bk                       # x_k = 1
                    g0, g1 = g(eta0), g(eta1)
                    v0, v1 = gprime(eta0), gprime(eta1)
                    if effect_type == 'dydx':
                        me[k] = (g1 - g0).mean()
                        row = _Tmv(X, v1 - v0) / n
                        row[k] = v1.sum() / n              # all-ones flip of col k
                    else:  # eydx : relative change (g1 - g0)/g0
                        me[k] = ((g1 - g0) / g0).mean()
                        p1 = v1 / g0                       # ∂/∂β via numerator g1
                        p0 = g1 * v0 / g0 ** 2             # ∂/∂β via denominator g0
                        row = _Tmv(X, p1 - p0) / n
                        row[k] = p1.sum() / n
                    J[k, :] = row
                else:                                      # 'mean' / 'median'
                    e0 = eta_star - x_star[k] * bk
                    e1 = e0 + bk
                    x0 = x_star.copy(); x0[k] = 0.0
                    x1 = x_star.copy(); x1[k] = 1.0
                    g0, g1 = g(e0), g(e1)
                    v0, v1 = gprime(e0), gprime(e1)
                    if effect_type == 'dydx':
                        me[k] = g1 - g0
                        J[k, :] = v1 * x1 - v0 * x0
                    else:  # eydx
                        me[k] = (g1 - g0) / g0
                        J[k, :] = (v1 / g0) * x1 - (g1 * v0 / g0 ** 2) * x0

    # ================================================================== #
    #  ex effects: full Jacobian, dummies treated as continuous
    # ================================================================== #
    else:  # 'eyex' or 'dyex'
        if at == 'overall':
            if effect_type == 'eyex':
                mu = g(eta)
                rate = gprime(eta) / mu                    # r = g′/μ
                rate_p = (gprime2(eta) * mu - gprime(eta) ** 2) / mu ** 2   # r′
            else:  # dyex
                rate = gprime(eta)                         # g′
                rate_p = gprime2(eta)                      # g″
            c = _Tmv(X, rate) / n                          # (1/n) Xᵀ rate
            me[:] = beta * c
            # J = diag(c) + (1/n) diag(β) (Xᵀ diag(rate′) X)
            # Dense: scale rows of X by rate′ via broadcasting -- O(np) temp,
            # then the O(np²) matmul. Never np.diag(rate_p) (that is n×n).
            # Sparse: diags(rate_p) stores only n nonzeros, so it stays cheap;
            # densify the small p×p result at the end.
            if isspmatrix(X):
                M = np.asarray((X.T @ (diags(rate_p) @ X)).todense())
            else:
                Xd = np.asarray(X)
                M = Xd.T @ (rate_p[:, None] * Xd)
            J[:] = np.diag(c) + (beta[:, None] / n) * M
        else:                                              # 'mean' / 'median'
            mu_s = g(eta_star)
            if effect_type == 'eyex':
                rate_s = gprime(eta_star) / mu_s
                rate_p_s = (gprime2(eta_star) * mu_s - gprime(eta_star) ** 2) / mu_s ** 2
            else:  # dyex
                rate_s = gprime(eta_star)
                rate_p_s = gprime2(eta_star)
            c = rate_s * x_star
            me[:] = beta * c
            # J = diag(c) + rate′(η*) (β⊙x*) x*ᵀ
            J[:] = np.diag(c) + rate_p_s * np.outer(beta * x_star, x_star)

    # ------------------------------------------------------------------ #
    #  delta-method covariance, drop intercept, inference, summary
    # ------------------------------------------------------------------ #
    # Drop intercept row from effects and covariance when present.
    me = me[fit.has_intercept:]
    cov = J @ V @ J.T
    cov = cov[fit.has_intercept:, fit.has_intercept:]

    return GLMMarginalEffects(
        me, cov, at, effect_type, dummy, dummy_method, dummies, fit
    )
