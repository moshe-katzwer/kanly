"""
Generalized additive models (GAM) for kanly GLMs.

``SparseGeneralizedAdditiveModel`` extends :class:`~kanly.regression.generalized_linear_models.model.SparseGeneralizedLinearModel`
to fit smooth terms via **penalized IRLS**, in the spirit of statsmodels
`GLMGam <https://www.statsmodels.org/stable/generated/statsmodels.gam.generalized_additive_model.GLMGam.html>`_.

Workflow
--------
1. User passes ``penalty`` and ``df`` dicts keyed by smooth covariate names.
2. Each penalized column in the formula is rewritten to a cubic B-spline basis
   ``bs(x, degree=3, df=..., include_intercept=False)``.
3. A block-diagonal **roughness penalty matrix** is assembled from
   :func:`~kanly.nonparametric.bspline.bspline_penalty` (integrated squared second
   derivative) and stored as the model's ``L2_penalty_matrix``.
4. :meth:`~kanly.regression.generalized_linear_models.model.SparseGeneralizedLinearModel.fit`
   calls :func:`~kanly.regression.generalized_linear_models.sparse_glm_internal.glm_internal`,
   which adds ``L2_penalty_matrix`` to the IRLS normal equations ``X'WX`` each
   iteration — the same general matrix-penalty path exposed by ``GLM``.

Entry point: ``kanly.api.gam`` / ``GAM`` (alias of ``GLM`` on this class).
"""
from __future__ import absolute_import, print_function

import time
import numpy as np

from kanly.formula.data_getter import SparseDataGetter
from kanly.formula.keys import ENDOG_KEY, INSTRUMENTS_KEY, FORMULA_DESIGN_INFO_KEY, NULL_ROWS_INFO_DICT_KEY, \
    VALID_OBS_ROWS_KEY, EXOG_KEY, HAS_IMPLICIT_CONSTANT_KEY, HAS_INTERCEPT_KEY
from kanly.formula.parse_formula import parse_formula
from kanly.nonparametric.bspline import bspline_penalty
from kanly.regression.cov_types import format_cov_kwds
from kanly.regression.generalized_linear_models.constants import DEFAULT_GLM_CHECK_CONSTANT_COLS, GLM_COV_TYPES, \
    DEFAULT_GLM_FORCE_IV_PROJECTION, DEFAULT_GLM_PROMPT_USER_FOR_MORE_ITERS, METHOD_IRLS, \
    DEFAULT_GLM_RESIDUAL_INCLUSION, DEFAULT_GLM_TEST_LEVEL, DEFAULT_GLM_FAMILY, DEFAULT_GLM_TOL, \
    DEFAULT_GLM_RESIDUAL_INCLUSION_ORDER, DEFAULT_GLM_COV_TYPE, DEFAULT_GLM_MAX_ITER
from kanly.regression.generalized_linear_models.model import SparseGeneralizedLinearModel
from kanly.regression.linear_models.penalized import _check_penalties
from kanly.utils.util import dict_2_dataframe


class SparseGeneralizedAdditiveModel(SparseGeneralizedLinearModel):
    """
    Sparse GLM with optional smooth (B-spline) terms via an L2 roughness matrix.

    Inherits all GLM families, links, IV, and covariance options from
    :class:`~kanly.regression.generalized_linear_models.model.SparseGeneralizedLinearModel`.
    Smooth covariates listed in ``penalty`` / ``df`` are expanded to spline bases
    before fit; see :meth:`build_model_from_formula` and :meth:`gam`.

    See Also
    --------
    gam : formula entry point.
    kanly.regression.generalized_linear_models.sparse_glm_internal.glm_internal :
        IRLS core that adds ``L2_penalty_matrix`` to ``X'WX``.
    """

    GAM = SparseGeneralizedLinearModel.GLM

    @staticmethod
    def gam(formula, data, penalty=None, df=None, start_params=None, tol=DEFAULT_GLM_TOL, max_iter=DEFAULT_GLM_MAX_ITER,
            alpha=0.0,l1_ratio=0.0, debug=False, family=DEFAULT_GLM_FAMILY, link=None, normalize=True, penalize_scale=False,
            use_t=True, test_level=DEFAULT_GLM_TEST_LEVEL, compute_cov=True, store_convergence_path=False,
            residual_inclusion=DEFAULT_GLM_RESIDUAL_INCLUSION, cov_kwds=None, cov_type=DEFAULT_GLM_COV_TYPE,
            line_search_fallback=True, pick_default_start=True, opt_method=None, index=None,
            specification_name=None, residual_inclusion_order=DEFAULT_GLM_RESIDUAL_INCLUSION_ORDER,
            prompt_user_for_more_iters=DEFAULT_GLM_PROMPT_USER_FOR_MORE_ITERS,
            force_iv_projection=DEFAULT_GLM_FORCE_IV_PROJECTION, check_constant_cols=DEFAULT_GLM_CHECK_CONSTANT_COLS):
        """
        Fit a generalized additive model (GAM) via penalized IRLS.

        Smooth terms are specified implicitly: for each key in ``penalty`` / ``df``,
        the corresponding formula column is replaced by a cubic B-spline basis before
        fitting. Setting ``penalty[x]=0`` recovers an unpenalized GLM on the spline
        expansion (many parameters, wiggly fit); larger values shrink toward a smoother
        curve.

        Parameters
        ----------
        formula : str
            GLM-style formula (e.g. ``'y ~ x1 + x2'``). Columns named in ``penalty``
            become spline bases; others enter linearly as in ``glm``.
        data : DataFrame
            Data source.
        penalty : dict
            Roughness weight per smooth variable, e.g. ``dict(x2=0.1)``. Values are
            multiplied by 2 when building the internal ``L2_penalty_matrix`` (to
            match the quadratic-penalty convention used in IRLS). Use ``0`` for
            no penalization on that smooth.
        df : dict
            Spline degrees of freedom (basis dimension) per smooth variable, e.g.
            ``dict(x2=20)``. Keys must match ``penalty``.
        family, link, cov_type, cov_kwds, use_t, test_level, tol, max_iter, debug
            Same as :meth:`~kanly.regression.generalized_linear_models.model.SparseGeneralizedLinearModel.glm`.
        alpha, l1_ratio
            Elastic-net on top of a GAM roughness matrix is not combined in the
            current IRLS path; leave at 0 for standard GAM fits.

        Returns
        -------
        SparseGLMRegressionResults
            ``fit.is_gam`` is True; ``fit.edf`` gives effective d.f. per coefficient;
            summary shows ``Df Model`` as sum of EDF for GAM fits.

        Examples
        --------
        >>> from kanly.api import gam
        >>> fit = gam('y ~ x', df, penalty=dict(x=0.05), df=dict(x=15),
        ...           family='poisson')

        See Also
        --------
        glm : unpenalized or elastic-net GLM without spline expansion.
        build_model_from_formula : constructs the L2 roughness matrix from
            ``penalty`` / ``df``.
        """
        cov_kwds = format_cov_kwds(cov_kwds)
        opt_method = METHOD_IRLS

        cov_type = cov_type.upper()
        if cov_type not in GLM_COV_TYPES:
            raise Exception("`cov_type` must be one of %s!" % str(GLM_COV_TYPES))

        _check_penalties(alpha, l1_ratio)

        model = SparseGeneralizedAdditiveModel.build_model_from_formula(
            formula, data, penalty=penalty, df=df,
            debug=debug, index=index, specification_name=specification_name,
            check_constant_cols=check_constant_cols)

        fit = model.fit(
            family=family, link=link,
            start_params=start_params, tol=tol, max_iter=max_iter,
            alpha=alpha, l1_ratio=l1_ratio, debug=debug,
            normalize=normalize, penalize_scale=penalize_scale, use_t=use_t, test_level=test_level,
            compute_cov=compute_cov,
            store_convergence_path=store_convergence_path,
            residual_inclusion=residual_inclusion, cov_type=cov_type, cov_kwds=cov_kwds,
            line_search_fallback=line_search_fallback, pick_default_start=pick_default_start,
            opt_method=opt_method, fit_intercept=model.has_intercept, first_column_constant=model.has_intercept,
            specification_name=specification_name, residual_inclusion_order=residual_inclusion_order,
            force_iv_projection=force_iv_projection, prompt_user_for_more_iters=prompt_user_for_more_iters,
        )

        return fit

    @staticmethod
    def build_model_from_formula(formula, data, debug=False, index=None, specification_name=None,
                                 drop_1_for_FE=True, cov_groups=None,
                                 df=None, penalty=None,
                                 check_constant_cols=DEFAULT_GLM_CHECK_CONSTANT_COLS):
        """Build a sparse GAM model: spline expansion plus an L2 roughness matrix.

        Steps:
        1. Parse the user formula and rewrite each ``penalty`` key ``c`` to
           ``bs(c, degree=3, df=df[c], include_intercept=False)``.
        2. Build the sparse design via :class:`~kanly.formula.data_getter.SparseDataGetter`.
        3. For each penalized spline term, fill the corresponding block of the
           L2 penalty matrix with
           ``(2 * penalty[c]) * bspline_penalty(knots)`` on the corresponding
           coefficient indices.
        4. Attach the matrix via
           :meth:`~kanly.regression.generalized_linear_models.model.SparseGeneralizedLinearModel._set_L2_penalty_matrix`.

        Parameters
        ----------
        formula : str
            Response and linear covariates (smooth columns will be expanded).
        data : DataFrame
            Data source.
        df : dict
            Spline basis dimension per smooth variable name.
        penalty : dict
            Roughness penalty weight per smooth variable name (``0`` = none).
        debug, index, specification_name, drop_1_for_FE, cov_groups,
        check_constant_cols
            Passed through to formula parsing / design construction.

        Returns
        -------
        SparseGeneralizedAdditiveModel
            Model with ``L2_penalty_matrix`` set; call :meth:`fit` to run
            penalized IRLS.
        """

        _t = time.time()

        data = dict_2_dataframe(data)

        result = parse_formula(formula, debug=debug)

        # Map original covariate names -> bs(...) formula tokens for penalized smooths
        new_exog_2_orig_map = dict()
        for i, c in enumerate(result[EXOG_KEY]):
            if c in penalty:
                replace = f'bs({c}, degree=3, df={df[c]}, include_intercept=False)'
                result[EXOG_KEY][i] = replace
                new_exog_2_orig_map[replace] = c

        formula = result[ENDOG_KEY][0] + ' ~ ' + ' + '.join(result[EXOG_KEY])

        result = SparseDataGetter.get_data(data=data, formula=formula, debug=debug, index=index,
                                           check_constant_cols=check_constant_cols, drop_1_for_FE=drop_1_for_FE)

        endog, endog_name = result[ENDOG_KEY].values.toarray().flatten(), result[ENDOG_KEY].column_names[0]
        exog, exog_names, exog_term_names \
            = result[EXOG_KEY].values, result[EXOG_KEY].column_names, result[EXOG_KEY].term_names

        fit_intercept = result[HAS_INTERCEPT_KEY]
        has_implicit_constant = result[HAS_IMPLICIT_CONSTANT_KEY]
        valid_obs_rows = result[VALID_OBS_ROWS_KEY]
        null_rows_info_dict = result[NULL_ROWS_INFO_DICT_KEY]
        formula_design_info = result[FORMULA_DESIGN_INFO_KEY]

        # Placeholder; the roughness matrix is assembled once p is known.
        L2_penalty_matrix = None
        model = SparseGeneralizedAdditiveModel(
            exog, endog, False, fit_intercept, has_implicit_constant, formula_design_info, True,
            L2_penalty_matrix, weights=None,
            endog_name=endog_name, exog_names=exog_names, weights_name=None, instruments=None,
            instrument_names=None, valid_obs_rows=valid_obs_rows, index=index,
            null_rows_info_dict=null_rows_info_dict, model_elapsed=time.time() - _t,
            specification_name=specification_name
        )

        p = len(model.exog_names)
        L2_penalty_matrix = np.zeros((p,p))

        # Block-diagonal roughness penalties on each penalized spline coefficient block
        fdi = model.formula_design_info
        for t in fdi.exog_terms:
            for v in t.state['numerical']:
                v_orig = new_exog_2_orig_map.get(v, None)
                if v_orig in penalty:
                    knots = t.state['numerical'][v]['bspline']['knots']
                    # factor 2 aligns user penalty with quadratic form added to X'WX in IRLS
                    penalty_matrix = (penalty[v_orig] * 2) * bspline_penalty(knots, include_intercept=False)
                    idx = fdi.exog_var_2_col_indices[v]
                    L2_penalty_matrix[np.ix_(idx, idx)] = penalty_matrix

        model._set_L2_penalty_matrix(L2_penalty_matrix, regularize_to_values=None)

        model.gam_penalty_arg = penalty
        model.gam_df_arg = df

        if debug:
            print(model)

        return model


# if __name__ == '__main__':
#     import numpy as np
#     import pandas as pd
#     import statsmodels.api as sm
#
#     from statsmodels.gam.api import GLMGam, BSplines
#
#     from kanly.api import glm
#     import matplotlib.pyplot as plt
#
#     np.random.seed(0)
#     n2 = 100
#     x2 = np.random.randn(n2)
#     x2.sort()
#     y2 = np.exp(-4.5 + .5 * x2 - .25 * x2 ** 2 + .4 * np.random.randn(n2))
#     data2 = pd.DataFrame(dict(x2=x2, y2=y2))
#     df_2 = 8
#     penalty2 = .05
#
#     fit_glm = SparseGeneralizedAdditiveModel.gam(
#         'y2 ~ x2', data2,
#         penalty=dict(x2=0),
#         df=dict(x2=df_2), family='poisson')
#
#     fit_gam = SparseGeneralizedAdditiveModel.gam(
#         'y2 ~ x2', data2,
#         penalty=dict(x2=penalty2),
#         df=dict(x2=df_2), family='poisson', cov_type='nonrobust')
#     print(fit_gam)
#
#     bs = BSplines(data2[['x2']], df=[df_2 + 1], degree=[3])
#
#     # penalization weight
#     alpha = np.array([penalty2])
#
#     fit_gamsm = GLMGam.from_formula('y2 ~ 1', data=data2, smoother=bs, alpha=alpha,
#                                   family=sm.families.Poisson()).fit()
#     print(fit_gamsm.summary())
#
#     print(fit_gam.edf)
#     print(fit_gamsm.edf)
#
#     plt.scatter(x2, y2, alpha=.3, c='y')
#     plt.plot(x2, fit_glm.fittedvalues, lw=3)
#     plt.plot(x2, fit_gam.fittedvalues, lw=3)
#     plt.plot(x2, fit_gamsm.fittedvalues, lw=3, ls='--')
#     plt.yscale('log')
#     plt.show()
