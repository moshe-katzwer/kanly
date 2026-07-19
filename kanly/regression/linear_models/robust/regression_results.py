"""Result container for fitted sparse robust linear regression models.

``SparseRobustLinearRegressionResults`` stores M-estimator fit metadata
(scale, cost, norm function, convergence diagnostics) and provides the
summary table and header/footer formatters used by the base class.
"""
from __future__ import absolute_import, print_function

import numpy as np

from kanly.regression.linear_models.robust.constants import DEFAULT_RLM_TEST_LEVEL, DEFAULT_RLM_USE_T
from kanly.regression.regression_results_base import RegressionResultsBase


class SparseRobustLinearRegressionResults(RegressionResultsBase):
    """Fitted results for a robust linear model estimated via M-estimation.

    Extends ``RegressionResultsBase`` with robust-regression-specific
    attributes and summary formatting.

    Key instance attributes set by ``__init__``:

    - ``params`` (ndarray)       — Coefficient estimates β̂, shape (p,).
    - ``bse`` (ndarray)          — Standard errors, shape (p,).
    - ``cov_params`` (ndarray)   — Covariance matrix, shape (p, p), or None.
    - ``cov_type`` (str)         — Covariance type (``'H1'``, …, ``'BOOTSTRAP'``).
    - ``resid`` (ndarray)        — Final residuals y − Xβ̂, shape (n,).
    - ``fittedvalues`` (ndarray) — Fitted values Xβ̂, shape (n,).
    - ``scale`` (float)          — MAD-based scale estimate σ̂.
    - ``pseudo_rsquared`` (float)— 1 − cost/naive_cost (robust analogue of R²).
    - ``cost`` (float)           — Final M-estimator cost Σρ(rᵢ/σ̂).
    - ``weights`` (ndarray)      — Final IRLS weights, shape (n,).
    - ``method`` (str)           — Always ``'IRLS'``.
    - ``M`` (RobustNormFunction) — Norm object used for estimation.
    - ``iteration_info`` (dict)  — Convergence diagnostics from ``rlm_internal``.
    - ``fit_elapsed`` (float)    — Wall-clock fit time in seconds.
    - ``df_model`` (int)         — Model degrees of freedom.
    - ``df_resid`` (int)         — Residual degrees of freedom.
    """

    def __init__(self, nobs, params, fittedvalues, compute_cov, cov_params, cov_type, resid, df_resid, df_model, scale,
                 pseudo_rsquared, cost, weights, model, M, fit_elapsed, use_t=DEFAULT_RLM_USE_T,
                 test_level=DEFAULT_RLM_TEST_LEVEL, keep_model=True, iteration_info=None, specification_name=None):
        """Initialise the results object and store all fit metadata.

        Args:
            nobs (int): Number of observations.
            params (ndarray): Coefficient estimates β̂, shape (p,).
            fittedvalues (ndarray): Fitted values Xβ̂, shape (n,).
            compute_cov (bool): Whether covariance was estimated.
            cov_params (ndarray or None): Covariance matrix (p, p), or None
                if ``compute_cov`` is False.
            cov_type (str): Covariance type string.
            resid (ndarray): Final residuals y − Xβ̂, shape (n,).
            df_resid (int): Residual degrees of freedom (n − p).
            df_model (int): Model degrees of freedom (p − 1 or p).
            scale (float): MAD-based scale estimate σ̂.
            pseudo_rsquared (float): Robust pseudo-R² = 1 − cost/naive_cost.
            cost (float): Final M-estimator cost Σρ(rᵢ/σ̂).
            weights (ndarray): Final IRLS weights, shape (n,).
            model (SparseRobustLinearModel): The model object; retained if
                ``keep_model`` is True.
            M (RobustNormFunction): The norm object used in estimation.
            fit_elapsed (float): Wall-clock fit time in seconds.
            use_t (bool): Use t-distribution for hypothesis tests.
            test_level (float): Significance level for CIs and p-values.
            keep_model (bool): Whether to attach ``model`` to this object.
            iteration_info (dict, optional): Convergence diagnostics from
                ``rlm_internal`` (``num_iters``, ``max_iter``, ``tol``,
                ``error``, ``converged``, ``force_scale``).
            specification_name (str, optional): Human-readable model label.
        """

        super().__init__(nobs, params, cov_params, df_model, df_resid, df_resid, exog_names=list(model.exog_names),
                         endog_name=model.endog_name, cov_type=cov_type, cov_kwds=None, test_level=test_level,
                         use_t=use_t, alpha=0, l1_ratio=0, specification_name=specification_name)

        self.set_properties_from_model(model, keep_model)

        self.compute_cov = compute_cov

        self.resid = resid
        self.fittedvalues = fittedvalues
        self.df_resid = df_resid
        self.df_model = df_model

        self.scale = scale
        self.pseudo_rsquared = pseudo_rsquared
        self.cost = cost
        self.weights = weights
        self.method = 'IRLS'

        self.M = M
        self.iteration_info = iteration_info
        self.fit_elapsed = fit_elapsed

    def get_result_type(self):
        """Return the display label for the results table header.

        Returns:
            str: ``'Robust Regression Results'``.
        """
        return 'Robust Regression Results'

    def get_header_info_array(self):
        """Return a 2-column array of label/value pairs for the summary header.

        Each row is a ``(label, value)`` pair displayed in the top section of
        the summary table.  Rows include:

        - dep. variable, weights, nobs
        - model type (``'rlm'``), method (``'IRLS'``), norm name
        - df_resid, df_model, use_t, covariance type
        - scale, force_scale flag, cost, convergence status
        - iteration count, model/fit elapsed time, max_iter, error, tolerance

        Returns:
            ndarray: Shape (n_rows, 2) array of string label/value pairs.
        """
        return np.array([
            ('dep. variable:', self.endog_name),
            ('weights:', self.weights_name),
            ('nobs:', self.nobs),
            ('model:', 'rlm'),
            ('method:', self.method),
            ('norm:', self.M.__class__.__name__),
            ('df resid:', self.df_resid),
            ('df model:', self.df_model),
            ('use_t:', self.use_t),
            ('covariance type:', self.cov_type.upper()),
            ('scale (est):', ("%.3e" % self.scale) if self.scale < .01 else np.round(self.scale, 3)),
            ('force scale:',  self.iteration_info['force_scale']),
            ('cost:', ("%.4e" % self.cost)),
            ('converged:', self.iteration_info['converged']),
            ('iterations:', self.iteration_info['num_iters']),
            ('model elapsed:', "%.2fs" % self.model_elapsed),
            ('fit elapsed:', "%.2fs" % self.fit_elapsed),
            ('max_iters:', self.iteration_info['max_iter']),
            ('error:', "%.2e" % self.iteration_info['error']),
            ('tol:', "%.2e" % self.iteration_info['tol']),
        ])

    def get_result_name(self):
        """Return a human-readable model name for display in results tables.

        Returns:
            str: ``'Robust Linear Model (M-estimation)'``.
        """
        return "Robust Linear Model (M-estimation)"

    def get_footer_info(self, *args, **kwargs):
        """Return footer text appended below the coefficient table.

        The iteration-info block is currently commented out.  When enabled it
        would display a ``pd.Series`` representation of ``iteration_info``.

        Returns:
            str: Empty string (footer is suppressed).
        """
        # if self.iteration_info is not None:
        #     iter_info_series = pd.Series(index=[f'{str(k)}:' for k in self.iteration_info.keys()],
        #                                  data=self.iteration_info.values())
        #     iter_info_str = '\n'.join(str(iter_info_series).split('\n')[:-1]) + '\n' #+ "=" * len_tbl
        # else:
        #     iter_info_str = ''
        # return iter_info_str
        return ''

    def __repr__(self):
        """Return ``str(self)`` so that repr and str are consistent.

        Returns:
            str: The full summary table string.
        """
        return str(self)

    def __str__(self):
        """Return the formatted summary table.

        Returns:
            str: Output of ``self.summary()``.
        """
        return self.summary()

    # def predict(self, data=None, params=None, integer_index=None, debug=False, *args, **kwargs):
    #     if params is None and data is None:
    #         return self.fittedvalues.copy()
    #     if params is None:
    #         params = self.params.copy()
    #     return self.model.predict(params, data=data, integer_index=integer_index, debug=debug)