"""
Bayesian Linear Regression with conjugate NormalInverseGammaPrior
"""
from __future__ import absolute_import, print_function

import time

import numpy as np

from kanly.utils.linalg_utils import sandwich_diagonal
from pandas import DataFrame
from scipy.linalg import pinv
from scipy.sparse import isspmatrix, csc_matrix

from kanly.bayes.bayesian_linear_regression_conjugate_prior.bayesian_linear_regression_results import BayesianLinearRegressionResults
from kanly.bayes.bayesian_linear_regression_conjugate_prior.normal_inverse_gamma import NormalInverseGamma
from kanly.formula.data_getter import (SparseDataGetter, EXOG_KEY, ENDOG_KEY, WEIGHTS_KEY, HAS_INTERCEPT_KEY,
                                       HAS_IMPLICIT_CONSTANT_KEY, ABSORB_KEY, INSTRUMENTS_KEY, FORMULA_DESIGN_INFO_KEY)
from kanly.regression.linear_model_base import LinearModelBase
from kanly.regression.linear_models.linear_model_2_quadratic_form import \
    _linear_model_components_2_quadratic_form_and_likelihood


class BayesianLinearRegressionConjugatePriorModel(LinearModelBase):
    """Closed-form Bayesian linear regression under a Normal-Inverse-Gamma prior.

    Exact (no MCMC) posterior over ``(beta, sigma2)`` for the linear model
    ``y = X beta + eps``, ``eps ~ N(0, sigma2)``. Two entry points are
    exposed on ``kanly.api``:

    - ``blm(formula, data, ...)``  — Patsy formula API.
    - ``BLM(y, X, ...)``           — matrix API.

    Examples
    --------
    Formula API with mildly informative priors on the slope:

    >>> import numpy as np, pandas as pd
    >>> from kanly.api import blm
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({'x':   rng.normal(size=200),
    ...                    'grp': rng.integers(0, 5, 200)})
    >>> df['y'] = 1.2 - 0.3 * df['x'] + 0.2 * rng.normal(size=200)
    >>> fit = blm('y ~ x + C(grp)', df,                    # doctest: +SKIP
    ...           mu0={'x': 0.0},
    ...           Lambda0={'x': 1.0})

    See Also
    --------
    :func:`blm`, :func:`BLM`.
    """

    def __init__(self, endog, exog, add_constant, has_intercept, has_implicit_constant, formula_design_info, data, index,
                 param_names=None,
                 exog_names=None, endog_name=None, weights=None, weights_name=None, a0=0, b0=0, mu0=None, Lambda0=None,
                 model_elapsed=0,
                 specification_name=None):
        """Initialize conjugate-prior Bayesian linear model.

        Args:
            endog: Outcome vector/matrix.
            exog: Design matrix.
            add_constant: Whether a constant term is added by constructor flow.
            has_intercept: Whether explicit intercept is present in ``exog``.
            has_implicit_constant: Whether regressors imply an implicit constant.
            formula: Optional formula string used to build model.
            from_formula: Whether model was created from formula parser.
            data: Original source data object.
            index: Optional row-index selector used during build.
            param_names: Optional ordered parameter names.
            exog_names: Optional regressor names.
            endog_name: Optional outcome name.
            weights: Optional observation weights.
            weights_name: Optional weight column name.
            a0: Inverse-gamma shape hyperparameter.
            b0: Inverse-gamma scale hyperparameter.
            mu0: Prior mean hyperparameter for beta.
            Lambda0: Prior precision hyperparameter for beta.
            model_elapsed: Model-build elapsed seconds.
            specification_name: Optional model label.
        """
        super().__init__(endog, exog, add_constant, has_intercept, has_implicit_constant, formula_design_info, weights=weights,
                         instruments=None, endog_name=endog_name, absorb=None, absorb_names=None, absorb_term_name=None,
                         cov_groups=None, cov_groups_name=None, exog_names=exog_names,
                         weights_name=weights_name,
                         instrument_names=None, index=index, valid_obs_rows=None,
                         null_rows_info_dict=None, method=None, specification_name=specification_name,
                         endog_regressors=None,
                         model_elapsed=model_elapsed, is_sure=False, parent_model=None)
        if param_names is None:
            param_names = list(self.exog_names) + ['__sigma2']
        self.param_names = param_names
        self.num_params = len(self.param_names)

        if isinstance(mu0, dict):
            mu0 = np.array([mu0.get(k, 0) for k in self.exog_names])
        if isinstance(Lambda0, dict):
            Lambda0 = np.diag([Lambda0.get(k, 0) for k in self.exog_names])

        self.prior = NormalInverseGamma(mu0, Lambda0, a0, b0, self.num_params - 1)

        _, self.quad_form = _linear_model_components_2_quadratic_form_and_likelihood(
            self.endog, self.exog, self.weights)

        self.XtX, self.Xty, self.yty = self.quad_form.XtX(), self.quad_form.Xty(), self.quad_form.yty()
        self._log_pdf_likelihood = lambda beta, sigma2: -self.quad_form(beta) / (2 * sigma2) - self.nobs / 2 * np.log(
            sigma2)

    def _log_pdf_prior(self, params):
        """Evaluate the conjugate prior kernel for ``[beta..., sigma2]``.

        Args:
            params: Parameter vector ``[beta..., sigma2]``.
        """
        return self.prior(params)

    def _log_pdf_posterior(self, params):
        """Evaluate posterior kernel as prior + Gaussian likelihood term.

        Args:
            params: Parameter vector ``[beta..., sigma2]``.
        """
        params = np.array(params)
        return self.prior(params) + self._log_pdf_likelihood(params[:-1], params[-1])

    def accepts_multi_outcome(self):
        """Indicate whether model supports multi-outcome targets.

        Args:
            None.
        """
        return False

    def predict(self, params, data=None, index=None, debug=False, *args, **kwargs):
        """Predict mean response values.

        Args:
            params: Coefficient vector, optionally including trailing ``sigma2``.
            data: Optional new data for prediction.
            index: Optional row index subset.
            debug: If True, emit debug details.
            *args: Forwarded positional args.
            **kwargs: Forwarded keyword args.
        """
        # if data is not None or index is not None:
        #     raise NotImplementedError("Not implemented for `index` or `data`")
        if len(params) == self.num_params:
            params = params[:-1]
        return self.get_linear_predictor(params, data=data, index=index, debug=debug)

    def fit(self, a0=None, b0=None, mu0=None, Lambda0=None) -> BayesianLinearRegressionResults:
        """Compute posterior hyperparameters analytically and return result summary object.

        Args:
            a0: Optional inverse-gamma shape override.
            b0: Optional inverse-gamma scale override.
            mu0: Optional prior mean override.
            Lambda0: Optional prior precision override.
        """

        __t0 = time.time()

        if not np.any([x is not None for x in (a0, b0, mu0, Lambda0)]):
            if a0 is None: a0 = self.prior.a
            if b0 is None: b0 = self.prior.b
            if mu0 is None: mu0 = self.prior.mu
            if Lambda0 is None: Lambda0 = self.prior.Lambda
            self.prior = NormalInverseGamma(mu0, Lambda0, a0, b0, self.num_params - 1)

        mu0, Lambda0, a0, b0, flat_beta \
            = self.prior.mu, self.prior.Lambda, self.prior.a, self.prior.b, self.prior.flat_beta

        # Closed-form posterior update for Normal-Inverse-Gamma prior.
        n, p = self.exog.shape
        Lambda_n = self.XtX + (0 if flat_beta else Lambda0)
        mu_n = pinv(Lambda_n).dot(self.Xty + (0 if flat_beta else mu0.dot(Lambda0)))

        a_n = a0 + n / 2
        b_n = b0 + (self.yty - mu_n.dot(Lambda_n).dot(mu_n)
                    + (0 if flat_beta else mu0.dot(Lambda0).dot(mu0))
                    ) / 2

        # fitted values
        if isspmatrix(self.exog):
            fittedvalues = self.exog.dot(csc_matrix(mu_n).reshape((-1, 1))).toarray().flatten()
        else:
            fittedvalues = self.exog.dot(mu_n)
        _y = self.endog.toarray().flatten() if isspmatrix(self.endog) else self.endog
        resid = _y - fittedvalues

        # get posterior predictive (not full, just diagonal)
        # TODO
        # scale_beta = pinv(Lambda_n)
        # t_df = 2 * a_n
        # var_beta = (b_n / a_n) * scale_beta * t_df / (t_df - 2)
        # sigma2 =
        # posterior_predictive_std = np.sqrt(sandwich_diagonal(self.exog, var_beta) + sigma2)

        rsquared = self._get_rsquared(_y, resid, self.is_weighted, self.weights)

        fit_elasped = time.time() - __t0

        fit = BayesianLinearRegressionResults(
            self.nobs, self.num_params,
            self.is_weighted,
            self.prior.copy(),
            NormalInverseGamma(mu_n, Lambda_n, a_n, b_n),
            self.has_intercept, self.has_implicit_constant,
            fittedvalues=fittedvalues, resid=resid,
            rsquared=rsquared,
            fit_elasped=fit_elasped, model_elapsed=self.model_elapsed,
            endog_name=self.endog_name,
            weights_name=self.weights_name,
            param_names=self.param_names, model=self, specification_name=self.specification_name
        )
        return fit

    @staticmethod
    def _get_rsquared(_y, resid, is_weighted, weights):
        """Compute weighted or unweighted R-squared from fitted residuals.

        Args:
            _y: Observed outcomes.
            resid: Residual vector.
            is_weighted: Whether weighted fitting was used.
            weights: Observation weights.
        """
        if is_weighted:
            if isspmatrix(weights):
                w = weights.toarray().flatten()
            else:
                w = weights
            wtd_mean = np.average(_y, weights=w)
            rsquared = 1.0 - np.average(resid ** 2, weights=w) / np.average((_y - wtd_mean) ** 2, weights=w)
        else:
            rsquared = 1.0 - np.sum(resid ** 2) / np.sum((_y - _y.mean()) ** 2)
        return rsquared

    def build_model(self, data, index=None, debug=False, a0=None, b0=None, mu0=None, Lambda0=None,
                    strip_non_exog=True, check_constant_cols=True, drop_1_for_FE=True, specification_name=None):
        """Rebuild model from stored formula and new inputs.

        Args:
            data: New data source.
            index: Optional index subset.
            debug: Debug flag.
            a0: Optional inverse-gamma shape override.
            b0: Optional inverse-gamma scale override.
            mu0: Optional prior mean override.
            Lambda0: Optional prior precision override.
            strip_non_exog: Reserved compatibility argument.
            check_constant_cols: Whether to check constant columns.
            drop_1_for_FE: Whether to drop one fixed-effect level.
            specification_name: Optional model label override.
        """

        if self.from_formula:
            if a0 is None:
                a0 = self.prior.a
            if b0 is None:
                b0 = self.prior.b
            if mu0 is None:
                mu0 = self.prior.mu
            if Lambda0 is None:
                Lambda0 = self.prior.Lambda
            if specification_name is None: specification_name = self.specification_name
            bmodel = self.build_model_from_formula(self.formula, data, mu0, Lambda0, a0, b0, index=index, debug=debug,
                                                   check_constant_cols=check_constant_cols, drop_1_for_FE=drop_1_for_FE,
                                                   specification_name=specification_name)
            return bmodel
        else:
            raise Exception("Can only rebuild a model from a formula!")

    @staticmethod
    def build_model_from_formula(formula, data, mu0=None, Lambda0=None, a0=0, b0=0, index=None, debug=False,
                                 specification_name=None, drop_1_for_FE=True, check_constant_cols=True):
        """Construct a conjugate-prior Bayesian linear model directly from formula/data.

        Args:
            formula: Regression formula.
            data: Input data.
            mu0: Prior mean hyperparameter.
            Lambda0: Prior precision hyperparameter.
            a0: Inverse-gamma shape hyperparameter.
            b0: Inverse-gamma scale hyperparameter.
            index: Optional row subset.
            debug: Debug mode flag.
            specification_name: Optional model label.
            drop_1_for_FE: Whether to drop one FE level.
            check_constant_cols: Whether to check constant columns.
        """

        __t0 = time.time()

        if isinstance(data, dict):
            data = DataFrame(data, copy=False)
        data_obj = SparseDataGetter.get_data(data, formula, index=index, debug=debug, drop_1_for_FE=drop_1_for_FE,
                                             check_constant_cols=check_constant_cols)

        if data_obj[INSTRUMENTS_KEY] is not None:
            raise Exception("No instruments in `blm`!")
        if data_obj[ABSORB_KEY] is not None:
            raise Exception("No absorb in `blm`!")

        X, exog_names = data_obj[EXOG_KEY].values, data_obj[EXOG_KEY].column_names
        y, endog_name = data_obj[ENDOG_KEY].values, data_obj[ENDOG_KEY].column_names[0]

        param_names = list(exog_names) + ['__sigma2']

        has_implicit_constant = data_obj[HAS_IMPLICIT_CONSTANT_KEY]
        has_intercept = data_obj[HAS_INTERCEPT_KEY]

        if data_obj[WEIGHTS_KEY] is not None:
            w, weights_name = data_obj[WEIGHTS_KEY].values, data_obj[WEIGHTS_KEY].column_names[0]
        else:
            w, weights_name = None, None

        formula_design_info = data_obj[FORMULA_DESIGN_INFO_KEY]

        model_elapsed = time.time() - __t0

        return BayesianLinearRegressionConjugatePriorModel(
            y, X, False, has_intercept, has_implicit_constant, formula_design_info, data, index,
            param_names=param_names,
            exog_names=exog_names, endog_name=endog_name, weights=w, weights_name=weights_name,
            a0=a0, b0=b0, mu0=mu0, Lambda0=Lambda0, model_elapsed=model_elapsed, specification_name=specification_name
        )

    @staticmethod
    def blm(formula, data, index=None, mu0=None, Lambda0=None, a0=0, b0=0, debug=False, specification_name=None):
        """Build-and-fit convenience wrapper for formula interface.

        Args:
            formula: Regression formula.
            data: Input data.
            index: Optional row subset.
            mu0: Prior mean hyperparameter.
            Lambda0: Prior precision hyperparameter.
            a0: Inverse-gamma shape hyperparameter.
            b0: Inverse-gamma scale hyperparameter.
            debug: Debug mode flag.
            specification_name: Optional model label.

        Examples
        --------
        Closed-form Bayesian linear regression with a default flat prior:

        >>> import numpy as np, pandas as pd
        >>> from kanly.api import blm
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({'x':   rng.normal(size=200),
        ...                    'grp': rng.integers(0, 5, 200)})
        >>> df['y'] = 1.2 - 0.3 * df['x'] + 0.2 * rng.normal(size=200)
        >>> fit = blm('y ~ x + C(grp)', df)                # doctest: +SKIP

        Mildly informative slope prior centred at zero:

        >>> fit_p = blm('y ~ x + C(grp)', df,              # doctest: +SKIP
        ...             mu0={'x': 0.0},
        ...             Lambda0={'x': 1.0})

        See Also
        --------
        :meth:`BLM` : matrix-form entry point.
        """
        bmodel = BayesianLinearRegressionConjugatePriorModel.build_model_from_formula(
            formula, data, index=index, debug=debug, mu0=mu0, Lambda0=Lambda0, a0=a0, b0=b0,
            specification_name=specification_name
        )
        return bmodel.fit()

    @staticmethod
    def BLM(y, X, add_constant=False, weights=None, mu0=None, Lambda0=None, a0=0, b0=0, param_names=None, endog_name=None,
            weights_name=None, specification_name=None):
        """Build-and-fit convenience wrapper for matrix interface.

        Args:
            y: Outcome vector.
            X: Design matrix.
            add_constant: Whether to add constant term.
            weights: Optional weights.
            mu0: Prior mean hyperparameter.
            Lambda0: Prior precision hyperparameter.
            a0: Inverse-gamma shape hyperparameter.
            b0: Inverse-gamma scale hyperparameter.
            param_names: Optional parameter names.
            endog_name: Optional outcome label.
            weights_name: Optional weights label.
            specification_name: Optional model label.

        Examples
        --------
        Closed-form Bayesian linear regression from numpy arrays:

        >>> import numpy as np
        >>> from kanly.api import BLM
        >>> rng = np.random.default_rng(0)
        >>> n = 200
        >>> X = np.column_stack([np.ones(n), rng.normal(size=n)])
        >>> y = 1.0 + 2.0 * X[:, 1] + 0.5 * rng.normal(size=n)
        >>> fit = BLM(y, X,                                # doctest: +SKIP
        ...           param_names=['Intercept', 'x', '__sigma2'])
        """

        bmodel = BayesianLinearRegressionConjugatePriorModel(
            y, X, add_constant, False, False, None, False, None, index=None,
            param_names=param_names, exog_names=None,
            endog_name=endog_name, weights=weights, weights_name=weights_name, a0=a0, b0=b0, mu0=mu0, Lambda0=Lambda0,
            specification_name=specification_name)

        return bmodel.fit()

    def _set_prior_from_dict(self, precision_dict, mean_dict=None, a0=None, b0=None):
        """Set conjugate prior from dictionaries keyed by parameter names.

        Args:
            precision_dict: Precision sparse_terms keyed by parameter.
            mean_dict: Mean sparse_terms keyed by parameter.
            a0: Optional inverse-gamma shape override.
            b0: Optional inverse-gamma scale override.
        """
        if a0 is None:
            a0 = self.a0
        if b0 is None:
            b0 = self.b0

        if mean_dict is None:
            mean_dict = dict()

        mu0 = np.array([mean_dict.get(k, 0.0) for k in self.param_names[:-1]])
        Lambda0 = np.diag([precision_dict.get(k, 0.0) for k in self.param_names[:-1]])

        self.prior = NormalInverseGamma(mu0, Lambda0, a0, b0)
        return {'a0': a0, 'b0': b0, 'mu0': mu0, 'Lambda0': Lambda0}

    # def compute_posterior_predictive(self, mu_beta, var_beta, sigma2, full=False):
    #     #TODO
    #     return BayesianLinearRegressionResults.__compute_posterior_predictive_internal(
    #         self.exog, mu_beta, var_beta, sigma2, full)

#
# if __name__ == '__main__':
#     import numpy as np
#     import pandas as pd
#     from kanly.api import lm, compare_fits
#
#     n = 300
#     np.random.seed(0)
#     df = pd.DataFrame({
#         'x': np.random.randn(n),
#         'grp': np.random.randint(0, 12, n),
#     })
#     df['y'] = 1.2 - 0.3 * df['x'] + .2 * np.random.randn(n)
#
#     fit_blm = BayesianLinearRegressionConjugatePriorModel.blm('y ~ x + C(grp)', df, debug=False,
#                                                               specification_name='blm')
#     fit_lm = lm('y ~ x + C(grp)', df, debug=False, specification_name='lm')
#     print(compare_fits([fit_blm, fit_lm]))
