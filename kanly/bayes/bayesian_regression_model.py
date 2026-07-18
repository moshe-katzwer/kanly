"""Bayesian wrappers over linear, generalized-linear, and nonlinear regression builders."""

from __future__ import absolute_import, print_function

from abc import abstractmethod

import numpy as np

from kanly.bayes.bayesian_model import BayesianModel
from kanly.regression.generalized_linear_models.constants import DEFAULT_GLM_FAMILY
from kanly.regression.generalized_linear_models.families import _get_family_and_link, Poisson, Binomial
from kanly.regression.generalized_linear_models.model import SparseGeneralizedLinearModel
from kanly.regression.linear_models.model import SparseLinearModel
from kanly.regression.linear_models.penalized.sparse_elastic_net_internal import _get_normalizing_factors
from kanly.regression.nonlinear_least_squares.model import \
    SparseNonlinearLeastSquaresModel, DEFAULT_NLLS_JAC_METHOD, DEFAULT_NLLS_DENSE_THRESHOLD_MB
from kanly.utils.util import dict_2_dataframe

SCALE_KEY = '__sigma2'


class BayesianRegressionModelFromFormula(BayesianModel):
    """
    Class for generating log posteriors for some common regression models, including

        - linear models (likelihood ``[y - (x' * params)] ~ Normal(0.0, __sigma2)``)
        - generalized linear models (likelihood `` y * (x' * params) - b(x' * params)`` 
            where ``b`` is the cumulant function)
        - nonlinear least squares (likelihood ``[y - f(x,params)] ~ Normal(0.0, __sigma2)``)

    and the prior is user specified.  Convenience methods for MCMC.
    """

    def __init__(self, base_model, log_likelihood_function, bounds, do_bounded_transform, priors, param_names,
                 specification_name=None, other_info=None, debug=False, nopython=False):
        """Initialize Bayesian regression wrapper over a fitted base-model object.

        Args:
            base_model: Underlying sparse regression model object.
            log_likelihood_function: Callable ``params -> loglik``.
            bounds: Optional parameter bounds dictionary.
            do_bounded_transform: Whether to sample/optimize in transformed space.
            priors: Optional prior specification dictionary.
            param_names: Ordered parameter names.
            specification_name: Optional model label.
            other_info: Optional metadata dictionary.
            debug: Debug flag.
            nopython: Whether to prefer nopython-compatible prior callables.
        """

        super().__init__(log_likelihood_function, bounds, do_bounded_transform, priors, param_names,
                         specification_name=specification_name, other_info=other_info, debug=debug, nopython=nopython)

        self.base_model = base_model
        self.nobs = self.base_model.nobs
        self.is_weighted = self.base_model.is_weighted

    @staticmethod
    def _parse_bounds(bounds, param_names):
        """Keep only relevant bounds and ensure the residual variance parameter is constrained positive.

        Args:
            bounds: User-supplied bounds dictionary or None.
            param_names: Ordered parameter names for this model.
        """
        if bounds is None:
            bounds = dict()
        if not isinstance(bounds, dict):
            raise NotImplementedError('`bounds` must be a dictionary keyed on parameter for now...')
        bounds = {k: p for k, p in bounds.items() if k in param_names}
        if SCALE_KEY not in bounds:
            bounds[SCALE_KEY] = (0.0, np.inf)
        return bounds

    @classmethod
    @abstractmethod
    def build_model_from_formula(*args, **kwargs) -> BayesianModel:
        """Abstract constructor from formula/data.

        Args:
            *args: Implementation-specific positional arguments.
            **kwargs: Implementation-specific keyword arguments.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_log_likelihood_function_from_model(*args, **kwargs):
        """Abstract likelihood-builder from base model internals.

        Args:
            *args: Implementation-specific positional arguments.
            **kwargs: Implementation-specific keyword arguments.
        """
        raise NotImplementedError

    def get_sum_weights(self):
        """Return total effective sample weight used in likelihood scaling.

        Args:
            None.
        """
        if self.base_model.is_weighted:
            return self.base_model.weights.sum()
        else:
            return self.nobs

    def get_elastic_net_log_prior(self, alpha, l1_ratio, penalize_intercept=False, normalize=False):
        """Build an elastic-net style log-prior over regression coefficients.

        Args:
            alpha: Scalar or dict of overall penalty strengths.
            l1_ratio: Mix between L1 and L2 penalties in ``[0, 1]``.
            penalize_intercept: Whether to include intercept in penalty.
            normalize: Whether to scale penalties by regressor norms.
        """

        if normalize:
            if isinstance(self.base_model, SparseNonlinearLeastSquaresModel):
                raise Exception("Can't normalize on NLLS model!")
            _, _, l2_norm_X = \
                _get_normalizing_factors(self.base_model.exog, self.is_weighted, self.base_model.weights)
        else:
            l2_norm_X = None

        return self._get_elastic_net_log_prior_internal(
            alpha, l1_ratio, self.param_names, self.get_sum_weights(), penalize_intercept=penalize_intercept,
            scales=l2_norm_X)

    @staticmethod
    def _get_elastic_net_log_prior_internal(alpha, l1_ratio, param_names, sum_wts,
                                            penalize_intercept=False, scales=None):
        """Internal factory for separable L1/L2 penalties as a log-prior function.

        Args:
            alpha: Scalar/vector penalty magnitudes.
            l1_ratio: Mix between L1 and L2 penalties.
            param_names: Ordered parameter names including trailing scale.
            sum_wts: Total effective weight (or sample size).
            penalize_intercept: Whether intercept is penalized.
            scales: Optional feature scaling factors.
        """

        assert 0 <= l1_ratio <= 1
        if isinstance(alpha, dict):
            alpha = np.array([alpha.get(k, 0) if (k != 'Intercept' or penalize_intercept) else 0.0
                              for k in param_names[:-1]])
        elif isinstance(alpha, (int, float)):
            alpha = np.array([alpha if (k != 'Intercept' or penalize_intercept) else 0.0
                              for k in param_names[:-1]])

        assert np.all(alpha >= 0)
        assert alpha.shape == (len(param_names) - 1,)

        if scales is not None:
            assert (
                    (isinstance(scales, (int, float)) and scales > 0) or
                    (np.shape(scales) == (len(param_names) - 1,) and np.all(np.array(scales) >= 0))
            )
            scales = np.array(scales)
        else:
            scales = 1.0

        _l1_penalties = 2 * alpha * sum_wts * l1_ratio * scales
        _l2_penalties = 2 * alpha * sum_wts * (1 - l1_ratio) / 2 * scales ** 2

        def _log_prior(params):
            """Evaluate the elastic-net log-prior for regression coefficients.

            Implements the elastic-net penalty
            ``-(L2_penalty * coef² + L1_penalty * |coef|) / (2 * sigma2)``
            which reduces to ridge (``l1_ratio=0``) or LASSO (``l1_ratio=1``).
            The last element of ``params`` is treated as ``sigma2`` (variance).

            Args:
                params: Full parameter vector; ``params[:-1]`` are regression
                    coefficients and ``params[-1]`` is ``sigma2``.

            Returns:
                Scalar log-prior value (non-positive).
            """
            return -(
                    (np.sum(_l2_penalties * params[:-1] ** 2) + np.sum(_l1_penalties * np.abs(params[:-1])))
                    / (2 * params[-1])
            )

        return _log_prior


class BayesianNonlinearLeastSquaresModel(BayesianRegressionModelFromFormula):
    """Bayesian nonlinear least-squares model with an explicit ``__sigma2`` scale parameter."""

    @classmethod
    def build_model_from_formula(
            cls, formula, data, do_bounded_transform=True, bounds=None, priors=None, nopython=False, index=None,
            debug=False, cov_groups=None, custom_functions=None, jac_method=DEFAULT_NLLS_JAC_METHOD,
            dense_threshold_mb=DEFAULT_NLLS_DENSE_THRESHOLD_MB, specification_name=None, is_variance_weights=True,
            t_df=np.inf
    ) -> BayesianModel:
        """
        :param priors: should be dict of log pdf callables, keyed on parameter name, *or* strings as defined in `text_prior_to_log_pdf`
        :param t_df: if np.inf (default) then normal LLF, else t-distributon
        :param is_variance_weights: in the likelihood whether to treat the weights as a variance scaling
               `is_variance_weights=True` or as a frequency weighting `is_variance_weights=False`
        :return:
        Args:
            cls: Class reference for alternate constructors.
            formula: Nonlinear least-squares formula string.
            data: Input dataset.
            do_bounded_transform: Whether to use bounded->unbounded transforms.
            bounds: Optional parameter bounds dict.
            priors: Optional prior dict.
            nopython: Whether to request nopython-compatible internals.
            index: Optional row index/subset.
            debug: Debug mode flag.
            cov_groups: Optional covariance-group specification.
            custom_functions: Optional custom callables used in formula.
            jac_method: Jacobian approximation method.
            dense_threshold_mb: Threshold for dense/sparse Jacobian handling.
            specification_name: Optional model label.
            is_variance_weights: Whether weights are interpreted as variance weights.
            t_df: Degrees of freedom for t likelihood (``np.inf`` => Gaussian).

        Examples
        --------
        Bayesian NLLS (exponential decay model) with MCMC inference:

        >>> import numpy as np, pandas as pd
        >>> from kanly.api import BayesianNonlinearLeastSquaresModel
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({'x': rng.normal(size=250)})
        >>> df['y'] = (1.0 + 3.0 * np.exp(-0.5 * df['x']) +
        ...            0.4 * rng.normal(size=250))
        >>> model = BayesianNonlinearLeastSquaresModel.build_model_from_formula(   # doctest: +SKIP
        ...     '[y] ~ {Intercept} + {beta} * exp({gamma} * [x])', df,
        ...     priors={'beta': 'norm(3.0, 1.0)'})
        >>> fit = model.sample([0.0, 1.0, -0.1, 0.5],          # doctest: +SKIP
        ...                    n_samples=5_000, n_burnin=1_000,
        ...                    n_chains=4)
        """

        nlls_model = SparseNonlinearLeastSquaresModel.build_model_from_formula(
            formula, data, do_njit=nopython, debug=debug, index=index, cov_groups=cov_groups,
            specification_name=None, custom_functions=custom_functions,
            dense_threshold_mb=dense_threshold_mb, jac_method=jac_method)

        param_names = list(nlls_model.param_names) + [SCALE_KEY]

        llf = cls.get_log_likelihood_function_from_model(nlls_model, t_df, is_variance_weights)

        bounds = cls._parse_bounds(bounds, param_names)

        return cls(
            nlls_model, llf, bounds, do_bounded_transform, priors, param_names,
            specification_name=specification_name, other_info={'t_df': t_df}, debug=debug,
            nopython=nopython
        )

    @staticmethod
    def get_log_likelihood_function_from_model(model: SparseNonlinearLeastSquaresModel, t_df=np.inf, is_variance_weights=True):
        """Return NLLS log-likelihood callable on parameter vector.

        Args:
            model: Sparse NLLS model object.
            t_df: Degrees of freedom for t likelihood.
            is_variance_weights: Whether model weights scale variance.
        """
        return model.get_log_likelihood_function(transform_scale=False, t_df=t_df,
                                                 is_variance_weights=is_variance_weights)

    @classmethod
    def model_type(cls):
        """Return model-type label.

        Args:
            cls: Class object.
        """
        return 'Nonlinear Least Squares'


class BayesianLinearModel(BayesianRegressionModelFromFormula):
    """Bayesian linear model using the sparse linear-model backend."""

    @classmethod
    def build_model_from_formula(cls, formula, data, do_bounded_transform=True,
                                 bounds=None, priors=None, index=None, debug=False,
                                 cov_groups=None, specification_name=None, nopython=False
                                 ) -> BayesianModel:
        """
        :param priors: should be dict of log pdf callables, keyed on parameter name, *or* strings as defined in `text_prior_to_log_pdf`
        :return:
        Args:
            cls: Class reference for alternate constructors.
            formula: Linear model formula.
            data: Input dataset.
            do_bounded_transform: Whether to use bounded transforms.
            bounds: Optional bounds dictionary.
            priors: Optional prior dictionary.
            index: Optional row subset.
            debug: Debug mode flag.
            cov_groups: Optional covariance-group settings.
            specification_name: Optional model label.
            nopython: Whether to request nopython-compatible prior handling.

        Examples
        --------
        Bayesian linear regression with MCMC and a slope prior:

        >>> import numpy as np, pandas as pd
        >>> from kanly.api import BayesianLinearModel
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({'x': rng.normal(size=200)})
        >>> df['y'] = 1.0 + 2.0*df['x'] + rng.normal(size=200)
        >>> model = BayesianLinearModel.build_model_from_formula(    # doctest: +SKIP
        ...     'y ~ x', df,
        ...     priors={'x': 'norm(0, 1)'})
        >>> fit = model.sample([0.0, 0.0, 1.0],              # [Intercept, x, sigma2]
        ...                    n_samples=5_000, n_burnin=1_000,
        ...                    n_chains=4)                           # doctest: +SKIP
        """

        lm_model = SparseLinearModel.build_model_from_formula(
            formula, data, debug=debug, index=index, cov_groups=cov_groups,
            specification_name=None)

        param_names = list(lm_model.exog_names) + [SCALE_KEY]

        llf = cls.get_log_likelihood_function_from_model(lm_model)

        bounds = cls._parse_bounds(bounds, param_names)

        return cls(
            lm_model, llf, bounds, do_bounded_transform, priors, param_names,
            specification_name=specification_name, debug=debug, nopython=nopython
        )

    @staticmethod
    def get_log_likelihood_function_from_model(model):
        """Wrap linear-model likelihood to consume ``sigma2`` rather than ``sigma``.

        Args:
            model: Sparse linear-model instance.
        """
        # Note that llf from linear model is function of sigma, not sigma^2
        llf = model.get_quadratic_form_and_llf()[0]

        def llf2(x):
            """Evaluate the linear-model log-likelihood parameterized in sigma² rather than sigma.

            Extracts the sigma value as ``x[-1] ** 0.5`` (square-root of the
            last element, which is treated as sigma²) and forwards it together
            with the regression coefficients ``x[:-1]`` to the original ``llf``.

            Args:
                x: Full parameter vector; ``x[:-1]`` are regression coefficients
                    and ``x[-1]`` is sigma² (variance).

            Returns:
                Scalar log-likelihood value.
            """
            return llf(x[:-1], x[-1] ** .5)

        return llf2

    # @staticmethod
    # def get_llf_obs_callable_from_model(model):
    #     return None

    @classmethod
    def model_type(cls):
        """Return model-type label.

        Args:
            cls: Class object.
        """
        return 'Linear Model'


class BayesianGeneralizedLinearModel(BayesianRegressionModelFromFormula):
    """Bayesian generalized linear model wrapper over sparse GLM likelihoods."""

    def __init__(self, glm_model, log_likelihood_function, bounds, do_bounded_transform, priors, param_names, family, link,
                 specification_name=None):
        """Initialize Bayesian GLM wrapper.

        Args:
            glm_model: Underlying sparse GLM model.
            log_likelihood_function: Callable ``params -> loglik``.
            bounds: Optional parameter bounds dictionary.
            do_bounded_transform: Whether to use transformed parameterization.
            priors: Optional prior dictionary.
            param_names: Ordered parameter names.
            family: GLM family object.
            link: GLM link object.
            specification_name: Optional model label.
        """
        super().__init__(glm_model, log_likelihood_function, bounds, do_bounded_transform, priors, param_names,
                         specification_name=specification_name)
        self.family = family
        self.link = link

    @classmethod
    def model_type(cls):
        """Return model-type label.

        Args:
            cls: Class object.
        """
        return 'Generalized Linear Model'

    @classmethod
    def build_model_from_formula(cls, formula, data, family=DEFAULT_GLM_FAMILY, link=None, debug=False, index=None,
                                 specification_name=None, drop_1_for_FE=True, check_constant_cols=True,
                                 do_bounded_transform=True, priors=None, bounds=None,
                                 ) -> BayesianModel:
        """Build Bayesian GLM from formula/data.

        Args:
            cls: Class reference for alternate constructors.
            formula: GLM formula.
            data: Input dataset.
            family: GLM family spec/object.
            link: Optional link spec/object.
            debug: Debug mode flag.
            index: Optional row subset.
            specification_name: Optional model label.
            drop_1_for_FE: Whether to drop one fixed-effect level.
            check_constant_cols: Whether to check constant columns.
            do_bounded_transform: Whether to apply bounded transforms.
            priors: Optional prior dictionary.
            bounds: Optional bounds dictionary.

        Examples
        --------
        Bayesian logistic regression with MCMC and normal priors:

        >>> import numpy as np, pandas as pd
        >>> from kanly.api import BayesianGeneralizedLinearModel
        >>> rng = np.random.default_rng(0)
        >>> n = 500
        >>> df = pd.DataFrame({'x': rng.normal(size=n)})
        >>> df['y'] = (rng.uniform(size=n) <
        ...            1/(1+np.exp(-(0.4 + 0.9*df['x'])))).astype(float)
        >>> model = BayesianGeneralizedLinearModel.build_model_from_formula(  # doctest: +SKIP
        ...     'y ~ x', df, family='binomial',
        ...     priors={'x': 'norm(0, 1)'})
        >>> fit = model.sample([0.0, 0.0],                    # doctest: +SKIP
        ...                    n_samples=5_000, n_burnin=1_000,
        ...                    n_chains=4)
        """
        data = dict_2_dataframe(data)

        glm_model = SparseGeneralizedLinearModel.build_model_from_formula(
            formula, data, debug=debug, index=index, specification_name=specification_name,
            drop_1_for_FE=drop_1_for_FE, cov_groups=None, check_constant_cols=check_constant_cols)

        family, link = _get_family_and_link(family, link)

        param_names = list(glm_model.exog_names) + [SCALE_KEY]
        bounds = cls._parse_bounds(bounds, param_names)

        llf = cls.get_log_likelihood_function_from_model(glm_model, family, link)

        return cls(
            glm_model, llf, bounds, do_bounded_transform, priors, param_names, family, link, debug=debug)

    @staticmethod
    def get_log_likelihood_function_from_model(model, family, link):
        """Build parameter-vector log-likelihood callable for GLM family/link.

        Args:
            model: Sparse GLM model instance.
            family: GLM family object.
            link: GLM link object.
        """
        _llf = model.get_log_likelihood_function(family=family, link=link)
        wts = model.weights if model.is_weighted else 1.

        def llf(params):
            """Evaluate the GLM log-likelihood parameterized over the full parameter vector.

            For Poisson and Binomial families, the scale is fixed at 1.0.
            For other families, the scale is extracted as ``params[-1] ** 0.5``
            (treating the last element as sigma²).

            Args:
                params: Full parameter vector; ``params[:-1]`` are linear
                    predictor coefficients and ``params[-1]`` is sigma²
                    (ignored for Poisson/Binomial).

            Returns:
                Scalar log-likelihood value.
            """
            if family.name in [Poisson.name, Binomial.name]:
                scale = 1.0
            else:
                scale = params[-1] ** .5
            return _llf(params[:-1], scale=scale, var_weights=wts)

        return llf


    # def amha(self, x0, step_cov=None, debug=False, specification_name=None, n_chains=DEFAULT_AM_N_CHAINS,
    #          n_burnin=DEFAULT_AM_N_BURNIN, n_samples=DEFAULT_AM_N_SAMPLES, max_processes=DEFAULT_AM_MAX_PROCESSES,
    #          seed=None, target_acceptance_rate=DEFAULT_AM_TARGET_ACCEPTANCE_RATE, draw_size=DEFAULT_AM_DRAW_SIZE,
    #          scaler0=None, min_scaler=DEFAULT_AM_MIN_SCALER, max_scaler=DEFAULT_AM_MAX_SCALER,
    #          scaler_adjust_rate=DEFAULT_AM_SCALER_ADJUST_RATE, thinning=DEFAULT_AM_THINNING,
    #          scaler_adjust_denom_power=DEFAULT_AM_SCALER_ADJUST_DENOM_POWER, resample_k=DEFAULT_AM_RESAMPLE_K,
    #          pbar_update_cadence=DEFAULT_AM_PBAR_UPDATE_CADENCE, do_adaptive=DEFAULT_AM_DO_ADAPTIVE,
    #          max_subchain_draws_burnin=DEFAULT_AM_MAX_SUBCHAIN_DRAWS_BURNIN,
    #          max_subchain_draws_sample=DEFAULT_AM_MAX_SUBCHAIN_DRAWS_SAMPlE,
    #          do_parallel=DEFAULT_AM_DO_PARALLEL,
    #          user_prompt_for_more_iters=DEFAULT_AM_USER_PROMPT_FOR_MORE_ITERS, proposal_df=DEFAULT_AM_PROPOSAL_DF,
    #          x0_is_original_scale=True, step_cov_adjust_rate=DEFAULT_AM_STEP_COV_ADJUST_RATE,
    #          fix_params=None, show_r_hat_ever_subchain=False,
    #          normalize_step_cov=DEFAULT_AM_NORMALIZE_STEP_COV) -> MCMCResults:
    #
    #     raise NotImplementedError("NEED TO UPDATE!")
    #
    #     if self.family.name in [Poisson.name, Binomial.name]:
    #         if fix_params is None:
    #             fix_params = dict()
    #         if SCALE_KEY not in fix_params:
    #             fix_params[SCALE_KEY] = 1.0
    #
    #     return super().amha(
    #         x0, step_cov=step_cov, debug=debug, specification_name=specification_name, n_chains=n_chains,
    #         n_burnin=n_burnin, n_samples=n_samples, max_processes=max_processes,
    #         seed=seed, target_acceptance_rate=target_acceptance_rate, draw_size=draw_size,
    #         scaler0=scaler0, min_scaler=min_scaler, max_scaler=max_scaler,
    #         scaler_adjust_rate=scaler_adjust_rate, thinning=thinning,
    #         scaler_adjust_denom_power=scaler_adjust_denom_power, resample_k=resample_k,
    #         pbar_update_cadence=pbar_update_cadence, do_adaptive=do_adaptive,
    #         max_subchain_draws_burnin=max_subchain_draws_burnin,
    #         max_subchain_draws_sample=max_subchain_draws_sample,
    #         do_parallel=do_parallel,
    #         user_prompt_for_more_iters=user_prompt_for_more_iters, proposal_df=proposal_df,
    #         x0_is_original_scale=x0_is_original_scale, step_cov_adjust_rate=step_cov_adjust_rate,
    #         fix_params=fix_params, show_r_hat_ever_subchain=show_r_hat_ever_subchain,
    #         normalize_step_cov=normalize_step_cov,
    #     )


# if __name__ == '__main__':
#     from kanly.api import elastic_net
#
#     np.random.seed(0)
#     n = 5000
#     x = 1.56 * np.random.randn(n)
#     z = np.random.rand(n)
#     y = 3 + 10 * x - 2 * z + np.random.randn(n) * 3
#     wts = .01 + np.random.rand(n)
#     g = np.random.randint(0, 25, n)
#     data = {'x': x, 'y': y, 'z': z, 'wts': wts, 'g': g}
#
#     model = BayesianNonlinearLeastSquaresModel.build_model_from_formula(
#         '[y] ~ {a}*[x]+{b} $ [wts]', data, do_njit=True, is_variance_weights=True)
#     fit = model.amh([0] * 2 + [1], debug=True, n_chains=6, n_samples=30000,
#                      max_subchain_draws=100000, max_processes=12, )
#     print(fit)
#
# #
# if __name__ == '__main__':
#     from kanly.api import elastic_net
#     np.random.seed(0)
#     n = 50
#     x = 1.56 * np.random.randn(n)
#     z = np.random.rand(n)
#     y = 3 + 10 * x - 2 * z + np.random.randn(n) * 3
#     wts = .01 + np.random.rand(n)
#     data = {'x': x, 'y': y, 'z': z, 'wts': wts}
#
#     bmodel1 = BayesianNonlinearLeastSquaresModel.build_model_from_formula('[y] ~ [x] + [z]', data)
#     bmodel2 = BayesianGeneralizedLinearModel.build_model_from_formula('y~poly(z,2)', data)
#
#
#     bmodel = BayesianLinearModel.build_model_from_formula('y ~ x + z', data)
#     bmodel.set_priors({'': bmodel.get_elastic_net_log_prior(alpha=4, l1_ratio=.85, normalize=True)})
#     fit_mcmc_data_model = bmodel.amh([0, 0, 0, 1], n_samples=100000, n_chains=6, max_subchain_draws=80000)
#     print(fit_mcmc_data_model)
#     print(fit_mcmc_data_model.map_params)
#     print(elastic_net('y ~ x+z', data, alpha=4, l1_ratio=.85, normalize=True))
#
#     #
#     # print(bmodel.maximize_likelihood([0, 0, 1])['params'])
#     # print(bmodel.transform(bmodel.maximize_likelihood([0, 0, 1], transformed=True)['params']))
#     #
#     # print(bmodel.maximize_posterior([0, 0, 1])['params'])
#     # print(bmodel.transform(bmodel.maximize_posterior([0, 0, 1], transformed=True)['params']))
#     #
#     #
#     # print(BayesianLinearModel.build_model_from_formula(
#     #     'y ~ x', data).maximize_posterior([0, 0, 1.]))
#     #
#     # print(BayesianNonlinearLeastSquaresModel.build_model_from_formula(
#     #     '[y] ~ {a}+{b}*[x]', data).maximize_posterior([0, 0, 1.]))
#     #
#     # print(BayesianLinearModel.build_model_from_formula(
#     #     'y ~ x', data).amh([0, 0, 1.]))
#     #
#     print(BayesianNonlinearLeastSquaresModel.build_model_from_formula(
#         '[y] ~ {a}+{b}*[x]', data).amh([0, 0, 1.]))
#
#     from kanly.api import lm
#
#     print(lm('y~x', data))
# #
# if __name__ == '__main__':
#     from kanly.api import elastic_net
#     import matplotlib.pyplot as plt
#
#     np.random.seed(0)
#     n = 10_000
#     x = 1.56 * np.random.randn(n)
#     z = np.random.rand(n)
#     y = 3 + 10 * x - 2 * z + np.random.randn(n) * 3
#     wts = .01 + np.random.rand(n)
#     data = {'x': x, 'y': y, 'z': z, 'wts': wts, 'g': np.random.randint(0, 12, n)}
#
#     model = BayesianNonlinearLeastSquaresModel.build_model_from_formula(
#         '[y] ~ {a}+{b}*[x]+[C(g,-1)]', data,
#         #ounds={'a': [5, 12.]},
#         #priors={'a': 'normal(10, .1)'},
#         do_njit=True,
#     )
#     fit = model.amha(
#         [10, 6] + [0]*11 + [1.], user_prompt_for_more_iters=False,
#         step_cov=np.zeros((14,14)).astype(float),
#         debug=True, show_r_hat_ever_subchain=False,
#         n_burnin=10_000,
#         max_subchain_draws_burnin=3_000,
#         max_subchain_draws_sample=20_000,
#         n_samples=2_000,
#         #fix_params={'C(g)[6]': 1},
#         stop_adaptation_after_burnin=True,
#         do_diff_evolution_mc=False,
#         n_chains=10,
#         max_processes=100,
#         pbar_update_cadence=1.0
#     )
#     print(fit)
#     # print(fit.model)
#     # print(fit.model.priors)
#     # #fit.diagnostic_plot('{a}+2*{b}')
#     # #plt.show()
#     # fit.amha(10_000, debug=True)
#     #
#     #
#     # print(fit.model.base_model.fit())
#     #
#     # # model.amha(
#     # #     [10, 6] + [0]*11 + [1.], user_prompt_for_more_iters=False,
#     # #     debug=True, show_r_hat_ever_subchain=False,
#     # #     n_burnin=10000,
#     # #     max_subchain_draws_burnin=10_000,
#     # #     max_subchain_draws_sample=200_000,
#     # #     n_samples=180_000,
#     # #     #fix_params={'C(g)[6]': 1},
#     # #     stop_adaptation_after_burnin=True,
#     # #     do_diff_evolution_mc=False,
#     # #     n_chains=3,
#     # #     max_processes=100,
#     # #     pbar_update_cadence=1.0
#     # # # )
#     # fit.kde('a')
#     # plt.show()
#     fit.amha(1_000, debug=True)
#     print(fit)
