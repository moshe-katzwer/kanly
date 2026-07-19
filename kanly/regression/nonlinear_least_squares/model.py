from __future__ import absolute_import, print_function

import textwrap
import time
import warnings

import numpy as np
from pandas import DataFrame, Series

import random

from kanly.bootstrap.bootstrap import do_bootstrap2, bootstrap_entire_procedure, \
    DEFAULT_BB_ALPHA, DEFAULT_BB_METHOD, DEFAULT_BB_MAX_PROCESSES, DEFAULT_BB_SEED
from kanly.formula.data_getter import expand_lag_terms_in_formula
from kanly.formula.formula_design_info import FormulaDesignInfo, FormulaDesignInfoBase
from kanly.formula.sparse_term_to_data_methods import get_nobs_from_index
from kanly.regression.linear_models.variance_covariance2 import SparseVarianceCovariance2
from kanly.regression.model_base import ModelBase
from kanly.regression.nonlinear_least_squares.constants import (
    DEFAULT_NLLS_EN_GTOL, DEFAULT_NLLS_EN_FTOL, DEFAULT_NLLS_EN_XTOL, DEFAULT_NLLS_EN_NUM_SHRINKAGE,
    DEFAULT_NLLS_EN_SHRINK_FACTOR, DEFAULT_NLLS_EN_MAX_ITER, DEFAULT_NLLS_EN_ACTIVE_SET, DEFAULT_NLLS_EN_ALPHA,
    DEFAULT_NLLS_EN_L1_RATIO, DEFAULT_NLLS_EN_NORMALIZE, DEFAULT_NLLS_KEEP_OPTIMIZATION_PATH,
    DEFAULT_NLLS_XTOL, DEFAULT_NLLS_GTOL, DEFAULT_NLLS_FTOL, DEFAULT_NLLS_DELTA, DEFAULT_NLLS_DELTA_FLOOR,
    DEFAULT_NLLS_MAX_ITER, DEFAULT_NLLS_X_SCALE, DEFAULT_NLLS_RHO_QUAD_MODEL_ACCEPT, DEFAULT_NLLS_RHO_QUAD_MODEL_REJECT,
    DEFAULT_NLLS_DELTA_INCREASE_FACTOR, DEFAULT_NLLS_DELTA_DECREASE_FACTOR, DEFAULT_NLLS_RHO_STEP_ACCEPT_FLOOR,
    DEFAULT_NLLS_NUM_REFLECTIONS, DEFAULT_NLLS_DO_BROYDEN_JAC_UPDATE, DEFAULT_NLLS_BROYDEN_JAC_UPDATE_CADENCE,
    DEFAULT_NLLS_DO_LINE_SEARCH, DEFAULT_NLLS_COV_TYPE, DEFAULT_NLLS_TEST_LEVEL, DEFAULT_NLLS_BOOTSTRAP_SEED,
    DEFAULT_NLLS_BOOTSTRAP_N_SAMPLES, DEFAULT_NLLS_BOOTSTRAP_USE_CORRECTION, NLLS_COV_TYPES, BOOTSTRAP,
    DEFAULT_NLLS_PROMPT_USER_FOR_MORE_ITERS, DEFAULT_NLLS_DENSE_THRESHOLD_MB,
    DEFAULT_NLLS_JAC_METHOD, DEFAULT_NLLS_TRY_NEWTON_STEP, DEFAULT_NLLS_REFLECTION_THETA,
    DEFAULT_NLLS_EN_JAC_METHOD, DEFAULT_NLLS_DO_ANALYTIC_JAC_JIT, DEFAULT_NLLS_SCALE_L2_PENALTIES,
    DEFAULT_NLLS_EN_SCALE_PENALTIES, DEFAULT_NLLS_EN_SELECTION,
    DEFAULT_NLLS_EN_ONE_DIM_SEARCH_INIT_VAL, DEFAULT_NLLS_EN_ONE_DIM_SEARCH_CADENCE,
    DEFAULT_NLLS_EN_ONE_DIM_SEARCH_MULTIPLIER)
from kanly.regression.cov_types import check_cov_kwds, format_cov_kwds
from kanly.regression.nonlinear_least_squares.formula.sparse_nonlinear_formula_parser import (
    build_prediction_function_from_formula, get_monomial_data, parse_str_to_var_names)
from kanly.regression.nonlinear_least_squares.function_callables.prediction_function import PredictionFunction
from kanly.regression.nonlinear_least_squares.function_callables.residual_function import ResidualFunction
from kanly.regression.nonlinear_least_squares.optimize.nlls_coordinate_descent_minimize_internal import \
    nlls_elastic_net_minimize_internal_coordinate_descent
from kanly.regression.nonlinear_least_squares.optimize.nlls_minimize_internal import nlls_minimize_internal
from kanly.regression.nonlinear_least_squares.regression_results import SparseNonlinearLeastSquaresRegressionResults
from kanly.sparse_data_frame import SparseDataFrame
from kanly.utils.linalg_utils import DEFAULT_DENSE_THRESHOLD_MB
from kanly.utils.dict_2_array import dict_2_array, dict_2_list
from scipy.sparse import isspmatrix
from scipy.special import loggamma

DEFAULT_LLF_T_DF = np.inf  # for use in llf, np.inf implies t-dist --> normal


class SparseNonlinearLeastSquaresModel(ModelBase):
    """Model object for sparse nonlinear least-squares regression.

    The model combines an endogenous vector, a generated (or user supplied)
    prediction function, optional weights/covariance groups, and metadata
    needed to fit NLLS by trust-region methods or elastic-net coordinate
    descent.  Formula-built models use the custom NLLS syntax with ``[data]``
    and ``{parameter}`` tokens.
    """

    def __init__(self, endog, prediction_func_callable, formula_design_info=None, do_njit=True,
                 endog_name=None, weights=None, weights_name=None, valid_row_indices=None, model_elapsed=0.0,
                 cov_groups=None, cov_groups_name=None, index=None, specification_name=None, param_names=None,
                 num_params=None):
        """Initialise a nonlinear least-squares model.

        Args:
            endog: Response vector.
            prediction_func_callable: ``PredictionFunction`` or compatible
                callable mapping parameters to fitted values.
            formula_design_info:
            data: Original source data, retained for rebuilding/prediction.
            do_njit: Whether generated prediction code is JIT compiled.
            endog_name: Optional response name for summaries.
            weights: Optional observation weights.
            weights_name: Optional weight variable name.
            valid_row_indices: Row indices retained after dropping invalid data.
            model_elapsed: Time spent constructing the model.
            cov_groups: Optional covariance/bootstrap group labels.
            cov_groups_name: Optional name of covariance grouping variable.
            index: Optional original row indexer.
            specification_name: Optional model label.
            param_names: Optional parameter names when the prediction callable
                does not expose them.
            num_params: Optional parameter count when it cannot be inferred.
        """

        self.nobs = getattr(prediction_func_callable, 'nobs', len(endog))

        super().__init__(self.nobs, endog, index=index, valid_obs_rows=None,
                         specification_name=specification_name, formula_design_info=formula_design_info,
                         model_elapsed=model_elapsed)

        if not isinstance(prediction_func_callable, PredictionFunction):
            warnings.warn(f'`prediction_func_callable` is not of type "PredictionFunction", '
                          f'is of type {type(prediction_func_callable)}')

        if endog.shape[0] != self.nobs:
            raise Exception(f'endog has nobs {endog.shape[0]} but model nobs is {self.nobs}!')

        self.endog_name = '<y>' if endog_name is None else endog_name

        self.from_formula = self.formula_design_info is not None
        if self.from_formula:
            self.formula = self.formula_design_info.formula
        else:
            self.formula = None

        self.prediction_function_callable = prediction_func_callable

        if isinstance(prediction_func_callable, PredictionFunction):
            self.param_names = prediction_func_callable.param_names.copy()
            self.num_params = prediction_func_callable.num_params
            self.func_str = self.prediction_function_callable.func_str
            self.pred_func_py_str = self.prediction_function_callable.prediction_func_python_code_str
            self.residual_function_callable = ResidualFunction(self.prediction_function_callable, self.endog)

        else:
            if num_params is None:
                num_params = getattr(prediction_func_callable, 'num_params', None)
            if num_params is None:
                if param_names is None:
                    raise Exception("Must specify `num_params` for number of parameters or give `param_names`!")
                else:
                    num_params = len(param_names)
            self.num_params = num_params
            if param_names is None:
                param_names = getattr(prediction_func_callable, 'param_names',
                                      ['<x%d>' % d for d in range(self.num_params)])
            self.param_names = param_names
            self.func_str = None
            self.pred_func_py_str = None

            self.residual_function_callable = ResidualFunction(self.prediction_function_callable, self.endog,
                                                               num_params=num_params)

        self.is_weighted = weights is not None
        if self.is_weighted:
            if weights_name is None:
                weights_name = '<weights>'
            weights = np.asarray(weights)
        else:
            weights_name = None
        self.weights = weights
        self.weights_name = weights_name

        self.valid_obs_rows = valid_row_indices

        self.do_njit = do_njit

        self.cov_groups = cov_groups
        self.cov_groups_name = cov_groups_name

    def __repr__(self):
        """Return the human-readable model description."""
        return str(self)

    def __str__(self):
        """Return a formatted description of the NLLS model and parameters."""
        formula_wrap = '\n'.join(textwrap.wrap(str(self.formula), width=55, subsequent_indent=' ' * 14))
        param_wrap = '\n'.join(textwrap.wrap(', '.join(self.param_names),
                                             width=55, subsequent_indent=' ' * 14))
        return (
                '=' * 70 + '\n' +
                f'Nonlinear Least Squares Model\n' +
                '-' * 70 + '\n' +
                f'Dep Var:      {self.endog_name}\n' +
                f'Nobs:         {self.nobs}\n' +
                f'Num Params:   {self.num_params}\n' +
                f'Weights:      {self.weights_name}\n' +
                '\n' +
                f"formula:      {formula_wrap}\n\n" +
                f"Params:       {param_wrap}\n" +
                '=' * 70 + '\n'
        )

    def reindex(self, idx, inplace=False):
        """Subset the model's response, prediction data, and weights.

        Args:
            idx: Row indexer or boolean mask.
            inplace: When ``True``, mutate this model; otherwise return a new
                reindexed model.

        Returns:
            ``None`` when ``inplace=True``; otherwise a
            ``SparseNonlinearLeastSquaresModel`` over the selected rows.
        """
        if inplace:
            self.endog = self.endog[idx].copy()
            self.prediction_function_callable.reindex(idx, inplace)
            self.nobs = self.prediction_function_callable.nobs
            if self.weights is not None:
                self.weights = self.weights[idx].copy()
            self.residual_function_callable = ResidualFunction(
                self.prediction_function_callable, self.endog)

        else:
            return SparseNonlinearLeastSquaresModel(
                self.endog[idx].copy(),
                self.prediction_function_callable.reindex(idx, inplace=False),
                weights=self.weights[idx].copy() if self.is_weighted else None,
                weights_name=self.weights_name,
                valid_row_indices=self.valid_obs_rows[idx].copy(),
            )

    @staticmethod
    def NLLS_EN(endog, prediction_func_callable, param_names=None, num_params=None,
                start_params=None, debug=False,
                weights=None, weights_name=None, endog_name=None,
                cov_groups=None, cov_groups_name=None,
                specification_name=None, do_njit=True,
                bounds=None, positive=False,
                alpha=DEFAULT_NLLS_EN_ALPHA, l1_ratio=DEFAULT_NLLS_EN_L1_RATIO,
                active_set=DEFAULT_NLLS_EN_ACTIVE_SET, max_iter=DEFAULT_NLLS_EN_MAX_ITER,
                ftol=DEFAULT_NLLS_EN_FTOL, gtol=DEFAULT_NLLS_EN_GTOL, xtol=DEFAULT_NLLS_EN_XTOL,
                num_shrinkage=DEFAULT_NLLS_EN_NUM_SHRINKAGE, shrink_factor=DEFAULT_NLLS_EN_SHRINK_FACTOR,
                normalize=DEFAULT_NLLS_EN_NORMALIZE,
                selection=DEFAULT_NLLS_EN_SELECTION,
                prompt_user_for_more_iters=DEFAULT_NLLS_PROMPT_USER_FOR_MORE_ITERS,
                regularize_to_values=None, jac_method=DEFAULT_NLLS_EN_JAC_METHOD,
                scale_penalties=DEFAULT_NLLS_EN_SCALE_PENALTIES,
                one_dim_search_cadence=DEFAULT_NLLS_EN_ONE_DIM_SEARCH_CADENCE,
                one_dim_search_multiplier=DEFAULT_NLLS_EN_ONE_DIM_SEARCH_MULTIPLIER,
                one_dim_search_init_value=DEFAULT_NLLS_EN_ONE_DIM_SEARCH_INIT_VAL,
                seed=0, cov_type=None, cov_kwds=None
                ):
        """
        If f(x;beta) is the prediction function, this minimizes over beta

            1/2 * sum_i (y_i - f(x_i;beta)) ** 2
                + n * sum_k alpha[k] * l1_ratio[k] * |beta[k]-regularize_to_values[k]|
                + n * sum_k alpha[k] * (1-l1_ratio[k])/2 * (beta[k]-regularize_to_values[k]) ** 2

        That is, we do elastic net, shrinking parameters `beta` to the values in `regularize_to_values` (default=0)

        Examples
        --------
        Matrix-form elastic-net NLLS with a user-supplied prediction callable:

        >>> import numpy as np
        >>> from kanly.api import NLLS_EN
        >>> rng = np.random.default_rng(0)
        >>> n = 500
        >>> x = rng.normal(size=n)
        >>> y = 1.0 + 3.0 * np.exp(-0.5 * x) + 0.4 * rng.normal(size=n)
        >>> def predict(beta):                             # parameter callable
        ...     return beta[0] + beta[1] * np.exp(beta[2] * x)
        >>> fit = NLLS_EN(                                  # doctest: +SKIP
        ...     y, predict, param_names=['Intercept', 'beta', 'gamma'],
        ...     start_params=np.array([0.0, 1.0, -0.1]),
        ...     alpha=0.01, l1_ratio=0.5)

        See Also
        --------
        :meth:`nlls_en` : formula entry point.
        """

        if cov_type is not None and cov_type.upper() != BOOTSTRAP:
            raise Exception("`cov_type` must be `None` or `'BOOTSTRAP'`!")
        cov_kwds = format_cov_kwds(cov_kwds)

        # TODO cov_groups
        model = SparseNonlinearLeastSquaresModel(
            endog, prediction_func_callable, formula=None, from_formula=False, data=None, do_njit=do_njit,
            endog_name=endog_name, weights=weights, weights_name=weights_name, valid_row_indices=None,
            model_elapsed=0.0,
            cov_groups=cov_groups, cov_groups_name=cov_groups_name, index=None, specification_name=specification_name,
            param_names=param_names, num_params=num_params
        )

        return model._do_fit_en_wrapper_internal(
            start_params=start_params, bounds=bounds, positive=positive, alpha=alpha, l1_ratio=l1_ratio,
            debug=debug, active_set=active_set, max_iter=max_iter, ftol=ftol, gtol=gtol, xtol=xtol,
            num_shrinkage=num_shrinkage, shrink_factor=shrink_factor, normalize=normalize,
            prompt_user_for_more_iters=prompt_user_for_more_iters,
            jac_method=jac_method, selection=selection, seed=seed,
            scale_penalties=scale_penalties, regularize_to_values=regularize_to_values,
            one_dim_search_cadence=one_dim_search_cadence,
            one_dim_search_multiplier=one_dim_search_multiplier,
            one_dim_search_init_value=one_dim_search_init_value, cov_type=cov_type, cov_kwds=cov_kwds)


    @staticmethod
    def nlls_en(formula, data, start_params=None, bounds=None, positive=False, debug=False,
                specification_name=None, index=None, do_njit=True,
                alpha=DEFAULT_NLLS_EN_ALPHA, l1_ratio=DEFAULT_NLLS_EN_L1_RATIO,
                active_set=DEFAULT_NLLS_EN_ACTIVE_SET, max_iter=DEFAULT_NLLS_EN_MAX_ITER,
                ftol=DEFAULT_NLLS_EN_FTOL, gtol=DEFAULT_NLLS_EN_GTOL, xtol=DEFAULT_NLLS_EN_XTOL,
                num_shrinkage=DEFAULT_NLLS_EN_NUM_SHRINKAGE, shrink_factor=DEFAULT_NLLS_EN_SHRINK_FACTOR,
                normalize=DEFAULT_NLLS_EN_NORMALIZE, custom_functions=dict(),
                selection=DEFAULT_NLLS_EN_SELECTION,
                prompt_user_for_more_iters=DEFAULT_NLLS_PROMPT_USER_FOR_MORE_ITERS,
                regularize_to_values=None, jac_method=DEFAULT_NLLS_EN_JAC_METHOD,
                scale_penalties=DEFAULT_NLLS_EN_SCALE_PENALTIES,
                one_dim_search_cadence=DEFAULT_NLLS_EN_ONE_DIM_SEARCH_CADENCE,
                one_dim_search_multiplier=DEFAULT_NLLS_EN_ONE_DIM_SEARCH_MULTIPLIER,
                one_dim_search_init_value=DEFAULT_NLLS_EN_ONE_DIM_SEARCH_INIT_VAL,
                seed=0, cov_type=None, cov_kwds=None
                ):
        """
        If f(x;beta) is the prediction function, this minimizes over beta
        
            1/2 * sum_i (y_i - f(x_i;beta)) ** 2
                + n * sum_k alpha[k] * l1_ratio[k] * |beta[k]-regularize_to_values[k]|
                + n * sum_k alpha[k] * (1-l1_ratio[k])/2 * (beta[k]-regularize_to_values[k]) ** 2
            
        That is, we do elastic net, shrinking parameters `beta` to the values in `regularize_to_values` (default=0)

        Examples
        --------
        Elastic-net penalised exponential model. Parameters in ``{...}``
        braces are estimated; data columns appear in ``[...]`` brackets:

        >>> import numpy as np, pandas as pd
        >>> from kanly.api import nlls_en
        >>> rng = np.random.default_rng(0)
        >>> n = 500
        >>> df = pd.DataFrame({'x': rng.normal(size=n)})
        >>> df['y'] = 1.0 + 3.0 * np.exp(-0.5 * df['x']) + 0.4 * rng.normal(size=n)
        >>> fit = nlls_en(                                  # doctest: +SKIP
        ...     '[y] ~ {Intercept} + {beta} * exp({gamma} * [x])', df,
        ...     start_params={'Intercept': 0.0, 'beta': 1.0, 'gamma': -0.1},
        ...     alpha={'Intercept': 0.0, 'beta': 0.01, 'gamma': 0.01},
        ...     l1_ratio=0.5)

        Shrink ``beta`` toward a non-zero target via
        ``regularize_to_values={'beta': 3.0}``.

        See Also
        --------
        :meth:`nlls` : unpenalised NLLS with full inference.
        """

        if cov_type is not None and cov_type.upper() != BOOTSTRAP:
            raise Exception("`cov_type` must be `None` or `'BOOTSTRAP'`!")

        model = SparseNonlinearLeastSquaresModel.build_model_from_formula(
            formula, data, debug=debug, index=index, do_njit=do_njit,
            cov_groups=ModelBase.get_cov_group_keyword(cov_kwds),
            specification_name=specification_name,
            custom_functions=custom_functions)

        return model._do_fit_en_wrapper_internal(
            start_params=start_params, bounds=bounds, positive=positive, alpha=alpha, l1_ratio=l1_ratio,
            debug=debug, active_set=active_set, max_iter=max_iter, ftol=ftol, gtol=gtol, xtol=xtol,
            num_shrinkage=num_shrinkage, shrink_factor=shrink_factor, normalize=normalize,
            prompt_user_for_more_iters=prompt_user_for_more_iters,
            jac_method=jac_method, selection=selection, seed=seed,
            scale_penalties=scale_penalties, regularize_to_values=regularize_to_values,
            one_dim_search_cadence=one_dim_search_cadence,
            one_dim_search_multiplier=one_dim_search_multiplier,
            one_dim_search_init_value=one_dim_search_init_value, cov_type=cov_type, cov_kwds=cov_kwds)

    def _do_fit_en_wrapper_internal(
            self, start_params, bounds, positive, alpha, l1_ratio,
            debug, active_set, max_iter, ftol, gtol, xtol, num_shrinkage, shrink_factor, normalize,
            prompt_user_for_more_iters, jac_method, selection, seed, scale_penalties, regularize_to_values,
            one_dim_search_cadence, one_dim_search_multiplier, one_dim_search_init_value, cov_type, cov_kwds):
        """Fit elastic-net NLLS and optionally wrap it in a bootstrap covariance run.

        Args:
            start_params: Optional starting parameter vector/dict.
            bounds: Optional bounds vector/dict.
            positive: Positivity constraint input.
            alpha: Elastic-net penalty strength.
            l1_ratio: L1 share of the elastic-net penalty.
            debug: Whether to print optimisation diagnostics.
            active_set: Whether to use active-set coordinate cycling.
            max_iter: Maximum coordinate-descent iterations.
            ftol: Objective-change tolerance.
            gtol: Subgradient tolerance.
            xtol: Coordinate-change tolerance.
            num_shrinkage: Maximum backtracking attempts per coordinate.
            shrink_factor: Backtracking shrinkage factor.
            normalize: Whether to normalise coordinate penalties by Jacobian scale.
            prompt_user_for_more_iters: Whether/how to prompt after max iterations.
            jac_method: Derivative method for coordinate descent.
            selection: Coordinate selection strategy.
            seed: Random seed for random selection.
            scale_penalties: Whether penalties scale with the mean objective_function.
            regularize_to_values: Penalty target values.
            one_dim_search_cadence: Optional full-direction search cadence.
            one_dim_search_multiplier: Full-direction search multiplier.
            one_dim_search_init_value: Initial full-direction search scalar.
            cov_type: Optional covariance type; only bootstrap is supported here.
            cov_kwds: Bootstrap keyword arguments.

        Returns:
            Fit result or bootstrap wrapper dict containing fit and bootstrap samples.
        """

        fit = self.fit_en(start_params=start_params, bounds=bounds, positive=positive, alpha=alpha, l1_ratio=l1_ratio,
                          debug=debug, active_set=active_set, max_iter=max_iter, ftol=ftol, gtol=gtol, xtol=xtol,
                          num_shrinkage=num_shrinkage, shrink_factor=shrink_factor, normalize=normalize,
                          prompt_user_for_more_iters=prompt_user_for_more_iters,
                          jac_method=jac_method, selection=selection, seed=seed,
                          scale_penalties=scale_penalties, regularize_to_values=regularize_to_values,
                          one_dim_search_cadence=one_dim_search_cadence,
                          one_dim_search_multiplier=one_dim_search_multiplier,
                          one_dim_search_init_value=one_dim_search_init_value)

        if cov_type is not None and  BOOTSTRAP in cov_type.upper():

            return self._bootstrap_en_subfunction(
                fit, cov_kwds=cov_kwds, bounds=bounds, positive=positive,
                num_shrinkage=num_shrinkage, shrink_factor=shrink_factor, normalize=normalize,
                selection=selection, alpha=alpha, l1_ratio=l1_ratio, active_set=active_set, max_iter=max_iter,
                ftol=ftol, gtol=gtol, xtol=xtol, regularize_to_values=regularize_to_values,
                jac_method=jac_method
            )

        else:
            return fit

    def _bootstrap_en_subfunction(
            self, fit, cov_kwds=None, bounds=None, positive=False, debug=False,
            num_shrinkage=DEFAULT_NLLS_EN_NUM_SHRINKAGE,
            shrink_factor=DEFAULT_NLLS_EN_SHRINK_FACTOR, normalize=DEFAULT_NLLS_EN_NORMALIZE,
            selection=DEFAULT_NLLS_EN_SELECTION,
            alpha=DEFAULT_NLLS_EN_ALPHA, l1_ratio=DEFAULT_NLLS_EN_L1_RATIO, active_set=DEFAULT_NLLS_EN_ACTIVE_SET,
            max_iter=DEFAULT_NLLS_EN_MAX_ITER, ftol=DEFAULT_NLLS_EN_FTOL, gtol=DEFAULT_NLLS_EN_GTOL,
            xtol=DEFAULT_NLLS_EN_XTOL, regularize_to_values=None,
            jac_method=DEFAULT_NLLS_EN_JAC_METHOD,
    ):
        """Run bootstrap resampling for an elastic-net NLLS fit.

        Args:
            fit: Existing elastic-net NLLS fit used for starting values.
            cov_kwds: Bootstrap options such as ``n_samples``, ``seed``, and
                ``alpha``.
            bounds: Optional parameter bounds.
            positive: Positivity constraint input.
            debug: Whether to print bootstrap diagnostics.
            num_shrinkage: Maximum backtracking attempts per coordinate.
            shrink_factor: Backtracking shrinkage factor.
            normalize: Whether to normalise coordinate penalties.
            selection: Coordinate selection strategy.
            alpha: Elastic-net penalty strength.
            l1_ratio: Elastic-net L1 ratio.
            active_set: Whether to use active-set cycling.
            max_iter: Maximum coordinate-descent iterations.
            ftol: Objective-change tolerance.
            gtol: Subgradient tolerance.
            xtol: Coordinate-change tolerance.
            regularize_to_values: Penalty target values.
            jac_method: Derivative method.

        Returns:
            Dict with the original ``fit`` and ``bootstrap_result`` list.
        """

        cov_kwds = format_cov_kwds(cov_kwds)

        temp_func = self.get_bootstrap_en_function(
            start_params=fit.params.copy(), bounds=bounds, positive=positive, debug=False,
            alpha=alpha, l1_ratio=l1_ratio,
            active_set=active_set, max_iter=max_iter,
            ftol=ftol, gtol=gtol, xtol=xtol,
            num_shrinkage=num_shrinkage, shrink_factor=shrink_factor,
            normalize=normalize, prompt_user_for_more_iters=False,
            selection=selection, regularize_to_values=regularize_to_values,
            jac_method=jac_method,
            return_series=False, check_convergence=False,
            weights=self.weights
        )

        bootstrap_result = bootstrap_entire_procedure(
            temp_func, self.nobs, blocks=self.cov_groups,
            n_samples=cov_kwds.get('n_samples', DEFAULT_NLLS_BOOTSTRAP_N_SAMPLES),
            seed=cov_kwds.get('seed', DEFAULT_BB_SEED),
            alpha=cov_kwds.get('alpha', DEFAULT_BB_ALPHA), return_type='list', debug=debug)

        return {'fit': fit, 'bootstrap_result': bootstrap_result}

    def _process_en_input_dicts(self, alpha, positive, start_params, regularize_to_values, bounds):
        """Convert elastic-net dict inputs keyed by parameter name to ordered arrays/lists.

        Args:
            alpha: Scalar/vector/dict penalty strength.
            positive: Boolean/vector/dict positivity constraints.
            start_params: Optional starting values.
            regularize_to_values: Optional penalty target values.
            bounds: Optional bounds by parameter.

        Returns:
            Tuple ``(alpha, positive, start_params, regularize_to_values, bounds)``
            ordered by ``self.param_names``.
        """

        alpha = dict_2_array(alpha, self.param_names, ignore_extra_keys=True)
        positive = dict_2_array(positive, self.param_names, default_value=False, dtype=bool)
        start_params = dict_2_array(start_params, self.param_names)
        regularize_to_values = dict_2_array(regularize_to_values, self.param_names)
        bounds = dict_2_list(bounds, self.param_names, default_value=(-np.inf, np.inf))

        # alpha = self._process_dict_input(alpha)
        # positive = self._process_dict_input(positive, default=False)
        # start_params = self._process_dict_input(start_params)
        # regularize_to_values = self._process_dict_input(regularize_to_values)
        # bounds = self._process_dict_input(bounds, default=(-np.inf, np.inf))

        return alpha, positive, start_params, regularize_to_values, bounds

    def fit_en(self, start_params=None, bounds=None, positive=False, debug=False,
               alpha=DEFAULT_NLLS_EN_ALPHA, l1_ratio=DEFAULT_NLLS_EN_L1_RATIO,
               specification_name=None, active_set=DEFAULT_NLLS_EN_ACTIVE_SET, max_iter=DEFAULT_NLLS_EN_MAX_ITER,
               ftol=DEFAULT_NLLS_EN_FTOL, gtol=DEFAULT_NLLS_EN_GTOL, xtol=DEFAULT_NLLS_EN_XTOL,
               num_shrinkage=DEFAULT_NLLS_EN_NUM_SHRINKAGE, shrink_factor=DEFAULT_NLLS_EN_SHRINK_FACTOR,
               normalize=DEFAULT_NLLS_EN_NORMALIZE, prompt_user_for_more_iters=DEFAULT_NLLS_PROMPT_USER_FOR_MORE_ITERS,
               selection=DEFAULT_NLLS_EN_SELECTION, regularize_to_values=None,
               jac_method=DEFAULT_NLLS_EN_JAC_METHOD, scale_penalties=DEFAULT_NLLS_EN_SCALE_PENALTIES,
               one_dim_search_cadence=DEFAULT_NLLS_EN_ONE_DIM_SEARCH_CADENCE,
               one_dim_search_multiplier=DEFAULT_NLLS_EN_ONE_DIM_SEARCH_MULTIPLIER,
               one_dim_search_init_value=DEFAULT_NLLS_EN_ONE_DIM_SEARCH_INIT_VAL,
               seed=0
               ) -> SparseNonlinearLeastSquaresRegressionResults:
        """Fit an elastic-net penalised nonlinear least-squares model.

        Args:
            start_params: Optional starting values, vector or dict keyed by
                parameter name.
            bounds: Optional parameter bounds, vector/list or dict.
            positive: Boolean or per-parameter positivity constraint.
            debug: Whether to print solver diagnostics.
            alpha: Elastic-net penalty strength.
            l1_ratio: L1 share of the penalty.
            specification_name: Optional label for the result.
            active_set: Whether to use active-set coordinate cycling.
            max_iter: Maximum coordinate-descent iterations.
            ftol: Objective-change tolerance.
            gtol: Subgradient tolerance.
            xtol: Coordinate-change tolerance.
            num_shrinkage: Maximum shrinkage attempts per coordinate.
            shrink_factor: Backtracking shrinkage factor.
            normalize: Whether to scale penalties by coordinate Jacobian norms.
            prompt_user_for_more_iters: Whether/how to prompt after max iterations.
            selection: Coordinate selection strategy.
            regularize_to_values: Target values for centred regularisation.
            jac_method: Derivative method for coordinate descent.
            scale_penalties: Whether penalties scale with sample size/weights.
            one_dim_search_cadence: Optional full-direction search cadence.
            one_dim_search_multiplier: Full-direction search multiplier.
            one_dim_search_init_value: Initial full-direction search scalar.
            seed: Random seed for random coordinate selection.

        Returns:
            ``SparseNonlinearLeastSquaresRegressionResults``.
        """

        alpha, positive, start_params, regularize_to_values, bounds \
            = self._process_en_input_dicts(alpha, positive, start_params, regularize_to_values, bounds)

        result = nlls_elastic_net_minimize_internal_coordinate_descent(
            self.residual_function_callable, weights=self.weights, x0=start_params, num_params=self.num_params,
            alpha=alpha,
            l1_ratio=l1_ratio, num_shrinkage=num_shrinkage, shrink_factor=shrink_factor,
            positive=positive, bounds=bounds, debug=debug, active_set=active_set,
            max_iter=max_iter, xtol=xtol, ftol=ftol, gtol=gtol, normalize=normalize,
            prompt_user_for_more_iters=prompt_user_for_more_iters,
            selection=selection, regularize_to_values=regularize_to_values,
            jac_method=jac_method, scale_penalties=scale_penalties,
            one_dim_search_cadence=one_dim_search_cadence,
            one_dim_search_multiplier=one_dim_search_multiplier,
            one_dim_search_init_value=one_dim_search_init_value,
            seed=seed,
        )
        solver_options = {'gtol': gtol, 'xtol': xtol, 'ftol': ftol, 'max_iter': max_iter, 'active_set': active_set,
                          'jac_method': jac_method, 'scale_penalties': scale_penalties,
                          'one_dim_search_cadence': one_dim_search_cadence,
                          'one_dim_search_multiplier': one_dim_search_multiplier,
                          'one_dim_search_init_value': one_dim_search_init_value,
                          'num_shrinkage': num_shrinkage,
                          'shrink_factor': shrink_factor,
                          'seed': seed,
                          'selection': selection
                          }

        df_resid = self.nobs - self.num_params
        df_self = self.num_params

        fittedvalues = self.endog - result['resid']

        resid = result['resid']
        wresid = result['wresid']

        wssr, wsst, rsquared, rsquared_adj, scale, scale_mle, llf = self.get_rsquared(resid, df_resid)

        return SparseNonlinearLeastSquaresRegressionResults(
            self, result['params'], None, 'NOT COMPUTED', dict(), resid, wresid, fittedvalues,
            self.nobs, df_resid,
            df_self, result['optimality'], rsquared, rsquared_adj, wssr, wsst, 'CD', result['converged'],
            result['iterations'], result['optimality'], result['cost'], scale, scale_mle, llf, result['penalty'],
            result['objective_function'], None, result['grad'], result['subgrad'], None, result['status'],
            result['message'], result['active_mask'],
            result['is_bounded'], solver_options, test_level=.05, use_t=True, fit_elapsed=result['fit_elapsed'],
            cov_elapsed=0.0, specification_name=specification_name, keep_model=True, root_loss_function=None,
            optimization_result=result, is_penalized=True,
            l1_penalty=result['l1_penalty'], l2_penalty=result['l2_penalty'], positive=positive, alpha=alpha,
            l1_ratio=l1_ratio)

    @staticmethod
    def NLLS(endog, prediction_func_callable, param_names=None, num_params=None,
             start_params=None,
             weights=None, weights_name=None, endog_name=None,
             cov_groups=None, cov_groups_name=None,
             bounds=None, Delta=DEFAULT_NLLS_DELTA, max_iter=DEFAULT_NLLS_MAX_ITER,
             xtol=DEFAULT_NLLS_XTOL, ftol=DEFAULT_NLLS_FTOL, gtol=DEFAULT_NLLS_GTOL, compute_cov=True,
             cov_type=DEFAULT_NLLS_COV_TYPE, cov_kwds=None, test_level=DEFAULT_NLLS_TEST_LEVEL,
             debug=False, subsample=None, subsample_seed=0, specification_name=None,
             do_njit=True,
             x_scale=DEFAULT_NLLS_X_SCALE, Delta_floor=DEFAULT_NLLS_DELTA_FLOOR,
             rho_quad_model_accept=DEFAULT_NLLS_RHO_QUAD_MODEL_ACCEPT,
             rho_quad_model_reject=DEFAULT_NLLS_RHO_QUAD_MODEL_REJECT,
             rho_step_accept_floor=DEFAULT_NLLS_RHO_STEP_ACCEPT_FLOOR,
             Delta_increase_factor=DEFAULT_NLLS_DELTA_INCREASE_FACTOR,
             Delta_decrease_factor=DEFAULT_NLLS_DELTA_DECREASE_FACTOR,
             do_line_search=DEFAULT_NLLS_DO_LINE_SEARCH,
             num_reflections=DEFAULT_NLLS_NUM_REFLECTIONS,
             do_broyden_jac_update=DEFAULT_NLLS_DO_BROYDEN_JAC_UPDATE,
             broyden_jac_update_cadence=DEFAULT_NLLS_BROYDEN_JAC_UPDATE_CADENCE,
             root_loss_function=None, f_scale=1.0, keep_optimization_path=DEFAULT_NLLS_KEEP_OPTIMIZATION_PATH,
             prompt_user_for_more_iters=DEFAULT_NLLS_PROMPT_USER_FOR_MORE_ITERS,
             try_newton_step=DEFAULT_NLLS_TRY_NEWTON_STEP,
             dense_threshold_mb=DEFAULT_NLLS_DENSE_THRESHOLD_MB,
             jac_method=DEFAULT_NLLS_JAC_METHOD, do_analytic_jac_jit=DEFAULT_NLLS_DO_ANALYTIC_JAC_JIT,
             theta=DEFAULT_NLLS_REFLECTION_THETA, l2_penalties=None, regularize_to_values=None,
             scale_l2_penalties=DEFAULT_NLLS_SCALE_L2_PENALTIES,
             ):
        """Construct and fit an NLLS model from a user-supplied prediction function.

        Args:
            endog: Response vector.
            prediction_func_callable: Callable mapping parameters to fitted values.
            param_names: Optional ordered parameter names.
            num_params: Optional parameter count when names/callable do not provide it.
            start_params: Optional starting parameter vector.
            weights: Optional observation weights.
            weights_name: Optional weight label.
            endog_name: Optional response name.
            cov_groups: Optional covariance/bootstrap group labels.
            cov_groups_name: Optional group label.
            bounds: Optional parameter bounds.
            Delta: Initial trust-region radius.
            max_iter: Maximum solver iterations.
            xtol: Parameter-step tolerance.
            ftol: Objective-change tolerance.
            gtol: Gradient tolerance.
            compute_cov: Whether to compute covariance estimates.
            cov_type: Covariance estimator type.
            cov_kwds: Covariance estimator options.
            test_level: Significance level for inference.
            debug: Whether to print diagnostics.
            subsample: Optional random subsample size for starting-value fit.
            subsample_seed: Seed for subsample selection.
            specification_name: Optional result label.
            do_njit: Whether prediction code is JIT compiled when applicable.
            x_scale: Parameter scaling, ``None``, scalar/vector, or ``'jac'``.
            Delta_floor: Minimum trust-region radius.
            rho_quad_model_accept: Ratio threshold for accepting model quality.
            rho_quad_model_reject: Ratio threshold for shrinking the radius.
            rho_step_accept_floor: Minimum ratio for accepting a step.
            Delta_increase_factor: Radius expansion factor.
            Delta_decrease_factor: Radius shrink factor.
            do_line_search: Whether to extend accepted steps by line search.
            num_reflections: Number of reflected bound segments to consider.
            do_broyden_jac_update: Whether to update quadratic approximation.
            broyden_jac_update_cadence: Fresh-Jacobian cadence under Broyden.
            root_loss_function: Optional robust root-loss function/name.
            f_scale: Robust-loss scale or ``'adaptive'``.
            keep_optimization_path: Whether to store optimisation path.
            prompt_user_for_more_iters: Whether/how to prompt after max iterations.
            try_newton_step: Whether to compare Newton steps.
            dense_threshold_mb: Dense Jacobian/covariance threshold in MB.
            jac_method: ``'analytic'``, ``'mid'``, or ``'fwd'``.
            do_analytic_jac_jit: Whether to JIT analytic Jacobian code.
            theta: Bound-interior truncation/reflection factor.
            l2_penalties: Optional L2 penalty weights.
            regularize_to_values: L2 penalty target values.
            scale_l2_penalties: Whether L2 penalties scale with sample size/weights.

        Returns:
            ``SparseNonlinearLeastSquaresRegressionResults``.

        Examples
        --------
        Matrix-form NLLS with a user-supplied prediction callable. Useful
        when your model lives in code (e.g. a numerical integrator) rather
        than a closed-form formula:

        >>> import numpy as np
        >>> from kanly.api import NLLS
        >>> rng = np.random.default_rng(0)
        >>> n = 250
        >>> x = rng.normal(size=n)
        >>> y = 1.0 + 3.0 * np.exp(-0.5 * x) + 0.4 * rng.normal(size=n)
        >>> def predict(beta):
        ...     return beta[0] + beta[1] * np.exp(beta[2] * x)
        >>> fit = NLLS(                                     # doctest: +SKIP
        ...     y, predict, param_names=['Intercept', 'beta', 'gamma'],
        ...     start_params=np.array([0.0, 1.0, -0.1]),
        ...     cov_type='HC1')
        >>> # fit.params       -> {'Intercept': 0.79, 'beta': 3.13, 'gamma': -0.49}

        See Also
        --------
        :meth:`nlls` : formula entry point.
        """

        cov_kwds = format_cov_kwds(cov_kwds)
        if cov_groups is not None:
            cov_kwds['groups'] = cov_groups

        # TODO cov_groups
        model = SparseNonlinearLeastSquaresModel(
            endog, prediction_func_callable, formula=None, from_formula=False, data=None, do_njit=do_njit,
            endog_name=endog_name, weights=weights, weights_name=weights_name, valid_row_indices=None,
            model_elapsed=0.0,
            cov_groups=cov_groups, cov_groups_name=cov_groups_name, index=None, specification_name=specification_name,
            param_names=param_names, num_params=num_params
        )

        fit = model.fit(
            start_params, bounds=bounds, Delta=Delta, Delta_floor=Delta_floor, max_iter=max_iter, xtol=xtol,
            ftol=ftol, gtol=gtol, compute_cov=compute_cov, cov_type=cov_type, cov_kwds=cov_kwds, debug=debug,
            subsample=subsample, subsample_seed=subsample_seed, test_level=test_level, x_scale=x_scale,
            rho_quad_model_accept=rho_quad_model_accept, rho_quad_model_reject=rho_quad_model_reject,
            rho_step_accept_floor=rho_step_accept_floor, Delta_increase_factor=Delta_increase_factor,
            Delta_decrease_factor=Delta_decrease_factor, specification_name=specification_name,
            num_reflections=num_reflections, do_broyden_jac_update=do_broyden_jac_update,
            broyden_jac_update_cadence=broyden_jac_update_cadence, do_line_search=do_line_search,
            root_loss_function=root_loss_function, f_scale=f_scale, keep_optimization_path=keep_optimization_path,
            prompt_user_for_more_iters=prompt_user_for_more_iters, jac_method=jac_method,
            try_newton_step=try_newton_step, theta=theta, dense_threshold_mb=dense_threshold_mb,
            do_analytic_jac_jit=do_analytic_jac_jit, l2_penalties=l2_penalties,
            regularize_to_values=regularize_to_values, scale_l2_penalties=scale_l2_penalties,
        )

        return fit

    @staticmethod
    def nlls(formula, data, start_params=None, bounds=None, Delta=DEFAULT_NLLS_DELTA, max_iter=DEFAULT_NLLS_MAX_ITER,
             xtol=DEFAULT_NLLS_XTOL, ftol=DEFAULT_NLLS_FTOL, gtol=DEFAULT_NLLS_GTOL, compute_cov=True,
             cov_type=DEFAULT_NLLS_COV_TYPE, cov_kwds=None, test_level=DEFAULT_NLLS_TEST_LEVEL,
             debug=False, index=None, subsample=None, subsample_seed=0, specification_name=None,
             do_njit=True, custom_functions=dict(),
             x_scale=DEFAULT_NLLS_X_SCALE, Delta_floor=DEFAULT_NLLS_DELTA_FLOOR,
             rho_quad_model_accept=DEFAULT_NLLS_RHO_QUAD_MODEL_ACCEPT,
             rho_quad_model_reject=DEFAULT_NLLS_RHO_QUAD_MODEL_REJECT,
             rho_step_accept_floor=DEFAULT_NLLS_RHO_STEP_ACCEPT_FLOOR,
             Delta_increase_factor=DEFAULT_NLLS_DELTA_INCREASE_FACTOR,
             Delta_decrease_factor=DEFAULT_NLLS_DELTA_DECREASE_FACTOR,
             do_line_search=DEFAULT_NLLS_DO_LINE_SEARCH,
             num_reflections=DEFAULT_NLLS_NUM_REFLECTIONS,
             do_broyden_jac_update=DEFAULT_NLLS_DO_BROYDEN_JAC_UPDATE,
             broyden_jac_update_cadence=DEFAULT_NLLS_BROYDEN_JAC_UPDATE_CADENCE,
             root_loss_function=None, f_scale=1.0, keep_optimization_path=DEFAULT_NLLS_KEEP_OPTIMIZATION_PATH,
             prompt_user_for_more_iters=DEFAULT_NLLS_PROMPT_USER_FOR_MORE_ITERS,
             try_newton_step=DEFAULT_NLLS_TRY_NEWTON_STEP,
             dense_threshold_mb=DEFAULT_NLLS_DENSE_THRESHOLD_MB,
             jac_method=DEFAULT_NLLS_JAC_METHOD, do_analytic_jac_jit=DEFAULT_NLLS_DO_ANALYTIC_JAC_JIT,
             theta=DEFAULT_NLLS_REFLECTION_THETA, l2_penalties=None, regularize_to_values=None,
             scale_l2_penalties=DEFAULT_NLLS_SCALE_L2_PENALTIES):
        """Build and fit an NLLS model from a formula and data.

        Args:
            formula: NLLS formula with ``[data]`` sparse_terms, ``{parameter}`` sparse_terms,
                optional categorical/polynomial sparse_terms, and optional ``$ [w]`` weights.
            data: pandas ``DataFrame`` or dict-like data.
            start_params: Optional starting values, vector or dict keyed by parameter.
            bounds: Optional parameter bounds, vector/list or dict keyed by parameter.
            Delta: Initial trust-region radius.
            max_iter: Maximum solver iterations.
            xtol: Parameter-step tolerance.
            ftol: Objective-change tolerance.
            gtol: Gradient tolerance.
            compute_cov: Whether to compute covariance estimates.
            cov_type: Covariance estimator type.
            cov_kwds: Covariance estimator options.
            test_level: Significance level for inference.
            debug: Whether to print diagnostics.
            index: Optional row subset/indexer.
            subsample: Optional random subsample size for starting-value fit.
            subsample_seed: Seed for subsample selection.
            specification_name: Optional result label.
            do_njit: Whether to JIT compile generated prediction code.
            custom_functions: Optional functions exposed to generated formula code.
            x_scale: Parameter scaling, ``None``, scalar/vector, or ``'jac'``.
            Delta_floor: Minimum trust-region radius.
            rho_quad_model_accept: Ratio threshold for accepting model quality.
            rho_quad_model_reject: Ratio threshold for shrinking the radius.
            rho_step_accept_floor: Minimum ratio for accepting a step.
            Delta_increase_factor: Radius expansion factor.
            Delta_decrease_factor: Radius shrink factor.
            do_line_search: Whether to extend accepted steps by line search.
            num_reflections: Number of reflected bound segments to consider.
            do_broyden_jac_update: Whether to update quadratic approximation.
            broyden_jac_update_cadence: Fresh-Jacobian cadence under Broyden.
            root_loss_function: Optional robust root-loss function/name.
            f_scale: Robust-loss scale or ``'adaptive'``.
            keep_optimization_path: Whether to store optimisation path.
            prompt_user_for_more_iters: Whether/how to prompt after max iterations.
            try_newton_step: Whether to compare Newton steps.
            dense_threshold_mb: Dense Jacobian/covariance threshold in MB.
            jac_method: ``'analytic'``, ``'mid'``, or ``'fwd'``.
            do_analytic_jac_jit: Whether to JIT analytic Jacobian code.
            theta: Bound-interior truncation/reflection factor.
            l2_penalties: Optional L2 penalty weights.
            regularize_to_values: L2 penalty target values.
            scale_l2_penalties: Whether L2 penalties scale with sample size/weights.

        Returns:
            ``SparseNonlinearLeastSquaresRegressionResults``.

        Examples
        --------
        Weighted exponential fit with bootstrap covariance. Data columns
        appear in ``[...]`` brackets; parameters to estimate appear in
        ``{...}`` braces; ``$ [w]`` introduces an observation weight column:

        >>> import numpy as np, pandas as pd
        >>> from kanly.api import nlls
        >>> rng = np.random.default_rng(0)
        >>> n = 250
        >>> df = pd.DataFrame({
        ...     'x': rng.normal(size=n),
        ...     'w': np.exp(rng.normal(size=n)),
        ... })
        >>> df['y'] = 1.0 + 3.0 * np.exp(-0.5 * df['x']) + 0.4 * rng.normal(size=n)
        >>> fit = nlls(                                     # doctest: +SKIP
        ...     '[y] ~ {Intercept} + {beta} * exp({gamma} * [x]) $ [w]', df,
        ...     max_iter=100,
        ...     cov_type='bootstrap',
        ...     cov_kwds={'n_samples': 1_000, 'max_processes': 4},
        ...     specification_name='Exponential')
        >>> print(fit)                                       # doctest: +SKIP
        ══════════════════════════════════════════════════════════════════
        Nonlinear Least Squares Results
        Exponential
        ══════════════════════════════════════════════════════════════════
        ...
        Intercept   0.7854  ****   0.1923   4.08  <0.001   0.4066    1.164
        beta         3.131  ****   0.2004  15.63  <0.001    2.737    3.526
        gamma      -0.4918  ****  0.02419 -20.33  <0.001  -0.5394  -0.4441

        Logistic regression with an analytic Jacobian and starting values:

        >>> df2 = pd.DataFrame({'x': rng.normal(size=10_000)})
        >>> df2['p'] = 1 / (1 + np.exp(-(0.4 + 0.9 * df2['x'])))
        >>> df2['y'] = (rng.uniform(size=10_000) < df2['p']).astype(float)
        >>> fit2 = nlls(                                     # doctest: +SKIP
        ...     '[y] ~ 1.0 / (1.0 + exp({alpha} + {beta} * [x]))', df2,
        ...     start_params={'alpha': -0.5, 'beta': -1.0},
        ...     jac_method='analytic', do_analytic_jac_jit=True,
        ...     x_scale='jac')

        Robust root losses (``root_loss_function='soft_l1'``, ``'huber'``, etc.)
        and parameter bounds (``bounds={'beta': [-1, 1]}``) are also supported.

        See Also
        --------
        :meth:`NLLS` : matrix-form entry point taking a prediction callable.
        :meth:`nlls_en` : elastic-net penalised variant.
        """

        cov_kwds = format_cov_kwds(cov_kwds)

        model = SparseNonlinearLeastSquaresModel.build_model_from_formula(
            formula, data, debug=debug, index=index, do_njit=do_njit,
            cov_groups=ModelBase.get_cov_group_keyword(cov_kwds), specification_name=specification_name,
            custom_functions=custom_functions, dense_threshold_mb=dense_threshold_mb, jac_method=jac_method)

        return model.fit(
            start_params, bounds=bounds, Delta=Delta, Delta_floor=Delta_floor, max_iter=max_iter, xtol=xtol,
            ftol=ftol, gtol=gtol, compute_cov=compute_cov, cov_type=cov_type, cov_kwds=cov_kwds, debug=debug,
            subsample=subsample, subsample_seed=subsample_seed, test_level=test_level, x_scale=x_scale,
            rho_quad_model_accept=rho_quad_model_accept, rho_quad_model_reject=rho_quad_model_reject,
            rho_step_accept_floor=rho_step_accept_floor, Delta_increase_factor=Delta_increase_factor,
            Delta_decrease_factor=Delta_decrease_factor, specification_name=specification_name,
            num_reflections=num_reflections, do_broyden_jac_update=do_broyden_jac_update,
            broyden_jac_update_cadence=broyden_jac_update_cadence, do_line_search=do_line_search,
            root_loss_function=root_loss_function, f_scale=f_scale, keep_optimization_path=keep_optimization_path,
            prompt_user_for_more_iters=prompt_user_for_more_iters, jac_method=jac_method,
            try_newton_step=try_newton_step, theta=theta, dense_threshold_mb=dense_threshold_mb,
            do_analytic_jac_jit=do_analytic_jac_jit, l2_penalties=l2_penalties,
            regularize_to_values=regularize_to_values, scale_l2_penalties=scale_l2_penalties
        )

    def get_bounds(self, bounds):
        """Convert bounds keyed by parameter name to an ordered bounds array.

        Args:
            bounds: ``None``, array-like ``(num_params, 2)``, or dict mapping
                parameter names to ``(lower, upper)`` pairs.

        Returns:
            Bounds in array/list form, or ``None`` when no bounds are supplied.
        """
        if bounds is not None:
            if isinstance(bounds, dict):
                bounds_arr = np.ones((self.num_params, 2))
                for i, nm in enumerate(self.param_names):
                    bounds_arr[i] = bounds.get(nm, [-np.inf, np.inf])
                bounds = bounds_arr
            return bounds

    def fit(self, start_params=None, Delta=DEFAULT_NLLS_DELTA, Delta_floor=DEFAULT_NLLS_DELTA_FLOOR,
            max_iter=DEFAULT_NLLS_MAX_ITER, xtol=DEFAULT_NLLS_XTOL,
            ftol=DEFAULT_NLLS_FTOL, gtol=DEFAULT_NLLS_GTOL, compute_cov=True, cov_type=DEFAULT_NLLS_COV_TYPE,
            cov_kwds=None, use_t=True, test_level=DEFAULT_NLLS_TEST_LEVEL, debug=False, subsample=None,
            subsample_seed=0, x_scale=DEFAULT_NLLS_X_SCALE, bounds=None, specification_name=None,
            rho_quad_model_accept=DEFAULT_NLLS_RHO_QUAD_MODEL_ACCEPT,
            rho_quad_model_reject=DEFAULT_NLLS_RHO_QUAD_MODEL_REJECT,
            rho_step_accept_floor=DEFAULT_NLLS_RHO_STEP_ACCEPT_FLOOR,
            Delta_increase_factor=DEFAULT_NLLS_DELTA_INCREASE_FACTOR,
            Delta_decrease_factor=DEFAULT_NLLS_DELTA_DECREASE_FACTOR,
            do_line_search=DEFAULT_NLLS_DO_LINE_SEARCH,
            num_reflections=DEFAULT_NLLS_NUM_REFLECTIONS,
            do_broyden_jac_update=DEFAULT_NLLS_DO_BROYDEN_JAC_UPDATE,
            broyden_jac_update_cadence=DEFAULT_NLLS_BROYDEN_JAC_UPDATE_CADENCE,
            root_loss_function=None, f_scale=1.0, keep_optimization_path=DEFAULT_NLLS_KEEP_OPTIMIZATION_PATH,
            prompt_user_for_more_iters=DEFAULT_NLLS_PROMPT_USER_FOR_MORE_ITERS, jac_method=DEFAULT_NLLS_JAC_METHOD,
            try_newton_step=DEFAULT_NLLS_TRY_NEWTON_STEP, theta=DEFAULT_NLLS_REFLECTION_THETA,
            jacobian_func=None, dense_threshold_mb=DEFAULT_DENSE_THRESHOLD_MB,
            do_analytic_jac_jit=DEFAULT_NLLS_DO_ANALYTIC_JAC_JIT,
            l2_penalties=None, regularize_to_values=None, scale_l2_penalties=DEFAULT_NLLS_SCALE_L2_PENALTIES
            ) -> SparseNonlinearLeastSquaresRegressionResults:
        """Fit the model by trust-region nonlinear least squares.

        Args:
            start_params: Optional starting values, vector or dict keyed by parameter.
            Delta: Initial trust-region radius.
            Delta_floor: Minimum trust-region radius.
            max_iter: Maximum solver iterations.
            xtol: Parameter-step tolerance.
            ftol: Objective-change tolerance.
            gtol: Gradient tolerance.
            compute_cov: Whether to compute covariance estimates after fitting.
            cov_type: Covariance estimator type.
            cov_kwds: Covariance estimator options.
            use_t: Whether to use t-distribution inference when supported.
            test_level: Significance level for inference.
            debug: Whether to print fitting/covariance diagnostics.
            subsample: Optional random subsample size for an initial fit.
            subsample_seed: Seed for subsample selection.
            x_scale: Parameter scaling, ``None``, scalar/vector, or ``'jac'``.
            bounds: Optional parameter bounds.
            specification_name: Optional result label.
            rho_quad_model_accept: Ratio threshold for accepting model quality.
            rho_quad_model_reject: Ratio threshold for shrinking the radius.
            rho_step_accept_floor: Minimum ratio for accepting a step.
            Delta_increase_factor: Radius expansion factor.
            Delta_decrease_factor: Radius shrink factor.
            do_line_search: Whether to extend accepted steps by line search.
            num_reflections: Number of reflected bound segments to consider.
            do_broyden_jac_update: Whether to update quadratic approximation.
            broyden_jac_update_cadence: Fresh-Jacobian cadence under Broyden.
            root_loss_function: Optional robust root-loss function/name.
            f_scale: Robust-loss scale or ``'adaptive'``.
            keep_optimization_path: Whether to store optimisation path.
            prompt_user_for_more_iters: Whether/how to prompt after max iterations.
            jac_method: ``'analytic'``, ``'mid'``, or ``'fwd'``.
            try_newton_step: Whether to compare Newton steps.
            theta: Bound-interior truncation/reflection factor.
            jacobian_func: Optional user-provided residual Jacobian function.
            dense_threshold_mb: Dense Jacobian/covariance threshold in MB.
            do_analytic_jac_jit: Whether to JIT analytic Jacobian code.
            l2_penalties: Optional L2 penalty weights.
            regularize_to_values: L2 penalty target values.
            scale_l2_penalties: Whether L2 penalties scale with sample size/weights.

        Returns:
            ``SparseNonlinearLeastSquaresRegressionResults``.
        """

        _t = time.time()

        cov_kwds = format_cov_kwds(cov_kwds)
        check_cov_kwds(cov_type, cov_kwds)

        # get params
        assert jac_method in ('analytic', 'mid', 'fwd') or jac_method is None
        if jac_method is not None and jacobian_func is not None:
            raise Exception("Cannot specify kwd `jac_method` and give a `jacobian_func`!")

        start_params_orig = np.array(start_params) if start_params is not None else None
        # start_params = self._process_dict_input(start_params, default=0.0)
        # l2_penalties = self._process_dict_input(l2_penalties)
        # regularize_to_values = self._process_dict_input(regularize_to_values)

        start_params = dict_2_array(start_params, self.param_names, default_value=0.0)
        l2_penalties = dict_2_array(l2_penalties, self.param_names, default_value=0.0)
        regularize_to_values = dict_2_array(regularize_to_values, self.param_names, default_value=0.0)

        bounds = self.get_bounds(bounds)

        if subsample is not None:
            if subsample < self.num_params:
                raise Exception
            if debug:
                print(f"\nEstimating a starting point on {subsample}/{self.nobs} random observations...\n")
            random.seed(subsample_seed)
            subsample_idx = random.sample(range(self.nobs), subsample)
            residual_function_callable_temp = self.residual_function_callable.reindex(subsample_idx, inplace=False)
            if self.is_weighted:
                weights_temp = self.weights[subsample_idx]
            else:
                weights_temp = None
            result_subsample = nlls_minimize_internal(
                residual_function_callable_temp, weights=weights_temp, x0=start_params,
                jacobian_func=jacobian_func,
                num_params=self.num_params, bounds=bounds, debug=debug,
                Delta=Delta, Delta_floor=Delta_floor, do_line_search=do_line_search,
                xtol=xtol * 100, ftol=ftol * 100, gtol=gtol * 100, max_iter=max_iter,
                x_scale=x_scale, num_reflections=num_reflections,
                rho_quad_model_accept=rho_quad_model_accept, rho_quad_model_reject=rho_quad_model_reject,
                rho_step_accept_floor=rho_step_accept_floor,
                Delta_increase_factor=Delta_increase_factor, Delta_decrease_factor=Delta_decrease_factor,
                do_broyden_jac_update=do_broyden_jac_update, broyden_jac_update_cadence=broyden_jac_update_cadence,
                root_loss_function=root_loss_function, f_scale=f_scale,
                keep_optimization_path=False, jac_method=jac_method,
                try_newton_step=try_newton_step, theta=theta,
                dense_threshold_mb=dense_threshold_mb, do_analytic_jac_jit=do_analytic_jac_jit,
                l2_penalties=l2_penalties, regularize_to_values=regularize_to_values,
                scale_l2_penalties=scale_l2_penalties
            )
            start_params = result_subsample.params
            if result_subsample.is_bounded:
                start_params = .995 * start_params + .005 * result_subsample.start_params

        cov_type = cov_type.upper()
        if cov_type not in NLLS_COV_TYPES:
            if BOOTSTRAP not in cov_type:
                raise Exception(f"{cov_type=} is invalid for NLLS!")

        if debug:
            print(f"\nBeginning full estimation...\n")

        if self.is_weighted:
            weighted_tss = np.sum(self.weights * (self.endog - np.average(self.endog, weights=self.weights)) ** 2) / 2
        else:
            weighted_tss = sum((self.endog - self.endog.mean()) ** 2) / 2

        result = nlls_minimize_internal(
            self.residual_function_callable, weights=self.weights,
            x0=start_params, jacobian_func=jacobian_func,
            num_params=self.num_params, bounds=bounds, Delta=Delta,
            max_iter=max_iter, xtol=xtol, ftol=ftol, gtol=gtol, debug=debug,
            x_scale=x_scale, num_reflections=num_reflections,
            rho_quad_model_accept=rho_quad_model_accept, rho_quad_model_reject=rho_quad_model_reject,
            rho_step_accept_floor=rho_step_accept_floor,
            Delta_floor=Delta_floor, do_line_search=do_line_search,
            Delta_increase_factor=Delta_increase_factor, Delta_decrease_factor=Delta_decrease_factor,
            do_broyden_jac_update=do_broyden_jac_update, broyden_jac_update_cadence=broyden_jac_update_cadence,
            root_loss_function=root_loss_function, f_scale=f_scale, keep_optimization_path=keep_optimization_path,
            prompt_user_for_more_iters=prompt_user_for_more_iters, jac_method=jac_method,
            wtd_total_sum_of_squares=weighted_tss, try_newton_step=try_newton_step, theta=theta,
            dense_threshold_mb=dense_threshold_mb, do_analytic_jac_jit=do_analytic_jac_jit,
            l2_penalties=l2_penalties, regularize_to_values=regularize_to_values,
            scale_l2_penalties=scale_l2_penalties
        )

        df_resid = self.nobs - self.num_params
        df_model = self.num_params

        resid = result.resid
        wresid = result.wresid
        fittedvalues = self.endog - resid

        wssr, wsst, rsquared, rsquared_adj, scale, scale_mle, llf = self.get_rsquared(resid, df_resid)

        method = 'TR' if bounds is None else 'TR-Refl'

        fit_elapsed = time.time() - _t

        fit = SparseNonlinearLeastSquaresRegressionResults(
            self, result._params, None, cov_type, cov_kwds, resid, wresid, fittedvalues, self.nobs, df_resid,
            df_model,
            None, rsquared, rsquared_adj, wssr, wsst, method, test_level=test_level, use_t=use_t,
            fit_elapsed=fit_elapsed, cov_elapsed=None, converged=result.converged, iterations=result.iterations,
            optimality=result.optimality, cost=result.cost, scale=scale, scale_mle=scale_mle, llf=llf,
            penalty=result.penalty, objective=result.cost + result.penalty, jac=result.jac,
            grad=result.grad, subgrad=None, v=result.v,
            status=result.status, message=result.message, active_mask=result.active_mask,
            is_bounded=result.is_bounded, root_loss_function=result.root_loss_function_orig,
            solver_options={
                'xtol': xtol, 'gtol': gtol, 'Delta': Delta, 'max_iter': max_iter, 'subsample': subsample,
                'subsample_seed': subsample_seed, 'start_params': start_params_orig,
                'x_scale': x_scale, 'rho_quad_model_accept': rho_quad_model_accept,
                'Delta_floor': Delta_floor,
                'rho_quad_model_reject': rho_quad_model_reject,
                'Delta_increase_factor': Delta_increase_factor, 'Delta_decrease_factor': Delta_decrease_factor,
                'do_broyden_jac_update': do_broyden_jac_update,
                'broyden_jac_update_cadence': broyden_jac_update_cadence,
                'do_line_search': do_line_search,
                'root_loss_function': root_loss_function,
                'f_scale': f_scale, 'try_newton_step': try_newton_step,
                'jac_method': jac_method, 'dense_threshold_mb': dense_threshold_mb,
                'do_analytic_jac_jit': do_analytic_jac_jit,
                'scale_l2_penalties': scale_l2_penalties
            },
            specification_name=specification_name,
            optimization_result=result, is_penalized=False,
            optimization_path=result.optimization_path,
            jacobian_function_callable=result.jacobian_function_callable,
            l2_penalty=result.l2_penalties, regularize_to_values=result.regularize_to_values,
        )

        if np.any(result.active_mask != 0):
            if BOOTSTRAP not in cov_type.upper():
                compute_cov = False
            if debug:
                print("Cannot compute variance covariance with active constraints, except for bootstrap!")

        # compute var-covar
        _t_cov = time.time()
        if compute_cov:

            self.set_covariance_groups(self.get_cov_group_keyword(cov_kwds))

            if BOOTSTRAP in cov_type:

                if debug:
                    print(f"\nBeginning bootstrap variance-covariance estimation with "
                          f"{cov_kwds.get('n_samples', DEFAULT_NLLS_BOOTSTRAP_N_SAMPLES)} samples...", end='')

                param_estimation_func = self.get_bootstrap_function(
                    start_params=result.params, bounds=bounds, Delta=Delta, Delta_floor=Delta_floor,
                    max_iter=max_iter, xtol=xtol,
                    ftol=ftol, gtol=gtol, debug=False, x_scale=x_scale,
                    rho_quad_model_accept=rho_quad_model_accept, rho_quad_model_reject=rho_quad_model_reject,
                    rho_step_accept_floor=rho_step_accept_floor, Delta_increase_factor=Delta_increase_factor,
                    Delta_decrease_factor=Delta_decrease_factor,
                    num_reflections=num_reflections, do_broyden_jac_update=do_broyden_jac_update,
                    broyden_jac_update_cadence=broyden_jac_update_cadence, do_line_search=do_line_search,
                    root_loss_function=None, f_scale=f_scale, keep_optimization_path=keep_optimization_path,
                    prompt_user_for_more_iters=prompt_user_for_more_iters, jac_method=jac_method,
                    try_newton_step=try_newton_step, theta=theta, check_convergence=True, return_series=False,
                    jacobian_func=result.jacobian_function_callable,
                    regularize_to_values=regularize_to_values, l2_penalties=l2_penalties,
                    scale_l2_penalties=scale_l2_penalties,
                    weights=self.weights
                )

                do_bootstrap2(self.nobs, fit, param_estimation_func, groups=self.cov_groups,
                              n_samples=cov_kwds.get('n_samples', DEFAULT_NLLS_BOOTSTRAP_N_SAMPLES),
                              method=cov_kwds.get('method', DEFAULT_BB_METHOD),
                              alpha=cov_kwds.get('alpha', DEFAULT_BB_ALPHA),
                              seed=cov_kwds.get('seed', DEFAULT_NLLS_BOOTSTRAP_SEED), debug=debug,
                              use_correction=cov_kwds.get('use_correction', DEFAULT_NLLS_BOOTSTRAP_USE_CORRECTION),
                              max_processes=cov_kwds.get('max_processes', DEFAULT_BB_MAX_PROCESSES),
                              test_level=test_level, group_name=self.cov_groups_name)

            else:

                if debug:
                    print(f"\nComputing {cov_type} variance-covariance...", end='')

                cluster_name = self.cov_groups_name
                cov_params, num_groups, df_t_dist, small_samp_correct, cov_time \
                    = SparseVarianceCovariance2.compute_cov_params(
                    cov_type, cov_kwds, use_t, df_resid, wssr, resid, result.normalized_cov_params,
                    is_sure=False, exog_absorb_instrumented=result.jac,
                    debug=debug, _time=None, groups=self.cov_groups, weights=self.weights)

                fit.set_cov_params(
                    cov_params, cov_type, ci_lo=None, ci_hi=None, test_level=test_level, df_t_dist=df_t_dist,
                    cluster_name=cluster_name, debug=debug)

            if debug:
                print(f"...complete ({'%.2f' % (time.time() - _t_cov)}s)\n")

        else:
            fit.cov_type = None

        fit.cov_elapsed = time.time() - _t_cov

        if debug:
            print('NLLS Estimation Complete!\n\n')

        if subsample is not None:
            fit.iterations += result_subsample.iterations
            fit.fit_elapsed += result_subsample.fit_elapsed

        return fit

    @staticmethod
    def build_model_from_formula(formula, data, do_njit=True, debug=False, index=None, cov_groups=None,
                                 specification_name=None, custom_functions=None,
                                 dense_threshold_mb=DEFAULT_NLLS_DENSE_THRESHOLD_MB,
                                 jac_method=DEFAULT_NLLS_JAC_METHOD) -> SparseNonlinearLeastSquaresRegressionResults:
        """Parse a formula/data pair into a ``SparseNonlinearLeastSquaresModel``.

        Args:
            formula: NLLS formula string.  The left side identifies the response
                as ``[y]``; the right side defines predictions using ``[data]``
                and ``{parameter}`` tokens; an optional ``$ [w]`` suffix supplies
                weights.
            data: pandas ``DataFrame`` or dict-like data.
            do_njit: Whether to JIT compile generated prediction code.
            debug: Whether to print model-construction diagnostics.
            index: Optional row subset/indexer.
            cov_groups: Optional covariance group labels or column/expression name.
            specification_name: Optional model label.
            custom_functions: Optional functions exposed to generated formula code.
            dense_threshold_mb: Dense Jacobian/covariance threshold in MB.
            jac_method: Jacobian method assigned to the prediction callable.

        Returns:
            ``SparseNonlinearLeastSquaresModel`` with invalid rows removed.

        Examples
        --------
        Build (without fitting) a model so you can call ``.fit(...)``
        repeatedly with different starting values / covariance options:

        >>> import numpy as np, pandas as pd
        >>> from kanly.regression.nonlinear_least_squares.model import \\
        ...     SparseNonlinearLeastSquaresModel
        >>> rng = np.random.default_rng(0)
        >>> n = 250
        >>> df = pd.DataFrame({'x': rng.normal(size=n)})
        >>> df['y'] = 1.0 + 3.0 * np.exp(-0.5 * df['x']) + 0.4 * rng.normal(size=n)
        >>> model = SparseNonlinearLeastSquaresModel.build_model_from_formula(
        ...     '[y] ~ {Intercept} + {beta} * exp({gamma} * [x])', df)
        >>> fit_ols  = model.fit(start_params={'beta': 1.0, 'gamma': -0.1})  # doctest: +SKIP
        >>> fit_hc1  = model.fit(cov_type='HC1')                              # doctest: +SKIP
        """

        _t = time.time()

        if isinstance(data, dict):
            data = DataFrame(data, copy=False)

        if custom_functions is None:
            custom_functions = dict()

        formula_copy = formula

        # TODO, doesn't really make sense in nonlinear setting actually
        # without getting very complicated
        # formula = expand_lag_terms_in_formula(formula)

        _, index = get_nobs_from_index(data, index)

        if debug:
            print("Nonlinear Least Squares...constructing model...\n")

        # -----------
        # Get Weights
        if debug:
            print("\tChecking weights...", end='')
        weights_name = None
        weights = None
        is_weighted = '$' in formula_copy
        valid_indices_weights = True
        if is_weighted:
            wt_split = formula_copy.split('$')
            assert len(wt_split) == 2
            formula_copy = wt_split[0]
            weights_name, _, _ = parse_str_to_var_names(wt_split[1])
            weights_name = weights_name[0].replace('[', '').replace(']', '').replace(' ', '')
            weights = get_monomial_data(weights_name, data, data_dict=None, index=index)
            valid_indices_weights = (weights >= 0.0) & np.isfinite(weights)
            # weights *= len(weights) / np.sum(weights[valid_indices_weights])
            if debug:
                print(f"weights are '{weights_name}' ({'%.2f' % (time.time() - _t)}s).")
        else:
            if debug:
                print(f"no weights  ({'%.2f' % (time.time() - _t)}s).")

        # Split the formula
        endog_str, exog_str = tuple(formula_copy.split('~'))

        # ---------
        # Get Endog
        endog_name, _, _ = parse_str_to_var_names(endog_str)
        if debug:
            print(f"\tGetting endog column {endog_name}", end='')
        endog_name = endog_name[0].replace('[', '').replace(']', '').replace(' ', '')
        endog = get_monomial_data(
            endog_name, data, data_dict=None, index=index)
        valid_indices_endog = np.isfinite(endog)
        if debug:
            print(f" ({'%.2f' % (time.time() - _t)}s).")

        # --------
        # Get Exog
        prediction_func_callable, exog_result, valid_indices_exog = build_prediction_function_from_formula(
            exog_str, data, do_njit=do_njit, debug=debug, _t=_t, index=index,
            custom_functions=custom_functions, dense_threshold_mb=dense_threshold_mb, jac_method=jac_method)

        # -----
        # Get valid indices and remove invalid rows from prediction
        # and residual fucntions
        if debug:
            print("\nChecking valid rows across endog, prediction function and weights...", end='')
        valid_obs_rows = valid_indices_weights & valid_indices_exog & valid_indices_endog

        if cov_groups is not None and isinstance(cov_groups, str):
            if debug:
                print(f"\nGetting cov groups {cov_groups}...", end='')
            cov_groups, cov_groups_name = ModelBase.get_covariance_groups_internal(
                endog.shape[0], cov_groups, data=data, index=index, valid_obs_rows=valid_obs_rows,
                current_cov_groups=None, current_cov_groups_name=None)
            if debug:
                print(f"done ({'%.2f' % (time.time() - _t)}s).")
        else:
            cov_groups, cov_groups_name = None, None

        if not np.all(valid_obs_rows):
            num_invalid = len(endog) - np.count_nonzero(valid_obs_rows)

            # Keep response, weights, covariance groups, and generated
            # prediction arrays on exactly the same valid-row subset.
            prediction_func_callable.reindex(valid_obs_rows, inplace=True)
            endog = endog[valid_obs_rows].copy()
            if weights is not None:
                weights = weights[valid_obs_rows].copy()

            if debug:
                print(f"{num_invalid} invalid rows found ({'%.2f' % (time.time() - _t)}s).")
        else:
            if debug:
                print(f"all valid ({'%.2f' % (time.time() - _t)}s).")

        if index is None:
            valid_obs_rows = np.arange(len(data))[valid_obs_rows]
        else:
            valid_obs_rows = np.arange(len(index))[valid_obs_rows]

        fdi = FormulaDesignInfoBase(formula, data)
        model = SparseNonlinearLeastSquaresModel(
            endog, prediction_func_callable, formula_design_info=fdi, endog_name=endog_name, weights=weights,
            weights_name=weights_name, valid_row_indices=valid_obs_rows, model_elapsed=time.time() - _t,
            do_njit=do_njit, cov_groups=cov_groups, cov_groups_name=cov_groups_name, index=index,
            specification_name=specification_name)

        if debug:
            print("\nModel complete!")
            print(model)

        return model

    def build_model(self, data, index=None, debug=False):
        """Rebuild this formula model on new data.

        Args:
            data: New data frame/dict to parse with ``self.formula``.
            index: Optional row subset/indexer.
            debug: Whether to print construction diagnostics.

        Returns:
            New ``SparseNonlinearLeastSquaresModel``.
        """
        return self.build_model_from_formula(
            self.formula, data, index=index, debug=debug, do_njit=self.do_njit)

    def get_rsquared(self, resid, df_resid):
        """
        :return:  wssr, wsst, rsquared, rsquared_adj, scale, scale_mle, llf

        Args:
            resid: Residual vector from a fit.
            df_resid: Residual degrees of freedom.

        Returns:
            Tuple ``(wssr, wsst, rsquared, rsquared_adj, scale, scale_mle, llf)``.
        """
        wssr = np.sum((self.weights if self.is_weighted else 1.0) * resid ** 2)
        wsst = np.sum((self.endog - np.average(self.endog, weights=self.weights if self.is_weighted else None)) ** 2
                      * (self.weights if self.is_weighted else 1.0))

        rsquared = 1.0 - wssr / wsst
        if df_resid > 0:
            rsquared_adj = 1 - (1 - rsquared) * (self.nobs - 1) / df_resid
        else:
            rsquared_adj = np.nan

        scale_mle = wssr / self.nobs
        scale = wssr / (self.nobs - self.num_params) if self.nobs > self.num_params else np.nan

        llf = -self.nobs / 2 * np.log(2 * np.pi * scale_mle) - wssr / (2 * scale_mle)
        if self.is_weighted:
            llf += 0.5 * np.log(self.weights).sum()

        return wssr, wsst, rsquared, rsquared_adj, scale, scale_mle, llf

    # TODO "ignore_column_mismatch"?
    def predict(self, params, data=None, index=None, debug=False, ignore_column_mismatch=None):
        """Generate predictions from parameter values, optionally on new data.

        Args:
            params: Parameter vector, dict, or pandas ``Series``.  Dict/Series
                inputs are aligned by parameter name and missing parameters are
                filled with zero.
            data: Optional new ``DataFrame``/``SparseDataFrame``.  When ``None``,
                predictions use the model's stored data.
            index: Optional row subset for rebuilding on ``data``.
            debug: Whether to print model-construction diagnostics when using
                new data.

        Returns:
            NumPy array of fitted values.
        """

        if data is None:
            model = self
        elif isinstance(data, (DataFrame, SparseDataFrame)):
            model = self.build_model(data, index=index, debug=debug)
        else:
            raise Exception(f'type {type(data)} not supported!')

        prediction_function_callable = model.prediction_function_callable
        param_names = model.param_names

        # work on param types
        if isinstance(params, (Series, dict)):

            if isinstance(params, dict):
                params = Series(index=list(params.keys()), data=list(params.values()))

            if not set(params.index) <= set(param_names):
                raise Exception(f"Supplied params has index {params.index} "
                                f", but column names for data are {param_names}!")
            params_new = Series(index=param_names, data=np.zeros(len(param_names)))
            params_new[params.index] = params

        elif isinstance(params, (np.ndarray, list)):
            params_new = params

        else:
            raise Exception(f"Unsupported params type {type(params)}!")

        return prediction_function_callable(params_new)

    # def fit_alternating_wip(
    #         self, start_params=None,
    #
    #         # TRF params
    #         Delta=DEFAULT_NLLS_DELTA, Delta_floor=DEFAULT_NLLS_DELTA_FLOOR,
    #         max_iter=DEFAULT_NLLS_MAX_ITER, xtol=DEFAULT_NLLS_XTOL,
    #         ftol=DEFAULT_NLLS_FTOL, gtol=DEFAULT_NLLS_GTOL, compute_cov=True, cov_type=DEFAULT_NLLS_COV_TYPE,
    #         cov_kwds=None, use_t=True, test_level=DEFAULT_NLLS_TEST_LEVEL, debug=False,
    #         x_scale=DEFAULT_NLLS_X_SCALE, bounds=None, specification_name=None,
    #         rho_quad_model_accept=DEFAULT_NLLS_RHO_QUAD_MODEL_ACCEPT,
    #         rho_quad_model_reject=DEFAULT_NLLS_RHO_QUAD_MODEL_REJECT,
    #         rho_step_accept_floor=DEFAULT_NLLS_RHO_STEP_ACCEPT_FLOOR,
    #         Delta_increase_factor=DEFAULT_NLLS_DELTA_INCREASE_FACTOR,
    #         Delta_decrease_factor=DEFAULT_NLLS_DELTA_DECREASE_FACTOR,
    #         do_line_search=DEFAULT_NLLS_DO_LINE_SEARCH,
    #         num_reflections=DEFAULT_NLLS_NUM_REFLECTIONS,
    #         do_broyden_jac_update=DEFAULT_NLLS_DO_BROYDEN_JAC_UPDATE,
    #         broyden_jac_update_cadence=DEFAULT_NLLS_BROYDEN_JAC_UPDATE_CADENCE,
    #         keep_optimization_path=DEFAULT_NLLS_KEEP_OPTIMIZATION_PATH,
    #         prompt_user_for_more_iters=DEFAULT_NLLS_PROMPT_USER_FOR_MORE_ITERS, jac_method=DEFAULT_NLLS_JAC_METHOD,
    #         try_newton_step=DEFAULT_NLLS_TRY_NEWTON_STEP, theta=DEFAULT_NLLS_REFLECTION_THETA,
    #
    #         # CD params
    #         active_set=DEFAULT_NLLS_EN_ACTIVE_SET, max_iter_cd=DEFAULT_NLLS_EN_MAX_ITER,
    #         ftol_cd=DEFAULT_NLLS_EN_FTOL, gtol_cd=DEFAULT_NLLS_EN_GTOL, xtol_cd=DEFAULT_NLLS_EN_XTOL,
    #         num_shrinkage=DEFAULT_NLLS_EN_NUM_SHRINKAGE, shrink_factor=DEFAULT_NLLS_EN_SHRINK_FACTOR,
    #         selection=DEFAULT_NLLS_EN_SELECTION,
    #
    #         max_outer_loops=10
    #         ):
    #
    #     cov_kwds = format_cov_kwds(cov_kwds)
    #     cov_kwds = cov_kwds.copy()
    #
    #     for i in range(max_outer_loops):
    #
    #         fit_trf = self.fit(
    #             start_params, bounds=bounds, Delta=Delta, Delta_floor=Delta_floor, max_iter=max_iter, xtol=xtol,
    #             ftol=ftol, gtol=gtol, compute_cov=False, cov_type=cov_type, cov_kwds=cov_kwds, debug=debug,
    #             test_level=test_level, x_scale=x_scale,
    #             rho_quad_model_accept=rho_quad_model_accept, rho_quad_model_reject=rho_quad_model_reject,
    #             rho_step_accept_floor=rho_step_accept_floor, Delta_increase_factor=Delta_increase_factor,
    #             Delta_decrease_factor=Delta_decrease_factor, specification_name=specification_name,
    #             num_reflections=num_reflections, do_broyden_jac_update=do_broyden_jac_update,
    #             broyden_jac_update_cadence=broyden_jac_update_cadence, do_line_search=do_line_search,
    #             root_loss_function=None, f_scale=1.0, keep_optimization_path=keep_optimization_path,
    #             prompt_user_for_more_iters=prompt_user_for_more_iters, jac_method=jac_method,
    #             try_newton_step=try_newton_step, theta=theta, use_t=use_t,
    #         )
    #
    #         print("%-6d" % i, "%12.4e" % fit_trf.cost, ' ', end='')
    #
    #         fit_en = self.fit_en(
    #             start_params=fit_trf.params.values.copy(), bounds=bounds, positive=False, alpha=0, l1_ratio=0,
    #             debug=debug, active_set=active_set, max_iter=max_iter_cd, ftol=ftol_cd, gtol=gtol_cd, xtol=xtol_cd,
    #             num_shrinkage=num_shrinkage, shrink_factor=shrink_factor, normalize=False,
    #             prompt_user_for_more_iters=prompt_user_for_more_iters, selection=DEFAULT_NLLS_EN_SELECTION)
    #
    #         start_params = fit_en.params.values.copy()
    #
    #         print("%12.4e" % (fit_en.cost * self.nobs),
    #               "%10.2e" % (fit_trf.cost / (fit_en.cost * self.nobs) - 1))
    #         if abs(fit_trf.cost / (fit_en.cost * self.nobs) - 1) < 1e-6:
    #             break
    #
    #     fit_trf = self.fit(
    #         start_params, bounds=bounds, Delta=Delta, Delta_floor=Delta_floor, max_iter=max_iter, xtol=xtol, ftol=ftol,
    #         gtol=gtol, compute_cov=compute_cov, cov_type=cov_type, cov_kwds=cov_kwds, debug=debug,
    #         test_level=test_level, x_scale=x_scale, rho_quad_model_accept=rho_quad_model_accept,
    #         rho_quad_model_reject=rho_quad_model_reject, rho_step_accept_floor=rho_step_accept_floor,
    #         Delta_increase_factor=Delta_increase_factor, Delta_decrease_factor=Delta_decrease_factor,
    #         specification_name=specification_name, num_reflections=num_reflections,
    #         do_broyden_jac_update=do_broyden_jac_update, broyden_jac_update_cadence=broyden_jac_update_cadence,
    #         do_line_search=do_line_search, root_loss_function=None, f_scale=1.0,
    #         keep_optimization_path=keep_optimization_path, prompt_user_for_more_iters=prompt_user_for_more_iters,
    #         jac_method=jac_method, try_newton_step=try_newton_step, theta=theta, use_t=use_t,
    #     )
    #     return fit_trf

    def get_bootstrap_en_function(
            self, start_params=None, bounds=None, positive=False, debug=False,
            alpha=DEFAULT_NLLS_EN_ALPHA, l1_ratio=DEFAULT_NLLS_EN_L1_RATIO,
            active_set=DEFAULT_NLLS_EN_ACTIVE_SET, max_iter=DEFAULT_NLLS_EN_MAX_ITER,
            ftol=DEFAULT_NLLS_EN_FTOL, gtol=DEFAULT_NLLS_EN_GTOL, xtol=DEFAULT_NLLS_EN_XTOL,
            num_shrinkage=DEFAULT_NLLS_EN_NUM_SHRINKAGE, shrink_factor=DEFAULT_NLLS_EN_SHRINK_FACTOR,
            normalize=DEFAULT_NLLS_EN_NORMALIZE, prompt_user_for_more_iters=DEFAULT_NLLS_PROMPT_USER_FOR_MORE_ITERS,
            selection=DEFAULT_NLLS_EN_SELECTION, regularize_to_values=None,
            jac_method=DEFAULT_NLLS_EN_JAC_METHOD,
            return_series=False, check_convergence=False,
            weights=None):
        """Build a bootstrap callback for elastic-net NLLS refits.

        Args:
            start_params: Starting parameters for each bootstrap refit.
            bounds: Optional parameter bounds.
            positive: Positivity constraint input.
            debug: Whether to print optimisation diagnostics.
            alpha: Elastic-net penalty strength.
            l1_ratio: Elastic-net L1 ratio.
            active_set: Whether to use active-set coordinate cycling.
            max_iter: Maximum coordinate-descent iterations.
            ftol: Objective-change tolerance.
            gtol: Subgradient tolerance.
            xtol: Coordinate-change tolerance.
            num_shrinkage: Maximum backtracking attempts per coordinate.
            shrink_factor: Backtracking shrinkage factor.
            normalize: Whether to normalise penalties by Jacobian scale.
            prompt_user_for_more_iters: Whether/how to prompt after max iterations.
            selection: Coordinate selection strategy.
            regularize_to_values: Penalty target values.
            jac_method: Derivative method.
            return_series: Whether to return pandas ``Series`` estimates.
            check_convergence: Whether non-converged bootstrap fits return ``None``.
            weights: Optional base sampling weights multiplied by bootstrap weights.

        Returns:
            Callable accepting a bootstrap-weight vector and returning estimated
            parameters.
        """

        alpha, positive, start_params, regularize_to_values, bounds \
            = self._process_en_input_dicts(alpha, positive, start_params, regularize_to_values, bounds)

        def __temp_func(boot_weights):
            """Refit the elastic-net model for one bootstrap draw.

            Args:
                boot_weights: Bootstrap frequency/weight vector.

            Returns:
                Parameter estimates, or ``None`` if convergence is required and
                the refit did not converge.
            """
            if weights is not None:
                # Combine bootstrap sampling weights with any original model
                # weights so both sources affect the refit objective_function.
                boot_weights *= weights
            temp_result = nlls_elastic_net_minimize_internal_coordinate_descent(
                self.residual_function_callable, weights=boot_weights, x0=start_params,
                num_params=self.num_params,
                alpha=alpha, l1_ratio=l1_ratio,
                num_shrinkage=num_shrinkage, shrink_factor=shrink_factor,
                positive=positive, bounds=bounds, debug=debug, active_set=active_set,
                max_iter=max_iter, xtol=xtol, ftol=ftol, gtol=gtol, normalize=normalize,
                prompt_user_for_more_iters=prompt_user_for_more_iters,
                selection=selection, regularize_to_values=regularize_to_values,
                jac_method=jac_method
            )

            if check_convergence:
                if not temp_result['converged']:
                    return None

            values = temp_result['params'].copy()
            if return_series:
                values = Series(values, index=self.param_names)

            del temp_result
            return values

        return __temp_func

    def get_bootstrap_function(
            self, start_params=None, Delta=DEFAULT_NLLS_DELTA, Delta_floor=DEFAULT_NLLS_DELTA_FLOOR,
            max_iter=DEFAULT_NLLS_MAX_ITER, xtol=DEFAULT_NLLS_XTOL, ftol=DEFAULT_NLLS_FTOL, gtol=DEFAULT_NLLS_GTOL,
            debug=False, x_scale=DEFAULT_NLLS_X_SCALE, bounds=None,
            rho_quad_model_accept=DEFAULT_NLLS_RHO_QUAD_MODEL_ACCEPT,
            rho_quad_model_reject=DEFAULT_NLLS_RHO_QUAD_MODEL_REJECT,
            rho_step_accept_floor=DEFAULT_NLLS_RHO_STEP_ACCEPT_FLOOR,
            Delta_increase_factor=DEFAULT_NLLS_DELTA_INCREASE_FACTOR,
            Delta_decrease_factor=DEFAULT_NLLS_DELTA_DECREASE_FACTOR,
            do_line_search=DEFAULT_NLLS_DO_LINE_SEARCH,
            num_reflections=DEFAULT_NLLS_NUM_REFLECTIONS,
            do_broyden_jac_update=DEFAULT_NLLS_DO_BROYDEN_JAC_UPDATE,
            broyden_jac_update_cadence=DEFAULT_NLLS_BROYDEN_JAC_UPDATE_CADENCE,
            root_loss_function=None, f_scale=1.0, keep_optimization_path=DEFAULT_NLLS_KEEP_OPTIMIZATION_PATH,
            prompt_user_for_more_iters=DEFAULT_NLLS_PROMPT_USER_FOR_MORE_ITERS, jac_method=DEFAULT_NLLS_JAC_METHOD,
            try_newton_step=DEFAULT_NLLS_TRY_NEWTON_STEP, theta=DEFAULT_NLLS_REFLECTION_THETA,
            return_series=False, check_convergence=False, jacobian_func=None,
            regularize_to_values=None, l2_penalties=None, scale_l2_penalties=None,
            weights=None
    ):
        """Build a bootstrap callback for trust-region NLLS refits.

        Args:
            start_params: Starting parameters for each bootstrap refit.
            Delta: Initial trust-region radius.
            Delta_floor: Minimum trust-region radius.
            max_iter: Maximum solver iterations.
            xtol: Parameter-step tolerance.
            ftol: Objective-change tolerance.
            gtol: Gradient tolerance.
            debug: Whether to print diagnostics.
            x_scale: Parameter scaling.
            bounds: Optional parameter bounds.
            rho_quad_model_accept: Ratio threshold for accepting model quality.
            rho_quad_model_reject: Ratio threshold for shrinking the radius.
            rho_step_accept_floor: Minimum ratio for accepting a step.
            Delta_increase_factor: Radius expansion factor.
            Delta_decrease_factor: Radius shrink factor.
            do_line_search: Whether to extend accepted steps by line search.
            num_reflections: Number of reflected bound segments to consider.
            do_broyden_jac_update: Whether to update quadratic approximation.
            broyden_jac_update_cadence: Fresh-Jacobian cadence under Broyden.
            root_loss_function: Optional robust root-loss function/name.
            f_scale: Robust-loss scale.
            keep_optimization_path: Whether to store optimisation path.
            prompt_user_for_more_iters: Whether/how to prompt after max iterations.
            jac_method: Jacobian method.
            try_newton_step: Whether to compare Newton steps.
            theta: Bound-interior truncation/reflection factor.
            return_series: Whether to return pandas ``Series`` estimates.
            check_convergence: Whether non-converged refits return ``None``.
            jacobian_func: Optional residual Jacobian callable.
            regularize_to_values: L2 penalty target values.
            l2_penalties: Optional L2 penalty weights.
            scale_l2_penalties: Whether L2 penalties scale with sample size/weights.
            weights: Optional base sampling weights multiplied by bootstrap weights.

        Returns:
            Callable accepting a bootstrap-weight vector and returning estimated
            parameters.
        """
        if self.param_names is None:
            param_names = ['x%d' % j for j in range(self.num_params)]
        else:
            param_names = self.param_names

        def __temp_func(boot_weights):
            """Refit the trust-region model for one bootstrap draw.

            Args:
                boot_weights: Bootstrap frequency/weight vector.

            Returns:
                Parameter estimates, or ``None`` if convergence is required and
                the refit did not converge.
            """
            if weights is not None:
                # Combine bootstrap sampling weights with any original model
                # weights so both sources affect the refit objective_function.
                boot_weights *= weights
            temp_result = nlls_minimize_internal(
                self.residual_function_callable, weights=boot_weights,
                x0=start_params, jacobian_func=jacobian_func,
                num_params=self.num_params, bounds=bounds, Delta=Delta,
                max_iter=max_iter, xtol=xtol, ftol=ftol, gtol=gtol, debug=debug,
                x_scale=x_scale, num_reflections=num_reflections,
                rho_quad_model_accept=rho_quad_model_accept, rho_quad_model_reject=rho_quad_model_reject,
                rho_step_accept_floor=rho_step_accept_floor,
                Delta_floor=Delta_floor, do_line_search=do_line_search,
                Delta_increase_factor=Delta_increase_factor, Delta_decrease_factor=Delta_decrease_factor,
                do_broyden_jac_update=do_broyden_jac_update, broyden_jac_update_cadence=broyden_jac_update_cadence,
                root_loss_function=root_loss_function, f_scale=f_scale,
                keep_optimization_path=keep_optimization_path,
                regularize_to_values=regularize_to_values, l2_penalties=l2_penalties,
                scale_l2_penalties=scale_l2_penalties,
                prompt_user_for_more_iters=prompt_user_for_more_iters, jac_method=jac_method,
                wtd_total_sum_of_squares=np.nan, try_newton_step=try_newton_step, theta=theta)

            if check_convergence:
                if not temp_result.converged:
                    return None

            values = temp_result.params.copy()
            if return_series:
                values = Series(values, index=param_names)

            del temp_result
            return values

        return __temp_func

    def get_residual_analytical_jacobian(self, dense_threshold_mb=DEFAULT_DENSE_THRESHOLD_MB, debug=False, do_jit=True):
        """Return an analytic Jacobian callable for residuals.

        Args:
            dense_threshold_mb: Dense Jacobian threshold in MB.
            debug: Whether to print automatic-differentiation diagnostics.
            do_jit: Whether to JIT compile generated Jacobian code.

        Returns:
            Tuple ``(jacobian_callable, info_dict)``.
        """
        return self.residual_function_callable.get_analytical_jacobian(
            dense_threshold_mb=dense_threshold_mb, debug=debug, do_jit=do_jit)

    def get_residual_analytical_partial_derivative(self, arg_number, debug=False, do_jit=True):
        """Return an analytic residual partial-derivative callable.

        Args:
            arg_number: Parameter index to differentiate with respect to.
            debug: Whether to print automatic-differentiation diagnostics.
            do_jit: Whether to JIT compile generated derivative code.

        Returns:
            Callable returning derivative values for one parameter.
        """
        return self.residual_function_callable.get_analytical_partial_derivative(
            arg_number, debug=debug, do_jit=do_jit)

    def get_residual_analytical_partial_derivatives(self, debug=False, do_jit=True):
        """Return analytic residual partial-derivative callables for all parameters.

        Args:
            debug: Whether to print automatic-differentiation diagnostics.
            do_jit: Whether to JIT compile generated derivative code.

        Returns:
            List of derivative callables, one per parameter.
        """
        return self.residual_function_callable.get_analytical_partial_derivatives(debug=debug, do_jit=do_jit)

    def get_log_likelihood_function(self, transform_scale=False, t_df=DEFAULT_LLF_T_DF, is_variance_weights=True):
        """

        :param is_variance_weights: in the likelihood whether to treat the weights as a variance scaling
               `is_variance_weights=True` or as a frequency weighting `is_variance_weights=False`
        :param t_df`, if `t_df=None`, a normal likelhood, elst t(df=t_df)

        Log Likelihood
        Assumes last argument is sigma^2, not sigma

        In case for variance weights:

            In normal case:
                \log\mathcal{L}&=-\frac{n}{2}\log\left\{ 2\pi\sigma^{2}\right\} +\frac{1}{2}\sum_{i}\log\left\{ w_{i}\right\} -\frac{1}{2\sigma^{2}}\sum_{i}w_{i}r_{i}^{2}(\beta)

            In t-distribution case:
                \log\mathcal{L}&=n\log\left(\frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right)}\right)-\frac{n}{2}\log\left(\pi\nu\right)-\frac{n}{2}\log\left(\tau^{2}\right)+\frac{1}{2}\sum_{i}\log w_{i}-\frac{\nu+1}{2}\sum_{i}\log\left\{ 1+w_{i}\frac{1}{\nu}\frac{\left(y_{i}-f\left(x_{i};\beta\right)\right)^{2}}{\tau^{2}}\right\}


        In case for frequency weights:

            In normal case:
                -\frac{\log\left(2\pi\sigma^{2}\right)}{2}\sum_{i}w_{i}-\frac{1}{2\sigma^{2}}\sum_{i}w_{i}r_{i}(\beta)^{2}

            In t-distribution case:
                << Not implemented yet >>

        MLE for both models is the same, just different scaling on scale parameter.
        MLE is the same for beta, but for scale we have

            sigma^2[ivw=True] = sigma^2[ivw=False] *  mean(weights)
        """

        n = self.nobs

        is_weighted = self.is_weighted
        if is_weighted:
            weights = self.weights
            half_sum_log_wts = np.log(weights).sum() / 2
        else:
            weights = 1
            half_sum_log_wts = 0.0

        # Normal case
        if t_df is None or t_df == 0 or t_df == np.inf:

            # variance weights
            if is_variance_weights:

                def llf(params):
                    """Evaluate the normal log-likelihood with variance weights.

                    Args:
                        params: Parameter vector with regression parameters
                            followed by ``sigma_sq`` (or log ``sigma_sq`` when
                            ``transform_scale=True``).

                    Returns:
                        Scalar log-likelihood value.
                    """
                    params = np.array(params)
                    resid = self.residual_function_callable(params[:-1])
                    sigma_sq = np.exp(params[-1]) if transform_scale else params[-1]
                    ssr = np.sum(weights * resid ** 2)
                    ll = (
                            -n / 2 * np.log(2 * np.pi * sigma_sq) + half_sum_log_wts
                            - 1 / (2 * sigma_sq) * ssr
                    )
                    return ll

                return llf

            # frequency weights
            else:

                if is_weighted:
                    sum_wts = np.sum(weights)
                else:
                    sum_wts = n

                def llf(params):
                    """Evaluate the normal log-likelihood with frequency weights.

                    Args:
                        params: Parameter vector with regression parameters
                            followed by ``sigma_sq`` (or log ``sigma_sq`` when
                            ``transform_scale=True``).

                    Returns:
                        Scalar log-likelihood value.
                    """
                    params = np.array(params)
                    resid = self.residual_function_callable(params[:-1])
                    sigma_sq = np.exp(params[-1]) if transform_scale else params[-1]
                    ssr = np.sum(weights * resid ** 2)
                    ll = (
                        (-1 / 2 * np.log(2 * np.pi * sigma_sq) * sum_wts
                         - 1 / (2 * sigma_sq) * ssr)
                    )
                    return ll

                return llf


        # t-distribution with `t_df` degrees of freedom
        else:
            assert isinstance(t_df, (float, int)) and t_df > 0

            # variance weights
            if is_variance_weights:

                constant_t = n * (
                        loggamma((t_df + 1) / 2)
                        - loggamma(t_df / 2)
                        - np.log(np.pi * t_df) / 2
                )

                def llf(params):
                    """Evaluate the t-distribution log-likelihood with variance weights.

                    Args:
                        params: Parameter vector with regression parameters
                            followed by the residual variance/scale parameter.

                    Returns:
                        Scalar log-likelihood value.
                    """
                    r = self.residual_function_callable(params[:-1])
                    return (
                            constant_t
                            + half_sum_log_wts
                            - n / 2 * np.log(params[-1])
                            - (t_df + 1) / 2 * np.log(1 + (weights * r ** 2) / (t_df * params[-1])).sum()
                    )

                return llf

            # frequency weights
            else:
                raise NotImplementedError("frequency weights (`is_var_weights=False`) '"
                                          "'not implemented for t distribution yet!")

    def get_log_likelihood_function_obs(self, transform_scale=False, t_df=DEFAULT_LLF_T_DF):
        """
        Log Likelihood
        Assumes last argument is sigma^2, not sigma

        \log\mathcal{L}&=-\frac{n}{2}\log\left\{ 2\pi\sigma^{2}\right\} +\frac{1}{2}\sum_{i}\log\left\{ w_{i}\right\} -\frac{1}{2\sigma^{2}}\sum_{i}w_{i}r_{i}^{2}(\beta)
        """

        is_weighted = self.is_weighted
        if is_weighted:
            weights = self.weights
            half_log_wts = np.log(weights) / 2
        else:
            weights = 1.0
            half_log_wts = 0.0

        # Normal case
        if t_df is None or t_df == 0 or t_df == np.inf:

            def llf_obs(x):
                """Evaluate per-observation normal log-likelihood values.

                Args:
                    x: Parameter vector with regression parameters followed by
                        ``sigma_sq`` (or log ``sigma_sq`` when transformed).

                Returns:
                    Vector of observation-level log-likelihood contributions.
                """
                sigma_sq = np.exp(x[-1]) if transform_scale else x[-1]
                return (
                        -1 / 2 * np.log(2 * np.pi * sigma_sq / weights)
                        - self.residual_function_callable(x[:-1]) ** 2 / (2 * sigma_sq / weights)
                )

            return llf_obs

        # t-distribution with `t_df` degrees of freedom
        else:
            assert isinstance(t_df, (float, int)) and t_df > 0

            constant_t = (
                    loggamma((t_df + 1) / 2)
                    - loggamma(t_df / 2)
                    - np.log(np.pi * t_df) / 2
            )

            def llf_obs(params):
                """Evaluate per-observation t log-likelihood values.

                Args:
                    params: Parameter vector with regression parameters followed
                        by residual variance/scale.

                Returns:
                    Vector of observation-level log-likelihood contributions.
                """
                r = self.residual_function_callable(params[:-1])
                return (
                        constant_t
                        + half_log_wts
                        - 1 / 2 * np.log(params[-1])
                        - (t_df + 1) / 2 * np.log(1 + (weights * r ** 2) / (t_df * params[-1]))
                )

            return llf_obs

    def get_log_likelihood_function_analytical_gradient(
            self, dense_threshold_mb=DEFAULT_DENSE_THRESHOLD_MB, debug=False, do_jit=True,
            transform_scale=False):
        """
        Assumes last argument is sigma^2, not sigma
        \log\mathcal{L}&=-\frac{n}{2}\log\left\{ 2\pi\sigma^{2}\right\} +\frac{1}{2}\sum_{i}\log\left\{ w_{i}\right\} -\frac{1}{2\sigma^{2}}\sum_{i}w_{i}r_{i}^{2}(\beta)
        frac{\partial\log\mathcal{L}}{\partial\sigma^{2}}&=-\frac{n}{2\sigma^{2}}+\frac{1}{2\sigma^{4}}\sum_{i}w_{i}r_{i}^{2}(\beta)
        frac{\partial\log\mathcal{L}}{\partial\beta_{j}}&=-\frac{1}{\sigma^{2}}\sum_{i}w_{i}r_{i}(\beta)\frac{\partial r_{i}(\beta)}{\partial\beta_{j}}\text{}
        """

        jac_resid_func, _ = self.get_residual_analytical_jacobian(
            dense_threshold_mb=dense_threshold_mb, debug=debug, do_jit=do_jit)
        n = self.nobs
        p = self.num_params
        is_weighted = self.is_weighted

        def __gradient(params):
            """Evaluate the analytic gradient of the normal log-likelihood.

            Args:
                params: Parameter vector with regression parameters followed by
                    scale/variance parameter.

            Returns:
                Gradient vector of length ``num_params + 1``.
            """
            sigma_sq = np.exp(params[-1]) if transform_scale else params[-1]
            beta = params[:-1]
            resid = self.residual_function_callable(beta)
            jac_resid = jac_resid_func(beta)
            is_sparse = isspmatrix(jac_resid)
            result = np.zeros(p + 1)

            wtd_resid = self.weights * resid if is_weighted else resid
            wssr = np.dot(wtd_resid, resid)
            if transform_scale:
                result[-1] = (-n + 1.0 / sigma_sq * wssr) / 2.0
            else:
                result[-1] = (-n / sigma_sq + 1.0 / (sigma_sq ** 2) * wssr) / 2.0

            for k in range(p):
                result[k] = -1.0 / sigma_sq * np.dot(wtd_resid, jac_resid.getcol(k).toarray().ravel()
                if is_sparse else jac_resid[:, k])

            return result

        return __gradient

#
# if __name__ == '__main__':
#     import numpy as np
#     import pandas as pd
#
#     from kanly.api import nlls
#
#     n = 1_000
#     np.random.seed(0)
#     df = pd.DataFrame({'x': np.random.randn(n)})
#     df['p'] = 1 / (1 + np.exp(-(.4 + .9 * df.x)))
#     df['w'] = 1+np.random.rand(n)
#     df['y'] = (np.random.rand(n) < df['p']).astype(float)
#
#     fit = nlls('[y] ~ 1.0 / (1.0 + np.exp({alpha} + {beta} * [x])) $ [w]',
#                df,
#                #cov_type='hc1',
#                debug=True,
#                jac_method='mid',
#                cov_type='bootstrap',
#                specification_name='logistic regression')
#
#     fit.model.get_residual_analytical_jacobian()
#
#     print(fit)
#
#     llf = fit.model.get_log_likelihood_function()
#     print(fit.llf)
#     print(llf(np.hstack([fit.params, fit.scale_mle])))
#
#     from scipy.stats import norm
#
#     print(np.sum(np.log(
#         norm.pdf(fit.model.residual_function_callable(fit.params), loc=0, scale=fit.scale_mle ** .5 / df.w ** .5))))
#
#     # print(fit)
#     #
#     # func = fit.model.get_bootstrap_function(start_params=fit.params)
#     #
#     # from kanly.bootstrap import bootstrap_entire_procedure
#     #
#     # print(bootstrap_entire_procedure(
#     #     nobs=n,
#     #     n_samples=250,
#     #     func=fit.model.get_bootstrap_en_function(alpha=.1, start_params=fit.params),
#     #     debug=True
#     # ))
#
# if __name__ == '__main__':
#     import numpy as np
#     import pandas as pd
#     import scipy as sp
#
#     from kanly.api import nlls
#
#     n = 1_000
#     np.random.seed(0)
#     df = pd.DataFrame({'x': np.random.randn(n)})
#     df['p'] = 1 / (1 + np.exp(-(.4 + .9 * df.x)))
#     df['w'] = 1+np.random.rand(n)
#     df['y'] = (np.random.rand(n) < df['p']).astype(float)
#
#     from numba import vectorize
#     import numba_scipy
#     @vectorize
#     def _gamma(x):
#         return sp.special.gamma(x)
#
#     fit = nlls('[y] ~ {a} * _gamma([x])',
#                df, debug=True,
#                do_njit=True,
#                custom_functions={'_gamma': _gamma})
#     print(fit)
#
#
#
# if __name__ == '__main__':
#     import numpy as np
#     import pandas as pd
#
#     from kanly.api import lm, elastic_net, nlls, compare_results, nlls_minimize_internal, nlls_en, NLLS_EN
#
#     n = 375
#     np.random.seed(0)
#     df = pd.DataFrame({
#         'x': np.random.randn(n),
#         'w': np.random.randint(1, 4, n),
#         'grp': np.random.randint(0, 12, n),
#     })
#     df['y'] = 1.2 - 0.3 * df['x'] + .6 * np.random.randn(n) + .15 * (df.grp == 2)
#
#     fit0 = nlls_en('[y] ~ {Intercept} + {x}*[x] + {I(x**2)}*[x**2] $ [w]', df,
#                    alpha={'x': .55}, l1_ratio=.4,
#                    normalize=False,
#                    debug=False,
#                    #cov_type='cluster',
#                    #cov_kwds={'groups': 'grp'},
#                    )
#
#
#     def pred_func(a):
#         return a[0] + a[1] * df.x + a[2] * df.x ** 2
#
#
#     # print(fit0)
#
#     fit1 = NLLS_EN(df.y, pred_func, param_names=['Intercept', 'x', 'I(x**2)'], weights=df.w,
#                    alpha={'x': .55}, l1_ratio=.4, normalize=False)
#
#     print(compare_results([fit0, fit1]))
#
#     # fit1 = SparseNonlinearLeastSquaresModel.NLLS(
#     #     df.y, pred_func, param_names=['Intercept', 'x', 'I(x**2)'], weights=df.w,
#     #     cov_type='cluster',
#     #     cov_groups=df.grp.values,
#     # )
#     # print(compare_results([fit1, fit0]))


