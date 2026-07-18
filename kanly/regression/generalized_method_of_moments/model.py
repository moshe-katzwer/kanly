"""Model builders and high-level fit methods for generalized method of moments.

This module defines ``SparseGeneralizedMethodOfMomentsModel``, the main GMM
container used by ``kanly.api.gmm``, ``gmm_iv_linear``, ``gmm_iv_nonlinear``,
and ``gmm_mle``.  A model stores observation-level moment functions, their
sample averages, a Jacobian for those average moments, fitting metadata, and
helpers for formula-based moment construction.
"""

from __future__ import absolute_import, print_function

import time

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.sparse import csc_matrix, diags, csr_matrix, isspmatrix

from kanly.bootstrap.bootstrap import do_bootstrap2, DEFAULT_BB_ALPHA, DEFAULT_BB_SEED, DEFAULT_BB_METHOD, \
    DEFAULT_BB_MAX_PROCESSES
from kanly.formula.data_getter import SparseDataGetter, parse_formula
from kanly.formula.formula_design_info import FormulaDesignInfo, FormulaDesignInfoBase
from kanly.formula.keys import (EXOG_KEY)
from kanly.formula.sparse_term_to_data_methods import get_nobs_from_index
from kanly.regression.cov_types import _parse_bootstrap
from kanly.regression.covariance_cluster_groups_model_base import CovarianceClusterGroupsModelBase
from kanly.regression.generalized_method_of_moments.constants import (
    DEFAULT_GMM_F_TOL, DEFAULT_GMM_X_TOL, DEFAULT_GMM_G_TOL, DEFAULT_GMM_MAX_ITER, ONE_STEP,
    DEFAULT_GMM_USE_T, DEFAULT_GMM_TEST_LEVEL, DEFAULT_GMM_COV_TYPE, DEFAULT_GMM_DELTA,
    DEFAULT_GMM_METHOD, GMM_METHODS, GMM_COV_TYPES, SANDWICH, BOOTSTRAP, CLUSTER,
    DEFAULT_GMM_BOOTSTRAP_N_SAMPLES, DEFAULT_ITERATIVE_GMM_X_TOL, DEFAULT_ITERATIVE_GMM_MAX_ITER,
    GMM_MODEL_TYPES, GMM_GENERAL, GMM_IV_LINEAR, GMM_MLE, GMM_IV_NONLINEAR
)
from kanly.regression.generalized_method_of_moments.gmm_internal import fit_gmm_outer_loop
from kanly.regression.generalized_method_of_moments.gmm_variance_covariance import get_gmm_var_covar, get_Omega
from kanly.regression.generalized_method_of_moments.regression_results import GMMRegressionResults
from kanly.regression.linear_models.model import SparseLinearModel
from kanly.regression.nonlinear_least_squares.formula import sparse_nonlinear_formula_parser as snfp
from kanly.regression.nonlinear_least_squares.formula.sparse_nonlinear_formula_parser import (
    build_prediction_function_from_formula)
from kanly.regression.nonlinear_least_squares.model import SparseNonlinearLeastSquaresModel
from kanly.utils.linalg_utils import (
    csc_matrix_by_column_array_broadcast, get_eigenvals_and_condition_number_internal, get_matrix_inverse_internal)

EPS = np.sqrt(np.finfo(float).eps)


class SparseGeneralizedMethodOfMomentsModel(CovarianceClusterGroupsModelBase):
    """Sparse GMM model object that estimates parameters from moment conditions.

    The model represents observation-level moments ``g_i(theta)`` as a sparse
    matrix with shape ``(nobs, num_moments)``.  Fitting minimizes
    ``E[g(theta)]' W E[g(theta)]`` and then packages estimates, covariance, and
    diagnostics in ``GMMRegressionResults``.

    Examples
    --------
    There are four formula entry points on ``kanly.api``:

    - ``gmm(formulas, data)``        — generic GMM with hand-written moments
    - ``gmm_iv_linear('y ~ x | z')``  — linear instrumental variables
    - ``gmm_iv_nonlinear(resid, instruments)`` — nonlinear IV
    - ``gmm_mle(formula_llf, data)`` — MLE as GMM via score equations

    Linear IV-GMM (instruments after ``|``):

    >>> import numpy as np, pandas as pd
    >>> from kanly.api import gmm_iv_linear
    >>> rng = np.random.default_rng(0)
    >>> n = 1_000
    >>> z1 = rng.normal(size=n); z2 = rng.normal(size=n)
    >>> x  = 0.5*z1 + 0.5*z2 + rng.normal(size=n)        # endogenous
    >>> y  = 1.0 + 2.0*x + rng.normal(size=n)
    >>> df = pd.DataFrame({'y': y, 'x': x, 'z1': z1, 'z2': z2})
    >>> fit = gmm_iv_linear('y ~ x | z1 + z2', df,
    ...                     do_2sls=True)                  # doctest: +SKIP
    >>> fit.params.round(2)                                 # doctest: +SKIP
    Intercept    1.00
    x            2.00
    dtype: float64

    See Also
    --------
    :func:`gmm`, :func:`gmm_iv_linear`, :func:`gmm_iv_nonlinear`,
    :func:`gmm_mle`.
    """

    def __init__(self, moment_func_obs, nobs, num_moments, num_params=None, param_names=None, specification_name=None,
                 model_elapsed=0, from_formula=False, formula=None, data=None, moment_vals=None,
                 moment_func_mean=None, moment_func_mean_jacobian=None, index=None, model_type=GMM_GENERAL,
                 cov_groups=None, cov_groups_name=None, valid_obs_rows=None
                 ):
        """Initialize a GMM model from observation-level moment functions.

        Args:
            moment_func_obs: Callable ``theta -> moments`` returning an
                observation-by-moment matrix.
            nobs: Number of observations represented by the moment function.
            num_moments: Number of moment conditions.
            num_params: Number of parameters. Optional when ``param_names`` is
                supplied.
            param_names: Ordered parameter names. Optional when ``num_params``
                is supplied.
            specification_name: Optional label used in printed summaries.
            model_elapsed: Seconds spent constructing the model object.
            from_formula: Whether this model was constructed from formula
                strings.
            formula: Formula or moment-formula metadata for display.
            data: Original data source used to build the model.
            moment_vals: Optional target moment values. Defaults to zero moments.
            moment_func_mean: Optional callable for average moments. If omitted,
                it is computed as the column mean of ``moment_func_obs``.
            moment_func_mean_jacobian: Optional Jacobian callable for average
                moments. If omitted, finite differences are used.
            index: Optional row index/subset used during construction.
            model_type: One of the GMM model type constants.
            cov_groups: Optional covariance cluster group labels.
            cov_groups_name: Optional cluster-group display name.
            valid_obs_rows: Optional valid-row mask inherited from formula
                parsing.
        """
        fdi = FormulaDesignInfoBase(None, data)
        super().__init__(nobs, index=index, valid_obs_rows=valid_obs_rows, formula_design_info=fdi,
                         specification_name=specification_name, cov_groups=cov_groups, cov_groups_name=cov_groups_name)

        self.moment_func_obs = self.get_gmm_function_wrapper(moment_func_obs)

        if moment_func_mean is None:
            self.moment_func_mean = self.get_moment_func_mean(self.moment_func_obs)
        else:
            self.moment_func_mean = moment_func_mean

        if moment_func_mean_jacobian is None:
            self.moment_func_mean_jacobian = self.get_jacobian_function(self.moment_func_mean, num_moments, num_params)
        else:
            self.moment_func_mean_jacobian = moment_func_mean_jacobian

        if param_names is None and num_params is None:
            raise Exception
        elif param_names is None:
            param_names = [f'<param{j}>' for j in range(num_params)]
        elif num_params is None:
            num_params = len(param_names)

        self.param_names = param_names
        self.num_params = num_params
        self.num_moments = num_moments
        self.model_elapsed = model_elapsed

        self.from_formula = from_formula
        self.formula = formula
        self.data = data
        self.moment_vals = moment_vals

        self.over_identified = self.num_params < self.num_moments
        if self.num_params > self.num_moments:
            raise Exception(f"Model is under-identified, {self.num_moments} moments < {self.num_params} params")

        self.index = index

        assert model_type in GMM_MODEL_TYPES
        self.model_type = model_type

    @staticmethod
    def get_moment_func_mean(moment_func_obs):
        """Build a sample-average moment function from observation moments.

        Args:
            moment_func_obs: Callable returning an ``nobs x num_moments`` matrix
                for a parameter vector and optional weights.

        Returns:
            Callable ``(theta, weights=None) -> 1-D average moment array``.
        """
        return lambda theta, weights=None: \
            np.asarray(moment_func_obs(theta, weights).mean(axis=0)).flatten()

    @staticmethod
    def get_jacobian_function(moment_func_mean, num_moments, num_params):
        """Create a finite-difference Jacobian for average moments.

        Args:
            moment_func_mean: Callable returning the sample-average moments.
            num_moments: Number of moment conditions.
            num_params: Number of parameters.

        Returns:
            Callable returning the ``num_moments x num_params`` Jacobian matrix.
        """
        def jacobian_wrapper(theta, weights=None, eps=EPS * 100):
            """Evaluate a forward-difference Jacobian of the mean moments.

            Args:
                theta: Parameter vector.
                weights: Optional bootstrap or frequency weights.
                eps: Relative finite-difference step size.

            Returns:
                Dense Jacobian matrix with rows for moments and columns for
                parameters.
            """
            J = np.zeros((num_moments, num_params))
            E_g = moment_func_mean(theta, weights)
            for i in range(num_params):
                dx = eps * (1.0 + np.abs(theta[i]))
                xi = theta.copy()
                xi[i] += dx
                E_g_i = moment_func_mean(xi, weights)

                J[:, i] = (E_g_i - E_g) / dx
            return J

        return jacobian_wrapper

    @staticmethod
    def get_gmm_function_wrapper(gmm_function):
        """Wrap an observation-level moment callable in sparse/weight handling.

        Args:
            gmm_function: Callable ``theta -> moments``.

        Returns:
            Callable that returns a sparse CSC moment matrix and optionally
            applies observation weights column-wise.
        """
        def gmm_function_wrapper(theta, weights=None):
            """Evaluate wrapped observation-level moments.

            Args:
                theta: Parameter vector.
                weights: Optional observation weights to multiply into each
                    moment row.

            Returns:
                Sparse CSC matrix of observation-level moments.
            """
            G = gmm_function(theta)
            if not isspmatrix(G):
                G = csc_matrix(G)
            if weights is not None:
                G = csc_matrix_by_column_array_broadcast(G, weights)
            return G
        return gmm_function_wrapper

    @staticmethod
    def build_model_from_formulas(formulas, data, specification_name=None, debug=False, do_njit=True,
                                  custom_functions=dict(), moment_vals=None, index=None):
        """Build a GMM model from formula strings describing moments.

        Args:
            formulas: List of moment specifications. A string is a single
                moment; a tuple of strings multiplies sub-moment callables
                together, e.g. residual times instrument.
            data: DataFrame-like or dict data source.
            specification_name: Optional label used in summaries.
            debug: Whether to print formula-building progress.
            do_njit: Whether generated formula functions should use Numba.
            custom_functions: Extra functions available to formula parsing.
            moment_vals: Optional nonzero target values for the moments.
            index: Optional row subset.

        Returns:
            ``SparseGeneralizedMethodOfMomentsModel`` ready to fit.
        """

        _t = time.time()

        if isinstance(data, dict):
            data = DataFrame(data, copy=False)

        if debug:
            print('Building GMM model from formulas')

        num_moments = len(formulas)
        formula_dict = dict()

        if moment_vals is not None:
            moment_vals = np.array(moment_vals).flatten().astype(float)
            assert moment_vals.shape[0] == num_moments

        for i, f in enumerate(formulas):
            if isinstance(f, str):
                formulas[i] = (f,)

        nobs, index = get_nobs_from_index(data, index)
        valid_indices_master = np.full(nobs, True, dtype=bool)

        for f in formulas:
            if debug:
                print(f"\n\tBuilding callable function for moment {f}...")
            func_subs = []
            for fsub in f:
                if debug:
                    print(f"\t\tBuilding callable function for sub-moment callable {fsub}...", end='')
                if fsub in formula_dict.keys():
                    func_subs.append(formula_dict[fsub])
                else:
                    # Each unique formula fragment is compiled once and reused
                    # across any moments that multiply by the same expression.
                    temp_func, _, valid_indices_exog = snfp.build_prediction_function_from_formula(
                        fsub, data, do_njit=do_njit, custom_functions=custom_functions, index=index)
                    formula_dict[fsub] = temp_func
                    func_subs.append(temp_func)
                    # if len(valid_indices_exog) < len(valid_indices_master):
                    #     print(valid_indices_exog)
                    #     valid_indices_exog_new = np.full(len(valid_indices_master), False)
                    #     valid_indices_exog_new[valid_indices_exog] = True
                    #     valid_indices_exog = valid_indices_exog_new
                    # print('r333 ', valid_indices_master.shape, valid_indices_exog.shape)
                    valid_indices_master &= valid_indices_exog
                if debug:
                    print(f'{"%.2fs" % (time.time() - _t)}')

        if np.count_nonzero(valid_indices_master) < len(data):
            if debug:
                print("\n\tRe-indexing prediction functions for valid indices...", end="")
            for f in func_subs:
                f.reindex(valid_indices_master, inplace=True)

        if debug:
            print('\n\n')

        param_names = set()
        for f in formula_dict.values():
            param_names |= set(f.param_names)
        param_names = sorted(list(param_names))
        num_params = len(param_names)

        nobs = set([f.nobs for f in formula_dict.values()])
        assert len(nobs) == 1
        nobs = nobs.pop()

        param_name_2_idx = pd.Series(index=param_names, data=np.arange(num_params))

        def moment_func_callable(theta):
            """Evaluate all formula-defined observation-level moments.

            Args:
                theta: Full parameter vector ordered by ``param_names``.

            Returns:
                Sparse CSC matrix with one column per moment formula.
            """

            theta = np.asarray(theta)
            func_val_dict = dict()
            fvals = []

            for i, fs in enumerate(formulas):
                fval = 1.0
                for f in fs:
                    if f in func_val_dict.keys():
                        fval *= func_val_dict.get(f)
                    else:
                        func = formula_dict[f]
                        # Formula fragments only receive the subset of theta
                        # corresponding to the parameters they actually use.
                        func_val_dict[f] = func(theta[param_name_2_idx[func.param_names]])
                        fval *= func_val_dict[f]
                if moment_vals is not None:
                    fval -= moment_vals[i]
                fvals.append(csr_matrix(fval))

            values = np.hstack([v.data for v in fvals])
            indices = np.hstack([v.indices for v in fvals])
            indptr = np.cumsum([0] + [len(v.data) for v in fvals])

            m = csc_matrix((values, indices, indptr), shape=(nobs, num_moments))

            return m

        # return moment_func_callable, param_names, nobs, num_moments, formula_dict
        model = SparseGeneralizedMethodOfMomentsModel(
            moment_func_callable, nobs,
            num_moments, num_params=num_params, param_names=param_names, specification_name=specification_name,
            model_elapsed=time.time() - _t, from_formula=True, formula=formulas, data=data, moment_vals=moment_vals,
            index=index, valid_obs_rows=index,
        )

        if debug:
            print(model)

        return model

    def __str__(self):
        """Return a human-readable model summary string.

        Returns:
            Text describing model dimensions, type, and formula moments when
            available.
        """
        return (
            "\n".join(
                ['=' * 50,
                 'GMM Model' + (f'      ({self.specification_name})' if self.specification_name is not None else ''),
                 f'Model Type:    {self.model_type}',
                 '-' * 50,
                 f'No. Obs:       {self.nobs}',
                 f'No. Params:    {self.num_params}',
                 f'No. Moments:   {self.num_moments}',
                 '',
                 ] + (
                    ([''] + [
                        f'Moment {j}: ' + (f if isinstance(f, str) else (' * '.join([f'({s})' for s in f])))
                        for j, f in enumerate(self.formula)
                    ]) if self.formula is not None else []
                ) + [
                    '=' * 50
                ]
            )
        )

    @staticmethod
    def GMM(moment_func, nobs, num_moments, num_params, param_names=None, W=None, debug=False, specification_name=None,
            use_t=DEFAULT_GMM_USE_T, test_level=DEFAULT_GMM_TEST_LEVEL, method=DEFAULT_GMM_METHOD, moment_vals=None,
            max_iter=DEFAULT_GMM_MAX_ITER, xtol=DEFAULT_GMM_X_TOL, ftol=DEFAULT_GMM_F_TOL, gtol=DEFAULT_GMM_G_TOL,
            Delta=DEFAULT_GMM_DELTA, cov_type=DEFAULT_GMM_COV_TYPE, cov_kwds=None, start_params=None,
            iterative_gmm_max_iter=DEFAULT_ITERATIVE_GMM_MAX_ITER, iterative_gmm_x_tol=DEFAULT_ITERATIVE_GMM_X_TOL,
            ) -> GMMRegressionResults:
        """Fit a GMM model from a raw observation-level moment callable.

        Args:
            moment_func: Callable ``theta -> nobs x num_moments`` moments.
            nobs: Number of observations.
            num_moments: Number of moment conditions.
            num_params: Number of parameters.
            param_names: Optional ordered parameter names.
            W: Optional initial weighting matrix.
            debug: Whether to print optimization progress.
            specification_name: Optional result label.
            use_t: Whether inference uses t distributions.
            test_level: Test size for confidence intervals and p-values.
            method: GMM weighting update method.
            moment_vals: Optional target moment values metadata.
            max_iter: Maximum trust-region iterations for each inner solve.
            xtol: Parameter-step convergence tolerance.
            ftol: Moment/objective_function_ convergence tolerance.
            gtol: Gradient convergence tolerance.
            Delta: Initial trust-region radius.
            cov_type: Covariance type.
            cov_kwds: Optional covariance keyword arguments.
            start_params: Optional starting parameter vector.
            iterative_gmm_max_iter: Maximum outer weighting updates for
                iterative GMM.
            iterative_gmm_x_tol: Parameter-change tolerance for iterative GMM.

        Returns:
            Fitted ``GMMRegressionResults``.

        Examples
        --------
        Matrix-form GMM with a user-supplied moment callable. Example:
        method-of-moments estimation of mean and variance of a Normal:

        >>> import numpy as np
        >>> from kanly.api import GMM
        >>> rng = np.random.default_rng(0)
        >>> y = rng.normal(loc=2.0, scale=3.0, size=500)
        >>> def moments(theta):
        ...     mu, sigma2 = theta
        ...     m1 = y - mu
        ...     m2 = (y - mu) ** 2 - sigma2
        ...     return np.column_stack([m1, m2])
        >>> fit = GMM(moments, nobs=500, num_moments=2, num_params=2,    # doctest: +SKIP
        ...           param_names=['mu', 'sigma2'],
        ...           start_params=np.array([0.0, 1.0]))
        >>> fit.params.round(2)                                          # doctest: +SKIP
        mu        2.05
        sigma2    9.13
        dtype: float64
        """
        model = SparseGeneralizedMethodOfMomentsModel(
            moment_func, nobs, num_moments, num_params=num_params, moment_vals=moment_vals,
            param_names=param_names, specification_name=specification_name,
            model_elapsed=0, from_formula=False, formula=None, data=None)
        return model.fit(
            W=W, method=method, max_iter=max_iter, xtol=xtol, start_params=start_params,
            ftol=ftol, gtol=gtol, Delta=Delta, debug=debug,
            iterative_gmm_max_iter=iterative_gmm_max_iter, iterative_gmm_x_tol=iterative_gmm_x_tol,
            use_t=use_t, test_level=test_level, cov_type=cov_type, cov_kwds=cov_kwds)

    @staticmethod
    def gmm(formulas, data, W=None, debug=False, use_t=DEFAULT_GMM_USE_T, specification_name=None, do_njit=True,
            start_params=None, moment_vals=None, index=None,
            test_level=DEFAULT_GMM_TEST_LEVEL,
            method=DEFAULT_GMM_METHOD, max_iter=DEFAULT_GMM_MAX_ITER,
            xtol=DEFAULT_GMM_X_TOL,
            ftol=DEFAULT_GMM_F_TOL, gtol=DEFAULT_GMM_G_TOL, Delta=DEFAULT_GMM_DELTA,
            cov_type=DEFAULT_GMM_COV_TYPE, cov_kwds=None,
            iterative_gmm_max_iter=DEFAULT_ITERATIVE_GMM_MAX_ITER, iterative_gmm_x_tol=DEFAULT_ITERATIVE_GMM_X_TOL,
            custom_functions=dict()
            ) -> GMMRegressionResults:
        """Fit a formula-defined GMM model.

        Args:
            formulas: Moment formula list. Strings define moments directly;
                tuples multiply formula fragments together.
            data: DataFrame-like or dict data source.
            W: Optional initial weighting matrix.
            debug: Whether to print model-building and fitting progress.
            use_t: Whether inference uses t distributions.
            specification_name: Optional result label.
            do_njit: Whether generated formula functions should use Numba.
            start_params: Optional starting parameter vector.
            moment_vals: Optional nonzero target values for moments.
            index: Optional row subset.
            test_level: Test size for confidence intervals and p-values.
            method: GMM weighting update method.
            max_iter: Maximum trust-region iterations for each inner solve.
            xtol: Parameter-step convergence tolerance.
            ftol: Moment/objective_function_ convergence tolerance.
            gtol: Gradient convergence tolerance.
            Delta: Initial trust-region radius.
            cov_type: Covariance type.
            cov_kwds: Optional covariance keyword arguments.
            iterative_gmm_max_iter: Maximum outer weighting updates for
                iterative GMM.
            iterative_gmm_x_tol: Parameter-change tolerance for iterative GMM.
            custom_functions: Extra functions available to formula parsing.

        Returns:
            Fitted ``GMMRegressionResults``.

        Examples
        --------
        Estimate ``mu`` and ``sigma2`` of a Normal distribution from the
        first two moment formulas:

        >>> import numpy as np, pandas as pd
        >>> from kanly.api import gmm
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({'y': rng.normal(loc=2.0, scale=3.0, size=500)})
        >>> fit = gmm(                                        # doctest: +SKIP
        ...     ['[y] - {mu}',
        ...      '([y] - {mu})**2 - {sigma2}'],
        ...     df, start_params={'mu': 0.0, 'sigma2': 1.0})

        Each entry in ``formulas`` is a moment ``E[g_k(y; theta)] = 0``;
        parameters appear in ``{...}`` braces, data columns in ``[...]``
        brackets, exactly as in :func:`nlls`.

        See Also
        --------
        :func:`gmm_iv_linear` : linear IV-GMM with formula ``'y ~ x | z'``.
        :func:`gmm_iv_nonlinear` : nonlinear IV-GMM.
        :func:`gmm_mle` : MLE expressed as GMM score equations.
        """
        model = SparseGeneralizedMethodOfMomentsModel.build_model_from_formulas(
            formulas, data, debug=debug, specification_name=specification_name, do_njit=do_njit,
            custom_functions=custom_functions, moment_vals=moment_vals, index=index)
        return model.fit(
            W=W, method=method, max_iter=max_iter, xtol=xtol, ftol=ftol, gtol=gtol, Delta=Delta, debug=debug,
            iterative_gmm_max_iter=iterative_gmm_max_iter, iterative_gmm_x_tol=iterative_gmm_x_tol,
            use_t=use_t, test_level=test_level, cov_type=cov_type, cov_kwds=cov_kwds, start_params=start_params)

    def objective(self, theta, W=None):
        """Evaluate the GMM quadratic objective_function_ at a parameter vector.

        Args:
            theta: Parameter vector.
            W: Optional weighting matrix. Defaults to identity.

        Returns:
            Scalar objective_function_ ``gbar(theta)' W gbar(theta)``.
        """
        if W is None:
            W = np.eye(self.num_moments)
        g_mean = self.moment_func_mean(theta)
        return g_mean.dot(W).dot(g_mean)

    def fit(self, W=None, start_params=None, method=DEFAULT_GMM_METHOD, use_t=DEFAULT_GMM_USE_T,
            test_level=DEFAULT_GMM_TEST_LEVEL, max_iter=DEFAULT_GMM_MAX_ITER, xtol=DEFAULT_GMM_X_TOL,
            ftol=DEFAULT_GMM_F_TOL, gtol=DEFAULT_GMM_G_TOL, Delta=DEFAULT_GMM_DELTA, debug=False, _time=None,
            compute_cov=True, cov_type=DEFAULT_GMM_COV_TYPE, cov_kwds=None,
            iterative_gmm_max_iter=DEFAULT_ITERATIVE_GMM_MAX_ITER, iterative_gmm_x_tol=DEFAULT_ITERATIVE_GMM_X_TOL):
        """Estimate model parameters and covariance.

        Args:
            W: Optional initial weighting matrix. Defaults to identity.
            start_params: Optional starting parameter vector. Defaults to zeros.
            method: GMM method: one-step, two-step, or iterative.
            use_t: Whether inference uses t distributions.
            test_level: Test size for confidence intervals and p-values.
            max_iter: Maximum trust-region iterations for each inner solve.
            xtol: Parameter-step convergence tolerance.
            ftol: Moment/objective_function_ convergence tolerance.
            gtol: Gradient convergence tolerance.
            Delta: Initial trust-region radius.
            debug: Whether to print fitting progress.
            _time: Optional externally supplied start time.
            compute_cov: Whether to compute covariance or bootstrap inference.
            cov_type: Covariance type, one of sandwich, cluster, or bootstrap.
            cov_kwds: Optional covariance keyword arguments.
            iterative_gmm_max_iter: Maximum outer weighting updates for
                iterative GMM.
            iterative_gmm_x_tol: Parameter-change tolerance for iterative GMM.

        Returns:
            Fitted ``GMMRegressionResults``.
        """

        if _time is None:
            _time = time.time()

        if cov_kwds is None:
            cov_kwds = dict()

        if self.over_identified:
            method = method.upper()
            assert method in GMM_METHODS
        else:
            # Exactly identified systems do not benefit from updating W.
            method = ONE_STEP

        cov_type = cov_type.upper()
        if cov_type not in GMM_COV_TYPES:
            if BOOTSTRAP not in cov_type:
                raise Exception(f"`cov_type` {cov_type} is not in {GMM_COV_TYPES}!")

        if debug:
            print(self)
            print()
            print(
                "\n".join(
                    ['=' * 50,
                     'GMM Estimation Parameters',
                     '-' * 50,
                     f'method:                  {method.upper()}',
                     f'max_iter (iterative):    {"%.2e" % iterative_gmm_x_tol}',
                     f'x_tol (iterative):       {iterative_gmm_max_iter}',
                     '',
                     f'max_iter (opt):          {max_iter}',
                     f'Delta (opt):             {Delta}',
                     f'xtol (opt):              {"%.2e" % xtol}',
                     f'ftol (opt):              {"%.2e" % gtol}',
                     f'gtol (opt):              {"%.2e" % ftol}',
                     '=' * 50
                     ]
                )
            )

        if W is None:
            # Identity weighting gives the standard first-step GMM objective_function_.
            W = np.eye(self.num_moments)
        else:
            W = np.asarray(W)

        if start_params is None:
            start_params = np.zeros(self.num_params)

        params, opt_result, n_iters, W = fit_gmm_outer_loop(
            self.nobs, self.moment_func_mean, self.moment_func_mean_jacobian, self.moment_func_obs, self.num_params,
            start_params, W, max_iter, xtol, ftol, gtol, Delta, debug, method, iterative_gmm_max_iter,
            iterative_gmm_x_tol, weights=None, _time=_time)

        self.set_covariance_groups(self.get_cov_group_keyword(cov_kwds))
        # Cluster labels are only used for cluster and bootstrap covariance
        # paths; sandwich covariance works with raw observation-level moments.
        cov_groups = self.cov_groups if cov_type in (CLUSTER, BOOTSTRAP) else None

        Omega, num_clusters = get_Omega(self.moment_func_obs, self.nobs, params, cluster_groups=cov_groups)

        df_model = self.num_params
        df_resid = self.nobs - df_model
        df_t_dist = df_resid if cov_type != CLUSTER else num_clusters - 1

        if cov_kwds.get('use_correction', True):
            if cov_type == SANDWICH:
                ss_correction = self.nobs / (self.nobs - self.num_params)
            elif cov_type == CLUSTER:
                ss_correction = (self.nobs - 1) / df_resid
                if num_clusters > 1:
                    ss_correction *= (num_clusters) / (num_clusters - 1)
        else:
            ss_correction = 1.0

        fval = opt_result['fval']
        converged = opt_result['converged']
        avg_moment_vals = opt_result['moments']
        message = opt_result['message']

        if compute_cov and cov_type in (CLUSTER, SANDWICH):
            var_covar, _, condition_number, eigenvals = get_gmm_var_covar(
                self.moment_func_mean_jacobian, self.moment_func_obs, self.nobs, params, W,
                ss_correction=ss_correction, Omega=Omega, cluster_groups=cov_groups)
        else:

            G = self.moment_func_mean_jacobian(params)
            GWG = G.T.dot(W).dot(G)
            eigenvals, condition_number = get_eigenvals_and_condition_number_internal(GWG)

            var_covar = None

        fit = GMMRegressionResults(
            self, self.nobs, params, var_covar, df_model, df_resid, df_t_dist, converged, fval, avg_moment_vals,
            n_iters, message, opt_result, W, Omega, method, self.num_params, self.num_moments,
            self.param_names, self.over_identified, eigenvals, condition_number,
            cov_type=cov_type, cov_kwds=cov_kwds.copy(), test_level=test_level,
            use_t=use_t, specification_name=self.specification_name, fit_elapsed=time.time() - _time,
            model_elapsed=self.model_elapsed, formula=self.formula, moment_vals=self.moment_vals
        )

        if compute_cov and BOOTSTRAP in cov_type:

            cov_type, cov_kwds = _parse_bootstrap(cov_type, cov_kwds)

            def param_estimation_func(bootstrap_weights):
                """Refit GMM parameters for one bootstrap weight draw.

                Args:
                    bootstrap_weights: Observation weights generated by the
                        bootstrap driver.

                Returns:
                    Parameter vector for a converged bootstrap fit, otherwise
                    ``None`` so the bootstrap driver can discard it.
                """
                params_boot, opt_result, _, _ = fit_gmm_outer_loop(
                    self.nobs, self.moment_func_mean, self.moment_func_mean_jacobian, self.moment_func_obs,
                    self.num_params, params, W, max_iter, xtol, ftol, gtol, Delta, False, method,
                    iterative_gmm_max_iter, iterative_gmm_x_tol, weights=bootstrap_weights)
                if opt_result['converged']:
                    return params_boot
                else:
                    return None

            do_bootstrap2(self.nobs, fit, param_estimation_func, groups=cov_groups,
                          n_samples=cov_kwds.get('n_samples', DEFAULT_GMM_BOOTSTRAP_N_SAMPLES),
                          seed=cov_kwds.get('seed', DEFAULT_BB_SEED), debug=debug, use_correction=True,
                          test_level=test_level, group_name=None,
                          method=cov_kwds.get('method', DEFAULT_BB_METHOD),
                          max_processes=cov_kwds.get('max_processes', DEFAULT_BB_MAX_PROCESSES),
                          alpha=cov_kwds.get('method', DEFAULT_BB_ALPHA))

        return fit

    @staticmethod
    def build_linear_iv_gmm_model_from_formula(formula, data, specification_name=None, index=None):
        """Build a linear IV-GMM model from a formula.

        Args:
            formula: Linear IV formula such as ``'y ~ x | z'``.
            data: DataFrame-like or dict data source.
            specification_name: Optional result label.
            index: Optional row subset.

        Returns:
            ``SparseGeneralizedMethodOfMomentsModel`` with moments
            ``E[z_i * (y_i - x_i'beta)] = 0``.
        """

        _t = time.time()
        linmod = SparseLinearModel.build_model_from_formula(formula, data, index=index)

        def moment_func_mean(theta, weights=None):
            """Evaluate mean linear IV moments.

            Args:
                theta: Coefficient vector.
                weights: Optional bootstrap weights.

            Returns:
                Average instrument-residual moments.
            """
            theta = csc_matrix(theta).reshape((-1, 1))
            u = linmod.endog - linmod.exog.dot(theta)
            if linmod.is_weighted:
                u = csc_matrix_by_column_array_broadcast(u, linmod.weights)
            if weights is not None:
                u = csc_matrix_by_column_array_broadcast(u, weights)
            g = linmod.instruments.transpose().dot(u)
            return g.toarray().flatten() / linmod.nobs

        def moment_func_mean_jacobian(theta, weights=None):
            """Evaluate the analytic Jacobian of linear IV mean moments.

            Args:
                theta: Coefficient vector, unused because the Jacobian is
                    constant for linear IV moments.
                weights: Optional bootstrap weights.

            Returns:
                Dense Jacobian matrix of average moments.
            """
            exog = linmod.exog
            if linmod.is_weighted:
                exog = csc_matrix_by_column_array_broadcast(exog, linmod.weights)
            if weights is not None:
                exog = csc_matrix_by_column_array_broadcast(exog, weights)
            frozen_jacobian = -(linmod.instruments.transpose().dot(exog).toarray()) / linmod.nobs
            return frozen_jacobian

        def moment_func_obs(theta, weights=None):
            """Evaluate observation-level linear IV moments.

            Args:
                theta: Coefficient vector.
                weights: Optional bootstrap weights.

            Returns:
                Sparse matrix with rows ``z_i * u_i(theta)``.
            """
            theta = csc_matrix(theta).reshape((-1, 1))
            u = linmod.endog - linmod.exog.dot(theta)
            mmo = linmod.instruments.copy()
            wts = u.toarray().flatten()
            if weights is not None:
                wts *= weights
            if linmod.is_weighted:
                wts *= linmod.weights
            mmo = csc_matrix_by_column_array_broadcast(mmo, wts)
            return mmo

        model_elapsed = time.time() - _t

        # Build display formulas corresponding to each instrument-residual
        # moment. The actual model uses the optimized sparse callables above.
        resid_formula = f'{linmod.endog_name} - (' + ' + '.join([f'{{{x}}}*[{x}]' for x in linmod.exog_names]) + ')'
        resid_formula = resid_formula.replace('*[Intercept]', '')
        gmm_formula = [(f'({resid_formula}) * [{z}]',) for z in linmod.instrument_names]
        if linmod.is_weighted:
            gmm_formula = [(f'{f[0]} * [{linmod.weights_name}]',) for f in gmm_formula]

        model = SparseGeneralizedMethodOfMomentsModel(
            moment_func_obs, linmod.nobs, linmod.instruments.shape[1], num_params=linmod.exog.shape[1],
            param_names=linmod.exog_names, specification_name=specification_name,
            model_elapsed=model_elapsed, from_formula=True, formula=gmm_formula, data=data, moment_vals=None,
            moment_func_mean=moment_func_mean, moment_func_mean_jacobian=moment_func_mean_jacobian,
            index=index, model_type=GMM_IV_LINEAR
        )

        model.exog = linmod.exog
        model.instruments = linmod.instruments
        model.endog = linmod.endog
        model.weights = linmod.weights

        return model

    @staticmethod
    def gmm_iv_linear(formula, data, specification_name=None, W=None, start_params=None, method=DEFAULT_GMM_METHOD,
                      use_t=DEFAULT_GMM_USE_T, test_level=DEFAULT_GMM_TEST_LEVEL, max_iter=DEFAULT_GMM_MAX_ITER,
                      xtol=DEFAULT_GMM_X_TOL, ftol=DEFAULT_GMM_F_TOL, gtol=DEFAULT_GMM_G_TOL, Delta=DEFAULT_GMM_DELTA,
                      debug=False, _time=None, compute_cov=True, cov_type=DEFAULT_GMM_COV_TYPE, cov_kwds=None,
                      iterative_gmm_max_iter=DEFAULT_ITERATIVE_GMM_MAX_ITER, iterative_gmm_x_tol=DEFAULT_ITERATIVE_GMM_X_TOL,
                      do_2sls=False, index=None):
        """Fit linear instrumental-variable GMM from a formula.

        Args:
            formula: Linear IV formula such as ``'y ~ x | z'``.
            data: DataFrame-like or dict data source.
            specification_name: Optional result label.
            W: Optional initial weighting matrix.
            start_params: Optional starting coefficient vector.
            method: GMM weighting update method.
            use_t: Whether inference uses t distributions.
            test_level: Test size for confidence intervals and p-values.
            max_iter: Maximum trust-region iterations for each inner solve.
            xtol: Parameter-step convergence tolerance.
            ftol: Moment/objective_function_ convergence tolerance.
            gtol: Gradient convergence tolerance.
            Delta: Initial trust-region radius.
            debug: Whether to print fitting progress.
            _time: Optional externally supplied start time.
            compute_cov: Whether to compute covariance or bootstrap inference.
            cov_type: Covariance type.
            cov_kwds: Optional covariance keyword arguments.
            iterative_gmm_max_iter: Maximum outer weighting updates for
                iterative GMM.
            iterative_gmm_x_tol: Parameter-change tolerance for iterative GMM.
            do_2sls: If ``True``, use the 2SLS weighting matrix and one-step
                fitting.
            index: Optional row subset.

        Returns:
            Fitted ``GMMRegressionResults``.

        Examples
        --------
        Two-stage least squares as a one-step IV-GMM with the
        ``Z'Z``-weighting matrix:

        >>> import numpy as np, pandas as pd
        >>> from kanly.api import gmm_iv_linear
        >>> rng = np.random.default_rng(0)
        >>> n = 1_000
        >>> z1 = rng.normal(size=n); z2 = rng.normal(size=n)
        >>> x  = 0.5*z1 + 0.5*z2 + rng.normal(size=n)
        >>> y  = 1.0 + 2.0*x + rng.normal(size=n)
        >>> df = pd.DataFrame({'y': y, 'x': x, 'z1': z1, 'z2': z2})
        >>> fit_2sls = gmm_iv_linear('y ~ x | z1 + z2', df,
        ...                          do_2sls=True)            # doctest: +SKIP

        Over-identified two-step efficient GMM (default ``method='TWO_STEP'``):

        >>> fit_two_step = gmm_iv_linear('y ~ x | z1 + z2', df,
        ...                              method='TWO_STEP')   # doctest: +SKIP
        """

        model = SparseGeneralizedMethodOfMomentsModel.build_linear_iv_gmm_model_from_formula(
            formula, data, specification_name=specification_name, index=index)

        if do_2sls:
            if W is not None:
                raise Exception("Cannot specify `do_2sls=True` and a value for `W`")
            # The 2SLS weighting matrix is the inverse of the instrument
            # second moment, optionally with variance weights.
            if model.weights is not None:
                W = model.instruments.transpose().dot(diags(model.weights)).dot(model.instruments).toarray()
            else:
                W = model.instruments.transpose().dot(model.instruments).toarray()
            W = get_matrix_inverse_internal(W / model.nobs)
            method = ONE_STEP

        fit = model.fit(
            W=W, method=method, max_iter=max_iter, xtol=xtol, ftol=ftol, gtol=gtol, Delta=Delta, debug=debug,
            iterative_gmm_max_iter=iterative_gmm_max_iter, iterative_gmm_x_tol=iterative_gmm_x_tol, use_t=use_t,
            test_level=test_level, cov_type=cov_type, cov_kwds=cov_kwds, start_params=start_params,
            compute_cov=compute_cov)
        return fit

    @staticmethod
    def build_nonlinear_iv_gmm_model_from_formula(resid_formula, instrument_formula, data, specification_name=None,
                                                  debug=False, index=None, do_njit=True):
        """Build nonlinear IV-GMM moments from residual and instrument formulas.

        Args:
            resid_formula: Nonlinear residual formula understood by the NLLS
                parser.
            instrument_formula: RHS-style formula for instruments.
            data: DataFrame-like or dict data source.
            specification_name: Optional result label.
            debug: Whether to print parsing progress.
            index: Optional row subset.
            do_njit: Whether generated formula functions should use Numba.

        Returns:
            ``SparseGeneralizedMethodOfMomentsModel`` with moments
            ``E[z_i * u_i(theta)] = 0``.
        """
        _t = time.time()

        instrument_formula = '1 ~ ' + instrument_formula
        result = parse_formula(instrument_formula, debug=debug)
        instrument_var_names = result[EXOG_KEY]
        instruments = SparseDataGetter._sparse_dmatrix_internal(instrument_var_names, data, index=index)
        instruments_mat = instruments.values
        nlls_model = SparseNonlinearLeastSquaresModel.build_model_from_formula(
            resid_formula, data, debug=debug, index=index, do_njit=do_njit)
        resid_func = nlls_model.residual_function_callable
        nobs = instruments_mat.shape[0]

        def moment_func_mean(theta, weights=None):
            """Evaluate average nonlinear IV moments.

            Args:
                theta: Nonlinear parameter vector.
                weights: Optional bootstrap weights.

            Returns:
                Average instrument-residual moments.
            """
            u = resid_func(theta)
            if weights is not None:
                u *= weights
            if nlls_model.is_weighted:
                u *= nlls_model.weights
            u = csc_matrix(u).reshape((-1, 1))
            g = instruments_mat.transpose().dot(u)
            return g.toarray().flatten() / nobs

        def moment_func_mean_jacobian(theta, weights=None):
            """Evaluate the nonlinear IV moment Jacobian.

            Args:
                theta: Nonlinear parameter vector.
                weights: Optional bootstrap weights.

            Returns:
                Dense Jacobian matrix of average moments.
            """
            jac = resid_func.jacobian(theta)
            I = instruments_mat
            if weights is not None:
                I = csc_matrix_by_column_array_broadcast(I, weights)
            if nlls_model.is_weighted:
                I = csc_matrix_by_column_array_broadcast(I, nlls_model.weights)
            if not isspmatrix(jac):
                jac = csc_matrix(jac)
            frozen_jacobian = (I.transpose().dot(jac).toarray()) / nobs
            return frozen_jacobian

        def moment_func_obs(theta, weights=None):
            """Evaluate observation-level nonlinear IV moments.

            Args:
                theta: Nonlinear parameter vector.
                weights: Optional bootstrap weights.

            Returns:
                Sparse matrix with rows ``z_i * u_i(theta)``.
            """
            u = resid_func(theta)
            if weights is not None:
                u *= weights
            if nlls_model.is_weighted:
                u *= nlls_model.weights
            mmo = csc_matrix_by_column_array_broadcast(instruments_mat, u)
            return mmo

        model_elapsed = time.time() - _t

        gmm_formula = [(f'({resid_formula}) * [{z}]',) for z in instruments.column_names]

        return SparseGeneralizedMethodOfMomentsModel(
            moment_func_obs, nobs, instruments_mat.shape[1], num_params=resid_func.num_params,
            param_names=nlls_model.param_names, specification_name=specification_name,
            model_elapsed=model_elapsed, from_formula=True, formula=gmm_formula, data=data, moment_vals=None,
            moment_func_mean=moment_func_mean, moment_func_mean_jacobian=moment_func_mean_jacobian,
            index=index, model_type=GMM_IV_NONLINEAR
        )

    @staticmethod
    def gmm_iv_nonlinear(resid_formula, instrument_formula, data, specification_name=None, W=None, start_params=None,
                         method=DEFAULT_GMM_METHOD, use_t=DEFAULT_GMM_USE_T, test_level=DEFAULT_GMM_TEST_LEVEL,
                         max_iter=DEFAULT_GMM_MAX_ITER, xtol=DEFAULT_GMM_X_TOL, ftol=DEFAULT_GMM_F_TOL,
                         gtol=DEFAULT_GMM_G_TOL, Delta=DEFAULT_GMM_DELTA, debug=False, _time=None, compute_cov=True,
                         cov_type=DEFAULT_GMM_COV_TYPE, cov_kwds=None, index=None,
                         iterative_gmm_max_iter=DEFAULT_ITERATIVE_GMM_MAX_ITER, do_njit=True,
                         iterative_gmm_x_tol=DEFAULT_ITERATIVE_GMM_X_TOL):
        """Fit nonlinear instrumental-variable GMM.

        Args:
            resid_formula: Nonlinear residual formula understood by the NLLS
                parser.
            instrument_formula: RHS-style formula for instruments.
            data: DataFrame-like or dict data source.
            specification_name: Optional result label.
            W: Optional initial weighting matrix.
            start_params: Optional starting parameter vector.
            method: GMM weighting update method.
            use_t: Whether inference uses t distributions.
            test_level: Test size for confidence intervals and p-values.
            max_iter: Maximum trust-region iterations for each inner solve.
            xtol: Parameter-step convergence tolerance.
            ftol: Moment/objective_function_ convergence tolerance.
            gtol: Gradient convergence tolerance.
            Delta: Initial trust-region radius.
            debug: Whether to print fitting progress.
            _time: Optional externally supplied start time.
            compute_cov: Whether to compute covariance or bootstrap inference.
            cov_type: Covariance type.
            cov_kwds: Optional covariance keyword arguments.
            index: Optional row subset.
            iterative_gmm_max_iter: Maximum outer weighting updates for
                iterative GMM.
            do_njit: Whether generated formula functions should use Numba.
            iterative_gmm_x_tol: Parameter-change tolerance for iterative GMM.

        Returns:
            Fitted ``GMMRegressionResults``.

        Examples
        --------
        Nonlinear IV-GMM: identify an exponential model where ``x`` is
        endogenous and ``z1, z2`` are instruments. Residuals use NLLS-style
        ``{parameter}`` and ``[data]`` tokens; instruments use ``+``-style
        patsy sparse_terms:

        >>> import numpy as np, pandas as pd
        >>> from kanly.api import gmm_iv_nonlinear
        >>> rng = np.random.default_rng(0)
        >>> n = 1_000
        >>> z1 = rng.normal(size=n); z2 = rng.normal(size=n)
        >>> x  = 0.5*z1 + 0.5*z2 + rng.normal(size=n)
        >>> y  = 1.0 + 3.0*np.exp(-0.5*x) + 0.4*rng.normal(size=n)
        >>> df = pd.DataFrame({'y': y, 'x': x, 'z1': z1, 'z2': z2})
        >>> fit = gmm_iv_nonlinear(                          # doctest: +SKIP
        ...     '[y] - ({Intercept} + {beta} * exp({gamma} * [x]))',
        ...     'z1 + z2', df,
        ...     start_params={'Intercept': 0.0, 'beta': 1.0, 'gamma': -0.1})
        """

        model = SparseGeneralizedMethodOfMomentsModel.build_nonlinear_iv_gmm_model_from_formula(
            resid_formula, instrument_formula, data, specification_name=specification_name,
            index=index, debug=debug, do_njit=do_njit)

        fit = model.fit(
            W=W, method=method, max_iter=max_iter, xtol=xtol, ftol=ftol, gtol=gtol, Delta=Delta, debug=debug,
            iterative_gmm_max_iter=iterative_gmm_max_iter, iterative_gmm_x_tol=iterative_gmm_x_tol, use_t=use_t,
            test_level=test_level, cov_type=cov_type, cov_kwds=cov_kwds, start_params=start_params,
            compute_cov=compute_cov)

        return fit

    @staticmethod
    def build_mle_gmm_model_from_formula(formula_ll, data, is_log_ll=False, specification_name=None, index=None,
                                         debug=False, do_njit=True):
        """Build an MLE-as-GMM model from a likelihood formula.

        Args:
            formula_ll: Likelihood or log-likelihood formula. Parameters are
                expressed with nonlinear formula parameter syntax.
            data: DataFrame-like or dict data source.
            is_log_ll: Whether ``formula_ll`` is already a log-likelihood
                contribution. If ``False``, ``np.log(...)`` is wrapped around it.
            specification_name: Optional result label.
            index: Optional row subset.
            debug: Whether to print parsing progress.
            do_njit: Whether generated formula functions should use Numba.

        Returns:
            ``SparseGeneralizedMethodOfMomentsModel`` whose moments are the
            likelihood score equations.
        """

        _t = time.time()

        if is_log_ll is False:
            formula_ll = f'np.log({formula_ll})'

        func = build_prediction_function_from_formula(
            formula_ll, data, index=index, debug=debug, do_njit=do_njit)[0]

        # For MLE, GMM moments are the score contributions d log L_i / d theta.
        gmm_formula = [f'(d/d{{{p}}}) * {formula_ll}' for p in func.param_names]

        model_elapsed = time.time() - _t

        return SparseGeneralizedMethodOfMomentsModel(
            func.jacobian, func.nobs, func.num_params,
            formula=gmm_formula, from_formula=True,
            num_params=func.num_params, param_names=func.param_names, specification_name=specification_name,
            model_elapsed=model_elapsed, data=data, moment_vals=None,
            index=index, model_type=GMM_MLE)

    @staticmethod
    def gmm_mle(formula_llf, data, is_log_llf=False, specification_name=None, index=None, W=None, start_params=None,
                method=DEFAULT_GMM_METHOD, use_t=DEFAULT_GMM_USE_T, test_level=DEFAULT_GMM_TEST_LEVEL,
                max_iter=DEFAULT_GMM_MAX_ITER, xtol=DEFAULT_GMM_X_TOL, ftol=DEFAULT_GMM_F_TOL,
                gtol=DEFAULT_GMM_G_TOL, Delta=DEFAULT_GMM_DELTA, debug=False, _time=None, compute_cov=True,
                cov_type=DEFAULT_GMM_COV_TYPE, cov_kwds=None, do_njit=True,
                iterative_gmm_max_iter=DEFAULT_ITERATIVE_GMM_MAX_ITER,
                iterative_gmm_x_tol=DEFAULT_ITERATIVE_GMM_X_TOL):
        """Fit MLE score equations using the GMM optimizer.

        Args:
            formula_llf: Likelihood or log-likelihood formula.
            data: DataFrame-like or dict data source.
            is_log_llf: Whether ``formula_llf`` is already a log-likelihood
                contribution. If ``False``, ``np.log(...)`` is wrapped around it.
            specification_name: Optional result label.
            index: Optional row subset.
            W: Optional initial weighting matrix.
            start_params: Optional starting parameter vector.
            method: GMM weighting update method.
            use_t: Whether inference uses t distributions.
            test_level: Test size for confidence intervals and p-values.
            max_iter: Maximum trust-region iterations for each inner solve.
            xtol: Parameter-step convergence tolerance.
            ftol: Moment/objective_function_ convergence tolerance.
            gtol: Gradient convergence tolerance.
            Delta: Initial trust-region radius.
            debug: Whether to print fitting progress.
            _time: Optional externally supplied start time.
            compute_cov: Whether to compute covariance or bootstrap inference.
            cov_type: Covariance type.
            cov_kwds: Optional covariance keyword arguments.
            do_njit: Whether generated formula functions should use Numba.
            iterative_gmm_max_iter: Maximum outer weighting updates for
                iterative GMM.
            iterative_gmm_x_tol: Parameter-change tolerance for iterative GMM.

        Returns:
            Fitted ``GMMRegressionResults``.

        Examples
        --------
        Estimate the parameters of a Normal by maximising its log-likelihood
        as a system of score equations:

        >>> import numpy as np, pandas as pd
        >>> from kanly.api import gmm_mle
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({'y': rng.normal(loc=2.0, scale=3.0, size=500)})
        >>> log_lik = '-0.5 * (([y] - {mu})**2 / {sigma2} + log({sigma2}))'
        >>> fit = gmm_mle(log_lik, df, is_log_llf=True,                # doctest: +SKIP
        ...               start_params={'mu': 0.0, 'sigma2': 1.0})
        """

        model = SparseGeneralizedMethodOfMomentsModel.build_mle_gmm_model_from_formula(
            formula_llf, data, is_log_ll=is_log_llf, specification_name=specification_name, index=index,
            do_njit=do_njit)

        return model.fit(
            W=W, method=method, max_iter=max_iter, xtol=xtol, ftol=ftol, gtol=gtol, Delta=Delta, debug=debug,
            iterative_gmm_max_iter=iterative_gmm_max_iter, iterative_gmm_x_tol=iterative_gmm_x_tol, use_t=use_t,
            test_level=test_level, cov_type=cov_type, cov_kwds=cov_kwds, start_params=start_params,
            compute_cov=compute_cov)

        model = SparseGeneralizedMethodOfMomentsModel.build_mle_gmm_model_from_formula(
            formula_llf, data, is_log_ll=is_log_llf, specification_name=specification_name, index=index,
            do_njit=do_njit)

        return model.fit(
            W=W, method=method, max_iter=max_iter, xtol=xtol, ftol=ftol, gtol=gtol, Delta=Delta, debug=debug,
            iterative_gmm_max_iter=iterative_gmm_max_iter, iterative_gmm_x_tol=iterative_gmm_x_tol, use_t=use_t,
            test_level=test_level, cov_type=cov_type, cov_kwds=cov_kwds, start_params=start_params,
            compute_cov=compute_cov)

