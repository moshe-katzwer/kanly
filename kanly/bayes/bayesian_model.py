"""Core Bayesian model container with priors, bounds transforms, and MCMC/MAP entry points."""

from __future__ import absolute_import, print_function

from collections.abc import Callable, Iterable
from inspect import signature

import numpy as np
import pandas as pd
from scipy.stats._distn_infrastructure import rv_frozen
from scipy.stats._multivariate import multi_rv_frozen

from kanly import __version__
from kanly.bayes.map.maximum_a_posteriori import map, mle, BayesianModelMaximizationResult
from kanly.bayes.mcmc.adaptive_metropolis.adaptive_metropolis_mcmc import amha as amha_internal
from kanly.bayes.mcmc.adaptive_metropolis.constants import \
    DEFAULT_AM_N_BURNIN, DEFAULT_AM_N_SAMPLES, \
    DEFAULT_AM_N_CHAINS, \
    DEFAULT_AM_THINNING, DEFAULT_AM_DRAW_SIZE, DEFAULT_AM_TARGET_ACCEPTANCE_RATE, DEFAULT_AM_MAX_PROCESSES, \
    DEFAULT_AM_PBAR_UPDATE_CADENCE, DEFAULT_AM_DO_ADAPTIVE, \
    DEFAULT_AM_DO_PARALLEL, DEFAULT_AM_USER_PROMPT_FOR_MORE_ITERS, \
    DEFAULT_AM_SCALER_ADJUST_RATE, DEFAULT_AM_SCALER_ADJUST_DENOM_POWER, \
    DEFAULT_AM_MIN_SCALER, DEFAULT_AM_MAX_SCALER, DEFAULT_AM_PROPOSAL_DF, \
    DEFAULT_AM_MAX_SUBCHAIN_DRAWS_BURNIN, DEFAULT_AM_MAX_SUBCHAIN_DRAWS_SAMPlE, \
    DEFAULT_AM_STEP_COV_INITIAL_SAMPLES, DEFAULT_AM_NORMALIZE_STEP_COV, \
    DEFAULT_AM_DO_DIFF_EVOLUTION_MC, DEFAULT_AM_DIFF_EVOLUTION_FRAC_BURNIN, \
    DEFAULT_AM_DIFF_EVOLUTION_MAX_DRAWS, DEFAULT_AM_DIFF_EVOLUTION_WEIGHT, DEFAULT_AM_STOP_ADAPTATION_AFTER_BURNIN, \
    DEFAULT_AM_DIFF_EVOLUTION_JUMP_CADENCE, DEFAULT_AM_SCALAR_JITTER_BOUNDS
from kanly.bayes.mcmc.coordinate_mala.constants import \
    DEFAULT_COORD_MALA_THINNING, DEFAULT_COORD_MALA_N_CHAINS, DEFAULT_COORD_MALA_DEBUG, \
    DEFAULT_COORD_MALA_DO_CAUCHY, DEFAULT_COORD_MALA_DO_MALA, DEFAULT_COORD_MALA_N_SAMPLES, \
    DEFAULT_COORD_MALA_TAU_ADJUST, DEFAULT_COORD_MALA_P_BAR_UPDATE_CADENCE, DEFAULT_COORD_MALA_TARGET_ACCEPTANCE_RATE, \
    DEFAULT_COORD_MALA_USER_PROMPT_FOR_MORE_ITERS, DEFAULT_COORD_MALA_DIFF_EVOLUTION_STEP_CADENCE, \
    DEFAULT_COORD_MALA_MAX_SUBCHAIN_DRAWS, \
    DEFAULT_COORD_MALA_DIFF_EVOLUTION_FRAC_BURNIN, DEFAULT_COORD_MALA_DIFF_EVOLUTION_MAX_DRAWS, \
    DEFAULT_COORD_MALA_FRAC_BURNIN
from kanly.bayes.mcmc.coordinate_mala.coordinate_mala_mcmc import mala as mala_internal
from kanly.bayes.mcmc.mcmc_results import MCMCResults
from kanly.bayes.parameter_transformations import get_transformation_vector_functions, bounds_2_transformations
from kanly.bayes.utils.rv_to_nopython_logpdf import convert_str_to_scipy_rv, get_nopython_logpdf
from kanly.optimize.bfgs_bounded_quasi_newton import bfgs_pqn
from kanly.parameter_collection import ParameterCollection
from kanly.stats.distributions.nopython_frozen_logpdf import \
    MULTIVARIATE_DISTRIBUTIONS as SUPPORTED_MULTIVARIATE_DISTRIBUTIONS
from kanly.utils.function_str_to_callable import get_callable_from_func_str
from kanly.utils.print_options import print_options

# from kanly.stats import IMPORT_STR
#exec(IMPORT_STR)
from kanly.stats.common import (
    jit, njit, numpy, np, np_linalg, scipy, sp, sp_linalg, sp_special, stats, logit, log_d_expit, expit, d_expit, sin,
    cos, tan, arcsin, arccos, arctan, cbrt, sqrt, log, log2, log10, exp, std_normal_cdf, normal_cdf, normal_pdf,
    normal_logpdf, log_normal_pdf, log_normal_logpdf, log_normal_cdf, logpdf_beta, logpdf_cauchy, logpdf_chi2,
    logpdf_expon, logpdf_f, logpdf_gamma, logpdf_genextreme, logpdf_halfcauchy, logpdf_halfnorm, logpdf_invgamma,
    logpdf_laplace, logpdf_logistic, logpdf_lognorm, logpdf_multivariate_normal, logpdf_multivariate_t, logpdf_norm,
    logpdf_pareto, logpdf_t, logpdf_truncnorm, logpdf_weibull_min, nopython_logpdf_beta, nopython_logpdf_cauchy,
    nopython_logpdf_chi2, nopython_logpdf_expon, nopython_logpdf_f, nopython_logpdf_gamma, nopython_logpdf_genextreme,
    nopython_logpdf_halfcauchy, nopython_logpdf_halfnorm, nopython_logpdf_invgamma, nopython_logpdf_laplace,
    nopython_logpdf_logistic, nopython_logpdf_lognorm, nopython_logpdf_multivariate_normal,
    nopython_logpdf_multivariate_t, nopython_logpdf_norm, nopython_logpdf_pareto, nopython_logpdf_t,
    nopython_logpdf_truncnorm, nopython_logpdf_weibull_min, nopython_pdf_beta, nopython_pdf_cauchy, nopython_pdf_chi2,
    nopython_pdf_expon, nopython_pdf_f, nopython_pdf_gamma, nopython_pdf_genextreme, nopython_pdf_halfcauchy,
    nopython_pdf_halfnorm, nopython_pdf_invgamma, nopython_pdf_laplace, nopython_pdf_logistic, nopython_pdf_lognorm,
    nopython_pdf_multivariate_normal, nopython_pdf_multivariate_t, nopython_pdf_norm, nopython_pdf_pareto,
    nopython_pdf_t, nopython_pdf_truncnorm, nopython_pdf_weibull_min, pdf_beta, pdf_cauchy, pdf_chi2, pdf_expon, pdf_f,
    pdf_gamma, pdf_genextreme, pdf_halfcauchy, pdf_halfnorm, pdf_invgamma, pdf_laplace, pdf_logistic, pdf_lognorm,
    pdf_multivariate_normal, pdf_multivariate_t, pdf_norm, pdf_pareto, pdf_t, pdf_truncnorm, pdf_weibull_min,
    __frozen_internal_logpdf_genextreme, __frozen_internal_logpdf_norm, __frozen_internal_logpdf_truncnorm,
    __frozen_internal_logpdf_beta, __frozen_internal_logpdf_cauchy, __frozen_internal_logpdf_laplace,
    __frozen_internal_logpdf_expon, __frozen_internal_logpdf_t, __frozen_internal_logpdf_gamma,
    __frozen_internal_logpdf_lognorm, __frozen_internal_logpdf_invgamma, __frozen_internal_logpdf_logistic,
    __frozen_internal_logpdf_chi2, __frozen_internal_logpdf_gennorm, __frozen_internal_logpdf_multivariate_normal,
    __frozen_internal_logpdf_multivariate_t, __frozen_internal_logpdf_halfnorm, __frozen_internal_logpdf_pareto,
    __frozen_internal_logpdf_halfcauchy, __frozen_internal_logpdf_loguniform, __frozen_internal_logpdf_f,
    __frozen_internal_logpdf_weibull_min, __frozen_internal_logpdf_dirichlet, __nopython_frozen_internal_logpdf_norm,
    __nopython_frozen_internal_logpdf_halfnorm, __nopython_frozen_internal_logpdf_beta,
    __nopython_frozen_internal_logpdf_cauchy, __nopython_frozen_internal_logpdf_laplace,
    __nopython_frozen_internal_logpdf_expon, __nopython_frozen_internal_logpdf_t,
    __nopython_frozen_internal_logpdf_multivariate_t, __nopython_frozen_internal_logpdf_gamma,
    __nopython_frozen_internal_logpdf_lognorm, __nopython_frozen_internal_logpdf_invgamma,
    __nopython_frozen_internal_logpdf_logistic, __nopython_frozen_internal_logpdf_chi2,
    __nopython_frozen_internal_logpdf_gennorm, __nopython_frozen_internal_logpdf_multivariate_normal,
    __nopython_frozen_internal_logpdf_truncnorm, __nopython_frozen_internal_logpdf_pareto,
    __nopython_frozen_internal_logpdf_halfcauchy, __nopython_frozen_internal_logpdf_loguniform,
    __nopython_frozen_internal_logpdf_genextreme, __nopython_frozen_internal_logpdf_f,
    __nopython_frozen_internal_logpdf_weibull_min, __nopython_frozen_internal_logpdf_dirichlet,
    get_frozen_logpdf_pareto, get_frozen_logpdf_norm, get_frozen_logpdf_truncnorm, get_frozen_logpdf_halfnorm,
    get_frozen_logpdf_beta, get_frozen_logpdf_cauchy, get_frozen_logpdf_laplace, get_frozen_logpdf_expon,
    get_frozen_logpdf_t, get_frozen_logpdf_gamma, get_frozen_logpdf_invgamma, get_frozen_logpdf_lognorm,
    get_frozen_logpdf_logistic, get_frozen_logpdf_gennorm, get_frozen_logpdf_chi2,
    get_frozen_logpdf_multivariate_normal, get_frozen_logpdf_halfcauchy, get_frozen_logpdf_multivariate_t,
    get_frozen_logpdf_uniform, get_frozen_logpdf_flat, get_frozen_logpdf_loguniform, get_frozen_logpdf_genextreme,
    get_frozen_logpdf_f, get_frozen_logpdf_weibull_min, get_frozen_logpdf_dirichlet, )


class BayesianModel(ParameterCollection):
    """Base container for Bayesian models: likelihood, priors, bounds, and sampling.

    ``BayesianModel`` is the central class for organizing models you want to
    optimize (MAP/MLE) or explore with MCMC.  Formula-based wrappers
    (:class:`~kanly.bayes.bayesian_regression_model.BayesianLinearModel`,
    :class:`~kanly.bayes.bayesian_regression_model.BayesianGeneralizedLinearModel`,
    :class:`~kanly.bayes.bayesian_regression_model.BayesianNonlinearLeastSquaresModel`)
    and code-block :class:`~kanly.bayes.data_model.DataModel` objects all
    delegate to this class after building a log-likelihood.

    At construction you supply:

    * A **log-likelihood** callable ``log_likelihood_function(params)``.
    * Optional **priors** (see :meth:`set_priors`).
    * Optional **bounds** and whether to use smooth reparameterization
      (``do_bounded_transform``; see :meth:`set_transformations`).

    The model then exposes:

    * ``log_posterior`` — log-likelihood + log-prior + bound handling on the
      **original** parameter scale.
    * ``log_posterior_transformed`` — same density expressed on an **unbounded**
      internal scale when bounds use reparameterization.
    * ``log_posterior_jacobian_adjustment`` — log-Jacobian so MCMC on
      transformed coordinates targets the correct posterior after mapping back.

    For original-scale parameters ``x`` and unbounded internal coordinates ``u``:

    .. math::

        \\log \\pi(x) = \\log \\pi_{\\mathrm{trans}}(\\mathrm{inv\\_transform}(x))

    .. math::

        \\log \\pi_{\\mathrm{trans}}(u) = \\log \\pi(\\mathrm{transform}(u))

    MCMC samplers typically evaluate
    ``log_posterior_transformed + log_posterior_jacobian_adjustment`` on ``u``,
    then store draws on the original scale via ``transform``.

    If there are no bounds, or ``do_bounded_transform=False``, sampling uses
    hard log-barriers at bounds instead of reparameterization.  Regression
    wrappers still impose ``__sigma2 > 0`` via bounds unless you override them.

    Parameters
    ----------
    See :meth:`__init__` for constructor arguments.

    Methods
    -------
    sample, amha, mala
        MCMC; ``sample`` optionally runs coordinate MALA warmup then adaptive
        Metropolis–Hastings with differential-evolution proposals.
    map, mle
        Posterior mode and maximum likelihood via bounded BFGS.

    Examples
    --------
    Build a 2-parameter Bayesian model by hand and run adaptive Metropolis:

    >>> import numpy as np
    >>> from kanly.api import BayesianModel
    >>> rng = np.random.default_rng(0)
    >>> y = rng.normal(loc=2.0, scale=3.0, size=200)
    >>> def log_lik(theta):                                # theta = (mu, log_sigma)
    ...     mu, log_sigma = theta
    ...     sigma = np.exp(log_sigma)
    ...     return -0.5 * np.sum((y - mu)**2 / sigma**2) \\
    ...            - len(y) * log_sigma
    >>> model = BayesianModel(log_lik,                     # doctest: +SKIP
    ...                       param_names=['mu', 'log_sigma'],
    ...                       priors={'mu': 'norm(0, 10)',
    ...                               'log_sigma': 'norm(0, 1)'})
    >>> fit = model.sample([0.0, 0.0], n_samples=5_000,    # doctest: +SKIP
    ...                    n_burnin=1_000, n_chains=4)
    >>> print(fit)                                          # doctest: +SKIP

    For most use cases the higher-level :class:`DataModel` builder (which
    parses `data_string` + `model_string` blocks) is more convenient — see
    :meth:`DataModel.build_data_model` and :meth:`DataModel.to_bayesian_model`.

    Aliases on ``kanly.api``: ``bmodel``.
    """

    def __init__(self, log_likelihood_function, bounds=None, do_bounded_transform=True, priors=None,
                 param_names=None, num_params=None, specification_name=None, other_info=None, debug=False,
                 nopython=False, parameter_groupings=None
                 ):
        """Construct a Bayesian model from a log-density and optional priors/bounds.

        Parameters
        ----------
        log_likelihood_function : callable
            Function ``params -> float`` returning the log-likelihood (or, if
            ``priors`` is empty, the full log-posterior).  ``params`` is a
            1-D array in ``param_names`` order, or values are passed through
            :meth:`~kanly.parameter_collection.ParameterCollection.dict_2_array`
            when you call samplers with a dict.
        bounds : dict, optional
            Mapping ``{parameter_name: (lower, upper)}``.  Use ``np.inf`` for
            one-sided limits.  Merged with support limits inferred from scipy
            priors when applicable.  Regression wrappers add ``__sigma2`` bounds
            automatically.
        do_bounded_transform : bool, default True
            If ``True`` and ``bounds`` is non-empty, bounded parameters are
            reparameterized to an unbounded space for MCMC/optimization, with a
            Jacobian correction on the log-density.  If ``False``, violations
            of bounds set the log-density to ``-inf`` (hard barrier).
        priors : dict, optional
            Prior specifications keyed by:

            * a **parameter name** (univariate prior on one coordinate),
            * a **tuple of names** (callable prior on that subset),
            * a **parameter-group name** from ``parameter_groupings`` (e.g.
              multivariate normal on a block of coefficients),
            * the empty string ``''`` for a **joint prior** on the full vector
              (string expression with ``{param}`` placeholders or a callable
              taking the full ``params`` array).

            Values may be scipy frozen distributions, strings like
            ``'norm(0, 1)'``, callables, or lists of priors to stack on the
            same key.  ``None`` or ``{}`` means flat priors (likelihood only).
        param_names : list of str, optional
            Names for each element of ``params``.  Required unless
            ``num_params`` is given.
        num_params : int, optional
            If ``param_names`` is omitted, parameters are named
            ``x0``, ``x1``, ...  Required unless ``param_names`` is given.
        specification_name : str, optional
            Label printed in summaries and stored on ``MCMCResults``.
        other_info : dict, optional
            Arbitrary metadata attached to the model (e.g. formula strings).
        debug : bool, default False
            Verbose logging while parsing priors and transformations.
        nopython : bool, default False
            When ``True``, attempt Numba-friendly log-pdf callables for scipy
            priors where supported.
        parameter_groupings : dict, optional
            Maps group names to lists of parameter names for block / multivariate
            priors.

        Raises
        ------
        Exception
            If neither ``param_names`` nor ``num_params`` is supplied.

        Attributes
        ----------
        log_likelihood_function
            User-supplied log-likelihood callable.
        log_posterior
            Callable on the original parameter scale (after :meth:`set_priors`
            and :meth:`set_transformations`).
        log_posterior_transformed
            Callable used by MCMC in transformed coordinates when applicable.
        transform, inv_transform
            Maps between bounded original space and unbounded sampling space.
        log_posterior_jacobian_adjustment
            Log-Jacobian adjustment for transformed-space MCMC.
        bounds
            Effective bounds dict after merging user input and prior support.
        priors, priors_orig
            Parsed prior specifications.

        See Also
        --------
        kanly.bayes.data_model.DataModel.to_bayesian_model
        kanly.bayes.bayesian_regression_model.BayesianLinearModel
        """

        if param_names is None:
            if num_params is None:
                raise Exception("Must supply either param names or number of params!")
            assert isinstance(num_params, int) and num_params > 0
            param_names = [f'x{d}' for d in range(num_params)]

        super().__init__(param_names, parameter_groupings)

        self.specification_name = specification_name
        self.other_info = other_info

        self.log_likelihood_function = log_likelihood_function

        # Build parameter transformations to respect bounds
        self.set_transformations(bounds, do_bounded_transform, debug=debug)

        # Build log-pdf from priors
        self.set_priors(priors, debug=debug, nopython=nopython)

        self.__version__ = __version__

    def _build_log_likelihood_transformed(self):
        """Build ``log_likelihood_function_transformed`` for MCMC/optimization.

        When ``do_bounded_transform`` is active, likelihood evaluation maps
        unbounded ``params`` through ``transform`` before applying the user
        likelihood and hard bound indicator.  Otherwise the transformed
        callable equals the bounded likelihood on the original scale.

        Sets attributes ``_log_likelihood_bounded`` and
        ``log_likelihood_function_transformed``.
        """
        def _log_likelihood(params):
            """Evaluate the log-likelihood including any hard-bound log-barrier.

            Adds ``lp_bounds_log_pdf(params)`` (which returns ``-inf`` when
            ``params`` violates any hard bound, and 0 otherwise) to the
            user-supplied log-likelihood function.

            Args:
                params: Parameter vector in the original (bounded) space.

            Returns:
                Scalar log-likelihood value (``-inf`` if bounds are violated).
            """
            return self.log_likelihood_function(params) + self.lp_bounds_log_pdf(params)

        self._log_likelihood_bounded = _log_likelihood
        if self.transformations is not None and len(self.transformations):

            def _log_likelihood_function_transformed(params):
                """Evaluate the log-likelihood at unbounded-space parameters.

                Maps ``params`` from the unbounded sampling space back to the
                original (bounded) space via ``self.transform`` before
                evaluating the bounded log-likelihood.

                Args:
                    params: Parameter vector in the unbounded sampling space.

                Returns:
                    Scalar log-likelihood value in the original space.
                """
                params_transformed = self.transform(params)
                return _log_likelihood(params_transformed)

        else:
            _log_likelihood_function_transformed = self._log_likelihood_bounded

        self.log_likelihood_function_transformed = _log_likelihood_function_transformed

    def _build_log_posterior(self, debug=False):
        """Rebuild ``log_posterior`` and ``log_posterior_transformed`` after prior/bound changes.

        Composes log-likelihood, summed log-priors, and bound indicators on the
        original scale, then wires the transformed-space variant used by MCMC.

        Parameters
        ----------
        debug : bool, default False
            Unused; retained for API compatibility with :meth:`set_priors`.
        """
        def _log_posterior(params):
            """Evaluate the log-posterior in the original (bounded) parameter space.

            Sums log-likelihood, log-prior, and the hard-bound log-barrier.

            Args:
                params: Parameter vector in the original (possibly bounded) space.

            Returns:
                Scalar log-posterior value (``-inf`` if bounds are violated).
            """
            return self.log_likelihood_function(params) + self.log_pdf_prior(params) + self.lp_bounds_log_pdf(params)

        self.log_posterior = _log_posterior
        self.log_posterior.__doc__ = \
            """The log posterior function parametrized on the original parameter space,
            which is possibly bounded.
            
            `self.log_posterior(x) = self.log_posterior_transformed(self.inv_transform(x))`
            """

        if self.transformations is not None and len(self.transformations):

            def _log_posterior_transformed(params):
                """Evaluate the log-posterior in the unbounded sampling space.

                Maps ``params`` to the original space via ``self.transform``
                before evaluating ``_log_posterior``.  Does not include the
                Jacobian adjustment; that is handled by
                ``self.log_posterior_jacobian_adjustment``.

                Args:
                    params: Parameter vector in the unbounded sampling space.

                Returns:
                    Scalar log-posterior value (``-inf`` if bounds are violated).
                """
                params_transformed = self.transform(params)
                return _log_posterior(params_transformed)

        else:
            _log_posterior_transformed = _log_posterior

        self.log_posterior_transformed = _log_posterior_transformed
        self.log_posterior_transformed.__doc__ = \
            """The log posterior function parametrized on the unbounded parameter space,
            a transformation of the original possibly bounded parameter space.  Requires the Jacobian adjustment 
            to be the correct density.
            
            `self.log_posterior_transformed(x) = self.log_posterior(self.transform(x))`
            """

    def set_transformations(self, bounds, do_bounded_transform, debug=False):
        """Configure bounds and reparameterization for MCMC/optimization.

        Parameters
        ----------
        bounds : dict or None
            ``{name: (lower, upper)}`` with finite or infinite limits.  ``None``
            is treated as no explicit bounds.
        do_bounded_transform : bool
            If ``True`` and there are bounds, each bounded coordinate gets a
            smooth bijection to ``(-inf, inf)`` and ``lp_bounds_log_pdf`` is
            identically zero (bounds enforced by the map).  If ``False``, bounds
            are enforced by returning ``-inf`` outside the interval.
        debug : bool, default False
            Print transformation source when ``True``.

        Notes
        -----
        Also sets ``self.transform``, ``self.inv_transform``,
        ``self.log_posterior_jacobian_adjustment``, and refreshes likelihood/
        posterior callables via :meth:`_build_log_likelihood_transformed` and
        :meth:`_build_log_posterior`.
        """

        if bounds is None:
            bounds = dict()

        assert np.all([len(v) == 2 for v in bounds.values()])
        self.bounds = {k: tuple(v) for k, v in bounds.items()}
        self.do_bounded_transform = do_bounded_transform

        if do_bounded_transform and len(bounds):
            # Enforce bounds through reparameterization (preferred for MCMC stability).
            self.lp_bounds_log_pdf = lambda x: 0.0
            transformations = bounds_2_transformations(bounds)
        else:
            lb, ub = np.full(self.num_params, -np.inf, ), np.full(self.num_params, np.inf, )
            for i, k in enumerate(self.param_names):
                if k in bounds:
                    lb[i], ub[i] = bounds[k]

            def _lp_bounds_log_pdf_internal(x):
                """Return 0 if all parameters are within bounds, else -inf.

                Acts as a hard-boundary log-indicator function: any parameter
                outside its ``[lb, ub]`` interval sends the log-posterior to
                ``-inf``, causing automatic rejection in any MCMC or optimizer
                step.

                Args:
                    x: Parameter vector in the original (bounded) space.

                Returns:
                    0.0 if all bounds are satisfied, ``-np.inf`` otherwise.
                """
                if np.any((x < lb) | (x > ub)):
                    return -np.inf
                else:
                    return 0

            self.lp_bounds_log_pdf = _lp_bounds_log_pdf_internal

            transformations = dict()

        _transform, _inv_transform, _log_posterior_jacobian_adjustment, _trans_func_str, _inv_trans_func_str, _lp_adj_func_str \
            = get_transformation_vector_functions(transformations, self.param_names)

        self.transformations = transformations
        self.transform = _transform
        self.inv_transform = _inv_transform
        self.log_posterior_jacobian_adjustment = _log_posterior_jacobian_adjustment

        self.transform.__doc__ = f"Converts parameters back to 'original' bounded parameter space of the model\n\n{_trans_func_str}"
        self.inv_transform.__doc__ = f"Converts parameters from 'original' bounded of the model to the transformed unbounded space\n\n{_inv_trans_func_str}"
        self.log_posterior_jacobian_adjustment.__doc__ = """"
        The jacobian adjustment to the log-posterior to ensure it is the correct density function.

        `self.log_posterior_transformed + self.log_posterior_jacobian_adjustment` should represent the same
        density as `self.log_posterior`\n\n%s.
        """ % _lp_adj_func_str

        self._build_log_likelihood_transformed()
        self._build_log_posterior()

    def set_priors(self, priors, debug=False, nopython=False):
        """Parse priors and rebuild the log-posterior.

        Parameters
        ----------
        priors : dict or None
            Prior specifications (see :class:`BayesianModel` constructor).
            Multiple priors per key may be passed as a list; they sum in log
            space.  String entries like ``'norm(0, 1)'`` are converted to scipy
            frozen distributions when possible.  scipy priors may tighten
            ``self.bounds`` to the distribution support and trigger a recursive
            call to :meth:`set_transformations`.
        debug : bool, default False
            Print parsing progress and conversions.
        nopython : bool, default False
            Prefer Numba-compiled log-pdf helpers for scipy priors when available.

        Notes
        -----
        Stores callables in ``self.priors_callables`` and
        ``self.prior_callables_dict``, then calls :meth:`_build_log_posterior`.
        """

        if priors is None:
            priors = dict()

        if debug:
            print("Setting priors...", end="")

        self.priors_orig = priors.copy() if priors is not None else None
        self.priors = {k: v for k, v in priors.items()
                       if not ((v is None) or (hasattr(v, 'len') and len(v) == 0))}
        self.prior_callables_dict = dict()
        self.priors_callables = []

        if len(priors) == 0:
            self._build_log_posterior(debug=debug)
            if debug:
                print("No priors!")
            return

        # Try to convert any prior that is a string to a scipy random variable
        # modifies `self.priors` inplace
        self.convert_str_prior_2_scipy_rv(self.priors,
                                          self.param_names + list(self.parameter_groupings.keys()),
                                          debug=debug)

        for prior_key, priors_k in self.priors.items():

            if prior_key not in self.prior_callables_dict:
                self.prior_callables_dict[prior_key] = []

            # EACH PRIOR KEY MAPS TO A LIST OF PRIORS (THAT STACK) FOR THAT KEY
            if not isinstance(priors_k, Iterable) or isinstance(priors_k, str):
                priors_k = [priors_k]

            for p_sub in priors_k:

                # If they prior maps to a certain parameter, do this
                if prior_key in self.param_names:

                    # if it is a scipy random variable, try to get the fast logpdf
                    # Otherwise, use the logpdf from the object
                    if isinstance(p_sub, rv_frozen):

                        try:
                            p_sub_func = get_nopython_logpdf(p_sub, nopython=nopython, logpdf_only=True)
                        except:
                            p_sub_func = p_sub.logpdf

                        p_sub_func = self.get_prior_function_index_callable(p_sub_func, self.param_2_idx[prior_key])
                        # Append to list of callables
                        self.prior_callables_dict[prior_key].append(p_sub_func)
                        self.priors_callables.append(p_sub_func)

                        bnds = p_sub.support()  # get support
                        set_bounds = False
                        if -np.inf < bnds[0] or bnds[1] < np.inf:
                            if prior_key not in self.bounds:
                                self.bounds[prior_key] = bnds
                                set_bounds = True
                            else:
                                if bnds[0] > self.bounds[prior_key][0] or bnds[1] < self.bounds[prior_key][1]:
                                    self.bounds[prior_key] = bnds
                                    set_bounds = True
                        if set_bounds:
                            if debug:
                                print(f"Setting bounds for '{prior_key}' as `{str(bnds)}`")
                            self.set_transformations(self.bounds, self.do_bounded_transform, debug=debug)

                    elif isinstance(p_sub, str):
                        p_sub_func = get_callable_from_func_str(p_sub, self.param_names, debug=debug)
                        self.priors_callables.append(p_sub_func)
                        self.prior_callables_dict[prior_key].append(p_sub_func)
                    elif isinstance(p_sub, Callable):
                        p_sub_func = p_sub
                        self.priors_callables.append(
                            self.get_prior_function_index_callable(p_sub_func, self.param_2_idx[prior_key]))
                        self.prior_callables_dict[prior_key].append(
                            self.get_prior_function_index_callable(p_sub_func, self.param_2_idx[prior_key]))
                    else:
                        raise Exception(f'"{p_sub}" (type "{type(p_sub)}") for prior on "{prior_key}" did not work!')

                # check if it is in a parameter grouping
                elif prior_key in self.parameter_groupings:
                    if isinstance(p_sub, (rv_frozen, multi_rv_frozen)):

                        try:
                            p_sub_func = get_nopython_logpdf(p_sub, nopython=nopython, logpdf_only=True)
                        except:
                            p_sub_func = p_sub.logpdf

                        p_sub_func = self.get_prior_function_index_callable(
                            p_sub_func, [self.param_2_idx[j] for j in self.parameter_groupings[prior_key]])

                        # Append to list of callables
                        self.prior_callables_dict[prior_key].append(p_sub_func)
                        self.priors_callables.append(p_sub_func)

                    elif isinstance(p_sub, Callable):
                        p_sub_func = self.get_prior_function_indices_list_callable(
                            p_sub, [self.param_2_idx[j] for j in self.parameter_groupings[prior_key]])
                        self.prior_callables_dict[prior_key].append(p_sub_func)
                        self.priors_callables.append(p_sub_func)

                    else:
                        raise Exception(f"Prior on a parameter grouping like '{prior_key}' must be a callable, "
                                        f"or use scipy.stats rv syntax for multivariate distributions.\nCurrently "
                                        f"supported are {SUPPORTED_MULTIVARIATE_DISTRIBUTIONS}!")

                # next check if it is on a tuple of parameter names
                # currently must be a callable!
                elif isinstance(prior_key, tuple):
                    if not set(prior_key) <= set(self.param_names):
                        raise Exception("If supplying tuple prior key, must be subset of parameters!")
                    if isinstance(p_sub, Callable) and len(signature(p_sub).parameters) == len(prior_key):
                        p_sub_func = self.get_prior_function_indices_list_callable(p_sub, [self.param_2_idx[j] for j in
                                                                                           prior_key])
                        self.prior_callables_dict[prior_key].append(p_sub_func)
                        self.priors_callables.append(p_sub_func)
                    else:
                        raise Exception(f"If specifying a prior on a subset of parameters, as in "
                                        f"{prior_key}, must have a callable with {len(prior_key)} "
                                        f"args, not whatever it is you supplied.")

                # otherwise, we treat it as a vector prior on ALL parameters
                # so should be either a callable that accepts the entire param vector
                # or a string that explicitly references parameters
                elif prior_key == '':

                    if isinstance(p_sub, str):
                        p_sub_func2 = get_callable_from_func_str(p_sub, self.param_names, debug=debug)
                    elif isinstance(p_sub, Callable):
                        p_sub_func2 = p_sub
                    else:
                        raise Exception(f'prior for `"{prior_key}" must be a string or callable!')

                    self.priors_callables.append(p_sub_func2)
                    self.prior_callables_dict[prior_key].append(p_sub_func2)

                else:
                    raise Exception(f"Valid prior keys are either (a) a parameter name string, "
                                    f"(b) a tuple of strings of parameter names, or "
                                    f"(c) the empty string `''` (to signify a prior on all parameters), "
                                    f"you supplied {prior_key}")

        if debug:
            print(f"done!\n\tPriors on {set(self.prior_callables_dict.keys())} params.")

        self._build_log_posterior(debug=debug)

    def log_pdf_prior(self, params):
        """Sum of log-prior contributions at ``params`` on the original scale.

        Parameters
        ----------
        params : array_like
            Parameter vector aligned with :attr:`param_names`.

        Returns
        -------
        float
            Total log-prior density (0.0 if no priors were registered).
        """
        if len(self.priors_callables):
            return np.sum([p(params) for p in self.priors_callables])
        else:
            return 0.0

    @staticmethod
    def convert_str_prior_2_scipy_rv(priors, param_names, debug=False):
        """Convert string-valued prior entries to scipy frozen RV objects where possible.

        Args:
            priors: Prior mapping to update in-place.
            param_names: Valid parameter names used to guard conversion.
            debug: Debug mode flag.
        """
        for prior_key, rvs in priors.items():
            if isinstance(rvs, str) or not isinstance(rvs, Iterable):
                rvs = [rvs]
            rvs_new = []
            for rv in rvs:
                if isinstance(rv, str):
                    if prior_key in param_names:
                        try:
                            rv2 = convert_str_to_scipy_rv(rv)
                            priors[prior_key] = rv2
                            if debug:
                                print(f"Converted '{rv}' into scipy random variable! {rv2}")
                            rv = rv2
                        except Exception as e:
                            if debug:
                                print(f"Could not convert str '{rv}' to scipy random variable!")

                rvs_new.append(rv)

            if len(rvs_new) == 1:
                rvs_new = rvs_new[0]

            priors[prior_key] = rvs_new

    @staticmethod
    def get_prior_function_index_callable(fun, idx):
        """Converts a univariate logpdf prior on the parameter at
        index `idx` to something on the full vector

        Args:
            fun: Univariate prior callable.
            idx: Parameter index (or index list/slice) to evaluate.
        """

        def temp(x):
            """Evaluate the univariate prior on the single coordinate ``x[idx]``.

            Args:
                x: Full parameter vector (array-like).

            Returns:
                Scalar log-pdf value from ``fun`` at ``x[idx]``.
            """
            x = np.asarray(x)
            x_sub = x[idx]
            return fun(x_sub)

        return temp

    @staticmethod
    def get_prior_function_indices_list_callable(fun, indices):
        """Converts a logpdf prior on the parameters at
        indices `indices` to something on the full vector

        Args:
            fun: Callable defined on selected coordinates.
            indices: Selected parameter indices.
        """
        assert len(indices) > 0

        def temp(x):
            """Evaluate the multi-variate prior on selected coordinates of ``x``.

            Extracts the coordinates at ``indices`` from the full vector and
            passes them as positional arguments to ``fun``.

            Args:
                x: Full parameter vector (array-like).

            Returns:
                Scalar log-pdf value from ``fun`` evaluated at the selected coordinates.
            """
            x = np.asarray(x)
            return fun(*(x[i] for i in indices))

        return temp

    def amha(self, start_params, step_cov=None, debug=False, specification_name=None, n_chains=DEFAULT_AM_N_CHAINS,
             n_burnin=DEFAULT_AM_N_BURNIN, n_samples=DEFAULT_AM_N_SAMPLES, max_processes=DEFAULT_AM_MAX_PROCESSES,
             seed=None, target_acceptance_rate=DEFAULT_AM_TARGET_ACCEPTANCE_RATE, draw_size=DEFAULT_AM_DRAW_SIZE,
             scaler0=None, min_scaler=DEFAULT_AM_MIN_SCALER, max_scaler=DEFAULT_AM_MAX_SCALER,
             scaler_adjust_rate=DEFAULT_AM_SCALER_ADJUST_RATE, thinning=DEFAULT_AM_THINNING,
             scaler_adjust_denom_power=DEFAULT_AM_SCALER_ADJUST_DENOM_POWER,
             # resample_k=DEFAULT_AM_RESAMPLE_K,
             pbar_update_cadence=DEFAULT_AM_PBAR_UPDATE_CADENCE, do_adaptive=DEFAULT_AM_DO_ADAPTIVE,
             max_subchain_draws_burnin=DEFAULT_AM_MAX_SUBCHAIN_DRAWS_BURNIN,
             max_subchain_draws_sample=DEFAULT_AM_MAX_SUBCHAIN_DRAWS_SAMPlE,
             do_parallel=DEFAULT_AM_DO_PARALLEL,
             user_prompt_for_more_iters=DEFAULT_AM_USER_PROMPT_FOR_MORE_ITERS, proposal_df=DEFAULT_AM_PROPOSAL_DF,
             start_params_is_original_scale=True, step_cov_initial_samples=DEFAULT_AM_STEP_COV_INITIAL_SAMPLES,
             fix_params=None,
             normalize_step_cov=DEFAULT_AM_NORMALIZE_STEP_COV,
             do_diff_evolution_mc=DEFAULT_AM_DO_DIFF_EVOLUTION_MC,
             diff_evolution_past_samples=None,
             diff_evolution_frac_burnin=DEFAULT_AM_DIFF_EVOLUTION_FRAC_BURNIN,
             diff_evolution_max_draws=DEFAULT_AM_DIFF_EVOLUTION_MAX_DRAWS,
             diff_evolution_weight=DEFAULT_AM_DIFF_EVOLUTION_WEIGHT,
             diff_evolution_jump_cadence=DEFAULT_AM_DIFF_EVOLUTION_JUMP_CADENCE,
             scalar_jitter_bounds=DEFAULT_AM_SCALAR_JITTER_BOUNDS,
             stop_adaptation_after_burnin=DEFAULT_AM_STOP_ADAPTATION_AFTER_BURNIN,
             callback_function=None
             ) -> MCMCResults:
        """Adaptive Metropolis–Hastings with optional differential evolution (AMHA).

        Runs multichain MCMC on ``log_posterior_transformed`` (plus Jacobian
        adjustment when bounds are reparameterized).  Proposals are multivariate
        Gaussian or Student-t with **adaptive covariance** ``step_cov`` and
        global scale ``scaler`` tuned toward ``target_acceptance_rate`` (default
        ~0.234).  When ``do_diff_evolution_mc=True``, proposals also move along
        differences of past samples (DE) mixed with Gaussian noise.

        Delegates to
        :func:`~kanly.bayes.mcmc.adaptive_metropolis.adaptive_metropolis_mcmc.amha`.

        Parameters
        ----------
        start_params : array_like or dict
            Initial value per chain (dict keys are parameter names; missing
            entries default to 0).
        step_cov : ndarray, optional
            Initial proposal covariance in transformed space.  ``None`` uses the
            identity (often poor for correlated posteriors; see :meth:`sample`
            MALA warmup).
        debug : bool, default False
            Verbose progress output.
        specification_name : str, optional
            Label on returned :class:`~kanly.bayes.mcmc.mcmc_results.MCMCResults`.
        n_chains, n_burnin, n_samples : int
            Chain count and draw counts for burn-in vs retained sampling phases.
            Burn-in draws are kept but labeled; relabel ex post on results.
        max_processes : int
            Cap on parallel workers when ``do_parallel=True``.
        seed : int, optional
            RNG seed for reproducibility.
        target_acceptance_rate : float
            Target Metropolis acceptance rate for ``scaler`` adaptation.
        draw_size : int
            Batch size for vectorized proposal draws.
        scaler0, min_scaler, max_scaler, scaler_adjust_rate, scaler_adjust_denom_power
            Global proposal scale initialization and Robbins–Monro-style updates.
        thinning : int
            Keep every ``thinning``-th draw.
        pbar_update_cadence : float
            Seconds between progress-bar updates when ``debug=True``.
        do_adaptive : bool
            Adapt ``step_cov`` from pooled chain history between blocks.
        max_subchain_draws_burnin, max_subchain_draws_sample : int
            Draws per chain before regrouping chains to refresh covariance and
            DE history during burn-in vs sampling.
        do_parallel : bool
            Run chains in parallel via Ray when available.
        user_prompt_for_more_iters : bool
            Prompt to extend sampling after diagnostics.
        proposal_df : float
            Student-t df for proposals; ``inf`` means Gaussian.
        start_params_is_original_scale : bool
            If ``True``, map starts through ``inv_transform`` before sampling.
        step_cov_initial_samples : int, optional
            Delay before first covariance update (defaults from ``n_burnin``).
        fix_params : dict, optional
            Parameters held fixed at given values.
        normalize_step_cov : bool
            Reserved; normalize proposal determinant.
        do_diff_evolution_mc : bool
            Mix DE direction proposals with Gaussian noise.
        diff_evolution_past_samples : ndarray, optional
            Seed DE history (e.g. from MALA warmup in :meth:`sample`).
        diff_evolution_frac_burnin, diff_evolution_max_draws
            Which past draws enter the DE pool.
        diff_evolution_weight : float
            Weight on DE direction vs Gaussian component in proposals.
        diff_evolution_jump_cadence : int
            How often to attempt a full-scale DE jump.
        scalar_jitter_bounds : tuple
            Bounds on extra scalar jitter in proposals.
        stop_adaptation_after_burnin : bool
            If ``True``, freeze ``step_cov``/``scaler`` adaptation after burn-in.
        callback_function : callable, optional
            ``callback(params) -> str`` each retained draw.

        Returns
        -------
        MCMCResults
            Chains, acceptance rates, and diagnostics on the original parameter
            scale after inverse transform.

        Notes
        -----
        Kanly's AMHA is **not** identical to textbook adaptive Metropolis alone:
        it adds DE proposals (ter Braak) and block-wise covariance updates (Haario
        et al.).  See the bayes package README *Sampling Functionality* and
        References there.

        See Also
        --------
        mala, sample
        """

        if specification_name is None:
            specification_name = self.specification_name

        start_params = np.array(self.dict_2_array(start_params))
        self.check_bounds_satisfied(start_params, start_params_is_original_scale)

        fit = amha_internal(
            self.log_posterior_transformed, start_params, start_params_is_original_scale=start_params_is_original_scale,
            step_cov=step_cov, log_posterior_jacobian_adjustment=self.log_posterior_jacobian_adjustment,
            param_names=self.param_names,
            bounds=None if self.do_bounded_transform else self.bounds_dict_2_array(),
            specification_name=specification_name, debug=debug,
            n_chains=n_chains, n_burnin=n_burnin, n_samples=n_samples, max_processes=max_processes, seed=seed,
            target_acceptance_rate=target_acceptance_rate, draw_size=draw_size, scaler0=scaler0,
            min_scaler=min_scaler, max_scaler=max_scaler, scaler_adjust_rate=scaler_adjust_rate,
            thinning=thinning, scaler_adjust_denom_power=scaler_adjust_denom_power,
            pbar_update_cadence=pbar_update_cadence, do_adaptive=do_adaptive,
            max_subchain_draws_burnin=max_subchain_draws_burnin, max_subchain_draws_sample=max_subchain_draws_sample,
            do_parallel=do_parallel,
            user_prompt_for_more_iters=user_prompt_for_more_iters, proposal_df=proposal_df,
            step_cov_initial_samples=step_cov_initial_samples, model=self, fix_params=fix_params,
            transformations=self.transformations,
            normalize_step_cov=normalize_step_cov,
            do_diff_evolution_mc=do_diff_evolution_mc,
            diff_evolution_past_samples=diff_evolution_past_samples,
            diff_evolution_frac_burnin=diff_evolution_frac_burnin,
            diff_evolution_max_draws=diff_evolution_max_draws, diff_evolution_weight=diff_evolution_weight,
            diff_evolution_jump_cadence=diff_evolution_jump_cadence,
            scalar_jitter_bounds=scalar_jitter_bounds,
            stop_adaptation_after_burnin=stop_adaptation_after_burnin,
            callback_function=callback_function
        )

        return fit

    def mala(self, start_params,
             n_chains=DEFAULT_COORD_MALA_N_CHAINS,
             n_samples=DEFAULT_COORD_MALA_N_SAMPLES, frac_burnin=DEFAULT_COORD_MALA_FRAC_BURNIN,
             thinning=DEFAULT_COORD_MALA_THINNING,
             # resample_k=DEFAULT_COORD_MALA_RESAMPLE_K,
             target_acceptance_rate=DEFAULT_COORD_MALA_TARGET_ACCEPTANCE_RATE, do_cauchy=DEFAULT_COORD_MALA_DO_CAUCHY,
             do_mala=DEFAULT_COORD_MALA_DO_MALA, start_params_is_original_scale=True, debug=DEFAULT_COORD_MALA_DEBUG,
             pbar_update_cadence=DEFAULT_COORD_MALA_P_BAR_UPDATE_CADENCE,
             user_prompt_for_more_iters=DEFAULT_COORD_MALA_USER_PROMPT_FOR_MORE_ITERS,
             tau_adjust=DEFAULT_COORD_MALA_TAU_ADJUST,
             diff_evolution_step_cadence=DEFAULT_COORD_MALA_DIFF_EVOLUTION_STEP_CADENCE,
             diff_evolution_frac_burnin=DEFAULT_COORD_MALA_DIFF_EVOLUTION_FRAC_BURNIN,  # TODO
             diff_evolution_max_draws=DEFAULT_COORD_MALA_DIFF_EVOLUTION_MAX_DRAWS,  # TODO
             max_subchain_draws=DEFAULT_COORD_MALA_MAX_SUBCHAIN_DRAWS,
             fix_params=None, specification_name=None, callback_function=None, tau0=None,
             ) -> MCMCResults:
        """Coordinate-wise Metropolis / MALA with optional DE jumps.

        Updates **one random coordinate** per micro-step using finite-difference
        partial derivatives of ``log_posterior_transformed``.  With
        ``do_mala=True`` (default), proposals include Langevin drift toward
        higher density; with ``do_mala=False``, plain coordinate random walk.
        Per-coordinate step sizes ``tau`` adapt toward
        ``target_acceptance_rate``.  Periodic full-dimensional DE steps use
        ``diff_evolution_step_cadence``.

        Delegates to
        :func:`~kanly.bayes.mcmc.coordinate_mala.coordinate_mala_mcmc.mala`.

        Parameters
        ----------
        start_params : array_like or dict
            Initial point(s) per chain.
        n_chains, n_samples : int
            Number of chains and total draws per chain (burn-in is a **fraction**
            ``frac_burnin``, unlike AMHA's fixed ``n_burnin`` count).
        frac_burnin : float
            Share of ``n_samples`` labeled burn-in (e.g. 0.25).
        thinning : int
            Keep every ``thinning``-th draw.
        target_acceptance_rate : float
            Target acceptance for ``tau`` adaptation (default ~0.57).
        do_cauchy : bool
            Use Cauchy instead of Gaussian proposal noise.
        do_mala : bool
            Gradient-informed MALA vs coordinate Metropolis.
        start_params_is_original_scale : bool
            Whether ``start_params`` are on the original bounded scale.
        debug, pbar_update_cadence, user_prompt_for_more_iters
            Verbosity and optional extension prompt.
        tau_adjust : float
            Learning rate for per-coordinate ``tau`` updates.
        diff_evolution_step_cadence, diff_evolution_frac_burnin, diff_evolution_max_draws
            DE jump scheduling and history pool.
        max_subchain_draws : int
            Draws before regrouping chains for adaptation.
        fix_params : dict, optional
            Fixed coordinates.
        specification_name : str, optional
            Run label.
        callback_function : callable, optional
            Per-draw callback.
        tau0 : array_like, optional
            Initial per-coordinate step sizes.

        Returns
        -------
        MCMCResults
            Includes ``other_info['cov_params_unbounded_space']`` used by
            :meth:`sample` to initialize AMHA ``step_cov``.

        Notes
        -----
        Useful alone on difficult posteriors and as **warmup** inside
        :meth:`sample` when ``do_mala_cd_warmup=True``.

        See Also
        --------
        amha, sample
        """

        if specification_name is None:
            specification_name = self.specification_name

        start_params = np.array(self.dict_2_array(start_params))
        self.check_bounds_satisfied(start_params, start_params_is_original_scale)

        fit = mala_internal(
            self.log_posterior_transformed, start_params,
            log_posterior_jacobian_adjustment=self.log_posterior_jacobian_adjustment,
            n_chains=n_chains, n_samples=n_samples, frac_burnin=frac_burnin, thinning=thinning,
            # resample_k=resample_k,
            target_acceptance_rate=target_acceptance_rate, do_cauchy=do_cauchy, do_mala=do_mala,
            start_params_is_original_scale=start_params_is_original_scale, debug=debug, pbar_update_cadence=pbar_update_cadence,
            transformations=self.transformations, user_prompt_for_more_iters=user_prompt_for_more_iters,
            param_names=self.param_names, tau_adjust=tau_adjust, fix_params=fix_params,
            diff_evolution_step_cadence=diff_evolution_step_cadence,
            diff_evolution_frac_burnin=diff_evolution_frac_burnin, diff_evolution_max_draws=diff_evolution_max_draws,
            max_subchain_draws=max_subchain_draws,
            specification_name=specification_name, model=self,
            callback_function=callback_function, tau0=tau0
        )

        return fit

    def sample(
            self, start_params,

            step_cov=None, debug=False, specification_name=None, n_chains=DEFAULT_AM_N_CHAINS,
            n_burnin=DEFAULT_AM_N_BURNIN, n_samples=DEFAULT_AM_N_SAMPLES, max_processes=DEFAULT_AM_MAX_PROCESSES,
            seed=None, target_acceptance_rate=DEFAULT_AM_TARGET_ACCEPTANCE_RATE, draw_size=DEFAULT_AM_DRAW_SIZE,
            thinning=DEFAULT_AM_THINNING,
            scaler0=None, min_scaler=DEFAULT_AM_MIN_SCALER, max_scaler=DEFAULT_AM_MAX_SCALER,
            scaler_adjust_rate=DEFAULT_AM_SCALER_ADJUST_RATE,
            scaler_adjust_denom_power=DEFAULT_AM_SCALER_ADJUST_DENOM_POWER,
            # resample_k=DEFAULT_AM_RESAMPLE_K, TODO remove
            pbar_update_cadence=DEFAULT_AM_PBAR_UPDATE_CADENCE, do_adaptive=DEFAULT_AM_DO_ADAPTIVE,
            max_subchain_draws_burnin=DEFAULT_AM_MAX_SUBCHAIN_DRAWS_BURNIN,
            max_subchain_draws_sample=DEFAULT_AM_MAX_SUBCHAIN_DRAWS_SAMPlE,
            do_parallel=DEFAULT_AM_DO_PARALLEL,
            user_prompt_for_more_iters=DEFAULT_AM_USER_PROMPT_FOR_MORE_ITERS, proposal_df=DEFAULT_AM_PROPOSAL_DF,
            start_params_is_original_scale=True, step_cov_initial_samples=DEFAULT_AM_STEP_COV_INITIAL_SAMPLES,
            fix_params=None,
            normalize_step_cov=DEFAULT_AM_NORMALIZE_STEP_COV,
            do_diff_evolution_mc=DEFAULT_AM_DO_DIFF_EVOLUTION_MC,
            diff_evolution_max_draws=DEFAULT_AM_DIFF_EVOLUTION_MAX_DRAWS,
            diff_evolution_past_samples=None,
            diff_evolution_weight=DEFAULT_AM_DIFF_EVOLUTION_WEIGHT,
            diff_evolution_frac_burnin=DEFAULT_AM_DIFF_EVOLUTION_FRAC_BURNIN,
            stop_adaptation_after_burnin=DEFAULT_AM_STOP_ADAPTATION_AFTER_BURNIN,

            do_mala_cd_warmup=False,
            keep_mala_warmup=True,

            n_samples_mala=DEFAULT_COORD_MALA_N_SAMPLES, frac_burnin_mala=DEFAULT_COORD_MALA_FRAC_BURNIN,
            n_chains_mala=None,
            thinning_mala=DEFAULT_COORD_MALA_THINNING,
            # resample_k=DEFAULT_COORD_MALA_RESAMPLE_K, TODO remove
            target_acceptance_rate_mala=DEFAULT_COORD_MALA_TARGET_ACCEPTANCE_RATE,
            do_cauchy_mala=DEFAULT_COORD_MALA_DO_CAUCHY,
            do_mala_mala=DEFAULT_COORD_MALA_DO_MALA,
            tau_adjust_mala=DEFAULT_COORD_MALA_TAU_ADJUST,
            max_subchain_draws_mala=DEFAULT_COORD_MALA_MAX_SUBCHAIN_DRAWS,

            diff_evolution_step_cadence_mala=DEFAULT_COORD_MALA_DIFF_EVOLUTION_STEP_CADENCE,
            diff_evolution_frac_burnin_mala=DEFAULT_COORD_MALA_DIFF_EVOLUTION_FRAC_BURNIN,
            diff_evolution_max_draws_mala=DEFAULT_COORD_MALA_DIFF_EVOLUTION_MAX_DRAWS,
            diff_evolution_jump_cadence=DEFAULT_AM_DIFF_EVOLUTION_JUMP_CADENCE,

            tau0_mala=None,

            scalar_jitter_bounds=DEFAULT_AM_SCALAR_JITTER_BOUNDS,

            callback_function=None,

    ) -> MCMCResults:
        """Default MCMC: optional coordinate MALA warmup, then AMHA main phase.

        Summary
        -------
        High-level sampler on this model.  When ``do_mala_cd_warmup=True``, runs
        coordinate MALA to learn per-parameter scales and a covariance matrix,
        then **AMHA** (adaptive Metropolis–Hastings + optional DE) for the main
        run.  When ``do_mala_cd_warmup=False`` (default), calls :meth:`amha`
        directly with your ``step_cov`` (identity if omitted).

        Warmup hands off to AMHA: last MALA draws as chain starts;
        ``step_cov`` from MALA sample covariance in unbounded space;
        optional ``diff_evolution_past_samples`` from MALA history when DE is
        enabled.  Set ``keep_mala_warmup=True`` to attach warmup
        :class:`~kanly.bayes.mcmc.mcmc_results.MCMCResults` under
        ``fit.other_info['mala_cd_warmup_fit']``.

        Notes
        -----
        Kanly's algorithms differ from textbook AM-only / full-dimensional MALA
        but reuse the same ideas.  Implementations:

        * AMHA — ``kanly.bayes.mcmc.adaptive_metropolis.adaptive_metropolis_mcmc.amha``
        * MALA — ``kanly.bayes.mcmc.coordinate_mala.coordinate_mala_mcmc.mala``

        **Coordinate MALA (optional warmup)**

        Warmup helps when parameters have very different scales or strong
        correlation, so a blind identity ``step_cov`` for AMHA would fail.

        Coordinate MALA explained:

            This "warmup" phase is often necessary in the presence of very complicated distributions that have
            correlation between parameters and/or very different scales between parameters.  For a d-dimensional
            distribution, going straight to Adaptive-Metropolis Hastings can be challenging as specifying the
            right starting variance-covariance for the proposals can lead to basically no exploration of the
            parameter space, since the proposal scales by coordinate are so out of whack with the distribution.

            Instead, this warmup steps samples coordinate by coordinate. It adjusts the size of the steps so that we
            are sampling in every direction, allowing us to
                (a) move rapidly towards higher density areas since we don't need to figure out multivariate steps, and
                (b) learn the right "scales" of each parameter for use in the AMH sampling step later.

            Essentially the algorithm is as follows:

            1. Initialize a starting point parameter value x(0)
            3. Initialize a scaler for each coordinate s(0,k) > 0 for all coordinates k
            4. Initialize "blocks", e.g. sequences for which we sample before regrouping and
               adjusting the algorithm.  E.g., if blocks = [1000, 1000, 10_000] we'd do 1000 runs, regroup
               then 1000 more runs, regroup, then a final 10_000 runs.

            for block in blocks:

                for chain in range(n_chains):

                    for i in range(number of samples in block):

                        a.  Draw a random coordinate "k" to sample along.

                        b.  Evaluate the partial derivative of the log posterior density at x(i) along coordinate "k",
                            call it g(x(i),k)

                        c.  Form the proposal step

                            j != k:  y(i+1)[j] = x(i)[j],
                            j == k:  y(i+1)[j] = x(i)[j] + s(i,k) * g(x(i),k) + sqrt(2*s(i,k)) * epsilon(i)

                            where s(i,k) is the scalar size for coordinate k, and epsilon(i) is white noise.  Our
                            proposals are then "moving" us towards higher density.
                            This is similar to Hamiltonian Monte Carlo,
                            except we are doing only one step (instead of a full path) and one coordinate at a time.

                        e.  [When doing differential evolution in MALA, we do something like the differential evolution
                            step described below in the AMH description, we don't repeat it here, but the proposal
                            is modified as you might expect.]

                        f.  Draw U(i) ~ Uniform[0,1]

                        e.  Accept the step if

                            log( U(i) )     <   log_posterior( y(i+1) ) - log_posterior( x(i) )
                                                + Q( y(i+1) | x(i) ) - Q( x(i) | y(i+1 ),

                            where Q( z | w ) is the probability of proposing z given we are at w.  We need to do this
                            "correction" to make the chain "reversible", since the proposals here are *not* symmetric.

                            Formally,

                            Q(z | w) = -1. / (4. * s(i,k)) * (w[k] - (z[k] + s(i,k) * g(z, k))) ** 2

                            and is derived from the normal distribution and g(.) is the gradient as above.

                            Set accept(i) = 1 if the above is true, else 0.
                            Here, `log_posterior` is the logarithm of the posterior density function.

                        f.  Update the current parameter location and add it to the sample:

                            x(i+1) = accept(i) * y(i+1) + (1 - accept(i)) * x(i)

                        g.  Update the scalar parameter for coordinate k, approximately as follows

                            s(i+1,k) = s(i,k) * (1 +
                                scalar_adjust_rate/(i+1)**scaler_adjust_denom_power * (accept(i) - target_acceptance_rate)
                            )

                > Regroup from the individual MCMC chains and
                    b.  Sample from the history across chains to form the sample for the
                        Differential Evolution step (the nu(i) directions above).

            > Use the final 3/4 of the MALA samples to
                (a) approximate the variance-covariance proposal matrix for the following AMHA, and
                (b) draw initial samples for the AMHA differential evolution.

        **Adaptive Metropolis–Hastings + differential evolution (main phase)**

        Adaptive Metropolis-Hastings + Differential Evolution Explained:

            This is where the actual sampling is done for the final approximation to the posterior distribution.

            It is divided into two sections:
                * Burnin: specified by number of draws in `n_burnin`, and
                * Sampling: specified by the number of draws in `n_samples`.

            Essentially the burnin samples are used to tune to the sampling algorithm, but are not included in
            the final results.  If `stop_adaptation_after_burnin=True`, then there are no further adaptations in
            the sampling distribution after the burnin samples are complete, otherwise there are still some small
            adaptations.  In practice this doesn't matter much and can lead to faster convergence, but in theory
            the sample only asymptotically converges to the correct target distribution if the adaptations go to zero.
            Note that burnin samples are not discarded, in the final results they are just labeled as burnin.

            This is a very simplified version of the code, but essentially the code essentially runs as follows:

            1. Initialize a starting point parameter value x(0)
            2. Initialize a starting point variance-covariance Sigma(0)
            3. Initialize a scaler s(0) > 0
            4. Initialize "blocks", e.g. sequences for which we sample before regrouping and
               adjusting the algorithm.  E.g., if blocks = [1000, 1000, 10_000] we'd do 1000 runs, regroup
               then 1000 more runs, regroup, then a final 10_000 runs.

            for block in blocks:

                for chain in range(n_chains):

                    for i in range(number of samples in block):

                        a.  Draw a proposal noise vector epsilon(i) ~ Normal(0, Sigma(block)), where
                           Sigma(i) is the current proposal distribution covariance matrix at iteration i.

                        b.  [If doing Differential Evolution,] draw a pair of past points x(j1(i)), x(j2(i))
                            from the history of samples, and form the direction:

                            nu(i) = x(j1(i)) - x(j2(i)).

                            The first "block" will not have this of course
                            unless the user supplies `diff_evolution_past_samples`.

                        c.  Form a proposal parameter point

                            y(i+1) = x(i) + scaler(i) * (
                                weight_DE * nu(i) + (1-weight_DE) * epsilon(i)
                            )

                        d.  Draw U(i) ~ Uniform[0,1]

                        e.  Accept the step if

                            log( U(i) ) < log_posterior( y(i+1) ) - log_posterior( x(i) ).

                            Set accept(i) = 1 if the above is true, else 0.
                            Here, `log_posterior` is the logarithm of the posterior density function.

                        f.  Update the current parameter location and add it to the sample:

                             x(i+1) = accept(i) * y(i+1) + (1 - accept(i)) * x(i)

                        g.  Update the scalar parameter, approximately as follows

                            s(i+1) = s(i) * (1 +
                                scalar_adjust_rate/(i+1)**scaler_adjust_denom_power * (accept(i) - target_acceptance_rate)
                            )

                > Regroup from the individual MCMC chains and
                    a.  Update the variance-covariance matrix `step_cov` across chains to
                        Sigma(block+1) = Var(all samples x across chains)
                    b.  Sample from the history across chains to form the sample for the
                        Differential Evolution step (the nu(i) directions above).

            > Use the non-burnin samples to form the final approximation to the posterior distribution.

        References
        ----------
        .. [0] Metropolis–Hastings:
           https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm
        .. [1] Haario et al. (2001), adaptive Metropolis:
           https://projecteuclid.org/journals/bernoulli/volume-7/issue-2/An-adaptive-Metropolis-algorithm/bj/1080222083.full
        .. [2] ter Braak (2006), DE-MCMC:
           https://doi.org/10.1007/s11222-006-8769-1
        .. [3] MALA overview:
           https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm

        Parameters
        ----------
            start_params: the starting point, can be a dict keyed on param names or a vector.  If multiple vectors are supplied,
                there should be one for each chain. There is no default value, you need to set an initialization.
                The dict need not include all parameters, parameters not included are set to zero.
            step_cov: an initial variance-covariance for the AMH proposal; if None defaults to identity.  A good guess
                      can make the algorithm work much faster, but typically this is hard to know, especially for
                      distributions where parameters are heavily correlated and/or have very different scales.  The MALA
                      warmup step is mostly to learn this matrix.
            debug: whether to have verbose output, default `False`
            specification_name: optional string naming the sample run, default None.
            n_chains: the number of chains to run
            n_burnin: the number of "burnin" draws to run that are not used in approximating the final distribution.
                      (Note that these are retained; you can change the "burnin" ex-post on the `MCMCResults` object.)
            n_samples: the number of "sample" draws to be used in approximating the final distribution.
            max_processes: the maximum number of cores to run on, default 8.
                           Will run about `n_chains`/`max_processes` chains per core.
            seed: the random seed initialization for the random sampling object, default 0
            target_acceptance_rate: The target acceptance rate for proposed samples in the adaptive metropolis sampling, default = .234.
            draw_size: the number of parameters to draw from proposal distribution at once; default 2000.  It is much
                       faster to draw many samples at once and loop through them than draw one at a time.
            thinning: keep every `thinning`'d samples, that is if `thinning`=4 we keep every fourth draw of the MCMC.
                      Default is 1 (no thinning of sample).
            scaler0: the initial scaling factor in the AMH MCMC; defaults to 2.38/sqrt(2*d) where d is number of parameters.
            min_scaler: the minimum scaling factor for the AMH MCMC; default 1e-12
            max_scaler: the maximum scaling factor for the AMH MCMC; default inf
            scaler_adjust_rate: default .5; the scaling on the proposal evolves as
                                `scaler *= (1 + scaler_adjust_rate / max(itr - 100, 1) ** scaler_adjust_denom_power)`
                                subject to bounds `min_scaler`, `max_scaler`.
            scaler_adjust_denom_power: default .35; the scaling on the proposal evolves as
                                       `scaler *= (1 + scaler_adjust_rate / max(itr - 100, 1) ** scaler_adjust_denom_power)`
                                       subject to bounds
            pbar_update_cadence: the time (in seconds) after which to update progress bars when debug=True; default=.7s
            do_adaptive: whether to adjust proposal distribution over time, default = True.  Adaptation stops after pre-specifief
                         `n_burnin` samples if `stop_adaptation_after_burnin=True`.
            max_subchain_draws_burnin: default 5_000; the number of draws to do on a chain before regrouping across the chains,
                                       adjusting the proposal variance-covariance, and re-sampling past draws for the differential
                                       evolution step.
            max_subchain_draws_sample: default 25_000; same as `max_subchain_draws_burnin` but for sampling cycles.
            do_parallel: whether to run the chains in parrallel using `ray` package, or sequentially.  Default is True.
            user_prompt_for_more_iters: flag for whether the sampling should terminate after the specified number of samples
                                        are reached, or whether to prompt the user to enter additional samples to draw after
                                        seeing convergence diagnostics.
            proposal_df: if a positive integer, sample from a t(df=proposal_df) distribution,
                         default is inf (for Gaussian proposals)
            start_params_is_original_scale: default True. Boolean for whether supplied `start_params` starting point is in the "original"
                                  parameter space, or the "transformed" (or "unconstrained") parameter space. Typically
                                  not something to pay attention to.
            step_cov_initial_samples: if `do_adaptive=True`, the number of draws to wait for before adjusting the proposal
                                      distribution to avoid degeneracy.  Default is None, which resolves to `max(n_burnin // 100, 50)`.
            fix_params: a dict keyed on parameter names, values as floats, for parameters to "fix" in the estimation, default None
            normalize_step_cov: default False; not currently used, for normalizing determinant of proposal distribution
            do_diff_evolution_mc: default True; flag for whether to do Differential Evolution sampling (True) or straight
                                  Adaptive Metropolis Hastings (False).  With differential evolution, proposal steps are in
                                  the direction of differences of past pairs of samples, plus some Gaussian noise tuned to the
                                  sampled distribution so far.  Otherwise, it is straight Adaptive Metropolis where proposals
                                  come from a Gaussian tuned to the burnin-sample of the distribution.
            diff_evolution_max_draws: default 25_000; the number of draws from the sampling history to use for the D.E. jumps.
            diff_evolution_past_samples: default None; optional array of past draws from which to begin the D.E. sampling immediately.
            diff_evolution_weight: default 0.95; the share of weight to give to the D.E. jump proposal vs the tuned Gaussian noise in AMH.
            diff_evolution_frac_burnin: default .25; only pull D.E. points for jumps exclusing first `diff_evolution_frac_burnin` share of
                                        samples.
            diff_evolution_jump_cadence: default np.inf; cadence with which to try a full (scaler=1) jump in direction of
                                         differential evolution direction (x(i1) - x(i2)) where x(i1),x(i2) are the D.E.
                                         draws at iteration i.  Don't experiment with this.
            stop_adaptation_after_burnin: default False. If True, freeze proposal
                                          adaptation after the burn-in phase.  For long chains this
                                          rarely changes practical results; in theory adaptation must
                                          vanish for exact asymptotic correctness.

            do_mala_cd_warmup: default False. If True, run coordinate MALA warmup before AMHA.
                               Recommended for strongly correlated or badly scaled posteriors.
            keep_mala_warmup: default True. If True, store warmup MCMCResults in
                              ``fit.other_info['mala_cd_warmup_fit']``; if False, warmup time is counted
                              but warmup draws are not attached to the returned object.
            n_samples_mala: the number of samples to be drawn, default 20_000
            frac_burnin_mala: the share of sample to discard as burnin for MALA; default .25
            n_chains_mala: default 4; the number of chains to run in the MALA warmup.
            thinning_mala: keep every `thinning`'d samples, that is if `thinning`=4 we keep every fourth draw of the MALA MCMC.
                           Default is 1 (no thinning of sample).
            target_acceptance_rate_mala: default 0.57; the target acceptance rate for the CD MALA step.
            do_cauchy_mala: default False; if True noise in proposal is Cauchy distributed rather than Gaussian.
            do_mala_mala: default True; whether to use the coordinate slope in forming the proposal.
                          If false, straight Metropolis Hastings by coordinate.
            tau_adjust_mala: learning rate for MALA per-coordinate step-size adaptation.
            max_subchain_draws_mala: max draws per chain before MALA regroups chains.
            diff_evolution_step_cadence_mala: default 20; every this many iterations, try a full DE step
                                              instead of a single-coordinate update.
            diff_evolution_frac_burnin_mala: default 0.5; fraction of history to sample from for D.E. jumps in the MALA warmup,
                                             e.g. 0.3 means only sample from last 30% of history.
            diff_evolution_max_draws_mala: default 2_500; the number of draws from the sampling history to use for the D.E. jumps in the MALA warmup
            tau0_mala: initial per-coordinate step sizes for MALA warmup.
            scalar_jitter_bounds: bounds on scalar jitter in AMHA proposals (see :meth:`amha`).
            callback_function: optional ``callback(params) -> str`` each retained draw.

        Returns
        -------
        MCMCResults
            Main-phase samples; may include ``other_info['mala_cd_warmup_fit']`` when
            ``do_mala_cd_warmup`` and ``keep_mala_warmup`` are True.

        See Also
        --------
        amha, mala
        """

        if debug:
            print('\n' * 3+ '=' * 100)
            settings_dict = dict(
                **dict(
                    debug=debug,
                    pbar_update_cadence=pbar_update_cadence,
                    user_prompt_for_more_iters=user_prompt_for_more_iters,

                    seed=seed,
                    do_adaptive=do_adaptive,
                    n_chains=n_chains,
                    n_burnin=n_burnin,
                    n_samples=n_samples,
                    target_acceptance_rate=target_acceptance_rate,
                    thinning=thinning,
                    max_subchain_draws_burnin=max_subchain_draws_burnin,
                    max_subchain_draws_sample=max_subchain_draws_sample,
                    draw_size=draw_size,

                    step_cov=None if None else "User Supplied",
                    scaler0=scaler0, min_scaler=min_scaler, max_scaler=max_scaler,
                    scaler_adjust_rate=scaler_adjust_rate,
                    scaler_adjust_denom_power=scaler_adjust_denom_power,

                    do_parallel=do_parallel,
                    max_processes=max_processes,

                    start_params_is_original_scale=start_params_is_original_scale,

                    normalize_step_cov=normalize_step_cov,
                    stop_adaptation_after_burnin=stop_adaptation_after_burnin,

                    do_diff_evolution_mc=do_diff_evolution_mc,
                    diff_evolution_max_draws=diff_evolution_max_draws,
                    diff_evolution_past_samples=None if diff_evolution_past_samples is None else len(diff_evolution_past_samples),
                    diff_evolution_weight=diff_evolution_weight,
                    diff_evolution_frac_burnin=diff_evolution_frac_burnin,
                    diff_evolution_jump_cadence=diff_evolution_jump_cadence,

                    scalar_jitter_bounds = str(scalar_jitter_bounds),

                    do_mala_cd_warmup=do_mala_cd_warmup
                ),
                **(
                    dict(
                        keep_mala_warmup=keep_mala_warmup,

                        n_samples_mala=n_samples_mala, frac_burnin_mala=frac_burnin_mala,
                        n_chains_mala=n_chains_mala,
                        thinning_mala=thinning_mala,
                        max_subchain_draws_mala=max_subchain_draws_mala,
                        do_mala_mala=do_mala_mala,
                        do_cauchy_mala=do_cauchy_mala,
                        tau_adjust_mala=tau_adjust_mala,
                        tau0_mala=tau0_mala,
                        target_acceptance_rate_mala=target_acceptance_rate_mala,

                        diff_evolution_step_cadence_mala=diff_evolution_step_cadence_mala,
                        diff_evolution_frac_burnin_mala=diff_evolution_frac_burnin_mala,
                        diff_evolution_max_draws_mala=diff_evolution_max_draws_mala,

                    ) if do_mala_cd_warmup else dict()
                )
            )
            print_options(settings_dict, title='SAMPLING SETTINGS')

        if specification_name is None:
            specification_name = self.specification_name

        start_params = np.array(self.dict_2_array(start_params))
        self.check_bounds_satisfied(start_params, start_params_is_original_scale)

        if do_mala_cd_warmup:

            if debug:
                print('\n' * 3 + '=' * 100)
                print('Beginning Coordinate MALA warmup fit to tune location and parameter scale...\n')

            if n_chains_mala is None:
                n_chains_mala = n_chains

            fit_mala = self.mala(
                start_params, n_chains=n_chains_mala,
                n_samples=n_samples_mala, frac_burnin=frac_burnin_mala,
                thinning=thinning_mala,
                # resample_k=None, # TODO remove
                target_acceptance_rate=target_acceptance_rate_mala,
                max_subchain_draws=max_subchain_draws_mala,
                do_cauchy=do_cauchy_mala,
                do_mala=do_mala_mala, start_params_is_original_scale=start_params_is_original_scale, debug=debug,
                pbar_update_cadence=pbar_update_cadence,
                user_prompt_for_more_iters=user_prompt_for_more_iters,
                diff_evolution_step_cadence=diff_evolution_step_cadence_mala,
                diff_evolution_max_draws=diff_evolution_max_draws_mala,
                diff_evolution_frac_burnin=diff_evolution_frac_burnin_mala,
                tau_adjust=tau_adjust_mala, fix_params=fix_params, specification_name=specification_name,
                callback_function=callback_function, tau0=tau0_mala
            )

            if debug:
                print("Intermediate initial MALA fit:")
                print(fit_mala)

            start_params = np.array(fit_mala.get_last_sample(n_chains))
            self.check_bounds_satisfied(start_params, start_params_is_original_scale)
            if do_diff_evolution_mc:
                diff_evolution_past_samples, step_cov = fit_mala.get_inv_transform_draws(
                    size=diff_evolution_max_draws, seed=seed)
            else:
                diff_evolution_past_samples = None
                step_cov = fit_mala.other_info['cov_params_unbounded_space']

            if debug:
                print('\nMALA Warmup fit completed...\n')
                print('=' * 100)

        if step_cov is not None and np.shape(step_cov) == tuple():
            step_cov = np.array([[step_cov]])

        fit = self.amha(
            start_params, step_cov=step_cov, debug=debug, specification_name=specification_name, n_chains=n_chains,
            n_burnin=n_burnin, n_samples=n_samples, max_processes=max_processes,
            seed=seed, target_acceptance_rate=target_acceptance_rate, draw_size=draw_size,
            scaler0=None, min_scaler=min_scaler, max_scaler=max_scaler,
            scaler_adjust_rate=scaler_adjust_rate, thinning=thinning,
            scaler_adjust_denom_power=scaler_adjust_denom_power,
            # resample_k=DEFAULT_AM_RESAMPLE_K, # TODO remove
            pbar_update_cadence=pbar_update_cadence, do_adaptive=do_adaptive,
            max_subchain_draws_burnin=max_subchain_draws_burnin,
            max_subchain_draws_sample=max_subchain_draws_sample,
            do_parallel=do_parallel,
            user_prompt_for_more_iters=user_prompt_for_more_iters, proposal_df=proposal_df,
            start_params_is_original_scale=start_params_is_original_scale,
            step_cov_initial_samples=step_cov_initial_samples,
            fix_params=fix_params,
            normalize_step_cov=normalize_step_cov,
            do_diff_evolution_mc=do_diff_evolution_mc,
            diff_evolution_past_samples=diff_evolution_past_samples,
            diff_evolution_frac_burnin=diff_evolution_frac_burnin,
            diff_evolution_max_draws=diff_evolution_max_draws,
            diff_evolution_weight=diff_evolution_weight,
            diff_evolution_jump_cadence=diff_evolution_jump_cadence,
            scalar_jitter_bounds=scalar_jitter_bounds,
            stop_adaptation_after_burnin=stop_adaptation_after_burnin,
            callback_function=callback_function
        )

        if do_mala_cd_warmup:
            fit.mcmc_time += fit_mala.mcmc_time + fit_mala.result_init_time
            if keep_mala_warmup:
                fit.other_info['mala_cd_warmup_fit'] = fit_mala
            else:
                del fit_mala

        return fit

    def check_bounds_satisfied(self, start_params, start_params_is_original_scale):
        """Ensure MCMC/optimization starts are finite and strictly inside bounds.

        Parameters
        ----------
        start_params : array_like
            One vector or a list of per-chain vectors.
        start_params_is_original_scale : bool
            If ``False``, values are on the unbounded transformed scale (only
            valid when ``do_bounded_transform=True``).

        Raises
        ------
        Exception
            Wrong dimension, non-finite values, or points on/outside bounds.
        """

        if not start_params_is_original_scale and not self.do_bounded_transform:
            raise Exception("Cannot supply starting point that is on transformed unbounded scale "
                            "When `do_bounded_transform` is False!")

        if np.ndim(start_params) == 1:
            start_params = [start_params]

        for x in start_params:
            if len(x) != self.num_params:
                raise Exception(f'`start_params` must have {self.num_params} length, but you supplied {len(x)}!')

        for x in start_params:
            for i, p in enumerate(self.param_names):

                x_i = x[i]

                if not np.isfinite(x_i):
                    raise Exception(f"\nParam {i} ('{p}'): value {x_i} is not finite!")

                if p not in self.bounds:
                    continue

                if not start_params_is_original_scale:
                    x_i = self.transformations[p].transform(x[i])

                if x_i <= self.bounds[p][0] or x_i >= self.bounds[p][1]:
                    raise Exception(f"Bounds not satisfied at starting point!"
                                    f"\nParam {i} ('{p}'): value {x_i} does not fall "
                                    f"strictly in bounds {self.bounds[p]}!")

    def maximize_posterior(self, start_params, transformed=False, **bfgs_pqn_kwargs):
        """
        If `transformed=True`, then `start_params` is taken to be on the transformed scale

        Args:
            start_params: Initial point.
            transformed: Whether ``start_params`` are already transformed.
            **bfgs_pqn_kwargs: Optimizer options forwarded to ``bfgs_pqn``.
        """
        return self._maximize_function_internal(
            self.log_posterior_transformed,
            start_params, transformed=transformed, name='log_posterior', **bfgs_pqn_kwargs)

    def maximize_likelihood(self, start_params, transformed=False, **bfgs_pqn_kwargs):
        """
        If `transformed=True`, then `start_params` is taken to be on the transformed scale

        Args:
            start_params: Initial point.
            transformed: Whether ``start_params`` are already transformed.
            **bfgs_pqn_kwargs: Optimizer options forwarded to ``bfgs_pqn``.
        """
        return self._maximize_function_internal(
            self.log_likelihood_function_transformed,
            start_params, transformed=transformed, name='log_likelihood', **bfgs_pqn_kwargs)

    def _maximize_function_internal(self, func, start_params, name=None, transformed=False, **bfgs_pqn_kwargs):
        """Shared bounded/unbounded maximization wrapper used by MAP and MLE methods.

        Args:
            func: Objective callable to maximize.
            start_params: Initial point.
            name: Optional objective_function label.
            transformed: Whether ``start_params`` are already transformed.
            **bfgs_pqn_kwargs: Optimizer options.
        """
        if not transformed:
            start_params = self.inv_transform(start_params)
        result = bfgs_pqn(func, start_params,
                          # bounds=None if self.do_bounded_transform else self.bounds_dict_2_array(), # TODO remove bounds
                          maximize=True, **bfgs_pqn_kwargs)
        params = result.x.copy()
        if not transformed:
            params = self.transform(params)
        return {'params': params,
                'optimization_result': result,
                'name': name,
                'transformed': transformed}

    def __str__(self):
        """Return a formatted summary string of this Bayesian model.

        Displays the number of parameters, model type, parameter names,
        their bounds, and the registered prior distributions.

        Returns:
            Multi-line string suitable for printing.
        """

        def prior_2_str(x):
            """Convert a prior object to a display string.

            Args:
                x: A prior specification: a string, a frozen scipy distribution,
                    a frozen multivariate distribution, or a callable.

            Returns:
                Human-readable string representation of the prior.
            """
            if isinstance(x, str):
                return x
            elif isinstance(x, rv_frozen):
                return f'{x.dist.name}({x.args}, {x.kwds})'
            elif isinstance(x, multi_rv_frozen):
                return f'{x._dist}(dim={x.dim})'
            elif isinstance(x, Callable):
                return 'Callable'
            else:
                raise Exception

        df_params = pd.DataFrame({'parameter': self.param_names})
        df_params['bounds'] = [str(self.bounds.get(k, '')) for k in self.param_names]
        prior_list = []
        for k in self.param_names:
            priors_k = []
            if k in self.priors_orig:
                priors_k.append(prior_2_str(self.priors_orig[k]))
            for j in self.priors_orig.keys():
                if isinstance(j, tuple) and k in j:
                    priors_k.append(prior_2_str(self.priors_orig[j]))
            for pg, grp in self.parameter_groupings.items():
                if k in grp and pg in self.priors:
                    priors_k.append(prior_2_str(self.priors_orig[pg]))

            if priors_k:
                prior_list.append(', '.join(priors_k))
            else:
                prior_list.append('')

        df_params['priors'] = prior_list

        width = max(len('Bayesian Statistical Model'), len(df_params.to_string().split('\n')[0]))
        bar = '─' * width
        dbl_bar = '═' * width

        s = (
            bar,
            f'Bayesian Statistical Model' + (
                '\n' + self.specification_name if self.specification_name is not None else ''),
            dbl_bar,
            f'Num Params: {self.num_params}',
            f'Type:       {self.model_type()}',
            f'',
            f'{df_params.to_string()}',
            bar,
        )
        return '\n'.join(s)

    def __repr__(self):
        """Return the formatted model summary string."""
        return str(self)

    @classmethod
    def model_type(cls):
        """Return the model type identifier string.

        Returns:
            The string ``'General'`` for the base ``BayesianModel`` class;
            subclasses override this to return a more specific label.
        """
        return 'General'

    def step_back_from_bounds(self, x, step=.001):
        """Nudge ``x`` slightly inward from finite bounds.

        Useful for starting optimizers or samplers when a user supplies values
        exactly on a bound.

        Parameters
        ----------
        x : ndarray
            Parameter vector on the original scale.
        step : float, ndarray, or dict, default 0.001
            Inward shift as a fraction of interval width (two-sided bounds),
            or per-parameter overrides.

        Returns
        -------
        ndarray
            Copy of ``x`` clipped inward where bounds are finite.
        """
        xcopy = x.copy()

        if self.bounds is None or len(self.bounds) == 0:
            return xcopy

        if isinstance(step, (int, float)):
            step = np.array([float(step)] * self.num_params)
        elif isinstance(step, dict):
            return self.dict_2_array(step, default_value=.0001)
        assert np.all(step < 1) and np.all(step >= 0)

        lb = np.ones(self.num_params) * -np.inf
        ub = np.ones(self.num_params) * np.inf
        for i, k in enumerate(self.param_names):
            if k in self.bounds:
                l, u = self.bounds[k]
                if l > -np.inf and u < np.inf:
                    dist = u - l
                    l += step[i] * dist
                    u -= step[i] * dist
                elif l > -np.inf and u == np.inf:
                    l += min(step[i], 3 * abs(xcopy[i] - l))
                elif l == -np.inf and u < np.inf:
                    u -= min(step[i], 3 * abs(u - xcopy[i]))
                lb[i] = l
                ub[i] = u

        return np.clip(xcopy, lb, ub)

    def get_frozen_function(self, function, fixed_params, transformed_space=False):
        """Restrict a full-vector callable to free coordinates only.

        Parameters
        ----------
        function : callable
            ``f(params) -> float`` on the full ``params`` vector.
        fixed_params : dict or None
            ``{name: value}`` for coordinates held fixed during optimization
            or sampling.
        transformed_space : bool, default False
            If ``True``, ``function`` is defined on unbounded coordinates and
            ``fixed_params`` values on the original scale are mapped through
            ``inv_transform``.

        Returns
        -------
        temp_func : callable
            ``f_reduced(z)`` where ``z`` has length ``n_unfixed``.
        n_unfixed : int
        fixed_param_ind, fixed_param_vals, unfixed_param_ind : ndarray
            Indexing metadata to rebuild the full vector.

        Notes
        -----
        Used by :meth:`map` and :meth:`mle` when ``fixed_params`` is set.
        """

        if fixed_params is None or len(fixed_params) == 0:
            return function, self.num_params, [], [], list(range(self.num_params))

        if transformed_space:
            fixed_params_transformed = {
                k: self.transformations.get(k).inv_transform(v)
                if k in self.transformations else v
                for k, v in fixed_params.items()
            }
        else:
            fixed_params_transformed = fixed_params

        fixed_param_ind = np.array([self.param_2_idx[k] for k in fixed_params_transformed])
        fixed_param_vals = np.array(list(fixed_params_transformed.values()))
        n_unfixed = self.num_params - len(fixed_params)
        unfixed_param_ind = np.array([i for i in range(self.num_params) if i not in fixed_param_ind])

        def temp_func(z):
            """Evaluate ``function`` with fixed parameters inserted at their indices.

            Constructs the full parameter vector by placing fixed values at
            ``fixed_param_ind`` and the free values ``z`` at ``unfixed_param_ind``.

            Args:
                z: 1-D array of free parameter values (length ``n_unfixed``).

            Returns:
                Scalar function value at the reconstituted full parameter vector.
            """
            x = np.zeros(self.num_params)
            x[fixed_param_ind] = fixed_param_vals
            x[unfixed_param_ind] = z
            return function(x)

        return temp_func, n_unfixed, fixed_param_ind, fixed_param_vals, unfixed_param_ind

    def map(self, start_params, fixed_params=None, use_transformed_scale=True, onesided_fd=None,
            maxiter=200, B0=1.0, xtol=1e-8, ftol=1e-8, gtol=1e-4, dx_fd=1e-6, momentum=.05, seed=0, debug=False,
            user_prompt_for_more_iters=False,
            ) -> BayesianModelMaximizationResult:
        """Maximum a posteriori (MAP) estimate via bounded BFGS.

        Maximizes ``log_posterior`` (likelihood + priors + bounds).  By default
        optimization runs in **transformed** unbounded coordinates when
        ``do_bounded_transform=True``, then maps the solution back to the
        original scale.

        Parameters
        ----------
        start_params, fixed_params : array_like or dict
            On the **original** parameter scale unless
            ``use_transformed_scale=False``.
        use_transformed_scale : bool, default True
            Optimize ``log_posterior_transformed`` in internal coordinates.
        onesided_fd, maxiter, B0, xtol, ftol, gtol, dx_fd, momentum, seed, debug,
        user_prompt_for_more_iters
            Forwarded to :func:`~kanly.bayes.map.maximum_a_posteriori.bfgs_pqn`.

        Returns
        -------
        BayesianModelMaximizationResult
            Mode, optimization diagnostics, and parameter names.

        See Also
        --------
        mle
        """

        return map(self, start_params, fixed_params=fixed_params, use_transformed_scale=use_transformed_scale, onesided_fd=onesided_fd,
                   maxiter=maxiter, B0=B0, xtol=xtol, ftol=ftol, gtol=gtol, dx_fd=dx_fd, momentum=momentum, seed=seed,
                   debug=debug, user_prompt_for_more_iters=user_prompt_for_more_iters)

    def mle(self, start_params, fixed_params=None, use_transformed_scale=True, onesided_fd=None,
            maxiter=200, B0=1.0, xtol=1e-8, ftol=1e-8, gtol=1e-4, dx_fd=1e-6, momentum=.05, seed=0, debug=False,
            user_prompt_for_more_iters=False, pbar_update_cadence=.3
            ) -> BayesianModelMaximizationResult:
        """Maximum likelihood estimate (flat prior on parameters).

        Same optimizer as :meth:`map`, but maximizes ``log_likelihood`` only
        (priors ignored).

        Parameters
        ----------
        start_params, fixed_params : array_like or dict
            On the original scale unless ``use_transformed_scale=False``.
        use_transformed_scale : bool, default True
            Optimize in transformed coordinates when bounds use reparameterization.
        onesided_fd, maxiter, B0, xtol, ftol, gtol, dx_fd, momentum, seed, debug,
        user_prompt_for_more_iters, pbar_update_cadence
            Optimizer options (see :meth:`map`).

        Returns
        -------
        BayesianModelMaximizationResult

        See Also
        --------
        map
        """

        return mle(self, start_params, fixed_params=fixed_params, use_transformed_scale=use_transformed_scale, onesided_fd=onesided_fd,
                   maxiter=maxiter, B0=B0, xtol=xtol, ftol=ftol, gtol=gtol, dx_fd=dx_fd, momentum=momentum, seed=seed,
                   debug=debug, user_prompt_for_more_iters=user_prompt_for_more_iters,
                   pbar_update_cadence=pbar_update_cadence)

    # def get_frozen_function(bmodel, function, fixed_params, return_sub_func=True, transform=False):
    #     fixed_params_transformed = {
    #         k: bmodel.transformations.get(k).inv_transform(v)
    #         if k in bmodel.transformations else v
    #         for k, v in fixed_params.items()
    #     }
    #     fixed_param_ind = np.array([bmodel.param_2_idx[k] for k in FIX_PARAMS])
    #     fixed_param_vals = np.array(list(fixed_params_transformed.values()))
    #     n_unfixed = num_params - len(fixed_params)
    #     unfixed_param_ind = np.array([i for i in range(num_params) if i not in fixed_param_ind])
    #
    #     def temp_func(z):
    #         x = np.zeros(num_params)
    #         x[fixed_param_ind] = fixed_param_vals
    #         x[unfixed_param_ind] = z
    #         return function(x)
    #
    #     return temp_func, n_unfixed, unfixed_param_ind
    #
    # def get_frozen_function(function, fixed_params):
    #     fixed_params_transformed = {
    #         k: bmodel.transformations.get(k).inv_transform(v)
    #         if k in bmodel.transformations else v
    #         for k, v in fixed_params.items()
    #     }
    #     fixed_param_ind = np.array([bmodel.param_2_idx[k] for k in FIX_PARAMS])
    #     fixed_param_vals = np.array(list(fixed_params_transformed.values()))
    #
    #     #    n_unfixed = num_params - len(fixed_params)
    #     #    unfixed_param_ind = np.array([i for i in range(num_params) if i not in fixed_param_ind])
    #
    #     def temp_func(z):
    #         x = z.copy()
    #         x[fixed_param_ind] = fixed_param_vals
    #         return function(x)
    #
    #     return temp_func

# def amha(log_likelihood_function, x0, bounds=None, do_bounded_transform=True, priors=None, param_names=None,
#          num_params=None, specification_name=None, other_info=None, debug=False, n_chains=DEFAULT_AM_N_CHAINS,
#          step_cov=None, n_burnin=DEFAULT_AM_N_BURNIN, n_samples=DEFAULT_AM_N_SAMPLES,
#          max_processes=DEFAULT_AM_MAX_PROCESSES, seed=None, target_acceptance_rate=DEFAULT_AM_TARGET_ACCEPTANCE_RATE,
#          draw_size=DEFAULT_AM_DRAW_SIZE, scaler0=None, min_scaler=DEFAULT_AM_MIN_SCALER,
#          max_scaler=DEFAULT_AM_MAX_SCALER, scaler_adjust_rate=DEFAULT_AM_SCALER_ADJUST_RATE,
#          thinning=DEFAULT_AM_THINNING, scaler_adjust_denom_power=DEFAULT_AM_SCALER_ADJUST_DENOM_POWER,
#          resample_k=DEFAULT_AM_RESAMPLE_K, pbar_update_cadence=DEFAULT_AM_PBAR_UPDATE_CADENCE,
#          do_adaptive=DEFAULT_AM_DO_ADAPTIVE,
#          max_subchain_draws_burnin=DEFAULT_AM_MAX_SUBCHAIN_DRAWS_BURNIN,
#          max_subchain_draws_sample=DEFAULT_AM_MAX_SUBCHAIN_DRAWS_SAMPlE,
#          do_parallel=DEFAULT_AM_DO_PARALLEL, user_prompt_for_more_iters=DEFAULT_AM_USER_PROMPT_FOR_MORE_ITERS,
#          proposal_df=DEFAULT_AM_PROPOSAL_DF, x0_is_original_scale=True,
#          step_cov_adjust_rate=DEFAULT_AM_STEP_COV_ADJUST_RATE, fix_params=None,
#          show_r_hat_ever_subchain=False,
#          normalize_step_cov=DEFAULT_AM_NORMALIZE_STEP_COV
#          ) -> MCMCResults:
#     """Adaptive Metropolis Hastings"""
#
#     bmodel = BayesianModel(log_likelihood_function, bounds=bounds, do_bounded_transform=do_bounded_transform,
#                            priors=priors, param_names=param_names, num_params=num_params,
#                            specification_name=specification_name, other_info=other_info)
#     fit = bmodel.amha(x0, step_cov=step_cov, debug=debug, specification_name=specification_name, n_chains=n_chains,
#                       n_burnin=n_burnin, n_samples=n_samples, max_processes=max_processes, seed=seed,
#                       target_acceptance_rate=target_acceptance_rate, draw_size=draw_size, scaler0=scaler0,
#                       min_scaler=min_scaler, max_scaler=max_scaler, scaler_adjust_rate=scaler_adjust_rate,
#                       thinning=thinning, scaler_adjust_denom_power=scaler_adjust_denom_power, resample_k=resample_k,
#                       pbar_update_cadence=pbar_update_cadence, do_adaptive=do_adaptive,
#                       max_subchain_draws_burnin=max_subchain_draws_burnin,
#                       max_subchain_draws_sample=max_subchain_draws_sample,
#                       do_parallel=do_parallel,
#                       user_prompt_for_more_iters=user_prompt_for_more_iters, proposal_df=proposal_df,
#                       x0_is_original_scale=x0_is_original_scale, step_cov_adjust_rate=step_cov_adjust_rate,
#                       fix_params=fix_params, show_r_hat_ever_subchain=show_r_hat_ever_subchain,
#                       normalize_step_cov=normalize_step_cov,
#                       )
#     return fit
#
#
# def mala(log_likelihood_function, x0, bounds=None, do_bounded_transform=True, priors=None, param_names=None,
#          num_params=None, specification_name=None, other_info=None, n_chains=DEFAULT_COORD_MALA_N_CHAINS,
#          n_samples=DEFAULT_COORD_MALA_N_SAMPLES, thinning=DEFAULT_COORD_MALA_THINNING,
#          resample_k=DEFAULT_COORD_MALA_RESAMPLE_K, target_acceptance_rate=DEFAULT_COORD_MALA_TARGET_ACCEPTANCE_RATE,
#          do_cauchy=DEFAULT_COORD_MALA_DO_CAUCHY, do_mala=DEFAULT_COORD_MALA_DO_MALA, x0_is_original_scale=True,
#          debug=DEFAULT_COORD_MALA_DEBUG, pbar_update_cadence=DEFAULT_COORD_MALA_P_BAR_UPDATE_CADENCE,
#          user_prompt_for_more_iters=DEFAULT_COORD_MALA_USER_PROMPT_FOR_MORE_ITERS,
#          tau_adjust=DEFAULT_COORD_MALA_TAU_ADJUST, fix_params=None) -> MCMCResults:
#     """Metropolis Adjusted Langevin MCMC"""
#
#     bmodel = BayesianModel(log_likelihood_function, bounds=bounds, do_bounded_transform=do_bounded_transform,
#                            priors=priors, param_names=param_names, num_params=num_params,
#                            specification_name=specification_name, other_info=other_info)
#
#     fit = bmodel.mala(x0, n_chains=n_chains, n_samples=n_samples,
#              thinning=thinning, resample_k=resample_k,
#              target_acceptance_rate=target_acceptance_rate, do_cauchy=do_cauchy,
#              do_mala=do_mala, x0_is_original_scale=x0_is_original_scale, debug=debug,
#              pbar_update_cadence=pbar_update_cadence,
#              user_prompt_for_more_iters=user_prompt_for_more_iters,
#              tau_adjust=tau_adjust, fix_params=fix_params)
#
#     return fit
#
# # #
# if __name__ == '__main__':
#     from scipy.stats import multivariate_normal
#     import matplotlib.pyplot as plt
#     from kanly.api import amha, read
#
#     F = multivariate_normal([3, 2.], [[1, .3], [.3, 10]])
#     # model = BayesianModel(
#     #                       #bounds={'a': [1, 10_000]},
#     #                       #priors={'b': 'normal(-10,.5)'}
#     #                       )
#     # fit = amha(F.logpdf, [2., .5], param_names=['a', 'b'], debug=True,
#     #           priors=None,
#     #           n_burnin=30_000, n_samples=60_000, user_prompt_for_more_iters=False, n_chains=5)
#     # print(fit)
#
#     #bmodel = BayesianModel
#     fit = amha(F.logpdf, [2., .5], param_names=['a', 'b'], debug=True,
#                bounds={'a': [1, 10_000]},
#                priors=None,
#                n_burnin=10_000,
#                n_samples=10_000,
#                user_prompt_for_more_iters=False, n_chains=5,
#                )
#     fit.save('fit_mcmc_example.txt')
#
#     fit2 = read('fit_mcmc_example.txt')
#     print(fit2)
#
#     print(fit.model.log_posterior_jacobian_adjustment.__doc__)
#     # print(fit)
#     # print(fit.ess)
#     # print(fit.get_ess_from_batched_means(selection='coordinate'))
#     # print(fit.get_ess_from_batched_means(selection='joint'))
#     # fit.multi_hist(fit.param_names)
#     # plt.show()
# #

#
#     fit = mod1.amha([.5, 1], debug=True, n_samples=3_000, n_burnin=3_000)
#     print(fit)
#
#     # check transforms
#     print(fit.map_params)
#     print(fit.transform(fit.inv_transform(fit.map_params)))
#
#     print(fit.max_log_posterior)
#     print(fit.log_posterior_original(fit.map_params))
#     print(fit.log_posterior(fit.inv_transform(fit.map_params)))
#
# if __name__ == '__main__':
#     from kanly.api import bayes_nlls_model
#
#     np.random.seed(0)
#     n = 100
#     x = 1.56 * np.random.randn(n)
#     z = np.random.rand(n)
#     y = 3 + 10 * x + np.random.randn(n) * 3
#     wts = .01 + np.random.rand(n)
#     data = {'x': x, 'y': y, 'z': z, 'wts': wts, 'g': np.random.randint(0, 12, n)}
#
#     # model = bayes_nlls_model(
#     #     '[y] ~ {a} + {b}*[x]',
#     #
#     #     data,
#     #     debug=False
#     # )
#     #
#     # print(model)
#     #
#     # print(model.amha([0,1,1], n_samples=30_000))
#
#     model = bayes_nlls_model(
#         '[y] ~ {a} + {b}*[x]',
#         data, priors={'a': 'half_norm(0, 25)', 'b': 'flat(14, 33)'},
#         debug=True
#     )
#
#     print(model)
#     print(model.amha([20, 30, 1], n_samples=10_000))
# #     print(model.amha([20, 30, 1], n_samples=10_000, do_parallel=False))
# #     print(model.mala([20, 30, 1], n_samples=10_000))
