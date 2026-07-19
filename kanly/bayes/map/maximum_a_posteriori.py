"""MAP/MLE utilities for :class:`kanly.bayes.bayesian_model.BayesianModel`."""

from __future__ import absolute_import, print_function, annotations

import numpy as np

from typing import TYPE_CHECKING
from copy import copy
import pprint

from pandas import Series

from kanly.optimize.bfgs_bounded_quasi_newton import bfgs_pqn
from kanly.dill_object import DillObject

if TYPE_CHECKING:
    from kanly.bayes.bayesian_model import BayesianModel


class BayesianModelMaximizationResult(DillObject):
    """Structured output from MAP/MLE optimization on a Bayesian model."""

    def __init__(self, model: BayesianModel, x, x_unbounded_space, fixed_params, max_log_posterior,
                 use_transformed_scale,
                 optimization_result, optimizer_options, minimand_function, minimand_function_frozen, n_unfixed,
                 fixed_param_ind, fixed_param_vals, unfixed_param_ind, inv_2nd_deriv):
        """Initialize optimization result container.

        Args:
            model: Source Bayesian model instance.
            x: Optimizer solution on original model scale.
            x_unbounded_space: Solution on transformed/unbounded scale.
            fixed_params: Optional mapping of coordinates held fixed.
            max_log_posterior: Objective value at optimum.
            use_transformed_scale: Whether optimization happened in transformed space.
            optimization_result: Raw optimizer return object.
            optimizer_options: Options dictionary passed to optimizer.
            minimand_function: Original objective_function callable.
            minimand_function_frozen: Objective with fixed params folded in.
            n_unfixed: Number of free coordinates optimized.
            fixed_param_ind: Fixed coordinate indices.
            fixed_param_vals: Fixed coordinate values.
            unfixed_param_ind: Optimized coordinate indices.
            inv_2nd_deriv: Approximate inverse second derivative summary.
        """
        self.model = model
        self.x = x
        self.x_unbounded_space = x_unbounded_space
        self.fixed_params = fixed_params
        self.max_log_posterior = max_log_posterior
        self.use_transformed_scale = use_transformed_scale
        self.optimization_result = optimization_result
        self.optimizer_options = optimizer_options
        self.minimand_function = minimand_function
        self.minimand_function_frozen = minimand_function_frozen
        self.n_unfixed = n_unfixed
        self.fixed_param_ind = fixed_param_ind
        self.fixed_param_vals = fixed_param_vals
        self.unfixed_param_ind = unfixed_param_ind
        self.inv_2nd_deriv = inv_2nd_deriv

    def __str__(self):
        """Return a pretty-printed string representation of this result object.

        Returns:
            String from ``pprint.pformat`` of the instance ``__dict__``.
        """
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        """Return the pretty-printed string representation.

        Returns:
            String returned by ``str(self)``.
        """
        return str(self)


def maximize_function(function, model: BayesianModel, x0, fixed_params=None, use_transformed_scale=True,
                      onesided_fd=None, maxiter=200, B0=1.0, xtol=1e-8, ftol=1e-8, gtol=1e-4, dx_fd=1e-6, momentum=.05,
                      seed=0, debug=False, user_prompt_for_more_iters=False, pbar_update_cadence=.3,
                      return_inv_2nd_derivatives=True,
                      ) -> BayesianModelMaximizationResult:
    """Maximize a scalar objective_function built on a Bayesian model parameterization.

    This is the shared worker used by :func:`map` and :func:`mle`.

    Args:
        function: Objective callable to minimize (typically negative log target).
        model: Bayesian model that provides transforms and parameter metadata.
        x0: Initial point on original model scale.
        fixed_params: Optional fixed-parameter mapping.
        use_transformed_scale: If True, optimize in unconstrained transformed space.
        onesided_fd: Optional finite-difference strategy flag for optimizer.
        maxiter: Maximum optimizer iterations.
        B0: Initial inverse-Hessian scaling for quasi-Newton updates.
        xtol: Parameter-change convergence tolerance.
        ftol: Objective-change convergence tolerance.
        gtol: Gradient-norm convergence tolerance.
        dx_fd: Finite-difference step size.
        momentum: Optimizer momentum/tail averaging factor.
        seed: Random seed for stochastic optimizer components.
        debug: If True, print optimizer diagnostics.
        user_prompt_for_more_iters: Whether optimizer may prompt for more iterations.
        pbar_update_cadence: Progress update cadence.
        return_inv_2nd_derivatives: Whether to estimate inverse second derivatives.
    """

    # Freeze fixed coordinates so the optimizer works only on the free subspace.
    function_frozen, n_unfixed, fixed_param_ind, fixed_param_vals, unfixed_param_ind \
        = model.get_frozen_function(function, fixed_params, transformed_space=use_transformed_scale)

    x0 = model.dict_2_array(x0)
    x0 = np.asarray(x0).astype(float)
    if use_transformed_scale:
        x0 = model.inv_transform(x0)
    else:
        x0 = x0.copy()

    z0 = x0[unfixed_param_ind]

    if not use_transformed_scale and model.bounds is not None:
        bounds = np.ones((2, model.num_params)) * np.inf
        bounds[0, :] = -np.inf
        for i, k in enumerate(model.param_names):
            if k in model.bounds:
                bounds[:, i] = model.bounds[k]
        bounds = bounds[:, unfixed_param_ind]
    else:
        bounds = None

    result = bfgs_pqn(
        function_frozen, z0, maxiter=maxiter, debug=debug, B0=B0,
        xtol=xtol, ftol=ftol, gtol=gtol, onesided_fd=onesided_fd,
        momentum=momentum, seed=seed, user_prompt_for_more_iters=user_prompt_for_more_iters,
        dx_fd=dx_fd,
        bounds=bounds,
    )

    x_optim = np.zeros(model.num_params)
    x_optim[unfixed_param_ind] = result.x
    if fixed_params is not None:
        x_optim[fixed_param_ind] = fixed_param_vals

    x_optim_unbdd = x_optim
    if use_transformed_scale:
        x_optim = model.transform(x_optim)

    log_posterior_value = -result.fun

    inv_2nd_deriv = None
    if return_inv_2nd_derivatives:

        temp_func = model.log_posterior_transformed if use_transformed_scale else model.log_posterior

        g2 = []
        f0 = temp_func(x_optim_unbdd)

        for i, k in enumerate(model.param_names):
            dx = min(max(abs(x_optim_unbdd[i]), 1.0), 10000) * 1e-6

            x0_copy = x_optim_unbdd.copy()
            x0_copy[i] += dx
            fh = temp_func(x0_copy)

            x0_copy = x_optim_unbdd.copy()
            x0_copy[i] -= dx
            fl = temp_func(x0_copy)

            g2.append((fh - 2 * f0 + fl) / dx ** 2)

        g2 = np.array(g2)
        inv_2nd_deriv = 1.0 / np.where(abs(g2) < 1e-20, 1e-20, g2)

    return BayesianModelMaximizationResult(
        model=model,
        x=Series(x_optim, index=model.param_names),
        x_unbounded_space=Series(x_optim_unbdd, index=model.param_names),
        fixed_params=copy(fixed_params) if fixed_params is not None else None,
        max_log_posterior=log_posterior_value,
        use_transformed_scale=use_transformed_scale,
        optimization_result=result,
        optimizer_options=dict(
            onesided_fd=onesided_fd, maxiter=maxiter,
            B0=B0, xtol=xtol, ftol=ftol, gtol=gtol, dx_fd=dx_fd,
            momentum=momentum, seed=seed),
        minimand_function=function,
        minimand_function_frozen=function_frozen,
        n_unfixed=n_unfixed,
        fixed_param_ind=fixed_param_ind,
        fixed_param_vals=fixed_param_vals,
        unfixed_param_ind=unfixed_param_ind,
        inv_2nd_deriv=Series(inv_2nd_deriv, index=model.param_names),
    )


def mle(model: BayesianModel, x0, fixed_params=None, use_transformed_scale=True, onesided_fd=None,
        maxiter=200, B0=1.0, xtol=1e-8, ftol=1e-8, gtol=1e-4, dx_fd=1e-6, momentum=.05, seed=0, debug=False,
        user_prompt_for_more_iters=False, pbar_update_cadence=.3, return_inv_2nd_derivatives=True,
        ) -> BayesianModelMaximizationResult:
    """Compute the maximum likelihood estimate via quasi-Newton optimization.

    Args:
        model: Bayesian model instance.
        x0: Initial point on original parameter scale.
        fixed_params: Optional fixed-parameter mapping.
        use_transformed_scale: Whether to optimize in transformed coordinates.
        onesided_fd: Optional finite-difference strategy.
        maxiter: Maximum optimizer iterations.
        B0: Initial inverse-Hessian scaling.
        xtol: Parameter tolerance.
        ftol: Objective tolerance.
        gtol: Gradient tolerance.
        dx_fd: Finite-difference step size.
        momentum: Optimizer momentum factor.
        seed: Random seed.
        debug: Verbose mode.
        user_prompt_for_more_iters: Whether optimizer can request more iterations.
        pbar_update_cadence: Progress update cadence.
        return_inv_2nd_derivatives: Whether to estimate curvature summary.
    """
    if use_transformed_scale:
        function = lambda x: -model.log_likelihood_function(x)
    else:
        function = lambda x: -model.log_likelihood_function_transformed(x)
    return maximize_function(function, model, x0, fixed_params=fixed_params,
                             use_transformed_scale=use_transformed_scale,
                             onesided_fd=onesided_fd, maxiter=maxiter, B0=B0, xtol=xtol, ftol=ftol, gtol=gtol,
                             dx_fd=dx_fd, momentum=momentum,
                             seed=seed, debug=debug, user_prompt_for_more_iters=user_prompt_for_more_iters,
                             pbar_update_cadence=pbar_update_cadence,
                             return_inv_2nd_derivatives=return_inv_2nd_derivatives)


def map(model: BayesianModel, x0, fixed_params=None, use_transformed_scale=True, onesided_fd=None,
        maxiter=200, B0=1.0, xtol=1e-8, ftol=1e-8, gtol=1e-4, dx_fd=1e-6, momentum=.05, seed=0, debug=False,
        user_prompt_for_more_iters=False, return_inv_2nd_derivatives=True,
        ) -> BayesianModelMaximizationResult:
    """Compute the maximum a-posteriori estimate via quasi-Newton optimization.

    Args:
        model: Bayesian model instance.
        x0: Initial point on original parameter scale.
        fixed_params: Optional fixed-parameter mapping.
        use_transformed_scale: Whether to optimize in transformed coordinates.
        onesided_fd: Optional finite-difference strategy.
        maxiter: Maximum optimizer iterations.
        B0: Initial inverse-Hessian scaling.
        xtol: Parameter tolerance.
        ftol: Objective tolerance.
        gtol: Gradient tolerance.
        dx_fd: Finite-difference step size.
        momentum: Optimizer momentum factor.
        seed: Random seed.
        debug: Verbose mode.
        user_prompt_for_more_iters: Whether optimizer can request more iterations.
        return_inv_2nd_derivatives: Whether to estimate curvature summary.
    """
    if use_transformed_scale:
        function = lambda x: -model.log_posterior_transformed(x)
    else:
        function = lambda x: -model.log_posterior(x)

    return maximize_function(function, model, x0, fixed_params=fixed_params,
                             use_transformed_scale=use_transformed_scale,
                             onesided_fd=onesided_fd, maxiter=maxiter, B0=B0, xtol=xtol, ftol=ftol, gtol=gtol,
                             dx_fd=dx_fd, momentum=momentum,
                             seed=seed, debug=debug, user_prompt_for_more_iters=user_prompt_for_more_iters,
                             return_inv_2nd_derivatives=return_inv_2nd_derivatives)

    # if __name__ == '__main__':
#
#     from kanly.api import bayes_nlls_model
#
#     np.random.seed(0)
#     n = 400
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
#         '[y] ~ {a} + {b}*[x] + {c}*[z]',
#         data,
#         # priors={'a': 'halfnorm(0, 25)', 'b': 'flat(14, 33)'},
#         bounds={'b': (0, 100)},
#         debug=True
#     )
#     print(model)
#
#     print(map(model, [.01, 3, 1, 1], use_transformed_scale=True, debug=True, B0=10000,
#               fixed_params={'c': 5}
#               ).x_map)
#
#     print(map(model, [.01, 3, 1, 1], use_transformed_scale=False, debug=True, B0=10000,
#               fixed_params={'c': 5}
#               ).x_map)
