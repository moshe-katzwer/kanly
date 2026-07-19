"""
A lightweight wrapper class for results from NLLS minimize internal call.
Not the full-blown kanly NLLS result, but good for quick and dirty results
using an arbitrary residual function.
"""
from __future__ import absolute_import, print_function

import numpy as np
import pandas as pd
import datetime

from kanly.regression.regression_results_base import TestingResultsBase
from kanly.regression.linear_models.variance_covariance2 import SparseVarianceCovariance2
from scipy.stats import t as t_dist


class NllsMinimizeInternalResult(TestingResultsBase):
    """Lightweight result object returned by the internal trust-region NLLS solver.

    Stores raw optimisation diagnostics, parameter estimates, covariance
    approximations, residuals, and a printable summary without requiring the
    full public ``SparseNonlinearLeastSquaresRegressionResults`` wrapper.
    """

    def __init__(self, params, active_mask, bounds, converged, cost, dF_over_F, dense_threshold_mb, fit_elapsed, grad,
                 hessian, is_bounded, iterations, jac, jac_method, jacobian_function_callable, l2_penalties, loss,
                 loss_scale, message, norm_dx, normalized_cov_params, optimality, optimization_path, penalty,
                 regularize_to_values, resid, root_loss_function_orig, root_loss_resid, scale_l2_penalties,
                 start_params, status, v, wresid, param_names=None, specification_name=None):
        """Initialise an internal NLLS optimisation result.

        Args:
            params: Final parameter vector.
            active_mask: Bound-activity mask (-1 lower, 0 inactive, 1 upper).
            bounds: Parameter bounds used during fitting.
            converged: Whether the solver met a convergence criterion.
            cost: Final weighted sum-of-squares cost.
            dF_over_F: Relative objective_function change at termination.
            dense_threshold_mb: Dense-matrix threshold used by Jacobian helpers.
            fit_elapsed: Total optimisation time in seconds.
            grad: Final gradient vector.
            hessian: Final Hessian/quadratic approximation matrix.
            is_bounded: Whether finite bounds were active in the problem.
            iterations: Number of outer iterations performed.
            jac: Final Jacobian matrix.
            jac_method: Jacobian method used by the solver.
            jacobian_function_callable: Callable used to evaluate Jacobians.
            l2_penalties: L2 penalties used in the objective_function, if any.
            loss: Final unpenalised cost/loss value.
            loss_scale: Scale used for robust root-loss residuals.
            message: Human-readable termination message.
            norm_dx: Norm of the final scaled parameter step.
            normalized_cov_params: Approximate ``(J'J)^{-1}`` matrix.
            optimality: Final infinity-norm optimality measure.
            optimization_path: Optional list of iterates/objective_function values.
            penalty: Final regularisation penalty value.
            regularize_to_values: Parameter target values for L2 penalties.
            resid: Final raw residual vector.
            root_loss_function_orig: Original robust loss argument.
            root_loss_resid: Final transformed residual vector.
            scale_l2_penalties: Whether L2 penalties were scaled by sample size.
            start_params: Initial parameter vector.
            status: Integer solver status code.
            v: Trust-region scaling vector at termination.
            wresid: Final weighted residual vector.
            param_names: Optional ordered parameter names.
            specification_name: Optional label for the fitted specification.
        """
        self._params = params
        self.active_mask = active_mask
        self.bounds = bounds
        self.converged = converged
        self.cost = cost
        self.dF_over_F = dF_over_F
        self.dense_threshold_mb = dense_threshold_mb
        self.fit_elapsed = fit_elapsed
        self.grad = grad
        self.hessian = hessian
        self.is_bounded = is_bounded
        self.iterations = iterations
        self.jac = jac
        self.jac_method = jac_method
        self.jacobian_function_callable = jacobian_function_callable
        self.l2_penalties = l2_penalties
        self.loss = loss
        self.loss_scale = loss_scale
        self.message = message
        self.norm_dx = norm_dx
        self.normalized_cov_params = normalized_cov_params
        self.optimality = optimality
        self.optimization_path = optimization_path
        self.penalty = penalty
        self.regularize_to_values = regularize_to_values
        self.resid = resid
        self.root_loss_function_orig = root_loss_function_orig
        self.root_loss_resid = root_loss_resid
        self.scale_l2_penalties = scale_l2_penalties
        self.start_params = start_params
        self.status = status
        self.v = v
        self.wresid = wresid
        self.num_params = len(self._params)
        self.nobs = len(self.resid)
        self.df_resid = self.nobs - self.num_params
        self.wssr = sum(self.wresid ** 2)
        self.scale = self.wssr / self.nobs

        self._cov_params = SparseVarianceCovariance2.compute_cov_params(
            'ols_small', dict(), False, self.df_resid, self.wssr, self.resid,
            self.normalized_cov_params, False, None
        )
        self._cov_params = (self.scale * self.normalized_cov_params.toarray()
                            * self.nobs / (self.nobs - self.num_params))
        self._bse = np.sqrt(np.diag(self._cov_params))

        if param_names is None:
            param_names = ['<x%d>' % d for d in range(self.num_params)]
        self.param_names = param_names

        self.params = pd.Series(self._params, index=self.param_names)
        self.bse = pd.Series(self._bse, index=self.param_names)
        self.cov_params = pd.DataFrame(self._cov_params, index=self.param_names, columns=self.param_names)
        self.date = datetime.datetime.today().strftime('%b %d, %Y')
        self.timestamp = datetime.datetime.today().strftime('%H:%M:%S')

        self.specification_name = specification_name

        self.model = dict()

    def summary(self, *args, **kwargs):
        """Build a printable summary table for the internal solver result.

        Args:
            *args: Ignored; present for compatibility with other result objects.
            **kwargs: Ignored; present for compatibility with other result objects.

        Returns:
            Multi-line string containing fit diagnostics and coefficient table.
        """
        header = pd.DataFrame([
            ('nobs:', self.nobs),
            ('df resid:', self.df_resid),
            ('cost:', self.cost),
            ('penalty:', self.penalty),
            ('scale:', self.scale),
            ('converged:', self.converged),
            ('iterations:', self.iterations),
            ('|dx|:', self.norm_dx),
            ('dF/F:', self.dF_over_F),
            ('fit time:', '%.3fs' % self.fit_elapsed),
            ('date:', self.date),
            ('time:', self.timestamp),
        ]).to_string(index=False, header=False)

        summary_df = pd.DataFrame(
            {
                'coef': self._params,
                'std err': self._bse,
            },
            index=self.param_names,
        )
        summary_df['t'] = summary_df['coef'] / summary_df['std err']
        summary_df['p>|t|'] = 2 * t_dist.cdf(-np.abs(summary_df['t']), self.df_resid)
        cv = t_dist.ppf(.975, self.df_resid)
        summary_df['[.025, '] = self._params - cv * self._bse
        summary_df[' .975]'] = self._params + cv * self._bse

        summary_df_strs = summary_df.to_string().split('\n')

        width = max([len(s) for s in header.split('\n')] + [len(s) for s in summary_df_strs])
        dbl_bar = '═' * width
        sng_bar = '─' * width

        summary = (dbl_bar + '\n' + 'NLLS Results'
                   + ('' if self.specification_name is None else ('\n' + self.specification_name))
                   + '\n' + dbl_bar + '\n\n' + header + '\n\n'
                   + dbl_bar + '\n' + summary_df_strs[0] + '\n' + sng_bar + '\n'
                   + '\n'.join(summary_df_strs[1:]) + '\n' + sng_bar
                   + f'\nSpherical errors, t-distribution with {self.df_resid} df'
                   + f'\nMessage: {self.message}'
                   + '\n' + self.get_version_str(width)
                   )
        return summary
