from __future__ import absolute_import, print_function

import pprint

from kanly.dill_object import DillObject


class OptimizationResult(DillObject):
    """Container for optimization outputs shared by solver implementations.

    The result object keeps the final objective_function_ value, parameter vector,
    convergence diagnostics, original callable, and options used to run an
    optimizer. Subclasses add solver-specific diagnostics such as Hessian
    approximations or binding-bound indicators.
    """

    def __init__(self, fun, x, num_params, grad, grad_projected, converged, message, time_elapsed, fun_callable,
                 ferr, xerr, iter, bounds, options, maximize, optimization_path):
        """Create an optimization result record.

        Args:
            fun: Final objective_function_ value, transformed back to the user's
                maximize/minimize convention.
            x: Final parameter vector.
            num_params: Number of optimized parameters.
            grad: Gradient at the final parameter vector.
            grad_projected: Projected gradient with entries blocked by active
                bounds set to zero.
            converged: Whether the optimizer met a convergence criterion.
            message: Human-readable convergence or failure message.
            time_elapsed: Wall-clock runtime in seconds.
            fun_callable: Original objective_function_ callable supplied by the user.
            ferr: Final relative objective_function_-change diagnostic.
            xerr: Final relative parameter-change diagnostic.
            iter: Number of optimizer iterations completed.
            bounds: Bounds used by the optimizer, or None for unbounded runs.
            options: Dictionary of solver options and starting values.
            maximize: Whether the user requested maximization.
            optimization_path: Optional list of per-iteration state snapshots.
        """
        self.fun = fun
        self.x = x
        self.num_params = num_params
        self.grad = grad
        self.grad_projected = grad_projected
        self.converged = converged
        self.message = message
        self.time_elapsed = time_elapsed
        self.fun_callable = fun_callable
        self.ferr = ferr
        self.xerr = xerr
        self.iter = iter
        self.bounds = bounds
        self.options = options
        self.maximize = maximize
        self.optimization_path = optimization_path

    def __str__(self):
        """Return a pretty-printed representation of stored result fields."""
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        """Return the same string representation used by ``str(result)``."""
        return str(self)

    def to_string(self, keys=None):
        """Return a pretty-printed subset of result fields.

        Args:
            keys: Optional iterable of field names to include. When omitted,
                all fields are included.

        Returns:
            A formatted string containing either all result fields or the
            requested subset. Missing requested keys are shown with value None.
        """
        if keys is None:
            return str(self)
        else:
            self_dict = self.__dict__
            return pprint.pformat({k: self_dict.get(k, None) for k in keys})
