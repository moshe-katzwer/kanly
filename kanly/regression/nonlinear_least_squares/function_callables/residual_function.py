from __future__ import absolute_import, print_function

import warnings
from typing import Callable

import numpy as np
from scipy.sparse import isspmatrix

from kanly.regression.nonlinear_least_squares.function_callables.prediction_function import PredictionFunction
from kanly.utils.linalg_utils import DEFAULT_DENSE_THRESHOLD_MB

from kanly.automatic_differentiation.elementary_functions import *  # don't delete
from kanly.dill_object import DillObject


class ResidualFunction(DillObject):
    """Callable residual wrapper for nonlinear least-squares models.

    Combines a prediction function ``f(params)`` with an endogenous vector
    ``y`` and returns residuals ``y - f(params)``.  Jacobians are the negative
    of prediction Jacobians, matching the residual definition.
    """

    def __init__(self, prediction_function_callable, endog, num_params=None):
        """Initialise a residual function.

        Args:
            prediction_function_callable: ``PredictionFunction`` or compatible
                callable that maps parameters to fitted values.
            endog: Observed response vector.
            num_params: Required only when ``prediction_function_callable`` does
                not expose ``num_params``.
        """
        if not isinstance(prediction_function_callable, PredictionFunction):
            warnings.warn("Not using a `PredictionFunction` object!")

        if hasattr(prediction_function_callable, 'num_params'):
            self.num_params = prediction_function_callable.num_params
        else:
            assert num_params is not None and isinstance(num_params, int) and num_params > 0
            self.num_params = num_params
        assert isinstance(prediction_function_callable, Callable)

        self.prediction_function_callable = prediction_function_callable

        self.endog = np.asarray(endog)
        self.nobs = len(endog)

    def copy(self, copy_data=True):
        """Return a copy of the residual function and optionally its data.

        Args:
            copy_data: When ``True``, copy the endogenous vector and prediction
                function data arrays; otherwise share them.

        Returns:
            New ``ResidualFunction`` instance.
        """
        prediction_function_copy = self.prediction_function_callable.copy(copy_data=copy_data)
        if copy_data:
            endog_copy = self.endog.copy()
        else:
            endog_copy = self.endog
        return ResidualFunction(prediction_function_copy, endog_copy)

    def __call__(self, params, idx=None):
        """Evaluate residuals at ``params``.

        Args:
            params: Parameter vector.
            idx: Optional row indexer for returning a subset.

        Returns:
            Residual vector ``endog - prediction``.
        """
        return self._residuals(params, idx=idx)

    def _residuals(self, params, idx=None):
        """Compute residuals ``y - f(params)`` with optional row subsetting.

        Args:
            params: Parameter vector.
            idx: Optional row indexer.

        Returns:
            NumPy residual vector.
        """
        if idx is not None:
            f = self.prediction_function_callable(params)[idx]
        else:
            f = self.prediction_function_callable(params)
        values = self.get_endog(idx=idx) - f
        return values

    def get_endog(self, idx=None):
        """Return the endogenous vector or a subset of it.

        Args:
            idx: Optional row indexer.

        Returns:
            ``self.endog`` or ``self.endog[idx]``.
        """
        if idx is None:
            return self.endog
        else:
            return self.endog[idx]

    def jacobian(self, params, f0=None, idx=None):
        """Evaluate the residual Jacobian at ``params``.

        Args:
            params: Parameter vector.
            f0: Optional pre-computed residual baseline.
            idx: Optional row indexer.

        Returns:
            Dense or sparse Jacobian of residuals with respect to parameters.
        """
        return self._jacobian(params, f0=f0, idx=idx)

    def _jacobian(self, params, f0=None, idx=None):
        """Compute ``-prediction_jacobian`` for the residual definition.

        Args:
            params: Parameter vector.
            f0: Optional pre-computed residual baseline.
            idx: Optional row indexer.

        Returns:
            Residual Jacobian with sign flipped from prediction Jacobian.
        """
        jac = self.prediction_function_callable.jacobian(params, f0=f0, idx=idx)
        if isspmatrix(jac):
            jac.data = -jac.data
        else:
            jac *= -1
        return jac

    def reindex(self, idx, inplace=False):
        """Subset residual data to a row indexer.

        Args:
            idx: Row indexer or boolean mask.
            inplace: When ``True``, mutate this object; otherwise return a new
                reindexed ``ResidualFunction``.

        Returns:
            ``None`` when ``inplace=True``; otherwise a reindexed copy.
        """
        if inplace:
            self.endog = self.endog[idx]
            self.prediction_function_callable.reindex(idx, inplace=True)
            self.nobs = len(self.endog)

        else:
            return ResidualFunction(self.prediction_function_callable.reindex(idx, inplace=False), self.endog[idx])

    def get_analytical_jacobian(self, dense_threshold_mb=DEFAULT_DENSE_THRESHOLD_MB, debug=False, do_jit=True):
        """
        :return: jacobian callable, dict of info

        Args:
            dense_threshold_mb: Dense Jacobian threshold in MB.
            debug: Whether to print analytic-Jacobian diagnostics.
            do_jit: Whether to JIT compile generated analytic Jacobian code.
        """
        temp_jac_func, jac_result = self.prediction_function_callable.get_analytical_jacobian(
            dense_threshold_mb=dense_threshold_mb, debug=debug, do_jac_jit=do_jit)
        jac_func = lambda x: -temp_jac_func(x)
        return jac_func, jac_result

    def get_analytical_partial_derivative(self, arg_number, debug=False, do_jit=True, return_info=False):
        """
        :return: partial derivative callable (negative of pred func partial), dict of info

        Args:
            arg_number: Parameter index to differentiate with respect to.
            debug: Whether to print automatic-differentiation diagnostics.
            do_jit: Whether to JIT compile generated derivative code.
            return_info: Whether to return derivative metadata.
        """
        result = self.prediction_function_callable.get_analytical_partial_derivative(
            arg_number, debug=debug, do_jit=do_jit)
        if return_info:
            return lambda x: -result[0](x), result[1]
        else:
            return lambda x: -result(x)

    def get_analytical_partial_derivatives(self, debug=False, do_jit=True, return_info=False):
        """
        :return: partial derivative callables (negative of pred func partial), dict of info

        Args:
            debug: Whether to print automatic-differentiation diagnostics.
            do_jit: Whether to JIT compile generated derivative code.
            return_info: Whether to return derivative metadata.
        """
        results = [self.get_analytical_partial_derivative(i, debug, do_jit=do_jit, return_info=return_info)
                   for i in range(self.num_params)]
        if return_info:
            return [r[0] for r in results], [r[1] for r in results]
        else:
            return results
