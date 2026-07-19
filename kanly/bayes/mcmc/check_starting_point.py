"""Validation utilities for MCMC initial states."""

from __future__ import absolute_import, print_function

import time

import numpy as np


def check_starting_point(x0s, log_posterior_transformed, log_posterior_jacobian_adjustment=None, debug=False):
    """Validate that starting points yield finite scalar log-density values.

    This also eagerly calls the provided functions, which can trigger one-time numba
    compilation before the main sampler loop starts.

    Args:
        x0s: One start vector or a list/array of start vectors (one per chain).
        log_posterior_transformed: Callable evaluating log-posterior on sampling scale.
        log_posterior_jacobian_adjustment: Optional Jacobian-adjustment callable.
        debug: If True, print timing/progress messages.
    """
    _t = time.time()
    if debug:
        print("Checking starting points for validity (and possibly doing `numba` compilation)...", end="")
    if np.ndim(x0s) == 1:
        x0s = [x0s]
    lp_vals = [log_posterior_transformed(x) for x in x0s]
    if not np.all([isinstance(l, (float, int)) for l in lp_vals]):
        raise Exception('log_posterior function not returning a scalar value!')
    if np.any(~np.isfinite(lp_vals)):
        raise Exception("Some log posterior values are infinite or nan at starting value!")
    if log_posterior_jacobian_adjustment is not None:
        jac_vals = [log_posterior_jacobian_adjustment(x) for x in x0s]
        if np.any(~np.isfinite(jac_vals)):
            raise Exception("Some jacobian adjustment values are infinite or nan at starting value!")

    if debug:
        print(f"({time.time()-_t:0.2f}s)\n")
