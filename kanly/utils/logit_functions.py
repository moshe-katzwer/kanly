from __future__ import absolute_import, print_function

import numpy as np
from numba.extending import overload


def logit(u):
    """Compute the logit (log-odds) of ``u``.

    The logit function maps a probability ``u ∈ (0, 1)`` to the real line:
    ``logit(u) = log(u / (1 - u))``.  It is the inverse of the sigmoid
    (expit) function.

    Args:
        u: Scalar or array-like of probability values strictly in ``(0, 1)``.

    Returns:
        Log-odds value(s) in ``(-∞, +∞)``.
    """
    return np.log(u / (1 - u))


def expit(v):
    """Compute the sigmoid (logistic) function of ``v``.

    Maps any real number to ``(0, 1)``:
    ``expit(v) = 1 / (1 + exp(-v))``.
    Inverse of ``logit``.

    Args:
        v: Scalar or array-like of real values.

    Returns:
        Value(s) in ``(0, 1)``.
    """
    return 1.0 / (1.0 + np.exp(-v))


def d_expit(v):
    """Compute the derivative of the sigmoid (expit) function.

    ``d/dv expit(v) = expit(v) * (1 - expit(v))``.  Used as the Jacobian
    factor when transforming parameters constrained to ``(0, 1)``.

    Args:
        v: Scalar or array-like of real values.

    Returns:
        Derivative value(s); always in ``(0, 0.25]``.
    """
    val = expit(v)
    return val * (1.0 - val)


def log_d_expit(v):
    """Compute the log of the derivative of the sigmoid function.

    ``log(d/dv expit(v)) = log(expit(v)) + log(1 - expit(v))``.
    Used as the log-Jacobian adjustment when working in transformed
    (unbounded) parameter space.

    Args:
        v: Scalar or array-like of real values.

    Returns:
        Log-derivative value(s); always in ``(-∞, -log(4)]``.
    """
    val = expit(v)
    return np.log(val) + np.log(1.0 - val)


@overload(logit)
def logit_impl(u):
    """Numba overload of ``logit`` enabling use inside ``@njit`` functions."""
    return logit


@overload(d_expit)
def d_expit_impl(v):
    """Numba overload of ``d_expit`` enabling use inside ``@njit`` functions."""
    return d_expit


@overload(expit)
def expit_impl(v):
    """Numba overload of ``expit`` enabling use inside ``@njit`` functions."""
    return expit


@overload(log_d_expit)
def log_d_expit_impl(v):
    """Numba overload of ``log_d_expit`` enabling use inside ``@njit`` functions."""
    return log_d_expit
