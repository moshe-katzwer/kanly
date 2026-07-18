"""
Functions for manipulating lag and differencing coefficients
as polynomials
"""

from __future__ import absolute_import, print_function

import numpy as np
from numpy.polynomial import Polynomial
from scipy.special import binom


def lag_2_polynomial(coefs, s=1) -> Polynomial:
    """
    Convert lag coefficients into a polynomial representation.

    Given coefficients ``[phi_1, phi_2, ...]`` and seasonal stride ``s``,
    constructs the polynomial ``1 - phi_1 * L^s - phi_2 * L^{2s} - ...`` by
    composing a linear polynomial with a degree-``s`` monomial.  The result is
    the standard lag-operator polynomial used to form AR and MA filters.

    Args:
        coefs: Lag coefficient vector.
        s: Seasonal period length.

    Returns:
        ``numpy.polynomial.Polynomial`` representing the lag operator polynomial
        ``1 - coefs[0]*L^s - coefs[1]*L^{2s} - ...``.  Returns
        ``Polynomial([1.0])`` when ``coefs`` is empty.
    """
    if len(coefs):
        temp = Polynomial(np.hstack([1, -np.array(coefs)]))
        temp = temp(Polynomial([0.] * s + [1]))
        return temp
    else:
        return Polynomial([1.0])


def polynomial_2_coef(poly: Polynomial) -> np.array:
    """
    Convert a polynomial representation back to lag coefficients.

    Extracts the lag coefficients as ``-poly.coef[1:]``, reversing the sign
    convention used by ``lag_2_polynomial`` where the polynomial is
    ``1 - phi_1*L - phi_2*L^2 - ...``.

    Args:
        poly: Polynomial object to convert.

    Returns:
        1-D NumPy array of lag coefficients ``[phi_1, phi_2, ...]``.
    """
    return -np.array(poly.coef[1:])


def get_differencing_polynomial(d: int, s: int = 1) -> Polynomial:
    """
    Build a nonseasonal or seasonal differencing polynomial.

    Constructs the polynomial ``(1 - L^s)^d`` using the binomial expansion,
    then substitutes ``L^s`` for ``L`` by composing with a degree-``s``
    monomial.  For ``d=0`` the identity polynomial ``1`` is returned.
    Setting ``s > 1`` produces a seasonal differencing polynomial such as
    ``(1 - L^4)^1`` for quarterly seasonal differencing.

    Args:
        d: Differencing order.
        s: Lag stride (1 for nonseasonal, >1 for seasonal).

    Returns:
        ``numpy.polynomial.Polynomial`` encoding the operator ``(1 - L^s)^d``.
    """
    assert d >= 0
    assert s >= 1
    if d == 0:
        return Polynomial([1.0])
    else:
        bv = binom(d, range(0, d + 1)) * ([1, -1] * d)[:d + 1]
        poly = Polynomial(bv)
        poly = poly(Polynomial([0] * s + [1]))
        return poly


def get_combined_differencing_polynomial(d, D=None, s=2):
    """
    Build the combined nonseasonal and seasonal differencing polynomial.

    Multiplies the nonseasonal differencing polynomial ``(1 - L)^d`` by the
    seasonal differencing polynomial ``(1 - L^s)^D``.  If ``D`` is ``None``
    or 0 only the nonseasonal polynomial is returned.

    Args:
        d: Nonseasonal differencing order.
        D: Seasonal differencing order (``None`` or 0 means no seasonal differencing).
        s: Seasonal period length; must be >= 2 when D >= 1.

    Returns:
        ``numpy.polynomial.Polynomial`` encoding the combined operator
        ``(1 - L)^d * (1 - L^s)^D``.
    """
    assert d >= 0
    poly = get_differencing_polynomial(d)
    if D is None or D == 0:
        return poly
    else:
        assert D >= 1
        assert s >= 2
        poly = poly * get_differencing_polynomial(D, s)
        return poly


def get_combined_differencing_coefs(d, D, s):
    """
    Return combined differencing coefficients.

    Convenience wrapper that converts the combined differencing polynomial
    ``(1 - L)^d * (1 - L^s)^D`` to a 1-D coefficient array via
    ``polynomial_2_coef``.

    Args:
        d: Nonseasonal differencing order.
        D: Seasonal differencing order.
        s: Seasonal period length.

    Returns:
        1-D NumPy array of lag coefficients for the combined differencing filter.
    """
    return polynomial_2_coef(get_combined_differencing_polynomial(d, D, s))


def get_differencing_coefs(d: int, s: int = 1):
    """
    Return coefficients for a differencing polynomial.

    Convenience wrapper around ``get_differencing_polynomial`` + ``polynomial_2_coef``
    that returns the lag coefficients for the single-component operator ``(1 - L^s)^d``
    without constructing the combined seasonal-nonseasonal polynomial.

    Args:
        d: Differencing order.
        s: Lag stride (1 for nonseasonal, >1 for seasonal).

    Returns:
        1-D NumPy array of lag coefficients for the differencing filter.
    """
    poly = get_differencing_polynomial(d, s)
    return polynomial_2_coef(poly)


def combine_lag_coefs_2_polynomial(*args) -> Polynomial:
    """
    Multiply multiple lag coefficient polynomials.

    Each element of ``args`` is a ``(coefs, s)`` tuple compatible with
    ``lag_2_polynomial``.  The function converts each to a Polynomial, takes
    their product, and pads the result to the expected total degree so that
    coefficient indexing remains unambiguous even when cancellations reduce
    the actual degree.

    Returns:
        ``numpy.polynomial.Polynomial`` that is the product of all supplied
        lag polynomials, zero-padded to the sum of their individual degrees.
    """
    polys = [lag_2_polynomial(*a) for a in args]
    result: Polynomial = np.prod(polys)
    deg = sum(len(a[0]) for a in args)
    if len(result.coef) == deg + 1:
        return result
    else:
        return Polynomial(np.hstack([result.coef, [0.0] * (deg + 1 - len(result.coef))]))


def combine_lag_coefs(*args):
    """
    Combine lag coefficient vectors into one coefficient vector.

    Convenience wrapper that calls ``combine_lag_coefs_2_polynomial`` and
    extracts the resulting coefficients via ``polynomial_2_coef``.  Use this
    to merge, for example, a nonseasonal AR vector with a seasonal AR vector
    into a single full-length lag-coefficient array.

    Returns:
        1-D NumPy array of lag coefficients for the combined lag polynomial.
    """
    return polynomial_2_coef(combine_lag_coefs_2_polynomial(*args))


def check_intersection(coefs, seasonal_coefs, s):
    """Make sure seasonal and regular lags don't share lag terms.

    Args:
        coefs: Nonseasonal lag coefficient vector.
        seasonal_coefs: Seasonal lag coefficient vector.
        s: Seasonal period length.

    Raises:
        Exception: If any nonzero nonseasonal lag also appears as a nonzero
            seasonal lag after multiplying by ``s``.
    """
    assert s >= 1
    intersections = (
            set([i + 1 for i, c in enumerate(coefs) if c != 0])
            & set([s * (i + 1) for i, c in enumerate(seasonal_coefs) if c != 0])
    )
    if len(intersections):
        raise Exception(f"Invalid Specification: lag(s) {intersections} appear in both"
                        f" seasonal and non-seasonal terms!")

# #
# if __name__ == '__main__':
#     print(combine_lag_coefs(
#         ([0, 0], 1),
#         ([0, 0], 1)
#     ))
