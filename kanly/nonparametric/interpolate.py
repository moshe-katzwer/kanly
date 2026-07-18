"""Piecewise polynomial interpolation in 1-D.

Provides a unified :func:`interp` factory that constructs an
:class:`Interpolator1d` object for one of several interpolation kinds:

- **cubic** (default): C2-continuous cubic spline with selectable boundary
  conditions (not-a-knot, natural, clamped).
- **quadratic**: C1-continuous quadratic spline.
- **linear**: piecewise-linear (C0).
- **nearest** / **previous** / **next**: step interpolants.

Convenience wrappers :func:`cubic_spline`, :func:`quadratic_spline`, and
:func:`linear_spline` are also provided.

The heavy inner-loop numerics are JIT-compiled with Numba (``@njit``) for
performance.  The :class:`Interpolator1d` result object exposes
:meth:`~Interpolator1d.derivative`, :meth:`~Interpolator1d.derivative2`, and
:meth:`~Interpolator1d.derivative3` methods in addition to direct evaluation.
"""
from __future__ import absolute_import, print_function

from numba import njit
import numpy as np

from kanly.dill_object import DillObject

LINEAR = 'linear'
CUBIC = 'cubic'
QUADRATIC = 'quadratic'
NEAREST = 'nearest'
PREVIOUS = 'previous'
NEXT = 'next'

BC_TYPE_NOT_A_KNOT = 'not-a-knot'
BC_TYPE_CLAMPED = 'clamped'
BC_TYPE_NATURAL = 'natural'
BC_TYPES = (BC_TYPE_CLAMPED, BC_TYPE_NATURAL, BC_TYPE_NOT_A_KNOT)
DEFAULT_BC_TYPE = BC_TYPE_NOT_A_KNOT


@njit(cache=True)
def _get_quadratic_spline_coeffs(x, y):
    """
    Constructs a quadratic spline through given points (x, y).

    Parameters:
        x (numpy array): x-coordinates of data points.
        ynew (numpy array): y-coordinates of data points.

    Returns:
        coeffs (list of tuples): [(a_i, b_i, c_i)] for each interval [x_i, x_{i+1}].
    """

    k = len(x)
    xnew_ = np.array([(x[i + 1] + x[i + 2]) / 2 for i in range(k - 3)])
    xnnew_ = np.zeros(k - 1)
    xnnew_[0] = x[0]
    xnnew_[-1] = x[-1]
    xnnew_[1:-1] = xnew_
    b = np.zeros(3 * k - 6)
    b[:len(y)] = y
    A = np.zeros((3 * k - 6, 3 * k - 6))
    A[0][0] = A[1][0] = 1
    A[0][1] = x[0]
    A[0][2] = x[0] ** 2
    A[1][1] = x[1]
    A[1][2] = x[1] ** 2
    A[k - 2][3 * k - 9] = A[k - 1][3 * k - 9] = 1
    A[k - 2][3 * k - 8] = x[k - 2]
    A[k - 2][3 * k - 7] = x[k - 2] ** 2
    A[k - 1][3 * k - 8] = x[k - 1]
    A[k - 1][3 * k - 7] = x[k - 1] ** 2
    for i in range(2, k - 2):
        A[i][3 * (i - 1)] = 1
        A[i][3 * (i - 1) + 1] = x[i]
        A[i][3 * (i - 1) + 2] = x[i] ** 2
    for i in range(k - 3):
        A[k + i][3 * i] = 1
        A[k + i][3 * i + 3] = -1
        A[k + i][3 * i + 1] = xnew_[i]
        A[k + i][3 * i + 4] = -xnew_[i]
        A[k + i][3 * i + 2] = xnew_[i] ** 2
        A[k + i][3 * i + 5] = -1 * xnew_[i] ** 2
        A[k + i + k - 3][3 * i + 1] = 1
        A[k + i + k - 3][3 * i + 4] = -1
        A[k + i + k - 3][3 * i + 2] = 2 * xnew_[i]
        A[k + i + k - 3][3 * i + 5] = -2 * xnew_[i]

    Coef = np.linalg.solve(A, b)
    Coef = Coef.reshape((k - 2, 3))
    Coeffs = np.zeros((k - 2, 3))
    for j in range(3):
        Coeffs[:, j] = Coef[:, 2 - j]
    grid = xnnew_

    return Coeffs, grid


# TODO remove
# @njit
# def _get_cubic_spline_coeffs_OLD(x, y):
#     """
#     Constructs a natural cubic spline interpolating the given points (x, y).

#     Parameters:
#         x (numpy array): x-coordinates of data points.
#         y (numpy array): y-coordinates of data points.

#     Returns:
#         spline_function (callable): Function to evaluate the spline at any x.
#     """
#     num_intervals = len(x) - 1  # Number of intervals
#     h = np.diff(x)              # Interval widths

#     # Step 1: Compute coefficients for the second derivative system
#     A = np.zeros((num_intervals + 1, num_intervals + 1))  # Coefficient matrix
#     b = np.zeros(num_intervals + 1)  # Right-hand side

#     # Natural spline boundary conditions (second derivative at endpoints = 0)
#     A[0, 0] = A[-1, -1] = 1  # Natural spline
#     b[0] = b[-1] = 0

#     # Fill system for second derivative continuity
#     for i in range(1, num_intervals):
#         A[i, i - 1] = h[i - 1] / 6
#         A[i, i] = (h[i - 1] + h[i]) / 3
#         A[i, i + 1] = h[i] / 6
#         b[i] = (y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]

#     # Solve for second derivatives (M values)
#     M = np.linalg.solve(A, b)

#     # Compute spline coefficients
#     coeffs = np.zeros((num_intervals, 5))
#     for i in range(num_intervals):
#         a_i = (M[i + 1] - M[i]) / (6 * h[i])
#         b_i = M[i] / 2
#         c_i = (y[i + 1] - y[i]) / h[i] - (2 * h[i] * M[i] + h[i] * M[i + 1]) / 6
#         d_i = y[i]
#         coeffs[i] = (a_i, b_i, c_i, d_i, x[i])

#     return coeffs


def _get_cubic_spline_coeffs(x, y, bc_type=DEFAULT_BC_TYPE, clamped_slopes=None):
    """Python wrapper that validates arguments before delegating to the Numba kernel.

    Normalises ``bc_type`` to lower-case, supplies default ``clamped_slopes``,
    and forwards to :func:`_get_cubic_spline_coeffs_njit`.

    Args:
        x: 1-D array of strictly-increasing knot x-coordinates.
        y: 1-D array of function values at ``x``, same length as ``x``.
        bc_type: Boundary condition type string – one of
            ``'not-a-knot'`` (default), ``'natural'``, or ``'clamped'``.
        clamped_slopes: ``(left_slope, right_slope)`` used when
            ``bc_type='clamped'``. Defaults to ``[0.0, 0.0]``.

    Returns:
        Tuple ``(coeffs, grid)`` where ``coeffs`` is an ``(n-1, 4)`` array
        of polynomial coefficients (descending powers) per segment, and
        ``grid`` is the ``x`` array used as the breakpoint sequence.
    """
    bc_type = bc_type.lower()
    assert bc_type in BC_TYPES
    if clamped_slopes is None:
        clamped_slopes = np.array([0.0, 0.0])
    else:
        clamped_slopes = np.asarray(clamped_slopes, dtype=np.float64)
    return _get_cubic_spline_coeffs_njit(x, y, bc_type, clamped_slopes)


@njit(cache=True)
def _get_cubic_spline_coeffs_njit(x, y, bc_type, clamped_slopes):
    """
    Compute cubic spline interpolation with specified boundary conditions.

    Parameters:
    x : array_like, shape (n,)
        Independent variable values (must be strictly increasing)
    y : array_like, shape (n,)
        Dependent variable values
    bc_type : str, optional
        Boundary condition type:
        'not-a-knot' (default), 'natural', or 'clamped'
    clamped_slopes : tuple, optional
        (left_slope, right_slope) for clamped boundary conditions.
        If None and bc_type='clamped', uses (0, 0) for natural clamped spline.

    Returns:
    callable
        A function that computes the interpolated value at any point
    """
    n = len(x)

    # Check inputs
    if len(y) != n:
        raise Exception("x and y must have the same length")
    if bc_type not in ['not-a-knot', 'natural', 'clamped']:
        raise Exception("bc_type must be 'not-a-knot', 'natural', or 'clamped'")
    if bc_type == 'not-a-knot' and n < 3:
        raise Exception("At least 3 points are needed for not-a-knot spline")
    if bc_type == 'clamped' and n < 2:
        raise Exception("At least 2 points are needed for clamped spline")

    # Calculate differences between points
    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("x must be strictly increasing")

    # Calculate slopes
    delta = np.diff(y) / h

    # Set up the tridiagonal system for second derivatives
    A = np.zeros((n, n))
    b = np.zeros(n)

    # Internal points (same for all boundary conditions)
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b[i] = 3 * (delta[i] - delta[i - 1])

    if bc_type == BC_TYPE_NOT_A_KNOT:
        # Not-a-knot boundary conditions
        if n >= 3:
            # First condition: third derivative continuous at x[1]
            A[0, 0] = h[1]
            A[0, 1] = -(h[0] + h[1])
            A[0, 2] = h[0]
            b[0] = 0

            # Second condition: third derivative continuous at x[-2]
            A[-1, -3] = h[-1]
            A[-1, -2] = -(h[-2] + h[-1])
            A[-1, -1] = h[-2]
            b[-1] = 0
    elif bc_type == BC_TYPE_NATURAL:
        # Natural spline boundary conditions (second derivative = 0 at ends)
        A[0, 0] = 1
        b[0] = 0

        A[-1, -1] = 1
        b[-1] = 0
    elif bc_type == BC_TYPE_CLAMPED:
        # Left endpoint condition
        A[0, 0] = 2 * h[0]
        A[0, 1] = h[0]
        b[0] = 3 * (delta[0] - clamped_slopes[0])

        # Right endpoint condition
        A[-1, -2] = h[-1]
        A[-1, -1] = 2 * h[-1]
        b[-1] = 3 * (clamped_slopes[1] - delta[-1])

    # Solve for second derivatives
    c_prime = np.linalg.solve(A, b)

    # Calculate spline coefficients
    # d x**2 + c x**3 + b x + a
    a = y[:-1]
    b = delta - h * (2 * c_prime[:-1] + c_prime[1:]) / 3
    c = c_prime[:-1]
    d = (c_prime[1:] - c_prime[:-1]) / (3 * h)

    coeffs = np.zeros((n - 1, 4))
    coeffs[:, 0] = d
    coeffs[:, 1] = c
    coeffs[:, 2] = b
    coeffs[:, 3] = a

    grid = x
    return coeffs, grid


@njit(cache=True)
def _get_segment(x_values, x, assume_sorted=False):
    """Find the left-interval index for each query point.

    For each element of ``x_values``, returns the index ``l`` such that
    ``x[l] <= x_values[i] < x[l+1]``.  A linear scan is used after
    optionally sorting the queries so that the walk through ``x`` never
    needs to backtrack.

    Args:
        x_values: 1-D array of query x-coordinates.
        x: 1-D sorted array of breakpoint coordinates (the grid interior, i.e.
           ``grid[:-1]`` for most splines).
        assume_sorted: If ``True`` the queries are assumed already sorted and
            the sort step is skipped.

    Returns:
        Tuple ``(left, index)`` where ``left[i]`` is the segment index for
        query ``x_values[i]``, and ``index`` is the permutation used to walk
        the queries in sorted order.
    """
    N = len(x_values)
    n = len(x)
    if assume_sorted:
        index = np.arange(N)
    else:
        index = np.argsort(x_values)
    left = np.zeros(N, dtype=np.int32)
    l = 0
    for i in index:
        xval = x_values[i]
        while l < n - 1 and x[l + 1] < xval:
            l += 1
        left[i] = l
    return left, index


@njit(cache=True)
def _evaluate_cubic_spline(x_values, coeffs, grid, assume_sorted=False):
    """Evaluate a cubic spline at arbitrary query points.

    Uses the local offset ``dx = x - x_left`` and Horner's scheme:
    ``d*dx^3 + c*dx^2 + b*dx + a``.

    Args:
        x_values: 1-D array of query x-coordinates.
        coeffs: ``(n-1, 4)`` coefficient array in descending-power order
            ``[d, c, b, a]`` per segment.
        grid: 1-D knot array of length ``n``; segment ``i`` covers
            ``[grid[i], grid[i+1])``.
        assume_sorted: Passed to :func:`_get_segment`.

    Returns:
        1-D array of spline values, same length as ``x_values``.
    """
    y_values = np.zeros(x_values.shape)
    left, index = _get_segment(x_values, grid[:-1], assume_sorted=assume_sorted)

    for i in index:
        dx = x_values[i] - grid[left[i]]
        y_values[i] = (
                +coeffs[left[i], 0] * dx ** 3
                + coeffs[left[i], 1] * dx ** 2
                + coeffs[left[i], 2] * dx
                + coeffs[left[i], 3]
        )

    return y_values


@njit(cache=True)
def _evaluate_cubic_spline_deriv(x_values, coeffs, grid, assume_sorted=False):
    """Evaluate the first derivative of a cubic spline at query points.

    Analytically differentiates the cubic: ``3d*dx^2 + 2c*dx + b``.

    Args:
        x_values: 1-D array of query x-coordinates.
        coeffs: ``(n-1, 4)`` coefficient array ``[d, c, b, a]`` per segment.
        grid: 1-D knot array used to locate segments.
        assume_sorted: Passed to :func:`_get_segment`.

    Returns:
        1-D array of first-derivative values.
    """
    y_values = np.zeros(x_values.shape)
    left, index = _get_segment(x_values, grid[:-1], assume_sorted=assume_sorted)

    for i in index:
        dx = x_values[i] - grid[left[i]]
        y_values[i] = (
                + 3 * coeffs[left[i], 0] * dx ** 2
                + 2 * coeffs[left[i], 1] * dx
                + coeffs[left[i], 2]
        )

    return y_values


@njit(cache=True)
def _evaluate_cubic_spline_deriv2(x_values, coeffs, grid, assume_sorted=False):
    """Evaluate the second derivative of a cubic spline at query points.

    Analytically differentiates twice: ``6d*dx + 2c``.

    Args:
        x_values: 1-D array of query x-coordinates.
        coeffs: ``(n-1, 4)`` coefficient array ``[d, c, b, a]`` per segment.
        grid: 1-D knot array used to locate segments.
        assume_sorted: Passed to :func:`_get_segment`.

    Returns:
        1-D array of second-derivative values.
    """
    y_values = np.zeros(x_values.shape)
    left, index = _get_segment(x_values, grid[:-1], assume_sorted=assume_sorted)

    for i in index:
        dx = x_values[i] - grid[left[i]]
        y_values[i] = (
                + 6 * coeffs[left[i], 0] * dx
                + 2 * coeffs[left[i], 1]
        )

    return y_values


@njit(cache=True)
def _evaluate_cubic_spline_deriv3(x_values, coeffs, grid, assume_sorted=False):
    """Evaluate the (constant) third derivative of a cubic spline.

    For a cubic polynomial ``d*dx^3 + ...`` the third derivative is ``6d``
    (piecewise constant per segment).

    Args:
        x_values: 1-D array of query x-coordinates.
        coeffs: ``(n-1, 4)`` coefficient array ``[d, c, b, a]`` per segment.
        grid: 1-D knot array used to locate segments.
        assume_sorted: Passed to :func:`_get_segment`.

    Returns:
        1-D array of third-derivative values.
    """
    y_values = np.zeros(x_values.shape)
    left, index = _get_segment(x_values, grid[:-1], assume_sorted=assume_sorted)

    for i in index:
        y_values[i] = 6 * coeffs[left[i], 0]

    return y_values


@njit(cache=True)
def _evaluate_quadratic_spline(x_values, coeffs, grid, assume_sorted=False):
    """Evaluate a quadratic spline at query points using raw x coordinates.

    Note: the quadratic spline stores global (non-local) coefficients
    ``[a, b, c]`` such that the polynomial is ``a*x^2 + b*x + c`` (not
    offset by the segment knot).

    Args:
        x_values: 1-D array of query x-coordinates.
        coeffs: ``(n-1, 3)`` coefficient array ``[a, b, c]`` per segment.
        grid: 1-D knot array used to locate segments.
        assume_sorted: Passed to :func:`_get_segment`.

    Returns:
        1-D array of spline values.
    """
    y_values = np.zeros(x_values.shape)
    left, index = _get_segment(x_values, grid[:-1], assume_sorted=assume_sorted)

    for i in index:
        xval = x_values[i]
        y_values[i] = (
                + coeffs[left[i], 0] * xval ** 2
                + coeffs[left[i], 1] * xval
                + coeffs[left[i], 2]
        )

    return y_values


@njit(cache=True)
def _evaluate_quadratic_spline_deriv(x_values, coeffs, grid, assume_sorted=False):
    """Evaluate the first derivative of a quadratic spline.

    Differentiating ``a*x^2 + b*x + c`` gives ``2a*x + b``.

    Args:
        x_values: 1-D array of query x-coordinates.
        coeffs: ``(n-1, 3)`` coefficient array ``[a, b, c]`` per segment.
        grid: 1-D knot array used to locate segments.
        assume_sorted: Passed to :func:`_get_segment`.

    Returns:
        1-D array of first-derivative values.
    """
    y_values = np.zeros(x_values.shape)
    left, index = _get_segment(x_values, grid[:-1], assume_sorted=assume_sorted)

    for i in index:
        xval = x_values[i]
        y_values[i] = (
                + 2.0 * coeffs[left[i], 0] * xval
                + coeffs[left[i], 1]
        )

    return y_values


@njit(cache=True)
def _evaluate_quadratic_spline_deriv2(x_values, coeffs, grid, assume_sorted=False):
    """Evaluate the (constant) second derivative of a quadratic spline.

    The second derivative of ``a*x^2 + b*x + c`` is ``2a`` (constant per
    segment).

    Args:
        x_values: 1-D array of query x-coordinates.
        coeffs: ``(n-1, 3)`` coefficient array ``[a, b, c]`` per segment.
        grid: 1-D knot array used to locate segments.
        assume_sorted: Passed to :func:`_get_segment`.

    Returns:
        1-D array of second-derivative values.
    """
    y_values = np.zeros(x_values.shape)
    left, index = _get_segment(x_values, grid[:-1], assume_sorted=assume_sorted)

    for i in index:
        y_values[i] = 2.0 * coeffs[left[i], 0]

    return y_values


@njit(cache=True)
def _get_linear_spline_coeffs(x, y):
    """Compute piecewise-linear (hat-function) interpolation coefficients.

    Each segment ``i`` stores ``[slope, y_left]`` so that the value at
    a query ``t`` is ``y_left + slope * (t - x[i])``.

    Args:
        x: 1-D array of strictly-increasing knot x-coordinates (length ``n``).
        y: 1-D array of function values at ``x`` (length ``n``).

    Returns:
        Tuple ``(coeffs, grid)`` where ``coeffs`` is an ``(n-1, 2)`` array
        of ``[slope, y_left]`` per segment, and ``grid`` is ``x`` itself.
    """
    n = len(x) - 1
    coeffs = np.zeros((n, 2))
    for i in range(n):
        slope = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
        coeffs[i] = (slope, y[i])
    return coeffs, x


@njit(cache=True)
def _evaluate_linear_spline(x_values, coeffs, grid, assume_sorted=False):
    """Evaluate a piecewise-linear spline at query points.

    Computes ``y_left + slope * (x - x_left)`` for each query.

    Args:
        x_values: 1-D array of query x-coordinates.
        coeffs: ``(n-1, 2)`` coefficient array ``[slope, y_left]`` per segment.
        grid: 1-D knot array of length ``n``.
        assume_sorted: Passed to :func:`_get_segment`.

    Returns:
        1-D array of interpolated values.
    """
    y_values = np.zeros(x_values.shape)
    left, index = _get_segment(x_values, grid[:-1], assume_sorted=assume_sorted)
    for i in index:
        dx = x_values[i] - grid[left[i]]
        slope = coeffs[left[i], 0]
        a = coeffs[left[i], 1]
        y_values[i] = a + slope * dx

    return y_values


@njit(cache=True)
def _evaluate_linear_spline_deriv(x_values, coeffs, grid, assume_sorted=False):
    """Evaluate the (constant) first derivative of a piecewise-linear spline.

    The derivative within each segment is simply the stored slope, which is
    constant over the interval.

    Args:
        x_values: 1-D array of query x-coordinates.
        coeffs: ``(n-1, 2)`` coefficient array ``[slope, y_left]`` per segment.
        grid: 1-D knot array used to locate segments.
        assume_sorted: Passed to :func:`_get_segment`.

    Returns:
        1-D array of slope values (one per query point).
    """
    y_values = np.zeros(x_values.shape)
    left, index = _get_segment(x_values, grid[:-1], assume_sorted=assume_sorted)
    for i in index:
        y_values[i] = coeffs[left[i], 0]

    return y_values


@njit(cache=True)
def _zero_function(x_values, coeffs, grid, assume_sorted=False):
    """Return an array of zeros with the same shape as ``x_values``.

    Used as a placeholder derivative evaluator for interpolation kinds whose
    higher-order derivatives are identically zero (e.g., the third derivative
    of a quadratic spline, or any derivative of a step interpolant).

    Args:
        x_values: 1-D array of query x-coordinates (only the shape is used).
        coeffs: Ignored.
        grid: Ignored.
        assume_sorted: Ignored.

    Returns:
        Zero-filled array of the same shape as ``x_values``.
    """
    return np.zeros(x_values.shape)


def evaluate_spline(x_values, coeffs, grid, _eval_func, assume_sorted=False):
    """Evaluate a spline (or one of its derivatives) at arbitrary query points.

    Thin Python wrapper that coerces ``x_values`` to a float64 array and then
    calls the Numba-compiled ``_eval_func``.  Scalar inputs are returned as
    scalars rather than length-1 arrays.

    Args:
        x_values: Scalar or array-like of query x-coordinates.
        coeffs: Coefficient array appropriate for ``_eval_func``.
        grid: 1-D breakpoint array.
        _eval_func: Callable with signature
            ``(x_values, coeffs, grid, assume_sorted) -> y_values``.
        assume_sorted: If ``True`` the queries are assumed already sorted in
            ascending order (avoids an internal ``argsort``).

    Returns:
        Interpolated values as a scalar (if input is scalar/length-1) or a
        1-D NumPy array.
    """
    x_values = np.atleast_1d(x_values)
    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = _eval_func(x_values, coeffs, grid, assume_sorted=assume_sorted)
    if len(x_values) == 1:
        return y_values[0]
    else:
        return y_values


@njit(cache=True)
def _get_nearest_spline_coeffs(x, y):
    """Compute 'nearest-neighbour' step-interpolant coefficients.

    Each segment is centred on a knot: the grid breakpoints are placed at the
    midpoints between consecutive knots so that a query snaps to the closest
    knot value.

    Args:
        x: 1-D array of knot x-coordinates (length ``n``).
        y: 1-D array of function values at knots (length ``n``).

    Returns:
        Tuple ``(coeffs, grid)`` where ``coeffs`` is an ``(n, 1)`` array of
        knot values, and ``grid`` has length ``n+1`` with the first/last
        entries equal to ``x[0]``/``x[-1]`` and interior entries at
        midpoints ``(x[i] + x[i+1]) / 2``.
    """
    coeffs = y.reshape((-1, 1))
    # Build a grid whose internal breakpoints sit halfway between consecutive
    # knots, so each query maps to the nearest knot.
    grid = np.zeros(len(x) + 1)
    grid[0] = x[0]
    grid[-1] = x[-1]
    grid[1:-1] = (x[1:] + x[:-1]) / 2
    return coeffs, grid


@njit(cache=True)
def _get_previous_spline_coeffs(x, y):
    """Compute 'previous' step-interpolant coefficients.

    A query in segment ``i`` returns ``y[i]`` (the value at the left knot).

    Args:
        x: 1-D array of knot x-coordinates (length ``n``).
        y: 1-D array of function values at knots (length ``n``).

    Returns:
        Tuple ``(coeffs, grid)`` where ``coeffs`` is an ``(n-1, 1)`` array of
        left-knot values and ``grid`` is ``x`` itself.
    """
    coeffs = y[:-1].reshape((-1, 1))
    grid = x
    return coeffs, grid


@njit(cache=True)
def _get_next_spline_coeffs(x, y):
    """Compute 'next' step-interpolant coefficients.

    A query in segment ``i`` returns ``y[i+1]`` (the value at the right knot).

    Args:
        x: 1-D array of knot x-coordinates (length ``n``).
        y: 1-D array of function values at knots (length ``n``).

    Returns:
        Tuple ``(coeffs, grid)`` where ``coeffs`` is an ``(n-1, 1)`` array of
        right-knot values and ``grid`` is ``x`` itself.
    """
    coeffs = y[1:].reshape((-1, 1))
    grid = x
    return coeffs, grid


@njit(cache=True)
def _evaluate_nearest_spline(x_values, coeffs, grid, assume_sorted=False):
    """Evaluate a step interpolant (nearest/previous/next) at query points.

    Simply looks up the pre-computed constant value for the segment that
    contains each query point.  All three step variants (nearest, previous,
    next) share this evaluator; the difference lies in how coefficients are
    populated by their respective ``_get_*_spline_coeffs`` functions.

    Args:
        x_values: 1-D array of query x-coordinates.
        coeffs: ``(m, 1)`` array of constant values per segment.
        grid: 1-D breakpoint array of length ``m+1``.
        assume_sorted: Passed to :func:`_get_segment`.

    Returns:
        1-D array of step-interpolated values.
    """
    y_values = np.zeros(x_values.shape)
    left, index = _get_segment(x_values, grid[:-1], assume_sorted=assume_sorted)
    for i in index:
        y_values[i] = coeffs[left[i], 0]
    return y_values


def interp(x, y, kind=CUBIC, assume_sorted=False, copy=False, **kwargs):
    """Construct a 1-D interpolator from data points.

    Fits a piecewise polynomial of the requested kind to the ``(x, y)``
    pairs and returns an :class:`Interpolator1d` that can be called as a
    function and exposes derivative methods.

    Args:
        x: 1-D array-like of x-coordinates.  Need not be sorted unless
            ``assume_sorted=True``.
        y: 1-D array-like of function values, same length as ``x``.
        kind: Interpolation kind.  Accepts a string name or its integer alias:

            - ``'cubic'`` / ``3`` (default): C2-smooth cubic spline.
            - ``'quadratic'`` / ``2``: C1-smooth quadratic spline.
            - ``'linear'`` / ``1``: piecewise-linear.
            - ``'nearest'`` / ``0``: nearest-neighbour step.
            - ``'previous'``: left-endpoint step (previous knot value).
            - ``'next'``: right-endpoint step (next knot value).

        assume_sorted: If ``True`` ``x`` is assumed to be strictly increasing
            and no sorting is performed; otherwise data are sorted internally.
        copy: When ``assume_sorted=True``, whether to make a defensive copy of
            ``x`` and ``y`` before storing them.  Ignored when
            ``assume_sorted=False`` (a sorted copy is always made in that case).
        **kwargs: Extra keyword arguments forwarded to the coefficient
            function, e.g. ``bc_type`` and ``clamped_slopes`` for cubic
            splines.

    Returns:
        :class:`Interpolator1d` – callable object that evaluates the spline
        and provides :meth:`~Interpolator1d.derivative`,
        :meth:`~Interpolator1d.derivative2`,
        :meth:`~Interpolator1d.derivative3` methods.

    Examples
    --------
    Cubic interpolation of a sinusoid sampled at 11 nodes:

    >>> import numpy as np
    >>> from kanly.api import interp
    >>> x = np.linspace(0, 2*np.pi, 11)
    >>> y = np.sin(x)
    >>> f = interp(x, y, kind='cubic')
    >>> f(np.pi).round(3)                          # doctest: +ELLIPSIS
    0.00...
    >>> f.derivative(np.pi).round(3)               # doctest: +SKIP
    -1.0

    The alias ``interp1d`` is identical to ``interp``.

    See Also
    --------
    `cubic_spline`, `quadratic_spline`, `linear_spline` : kind-specific wrappers.
    """

    assert isinstance(kind, (str, int))
    if isinstance(kind, str):
        kind = kind.lower()

    if not assume_sorted:
        index = np.argsort(x)
        x, y = x[index], y[index]
    else:
        if copy:
            x, y = x.copy(), y.copy()
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)

    if kind == CUBIC or kind == 3:
        kind = CUBIC
        coef_func, eval_func, deriv, deriv2, deriv3 \
            = _get_cubic_spline_coeffs, _evaluate_cubic_spline, \
            _evaluate_cubic_spline_deriv, _evaluate_cubic_spline_deriv2, _evaluate_cubic_spline_deriv3
    elif kind == LINEAR or kind == 1:
        kind = LINEAR
        coef_func, eval_func, deriv, deriv2, deriv3 \
            = _get_linear_spline_coeffs, _evaluate_linear_spline, \
            _evaluate_linear_spline_deriv, _zero_function, _zero_function
    elif kind == QUADRATIC or kind == 2:
        kind = QUADRATIC
        coef_func, eval_func, deriv, deriv2, deriv3 \
            = _get_quadratic_spline_coeffs, _evaluate_quadratic_spline, \
            _evaluate_quadratic_spline_deriv, _evaluate_quadratic_spline_deriv2, _zero_function
    elif kind == NEAREST or kind == 0:
        kind = NEAREST
        coef_func, eval_func, deriv, deriv2, deriv3 \
            = _get_nearest_spline_coeffs, _evaluate_nearest_spline, \
            _zero_function, _zero_function, _zero_function  # since f', f''=0
    elif kind == PREVIOUS:
        coef_func, eval_func, deriv, deriv2, deriv3 \
            = _get_previous_spline_coeffs, _evaluate_nearest_spline, \
            _zero_function, _zero_function, _zero_function  # since f', f''=0
    elif kind == NEXT:
        coef_func, eval_func, deriv, deriv2, deriv3 \
            = _get_next_spline_coeffs, _evaluate_nearest_spline, \
            _zero_function, _zero_function, _zero_function  # since f', f''=0
    else:
        raise NotImplementedError(f'{kind=}!')

    coeffs, grid = coef_func(x, y, **kwargs)
    return Interpolator1d(x, y, eval_func, deriv, deriv2, deriv3, kind, coeffs, grid)


def cubic_spline(x, y, bc_type=DEFAULT_BC_TYPE, clamped_slopes=None):
    """Fit a cubic spline to data, with selectable boundary conditions.

    Convenience wrapper around :func:`interp` with ``kind='cubic'``.

    Args:
        x: 1-D array-like of x-coordinates (need not be sorted).
        y: 1-D array-like of function values, same length as ``x``.
        bc_type: Boundary condition type – one of ``'not-a-knot'`` (default),
            ``'natural'``, or ``'clamped'``.
        clamped_slopes: ``(left_slope, right_slope)`` used when
            ``bc_type='clamped'``; defaults to ``[0.0, 0.0]``.

    Returns:
        :class:`Interpolator1d` with cubic polynomial segments.

    Examples
    --------
    Cubic spline interpolation with natural boundary conditions:

    >>> import numpy as np
    >>> from kanly.api import cubic_spline
    >>> x = np.linspace(-1, 1, 9)
    >>> y = x ** 3
    >>> f = cubic_spline(x, y, bc_type='natural')
    >>> f(0.5).round(3)                          # doctest: +ELLIPSIS
    0.125
    """
    return interp(x, y, kind=CUBIC, bc_type=bc_type, clamped_slopes=clamped_slopes)


def quadratic_spline(x, y):
    """Fit a quadratic spline to data.

    Convenience wrapper around :func:`interp` with ``kind='quadratic'``.

    Args:
        x: 1-D array-like of x-coordinates (need not be sorted).
        y: 1-D array-like of function values, same length as ``x``.

    Returns:
        :class:`Interpolator1d` with quadratic polynomial segments.

    Examples
    --------
    Quadratic spline of a noisy parabola:

    >>> import numpy as np
    >>> from kanly.api import quadratic_spline
    >>> x = np.linspace(-3, 3, 13)
    >>> y = x ** 2
    >>> f = quadratic_spline(x, y)
    >>> f(0.5).round(2)                          # doctest: +SKIP
    0.25
    """
    return interp(x, y, kind=QUADRATIC)


def linear_spline(x, y):
    """Fit a piecewise-linear spline to data.

    Convenience wrapper around :func:`interp` with ``kind='linear'``.

    Args:
        x: 1-D array-like of x-coordinates (need not be sorted).
        y: 1-D array-like of function values, same length as ``x``.

    Returns:
        :class:`Interpolator1d` with linear polynomial segments.

    Examples
    --------
    Piecewise-linear interpolation between knots:

    >>> import numpy as np
    >>> from kanly.api import linear_spline
    >>> x = np.array([0.0, 1.0, 2.0, 4.0])
    >>> y = np.array([0.0, 2.0, 1.0, 5.0])
    >>> f = linear_spline(x, y)
    >>> f(1.5).round(3)                          # doctest: +ELLIPSIS
    1.5
    """
    return interp(x, y, kind=LINEAR)


class Interpolator1d(DillObject):
    """A fitted 1-D piecewise polynomial interpolator.

    Instances are constructed by :func:`interp` (or the convenience wrappers
    :func:`cubic_spline`, :func:`quadratic_spline`, :func:`linear_spline`) and
    are not normally created directly.

    Once fitted, the object is callable and also exposes derivative methods up
    to third order.  Calling ``f(x)`` is equivalent to ``f.__call__(x)``.

    Attributes:
        x: Sorted training x-coordinates.
        y: Training function values (aligned with ``x``).
        kind: String name of the interpolation kind (e.g. ``'cubic'``).
        coeffs: Array of per-segment polynomial coefficients.
        grid: 1-D breakpoint array used to locate segments during evaluation.
        power: Polynomial degree (0 for step interpolants, 1 linear, 2
            quadratic, 3 cubic).
    """

    def __init__(self, x, y, func, deriv, deriv2, deriv3, kind, coeffs, grid):
        """Initialise a fitted 1-D interpolator.

        Args:
            x: Sorted 1-D array of training x-coordinates.
            y: 1-D array of function values at ``x``.
            func: Numba-compiled evaluator function for the spline value.
            deriv: Numba-compiled evaluator for the first derivative.
            deriv2: Numba-compiled evaluator for the second derivative.
            deriv3: Numba-compiled evaluator for the third derivative.
            kind: String identifier of the interpolation kind – must be one of
                the module-level constants (``CUBIC``, ``LINEAR``, etc.).
            coeffs: Pre-computed coefficient array returned by the
                ``_get_*_coeffs`` helper.
            grid: Breakpoint array returned alongside ``coeffs``.
        """
        self.x = x
        self.y = y
        self._func = func
        self._deriv = deriv
        self._deriv2 = deriv2
        self._deriv3 = deriv3
        assert kind in (CUBIC, LINEAR, QUADRATIC, NEAREST, PREVIOUS, NEXT)
        self.kind = kind
        self.coeffs = coeffs
        # Map each kind to its polynomial degree for display purposes.
        self.power = {LINEAR: 1, CUBIC: 3, NEAREST: 0, QUADRATIC: 2, NEXT: 0, PREVIOUS: 0}[self.kind]
        self.grid = grid

    def __call__(self, x_values, assume_sorted=False):
        """Evaluate the interpolant at ``x_values``.

        Args:
            x_values: Scalar or 1-D array of query x-coordinates.
            assume_sorted: If ``True``, queries are assumed sorted (avoids
                an internal ``argsort`` for performance).

        Returns:
            Scalar or 1-D array of interpolated values.
        """
        return evaluate_spline(x_values, self.coeffs, self.grid, self._func, assume_sorted=assume_sorted)

    def derivative(self, x_values, assume_sorted=False):
        """Evaluate the first derivative of the interpolant at ``x_values``.

        Args:
            x_values: Scalar or 1-D array of query x-coordinates.
            assume_sorted: If ``True``, queries are assumed sorted.

        Returns:
            Scalar or 1-D array of first-derivative values.  Returns zero
            everywhere for step interpolants (``nearest``, ``previous``,
            ``next``).
        """
        return evaluate_spline(x_values, self.coeffs, self.grid, self._deriv, assume_sorted=assume_sorted)

    def derivative2(self, x_values, assume_sorted=False):
        """Evaluate the second derivative of the interpolant at ``x_values``.

        Args:
            x_values: Scalar or 1-D array of query x-coordinates.
            assume_sorted: If ``True``, queries are assumed sorted.

        Returns:
            Scalar or 1-D array of second-derivative values.  Returns zero
            everywhere for linear and step interpolants.
        """
        return evaluate_spline(x_values, self.coeffs, self.grid, self._deriv2, assume_sorted=assume_sorted)

    def derivative3(self, x_values, assume_sorted=False):
        """Evaluate the third derivative of the interpolant at ``x_values``.

        Args:
            x_values: Scalar or 1-D array of query x-coordinates.
            assume_sorted: If ``True``, queries are assumed sorted.

        Returns:
            Scalar or 1-D array of third-derivative values.  Returns zero
            everywhere for quadratic, linear, and step interpolants.
        """
        return evaluate_spline(x_values, self.coeffs, self.grid, self._deriv3, assume_sorted=assume_sorted)

    def __repr__(self):
        """Return the same string as :meth:`__str__`."""
        return str(self)

    def __str__(self, decimals=4):
        """Return a human-readable summary of the piecewise polynomial.

        Lists each segment's polynomial expression alongside its x-interval.
        For cubic splines the local offset variable ``h(x) = x - x_left`` is
        printed to make the segment polynomials easier to read.

        Args:
            decimals: Number of decimal places used for coefficient formatting.

        Returns:
            Multi-line string with one row per segment.
        """
        power = self.power
        s = self.kind
        n = len(self.coeffs)

        x_str = 'x' if power < 3 else 'h(x)'
        monomial_str = lambda p: f'{x_str}^{p}' if p >= 2 else (x_str if p == 1 else '')

        for i, c in enumerate(self.coeffs):
            s += '\n' + ' + '.join(
                [f'{c[j]:{3 + decimals}.{decimals}f} {monomial_str(self.power - j)}' for j in range(self.power + 1)])
            s += f', {self.grid[i]:{3 + decimals}.{decimals}f} <= x <= {self.grid[i + 1]:{3 + decimals}.{decimals}f}'
            if power == 3:
                s += f', {"" if i < n - 1 else " " * 14}  h(x) = x - {c[-1]:{3 + decimals}.{decimals}f}'

        return s
