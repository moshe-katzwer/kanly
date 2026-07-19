"""Lightweight metadata object for parsed model-parameter tokens."""

from __future__ import absolute_import, print_function, annotations

import pprint

SCALAR_PARAM_TYPE = 'scalar'
VECTOR_PARAM_TYPE = 'vector'
DUMMY_PARAM_TYPE = 'dummy'
POLYNOMIAL_PARAM_TYPE = 'poly'
SPLINE_PARAM_TYPE = 'spline'


class Parameter(object):
    """Container for one parsed parameter declaration from a model code block."""

    def __init__(self, param_string, param_string_unbounded, name, bounds, param_type, dim, other_info):
        """Initialize parsed parameter metadata.

        Args:
            param_string: Original raw token (including any bounds syntax).
            param_string_unbounded: Canonical token with bounds markers removed.
            name: Resolved parameter/group name.
            bounds: Optional ``(lb, ub)`` tuple or None.
            param_type: Parameter family label (scalar/vector/dummy/poly/spline).
            dim: Parameter dimensionality (for vector-like parameter declarations).
            other_info: Extra parser metadata needed for special parameter types.
        """
        self.param_string = param_string
        self.param_string_unbounded = param_string_unbounded
        self.name = name
        self.bounds = bounds
        self.param_type = param_type
        self.dim = dim
        self.other_info = other_info

    def __repr__(self):
        """Return readable object representation.

        Args:
            None.
        """
        return str(self)

    def __str__(self):
        """Pretty-print stored parameter metadata.

        Args:
            None.
        """
        return pprint.pformat(self.__dict__)
