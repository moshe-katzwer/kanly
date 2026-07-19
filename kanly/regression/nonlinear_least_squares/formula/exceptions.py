from __future__ import absolute_import, print_function


class FormulaException(Exception):
    """Exception raised when a nonlinear least-squares formula cannot be parsed.

    This package uses a custom formula language with ``[data]`` sparse_terms,
    ``{parameter}`` sparse_terms, categorical expansions, and optional weights.  This
    exception marks failures that are specific to parsing or validating that
    formula syntax.
    """
    pass
