"""Custom exception types for the kanly linear models package.

These thin ``Exception`` subclasses allow callers to catch specific
failure modes with ``except`` without relying on string matching against
generic exception messages.
"""

from __future__ import absolute_import, print_function


class IVUnderidentifiedException(Exception):
    """Raised when an instrumental-variables model is under-identified.

    An IV model is under-identified when the number of excluded
    instruments is less than the number of endogenous regressors, making
    the coefficient on the endogenous regressors unidentified.
    """
    pass


class RidgeException(Exception):
    """Raised for general ridge-regression configuration errors.

    Covers situations such as an invalid combination of ridge penalty
    settings or a ridge model that cannot be constructed from the
    supplied inputs.
    """
    pass


class RidgeParameterException(Exception):
    """Raised when the ridge penalty parameter specification is invalid.

    Examples include a negative penalty value, an unsupported penalty
    container type, or missing required keys in the ``ridge_kwds`` dict
    (e.g. ``'alpha'`` not supplied).
    """
    pass