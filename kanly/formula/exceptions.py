"""Formula-package specific exception types."""
from __future__ import absolute_import, print_function


class MissingColumnException(Exception):
    """Raised when a formula references a column not present in the input data."""
    pass


class MissingDataException(Exception):
    """Raised when required data are missing for formula matrix construction."""
    pass


class AbsorbAndNoInterceptException(Exception):
    """Raised when absorb/fixed effects are requested with an explicit ``-1`` formula."""
    pass
