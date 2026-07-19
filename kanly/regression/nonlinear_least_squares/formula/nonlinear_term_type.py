from __future__ import absolute_import, print_function

from enum import Enum


class NonlinearTermType(Enum):
    """Enumeration of special term types supported by the NLLS formula parser.

    Values describe how a bracketed term such as ``[x]``, ``[poly(x, 3)]``, or
    ``[C(group, -1)]`` should be expanded into generated prediction code and
    parameter names.
    """

    MONOMIAL = 0
    POLYNOMIAL = 1
    MULTIVARIATE_POLYNOMIAL = 2
    CHEBYSHEV = 3
    CATEGORICAL = 4

    @staticmethod
    def _get_term_type(term_str: str) -> (int, str):
        """Classify a formula term and return the parser payload.

        Args:
            term_str: Raw term text extracted from inside ``[...]`` in an NLLS
                formula, e.g. ``'x'``, ``'poly(x, 2)'``, or ``'C(group,-1)'``.

        Returns:
            Dict with ``term_type`` set to a ``NonlinearTermType`` and
            ``term_arg_str`` set to the inner argument string that the
            specialised parser should consume.
        """

        if term_str[:5] == 'poly(':
            return {'term_type': NonlinearTermType.POLYNOMIAL, 'term_arg_str': term_str[5:-1]}

        elif term_str[:6] == 'polym(':
            return {'term_type': NonlinearTermType.MULTIVARIATE_POLYNOMIAL, 'term_arg_str': term_str[6:-1]}

        elif term_str[:5] == 'cheb(':
            return {'term_type': NonlinearTermType.CHEBYSHEV, 'term_arg_str': term_str[5:-1]}

        elif term_str[:2] == 'C(':
            return {'term_type': NonlinearTermType.CATEGORICAL, 'term_arg_str': term_str[2:-1]}

        else:
            return {'term_type': NonlinearTermType.MONOMIAL, 'term_arg_str': term_str}
