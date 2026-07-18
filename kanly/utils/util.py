from __future__ import absolute_import, print_function

import inspect

from scipy.sparse import isspmatrix
from pandas import DataFrame
from kanly.sparse_data_frame import SparseDataFrame


def dict_2_dataframe(data):
    """Coerce ``data`` to a ``pd.DataFrame`` if it is not already one.

    Passes ``DataFrame`` and ``SparseDataFrame`` instances through unchanged;
    wraps any other value with ``pd.DataFrame(data)``.

    Args:
        data: ``DataFrame``, ``SparseDataFrame``, or any dict-like / 2-D
            array-like accepted by ``pd.DataFrame()``.

    Returns:
        A ``pd.DataFrame`` (or ``SparseDataFrame``) containing the input data.
    """
    if isinstance(data, (DataFrame, SparseDataFrame)):
        return data
    else:
        return DataFrame(data)


def round_list(lst, sig_figs, uniform_decimal_output=False):
    """Format a list of numbers to a given number of significant figures.

    Args:
        lst: Iterable of numeric values to format.
        sig_figs: Number of significant figures (or decimal places when
            ``uniform_decimal_output=True``).
        uniform_decimal_output: When ``True``, use ``%.<sig_figs>f`` fixed-
            point format (uniform decimal places); when ``False`` use
            ``%.<sig_figs>g`` (significant figures, may use scientific
            notation).

    Returns:
        List of formatted strings.
    """
    return [(('%.' + str(sig_figs) + 'f') if uniform_decimal_output else ('%.' + str(sig_figs) + 'g')) % l
            for l in lst]


def none_copy(x):
    """Return a copy of ``x``, or ``None`` / the string unchanged.

    Calls ``x.copy()`` for objects that support it (e.g. NumPy arrays,
    pandas Series/DataFrames).  Returns ``None`` or string values directly
    without copying.

    Args:
        x: Object to copy, ``None``, or a ``str``.

    Returns:
        Copy of ``x`` or the original value for ``None``/string types.

    Raises:
        Exception: If ``x`` is not ``None``, not a string, and does not have
            a ``copy`` method.
    """
    if x is None:
        return x
    elif isinstance(x, str):
        return x
    elif hasattr(x, 'copy'):
        return x.copy()
    else:
        raise Exception("arg of type '%s' has no 'copy' method!" % (type(x)))


def to_dense_helper(x, flatten=False, copy=True):
    """Convert a sparse matrix (or pass through a dense array) to a dense NumPy array.

    Returns ``None`` unchanged.  Converts sparse inputs via ``toarray()`` and
    always returns a copy.

    Args:
        x: Sparse matrix, dense NumPy array, or ``None``.
        flatten: When ``True``, return the result as a 1-D flattened array.
        copy: Unused; a copy is always returned for consistency.

    Returns:
        Dense NumPy array, or ``None`` when ``x`` is ``None``.
    """

    if x is None:
        return None

    if isspmatrix(x):
        x = x.toarray().copy()

    if flatten:
        return x.flatten().copy()
    else:
        return x.copy()


def str_to_args(arg_str: str) -> list:
    """
    Example: '[x+3], f(x+np.log(1)), 5, hello' --> ['[x+3]', 'f(x+np.log(1))', '5', 'hello']
    """

    # arg_str = arg_str.replace(' ', '') + ',' # TODO why was it like this originally?
    arg_str = arg_str.strip() + ','

    open_char = {'{', '(', '['}
    close_char = {'}', ')', ']'}
    args = []
    cur_term = ''

    open_stack = 0

    for c in arg_str:
        if c in open_char:
            open_stack += 1
            cur_term += c
        elif c == ',':
            if open_stack == 0:
                args.append(cur_term)
                cur_term = ''
            else:
                cur_term += ','
        elif c in close_char:
            cur_term += c
            open_stack -= 1
        else:
            cur_term += c

    return args


def print_iter_info(info_list, is_header=False, is_footer=False):
    """Print a formatted optimiser/MCMC iteration-progress row.

    Each element of ``info_list`` is a dict describing one column:
    ``{'name': str, 'len': int, 'format': str, 'value': any}``.  The header
    and footer variants are framed with ``=`` / ``-`` separators.

    Args:
        info_list: List of column descriptor dicts.  Each dict must have:
            - ``'name'``: Column heading string.
            - ``'len'``: Column width in characters.
            - ``'format'``: ``printf``-style format string (e.g. ``'%8.4f'``).
            - ``'value'``: Value to display (used for non-header rows).
        is_header: When ``True``, print column names with a double-line
            separator above and a single-line separator below.
        is_footer: When ``True``, print a closing ``=`` separator line.
    """
    s = ""
    for id in info_list:
        if is_header:
            s += f"%{id['len']}s" % id['name']
        else:
            s += id['format'] % id['value']

    if is_header:
        s = "\n" + '=' * len(s) + "\n" + s + "\n" + "-" * len(s)

    elif is_footer:
        s = '=' * len(s)

    print(s)


def get_eval_env_depth():
    """Return the depth of the current Python call stack above this function.

    Walks ``inspect.currentframe()`` up to the topmost frame and counts the
    number of intermediate frames.  Useful for determining how many levels
    deep an ``eval`` or ``exec`` call is occurring, which affects formula
    variable lookup scopes.

    Returns:
        Non-negative integer call-stack depth, or ``0`` on any exception
        (e.g. frames not available in all Python implementations).
    """
    try:
        frame = inspect.currentframe()
        fback = frame.f_back
        k = 0
        while fback.f_back is not None:
            fback = fback.f_back
            k += 1
        return k
    except Exception:
        return 0
    finally:
        del frame
