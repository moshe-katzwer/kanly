from __future__ import absolute_import, print_function, annotations

import re
from collections import OrderedDict


def parse_code_str(func_str, start, end, unique=True):
    """Extract all tokens in ``func_str`` that are delimited by ``start``…``end``.

    Strips comment lines and inline comments from the source string before
    searching, then applies a regex look-behind / look-ahead pattern to find
    all substrings that appear between the ``start`` and ``end`` delimiters.

    Typical usage: extract parameter names from a template expression such as
    ``"{alpha} * x + {beta}"`` with ``start='{'``, ``end='}'``.

    Args:
        func_str: Source string to search (may be a multi-line code snippet).
        start: Delimiter string marking the beginning of each token.
            Special regex characters ``[``, ``]``, and ``$`` are escaped
            automatically.
        end: Delimiter string marking the end of each token (same escaping
            applied).
        unique: When ``True`` (default), return only the first occurrence of
            each token (order-preserving deduplication via ``OrderedDict``).

    Returns:
        Tuple ``(var_names, cleaned_func_str)`` where ``var_names`` is the
        list of extracted token strings and ``cleaned_func_str`` is the source
        with comments removed and consecutive whitespace collapsed.
    """
    start = start.replace('[', '\[').replace(']', '\]').replace('$', '\$')
    end = end.replace('[', '\[').replace(']', '\]').replace('$', '\$')

    while '  ' in func_str:
        func_str = func_str.replace('  ', ' ')

    func_str_temp_list = func_str.split('\n')
    func_str_temp = ''
    for v in func_str_temp_list:

        if len(v.strip()) == 0:
            func_str_temp += '\n' + v
        elif v.strip()[0] == '#':
            continue
        else:
            func_str_temp += '\n' + v.split('#')[0]

    pattern = r'(?<=' + start + r').+?(?=' + end + ')'

    func_str_temp_list = func_str_temp.split('\n')

    var_names = []
    stride = 1 + (start == end)
    for s in func_str_temp_list:
        var_names += [x.strip() for x in re.findall(pattern, s)][::stride]

    if unique:
        var_names = list(OrderedDict.fromkeys(var_names))

    return var_names, func_str
