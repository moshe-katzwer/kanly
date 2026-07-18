"""Helpers for expanding ``poly(...)`` formula sparse_terms into monomial sparse_terms.

The parser treats ``poly(x, k)`` as a shorthand and expands it into explicit
``I(x**j)`` sparse_terms so downstream sparse-term logic can operate on ordinary
tokens/interactions.
"""
from __future__ import absolute_import, print_function

import re

import numpy as np

from kanly.utils.util import str_to_args


def find_first_poly_in_var(var_name):
    """Return the first ``poly(...)`` argument payload found in a term string.

    Args:
        var_name (str): A formula term, e.g. ``"poly(x,3):C(g)"``.

    Returns:
        str or None: Text inside the first ``poly(...)`` call, or None if not
        present.
    """
    polys = re.findall(r'(?<=poly\().+?(?=\))', var_name)
    if len(polys):
        return polys[0]
    else:
        return None


def monomial_power_str(regressor, power):
    """Convert a base regressor and exponent into formula-token text.

    Args:
        regressor (str): Regressor token, e.g. ``"x"``.
        power (int): Exponent to encode.

    Returns:
        str: Empty string for exponent 0, base token for exponent 1, and
        ``I(base**power)`` for higher exponents.

    Raises:
        Exception: If ``power < 0``.
    """
    if power < 0:
        raise Exception("Power must be >= 0!")
    elif power == 0:
        return ''
    elif power == 1:
        return regressor
    else:
        return f'I({regressor}**{power})'


def replace_poly_in_var_name_with_monomials_exploded(var_name):
    """Recursively expand one ``poly(...)`` occurrence into explicit monomials.

    Supported forms:
      - ``poly(x, 3)`` -> powers 0..3
      - ``poly(x, 3, -1)`` -> powers 1..3 (drop constant)
      - ``poly(x, [1,3,5])`` -> explicit exponent set

    The function is recursive so nested/multiple ``poly`` tokens in the same
    term are fully exploded.

    Args:
        var_name (str): Original term string.

    Returns:
        str or list[str]: Original term (if no ``poly`` present) or an expanded
        list of monomialized term strings.

    Raises:
        Exception: If polynomial order arguments are malformed.
    """
    poly_search_result = find_first_poly_in_var(var_name)
    if poly_search_result is not None:
        args = str_to_args(poly_search_result)

        drop_const = False
        if len(args) == 3:
            try:
                drop_const = int(args[2]) == -1
            except:
                raise Exception(f'Third argument for "poly" must be "-1", not{args[2]}!')

        regressor, order = args[0], args[1]
        try:
            order = int(order)
            exponents = range(drop_const, order + 1)
        except:
            try:
                order = order.translate({ord(c): None for c in '()[] '}).split(',')
                exponents = sorted(np.array(order).astype(int))
            except:
                raise Exception("the `order` in a polynomial must be a list of integers or an int")

        lists = [
            replace_poly_in_var_name_with_monomials_exploded(
                var_name.replace(f'poly({poly_search_result})',
                                 f'{monomial_power_str(regressor, j)}'))
            for j in exponents]
        to_return = []
        for l in lists:
            if isinstance(l, list):
                to_return += l
            else:
                to_return.append(l)
        to_return = [t for t in to_return if t != '']
        to_return = [t.replace("::", ":") for t in to_return]
        to_return = [t.lstrip(":").rstrip(":") for t in to_return]

        return to_return

    else:
        return var_name


def replace_poly_in_var_names_with_monomials_exploded(var_names):
    """Expand ``poly(...)`` shorthand across a list of term strings.

    Args:
        var_names (list[str]): Term names potentially containing ``poly`` calls.

    Returns:
        list[str]: Flattened list with all polynomial shorthand expanded.
    """
    var_names_new = []
    for v in var_names:
        v = replace_poly_in_var_name_with_monomials_exploded(v)
        if isinstance(v, list):
            var_names_new += v
        else:
            var_names_new.append(v)
    return var_names_new

# from formula
# @staticmethod
# def polynomial_term_maker(var_2_exponent_dict, scale=False):
#     str_base_orig = ':'.join(
#         'I(%s**%s%s%s)' % (var, '{' * (2 ** (i + 1)), 'var' + str(i), '}' * (2 ** (i + 1)))
#         for i, var in enumerate(var_2_exponent_dict.keys()))
#
#     str_list = [str_base_orig]
#
#     for i, (k, v) in enumerate(var_2_exponent_dict.items()):
#         var = 'var%d' % i
#         temp_list = []
#         for str_base in str_list.copy():
#             str_base = str_base.replace('{{%s}}' % var, '{}')
#             for j in range(v + 1):
#                 cur_string = str_base.format(j)
#                 temp_list.append(cur_string)
#         str_list = temp_list
#
#     temp_list = []
#     for j, s in enumerate(str_list):
#         split = s.split(':')
#         temp_split = []
#         for i, v in enumerate(split):
#             if '**0' in v:
#                 continue
#             elif '**1' in v:
#                 v = v.replace('**1)', '').replace('I(', '')
#
#             if '**' in v and scale:
#                 exp = v[v.index('**'):].split('**')[1].split(')')[0]
#                 print(exp)
#                 if int(exp) > 1:
#                     print(v, '**' + exp + ')')
#                     v = v.replace('**' + exp + ')', '**' + exp + "/10**" + exp + ')')
#
#             temp_split.append(v)
#
#         if temp_split:
#             temp_list.append(':'.join(temp_split))
#
#     return temp_list
#
# @staticmethod
# def polynomial_formula_maker(endog_name, exog_polynomial_dict, other_exog=None, scale=False):  # todo other exog
#     return endog_name + ' ~ ' + ' + '.join(SparseFormula.polynomial_term_maker(exog_polynomial_dict, scale=scale))
