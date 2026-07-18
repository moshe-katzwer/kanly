from __future__ import absolute_import, print_function, annotations


def parse_str_2_tuple(string, parse_parenthesis=False, level=0):
    """
    Respects parenthesis

    print((parse_str_2_tuple('a,(b,c)', parse_parenthesis=False))) -->
        ['a', '(b, c)']


    print((parse_str_2_tuple('a,(b,c)', parse_parenthesis=True))) -->
        ['a', ['b', 'c']]
    """

    parenth_stack = 0
    bracket_stack = 0
    curly_stack = 0
    for i, c in enumerate(string):
        if c == '(':
            parenth_stack += 1
        elif c == ')':
            parenth_stack -= 1
        elif c == '[':
            bracket_stack += 1
        elif c == ']':
            bracket_stack -= 1
        elif c == '{':
            curly_stack += 1
        elif c == '}':
            curly_stack -= 1
        elif c == ',' and parenth_stack == 0 and bracket_stack == 0:
            return [string[:i]] + parse_str_2_tuple(string[i + 1:], parse_parenthesis)

    result = [string]

    if parse_parenthesis:
        for i, c in enumerate(result):
            if c[0] in ('{', '(', '['):
                result[i] = parse_str_2_tuple(c[1:-1], True, level+1)

    return result
