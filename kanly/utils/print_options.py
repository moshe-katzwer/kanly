from __future__ import absolute_import, print_function


def print_options(options, title='OPTIONS'):
    """Pretty-print a dictionary of options as a titled key–value table.

    Output is formatted as a box with ``─`` separators and left-aligned keys
    padded to the width of the longest key.

    Args:
        options: Mapping of option names (strings) to their values.
        title: Header text displayed at the top of the table.  Defaults to
            ``'OPTIONS'``.
    """
    width_key = max([len(s) for s in options.keys()])
    width_val = max([len(str(s)) for s in options.values()])
    width = max(width_key + 3 + width_val, len(title))
    print()
    print('-' * width)
    print(title)
    print('=' * width)
    for i in options.items():
        print(f"%{width_key}s:  %s" % i)
    print('-' * width)
    print()
