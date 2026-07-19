from __future__ import absolute_import, print_function

from kanly.utils.util import str_to_args


class ArgumentParser(object):
    """Parse argument strings for special NLLS formula sparse_terms.

    The nonlinear formula parser delegates ``poly(...)``, ``polym(...)``,
    ``cheb(...)``, and ``C(...)`` argument handling to this class.  The helpers
    preserve the compact formula syntax while returning explicit tuples used to
    build generated prediction code and parameter names.
    """

    @staticmethod
    def get_override_name(arg_str: str) -> list:
        """
        If args end in `;___`, it will override the
        variable name with `___`

        Args:
            arg_str: Raw comma-separated argument string, optionally ending in
                ``; name`` to override generated parameter prefixes.

        Returns:
            Tuple ``(arg_str_without_override, override_name)`` where
            ``override_name`` is ``None`` if no override suffix is present.
        """
        if ';' in arg_str:
            v = arg_str.split(';')
            arg_str, override_name = ';'.join(v[:-1]), v[-1].strip()
        else:
            override_name = None
        return arg_str, override_name

    @staticmethod
    def str_to_args(arg_str: str) -> list:
        """Split a formula argument string on top-level commas.

        Args:
            arg_str: Argument string such as ``'x, [1,2,3]'``.

        Returns:
            List of argument substrings, preserving nested brackets and
            function calls.
        """
        return str_to_args(arg_str)

    @staticmethod
    def _parse_categorical_str_to_args(cat_str: str) -> (list, bool):
        """
        Examples:
            (1) 'x'           -->   (['x'], False)

            (2) 'x, -1'       -->   (['x'], True)

            (3) 'x, y, -1'    -->   (['x', 'y'], True)

            'x, y; name' will override the name with `name`

        Args:
            cat_str: Argument string from a ``C(...)`` categorical term.

        Returns:
            Tuple ``(variables, drop1, override_name)`` where ``variables`` are
            the categorical columns or expressions, ``drop1`` indicates whether
            the first level should be omitted, and ``override_name`` optionally
            overrides generated parameter names.
        """

        cat_str, override_name = ArgumentParser.get_override_name(cat_str)

        args = ArgumentParser.str_to_args(cat_str)

        drop1 = False
        try:
            drop1 = (int(args[-1]) == -1)
        except:
            pass

        if drop1:
            variables = args[:-1]
        else:
            variables = args

        return variables, drop1, override_name

        # TODO switch to more pythonic functional view, see _internal
        # args = ArgumentParser.str_to_args(cat_str)
        # new_str = ''
        # vals = {'vars': [], 'drop1': [], 'name': []}
        #
        # for arg in args:
        #     if arg[:6] == 'drop1=':
        #         vals['drop1'].append(arg)
        #     elif arg[:5] == 'name=':
        #         vals['name'].append(arg)
        #     else:
        #         vals['vars'].append(f"'{arg}'")
        #
        # for v in vals.values():
        #     if len(v):
        #         new_str += ','.join(v) + ','
        #
        # def _internal(*args, drop1=False, name=None):
        #     return args, drop1, name
        #
        # exec(f'_Z = _internal({new_str})')
        # print(locals())
        # return Z

    @staticmethod
    def _parse_cheb_str_to_args(poly_str: str) -> (str, int, bool):
        """
        Examples:
            (1)  'x, 2'        -->    ('x', 2, False)
            (2)  'x, 2, -1'    -->    ('x', 2, True)

            'x, y; name' will override the name with `name`

        Args:
            poly_str: Argument string from a ``cheb(...)`` term.

        Returns:
            Tuple ``(var_name, max_exponent, drop1, override_var_name)`` for
            Chebyshev basis expansion.
        """

        poly_str, override_var_name = ArgumentParser.get_override_name(poly_str)

        args = ArgumentParser.str_to_args(poly_str)
        if len(args) not in [2, 3]:
            raise Exception()

        var_name = args[0]

        drop1 = False
        if len(args) == 3:
            try:
                drop1 = (int(args[2]) == -1)
            except:
                raise Exception()

        try:
            max_exponent = int(args[1])
            assert max_exponent >= 1
        except:
            raise Exception

        return var_name, max_exponent, drop1, override_var_name

    @staticmethod
    def _parse_poly_str_to_args(poly_str):
        """
        Examples:
            (1)   'x, 2'		-->		('x', range(0, 3))
            (2)   'x, 2, -1'	-->		('x', range(1, 3))
            (3)   'x, [1,2,5]'  -->		('x', [1, 2, 5])

        Args:
            poly_str: Argument string from a ``poly(...)`` term.

        Returns:
            Tuple ``(var_name, exponents, override_name)`` where ``exponents``
            is either a ``range`` or explicit list of integer powers.
        """

        poly_str, override_name = ArgumentParser.get_override_name(poly_str)

        args = ArgumentParser.str_to_args(poly_str)
        if len(args) not in [2, 3]:
            raise Exception()

        var_name = args[0]

        drop1 = False
        if len(args) == 3:
            try:
                drop1 = (int(args[2]) == -1)
            except:
                raise Exception()

        try:
            exponents = int(args[1])
            exponents = range(int(drop1), exponents + 1)
        except:
            if len(args) == 3:
                raise Exception()
            try:
                exponents = [int(e) for e in args[1][1:-1].split(',')]
            except:
                raise Exception()

        return var_name, exponents, override_name

    @staticmethod
    def _parse_polym_str_to_args(arg_str: str) -> (list, int, bool):
        """
        Examples:
            (1)   'x, y, 2'		    -->		(['x', 'y'], 2, False)
            (2)   'x, y, 2, -1'		-->		(['x', 'y'], 2, True)

        Args:
            arg_str: Argument string from a ``polym(...)`` multivariate
                polynomial term.

        Returns:
            Tuple ``(var_names, exponent, drop1)`` where ``var_names`` are the
            input variables and ``exponent`` is the maximum total degree.
        """

        args = ArgumentParser.str_to_args(arg_str)

        drop1 = False
        try:
            drop1 = (int(args[-1]) == -1)
        except:
            pass

        if drop1:
            args = args[:-1]

        assert len(args) >= 3
        exponent = int(args[-1])
        var_names = args[:-1]

        return var_names, exponent, drop1


# if __name__ == '__main__':
#     def main():
#         import numpy as np
#         from kanly.regression.nonlinear_least_squares.formula.sparse_nonlinear_formula_parser import \
#             build_prediction_function_from_formula
#         np.random.seed(0)
#
#         n = 100
#         data = dict(g1=np.random.randint(0, 3, n), g2=np.random.randint(7, 10, n).astype(str))
#         func = build_prediction_function_from_formula('[C(g1,g2,-1)]', data)
#         print(func)
#         print(func[0].__dict__)
#         print(func[0].num_params)
#
#     main()
