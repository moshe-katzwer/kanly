"""Symbolic automatic differentiation for a restricted arithmetic / NumPy grammar.

Parses a flattened expression string into an :class:`AutoDiffGraphNode` tree, applies the
chain and product rules to emit Python source for partial derivatives, then ``exec` is
optional Numba-JIT callables for Jacobians, gradients, and Hessians.

Only elementary functions listed in :data:`kanly.automatic_differentiation.elementary_functions.DERIV_FUNC_NAME_DICT`
and the parsing rules implemented here are supported.
"""
from __future__ import absolute_import, print_function

import re
from itertools import chain

import numpy as np
from numba import jit  # don't delete! -- it's used in `exec` block

from kanly.utils.linalg_utils import DEFAULT_DENSE_THRESHOLD_MB, DenseThreshold
from kanly.automatic_differentiation.elementary_functions import *  # don't delete
from scipy.sparse import csc_matrix  # don't delete

NEG_POWER_TOKEN = '▼'
POWER_TOKEN = '^'


# ``**`` is rewritten to POWER_TOKEN / NEG_POWER_TOKEN so splitting on ``^`` is unambiguous.


@jit
def _jit_sum(z):
    """Return ``z`` as float if scalar-shaped, else ``np.sum(z)`` (used in generated code paths)."""
    if np.shape(z) == tuple():
        return float(z)
    else:
        return np.sum(z)


def build_jacobian_from_string(
        func_str, num_params, nobs, other_args='', dense_threshold_mb=DEFAULT_DENSE_THRESHOLD_MB, debug=False,
        do_jit=True):
    """Convenience wrapper: parse ``func_str`` and return :meth:`AutoDiffGraphNode.get_analytical_jacobian` result dict."""
    return AutoDiffGraphNode(func_str).get_analytical_jacobian(
        num_params, nobs, other_args=other_args, dense_threshold_mb=dense_threshold_mb, debug=debug, do_jit=do_jit)


def build_partial_derivative_from_string(func_str, arg_number, nobs, other_args='', debug=False, do_jit=True,
                                         return_info=False):
    """Convenience wrapper: ∂/∂``params[arg_number]`` for ``func_str`` (callable or tuple with info)."""
    return AutoDiffGraphNode(func_str).get_analytical_partial_derivative(
        arg_number, nobs, other_args=other_args, debug=debug, do_jit=do_jit, return_info=return_info)


FIXED_EFFECT = 'FIXED_EFFECT'
FIXED_EFFECT_DERIVATIVE = 'FIXED_EFFECT_DERIVATIVE'

OP_ADD = 'OP_ADD'
OP_SUBTRACT = 'OP_SUBTRACT'
OP_DIVIDE = 'OP_DIVIDE'
OP_MULTIPLY = 'OP_MULTIPLY'
OP_EXPONENTIATION = 'OP_EXPONENTIATION'

FUNCTION = 'FUNCTION'
EXPRESSION = 'EXPRESSION'
VARIABLE = 'VARIABLE'
PARAMETER = 'PARAMETER'
FLOAT = 'FLOAT'


class AutoDiffGraphNode(object):
    """One node in a parsed expression tree for symbolic differentiation.

    ``type`` is one of the ``OP_*``, ``FUNCTION``, ``EXPRESSION``, ``PARAMETER``, ``FLOAT``,
    ``VARIABLE``, ``FIXED_EFFECT``, or ``FIXED_EFFECT_DERIVATIVE`` module constants. Children
    hold sub-expressions for operators and function arguments.
    """

    DERIV_FUNC_NAME_DICT = DERIV_FUNC_NAME_DICT

    @staticmethod
    def _expression_is_fixed_effect_derivative(name):
        """Return True if ``name`` matches the fixed-effect indicator derivative template (zeros/ones mask)."""
        fixed_eff_deriv_template = 'np.ones(len(int_arr[:,]))*(int_arr[:,]==)'
        name_minus_digits = re.sub(r'[0-9]', '', name).replace(' ', '')
        return name_minus_digits == fixed_eff_deriv_template

    def __init__(self, name, parent=None, debug=False):
        """Parse ``name`` into a typed node with children (recursive descent).

        Parameters
        ----------
        name : str
            Substring of the objective_function after normalization (operators, calls, ``params[k]``, etc.).
        parent : AutoDiffGraphNode or None
            Parent node in the tree, if any.
        debug : bool
            If True, print parse progress.
        """

        name = AutoDiffGraphNode._strip(name)

        if AutoDiffGraphNode._expression_is_fixed_effect_derivative(name):
            self.name = name
            self.type = FIXED_EFFECT_DERIVATIVE
            self.children = []
            return

        # print('! ', name, name[0])
        # if name[0] == '-':
        #     name_ = name.replace(' ', '')
        #     if len(name_) > 1:
        #         if name[1].isalpha():
        #             name = '(-1.0) * ' + name[1:]

        if len(name) >= 2:
            if name[:3] == '1* ':
                name = name[3:]
            if name[:4] == '1.0* ':
                name = name[4:]
            elif name[:2] == '1*' and name[2].isalpha():
                name = name[3:]
            elif name[:4] == '1.0*' and name[4].isalpha():
                name = name[4:]
            else:
                for k in [' ', '+', '-', '*', '/', POWER_TOKEN]:
                    if name[:3] == f'1*{k}':
                        name = name[3:]
                        break

        if debug:
            print(f'Parsing {name}...')

        assert name.count('(') == name.count(')')
        assert name.count('[') == name.count(']')
        if name == '':
            print(parent)
            raise Exception
        self.name = name

        self.parent = parent
        self.func_name = None

        self.fe_param_numbers = None
        self.fe_param_int_arr_idx = None

        if name == '+':
            self.type = OP_ADD
        elif name == '-':
            self.type = OP_SUBTRACT
        elif name == '/':
            self.type = OP_DIVIDE
        elif name == '*':
            self.type = OP_MULTIPLY
        elif name == POWER_TOKEN or name == NEG_POWER_TOKEN:
            self.type = OP_EXPONENTIATION
        elif (re.sub(r'[0-9]', '', name.replace(':', '')) == 'params[][int_arr[,]]'
              or re.sub(r'[0-9]', '', name.replace(':', '')) == 'np.hstack((np.zeros(),params[]))[int_arr[,]]'):
            self.type = FIXED_EFFECT

            v = name
            for s in '()[]:,':
                v = v.replace(s, f' {s} ')
            ints = [int(s) for s in v.split(' ') if s.isnumeric()]
            self.fe_param_int_arr_idx = ints[-1]
            self.fe_param_numbers = tuple(ints[-3:-1])
            self.fe_offset = len(ints) == 4

        elif name[:7] == 'params[' and name[-1] == ']' and not np.any(
                [z in name for z in '+-/*()' + NEG_POWER_TOKEN + POWER_TOKEN]):
            self.type = PARAMETER
        else:
            try:
                name_ = name.replace(' ', '')
                float(name_)
                self.type = FLOAT
                self.name = name_
            except Exception as e:
                if np.any([z in name for z in '+-/*()' + POWER_TOKEN + NEG_POWER_TOKEN]):
                    self.type = EXPRESSION
                else:
                    self.type = VARIABLE

        self.build_children(debug=debug)

        assert len(self.children) <= 3

        if self.type == EXPRESSION:
            if self.is_leaf():
                raise Exception(f"An expression type cannot be a leaf! {self.name}")
            assert len(self.children) == 3 or (len(self.children) == 2 and self.children[0].name == '-')

        elif self.type == FUNCTION:
            assert len(self.children) == 1
            # print("@@@ ", self.name, self.type, self.func_name , self.func_name in self.DERIV_FUNC_NAME_DICT)
            assert self.func_name in self.DERIV_FUNC_NAME_DICT

        self.name = self.name.replace(POWER_TOKEN, '**')
        self.name = self.name.replace(NEG_POWER_TOKEN, '**-')
        while '  ' in self.name:
            self.name = self.name.replace('  ', ' ')

        self.prune_numeric()

    def prune_numeric(self):
        """Fold binary expressions whose both operands are numeric literals into a single FLOAT leaf."""
        if self.is_leaf():
            return
        elif len(self.children) == 3:
            l, op, r = self.children
            if l.type == FLOAT and r.type == FLOAT:
                self.type = FLOAT
                self.children = []
                exec_dict = dict(globals())
                exec(f'__temp__ = str({l.name}{op.name}{r.name})', exec_dict)
                self.name = exec_dict['__temp__']

    def build_children(self, debug=False):
        """Populate ``children`` for EXPRESSION nodes by additive / multiplicative / power / function splits."""

        # debug = True

        if self.type != EXPRESSION:
            self.children = []
            return

        # First, split all additive sparse_terms
        tokens = self.get_additive_term_strings(self.name)
        # if debug: print("\t**** ", self.name, '\n\t', tokens)
        # raise Exception
        if len(tokens) > 1:
            self.children = [AutoDiffGraphNode(t, parent=self, debug=debug) for t in tokens]
            return

        # This cannot be split additively, so split on arithemtic
        # *, /, ^
        # if len(tokens) < 1:
        # assert tokens[0] == self.name
        tokens = self.get_multiplicative_term_strings(self.name)
        # if debug: print("\tc ", self.name, tokens)
        if len(tokens) > 1:
            self.children = [AutoDiffGraphNode(t, parent=self, debug=debug) for t in tokens]
            return

        # Check for exponentiation
        if POWER_TOKEN in self.name or NEG_POWER_TOKEN in self.name:
            tokens = self.get_exponential_term_strings(self.name)
            if len(tokens) > 1:
                self.children = [AutoDiffGraphNode(t, parent=self, debug=debug) for t in tokens]
                return

        # This must be a function or an irreducible expression
        self.type = FUNCTION if '(' in self.name else EXPRESSION

        if self.type == FUNCTION:
            start = self.name.find('(')
            self.func_name = self.name[:start]
            end = self.name.rfind(')')
            self.children = [AutoDiffGraphNode(self.name[start:end + 1], parent=self, debug=debug)]
        else:
            self.children = []

        if self.type == FIXED_EFFECT:
            assert self.is_leaf()

    @staticmethod
    def _tokenize(name, delimiters):
        """Split ``name`` on ``delimiters`` characters only at parenthesis depth zero."""
        cur_token = ''
        tokens = []
        parenth_count = 0
        for i, c in enumerate(name):
            cur_token += c
            if c == '(':
                parenth_count += 1
            elif c == ')':
                parenth_count -= 1
            elif c in delimiters:
                if parenth_count == 0:
                    cur_token = cur_token[:-1]
                    tokens += [cur_token, c]
                    cur_token = ''

        tokens += [cur_token]
        return tokens

    @staticmethod
    def get_exponential_term_strings(name):
        """Split ``name`` on the outermost ``^`` / negative-power token into base and exponent substrings."""

        name = AutoDiffGraphNode._strip(name)

        if name[0] == '-':
            name = "(-1) * " + name[1:]

        parenth_count = 0
        tokens = []
        cur_token = ''

        for i, s in enumerate(name):
            cur_token += s
            if s == '(':
                parenth_count += 1
            elif s == ')':
                parenth_count -= 1
            elif s in (POWER_TOKEN, NEG_POWER_TOKEN):
                if parenth_count == 0:
                    tokens.append(AutoDiffGraphNode._strip(cur_token[:-1]))
                    tokens.append(POWER_TOKEN)
                    break

        if s == NEG_POWER_TOKEN:
            tokens.append('-' + name[i + 1:])
        else:
            tokens.append(name[i + 1:])

        tokens = [t for t in tokens if t != '']
        if len(tokens) == 0:
            return []

        if tokens[0] in (NEG_POWER_TOKEN, POWER_TOKEN):
            raise Exception

        return tokens

    @staticmethod
    def get_multiplicative_term_strings(name):
        """Split ``name`` on top-level ``*`` / ``/`` into a small list (numerator / denominator form if ``/``)."""
        name = AutoDiffGraphNode._strip(name)

        if name[0] == '-':
            name = "(-1) * " + name[1:]

        tokens = AutoDiffGraphNode._tokenize(name, '*/')

        if len(tokens) == 1:
            return tokens

        if '/' in tokens:
            numer_tokens = [tokens[0]]
            denom_tokens = []
            for j in range(1, len(tokens), 2):
                if tokens[j].strip() == '/':
                    denom_tokens.append(tokens[j + 1])
                elif tokens[j].strip() == '*':
                    numer_tokens.append(tokens[j + 1])
                else:
                    raise Exception

            tokens = [f"{' * '.join(numer_tokens)}", '/', f"{' * '.join(denom_tokens)}"]

        else:
            tokens = [tokens[0], '*', ' * '.join([t for t in tokens[2:] if t.strip() != '*'])]

        return tokens

    @staticmethod
    def get_additive_term_strings(name):
        """Split ``name`` on top-level ``+`` / ``-`` into summed vs subtracted chunks (may return length 1)."""
        name = AutoDiffGraphNode._strip(name)

        tokens = AutoDiffGraphNode._tokenize(name, '+-')
        tokens = [t for t in tokens if t != '']

        if tokens[0] == '-':
            tokens = ['0', '-'] + tokens[1:]

        if len(tokens) == 1:
            return tokens

        if '-' in tokens:
            add_tokens = [tokens[0]]
            sub_tokens = []
            for j in range(1, len(tokens), 2):
                if tokens[j] == '-':
                    sub_tokens.append(tokens[j + 1])
                else:
                    add_tokens.append(tokens[j + 1])

            if add_tokens[0] == '0' and len(add_tokens) > 1:
                add_tokens = add_tokens[1:]

            tokens = [f"{' + '.join(add_tokens)}", '-', f"{' + '.join(sub_tokens)}"]

        else:
            tokens = [tokens[0], '+', ' + '.join(tokens[2:])]

        return tokens

    @staticmethod
    def _strip(func_str):
        """Normalize whitespace, map ``**`` to internal power tokens, strip redundant parens and ``1*`` factors."""
        for j in ['\n', '\t']:
            func_str = func_str.replace(j, ' ')
        if func_str == '':
            return ''
        func_str = func_str.replace(' ', '')
        func_str = func_str.replace('**-', NEG_POWER_TOKEN)
        func_str = func_str.replace('**', POWER_TOKEN)
        func_str = func_str.replace('*+', '*')
        func_str = func_str.replace('/+', '/')
        func_str = func_str.replace('((+', '(')
        for c in ['+', '-']:
            while c * 2 in func_str:
                func_str = func_str.replace(c * 2, c)

        while '((1))' in func_str:
            func_str = func_str.replace('((1))', '(1)')
        func_str = func_str.replace('(1)*', '')
        func_str = func_str.replace('*(1)', '')

        while func_str[0] == '(' and func_str[-1] == ')':
            func_str_temp = func_str[1:-1]
            parenth_count = 0
            do_strip = True
            for i, s in enumerate(func_str_temp):
                if s == '(':
                    parenth_count += 1
                elif s == ')':
                    parenth_count -= 1
                if parenth_count < 0:
                    do_strip = False
                    break
            if do_strip:
                func_str = func_str_temp
                func_str = func_str.strip(' ')
            else:
                break

            # for z in ('+', '-', '/', '*', '^'):
            #     while f' {z}' in func_str:
            #         func_str = func_str.replace(f' {z}', z)
            #     while f'{z} ' in func_str:
            #         func_str = func_str.replace(f'{z} ', z)
            #     func_str = func_str.replace(z, f' {z} ')
            func_str.replace(' ', '')

        return func_str

    def __str__(self):
        """ASCII tree via :meth:`print_token`."""
        return self.print_token()

    def __repr__(self):
        """Same as :meth:`__str__` for interactive display."""
        return self.print_token()

    def print_token(self, level=0):
        """Pretty-print this node and descendants with indentation ``level``."""
        tab = "\t"
        s = (
            f'{tab * level} --> "{self.name}" <{self.type}'
            f'{(", " + self.func_name) if self.func_name is not None else ""}>'
        )
        if hasattr(self, 'children'):
            for c in self.children:
                s += '\n' + c.print_token(level + 1)
        return s

    def is_leaf(self):
        """True if this node has no child sub-expressions."""
        return len(self.children) == 0

    def is_operator(self):
        """True if ``type`` is one of ``OP_ADD``, ``OP_SUBTRACT``, ``OP_MULTIPLY``, ``OP_DIVIDE``, ``OP_EXPONENTIATION``."""
        return self.type[:3] == 'OP_'

    def is_root(self):
        """True if ``parent`` is None."""
        return self.parent is None

    def has_fixed_effect(self):
        """True if this subtree contains a ``FIXED_EFFECT`` node."""
        return self.type == FIXED_EFFECT or np.any([c.has_fixed_effect() for c in self.children])

    def get_partial_derivative_expression(self, arg_number, debug=False):
        """Return Python source string for ∂/∂``params[arg_number]`` of this subtree."""
        func_str = self._deriv_expression_internal(arg_number, debug)
        func_str = AutoDiffGraphNode._strip(func_str)
        func_str = func_str.replace(POWER_TOKEN, '**')
        func_str = func_str.replace(NEG_POWER_TOKEN, '**-')
        # func_str = func_str.replace('1.0*', '')  # incorrect, e.g. 101.0*x
        # func_str = func_str.replace('1*', '')  # incorrect, e.g. 41*x
        func_str = func_str.replace('**(1.0)', '')
        return func_str

    def _deriv_expression_internal(self, arg_number, debug=False):
        """Recursive symbolic differentiation; returns a fragment of Python source (not evaluated here)."""

        if self.type == FIXED_EFFECT_DERIVATIVE:
            return '0'

        if self.type == FIXED_EFFECT:
            if self.fe_param_numbers[0] <= arg_number < self.fe_param_numbers[1]:
                return f'np.ones(len(int_arr[:,{self.fe_param_int_arr_idx}])) * (int_arr[:,{self.fe_param_int_arr_idx}] == {self.fe_offset + arg_number - self.fe_param_numbers[0]})'
            else:
                return '0'

        arg = f'params[{arg_number}]'

        if arg not in self.name:
            if not self.has_fixed_effect():
                return '0'

        if self.is_leaf():
            if self.is_operator():
                return None
            elif self.type in (FLOAT, VARIABLE):
                return '0'
            elif self.type == PARAMETER:
                if self.name == arg:
                    return '1'
                else:
                    return '0'

            raise Exception("Should exit before here...")

        else:
            if self.type == FUNCTION:
                return (
                        self.DERIV_FUNC_NAME_DICT[self.func_name](self.children[0].name)
                        + " * (" + self.children[0].get_partial_derivative_expression(arg_number) + ")"
                )

            elif self.type == EXPRESSION:
                if len(self.children) == 2:
                    return f'(-{self.children[1].get_partial_derivative_expression(arg_number)})'

                elif len(self.children) == 3:
                    left, oper, right = self.children
                    dleft, dright = left.get_partial_derivative_expression(
                        arg_number), right.get_partial_derivative_expression(arg_number)

                    if dleft == '0' and dright == '0':
                        return '0'

                    if oper.name in ('+', '-'):
                        if dleft == '0':
                            return f'{oper.name if oper.name == "-" else ""}({dright})'
                        elif dright == '0':
                            return f'({dleft})'
                        return f'(({dleft}) {oper.name} ({dright}))'

                    elif oper.name == '*':
                        if dleft == '0':
                            return f'(({left.name}) * ({dright}))'
                        elif dright == '0':
                            return f'(({dleft}) * ({right.name}))'
                        return f'(({left.name}) * ({dright}) + ({dleft}) * ({right.name}))'

                    elif oper.name == '/':
                        if dleft == '0':
                            return (
                                f'((-({left.name})*({dright})'
                                f' / (({right.name}) ** 2 )))'
                            )
                        elif dright == '0':
                            return (
                                f'(({dleft}) / ({right.name}))'
                            )
                        else:
                            return (
                                f'((({dleft}) * ({right.name}) - ({left.name})*({dright}))'
                                f' / (({right.name}) ** 2 ))'
                            )
                    elif oper.name == '**':

                        # q(x) = f(x)^g(x)
                        # q'(x) = f(x)^g(x) g'(x) log(f(x)) + g(x) f'(x) f(x)^{g(x)-1}

                        f = left.name
                        df = dleft
                        g = right.name
                        dg = dright

                        if dg == '0' and df == '0':
                            return '0'
                        elif dg == '0':
                            if right.type == FLOAT:
                                g = float(g)
                                if abs(g - 1) > 1e-5:
                                    return f'({g}*({df}))*({f})**({g - 1})'
                                else:
                                    return f'({g}*({df}))'
                            else:
                                return f'({g})*({df})*({f})**(({g})-1)'
                        elif df == '0':
                            return f'({f})**({g})*({dg})*np.log({f})'
                        else:
                            return f'({f})**({g})*({dg})*np.log({f}) + ({g})*({df})*(({f})**(({g})-1))'

            print("ERROR IN DERIVING: ", (self.name, self.children, len(self.children)))

            raise Exception("Should exit before here...")

    def traverse_names(self):
        """Collect distinct sub-expression strings under this node (sorted by length) for Jacobian CSE."""
        val = []
        if self.type in [EXPRESSION, FUNCTION, FIXED_EFFECT]:
            val += [self.name]
        for n in self.children:
            val += n.traverse_names()
        return sorted(val, key=lambda z: len(z))

    def get_analytical_jacobian(self, num_args, nobs, other_args='', dense_threshold_mb=DEFAULT_DENSE_THRESHOLD_MB,
                                debug=False, do_jit=False):
        """Build and ``exec`` a Jacobian function ``params ->`` dense array or sparse ``csc_matrix``.

        Returns a dict with ``jacobian_callable``, ``func_str_code``, ``graph``, ``return_dense``, ``do_jit``.
        """

        return_dense = DenseThreshold.is_below_threshold_dim((nobs, num_args), dense_threshold_mb)
        if not return_dense:
            if debug:
                print("Can't do `jit` with sparse return value")
            do_jit = False
        if nobs == 1:
            return_dense = True

        deriv_strs = [self.get_partial_derivative_expression(arg_number, debug=debug) for arg_number in range(num_args)]
        sub_terms = self.traverse_names()
        # 
        # # print('\n\n\n\n')
        # for i, d in enumerate(deriv_strs):
        #     print(i, d)
        pd_nodes = [AutoDiffGraphNode(d) for d in deriv_strs]
        sub_terms = self.traverse_names()
        sub_terms += list(chain.from_iterable(p.traverse_names() for p in pd_nodes))
        sub_terms = sorted(set(sub_terms), key=lambda z: len(z))
        sub_terms_coded = []

        # Hoist repeated sub-expressions shared across columns into temps (cheap common subexpr elimination).
        for i, c in enumerate(sub_terms):
            if np.count_nonzero([c in d for d in deriv_strs]) >= 2:
                temp = f'___TEMP_{i}'
                sub_terms_coded.append(f'{temp} = {c}')
                for k, deriv_str in enumerate(deriv_strs):
                    deriv_strs[k] = deriv_str.replace(c, temp)
                for j in range(i + 1, len(sub_terms)):
                    sub_terms[j] = sub_terms[j].replace(c, temp)

        deriv_str_join = []
        if return_dense:
            if nobs == 1:
                deriv_str_join.append(f'jac_value = np.zeros({num_args})')
            else:
                deriv_str_join.append(f'jac_value = np.zeros(({nobs},{num_args}))')
            for i, d in enumerate(deriv_strs):
                if nobs == 1:
                    deriv_str_join.append(f'jac_value[{i}] = {d}')
                else:
                    deriv_str_join.append(f'jac_value[:,{i}] = {d}')

        else:
            deriv_str_join.append('data, indices, indptr = [], [], [0]')
            deriv_str_join.append('lo_ind = 0')
            for i, d in enumerate(deriv_strs):
                deriv_str_join.append('\n')
                deriv_str_join.append(f'temp = {d}')
                deriv_str_join.append(f'if np.shape(temp) == tuple(): temp = np.full({nobs}, temp)')
                # deriv_str_join.append(f'print({i}, temp, np.nonzero(temp))')
                deriv_str_join.append(f'nz = list(np.nonzero(temp)[0])')
                deriv_str_join.append(f'data += list(temp[nz])')
                deriv_str_join.append(f'indices += nz')
                deriv_str_join.append(f'indptr.append(lo_ind+len(nz))')
                deriv_str_join.append('lo_ind += len(nz)')
            deriv_str_join.append(f'\n')
            # deriv_str_join.append(f'jac_value = data, indices, indptr')
            deriv_str_join.append(f'jac_value = csc_matrix((data, indices, indptr), shape=({nobs}, {num_args}))')

        nt = '\n    '

        func_str_code = f"""
{"@jit" if do_jit else ""}
def __jacobian(params, {other_args}):

    # name = {self.name}
    # jit = {do_jit}
    # dense = {return_dense}

    {nt.join(sub_terms_coded)}
    
    {nt.join(deriv_str_join)}
   
    return jac_value
"""

        if debug:
            print(func_str_code)

        exec_dict = dict(globals())
        exec(func_str_code, exec_dict)
        __jacobian = exec_dict['__jacobian']

        def jacobian(params, *args, **kwargs):
            params = np.asarray(params)
            return __jacobian(params, *args, **kwargs)

        return {
            'jacobian_callable': jacobian,
            'func_str_code': func_str_code,
            'graph': self,
            'return_dense': return_dense,
            'do_jit': do_jit
        }

    def get_analytical_partial_derivative(self, arg, nobs, other_args='', debug=False, do_jit=False, return_info=False):
        """Return callable for column ``arg`` of the Jacobian (optionally with expression/code metadata)."""
        deriv_expr = self.get_partial_derivative_expression(arg, debug)
        func_str_code = f"""
{"@jit" if do_jit else ""}
def __partial_deriv_callable(params, {other_args}):
    return {deriv_expr}
"""
        exec_dict = dict(globals())
        exec(func_str_code, exec_dict)
        __partial_deriv_callable = exec_dict['__partial_deriv_callable']

        def partial_deriv_callable(params, *args, **kwargs):
            params = np.asarray(params)
            v = __partial_deriv_callable(params, *args, **kwargs)
            if np.ndim(v) == 0 and nobs > 1:
                return np.full(nobs, float(v))
            else:
                return v

        if return_info:
            return partial_deriv_callable, {
                'partial_derivative_expression': deriv_expr,
                'func_str_code': func_str_code,
                'graph': self,
                'arg': arg,
                'do_jit': do_jit
            }
        else:
            return partial_deriv_callable

    def get_analytical_partial_derivatives(self, num_args, nobs, other_args='', debug=False, do_jit=False,
                                           return_info=False):
        """List of partial derivative callables for ``arg`` in ``0 .. num_args-1`` (pair with info if requested)."""
        result = [
            self.get_analytical_partial_derivative(
                arg, nobs, other_args=other_args, debug=debug, do_jit=do_jit, return_info=return_info)
            for arg in range(num_args)]
        if return_info:
            return [r[0] for r in result], [r[1] for r in result]
        else:
            return result

    def get_analytical_hessian(self, num_args, nobs, other_args='', debug=False, do_jit=False, agg_func=None,
                               assume_symmetric=True, return_info=True):
        """Hessian of ``mean`` or ``mean_squared`` scalar summary; requires ``agg_func`` when ``nobs > 1``."""

        if agg_func is not None:
            if agg_func == 'mean':
                node = self
            elif agg_func == 'mean_squared':
                node = AutoDiffGraphNode(f'({self.name})**2')
            else:
                raise Exception('`agg_func` must be "mean" or "mean_squared"')
        else:
            node = self

        # print(f"\n\n\nNODE = \n{node}\n\n\n")

        if nobs != 1 and agg_func is None:
            raise Exception("Need to specify aggregation `agg_func` if nobs > 1!")

        # if nobs != 1:
        #    raise NotImplementedError(
        #        "Haven't yet implemented `agg_func` case for Hessian when function is vector-values!")

        deriv_exprs = [node.get_partial_derivative_expression(arg) for arg in range(num_args)]
        pd_nodes = [AutoDiffGraphNode(d) for d in deriv_exprs]

        second_deriv_exprs = [[pdn.get_partial_derivative_expression(arg) for arg in range(num_args)]
                              for pdn in pd_nodes]

        if debug:
            print("Second Deriv Expressions:")
            for i in range(num_args):
                for j in range(num_args):
                    print('\t', i, j, second_deriv_exprs[i][j])

        func_str_code = (
            f'\n{"@jit" if do_jit else ""}'
            f'\ndef __temp__(params, {other_args}):'
            f'\n    __hess = np.zeros(({num_args},{num_args}), dtype=float)'
        )
        for i in range(num_args):
            for j in range((i + 1) if assume_symmetric else num_args):
                func_str_code += (
                    f'\n'
                    f'\n    __hess[{i},{j}] = np.mean({second_deriv_exprs[i][j]})'
                )

        if assume_symmetric:
            for i in range(num_args):
                for j in range(i + 1, num_args):
                    func_str_code += f'\n\n    __hess[{i},{j}] = __hess[{j},{i}]'

        func_str_code += '\n\n    return __hess\n'

        if debug:
            print(func_str_code)

        exec_dict = dict(globals())
        exec(func_str_code, exec_dict)
        __hessian = exec_dict['__temp__']

        def hessian_callable(params, *args, **kwargs):
            params = np.asarray(params)
            return __hessian(params, *args, **kwargs)

        if return_info:
            return hessian_callable, {
                'second_derivative_expressions': second_deriv_exprs,
                'func_str_code': func_str_code,
                'graph': self,
                'do_jit': do_jit
            }
        else:
            return hessian_callable

    def get_analytical_gradient(self, num_args, nobs, other_args='', debug=False, do_jit=False, agg_func=None,
                                return_info=True):
        """Gradient of ``mean`` or ``mean_squared`` scalar summary; requires ``agg_func`` when ``nobs > 1``."""

        if agg_func is not None:
            if agg_func == 'mean':
                node = self
            elif agg_func == 'mean_squared':
                node = AutoDiffGraphNode(f'({self.name})**2')
            else:
                raise Exception('`agg_func` must be "mean" or "mean_squared"')
        else:
            node = self

        if nobs != 1 and agg_func is None:
            raise Exception("Need to specify aggregation `agg_func` if nobs > 1!")

        deriv_exprs = [node.get_partial_derivative_expression(arg) for arg in range(num_args)]

        func_str_code = (
            f'\n{"@jit" if do_jit else ""}'
            f'\ndef __temp__(params, {other_args}):'
            f'\n    __grad = np.zeros({num_args}, dtype=float)'
        )
        for i in range(num_args):
            func_str_code += (
                f'\n'
                f'\n    __grad[{i}] = np.mean({deriv_exprs[i]})'
            )

        func_str_code += '\n\n    return __grad\n'

        if debug:
            print(func_str_code)

        exec_dict = dict(globals())
        exec(func_str_code, exec_dict)
        __gradient = exec_dict['__temp__']

        def gradient_callable(params, *args, **kwargs):
            params = np.asarray(params)
            return __gradient(params, *args, **kwargs)

        if return_info:
            return gradient_callable, {
                'func_str_code': func_str_code,
                'graph': self,
                'do_jit': do_jit
            }
        else:
            return gradient_callable
