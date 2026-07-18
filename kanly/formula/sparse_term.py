"""Term parsing primitives for sparse Patsy-like formula handling.

This module defines:
  - ``NumericalControl`` for parsing numeric controls like ``L(...)`` and
    ``center(...)``.
  - ``SparseTerm`` for representing formula sparse_terms as categorical and numerical
    factors, plus ordering/comparison/drop-one logic.
"""
from __future__ import absolute_import, print_function

import itertools
import pprint
import re
from collections import UserString

import numpy as np

from kanly.formula.util import temp_replace, is_float
from kanly.utils.util import str_to_args


class NumericalControl(object):
    """Parsed representation of one numerical control token from a formula term."""
    def __init__(self, term_name):
        """Parse lag/center wrappers around an underlying numeric token.

        Args:
            term_name (str): Raw token text, e.g. ``"x"``, ``"L(x,2,g)"``,
                or ``"center(x,w)"``.
        """
        self.term_name_original_string = term_name
        self.term_name = term_name

        self.term_name, self.center_weights, self.center = self.parse_center_args(self.term_name)
        self.term_name, self.standardize_weights, self.standardize = self._parse_standardize_args(self.term_name)

        if self.center and self.standardize:
            raise Exception("Cannot center and standardize!")

        self.term_name, self.lags, self.lag_groups = self.parse_lag_args(self.term_name)

       # trend values are exponents on trend, (linear trend)**e
        self.term_name, self.is_trend_term, self.trend_exponents = self.parse_trend_args(self.term_name)

        # seasonal values are period of seasonality, e.g. seasonal(12) defines 12-period seasonality fixed effects
        self.term_name, self.is_seasonal_term, self.seasonal_periods = self.parse_seasonal_args(self.term_name)

        (self.term_name, self.is_bspline_term, self.bspline_degree, self.bspline_df, self.bspline_include_intercept,
         self.bspline_knots, self.bspline_lower_bound, self.bspline_upper_bound
         ) = self.parse_bspline_args(self.term_name)

    def is_lag_term(self):
        """Return whether this control is a lag/lead transform."""
        return self.lags is not None

    def is_center(self):
        """Return whether this control uses ``center(...)`` demeaning."""
        return self.center

    def is_standardize(self):
        """Return whether this control uses ``standardize(...)`` scaling."""
        return self.standardize

    def has_lag_groups(self):
        """Return whether lagging is group-specific."""
        return self.lag_groups is not None

    def __eq__(self, other):
        """Compare parsed control attributes for semantic equality."""
        if not isinstance(other, NumericalControl):
            return False
        return (self.term_name == other.term_name and
                self.lag_groups == other.lag_groups and
                self.lags == other.lags and
                self.term_name_original_string == other.term_name_original_string)

    def __hash__(self):
        """Hash based on string representation for set/dict usage."""
        return hash(str(self))

    def __str__(self):
        """Pretty-print parsed attributes for debugging."""
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        """Mirror ``__str__`` in interactive displays."""
        return str(self)

    @staticmethod
    def _parse_standardize_args(name):
        """Parse ``standardize(var[, weights])`` wrapper if present.

        Args:
            name (str): Candidate control text.

        Returns:
            tuple: ``(base_name, center_weights, is_demean)``.
        """
        if name.strip()[:12] == 'standardize(':
            args = str_to_args(name.strip()[12:-1])
            if len(args) == 1:
                name, standardize_weights = args[0], None
            elif len(args) == 2:
                name, demean_weights = args[0], args[1]
            return name, standardize_weights, True
        else:
            return name, None, False

    @staticmethod
    def _parse_string_to_bool_or_int(x: str):
        x = x.replace(' ', '')
        if (xl := x.lower()) in ('true', 'false'):
            return xl == 'true'
        try:
            return int(x)
        except ValueError:
            raise ValueError(f"Cannot parse '{x}' as bool or int.")

    @staticmethod
    def parse_bspline_args(name):
        if name.strip()[:3] == 'bs(':
            args = str_to_args(name.strip()[3:-1])
            term_name = args[0]
            arg_names = ['degree', 'df', 'include_intercept', 'knots', 'lower_bound', 'upper_bound']
            if len(args) - 1:
                arg_dict = {'include_intercept': False, 'degree': 3, 'df': 6, 'knots': None,
                            'lower_bound': None, 'upper_bound': None}
                has_kw_arg = False
                for i, z in enumerate(args[1:]):
                    spl = z.split('=')
                    if len(spl) == 1:
                        if has_kw_arg:
                            raise Exception
                        arg_dict[arg_names[i]] = NumericalControl._parse_string_to_bool_or_int(spl[0])
                    elif len(spl) == 2:
                        has_kw_arg = True
                        arg_dict[spl[0].replace(' ', '')] = NumericalControl._parse_string_to_bool_or_int(spl[1])
                    else:
                        raise Exception
            return (
                term_name, True, arg_dict['degree'], arg_dict['df'], arg_dict['include_intercept'],
                arg_dict['knots'], arg_dict['lower_bound'], arg_dict['upper_bound']
            )

        else:
            return name, False, None, None, None, None, None, None

    @staticmethod
    def parse_seasonal_args(name):
        """Parse ``seasonal(seasonal_periods)`` wrapper if present.

        Args:
            name (str): Candidate control text.

        Returns:
            tuple: ``(term_name, is_seasonal_term, seasonal_periods)``.
        """
        if name.strip()[:9] == 'seasonal(':
            args = str_to_args(name.strip()[9:-1])
            assert len(args) == 1
            args = args[0].strip()
            if args[0] == '[':
                seasonal_periods = tuple(sorted(set([int(e) for e in str_to_args(args[1:-1])])))
            else:
                seasonal_periods = tuple([int(args)])

            # Phase 1: drop any period whose span is fully contained in a larger period's span
            seasonal_periods = [
                p for p in seasonal_periods
                if not any(q % p == 0 and q != p for q in seasonal_periods)
            ]

            return name, True, seasonal_periods
        else:
            return name, False, None

    @staticmethod
    def parse_trend_args(name):
        """Parse ``trend(trend_exponents)`` wrapper if present.

        Args:
            name (str): Candidate control text.

        Returns:
            tuple: ``(term_name, is_trend_term, trend_exponents)``.
        """
        if name.strip()[:6] == 'trend(':
            args = str_to_args(name.strip()[6:-1])
            assert len(args)==1
            args = args[0].strip()
            if args[0] == '[':
                trend_exponents = tuple(sorted(set([int(e) for e in str_to_args(args[1:-1])])))
            else:
                trend_exponents = tuple([int(args)])
            return name, True, trend_exponents
        else:
            return name, False, None

    @staticmethod
    def parse_center_args(name):
        """Parse ``center(var[, weights])`` wrapper if present.

        Args:
            name (str): Candidate control text.

        Returns:
            tuple: ``(base_name, center_weights, is_demean)``.
        """
        if name.strip()[:7] == 'center(':
            args = str_to_args(name.strip()[7:-1])
            if len(args) == 1:
                name, demean_weights = args[0], None
            elif len(args) == 2:
                name, demean_weights = args[0], args[1]
            print('\n\t', name)
            return name, demean_weights, True
        else:
            return name, None, False

    @staticmethod
    def parse_lag_args(name):
        """Parse ``L(var[, lags[, groups]])`` wrapper if present.

        Args:
            name (str): Candidate control text.

        Returns:
            tuple: ``(base_name, lags_or_none, lag_groups_or_none)``.

        Raises:
            Exception: If too many lag arguments are provided.
        """
        if name.strip()[:2] == 'L(':
            v = name.strip()[2:-1]
            args = str_to_args(v)
            if len(args) == 1:
                var, lags, lag_groups = args[0], 1, None
            elif len(args) == 2:
                var, lags, lag_groups = args[0], int(args[1]), None
            elif len(args) == 3:
                grps = args[2].strip()
                if grps[0] == '[':
                    grps = tuple(sorted(set(str_to_args(grps[1:-1]))))
                else:
                    grps = tuple([grps])
                var, lags, lag_groups = args[0], int(args[1]), grps
            else:
                raise Exception('L(.) function can have at most 1-3 args!')

            return var, lags, lag_groups

        else:
            return name, None, None


class CategoricalControl(UserString):

    def __init__(self, name):
        super().__init__(name)
        self.state = dict()


INTERCEPT = CategoricalControl('Intercept')


class SparseTerm(object):
    """Formula term represented as categorical and numerical control factors."""

    FULL_CATEGORICAL = 'FULL_CATEGORICAL'
    NUMERICAL = 'NUMERICAL'
    MIXED = 'MIXED'

    def __init__(self, categorical_controls=None, numerical_controls=None, var_name=None):
        """Build a normalized term object from control lists.

        Args:
            categorical_controls (list[str] or None): Categorical factors.
            numerical_controls (list[NumericalControl] or None): Numeric factors.
            var_name (str or None): Original source term string.

        Raises:
            Exception: If both categorical and numerical controls are empty.
        """

        if categorical_controls is None:
            categorical_controls = []
        if numerical_controls is None:
            numerical_controls = []

        self.categorical_controls: list[CategoricalControl] = sorted(set(categorical_controls))
        self.numerical_controls: list[NumericalControl] = sorted(set(numerical_controls), key=lambda t: t.term_name_original_string)

        if len(self) == 0:
            raise Exception("Must supply variables!!")

        if len(self.numerical_controls) == 0:
            self.term_type = self.FULL_CATEGORICAL
        elif len(self.categorical_controls) > 0:
            self.term_type = self.MIXED
        else:
            self.term_type = self.NUMERICAL

        self.var_name = var_name
        self.state = {
            'numerical': {c.term_name_original_string: dict() for c in self.numerical_controls},
            'categorical': dict(),
        }

    def is_numerical(self):
        """Return True when the term contains only numerical controls."""
        return self.term_type == self.NUMERICAL

    def is_full_categorical(self):
        """Return True when the term contains only categorical controls."""
        return self.term_type == self.FULL_CATEGORICAL

    def is_mixed(self):
        """Return True when the term mixes categorical and numerical controls."""
        return self.term_type == self.MIXED

    def is_intercept(self):
        """Return True when this term is exactly the synthetic intercept term."""
        return self == SparseTerm(categorical_controls=[INTERCEPT])

    def is_monomial(self):
        """Return True when term has exactly one total factor."""
        return len(self.numerical_controls) + len(self.categorical_controls) == 1

    def __hash__(self):
        """Hash by canonical string form."""
        return hash(str(self))

    def __str__(self):
        """Render term as colon-joined text (categoricals as ``C(...)``)."""
        return ":".join(
            [str(s.term_name_original_string) for s in self.numerical_controls]
            + [f"C({s})" if s != INTERCEPT else 'Intercept' for s in self.categorical_controls]
        )

    def __lt__(self, other):
        """Strict subset-like ordering across categorical/numerical sets."""
        return self <= other and self != other

    def __le__(self, other):
        """Subset-like ordering across categorical/numerical sets."""
        return (
                set(self.categorical_controls) <= set(other.categorical_controls)
                and set(self.numerical_controls) <= set(other.numerical_controls)
        )

    def __gt__(self, other):
        """Strict superset-like ordering across categorical/numerical sets."""
        return self >= other and self != other

    def __ge__(self, other):
        """Superset-like ordering across categorical/numerical sets."""
        return (
                set(self.categorical_controls) >= set(other.categorical_controls)
                and set(self.numerical_controls) >= set(other.numerical_controls)
        )

    def __eq__(self, other):
        """Return semantic equality of categorical/numerical factor sets."""
        return (
            set(self.categorical_controls) == set (other.categorical_controls)
            and set(self.numerical_controls) == set(other.numerical_controls)
        )

    def __neq__(self, other):
        """Return inverse of ``__eq__``."""
        return not self == other

    def __len__(self):
        """Return total number of categorical + numerical factors."""
        return len(self.categorical_controls) + len(self.numerical_controls)

    def __repr__(self):
        """Mirror ``__str__`` for debugging output."""
        return str(self)

    def get_col_labels_for_term(self):
        """Get tokenized column labels for this term's original text."""
        return self._get_tokens_for_term(self.var_name)

    @staticmethod
    def get_col_labels_for_terms(terms):
        """Collect unique column tokens across a list of sparse_terms."""
        return set(itertools.chain.from_iterable([t.get_tokens_for_term()['column_tokens'] for t in terms]))

    @staticmethod
    def parse_to_list(term_string):
        """Parse a term string into ``('categorical'|'numerical', name)`` tuples.

        Args:
            term_string (str): Single term text, potentially with ``:``.

        Returns:
            tuple or list: A tuple for monomials, or list of tuples for
            interactions split by ``:``.
        """

        term_strings = term_string.split(':')
        term_strings = [t.strip() for t in term_strings]

        if len(term_strings) == 1:
            regressor = term_strings[0]
        else:
            return [SparseTerm.parse_to_list(_s) for _s in term_strings]

        z = re.search(r'(^C\().*(\))', regressor)
        if z:
            return 'categorical', regressor[2:-1]

        return 'numerical', regressor

    @staticmethod
    def parse_to_term(term_string):
        """Parse term text into a normalized ``SparseTerm`` object."""

        var_list = SparseTerm.parse_to_list(term_string)
        cat_controls = []
        num_controls = []

        if isinstance(var_list, tuple):
            var_list = [var_list]

        for _type, name in var_list:

            if _type == 'categorical':
                cat_controls.append(CategoricalControl(name))
            else:
                num_controls.append(NumericalControl(name))

        return SparseTerm(cat_controls, num_controls, var_name=term_string)

    @staticmethod
    def parse_to_terms(term_strings, do_absorb=False, debug=False):
        """Parse a list of term strings and compute drop-one metadata.

        Handles intercept insertion/removal and prunes nested/redundant sparse_terms
        according to categorical/mixed containment rules.

        Args:
            term_strings (list[str]): RHS term strings.
            do_absorb (bool): Whether absorb logic is active.
            debug (bool): Print dropped sparse_terms during pruning.

        Returns:
            tuple: ``(terms_array, drop1_dict)``.
        """

        add_intercept = "-1" not in term_strings
        if not add_intercept:
            term_strings.remove('-1')

        terms = np.array(sorted(
            ([SparseTerm(categorical_controls=[INTERCEPT], numerical_controls=[], var_name='Intercept')]
            if add_intercept else [])
            + [SparseTerm.parse_to_term(t) for t in term_strings], key=len))

        n = len(terms)
        valid = np.array([True] * n)

        for i in range(n):
            for j in range(i + 1, n):

                if (
                        terms[i] <= terms[j] and
                        ((terms[i].term_type == SparseTerm.FULL_CATEGORICAL
                          and terms[j].term_type == SparseTerm.FULL_CATEGORICAL)
                         or (terms[i].term_type == SparseTerm.NUMERICAL
                             and terms[j].term_type == SparseTerm.MIXED)
                        )
                ):
                        valid[i] = False
                        if debug:
                            print(f"\n\tDropping {str(terms[i])}' contained in {str(terms[j])}")
                        break

        if len(terms):
            terms = terms[valid]

        return terms, SparseTerm.build_drop_1_list(terms, do_absorb)

    def to_monomials(self):
        """
        Split this term into single-factor monomial sparse_terms.
        Preserves state --- shared state dict across these children.
        Mods to "children" monomials affect "parent" combined term
        """
        result = []
        for n in self.numerical_controls:
            temp = SparseTerm(numerical_controls=[n])
            temp.state['numerical'][n.term_name_original_string] = self.state['numerical'][n.term_name_original_string]
            result.append(temp)

        # We do not split categoricals into "monomials"
        # because they are parsed together anyway
        if len(self.categorical_controls):
            temp = SparseTerm(categorical_controls=self.categorical_controls)
            temp.state['categorical'] = self.state['categorical']
            result.append(temp)

        return result

    @staticmethod
    def build_drop_1_list(term_list, do_absorb=False):
        """Build mapping indicating which categorical blocks should drop one level.

        Args:
            term_list (Sequence[SparseTerm]): Ordered parsed sparse_terms.
            do_absorb (bool): Whether absorb/fixed effects are present.

        Returns:
            dict[str, bool]: Map from original term name to drop-one flag.
        """

        if do_absorb:
            has_intercept = False
        else:
            has_intercept = term_list[0] == SparseTerm(categorical_controls=[INTERCEPT])

        monomials_recovered = {'Intercept'} if has_intercept or do_absorb else set()
        drop1 = {'Intercept': False}

        numerical_interacted_list = set()
        num_seasonal = 0

        for t in term_list[has_intercept:]:

            if t.term_type == t.FULL_CATEGORICAL:
                drop1[t.var_name] = 'Intercept' in monomials_recovered
                monomials_recovered.add('Intercept')

            elif t.term_type == t.MIXED:
                # numerical interacted multiple categorical separate
                # sparse_terms check #num_ctrl = t.numerical controls[0][1]

                num_ctrl = ":".join(sorted(
                    [x.term_name for x in t.numerical_controls]
                ))

                if num_ctrl in numerical_interacted_list:
                    drop1[t.var_name] = True
                else:
                    numerical_interacted_list.add(num_ctrl)
                    drop1[t.var_name] = num_ctrl in monomials_recovered

            elif t.numerical_controls[0].is_seasonal_term:
                drop1[t.var_name] = True if has_intercept else num_seasonal > 0
                num_seasonal += 1

            else:
                monomials_recovered.add(t.numerical_controls[0].term_name)
                drop1[t.var_name] = False

        return drop1

    def get_tokens_for_term(self):
        """Convenience wrapper around ``_get_tokens_for_term``."""
        return SparseTerm._get_tokens_for_term(self.var_name)

    @staticmethod
    def _get_tokens_for_term(var_name):
        """Extract referenced data-column tokens and global function tokens.

        The parser masks quoted/Q(...) segments, removes known syntax/function
        scaffolding, and then classifies identifiers as either data columns
        or global function references.

        Args:
            var_name (str): Term text to tokenize.

        Returns:
            dict: ``{'column_tokens': set[str], 'global_tokens': set[str]}``.
        """

        v = var_name.replace('\'', '"')
        v, q_dict = temp_replace(v, r'Q\(.*?\)', '@')
        v, dbt_qt_dict = temp_replace(v, r'".*?"', '#')
        for c in ['C(', 'I(',  'poly(', 'Q(', 'L('] + list('`~!@#$%^&*()<>,/?;:[]()+=\\|'):
            v = v.replace(c, ' ')
        tokens = [s.strip() for s in v.split(' ')]
        tokens = set([s for s in tokens if s != '' and not is_float(s)])
        tokens |= set([re.findall(r'".*?"', q)[0].replace('"', '') for q in q_dict.keys()])

        # remove function calls
        global_func_remove = set()
        for t in tokens:
            if t.split('.')[0] in globals():
                global_func_remove.add(t)

        column_tokens = tokens - global_func_remove

        return {'column_tokens': column_tokens, 'global_tokens': global_func_remove}
