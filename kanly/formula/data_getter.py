"""Primary sparse formula data-construction pipeline.

``SparseDataGetter`` coordinates formula parsing, term-to-matrix conversion,
null-row alignment, fixed-effect handling, and optional IV/weights metadata.
It is the main backend for formula-based APIs across regression modules.
"""
from __future__ import absolute_import, print_function

import itertools
import time

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, hstack as sphstack

from kanly.formula.formula_design_info import FormulaDesignInfo
from kanly.formula.exceptions import (MissingDataException, AbsorbAndNoInterceptException)
from kanly.formula.invalid_row_functions import (
    default_weights_invalid_row_func, default_numerical_invalid_row_func, default_categorical_invalid_row_func)
from kanly.formula.keys import (ENDOG_KEY, ENDOG_REGRESSORS_KEY, EXOG_KEY, INDEX_KEY, NULL_ROWS_INFO_DICT_KEY,
                                INSTRUMENTS_KEY, INSTRUMENT_REGRESSORS_KEY, ABSORB_KEY, ABSORB_NAME_KEY, WEIGHTS_KEY,
                                VALID_OBS_ROWS_KEY, HAS_INTERCEPT_KEY, HAS_IMPLICIT_CONSTANT_KEY, TIME_ELAPSED_KEY,
                                FORMULA_DESIGN_INFO_KEY)
from kanly.formula.parse_formula import get_null_indices_for_formula, parse_formula
from kanly.formula.sparse_formula_data_object import SparseFormulaDataObj
from kanly.formula.sparse_term import SparseTerm
from kanly.formula.sparse_term_to_data_methods import (get_categorical_control_data, get_numerical_control_data,
                                                       get_nobs_from_index)
from kanly.sparse_data_frame import SparseDataFrame
from kanly.utils.parse_string_2_tuple import parse_str_2_tuple
from kanly.utils.util import dict_2_dataframe


class SparseDataGetterException(Exception):
    """Raised for invalid usage of sparse data getter helpers."""
    pass


class SparseDataGetter(object):
    """Facade for building sparse design/response matrices from formulas."""

    @staticmethod
    def _adjust_var_name_list_for_no_intercept(var_list):
        """Normalize trailing ``-1`` marker placement in variable list."""
        if var_list[-1][-2:] == '-1':
            var_list = var_list[:-1] + [var_list[-1][:-2]] + ['-1']
        return var_list

    @staticmethod
    def get_columns_for_term(term, data, term_dict=None, debug=False, numerical_invalid_row_func=None,
                             categorical_invalid_row_func=None, index=None):
        """
        return column data, column names, and indices of null values

        Args:
            term (SparseTerm): Parsed term to evaluate.
            data (DataFrame): Source data.
            term_dict (dict, optional): Cache of already-built term matrices.
            debug (bool): Print cache/build diagnostics.
            numerical_invalid_row_func (callable, optional): Numeric null rule.
            categorical_invalid_row_func (callable, optional): Categorical rule.
            index (array-like, optional): Optional row subset.

        Returns:
            SparseFormulaDataObj: Matrix block for the requested term.
        """

        if term_dict is not None:
            if term in term_dict.keys():
                if debug:
                    print("\n\t\tterm %s already cached! "  % str(term), end="")
                return term_dict[term]

        if numerical_invalid_row_func is None:
            numerical_invalid_row_func = default_numerical_invalid_row_func
        if categorical_invalid_row_func is None:
            categorical_invalid_row_func = default_categorical_invalid_row_func

        n_rows, index = get_nobs_from_index(data, index)

        if term.is_intercept():
            return SparseFormulaDataObj(
                coo_matrix((np.ones(n_rows), (range(n_rows), [0] * n_rows)),
                           shape=(n_rows, 1)).tocsc(),
                column_names=['Intercept'], null_rows=set())

        elif term.is_full_categorical():

            return get_categorical_control_data(
                term, data, term_dict=term_dict, debug=debug,
                invalid_row_func=categorical_invalid_row_func, index=index)

        elif len(term) == 1:

            if term.is_numerical():
                return get_numerical_control_data(
                    term, data, debug=debug, term_dict=term_dict,
                    invalid_row_func=numerical_invalid_row_func,
                    return_type='sparse', index=index)
            else:
                raise Exception("Should not have a 1-length term that isn't numerical (or was categorical before)!")

        else:
            results_multinomial = []

            monomial_terms = term.to_monomials()
            for x in monomial_terms:
                result = SparseDataGetter.get_columns_for_term(
                    x, data, term_dict=term_dict, debug=debug,
                    numerical_invalid_row_func=numerical_invalid_row_func,
                    categorical_invalid_row_func=categorical_invalid_row_func,
                    index=index
                )
                results_multinomial.append(result)

            return SparseDataGetter.reduce_columns_for_multinomial(results_multinomial)

    @staticmethod
    def _get_term_dict(cache_intermediate):
        """Interpret cache configuration and return a cache dict or None."""
        if isinstance(cache_intermediate, dict):
            return cache_intermediate
        elif isinstance(cache_intermediate, bool):
            return dict() if cache_intermediate else None
        else:
            raise Exception("`cache_intermediate` must be bool or dict!")

    @staticmethod
    def _sparse_dmatrix_internal(var_names, data, do_absorb=False, debug=False,
                                 check_constant_cols=True, is_endog_regressor=None,
                                 cache_intermediate=True, _time=None, return_dense=False,
                                 numerical_invalid_row_func=None, categorical_invalid_row_func=None,
                                 drop_1_for_FE=True, name=None, index=None):

        if _time is None:
            _time = time.time()

        if debug:
            print("\nBuilding sparse_terms for evaluation: vars='%s'..." % str(var_names), end='')

        terms, drop1_from_term_dict = SparseTerm.parse_to_terms(var_names, do_absorb=do_absorb, debug=debug)

        if debug:
            print("%.3f s" % (time.time() - _time))

        return SparseDataGetter._sparse_dmatrix_internal_from_terms(
            terms, data=data,
            do_absorb=do_absorb, debug=debug,
            check_constant_cols=check_constant_cols, is_endog_regressor=is_endog_regressor,
            cache_intermediate=cache_intermediate, _time=_time,
            return_dense=return_dense,
            numerical_invalid_row_func=numerical_invalid_row_func,
            categorical_invalid_row_func=categorical_invalid_row_func,
            drop_1_for_FE=drop_1_for_FE,
            drop1_from_term_dict=drop1_from_term_dict,
            name=name, index=index
        )

    @staticmethod
    def _sparse_dmatrix_internal_from_terms(
            terms, data, do_absorb=False, debug=False,
            check_constant_cols=True, is_endog_regressor=None,
            cache_intermediate=True, _time=None, return_dense=False,
            numerical_invalid_row_func=None, categorical_invalid_row_func=None,
            drop_1_for_FE=True, drop1_from_term_dict=None,
            name=None, index=None, keep_all_columns=None):
        """Build one sparse design matrix block from parsed RHS term names.

        Args:
            var_names (list[str]): Parsed RHS term names.
            data (DataFrame): Source data.
            do_absorb (bool): Whether absorb/fixed-effects mode is enabled.
            debug (bool): Print verbose build diagnostics.
            check_constant_cols (bool): Drop all-constant redundant columns.
            is_endog_regressor (dict, optional): IV metadata by term name.
            cache_intermediate (bool|dict): Cache controls for term matrices.
            _time (float, optional): Timing anchor for debug output.
            return_dense (bool): Return dense ndarray instead of sparse matrix.
            numerical_invalid_row_func (callable, optional): Numeric null rule.
            categorical_invalid_row_func (callable, optional): Categorical rule.
            drop_1_for_FE (bool): Apply drop-one coding for FE blocks.
            name (str, optional): Block name.
            index (array-like, optional): Optional row subset.

        Returns:
            SparseFormulaDataObj: Built design matrix with metadata.
        """

        if numerical_invalid_row_func is None:
            numerical_invalid_row_func = default_numerical_invalid_row_func
        if categorical_invalid_row_func is None:
            categorical_invalid_row_func = default_categorical_invalid_row_func

        if debug:
            print("Getting raw design matrices...", end='')

        term_dict = SparseDataGetter._get_term_dict(cache_intermediate)

        design_matrices = []
        for i, t in enumerate(terms):
            if debug:
                print(("\n" if i == 0 else "") + "\tGetting columns for %s ... " % t.var_name, end='')
            design_matrices.append(
                SparseDataGetter.get_columns_for_term(
                    t, data, term_dict=term_dict, debug=debug,
                    numerical_invalid_row_func=numerical_invalid_row_func,
                    categorical_invalid_row_func=categorical_invalid_row_func,
                    index=index
                )
            )
            if debug:
                print("shape=%s, no. invalid rows=%d, nnz=%d, time=%.3f" % (
                    str(design_matrices[-1].values.shape), len(design_matrices[-1].null_rows),
                    design_matrices[-1].values.nnz, time.time() - _time))

        if is_endog_regressor is not None:
            endog_reg_cols = np.array(
                [np.array([is_endog_regressor.get(t.var_name, False)] * d.values.shape[1])
                 for d, t in zip(design_matrices, terms)], dtype=object)
        else:
            endog_reg_cols = None

        nan_rows = set.union(*[r.null_rows for r in design_matrices])

        if debug:
            print("Raw design matrices done! (%.3f s)" % (time.time() - _time))

        nobs_temp, index = get_nobs_from_index(data, index)

        has_non_const_or_keep_cols = np.array([True] * len(terms))

        if check_constant_cols or drop_1_for_FE:

            valid_row_boolean_idx = np.array([True] * nobs_temp)
            valid_row_boolean_idx[list(nan_rows)] = False

            if debug:
                print("Finding constant columns...", end='')

            num_dropped_const_cols = 0

            found_constant = False   # we don't drop a constant column if it is the first example (i.e. intercept)
            for i, term in enumerate(terms):
                mat, cols_mat = design_matrices[i].values, np.array(design_matrices[i].column_names)

                # for when we predict
                if keep_all_columns:
                    cols_to_keep = np.array([True] * mat.shape[1])
                else:

                    if 'cols_to_keep' in term.state:
                        cols_to_keep, found_constant = term.state['cols_to_keep']
                    else:
                        cols_to_keep, found_constant = SparseDataGetter.get_nonconstant_cols_from_csc_matrix(
                            mat, found_constant, valid_row_boolean_idx=valid_row_boolean_idx if len(nan_rows) else None)
                        term.state['cols_to_keep'] = cols_to_keep, found_constant
                    num_dropped_const_cols += mat.shape[1] - cols_to_keep.sum()

                if np.sum(cols_to_keep) == 0:
                    has_non_const_or_keep_cols[i] = False
                elif np.sum(cols_to_keep) == mat.shape[1]:
                    pass
                else:
                    mat, cols_mat = mat[:, cols_to_keep], list(cols_mat[cols_to_keep])
                    design_matrices[i] = SparseFormulaDataObj(
                        mat, cols_mat, nan_rows, term_names=design_matrices[i].term_names,
                        endog_reg_cols=design_matrices[i].endog_reg_cols)
                    if endog_reg_cols is not None:
                        endog_reg_cols[i] = endog_reg_cols[i][cols_to_keep]

                if debug:
                    print(('\n' if i == 0 else "") + "\t %d of %d all-constant cols for %s in in %.3f s"
                          % (mat.shape[1] - cols_to_keep.sum(), mat.shape[1], term.var_name, time.time() - _time))


        if drop_1_for_FE:
            design_matrices, terms, drop_1_from_term_array = SparseDataGetter.drop_1_for_fixed_effects(
                design_matrices, terms, drop1_from_term_dict, has_non_const_or_keep_cols, endog_reg_cols,
                debug=debug, _time=_time)

        var_2_col_indices = dict()
        offset = 0
        for i, t in enumerate(terms):
            num_cols = design_matrices[i].values.shape[1]
            var_2_col_indices[t.var_name] = np.arange(offset, num_cols + offset)
            offset += num_cols

        if endog_reg_cols is not None:
            endog_reg_cols = endog_reg_cols[has_non_const_or_keep_cols]
            endog_reg_cols = np.hstack(endog_reg_cols)

        cols = list(itertools.chain.from_iterable((d.column_names for d in design_matrices)))
        design_matrix = sphstack(
            [d.values.tocsc() for d in design_matrices]).tocsc().astype(float)

        del term_dict

        if return_dense:
            design_matrix = design_matrix.toarray()
            if design_matrix.shape[1] == 1:
                design_matrix = design_matrix.flatten()

        return SparseFormulaDataObj(design_matrix, cols, nan_rows, name=name,
                                    term_names=[t.var_name for t in terms],# list(var_names),
                                    sparse_terms=terms,
                                    drop_1_dict=drop1_from_term_dict,
                                    endog_reg_cols=endog_reg_cols,
                                    var_2_col_indices=var_2_col_indices)

    @staticmethod
    def parse_exog_names_list_to_formula(exog_names):
        """Convert parsed exogenous term names back to an RHS formula string."""
        if len(exog_names) == 0:
            return '1'
        else:
            #exog_names = [x.replace(' ', '') for x in exog_names]
            if '-1' in exog_names:
                #exog_names = list(filter('-1'.__ne__, [x.replace(' ', '') for x in exog_names]))
                return ' + '.join(exog_names) + " -1"
            else:
                return ' + '.join(exog_names)

    @staticmethod
    def parse_to_variable_lists_helper(rhs, lhs='_temp_'):
        """
        :param lhs: left hand side of formula
        :param rhs: right hand side of formula

        Returns:
            tuple[list[str], list[str]]: Parsed lhs and rhs term-name lists.
        """

        if isinstance(rhs, str):
            rhs_formula = rhs
        else:
            rhs_formula = SparseDataGetter.parse_exog_names_list_to_formula(rhs)
        formula = lhs + " ~ " + rhs_formula
        result = parse_formula(formula)
        lhs, rhs = result[ENDOG_KEY], result[EXOG_KEY]
        rhs = list(filter('1'.__ne__, rhs)) # TODO [x.replace(' ', '') for x in rhs]))
        return lhs, rhs

    @staticmethod
    def drop_1_for_fixed_effects(design_matrices, terms, drop1, has_non_const_or_keep_cols,
                                 endog_reg_cols, debug=False, _time=None):
        """Drop one FE column per flagged term to avoid multicollinearity.

        Args:
            design_matrices (Sequence[SparseFormulaDataObj]): Term matrices.
            terms (Sequence[SparseTerm]): Parsed sparse_terms.
            drop1 (dict): Term-name to drop-one flag.
            has_non_const_or_keep_cols (ndarray[bool]): Per-term keep mask.
            endog_reg_cols (ndarray|None): IV endog-regressor flags.
            debug (bool): Print drop operations.
            _time (float, optional): Timing anchor.

        Returns:
            tuple: ``(design_matrices, sparse_terms, drop1_flags_array)`` after drops.
        """

        if _time is None:
            _time = time.time()

        if debug:
            print("Leave one out for FEs...")

        drop_1_arr = np.array([drop1[t.var_name] for t in terms])
        for i, term in enumerate(terms):

            if drop1[term.var_name]:

                mat, cols_mat, null_rows = (
                    design_matrices[i].values, np.array(design_matrices[i].column_names),
                    design_matrices[i].null_rows)

                if mat.shape[1] > 1:
                    col_drop = cols_mat[0]

                    if debug:
                        print("\tDropping column %s for multicollinearity" % col_drop)
                    mat, cols_mat = mat[:, 1:], cols_mat[1:]

                    design_matrices[i] = SparseFormulaDataObj(mat, cols_mat, null_rows)

                    if endog_reg_cols is not None:
                        endog_reg_cols[i] = endog_reg_cols[i][1:]
                else:
                    has_non_const_or_keep_cols[i] = False

        drop1 = drop_1_arr[has_non_const_or_keep_cols]
        terms = np.array(terms)[has_non_const_or_keep_cols]
        design_matrices = np.array(design_matrices)[has_non_const_or_keep_cols]

        if debug:
            print("Leave one out for fixed effects complete... %.3f s" % (time.time() - _time))
            print()

        return design_matrices, terms, drop1

    @staticmethod
    def reduce_columns_for_multinomial(results):
        """Expand interaction sparse_terms by multiplying component sparse blocks.

        Args:
            results (Sequence[SparseFormulaDataObj]): Monomial component blocks.

        Returns:
            SparseFormulaDataObj: Interaction block with cartesian-product cols.
        """
        if len(results) < 2:
            raise Exception("Should not hit this function with list of 1 result")

        elif len(results)  == 2:
            columns = ['%s:%s' % (c1, c2)
                       for c1 in results[0].column_names for c2 in results[1].column_names]
            blocks = [results[0].values[:, k].multiply(results[1].values).tocsc()
                      for k in range(results[0].values.shape[1])]

            nulls = results[0].null_rows | results[1].null_rows

            if len(blocks) == 1:
                return SparseFormulaDataObj(blocks [0], columns, nulls)
            else:
                return SparseFormulaDataObj(sphstack(blocks).tocsc(), columns, nulls)
        else:
            return SparseDataGetter.reduce_columns_for_multinomial(
                (results[0], SparseDataGetter.reduce_columns_for_multinomial(results[1:])))

    @staticmethod
    def get_parse_formula_dict_to_var_name_list(parsed_formula):
        """Flatten parsed formula dict into a single list of referenced sparse_terms."""
        term_strings = [parsed_formula[ENDOG_KEY]] + parsed_formula[EXOG_KEY]
        if parsed_formula[WEIGHTS_KEY] is not None:
            term_strings += [parsed_formula[WEIGHTS_KEY]]
        if parsed_formula[INSTRUMENTS_KEY] is not None:
            term_strings += parsed_formula[INSTRUMENTS_KEY]
        return term_strings

    @staticmethod
    def get_null_indices_for_formula(formula, data, absorb=None, debug=False):
        """Proxy to module-level null-row detection helper."""
        return get_null_indices_for_formula(formula, data, absorb=absorb, debug=debug)

    @staticmethod
    def get_data(data: pd.DataFrame | dict | SparseDataFrame, 
                formula: str,
                 absorb=None, numerical_invalid_row_func=None,
                 categorical_invalid_row_func=None, weights_invalid_row_func=None, absorb_invalid_row_func=None,
                 check_constant_cols=False, debug=False, _time=None, fail_on_missing=False,
                 cache_intermediate=True, sum_to_n=False, fail_on_iv=False, fail_on_absorb=False,
                 fail_on_weights=False, index=None, test_formula_on_dummy=True, groups=None,
                 drop_1_for_FE=True, exog_only=False) -> SparseFormulaDataObj:
        """Public entry point: parse formula and return all aligned data blocks.

        Args:
            data (DataFrame|dict): Input data.
            formula (str): Full formula string.
            absorb (str|list[str]|None): Optional absorb/fixed effects.
            numerical_invalid_row_func (callable, optional): Numeric null rule.
            categorical_invalid_row_func (callable, optional): Categorical rule.
            weights_invalid_row_func (callable, optional): Weights null rule.
            absorb_invalid_row_func (callable, optional): Absorb null rule.
            check_constant_cols (bool): Drop redundant constant columns.
            debug (bool): Print diagnostics.
            _time (float, optional): Timing anchor.
            fail_on_missing (bool): Raise if null rows are found.
            cache_intermediate (bool|dict): Cache controls.
            sum_to_n (bool): Normalize weights to sum to n.
            fail_on_iv (bool): Reject formulas with instruments.
            fail_on_absorb (bool): Reject absorb argument usage.
            fail_on_weights (bool): Reject weights usage.
            index (array-like, optional): Optional row subset.
            test_formula_on_dummy (bool): Validate formula on dummy data first.
            groups (Any): Reserved for covariance grouping (unused here).
            drop_1_for_FE (bool): Apply drop-one coding for FE blocks.

        Returns:
            dict: Structured bundle keyed by ENDOG/EXOG/INSTRUMENTS/ABSORB/
            WEIGHTS plus metadata keys.
        """

        if _time is None:
            _time = time.time()

        data = dict_2_dataframe(data)

        # Try to expand out lag short-hand
        formula = expand_lag_terms_in_formula(formula)

        # Check to make sure the formula can also be parsed on a dummy example before full-blown
        # data construction
        if test_formula_on_dummy:
            if debug:
                print("\tMaking sure formula works with supplied data...")
            SparseDataGetter.test_formula_on_dummy_data(data, formula, absorb=absorb ) #, cov_groups=cov_groups)

        return SparseDataGetter._get_data_internal(
            data, formula, absorb=absorb, numerical_invalid_row_func=numerical_invalid_row_func,
            categorical_invalid_row_func=categorical_invalid_row_func,
            weights_invalid_row_func=weights_invalid_row_func, absorb_invalid_row_func=absorb_invalid_row_func,
            check_constant_cols=check_constant_cols, debug=debug, _time=_time,
            fail_on_missing=fail_on_missing, cache_intermediate=cache_intermediate, sum_to_n=sum_to_n,
            fail_on_iv=fail_on_iv, fail_on_absorb=fail_on_absorb, fail_on_weights=fail_on_weights,
            index=index, #cov_groups=cov_groups,
            drop_1_for_FE=drop_1_for_FE, exog_only=exog_only
        )

    @staticmethod
    def _get_data_internal(
            data, formula, absorb=None, numerical_invalid_row_func=None, categorical_invalid_row_func=None,
            weights_invalid_row_func=None, absorb_invalid_row_func=None, check_constant_cols=False, debug=False, _time=None,
            fail_on_missing=False, cache_intermediate=True, sum_to_n=False, fail_on_iv=False, fail_on_absorb=False,
            fail_on_weights=False, index=None, drop_1_for_FE=True, keep_all_columns=False, exog_only=False):
        """Internal worker that builds and aligns all formula-derived data blocks."""

        if _time is None:
            _time = time.time()

        # unpack the formula using patsy
        result = parse_formula(formula, debug=debug)
        exog_names = result[EXOG_KEY]
        instrument_names = result[INSTRUMENTS_KEY]
        endog_name = result[ENDOG_KEY]
        weights = result[WEIGHTS_KEY]

        # Make sure everything specified is present
        # test_tokens(, data, debug=debug) # TODO!!!!! tokens

        if fail_on_absorb and absorb is not None:
            raise Exception("Cannot supply `absorb` arg when `fail_on_absorb=True`")

        if fail_on_iv and instrument_names:
            raise Exception("Cannot specify `instruments` when `fail_on_instruments=True`")

        if fail_on_weights and weights:
            raise Exception("Cannot specify `weights` when `fail_on_weights=True`")

        # TODO remove
        # cols_needed_for_formula = SparseFormula.get_column_labels_needed_for_formula(formula)
        # if not set(cols_needed_for_formula) <= set(data.columns):
        #     raise MissingColumnException("Cannot find columns %s in data from formula" %
        #                                  str(set(cols_needed_for_formula) - set(data.columns)))

        if instrument_names is not None:
            if not isinstance(instrument_names, list):
                raise Exception("`instruments` must be a list of strings!")
        else:
            instrument_names = []

        is_iv = len(instrument_names) > 0
        exog_names = list(exog_names)  # so we don't overwrite

        if '-1' in [e.replace(' ', '') for e in exog_names] and absorb:
            raise AbsorbAndNoInterceptException("Cannot specify no intercept in formula and absorb a fixed effed!")

        # instrument_regressors = list(instrument_names)

        if is_iv:
            is_endog_regressor = {x: x not in instrument_names for x in exog_names}
            endog_regressors = sorted(x for x in exog_names if x not in instrument_names)
        else:
            is_endog_regressor = None
            endog_regressors = None

        do_absorb = (absorb is not None and len(absorb) > 0)
        has_intercept = not ('-1' in exog_names or do_absorb)

        if do_absorb:
            absorb_name = absorb
            if isinstance(absorb, str):
                absorb = [absorb]
            absorb_name = ':'.join(absorb)
            if len(absorb) > 1:
                if debug:
                    print("Absorbing more than one fixed effect not supported: %s will be interacted" % str(absorb))
            exog_names.append('-1')
            if is_iv:
                instrument_names.append('-1')
        else:
            absorb_name = None
            absorb = None

        # ----------------------------------------- #
        # Initialize 'groups' (used for covariance) #
        # if cov_groups is not None:
        #     if debug:
        #         print("Getting the groups for inference...", end='')
        #
        #     def get_cov_groups(name, data_df, integer_index=None):
        #         if name in data_df.columns:
        #             vals = data_df[name].copy()
        #             if isinstance(vals, pd.Series):
        #                 vals = vals.values
        #         else:
        #             vals = dmatrix(f'{name} -1', data_df, NA_action=NAAction(NA_types=[]))
        #         if integer_index is not None:
        #             vals = vals[integer_index]
        #         null_rows = set(np.arange(len(vals))[pd.Series(vals).isnull()])
        #         return SparseFormulaDataObj(vals, [name], null_rows=null_rows, name=name)
        #
        #     cov_groups_obj = get_cov_groups(cov_groups, data, integer_index=integer_index)
        #
        #     if debug:
        #         print("%.3f s" % (time.time() - _time))
        #
        # else:
        #     cov_groups_obj = None

        # ------------------------------------ #
        # Initialize Regressors and Regressand #

        # ------------------
        # Build Sparse Terms
        if isinstance(endog_name, str):
            endog_name = list(endog_name)
        endog_name.append('-1')
        endog_terms, _ = SparseTerm.parse_to_terms(endog_name, debug=debug)
        exog_terms, exog_drop1_from_term_dict \
            = SparseTerm.parse_to_terms(exog_names, do_absorb=do_absorb, debug=debug)

        if is_iv:
            instrument_terms, instr_drop1_from_term_dict \
                = SparseTerm.parse_to_terms(instrument_names, do_absorb=do_absorb, debug=debug)
        else:
            instrument_terms, instr_drop1_from_term_dict = None, None

        if weights is not None:
            weights_terms, _ = SparseTerm.parse_to_terms([weights, '-1'], debug=debug)
        else:
            weights_terms = None

        if absorb is not None:
            absorb_terms, _ = SparseTerm.parse_to_terms([absorb_name, '-1'], debug=debug)
        else:
            absorb_terms = None

        return SparseDataGetter._get_data_internal_from_terms(
            data,
            endog_terms,
            exog_terms, exog_drop1_from_term_dict, is_endog_regressor,
            instrument_terms, instr_drop1_from_term_dict,
            weights_terms, absorb_terms,
            do_absorb,
            formula=formula,
            numerical_invalid_row_func=numerical_invalid_row_func,
            categorical_invalid_row_func=categorical_invalid_row_func,
            weights_invalid_row_func=weights_invalid_row_func, absorb_invalid_row_func=absorb_invalid_row_func,
            check_constant_cols=check_constant_cols, debug=debug, _time=_time, fail_on_missing=fail_on_missing,
            cache_intermediate=cache_intermediate, sum_to_n=sum_to_n, fail_on_iv=fail_on_iv,
            fail_on_absorb=fail_on_absorb, fail_on_weights=fail_on_weights, index=index, drop_1_for_FE=drop_1_for_FE,
            keep_all_columns=keep_all_columns,
            exog_only=exog_only,
        )

    @staticmethod
    def _get_data_internal_from_terms(
            data,
            endog_terms,
            exog_terms, exog_drop1_from_term_dict, is_endog_regressor,
            instrument_terms, instr_drop1_from_term_dict,
            weights_terms, absorb_terms,
            do_absorb,
            formula=None,
            numerical_invalid_row_func=None, categorical_invalid_row_func=None,
            weights_invalid_row_func=None, absorb_invalid_row_func=None, check_constant_cols=False, debug=False,
            _time=None,
            fail_on_missing=False, cache_intermediate=True, sum_to_n=False, fail_on_iv=False, fail_on_absorb=False,
            fail_on_weights=False, index=None, drop_1_for_FE=True, keep_all_columns=False,
            exog_only=False
    ):
        """Internal worker that builds and aligns all formula-derived data blocks."""

        if _time is None:
            _time = time.time()

        if numerical_invalid_row_func is None:
            numerical_invalid_row_func = default_numerical_invalid_row_func
        if categorical_invalid_row_func is None:
            categorical_invalid_row_func = default_categorical_invalid_row_func
        if absorb_invalid_row_func is None:
            absorb_invalid_row_func = default_categorical_invalid_row_func
        if weights_invalid_row_func is None:
            weights_invalid_row_func = default_weights_invalid_row_func

        if isinstance(cache_intermediate, dict):
            term_dict = cache_intermediate
        else:
            term_dict = dict() if cache_intermediate else False

        has_intercept = ('Intercept' in [t.var_name for t in exog_terms] or do_absorb)
        has_implicit_constant = (not has_intercept) and (do_absorb or np.any(
            [t.term_type == SparseTerm.FULL_CATEGORICAL for t in exog_terms]))

        # Exog (X)
        if debug:
            print("Exog names are %s" % str([t.var_name for t in exog_terms]))

        exog_obj: SparseFormulaDataObj = SparseDataGetter._sparse_dmatrix_internal_from_terms(
            exog_terms, data, do_absorb=do_absorb, debug=debug,
            check_constant_cols=check_constant_cols,
            is_endog_regressor=is_endog_regressor,
            numerical_invalid_row_func=numerical_invalid_row_func,
            categorical_invalid_row_func=categorical_invalid_row_func,
            cache_intermediate=term_dict, _time=_time, name=EXOG_KEY,
            index=index, drop_1_for_FE=drop_1_for_FE, drop1_from_term_dict=exog_drop1_from_term_dict,
            keep_all_columns=keep_all_columns,
        )

        # return just exog
        if exog_only:
            time_elapsed = time.time() - _time
            return {
                EXOG_KEY: exog_obj,
                HAS_IMPLICIT_CONSTANT_KEY: has_implicit_constant,
                HAS_INTERCEPT_KEY: has_intercept, TIME_ELAPSED_KEY: time_elapsed,
            }


        # Endog (y)
        endog_obj: SparseFormulaDataObj = SparseDataGetter._sparse_dmatrix_internal_from_terms(
                endog_terms, data, do_absorb=do_absorb, debug=debug, check_constant_cols=False,
                numerical_invalid_row_func=numerical_invalid_row_func, cache_intermediate=term_dict, _time=_time,
                index=index, drop_1_for_FE=False, keep_all_columns=True)

        if fail_on_missing and len(endog_obj.null_rows) > 0:
            raise MissingDataException("Endog has missing data in rows %s!" % str(endog_obj.null_rows))

        if fail_on_missing and len(exog_obj.null_rows) > 0:
            raise MissingDataException("Exog has missing data in rows %s!" % str(exog_obj.null_rows))

        # ---------------------- #
        # Initialize Instruments #
        is_iv = instrument_terms is not None
        if is_iv:
            instr_obj: SparseFormulaDataObj = SparseDataGetter._sparse_dmatrix_internal_from_terms(
                instrument_terms, data, do_absorb=do_absorb,
                name=INSTRUMENTS_KEY,
                debug=debug, check_constant_cols=check_constant_cols,
                numerical_invalid_row_func=numerical_invalid_row_func,
                categorical_invalid_row_func=categorical_invalid_row_func,
                cache_intermediate=term_dict, _time=_time,
                index=index,
                drop_1_for_FE=drop_1_for_FE, drop1_from_term_dict=instr_drop1_from_term_dict,
                keep_all_columns=keep_all_columns
            )
            instr_var_2_col = instr_obj.var_2_col_indices
            instr_drop_1_dict = instr_obj.drop_1_dict
            if fail_on_missing and len(instr_obj.null_rows) > 0:
                raise MissingDataException("Instruments have missing data in rows %s!" % str(instr_obj.null_rows))
        else:
            instr_obj = None
            instr_var_2_col = None
            instr_drop_1_dict = None

        # ------------------------------------------- #
        # Interact absorbed FEs and get design matrix #
        # TODO FIX ABSORB
        if absorb_terms is not None:
            absorb_obj = get_categorical_control_data(
                absorb_terms[0].var_name, data, debug=debug, term_dict=term_dict,
                invalid_row_func=absorb_invalid_row_func, name=ABSORB_KEY, index=index,
            )
            absorb_name = absorb_terms[0].var_name
        else:
            absorb_terms = None
            absorb_obj = None
            absorb_name = None

        # --------------- #
        # Get the weights #
        if weights_terms is not None:
            if debug:
                print("Getting the weights...", end='')

            weights_obj: SparseFormulaDataObj = get_numerical_control_data(
                weights_terms[0], data, term_dict=term_dict, index=index,
                invalid_row_func=weights_invalid_row_func, return_type='dense', name=WEIGHTS_KEY)
            if sum_to_n:
                weights_obj.values.data *= weights_obj.values.shape[0] / weights_obj.values.sum()
            if debug:
                print("%.3f s" % (time.time() - _time))
        else:
            weights_obj = None

        # -------------- #
        # Get valid rows #
        valid_obs_rows, null_rows_info_dict = SparseFormulaDataObj.slice_null_rows_from_sfdo_list(
            [endog_obj, exog_obj, absorb_obj, instr_obj, weights_obj], #, cov_groups_obj],
            debug=debug, _time=_time)

        time_elapsed = time.time() - _time

        if is_iv:
            endog_regressors = [x for x in exog_obj.column_names if x not in instr_obj.column_names]
        else:
            endog_regressors = None

        # Keep the settings for later
        settings = dict(
            numerical_invalid_row_func=numerical_invalid_row_func,
            categorical_invalid_row_func=categorical_invalid_row_func,
            weights_invalid_row_func=weights_invalid_row_func, absorb_invalid_row_func=absorb_invalid_row_func,
            check_constant_cols=check_constant_cols, fail_on_missing=fail_on_missing,
            cache_intermediate=cache_intermediate, sum_to_n=sum_to_n, fail_on_iv=fail_on_iv,
            fail_on_absorb=fail_on_absorb, fail_on_weights=fail_on_weights, drop_1_for_FE=drop_1_for_FE,
            keep_all_columns=keep_all_columns, exog_only=exog_only
        )

        if formula is None:
            formula = ' + '.join([t.var_name for t in endog_terms]) + ' ~ '
            formula += ' + '.join([t.var_name for t in exog_terms])
            if is_iv:
                formula += ' + '.join([t.var_name for t in instrument_terms])
            if weights_terms is not None:
                formula += ' $ ' + weights_terms[0].var_name

        fdi = FormulaDesignInfo(formula, data, endog_terms,
                                exog_terms, exog_obj.var_2_col_indices,
                                is_endog_regressor,
                                exog_drop1_from_term_dict,
                                settings,
                                do_absorb,
                                weight_terms=weights_terms,
                                instruments_terms=instrument_terms, instruments_var_2_col_indices=instr_var_2_col,
                                instruments_drop_1_dict=instr_drop_1_dict,
                                absorb_terms=absorb_terms)

        return_dict = {
            ENDOG_KEY: endog_obj, EXOG_KEY: exog_obj, INSTRUMENTS_KEY: instr_obj,
            ABSORB_KEY: absorb_obj, WEIGHTS_KEY: weights_obj,

            ABSORB_NAME_KEY: absorb_name, ENDOG_REGRESSORS_KEY: endog_regressors,
            # INSTRUMENT_REGRESSORS_KEY: instrument_regressors,
            HAS_IMPLICIT_CONSTANT_KEY: has_implicit_constant,
            HAS_INTERCEPT_KEY: has_intercept, TIME_ELAPSED_KEY: time_elapsed,

            VALID_OBS_ROWS_KEY: valid_obs_rows,
            NULL_ROWS_INFO_DICT_KEY: null_rows_info_dict,
            INDEX_KEY: index,

            FORMULA_DESIGN_INFO_KEY: fdi,

            #COV_GROUPS_KEY: cov_groups_obj,
            #COV_GROUPS_NAME_KEY: cov_groups,
        }

        return return_dict

    @staticmethod
    def test_formula_on_dummy_data(data, formula, absorb=None):  # , cov_groups=None):
        """Smoke-test formula parsing/building using synthetic non-missing data."""
        dummy_data = pd.DataFrame(columns=data.columns, data=np.ones((3, len(data.columns))).astype(float))
        SparseDataGetter._get_data_internal(dummy_data, formula, absorb=absorb,
                                            check_constant_cols=False, drop_1_for_FE=False,
                                            )  # , cov_groups=cov_groups)

    @staticmethod
    def get_nonconstant_cols_from_csc_matrix(matrix, found_constant, valid_row_boolean_idx=None):
        """Identify non-constant columns in a CSC matrix.

        Keeps at most one non-zero constant column (typically intercept) and
        marks remaining constant columns for dropping.

        Args:
            matrix (scipy.sparse.csc_matrix): Matrix to inspect.
            found_constant (bool): Whether a retained constant already exists.
            valid_row_boolean_idx (ndarray[bool], optional): Optional mask used
                to ignore invalid/null rows during constancy checks.

        Returns:
            tuple:
                - ndarray[bool]: Columns-to-keep mask.
                - bool: Updated ``found_constant`` flag.
        """

        constant_cols = []
        zero_cols = []
        for j in range(matrix.shape[1]):

            x_col_data = matrix.data[matrix.indptr[j]:matrix.indptr[j + 1]]
            v = np.zeros(matrix.shape[0])
            v[matrix.indices[matrix.indptr[j]:matrix.indptr[j + 1]]] = x_col_data
            if valid_row_boolean_idx is not None:
                v = v[valid_row_boolean_idx]

            zero_cols.append(np.abs(v).sum() < 1e-6)

            if zero_cols[-1]:
                constant_cols.append(True)
            else:
                constant_cols.append(np.std(v) < 1e-6)

        constant_cols = np.array(constant_cols)
        zero_cols = np.array(zero_cols)

        if not found_constant:
            nz = np.nonzero(constant_cols & ~zero_cols)[0]
            if len(nz):
                constant_cols[nz[0]] = False
                found_constant = True

        cols_to_keep = ~constant_cols

        return cols_to_keep, found_constant

    @staticmethod
    def sparse_dmatrix(formula, data, do_absorb=False, debug=False, check_constant_cols=True, is_endog_regressor=None,
                       cache_intermediate=True, _time=None, return_dense=False,
                       numerical_invalid_row_func=None, categorical_invalid_row_func=None,
                       drop_1_for_FE=True, name=None, index=None):
        """Build RHS sparse design matrix from a formula string.

        Args mirror ``_sparse_dmatrix_internal`` except ``formula`` is parsed
        to exogenous term names first.

        Returns:
            SparseFormulaDataObj: Exogenous design-matrix object.

        Examples
        --------
        Build a sparse design matrix from a Patsy-style formula. The result
        has ``.values`` (a CSC sparse matrix) and ``.column_names``:

        >>> import numpy as np, pandas as pd
        >>> from kanly.api import sparse_dmatrix
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({'x': rng.normal(size=10),
        ...                    'g': rng.integers(0, 3, 10)})
        >>> X = sparse_dmatrix('~ x + C(g)', df)             # doctest: +SKIP
        >>> X.values.shape                                    # doctest: +SKIP
        (10, 4)
        >>> X.column_names                                    # doctest: +SKIP
        ['Intercept', 'x', 'C(g)[1]', 'C(g)[2]']
        """

        if _time is None:
            _time = time.time()

        parsed_formula = parse_formula(formula)
        var_names = parsed_formula[EXOG_KEY]

        data_obj = SparseDataGetter._sparse_dmatrix_internal(
            var_names, data, do_absorb=do_absorb, debug=debug, check_constant_cols=check_constant_cols,
            is_endog_regressor=is_endog_regressor, numerical_invalid_row_func=numerical_invalid_row_func,
            categorical_invalid_row_func=categorical_invalid_row_func, _time=_time, name=name,
            index=index, drop_1_for_FE=drop_1_for_FE, cache_intermediate=cache_intermediate,
            return_dense=return_dense)
        data_obj.name = name

        return data_obj

    @staticmethod
    def get_endog_matrix(endog_names, data, do_absorb=False, debug=False, check_constant_cols=True,
                         is_endog_regressor=None, term_dict=None, _time=None, return_dense=False,
                         numerical_invalid_row_func=None, categorical_invalid_row_func=None,
                         drop_1_for_FE=True, index=None):

        """Build response matrix block for one or multiple endog sparse_terms."""
        if len(endog_names) == 1:
            return get_numerical_control_data(
                endog_names[0], data, invalid_row_func=numerical_invalid_row_func, name=ENDOG_KEY,
                term_dict=term_dict, index=index)
        else:
            return SparseDataGetter._sparse_dmatrix_internal(
                endog_names + ['-1'], data, do_absorb=do_absorb, debug=debug,
                check_constant_cols=check_constant_cols,
                is_endog_regressor=is_endog_regressor, numerical_invalid_row_func=numerical_invalid_row_func,
                categorical_invalid_row_func=categorical_invalid_row_func, _time=_time, name=ENDOG_KEY,
                index=index, drop_1_for_FE=drop_1_for_FE, cache_intermediate=term_dict,
                return_dense=return_dense)

    @staticmethod
    def sparse_dmatrices(formula, data, do_absorb=False, debug=False, check_constant_cols=True,
                         is_endog_regressor=None, cache_intermediate=True, _time=None, return_dense=False,
                         numerical_invalid_row_func=None, categorical_invalid_row_func=None,
                         drop_1_for_FE=True, index=None):
        """Build both endog and exog sparse matrices from a full formula.

        Returns:
            tuple[SparseFormulaDataObj, SparseFormulaDataObj]: ``(endog, exog)``.

        Examples
        --------
        Split a formula into response and design-matrix objects in one
        pass (analogous to ``patsy.dmatrices`` but sparse and faster on
        wide categorical designs):

        >>> import numpy as np, pandas as pd
        >>> from kanly.api import sparse_dmatrices
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({'y': rng.normal(size=10),
        ...                    'x': rng.normal(size=10),
        ...                    'g': rng.integers(0, 3, 10)})
        >>> y_obj, X_obj = sparse_dmatrices('y ~ x + C(g)', df)  # doctest: +SKIP
        >>> y_obj.values.shape, X_obj.values.shape               # doctest: +SKIP
        ((10, 1), (10, 4))
        """

        if _time is None:
            _time = time.time()

        parsed_formula = parse_formula(formula)

        term_dict = SparseDataGetter._get_term_dict(cache_intermediate)

        # endog_name = parsed_formula[ENDOG_KEY]
        # if isinstance(endog_name, list) and len(endog_name) == 1:
        #     endog_name = endog_name[0]
        endog_obj = SparseDataGetter.get_endog_matrix(
            parsed_formula[ENDOG_KEY],
            # endog_name,
            data, check_constant_cols=check_constant_cols,
            is_endog_regressor=is_endog_regressor, numerical_invalid_row_func=numerical_invalid_row_func,
            categorical_invalid_row_func=categorical_invalid_row_func, _time=_time,
            index=index, drop_1_for_FE=drop_1_for_FE, term_dict=term_dict,
            return_dense=return_dense)

        exog_obj = SparseDataGetter._sparse_dmatrix_internal(
            parsed_formula[EXOG_KEY], data, do_absorb=do_absorb, debug=debug, check_constant_cols=check_constant_cols,
            is_endog_regressor=is_endog_regressor, numerical_invalid_row_func=numerical_invalid_row_func,
            categorical_invalid_row_func=categorical_invalid_row_func, _time=_time, name=EXOG_KEY,
            index=index, drop_1_for_FE=drop_1_for_FE, cache_intermediate=term_dict,
            return_dense=return_dense)

        return endog_obj, exog_obj


def expand_lag_terms_in_formula(formula):
    """
    Expands sparse_terms of the form
        `L(x,range(1,3)` to `L(x,1) + L(x,2)`
    or
        `L(x,[3,8,9])` to `L(x,3) + L(x,8) + L(x,9)`

    Args:
        formula (str): Input formula string that may contain iterable lag specs.

    Returns:
        str: Formula with iterable lag specs expanded into explicit lag sparse_terms.
    """

    lag_vars_found = []
    for j in range(len(formula) - 2):
        if formula[j:j + 2] == 'L(':
            paren_counter = 1
            i = j + 2
            while i < len(formula):
                if formula[i] == ')':
                    paren_counter -= 1
                elif formula[i] == '(':
                    paren_counter += 1

                if paren_counter == 0:
                    lag_vars_found.append(formula[j + 2:i])
                    break
                i += 1

    for z in lag_vars_found:
        zsplit = parse_str_2_tuple(z)
        if len(zsplit) > 1:
            a = eval(f'{zsplit[1]}')
            if isinstance(a, int):
                a = [a]
            elif hasattr(a, '__iter__'):
                a = list(a)
            else:
                raise Exception('Second argument in `L(<var>, ., ...)` must be integer for lag'
                                ', or iterable of integers')
            assert np.all([isinstance(j, int) for j in a])

            if len(zsplit) == 3:
                suff = f',{zsplit[2]}'
            else:
                suff = ''

            formula = formula.replace(
                f'L({z})',
                ' + '.join([f'L({zsplit[0]},{j}{suff})' for j in a])
            )
    return formula


if __name__ == '__main__':

    import pandas as pd

    import pandas as pd
    import numpy as np
    from kanly.api import lm, LM

    n = 1000
    np.random.seed(0)

    x = 10 + np.random.randn(n)
    x2 = -3 + x + np.random.randn(n)
    z = [f'v{s}' for s in np.random.randint(0,2,n).astype(str)]
    y = 1.5 + .5 * x + .2 * np.random.randn(n)
    v = np.random.rand(n)
    g = np.random.randint(0, 4, n)
    w = np.random.randn(n)
    a = np.random.randint(0,10,n)

    df = pd.DataFrame(dict(x=x, g=g, z=z, y=y, w=w, x2=x2,a=a,v=v))

    formula = 'I(y+0) ~ C(g):C(z) + x | C(g):C(z) + x2 $ I(np.exp(w))'
    #formula = 'I(y+0) ~ C(g)*z + x | C(g)*z + x2'
    #formula = 'I(y+0) ~ C(g)*z + x'

    print(fit:=lm(formula, df))#, absorb='a'))

    # ret_val = SparseDataGetter.get_data(df, formula) #, absorb='a')

    # print(LM(ret_val[ENDOG_KEY].values, ret_val[EXOG_KEY].values, instruments=ret_val[INSTRUMENTS_KEY].values,
    #          weights=ret_val[WEIGHTS_KEY].values,
    #          absorb=ret_val[ABSORB_KEY].values))
