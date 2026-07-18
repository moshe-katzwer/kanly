"""Container object for sparse formula matrices and row-validity metadata."""
from __future__ import absolute_import, print_function

import time

import numpy as np
from scipy.sparse import isspmatrix


class SparseFormulaDataObj(object):
    """Bundle matrix values, column names, and null-row metadata for one block."""

    def __init__(self, values, column_names, null_rows, drop_1_dict=None,
                 term_names=None, sparse_terms=None, endog_reg_cols=None, var_2_col_indices=None,
                 name=None):
        """Initialize a formula data bundle.

        Args:
            values (ndarray or sparse matrix): Data block values.
            column_names (list[str] or None): Column names for ``values``.
            null_rows (set[int] or None): Invalid/null row indices.
            drop_1_dict (dict, optional): Term-level drop-one metadata.
            term_names (list[str], optional): Source term names.
            sparse_terms(list[SparseTerm], optional): the actual SparseTerm objects
            endog_reg_cols (ndarray, optional): Flags for endogenous regressors.
            var_2_col_indices (dict, optional): Maps each formula term name to
                the 0-based column indices of that term in ``values`` (e.g.
                ``{'C(g)': [3, 4, 5, 6]}``).  Copied onto linear models as
                ``exog_term_to_indices`` / ``instrument_term_to_indices``.
            name (str, optional): Logical block name (ENDOG/EXOG/etc.).
        """
        self.values = values
        self.column_names = column_names
        self.null_rows = null_rows
        self.sparse_terms = sparse_terms
        self.term_names = term_names
        self.endog_reg_cols = endog_reg_cols
        self.drop_1_dict = drop_1_dict
        self.var_2_col_indices = var_2_col_indices
        self.name = name

    def __repr__(self):
        """Return string form for debugging."""
        return str(self)

    def __str__(self):
        """Serialize internal attributes as a dict-like string."""
        return str(self.__dict__)

    def slice_null_rows(self, valid_obs_rows):
        """Restrict values to valid rows and drop empty sparse columns.

        Args:
            valid_obs_rows (array-like[int]): Sorted indices of rows to keep.
        """
        self.values = self.values[valid_obs_rows]
        if isspmatrix(self.values):
            if len(self.values.shape) == 2:
                nnz = np.array([self.values.getcol(j).nnz for j in range(self.values.shape[1])]) > 0
                if sum(nnz) < self.values.shape[1]:
                    idx = np.arange(self.values.shape[1])[nnz]
                    self.values = self.values[:, idx]
                    self.column_names = list(np.array(self.column_names)[idx])
                    self.null_rows = set()

    @staticmethod
    def construct_empty():
        """Construct an empty placeholder object."""
        return SparseFormulaDataObj(None, None, None)

    @staticmethod
    def slice_null_rows_from_sfdo_list(sfdo_list, debug=False, _time=None):
        """Union null rows across blocks, row-slice blocks, and return valid rows.

        Args:
            sfdo_list (list[SparseFormulaDataObj or None]): Data blocks to align.
            debug (bool): Print null-row diagnostics.
            _time (float, optional): Start time for debug timing output.

        Returns:
            tuple:
                - ndarray: Sorted valid observation row indices.
                - dict: Per-block null-row info keyed by object name.
        """

        if _time is None:
            _time = time.time()

        if debug:
            print("\nNull/Invalid Rows (by integer index): ")

        null_rows = set()
        null_rows_info_dict = dict()
        valid_obs_rows = None
        for i, o in enumerate(sfdo_list):
            if o is not None:
                if valid_obs_rows is None:
                    valid_obs_rows = set(range(o.values.shape[0]))
                null_rows_info_dict[o.name] = o.null_rows.copy()
                if o.null_rows:
                    if debug:
                        null_list = list(o.null_rows)[:1000]
                        print(f"\t{o.name if o.name is not None else f'({i})'}'"
                              f" has null rows {null_list} (of {len(o.null_rows)}!")
                    null_rows |= o.null_rows
        if debug and not null_rows:
            print("\tNone! :)")

        if valid_obs_rows is None:
            raise Exception
        valid_obs_rows = np.array(sorted(valid_obs_rows - null_rows))

        # ----------------------------------------------------------------
        # row-slice all the data objects to handle the missing/invalid/null
        if null_rows:
            if debug:
                print('Reshaping data to handle null_rows...', end='')

            for obj in sfdo_list:
                if obj is not None:
                    obj.slice_null_rows(valid_obs_rows)
                    
            if debug:
                print(f"{'%.3f' % (time.time() - _time)}s")

        return valid_obs_rows, null_rows_info_dict