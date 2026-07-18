"""Shared key constants used by formula parsing and data construction.

These strings are used as dictionary keys when passing intermediate formula
artifacts (endog/exog/weights/instruments/absorb metadata) between
``parse_formula``, ``SparseDataGetter``, and downstream model code.
"""
from __future__ import absolute_import, print_function

# Name used in result tables when no weights column is supplied.
NULL_WEIGHTS_NAME = '-'
# Generic key for token lists parsed from a formula block.
TOKENS_KEY = 'TOKENS'
# Formula left-hand side (response).
ENDOG_KEY = 'ENDOG'
# Formula right-hand side (design sparse_terms).
EXOG_KEY = 'EXOG'

# Optional observation weights.
WEIGHTS_KEY = 'WEIGHTS'
# Optional instrumental-variable sparse_terms (right side of "|").
INSTRUMENTS_KEY = 'INSTRUMENTS'

# Optional absorb/fixed-effect metadata.
ABSORB_KEY = 'ABSORB'
ABSORB_NAME_KEY = 'ABSORB_NAME'

# Misc metadata returned by formula/data-getter pipeline.
FORMULA_KEY = 'FORMULA'
NULL_ROWS_INFO_DICT_KEY = 'NULL_ROWS_INFO_DICT'
INDEX_KEY = 'INDEX'
HAS_INTERCEPT_KEY = 'HAS_INTERCEPT'
VALID_OBS_ROWS_KEY = 'VALID_OBS_ROWS'

# Optional control-function IV internals.
ENDOG_REGRESSORS_KEY = 'ENDOG_REGRESSORS'
INSTRUMENT_REGRESSORS_KEY = 'INSTRUMENT_REGRESSORS'
HAS_IMPLICIT_CONSTANT_KEY = 'HAS_IMPLICIT_CONSTANT'
TIME_ELAPSED_KEY = 'TIME_ELAPSED'

FORMULA_DESIGN_INFO_KEY = 'FORMULA_DESIGN_INFO'

# Sentinel column-name token used internally for synthetic intercept handling.
RETURN_CONSTANT_COLUMN_TERM_NAME = '__________CONSTANT__________abdfefghijklmnopqrstuvZYXWVUTSRQ12345_' * 2
