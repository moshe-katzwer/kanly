from __future__ import absolute_import, print_function

import copy
import pprint

from pandas import DataFrame

from kanly.dill_object import DillObject
from kanly.formula.sparse_term import SparseTerm
from kanly.sparse_data_frame import SparseDataFrame


class FormulaDesignInfoBase(DillObject):
    def __init__(self, formula, data):
        self.formula = formula
        self.data = data


class FormulaDesignInfo(FormulaDesignInfoBase):

    def __init__(self, formula: str,
                 data: (DataFrame, SparseDataFrame),
                 endog_terms: list[SparseTerm],

                 exog_terms: list[SparseTerm],
                 exog_var_2_col_indices: dict,
                 exog_is_endog_regressor: dict,
                 exog_drop_1_dict: dict,

                 settings: dict,
                 do_absorb: bool,

                 weight_terms=None,

                 instruments_terms: (list[SparseTerm], None) = None,
                 instruments_var_2_col_indices: (dict, None) = None,
                 instruments_drop_1_dict: (dict, None) = None,

                 absorb_terms=None,
                 ):

        super().__init__(formula, data)

        self.endog_terms = endog_terms

        self.exog_terms = exog_terms
        self.exog_is_endog_regressor = exog_is_endog_regressor
        self.exog_var_2_col_indices = exog_var_2_col_indices
        self.exog_term_names = list(self.exog_var_2_col_indices.keys())
        self.exog_drop_1_dict = exog_drop_1_dict

        self.settings = settings
        self.do_absorb = do_absorb

        self.weights_terms = weight_terms

        self.instruments_terms = instruments_terms
        self.instruments_var_2_col_indices = instruments_var_2_col_indices
        self.instrument_term_names = None if instruments_terms is None else list(self.exog_var_2_col_indices.keys())
        self.instruments_drop_1_dict = instruments_drop_1_dict

        self.absorb_terms = absorb_terms

    def __str__(self):
        tab2 = ' ' * 8
        res = ['Formula Design Info:',
               '    formula:\n' + tab2 + self.formula,
               '    endog:\n' + tab2 + pprint.pformat([t.var_name for t in self.endog_terms]),
               '    exog:\n' + tab2 + pprint.pformat([t.var_name for t in self.exog_terms]),
               ]
        for at in ['instruments', 'weights', 'absorb']:
            x = getattr(self, f'{at}_terms')
            if x is not None:
                res.append(f'    {at}:\n' + tab2 + pprint.pformat([t.var_name for t in x]))
        return '\n'.join(res)

    def __repr__(self):
        return self.__str__()

    def clone(self):
        return copy.deepcopy(self)

    def get_design_data(self, data=None, **kwargs):
        if data is None:
            data = self.data
        from kanly.formula.data_getter import SparseDataGetter  # circular
        settings = copy.deepcopy(self.settings)
        settings.update({k: v for k, v in kwargs.items() if k in settings})
        return SparseDataGetter._get_data_internal_from_terms(
            data,
            self.endog_terms,
            self.exog_terms, self.exog_drop_1_dict, self.exog_is_endog_regressor,
            self.instruments_terms, self.instruments_drop_1_dict,
            self.weights_terms, self.absorb_terms,
            self.do_absorb,
            formula=self.formula,
            **settings,
        )

    def get_design_data_exog(self, data=None, **kwargs):
        if data is None:
            data = self.data
        from kanly.formula.data_getter import SparseDataGetter  # circular
        settings = copy.deepcopy(self.settings)
        settings.update({k: v for k, v in kwargs.items() if k in settings})
        settings['exog_only'] = True
        return SparseDataGetter._get_data_internal_from_terms(
            data,
            self.endog_terms,
            self.exog_terms, self.exog_drop_1_dict, self.exog_is_endog_regressor,
            self.instruments_terms, self.instruments_drop_1_dict,
            self.weights_terms, self.absorb_terms,
            self.do_absorb,
            formula=self.formula,
            **settings,
        )

# if __name__ == '__main__':
#     import numpy as np
#     import pandas as pd
#     from kanly.api import lm, LM
#     from kanly.formula.keys import EXOG_KEY
#
#     n = 100
#     np.random.seed(0)
#     x = np.random.randn(n)
#     x = np.array(sorted(x))
#     df = pd.DataFrame({
#         'x': x,
#         'grp': np.random.randint(0, 12, n),
#     })
#     df['y'] = 1.2 - 0.3 * df['x'] + .2 * np.random.randn(n)
#
#     fit = lm('center(y) ~ center(x)-1', df)
#
#     print(fit.summary())
#     print(fit.model.formula_design_info)
#
#     exog_terms = fit.model.formula_design_info.exog_terms
#     exog_drop1_dict = fit.model.formula_design_info.exog_drop_1_dict
#     settings = fit.model.formula_design_info.settings
#
#     fdi: FormulaDesignInfo = fit.model.formula_design_info
#     retval = fdi.get_design_data(df[:15])
#     print( retval[EXOG_KEY].values.toarray().shape)
#     print(fit.params.shape)
#
#     import matplotlib.pyplot as plt
#
#     plt.scatter(x-x.mean(), df.y-df.y.mean(), alpha=.5, color='grey')
#     plt.scatter(x-x.mean(), fit.fittedvalues, color='b', lw=2)
#     plt.scatter(df.x[:15]-x.mean(),
#                 retval[EXOG_KEY].values.toarray().dot(fit.params),
#                 color='r', lw=2)
#
#     fit_sub = lm('center(y) ~ center(x)-1', df[:15])
#     plt.scatter(df[:15].x-df[:15].x.mean(), fit_sub.fittedvalues, color='g', lw=2)
#
#     plt.show()
