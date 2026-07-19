from __future__ import absolute_import, print_function

from kanly.formula.sparse_term import SparseTerm
from kanly.formula.sparse_term_to_data_methods import get_categorical_interacted_indices


class ClusterGroupObject(object):

    def __init__(self, group_name_2_vals):
        self.group_name_2_vals = group_name_2_vals
        self.num_groups = len(self.group_name_2_vals)

    @staticmethod
    def get_cluster_group_object_from_data(
            groups, data, index=None, remove_null_columns=True, debug=False):
        if isinstance(groups, str):
            groups = [groups, ]
        if len(groups) not in (1, 2):
            raise Exception("Must have 1 or 2 cluster groups categories!")

        # add interaction for 2-way FE
        elif len(groups) == 2:
            groups.append(list(groups))

        groups = [[g] if isinstance(g, str) else g for g in groups]
        print("ZZZ ", groups)

        group_name_2_vals = dict()
        for g in groups:
            term_string = ':'.join([f'C({x})' for x in g])
            nobs, col_indices, nulls, param_col_names, num_cat = get_categorical_interacted_indices(
                SparseTerm.parse_to_term(term_string), data, debug=debug, term_dict=None, invalid_row_func=None,
                remove_null_columns=remove_null_columns, index=index)
            group_name_2_vals[':'.join(g)] = {'col_indices': col_indices, 'nulls': nulls, 'num_categories': num_cat}

        return group_name_2_vals


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from kanly.formula.sparse_term import SparseTerm
    from kanly.api import lm

    n = 10_000_000
    df = pd.DataFrame({
        'g1': np.random.randint(0, 5, n).astype(int),
        'g2': np.random.randint(0, 3, n),
        'y': np.random.randn(n)
    })
    # df['g1'] = [f'_{g}' for g in df.g1]

    print(lm('y~C(g1>0)', df))
    print('=' * 100)

    X = ClusterGroupObject.get_cluster_group_object_from_data([('g1>0', 'g2',)], df)
    print(X)

    X = ClusterGroupObject.get_cluster_group_object_from_data(['g1>0', 'g2', ], df)
    print(X)
