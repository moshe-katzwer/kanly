from __future__ import absolute_import, print_function

import itertools

import numpy as np
from numpy import ndarray
from pandas import DataFrame, Series
from scipy.sparse import isspmatrix, csc_matrix


def num_cols(d):
    if len(d.shape) == 1:
        return 1
    else:
        return d.shape[1]


class SparseDataFrame(object):

    def __init__(self, data, columns=None, index=None, copy=False):

        if not isinstance(data, list):
            data = [{'data': data, 'columns': columns, 'index': index}]
        self.data = data

        for d in data:
            d['data'] = d['data'].copy()
            d['columns'] = d.get(
                'columns',
                d['data'].columns if isinstance(d['data'], DataFrame)
                else (d['data'].name if isinstance(d['data'], Series) else None)
                )

        var_cnt = 0
        self.col_2_ind_data_union = dict()
        self.ind_2_data_num = dict()
        self.col_2_data_num = dict()
        self.col_2_ind_data = dict()
        self.dtypes = dict()
        for n_d, d in enumerate(data):
            nc = num_cols(d['data'])
            if d.get('columns', None) is None:
                d['columns'] = [f'__x{j}' for j in range(var_cnt, var_cnt + nc)]
            self.col_2_ind_data.update({c: i for c, i in zip(d['columns'], range(nc))})
            self.col_2_ind_data_union.update({c: i for c, i in zip(d['columns'], range(var_cnt, var_cnt + nc))})
            self.ind_2_data_num.update({i: n_d for i in range(var_cnt, var_cnt + nc)})
            self.col_2_data_num.update({c: n_d for c in d['columns']})
            self.dtypes.update(self._get_dtypes(d['data'], d['columns']))
            var_cnt += nc

        self.columns = list(itertools.chain.from_iterable([d['columns'] for d in data]))
        if len(self.columns) != len(set(self.columns)):
            raise Exception("Duplicate column names!")

        # print(self.col_2_ind_data_union)
        # print(self.col_2_ind_data)
        # print(self.ind_2_data_num)
        # print(self.col_2_data_num)
        # print(self.dtypes)

        if len(set([d['data'].shape[0] for d in self.data])) != 1:
            raise Exception("Different row lengths!")
        self.shape = (self.data[0]['data'].shape[0], var_cnt)

        self.nnz = sum([
            d['data'].nnz if isspmatrix(d['data']) else np.prod(d['data'].shape)
            for d in data
        ])

        # for func_name in ['sum', 'mean', 'max', 'min', 'argmax', 'argmin']:
        #     setattr(self, func_name, getattr(self.data, func_name))

    @staticmethod
    def _get_dtypes(data, columns):
        if isspmatrix(data) or isinstance(data, ndarray) or isinstance(data, Series):
            return {c: data.dtype for c in columns}
        elif isinstance(data, DataFrame):
            return {c: data.dtypes.iloc[i] for i, c in enumerate(columns)}
        else:
            raise Exception

    @staticmethod
    def _convert_column_type(values, return_type='ndarray'):
        if isspmatrix(values):
            if return_type == 'sparse':
                return csc_matrix(values).reshape((-1, 1))
            elif return_type == 'ndarray':
                return values.toarray().flatten()
            elif return_type == 'Series':
                return Series(values.toarray().flatten())
        elif isinstance(values, ndarray):
            if return_type == 'sparse':
                return csc_matrix(values).reshape((-1, 1))
            elif return_type == 'ndarray':
                return values.flatten()
            elif return_type == 'Series':
                return Series(values.flatten())
        elif isinstance(values, Series):
            if return_type == 'sparse':
                return csc_matrix(values).reshape((-1, 1))
            elif return_type == 'ndarray':
                return values.values
            elif return_type == 'Series':
                return values
        else:
            raise Exception

    @staticmethod
    def _hstack(values, return_type='ndarray'):
        if return_type == 'ndarray':
            return np.hstack([v.reshape((-1,1)) for v in values])
        else:
            raise NotImplementedError(f'return_type={return_type}')

    def __getitem__(self, key, return_type='ndarray'):

        if isinstance(key, str):
            try:
                data = self.data[self.col_2_data_num[key]]['data']
                if isinstance(data, ndarray) or isspmatrix(data):
                    values = data[:, self.col_2_ind_data[key]]
                elif isinstance(data, DataFrame):
                    values = data.iloc[:, self.col_2_ind_data[key]]
                elif isinstance(data, Series):
                    values = data.values
                return self._convert_column_type(values, return_type=return_type)
            except Exception as e:
                raise Exception(f'key "{key}" not in columns!')

        else:
            try:
                key = list(iter(key))
                values = (self.__getitem__(k, return_type=return_type) for k in key)
                return self._hstack(values, return_type=return_type)
            except:
                raise Exception("TODO")  # TODO

    def __str__(self):
        return "<%d x %d data structure with columns %s, %d non-zero elements (%.3f%%)>" % (
            self.shape[0], self.shape[1], str(self.columns), self.nnz, 100.0*self.nnz/np.prod(self.shape))

    def __repr__(self):
        return str(self)

    def __len__(self):
        return self.shape[0]

    def loc(self):
        raise NotImplementedError("Cannot row index a `SparseDataFrame`, only column access!")
    #
    # def _get_bootstrap_sample_generator(self, n_samples, seed, groups=None, debug=False):
    #
    #     rand = np.random.RandomState(seed)
    #     data_csr = self.data.tocsr(copy=False)
    #
    #     if groups is None:
    #         indexes = (rand.choice(self.index, self.shape[0], replace=True) for i in range(n_samples))
    #         generator = ((SparseDataFrame(data_csr[idx, :], columns=self.columns), idx)
    #                      for idx in indexes)
    #         return generator, self.shape[0]
    #
    #     else:
    #         unique = self[groups].unique()
    #         num_unique = len(unique)
    #         boot_col_2_int = dict(zip(unique, range(num_unique)))
    #         index_to_group_int = self[groups].map(boot_col_2_int)
    #         grp_mat = coo_matrix((np.ones(len(self)), (np.arange(len(self)).astype(int), index_to_group_int))
    #                              ).tocsc()
    #
    #         groups_to_enum_dict, group_to_row_lists_dict \
    #             = AbsorbTools.get_absorb_mappings(SparseFormulaDataObj(grp_mat, column_names=unique,
    #                                                                    null_rows=set()))
    #
    #         def _sample_groups(groups_selected):
    #             idx = list(itertools.chain.from_iterable(group_to_row_lists_dict[u] for u in groups_selected))
    #             return SparseDataFrame(data_csr[idx, :], columns=self.columns), idx
    #
    #         selecteds = (rand.choice(unique, num_unique, replace=True) for j in range(n_samples))
    #         return (_sample_groups(s) for s in selecteds), num_unique
#
#
# if __name__ == '__main__':
#
#     import pandas as pd
#     from kanly.api import ols
#
#     n, k = 20, 5
#     np.random.seed(0)
#     X = np.random.randn(n, k)
#     df = pd.DataFrame(data=X, columns=[f'z_{j}' for j in range(k)])
#     df['y'] = np.random.randn(n) + df.z_1
#     df['grps'] = np.random.randint(0, 5, n)
#     # sdf = SparseDataFrame(csc_matrix(df.values), columns=df.columns)
#     # print(sdf.todataframe())
#     # print(df)
#     # df.index = np.random.choice(np.arange(n*10), n, replace=False)
#     #
#     # print(sdf['z_1'], f"\ndiff={max(np.abs(sdf['z_1'] - df['z_1']))}")
#     # print(sdf[['z_1', 'y']], f"\ndiff={np.max(np.abs((sdf[['z_1', 'y']] - df[['z_1', 'y']]).values))}")
#     #
#     # fit = ols2('y ~ z_1', sdf).fit(cov_type='cluster', cov_kwds={'groups': df.grps})
#     # print(fit.summary())
#     #
#     # print(isinstance(sdf, SparseDataFrame))
#     #
#     # fit = ols('y ~ z_1', sdf, cov_type='cluster', cov_kwds={'groups': 'grps'})
#     # print(fit)
#
#     sdf = SparseDataFrame(
#         [
#             {'data': df['y']},
#             {'data': df[['grps', 'z_1']]},
#             {'data': df[['z_2']].values, 'columns': ['z_2']},
#             {'data': csc_matrix(X), 'columns': [f's{k}' for k in range(X.shape[1])]}
#         ] + [
#             {'data': csc_matrix((df['grps'] == k).astype(float)).reshape((-1, 1)).tocsc(),
#              'columns': [f'grp_eq_{k}']} for k in range(4)
#         ]
#     )
#
#     print([csc_matrix((df['grps'] == k).astype(int)).reshape((-1, 1)).tocsc() for k in range(4)])
#
#     print(sdf['y'])
#     print(sdf['z_2'])
#     print(sdf[['y', 's1', 'z_2']])
#     print(sdf['s1'])
#     print(sdf['grp_eq_0'])
#     print(str(sdf))
#
#     fit = ols('y ~ s1 + grp_eq_0', sdf)
#     print(fit)
#
#     df['grp_eq_0'] = (df.grps == 0)
#     df['s1'] = X[:, 1]
#     fit = ols('y ~ s1 + grp_eq_0', df)
#     print(fit)