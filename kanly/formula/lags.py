"""
Computes lags of a time series
"""
from __future__ import absolute_import, print_function

import pandas as pd


def _do_lags(term, col_name, vals, data):
    """Apply lag/lead transformation encoded in a numerical control term.

    Supports both plain shifts (entire vector) and grouped shifts where lagging
    occurs independently within each provided lag group.

    Args:
        term (SparseTerm): Parsed term that may contain an ``L(...)`` control.
        col_name (str): Current output column name.
        vals (ndarray): Column values before lagging.
        data (DataFrame): Source data; required when group-based lagging is
            requested.

    Returns:
        tuple:
            - ndarray: Lagged/lead values.
            - str: Updated column name with lag prefix metadata.
    """
    assert term.is_monomial()

    if term.numerical_controls[0].is_lag_term():
        lags = term.numerical_controls[0].lags
        prefix = 'L' if lags == 1 else ('F' if lags == -1 else (f'L{lags}' if lags > 1 else f'F{-lags}'))

        if term.numerical_controls[0].has_lag_groups():

            v = vals[:, 0] if len(vals.shape) > 1 else vals

            grps = term.numerical_controls[0].lag_groups
            grps = list(grps)
            # If rows were reindexed/shuffled earlier, preserve original index
            # so groupby(...).shift aligns exactly with the working dataset.
            df_temp = pd.DataFrame({'temp_val': v}, index=data.index)
            df_temp[grps] = data[grps]
            # print(">>>> ", df_temp.groupby(grps)['temp_val'].shift(lags))
            vals = df_temp.groupby(grps)['temp_val'].shift(lags).values
            col_name = f'{prefix}[{col_name}|{grps if len(grps) > 1 else grps[0]}]'

        else:
            vals = vals.flatten()
            col_name_temp = col_name[0] if isinstance(col_name, list) else col_name
            col_name = f'{prefix}[{col_name_temp}]'
            vals = pd.Series(vals).shift(lags).values

        vals.reshape((-1, 1))

    return vals, col_name


# if __name__ == '__main__':
#
#     def main():
#         import pandas as pd
#         import numpy as np
#         from kanly.api import nlls, lm
#         from kanly.formula.sparse_term import SparseTerm
#         import pprint
#
#         M = 3
#         T = 5
#         np.random.seed(0)
#         y = np.random.randn(M * T)
#         y[1:] += .7 * y[:-1]
#         df = pd.DataFrame({'y': y,
#                            't': np.tile(range(T), M),
#                            'g': np.repeat(range(M), T),
#                            })
#
#         #df = df.sort_values(by=['t', 'g'])#.reset_index()
#         print(df)
#
#         fit = lm('y ~ L(y,1,g)', df)
#         #
#         # df2 = pd.DataFrame(fit.model.exog.toarray())
#         # df2['y true'] = df.y[fit.valid_obs_rows].values
#         # print(df2)
#
#         print(fit)
#
#     main()
