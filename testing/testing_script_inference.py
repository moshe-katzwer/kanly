from __future__ import absolute_import, print_function

import pandas as pd
import numpy as np

import statsmodels.formula.api as smf
from numpy.testing import assert_array_almost_equal

from kanly.api import lm, compare_results

np.random.seed(0)
n = 300
df = pd.DataFrame({'x': np.random.randint(0, 4, n), 'z': np.arange(n),
                   'w': np.random.randn(n),
                   'grp1': np.random.randint(0, 40, n),
                   'grp2': np.random.randint(0, 2, n).astype(str),
                   'grp3': np.random.randint(0, 5, n),
                   'wtsvar': .5 + np.random.rand(n)},
                   index=np.random.choice(np.arange(10 * n), n, replace=False)  # TODO
                  )

df.sort_values(by='grp2', inplace=True)

# df = pd.DataFrame({'x': np.random.randint(0, 4, n), 'z': np.arange(n),
#                    'w': np.random.randn(n),
#                    'grp1': np.random.randint(0, 40, n),
#                    'grp2': np.repeat(np.arange(30), n//30).astype(str),
#                    'grp3': np.random.randint(0, 5, n),
#                    'wtsvar': .5 + np.random.rand(n)},
#                    index=np.random.choice(np.arange(10 * n), n, replace=False)  # TODO
#                   )

e = np.random.randn(n)
e[1:] += .4 * e[:-1]
df['y'] = 15.3 - 2.5 * df.x + 3 * (df.grp2 == 1) + e

failed = []

cov_param_dict = {
    'hac-panel': [
        ({'maxlags': 2, 'kernel': 'bartlett', 'use_correction': False, 'groups': 'grp2'},
         {'maxlags': 2, 'kernel': 'bartlett', 'use_correction': False, 'groups': df.grp2, 'df_correction': False}),
        ({'maxlags': 2, 'kernel': 'uniform', 'use_correction': False, 'groups': 'grp2'},
         {'maxlags': 2, 'kernel': 'uniform', 'use_correction': False, 'groups': df.grp2, 'df_correction': False}),
        # ({'maxlags': 2, 'kernel': 'uniform', 'use_correction': True, 'groups': 'grp2'}, TODO
        # {'maxlags': 2, 'kernel': 'uniform', 'use_correction': 'hac-panel', 'groups': df.grp2, 'df_correction': False}),
    ],
    'ols_small': [(dict(), dict())],
    'nonrobust': [(dict(), dict())],
    'hc0': [(dict(), dict())],
    'hc1': [(dict(), dict())],
    'hc2': [(dict(), dict())],
    'hc3': [(dict(), dict())],
    'cluster': [({'groups': 'grp1'},
                 {'groups': df['grp1']}),
                ({'groups': ('grp1', 'grp3')},
                 {'groups': (df.grp1, df.grp3)}),
                ({'groups': 'grp1', 'use_correction': False},
                 {'groups': df['grp1'], 'use_correction': False}),
                ],
    'hac': [({'maxlags': 2, 'kernel': 'bartlett', 'use_correction': False},
             {'maxlags': 2, 'kernel': 'bartlett', 'use_correction': False}),
            ({'maxlags': 4, 'kernel': 'bartlett', 'use_correction': False},
             {'maxlags': 4, 'kernel': 'bartlett', 'use_correction': False}),
            ({'maxlags': 0, 'kernel': 'uniform', 'use_correction': False, },
             {'maxlags': 0, 'kernel': 'uniform', 'use_correction': False}),
            ({'maxlags': 4, 'kernel': 'uniform', 'use_correction': False},
             {'maxlags': 4, 'kernel': 'uniform', 'use_correction': False}),
            ({'maxlags': 2, 'kernel': 'uniform', 'use_correction': False},
             {'maxlags': 2, 'kernel': 'uniform', 'use_correction': False}),
            ({'maxlags': 2, 'kernel': 'bartlett', 'use_correction': True},
             {'maxlags': 2, 'kernel': 'bartlett', 'use_correction': True}),
            ({'maxlags': 4, 'kernel': 'bartlett', 'use_correction': True},
             {'maxlags': 4, 'kernel': 'bartlett', 'use_correction': True}),
            ({'maxlags': 0, 'kernel': 'uniform', 'use_correction': True, },
             {'maxlags': 0, 'kernel': 'uniform', 'use_correction': True}),
            ({'maxlags': 4, 'kernel': 'uniform', 'use_correction': True},
             {'maxlags': 4, 'kernel': 'uniform', 'use_correction': True}),
            ({'maxlags': 2, 'kernel': 'uniform', 'use_correction': True},
             {'maxlags': 2, 'kernel': 'uniform', 'use_correction': True}),
            ],
}

for cov_type, cov_kwd_list in cov_param_dict.items():

    print()
    print("#" * 50)
    print()

    for cov_kwds_ch, cov_kwds_sm in cov_kwd_list:

        for use_t in [True, False]:

            for is_weighted in [False, True]:

                fit_kn = lm('y ~ C(grp2) + x' + (' $ wtsvar' if is_weighted else ''), df,
                            # debug=True,
                            cov_type=cov_type, cov_kwds=cov_kwds_ch, use_t=use_t)

                if is_weighted:
                    weights = df.wtsvar
                else:
                    weights = np.ones(len(df))

                if cov_type == 'ols_small' or cov_type == 'nonrobust':
                    fit_sm = smf.wls('y ~ x + C(grp2)', df, weights=weights).fit(use_t=use_t)
                else:
                    fit_sm = smf.wls('y ~ x + C(grp2)', df, weights=weights).fit(
                        cov_type=cov_type, cov_kwds=cov_kwds_sm, use_t=use_t)

                # print(fit_ch, fit_sm.summary())

                print('\n' + "-"*50)
                spec = (f"cov_type={cov_type}", f"cov_kwds={cov_kwds_ch}", f'use_t={use_t}',
                        f'is_weighted={is_weighted}')
                print('\t', spec)


                try:
                    # sort because of different param ordering
                    assert_array_almost_equal(sorted(fit_sm.params), sorted(fit_kn.params))
                    print("\tpassed params")

                    try:
                        assert_array_almost_equal(
                            sorted(fit_sm.bse * (
                                np.sqrt((n - 3) / n) if cov_type == 'nonrobust' else 1)),
                            sorted(fit_kn.bse)
                        )
                        print("\tpassed std err")

                    except:
                        # print(fit_sm.summary(), fit_ch)
                        print(fit_sm.bse, "\n", fit_kn.bse)
                        print(fit_sm.bse.values / fit_kn.bse.values)
                        print(fit_sm._cov_params() / fit_kn._cov_params().values)
                        failed.append(('STD ERR', spec))
                        print("\tFAILED std err")
                        raise Exception

                except:
                    # print(fit_sm.summary(), fit_ch)
                    failed.append(('PARAMS', spec))
                    print(compare_results([fit_sm, fit_kn]))
                    print("\tFAILED params")
                    raise Exception

if failed:
    print("\n\nFailed: ")
    for j in failed:
        print("\t", j)
else:
    print("\n\nALL PASSED")
