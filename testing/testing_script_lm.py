import traceback as tb

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from scipy.sparse import hstack, csc_matrix
from statsmodels.sandbox.regression.gmm import IV2SLS

from kanly.api import lm, LM
from kanly.formula.data_getter import SparseDataGetter

np.random.seed(0)
n = 120
df = pd.DataFrame({'x': np.random.randint(0, 4, n), 'z': np.arange(n),
                   'x_zero': 0.0,
                   'w': np.random.randn(n), 'grp1': np.random.randint(0, 4, n), 'city': np.random.randint(0, 3, n),
                   'grp2': np.random.randint(0, 2, n).astype(str),
                   'wtsvar': .5 + np.random.rand(n)},
                  index=np.random.choice(np.arange(10 * n), n, replace=False)  # TODO
                  )

e = np.random.randn(n)
df['z'] = -3 + .15 * df.w + .4 * e + 2 * np.random.randn(n)
df['y'] = 3 + 1.2 * df['x'] + df.z + 3 * e + df.city - 1.2 * df['grp2'].map(lambda x: int(x))
df['q'] = np.random.randn(n)
df['_one_'] = 1
df['city_grp2'] = df['city'].astype(str) + "_" + df['grp2'].astype(str)

df_nulls = df.copy()
df_nulls.loc[37, 'z'] = np.nan
df_nulls.loc[[229, 885], 'y'] = np.nan
df_nulls.loc[[317, 845], 'wtsvar'] = -1
df_nulls.loc[[344, 229, 536], 'wtsvar'] = 0.0

df_nulls.loc[384, 'grp1'] = None

df_nulls.loc[740, 'city'] = None
df_nulls.loc[740, 'city_grp2'] = None

debug = False

failed_runs = []
num_successes = 0
num_runs = 0

_int_idx_ints = np.random.choice(range(50), int(3 / 4 * n), replace=True)

_int_idx_bools = np.array([False] * n)
_int_idx_bools[np.random.choice(range(50), int(3 / 4 * n), replace=True)] = True

for do_iv in [False, True]:
    for do_absorb in [None, 'city', ('city', 'grp2')]:
        for has_intercept in [False, True]:
            for with_nulls in [False, True]:
                for use_int_idx in [False, 'int', 'bool']:
                    for use_patsy_weights in [True]:  # REMOVED WEIGHTS ARGUMENT FROM LM FUNC!
                        for scale_design_matrix in [False, True]:
                            for do_weighted in [False, True]:  # TODO DENSE 4444

                                tbl = None

                                if not use_int_idx:
                                    integer_idx = None
                                elif use_int_idx == 'int':
                                    integer_idx = _int_idx_ints
                                elif use_int_idx == 'bool':
                                    integer_idx = _int_idx_bools

                                if not has_intercept and do_absorb is not None:
                                    continue

                                num_runs += 1

                                try:
                                    print('\n' * 6 + '=' * 150)
                                    print(("nulls=%5s, instrumental_variables=%5s, wtd=%5s, absorb=%5s, intercept=%5s, "
                                           "use_patsy_weights=%5s, use_int_idx=%5s, scale_design_matrix=%5s")
                                          % (with_nulls, do_iv, do_weighted, do_absorb, has_intercept, use_patsy_weights, use_int_idx, scale_design_matrix))

                                    intercept_str = '' if has_intercept else '   -1'
                                    # weights = 'I(wtsvar/2)' if do_weighted and not use_patsy_weights else None
                                    weight_str = ' $ I(wtsvar/2)' if do_weighted and use_patsy_weights else ''
                                    instruments = '| w + I(q**2) + x_zero' + intercept_str if do_iv else ''

                                    formula = 'y ~ z + I(q**2) + x_zero' + intercept_str + instruments + weight_str

                                    data = df_nulls if with_nulls else df

                                    formula_for_null = formula #+ ("$ I(wtsvar/2)" if do_weighted else "")
                                    null_rows = set(df.index[list(SparseDataGetter.get_null_indices_for_formula(
                                        formula_for_null, data, absorb=do_absorb, debug=False))])

                                    if with_nulls:
                                        valid_rows = df.index[df.index.isin(list(set(df.index) - null_rows))]
                                    else:
                                        valid_rows = df.index

                                    # with model
                                    fit = lm(formula, data,
                                             absorb=do_absorb, debug=debug,
                                             cov_type='HC1', use_t=True, scale_design_matrix=scale_design_matrix,
                                             index=integer_idx)
                                    print(fit.summary())
                                    # if not do_iv:
                                    #     fit.predict(df)

                                    # without model
                                    fit2 = lm(formula, data,
                                              absorb=do_absorb, debug=debug, scale_design_matrix=scale_design_matrix,
                                              cov_type='HC1', use_t=True, keep_model=False,
                                              index=integer_idx)
                                    print(fit2.null_rows_info_dict)
                                    fit2.summary()

                                    kanly_params = [fit.params['z'], fit.params['I(q**2)']]

                                    if use_int_idx:
                                        data_sm = data.iloc[integer_idx]
                                    else:
                                        data_sm = data

                                    data_sm = data_sm.loc[data_sm.index.isin(valid_rows)].copy()

                                    # By hand
                                    if do_absorb is None:
                                        G = np.ones((len(data_sm), 1))
                                    elif do_absorb == 'city':
                                        G = patsy.dmatrix('C(city) -1', data_sm)
                                    else:
                                        G = patsy.dmatrix('C(city_grp2) -1', data_sm)

                                    X = patsy.dmatrix('z + I(q**2) -1', data_sm)
                                    Z = patsy.dmatrix('w + I(q**2) -1', data_sm) if do_iv else X

                                    y = patsy.dmatrix('y -1', data_sm)

                                    w = data_sm.wtsvar.values if do_weighted else None
                                    W = np.diag(w) if do_weighted else np.eye(len(data_sm))

                                    # DON'T NEED THIS HERE BECAUSE WE DO IT IN THE LM FUNCTION NOW
                                    # _X_REG = hstack(
                                    #     [csc_matrix(np.ones(y.shape)), X]) if has_intercept and not do_absorb else X
                                    # _Z_REG = hstack(
                                    #     [csc_matrix(np.ones(y.shape)), Z]) if has_intercept and not do_absorb else Z
                                    _X_REG, _Z_REG = X, Z
                                    add_constant = has_intercept and not do_absorb

                                    REG_params = LM(
                                        y, _X_REG, add_constant=add_constant, weights=w, instruments=_Z_REG if do_iv else None,
                                        absorb=G if do_absorb else None,
                                        scale_design_matrix=scale_design_matrix
                                    )._params[-2:]

                                    REG_params_sparse = LM(
                                        y, csc_matrix(_X_REG), add_constant=add_constant, weights=w,
                                        instruments=_Z_REG if do_iv else None,
                                        absorb=G if do_absorb else None,
                                        scale_design_matrix=scale_design_matrix
                                    )._params[-2:]

                                    # if w is not None:
                                    #     print('*****>>> ', np.linalg.pinv(_X_REG.T.dot(W).dot(_X_REG)).dot(_X_REG.T.dot(W).dot(y)))

                                    # Weight the matrices
                                    rtW = np.sqrt(W)
                                    Gw = rtW.dot(G)
                                    Xw = rtW.dot(X)
                                    Zw = rtW.dot(Z)
                                    yw = rtW.dot(y)

                                    print(data_sm.shape, Xw.shape, rtW.shape, X.shape, fit.nobs)

                                    if has_intercept:
                                        # Build annihalator
                                        annihalator_G = Gw.dot(np.linalg.pinv(Gw.T.dot(Gw))).dot(Gw.T)

                                        # Absorb
                                        absorbed_y = yw - annihalator_G.dot(yw)
                                        absorbed_X = Xw - annihalator_G.dot(Xw)
                                        absorbed_Z = Zw - annihalator_G.dot(Zw)
                                    else:
                                        absorbed_y = yw
                                        absorbed_X = Xw
                                        absorbed_Z = Zw

                                    # First stage IV
                                    Pi = np.linalg.pinv(absorbed_Z.T.dot(absorbed_Z)).dot(absorbed_Z.T.dot(absorbed_X))
                                    X_pred = absorbed_Z.dot(Pi)
                                    # if has_intercept:
                                    #     Pi_to_compare = np.zeros((3, 3))
                                    #     Pi_to_compare[1:, 1:] = Pi
                                    #     Pi_to_compare[0, 0] = 1.0
                                    #     if w is None:
                                    #         Pi_to_compare[0, 1] = X[:, 1].mean() - X_pred[:, 1].mean()
                                    #     else:
                                    #         Pi_to_compare[0, 1] = (
                                    #             np.average(Xw[:, 1], weights=np.sqrt(w))
                                    #             - np.average(Xw[:, 1], weights=np.sqrt(w))
                                    #         )
                                    #
                                    # else:
                                    #     Pi_to_compare = Pi

                                    # Second stage IV
                                    beta = np.linalg.pinv(X_pred.T.dot(X_pred)).dot(X_pred.T.dot(absorbed_y))

                                    sm_bse = None

                                    if do_absorb is None:
                                        sm_absorb_str = ''
                                    elif do_absorb == 'city':
                                        sm_absorb_str = ' + C(city) '
                                    else:
                                        sm_absorb_str = ' + C(city)*C(grp2)'

                                    if do_iv:
                                        data_sm.loc[:, 'z_pred'] = smf.wls('z ~ w + I(q**2)' + sm_absorb_str + intercept_str,
                                                               data_sm, weights=(data_sm['wtsvar'] if do_weighted else data_sm['_one_'])).fit().fittedvalues
                                        fitsm = smf.wls('y ~ z_pred + I(q**2)' + sm_absorb_str + intercept_str,
                                                        data_sm, weights=(data_sm['wtsvar'] if do_weighted else data_sm['_one_'])).fit()
                                        params = fitsm.params[['z_pred', 'I(q ** 2)']]

                                        fit_sm_iv2sls = IV2SLS(absorbed_y, absorbed_X, absorbed_Z).fit()

                                    else:
                                        fitsm = smf.wls('y ~ z + I(q**2)' + sm_absorb_str + intercept_str,
                                                        data_sm, weights=(data_sm['wtsvar'] if do_weighted else data_sm['_one_']))\
                                            .fit(use_correction=True, use_t=True, data_correction=False, cov_type='HC1')
                                        params = fitsm.params[['z', 'I(q ** 2)']]
                                        sm_bse = fitsm.bse[['z', 'I(q ** 2)']]
                                        fit_sm_iv2sls = None

                                    print(fitsm.summary())
                                    if fit_sm_iv2sls is not None:
                                        print(fit_sm_iv2sls.summary())

                                    tbl = pd.DataFrame({'by_hand': np.array(beta).flatten(), 'kanly': kanly_params,
                                                        'sm': list(params),
                                                        'kanly_REG': REG_params,
                                                        'kanly_REG_sparse': REG_params_sparse,
                                                        },
                                                       index=['z', 'I(q**2)'])
                                    if fit_sm_iv2sls is not None:
                                        tbl['sm_iv'] = fit_sm_iv2sls.params

                                    print(("\n\t\tnulls=%5s, instrumental_variables=%5s, wtd=%5s, absorb=%5s, intercept=%5s, "
                                           "use_patsy_weights=%5s, use_int_idx=%5s, scale_design_matrix=%5s")
                                          % (
                                          with_nulls, do_iv, do_weighted, do_absorb, has_intercept, use_patsy_weights,
                                          use_int_idx, scale_design_matrix))
                                    print("\nPARAM TABLE:")
                                    print(tbl.to_string())
                                    assert_array_almost_equal(tbl.kanly, tbl.sm)
                                    assert_array_almost_equal(tbl.kanly, tbl.by_hand)
                                    assert_array_almost_equal(tbl.kanly_REG, tbl.by_hand)
                                    assert_array_almost_equal(tbl.kanly_REG_sparse, tbl.by_hand)
                                    if do_iv:
                                        assert_array_almost_equal(tbl.kanly, tbl.sm_iv)
                                        assert_array_almost_equal(
                                            Pi,
                                            fit.instrument_info.instrument_params.toarray()[-2:, -2:])

                                    assert_almost_equal(fitsm.nobs, fit.nobs)

                                    if do_iv:
                                        if not do_absorb:
                                            assert_almost_equal(fit_sm_iv2sls.df_model, fit.df_model)
                                            assert_almost_equal(fit_sm_iv2sls.rsquared, fit.rsquared)
                                            #assert_almost_equal(fit_sm_iv2sls.rsquared_adj, fit.rsquared_adj) # TODO
                                            #assert_array_almost_equal(fit_sm_iv2sls.bse[-2:], fit.bse[['z', 'I(q**2)']])
                                    else:
                                        assert_almost_equal(fitsm.fittedvalues, fit.fittedvalues)
                                        assert_almost_equal(fitsm.rsquared, fit.rsquared)
                                        assert_almost_equal(fitsm.rsquared_adj, fit.rsquared_adj)
                                        assert_almost_equal(fitsm.resid, fit.resid)
                                        assert_almost_equal(fitsm.df_resid, fit.df_resid)
                                        assert_almost_equal(fitsm.df_model, fit.df_model)
                                        assert_array_almost_equal(sm_bse, fit.bse[['z', 'I(q**2)']])
                                        assert_almost_equal(fitsm.llf, fit.llf)
                                        assert_almost_equal(fitsm.aic, fit.aic)
                                        assert_almost_equal(fitsm.bic, fit.bic)

                                        #print(sm_bse, fit.bse[['z', 'I(q**2)']])
                                    #from kanly.formula.formula import SparseFormula
                                    #print(SparseFormula.get_null_indices_for_formula(formula, df_nulls))

                                    num_successes += 1

                                except Exception as e:

                                    failed_runs.append((
                                        "nulls=%5s, instrumental_variables=%5s, wtd=%5s, absorb=%5s, intercept=%5s, use_patsy_weights=%5s, use_int_idx=%5s"
                                        % (with_nulls, do_iv, do_weighted, do_absorb, has_intercept, use_patsy_weights,
                                           use_int_idx),
                                        null_rows, set(df.index) - set(df.index[fit.model.valid_obs_rows]),
                                        tbl, ''.join(tb.format_exception(None, e, e.__traceback__)))
                                    )

                                    #raise e


print("\n" * 10, "#" * 100)

print(f'{num_successes}/{num_runs} specifications succeeded!')

if failed_runs:
    print("FAILED = %d" % len(failed_runs))
    for f in failed_runs:
        print()
        for k in f:
            print(k)
else:
    print("ALL SUCCESSFUL!")
