from __future__ import absolute_import, print_function

from kanly.api import elastic_net, lm
import pandas as pd
import numpy as np

def isc(data, outcome, unit_col, time_col, treatment_col, covariate_cols=None, max_iter=200, alpha=.0001,
        l1_ratio=.9, positive=True, tol=1e-6, shift=0, do_sc=False, debug=False):
    if covariate_cols is None:
        covariate_cols = []

    cov_mats = dict()
    for c in covariate_cols:
        if debug:
            print("** ", c)
        cov_mats[c] = data.pivot_table(values=[c], index=time_col, columns=unit_col)
        cov_mats[c].columns = ['_'.join(col).strip() for col in cov_mats[c].columns.values]

    unit_cols = sorted(data[unit_col].unique())
    data_piv = data.pivot_table(values=[outcome], index=time_col, columns=unit_col)
    treatment_piv = data.pivot_table(values=treatment_col, index=time_col, columns=unit_col)

    if shift:
        if debug:
            print("###### ", shift)
        idx0 = data_piv.index.copy()
        data_piv = pd.concat([data_piv.iloc[shift:], data_piv.iloc[:shift]])
        data_piv.index = idx0
        for c in covariate_cols:
            idx0 = cov_mats[c].index.copy()
            cov_mats[c] = pd.concat([cov_mats[c].iloc[shift:], cov_mats[c].iloc[:shift]])
            cov_mats[c].index = idx0

    data_piv.columns = ['_'.join(col).strip() for col in data_piv.columns.values]
    units = treatment_piv.columns.copy()
    times = treatment_piv.index.copy()
    treatment_units = {u: np.nonzero(treatment_piv.values[:, i])[0][0]
                       for i, u in enumerate(units)
                       if treatment_piv.values[:, i].sum() != 0
                       }

    min_treatment = min(treatment_units.values())
    pred = dict()
    fits = dict()
    data[f'{outcome}_predicted'] = np.nan
    data[f'{treatment_col}_predicted'] = np.nan

    treatment_df_dummy = treatment_piv.copy()
    treatment_df_dummy.columns = data_piv.columns.copy()

    for i, u in enumerate(unit_cols):
        if u not in treatment_units.keys():
            continue

        formula = f"{outcome}_{u} ~ " + " + ".join([f"{outcome}_{j}" for j in units if j != u])
        if len(covariate_cols):
            formula += " + " + ' + '.join([f'__{c}_{u}' for c in covariate_cols])

        if debug:
            print(formula)

        data_piv_u = data_piv.copy()
        for c in covariate_cols:
            data_piv_u[f'__{c}_{u}'] = cov_mats[c][f'{c}_{u}'].values

        for c in covariate_cols:
            treatment_df_dummy[f'__{c}_{u}'] = cov_mats[c][f'{c}_{u}'].values

        # print(data_piv_u.columns)
        # display(data_piv_u)

        __alpha = {f"{outcome}_{j}": alpha for j in unit_cols if j != u}
        if do_sc:
            for u2 in treatment_units.keys():
                __alpha[f"{outcome}_{u2}"] = 1e20

        if positive:
            positive = {f"{outcome}_{j}": True for j in unit_cols if j != u}

        # __alpha = alpha
        print(__alpha)
        fit_en = elastic_net(
            formula
            ,
            data_piv_u,
            index=(
                data_piv_u.index < data_piv_u.index[treatment_units[u]]
                if do_sc else
                data_piv_u.index < data_piv_u.index[min_treatment]
            ),
            alpha=__alpha, l1_ratio=l1_ratio, normalize=True,
            fit_intercept=True, positive=positive,
            debug=False, max_iter=max_iter, tol=tol,
            apply_scaling=False,
            refit=False,
        )

        if debug:
            print(fit_en)
        pred[u] = fit_en.predict(data_piv_u)
        fits[u] = fit_en
        data.loc[(data[unit_col] == u), f'{outcome}_predicted'] = pred[u]
        data.loc[(data[unit_col] == u), f'{outcome}_resid'] = data_piv_u[f'{outcome}_{u}'].values - pred[u]

        par2 = fit_en.params.copy(deep=True)
        par2['Intercept'] = 0.0
        for c in covariate_cols:
            par2[f'__{c}_{u}'] = 0.0
        data.loc[(data[unit_col] == u), f'{treatment_col}_predicted'] = fit_en.predict(treatment_df_dummy, params=par2)

    fit_final = lm(f'{outcome}_resid ~ I({treatment_col} - {treatment_col}_predicted):C({unit_col}) -1',
                   data,
                   index=(data[time_col] >= data[time_col][min_treatment]) & data[unit_col].isin(treatment_units.keys())
                   )

    fit_final_pooled = lm(f'{outcome}_resid ~ I({treatment_col} - {treatment_col}_predicted) -1',
                          data,
                          index=(data[time_col] >= data[time_col][min_treatment]) & data[unit_col].isin(
                              treatment_units.keys())
                          )

    if debug:
        display(fit_final)
        display(fit_final_pooled)

    return {'pred': pred, 'treatment_units': treatment_units, 'data_pivot': data_piv, 'times': times,
            'fits_first_stage': fits, 'fit_second_stage': fit_final, 'fit_second_stage_pooled': fit_final_pooled,
            'outcome': outcome}
