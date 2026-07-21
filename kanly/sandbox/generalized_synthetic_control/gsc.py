from __future__ import absolute_import, print_function

import numpy as np
import pandas as pd

from tqdm import tqdm
from kanly.regression.nonlinear_least_squares.optimize.nlls_coordinate_descent_minimize_internal import \
    nlls_elastic_net_minimize_internal_coordinate_descent


def get_gsc_pred_func(data, outcome_col, unit_col, time_col, treatment_cols, covariate_cols=None,
                      log_transform=False, debug=True, pooled_treatment=False, shift=0, weight_col=None,
                      fit_only_treatment_units=True, weight_control_only=False,
                      permute=None):
    if covariate_cols is None:
        covariate_cols = []

    df_unit_pivot = pd.pivot(data, index=time_col, columns=unit_col, values=outcome_col)
    unit_list, period_list = list(df_unit_pivot.columns), list(df_unit_pivot.index)

    outcome_matrix = df_unit_pivot.reset_index(drop=True).values  # T x N

    # WEIGHTS
    if weight_col is None:
        outcome_weight_matrix = None
    else:
        df_outcome_wt_pivot = pd.pivot(data, index=time_col, columns=unit_col, values=weight_col)
        outcome_weight_matrix = df_outcome_wt_pivot.reset_index(drop=True).values

    n_time, n_units = outcome_matrix.shape

    # TREATMENT
    df_trt_pivots = [pd.pivot(data, index=time_col, columns=unit_col, values=tc)
                     for tc in treatment_cols]
    treatment_matrices = [d.reset_index(drop=True).values for d in df_trt_pivots]  # T x (N * num_treatments)

    if permute is not None:
        treatment_matrices = [trm[:, permute] for trm in treatment_matrices]

    treatment_units_bool = np.vstack(
        [np.abs(tm).sum(axis=0) > 0 for tm in treatment_matrices])  # row is a treatment, col a unit
    n_trtd = dict()
    trtd_units = dict()
    for i_trt, t in enumerate(treatment_units_bool):
        n_trtd[i_trt] = np.count_nonzero(t)
        trtd_units[i_trt] = np.arange(n_units)[t]
        assert n_trtd[i_trt]
    treatment_units_cols = treatment_units_bool.sum(axis=0).astype(bool)
    not_treatment_units_cols = ~treatment_units_cols
    treatment_matrix = np.hstack(treatment_matrices)

    if shift != 0:
        treatment_matrix = np.vstack((treatment_matrix[shift:, :], treatment_matrix[:shift, :]))

    # COVARIATES
    covariate_cols = ['__Intercept__'] + covariate_cols.copy()
    data['__Intercept__'] = 1.0
    covariate_matrix = pd.pivot(data, index=time_col, columns=unit_col, values=covariate_cols).values
    n_covariates = len(covariate_cols)
    del data['__Intercept__']
    covariate_cols[0] = 'Intercept'

    def sc_weight_params_2_matrix(param_weights):
        param_weights = np.asarray(param_weights).reshape((n_units, n_units - 1))
        v1 = np.hstack((np.tril(param_weights, -1), np.zeros((n_units, 1))))
        v2 = np.hstack((np.zeros((n_units, 1)), np.triu(param_weights, 0)))
        v1 + v2
        return (v1 + v2).transpose()

    def resid_func(params):

        params = np.asarray(params).ravel()

        if pooled_treatment:
            n_trtd_params = len(treatment_cols)
        else:
            n_trtd_params = np.sum(list(n_trtd.values()))

        weight_params, treatment_params, covariate_params \
            = params[:n_units * (n_units - 1)], \
            params[n_units * (n_units - 1):(n_units * (n_units - 1) + n_trtd_params)], \
            params[(n_units * (n_units - 1) + n_trtd_params):]

        sc_weight_param_mat = sc_weight_params_2_matrix(weight_params)
        if fit_only_treatment_units:
            sc_weight_param_mat[:, not_treatment_units_cols] = 0.0
        if weight_control_only:
            sc_weight_param_mat[treatment_units_cols, :] = 0.0

        trt_params_expanded = []
        j = 0
        for i, t in enumerate(treatment_cols):
            trt_params_t = np.zeros(n_units)
            # print(len(trt_params_expanded), n_trtd[i], len(treatment_units_bool))
            if pooled_treatment:
                trt_params_t[treatment_units_bool[i]] = np.ones(n_trtd[i]) * treatment_params[i]
            else:
                trt_params_t[treatment_units_bool[i]] = treatment_params[j:j + n_trtd[i]]
                j = j + n_trtd[i]
            trt_params_expanded.append(trt_params_t)
        trt_params_expanded = np.hstack(trt_params_expanded)

        trt_params_expanded = np.diag(trt_params_expanded)
        trt_pred = treatment_matrix.dot(trt_params_expanded)

        trt_pred = sum([trt_pred[:, j * n_units:(j + 1) * n_units] for j in range(len(treatment_cols))])

        # print(pd.DataFrame(covariate_matrix))
        covariate_params = covariate_params.reshape((n_units, n_covariates))
        # print(covariate_params)
        covariate_params = np.hstack([np.diag(z) for z in covariate_params.T]).T
        # print(covariate_params)

        covar_pred = covariate_matrix.dot(covariate_params)

        #         print(trt_pred.shape)
        #         print(covar_pred.shape)
        #         print(outcome_matrix.shape)

        resid1 = outcome_matrix - (trt_pred + covar_pred)
        resid = resid1 - resid1.dot(sc_weight_param_mat)
        if weight_col is not None:
            resid *= outcome_weight_matrix

        return resid, {
            "outcomes": outcome_matrix,
            'sc_weight_param_mat': sc_weight_param_mat,
            'outcome_weight_matrix': outcome_weight_matrix,

            'treatment_units_cols': treatment_units_cols,

            'resid': resid,

            "resid1": resid1,
            "trd_pred": trt_pred,

            "covar_pred": covar_pred,
            "covariate_params": covariate_params,
            "covariate_matrix": covariate_matrix,

            "trt_params_expanded": trt_params_expanded,
            "treatment_matrix": treatment_matrix,
        }

    param_names = [
        f'omega_{unit_list[i]}_{unit_list[j]}'
        for i in range(n_units) for j in range(n_units) if i != j]
    for n_t, treatment_col in enumerate(treatment_cols):
        param_names += ([f'{treatment_col}_{unit_list[i]}' for i in trtd_units[n_t]]
                        if not pooled_treatment else
                        [treatment_col]
                        )
    param_names += [f'{x}_{unit_list[i]}' for i in range(n_units) for x in covariate_cols]

    # print(param_names)
    # print(len(param_names))

    return resid_func, param_names, n_units, n_time, unit_list, period_list, outcome_matrix


def gsc(data, outcome_col, unit_col, time_col, treatment_cols, covariate_cols=None, weight_col=None,
        log_transform=False, debug=True, pooled_treatment=False, positive=False, alpha=1, l1_ratio=.9,
        max_iter=500, ftol=1e-6, xtol=1e-6, weight_control_only=False,
        prompt_user_for_more_iters=False, selection='random', fit_only_treatment_units=True,
        do_conformal=False, conformal_stride=1,
        do_permutation=False, num_permutation_draws=100,
        ):
    if debug:
        print('Making the residual function...', end='')

    if isinstance(treatment_cols, str):
        treatment_cols = [treatment_cols]
    if covariate_cols is not None and isinstance(covariate_cols, str):
        covariate_cols = [covariate_cols]

    resid_func, param_names, n_units, n_time, unit_list, period_list, outcome_matrix = get_gsc_pred_func(
        data, outcome_col, unit_col, time_col, treatment_cols, covariate_cols=covariate_cols,
        log_transform=log_transform, debug=debug, pooled_treatment=pooled_treatment, weight_col=weight_col,
        fit_only_treatment_units=fit_only_treatment_units, weight_control_only=weight_control_only)
    if debug:
        print('done')

    n_params = len(param_names)
    #     alpha_vec = np.zeros(n_params)
    #     print(outcome_matrix.shape)
    #     z = np.tile(outcome_matrix.std(axis=0), n_units)
    #     for i in range(n_units):
    #         z[(n_units + 1) * i] = 0
    #     z = z[z > 0]
    #     alpha_vec[:n_units * (n_units - 1)] = alpha / z

    alpha_vec = np.zeros(n_params)
    alpha_vec[:n_units * (n_units - 1)] = alpha

    if positive:
        bounds = np.array([(0.0, np.inf)] * n_units * (n_units - 1)
                          + [(-np.inf, np.inf)] * (len(param_names) - n_units * (n_units - 1)))
    else:
        bounds = None

    # print("*** bounds ", bounds)

    optimization_result = nlls_elastic_net_minimize_internal_coordinate_descent(
        lambda p: resid_func(p)[0].flatten(),
        num_params=n_params,
        bounds=bounds,
        max_iter=max_iter,
        ftol=ftol,
        xtol=xtol,
        alpha=alpha_vec,
        l1_ratio=l1_ratio,
        debug=debug,
        selection=selection,
        prompt_user_for_more_iters=prompt_user_for_more_iters
    )

    params = pd.Series(index=param_names, data=optimization_result['params'])
    resid, resid_info = resid_func(optimization_result['params'])

    fittedvalues = [(resid_info['covar_pred'][:, j] + resid_info['trd_pred'][:, j]
                     + resid_info['resid1'].dot(resid_info['sc_weight_param_mat'])[:, j])
                    for j in range(n_units)]

    result = {
        'data': data,
        'params': params,
        'optimization_result': optimization_result,
        'resid': pd.DataFrame(resid, index=period_list, columns=unit_list),
        'resid_info': resid_info,
        'fittedvalues': pd.DataFrame(np.array(fittedvalues).T, index=period_list, columns=unit_list),
        'resid_func': resid_func,
        'param_names': param_names,
        'pooled_treatment': pooled_treatment,
        'outcome_col': outcome_col,
        'unit_col': unit_col,
        'time_col': time_col,
        'treatment_cols': treatment_cols,
        'covariate_cols': list(covariate_cols) + [] if covariate_cols is not None else None,
        'log_transform': log_transform,
        'unit_list': unit_list,
        'period_list': period_list,
        'bounds': bounds,
        'fit_only_treatment_units': fit_only_treatment_units,
        'n_time': n_time,
        'n_units': n_units,
        'n_params': n_params,
        'positive': positive,
        'weight_control_only': weight_control_only,
        'nlls_en_options': {
            'alpha_vec': alpha_vec,
            'ftol': ftol,
            'xtol': xtol,
            'max_iter': max_iter,
            'selection': selection,
            'alpha': alpha,
            'l1_ratio': l1_ratio,
            'bounds': bounds,
        },
        'conformal_inference_info': None,
        'permutation_inference_info': None,
    }

    if do_conformal:
        conformal_inference(result, conformal_stride, debug)

    if do_permutation:
        permutation_inference(result, num_permutation_draws, debug)

    return result


def conformal_inference(gsc_result_dict, conformal_stride=1, debug=False):
    params_conformal = []
    for shift in tqdm(range(1, gsc_result_dict['n_time'], conformal_stride)):
        resid_func_k, *_ = get_gsc_pred_func(
            data=gsc_result_dict['data'], outcome_col=gsc_result_dict['outcome_col'],
            unit_col=gsc_result_dict['unit_col'], time_col=gsc_result_dict['time_col'],
            treatment_cols=gsc_result_dict['treatment_cols'], covariate_cols=gsc_result_dict['covariate_cols'],
            weight_control_only=gsc_result_dict['weight_control_only'], log_transform=gsc_result_dict['log_transform'],
            debug=debug, pooled_treatment=gsc_result_dict['pooled_treatment'], shift=shift)
        params_conformal.append(
            nlls_elastic_net_minimize_internal_coordinate_descent(
                lambda p: resid_func_k(p)[0].flatten(),
                num_params=gsc_result_dict['n_params'],
                x0=gsc_result_dict['params'],
                bounds=gsc_result_dict['nlls_en_options']['bounds'],
                alpha=gsc_result_dict['nlls_en_options']['alpha_vec'],
                max_iter=gsc_result_dict['nlls_en_options']['max_iter'],
                xtol=gsc_result_dict['nlls_en_options']['xtol'],
                ftol=gsc_result_dict['nlls_en_options']['ftol'],
                l1_ratio=gsc_result_dict['nlls_en_options']['l1_ratio'],
                debug=False,
                selection=gsc_result_dict['nlls_en_options']['selection'],
            )['params']
        )

    params_conformal = np.array(params_conformal)
    params_conformal = pd.DataFrame(params_conformal, columns=gsc_result_dict['param_names'])

    gsc_result_dict['conformal_inference_info'] = {
        'params_conformal': params_conformal,
        'conformal_stride': conformal_stride,
    }

    return params_conformal



def permutation_inference(gsc_result_dict, num_draws=50, debug=False, seed=0, shifts=None):
    rand = np.random.RandomState(seed)
    if shifts is None:
        shifts = rand.randint(gsc_result_dict['n_time'], size=num_draws)
    elif shifts == 0:
        shifts = [0] * num_draws
    else:
        raise Exception

    params_permute = []
    permutations = []

    for i in tqdm(range(num_draws), total=num_draws):
        perm = rand.permutation(np.arange(gsc_result_dict['n_units']))
        permutations.append(perm)

        resid_func_k, *_ = get_gsc_pred_func(
            data=gsc_result_dict['data'], outcome_col=gsc_result_dict['outcome_col'],
            unit_col=gsc_result_dict['unit_col'], time_col=gsc_result_dict['time_col'],
            treatment_cols=gsc_result_dict['treatment_cols'],
            covariate_cols=gsc_result_dict['covariate_cols'],
            weight_control_only=gsc_result_dict['weight_control_only'],
            log_transform=gsc_result_dict['log_transform'],
            debug=debug, pooled_treatment=gsc_result_dict['pooled_treatment'], shift=shifts[i],
            permute=perm,
        )

        params_permute.append(
            nlls_elastic_net_minimize_internal_coordinate_descent(
                lambda p: resid_func_k(p)[0].flatten(),
                num_params=gsc_result_dict['n_params'],
                x0=gsc_result_dict['params'],
                bounds=gsc_result_dict['nlls_en_options']['bounds'],
                alpha=gsc_result_dict['nlls_en_options']['alpha_vec'],
                max_iter=gsc_result_dict['nlls_en_options']['max_iter'],
                xtol=gsc_result_dict['nlls_en_options']['xtol'],
                ftol=gsc_result_dict['nlls_en_options']['ftol'],
                l1_ratio=gsc_result_dict['nlls_en_options']['l1_ratio'],
                debug=False,
                selection=gsc_result_dict['nlls_en_options']['selection'],
            )['params']
        )

    params_permute = np.array(params_permute)
    params_permute = pd.DataFrame(params_permute, columns=gsc_result_dict['param_names'])

    gsc_result_dict['permutation_inference_info'] = {
        'params_permute': params_permute,
        'num_draws': num_draws,
        'seed': seed,
        'shifts': shifts,
        'permutations': permutations,
    }

    return params_permute
