passed = []
failed = []

for script in [
    'testing_script_inference',
    'testing_script_autodiff',
    'testing_script_autodiff_elementary_functions',
    'testing_script_nlls_llf',
    'testing_script_nlls',
    'testing_script_lm',
    'testing_script_glm',
    'testing_script_en',
    'testing_lags',
    'testing_script_prediction',
    'testing_script_lm_ridge',
    'testing_script_indexing',
    'testing_script_multioutcome',
    'testing_script_missing_data_indexing',
    'testing_script_two_way_clustering',
    'testing_lags_nlls',
    'testing_script_inference',
    'testing_joint_bootstrap',
    'testing_iv_gmm_linear',
    'testing_glm_iv',
    'testing_n_equals_p',
    'testing_demean',
]:

    try:
        exec(f'import {script}')
        passed.append(script)
        print(f"{script} passed")
    except:
        failed.append(script)
        print(f"{script} FAILED!")

print(("\n"+"#"*100)*3)

print(f'PASSED = {passed}')
print(f'FAILED = {failed}')
