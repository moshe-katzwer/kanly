passed = []
failed = []

for script in [
    'testing_script_auto_diff_hessian',
    'testing_script_autodiff',
    'testing_script_autodiff_elementary_functions',
    'testing_script_autodiff_hessian_vector_function',
    'testing_script_autodiff_str_to_callable',
]:

    try:
        exec(f'import {script}')
        passed.append(script)
        print(f"{script} passed")
    except:
        failed.append(script)
        print(f"{script} FAILED!")

print(("\n" + "#" * 100) * 3)

print(f'PASSED = {passed}')
print(f'FAILED = {failed}')
