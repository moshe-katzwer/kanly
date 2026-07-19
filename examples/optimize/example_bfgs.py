from kanly.api import bfgs_pqn
from kanly.api import func_str_to_callable

from kanly.utils.dict_2_array import dict_2_array

func = func_str_to_callable(f'({{x}}-1)**2 + ({{y}}+2)**2')
x0 = dict_2_array({'x': 0, 'y': 2}, func.param_names)
result = bfgs_pqn(func, x0=x0, maxiter=10, maximize=False, ftol=1e-12, gtol=1e-12)

print(f'{func=}')
print('\n')
print(f'{result.x=}, {result.ferr=}, {result.gnorm=}')

"""
func={   'callable_function': <function __func__temp__ at 0x143ef6ee0>,
    'func_str': '(params[0]-1)**2 + (params[1]+2)**2',
    'nobs': 1,
    'num_params': 2,
    'other_args': '',
    'param_names': ['x', 'y'],
    'python_code_str': '\n'
                       'def __func__temp__(params, ): params = '
                       'np.asarray(params); return (params[0]-1)**2 + '
                       '(params[1]+2)**2'}
result.x=array([ 1.00000001, -2.00000006]), result.ferr=8.248613298197706e-13, result.gnorm=1.198790486853639e-07
"""