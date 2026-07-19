from kanly.api import func_str_to_callable

f = func_str_to_callable('[y] * {gamma} + {alpha} ** 2 * [x] * np.log([z]*{beta})', debug=True)

print(f)

part_deriv_callables, return_info = f.get_analytical_partial_derivatives(return_info=True)

for i, d in enumerate(return_info):
    print(f'{i:3d}  {f.param_names[i]:10s}{d["partial_derivative_expression"]}')

# Use $ as delimiter

f = func_str_to_callable('[y] * $gamma$ + $alpha$ ** 2 * [x] * np.log([z]*$beta$)',
                         param_delimiters=('$', '$'), debug=True)

print(f)

part_deriv_callables, return_info = f.get_analytical_partial_derivatives(return_info=True)

for i, d in enumerate(return_info):
    print(f'{i:3d}  {f.param_names[i]:10s}{d["partial_derivative_expression"]}')

