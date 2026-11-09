"""Helpers for mapping bounded parameter spaces to unconstrained space."""

from __future__ import absolute_import, print_function

import numpy as np
from abc import abstractmethod

from numba import njit  # don't delete!

from kanly.utils.logit_functions import expit, logit, log_d_expit
from kanly.dill_object import DillObject

LOWER_BOUND = 'LOWER_BOUND'
UPPER_BOUND = 'UPPER_BOUND'
INTERVAL = 'INTERVAL'
UNBOUNDED = 'UNBOUNDED'

assert 'njit' in locals()


class TransformedParameter(DillObject):
    """
    Variable transformations from bounded to unbounded domains.
    This gives us the inverse transforms from the unbounded space parameter 'y':

    --------------------------------------------------------------------------------------------
    Original           |  `transform`                       |   `inv_transform`
    Domain             |  (back to original)                |   (to unbounded)
    --------------------------------------------------------------------------------------------
    x in (a,    inf)   |   x = a + exp(y)                   |   y = log(x - a)
    x in (-inf, b  )   |   x = b - exp(y)                   |   y = log(b - x)
    x in (a,    b  )   |   x = a + (b-a) / (1+exp(-y))    |   y = log((x-a)/(b-x))
    --------------------------------------------------------------------------------------------
    [-inf < a < b < inf]

    `transform` converts back to original bounded parameter space
    `inv_transform` converts from original to unbounded space.

    Note that if f(x) is the proportional to the original density and y = g(x) is
    some change of variables, then the density "h" on "y" is

        h(y) = f( g^(-1) (y) ) * | {d/dy} g^(-1) (y) |

    where

        g^(-1) (.)                   ==  `transform`
        g(.)                         ==  `inv_transform`
        log |{d/dy} g^(-1) (.)|      ==  `lp_adjust`

    | {d/dy} g^(-1) (y) | is the so-called "Jacobian adjustment".
    """

    def __init__(self, name, lb, ub):
        """Initialize transformed-parameter helper.

        Args:
            name: Parameter name.
            lb: Lower bound.
            ub: Upper bound.
        """
        self.name = name
        self.lb = lb
        self.ub = ub
        assert lb < ub

        if lb == -np.inf and ub == np.inf:
            self.transform_str = f'{{{self.name}}}'
            self.inv_transform_str = f'{{{self.name}}}'
            self.transform = lambda z: z
            self.inv_transform = lambda z: z
            self.lp_adjust = lambda z: 0.0
            self.lp_adjust_str = '0.0'
            self.type = UNBOUNDED

        elif ub == np.inf:
            self.transform_str = f'({lb} + np.exp({{{self.name}}}))'
            self.inv_transform_str = f'np.log({{{self.name}}} - ({lb}))'
            self.transform = lambda z: lb + np.exp(z)
            self.inv_transform = lambda z: np.log(z - lb)
            self.lp_adjust = lambda z: z
            self.lp_adjust_str = f'{{{self.name}}}'
            self.type = LOWER_BOUND

        elif lb == -np.inf:
            self.transform_str = f'({ub} - np.exp({{{self.name}}}))'
            self.inv_transform_str = f'np.log({ub} - {{{self.name}}})'
            self.transform = lambda z: ub - np.exp(z)
            self.inv_transform = lambda z: np.log(ub - z)
            self.lp_adjust = lambda z: z
            self.lp_adjust_str = f'{{{self.name}}}'
            self.type = UPPER_BOUND

        else:
            self.transform_str = f'({lb} + ({ub - lb}) * expit({{{self.name}}}))'
            self.inv_transform_str = f'logit(({{{self.name}}} - ({lb}))/({ub - lb}))'
            self.transform = lambda z: lb + (ub - lb) * expit(z)
            self.inv_transform = lambda z: logit((z - lb) / (ub - lb))
            log_width = np.log(ub - lb)
            self.lp_adjust = lambda z: log_width - np.logaddexp(0.0, -z) - np.logaddexp(0.0, z)
            self.lp_adjust_str = (
                f'({log_width} - np.logaddexp(0.0, -{{{self.name}}})'
                f' - np.logaddexp(0.0, {{{self.name}}}))'
            )
            self.type = INTERVAL

    @staticmethod
    def get_identity_transform(name):
        """Return identity transform object for unconstrained parameter.

        Args:
            name: Parameter name.
        """
        return TransformedParameter(name, -np.inf, np.inf)

    def __call__(self, x):
        """Apply forward transform.

        Args:
            x: Input value/array.
        """
        return self.transform(x)

    def __str__(self):
        """Return readable transform expression.

        Args:
            None.
        """
        return f'{{{self.name}}}   -->   {self.transform_str}'

    def __repr__(self):
        """Return representation string.

        Args:
            None.
        """
        return str(self)


def bounds_2_transformations(bounds):
    """Build a ``name -> TransformedParameter`` mapping from bound tuples.

    Args:
        bounds: Mapping from parameter name to ``(lb, ub)``.
    """
    transformations = {c: TransformedParameter(c, *b) for c, b in bounds.items()}
    return transformations


def transform_nonlinear_formula(formula, bounds):
    """Replace ``{param}`` placeholders in a formula using parameter transforms.

    Args:
        formula: Formula/source string containing placeholders.
        bounds: Mapping from parameter name to bounds.
    """
    transformations = bounds_2_transformations(bounds)
    while '{ ' in formula:
        formula = formula.replace('{ ', '{')
    while ' }' in formula:
        formula = formula.replace(' }', '}')
    for k, v in transformations.items():
        formula = formula.replace(f'{{{k}}}', v.str_replace)

    return formula, transformations


def get_transformation_vector_functions(transformations, param_names, debug=False):
    """Create vectorized transform/inverse/Jacobian callables for an ordered parameter list.

    Args:
        transformations: Mapping from parameter name to transform object.
        param_names: Ordered parameter names.
        debug: If True, print generated function source.
    """
    if transformations is None or len(transformations) == 0:
        def _transform_null(x, inplace=False):
            """Identity forward transform for models with no parameter transformations.

            Args:
                x: Parameter vector.
                inplace: When ``False`` (default) returns a copy; when ``True``
                    returns ``x`` unchanged.

            Returns:
                ``x`` unchanged or a copy of it.
            """
            return x if inplace else np.array(x)

        def _invtransform_null(x, inplace=False):
            """Identity inverse transform for models with no parameter transformations.

            Args:
                x: Parameter vector.
                inplace: When ``False`` returns a copy; when ``True`` returns ``x``.

            Returns:
                ``x`` unchanged or a copy of it.
            """
            return x if inplace else np.array(x)

        def _lp_jacobian_adjustment_null(x):
            """Return zero Jacobian log-determinant (no reparameterization applied).

            Args:
                x: Parameter vector (unused).

            Returns:
                Scalar 0.0.
            """
            return 0.0

        return _transform_null, _invtransform_null, _lp_jacobian_adjustment_null, \
            '(No transform)', '(No transform)', '(No transform)'

    trans_func_str = '\n'.join([
        f'@njit',
        f'def _transform_base(params):',
        '\n'.join([f"    params[{j}] = {transformations[k].transform_str}"
                   for j, k in enumerate(param_names) if k in transformations]),
        f'    return params',
        '',
    ])

    inv_trans_func_str = '\n'.join([
        f'@njit',
        f'def _inv_transform_base(params):',
        '\n'.join([f"    params[{j}] = {transformations[k].inv_transform_str}"
                   for j, k in enumerate(param_names) if k in transformations
                   if transformations[k].type != UNBOUNDED
                   ]),
        f'    return params',
        '',
    ])

    lp_adj_func_str = '\n'.join([
        f'@njit',
        f'def _lp_jacobian_adjustment_base(params):',
        "    return " + " + ".join(
            p.lp_adjust_str for p in transformations.values()
        ),
        '',
    ])

    for i, k in enumerate(param_names):
        trans_func_str = trans_func_str.replace(f'{{{k}}}', f'params[{i}]')
        inv_trans_func_str = inv_trans_func_str.replace(f'{{{k}}}', f'params[{i}]')
        lp_adj_func_str = lp_adj_func_str.replace(f'{{{k}}}', f'params[{i}]')

    # print(">>>>>> ", transformations)
    # print("\n\n\n", inv_trans_func_str)

    if debug:
        print(trans_func_str)
        print(inv_trans_func_str)
        print(lp_adj_func_str)

    # Build numba-ready kernels from generated source strings.
    exect_dict = dict(globals())
    exec(trans_func_str,exect_dict)
    exec(inv_trans_func_str,exect_dict)
    exec(lp_adj_func_str,exect_dict)

    _tr_base = exect_dict['_transform_base']
    _inv_tr_base = exect_dict['_inv_transform_base']
    _lp_adj_base = exect_dict['_lp_jacobian_adjustment_base']

    def _transform(params, inplace=False):
        """Map an unbounded sampling vector to the original parameter space.

        Copies ``params`` (unless ``inplace=True``) and then calls the
        Numba-JIT compiled ``_transform_base`` kernel generated from
        ``trans_func_str``.

        Args:
            params: Parameter vector in the unbounded sampling space.
            inplace: When ``True``, requires a floating-point NumPy array and
                modifies it in place; otherwise works on a fresh floating copy.

        Returns:
            NumPy array in the original (possibly bounded) parameter space.
        """
        if inplace:
            if not isinstance(params, np.ndarray) or not np.issubdtype(params.dtype, np.floating):
                raise ValueError('inplace=True requires a floating-point NumPy array')
        else:
            params = np.array(params, dtype=float, copy=True)
        return _tr_base(params)

    def _inv_transform(params, inplace=False):
        """Map an original parameter vector to the unbounded sampling space.

        Copies ``params`` (unless ``inplace=True``) and then calls the
        Numba-JIT compiled ``_inv_transform_base`` kernel.

        Args:
            params: Parameter vector in the original (possibly bounded) space.
            inplace: When ``True``, requires a floating-point NumPy array and
                modifies it in place; otherwise works on a fresh floating copy.

        Returns:
            NumPy array in the unbounded sampling space.
        """
        if inplace:
            if not isinstance(params, np.ndarray) or not np.issubdtype(params.dtype, np.floating):
                raise ValueError('inplace=True requires a floating-point NumPy array')
        else:
            params = np.array(params, dtype=float, copy=True)
        return _inv_tr_base(params)

    def _lp_jacobian_adjustment(params):
        """Compute the log-determinant of the Jacobian of the forward transform.

        Sums the per-parameter log-absolute-derivative contributions from
        the Numba-JIT compiled ``_lp_jacobian_adjustment_base`` kernel.
        This value is added to the log-posterior to correct the density
        when sampling in the unbounded space.

        Args:
            params: Parameter vector in the unbounded sampling space.

        Returns:
            Scalar log-Jacobian adjustment (sum of log|d transform / d param|).
        """
        params = np.asarray(params)
        return _lp_adj_base(params)

    return _transform, _inv_transform, _lp_jacobian_adjustment, trans_func_str, inv_trans_func_str, lp_adj_func_str


def convert_samples_to_original_space(chain_results, transformations, num_params, debug=False, key='samples', window_start=None):
    """Summarize unbounded draws, then map stored draws to the original space.

    The returned population covariance and mean are always computed on the
    unbounded sampling scale, including models with no transformations.
    ``window_start`` excludes warm-up from those statistics; all stored draws
    are subsequently converted for presentation. Each result dictionary is
    updated in place with a floating-point sample array.

    Args:
        chain_results: List of per-chain result dictionaries.
        transformations: Optional mapping from index to forward transform callable.
        num_params: Number of parameters.
        debug: Debug mode flag.
        key: Entry in chain dict containing sample arrays.
        window_start: Optional first retained row for covariance/mean estimates.

    Returns:
        Covariance with shape ``(num_params, num_params)`` and mean with shape
        ``(num_params,)``. A single retained draw has zero population covariance.
    """
    transformations = {} if transformations is None else transformations
    start = 0 if window_start is None else window_start
    if isinstance(start, (bool, np.bool_)) or not isinstance(start, (int, np.integer)) or start < 0:
        raise ValueError('window_start must be a nonnegative integer or None')
    if debug:
        print("Computing variance-covariance on unbounded scale...", end="")

    sample_arrays = []
    count = 0
    mean = np.zeros(num_params, dtype=float)
    sum_squared_deviations = np.zeros((num_params, num_params), dtype=float)
    for chain in chain_results:
        samples = np.asarray(chain[key], dtype=float)
        if samples.ndim != 2 or samples.shape[1] != num_params:
            raise ValueError(f'{key} must have shape (n_draws, num_params)')
        if start > len(samples):
            raise ValueError('window_start exceeds the number of stored draws')
        sample_arrays.append(samples)
        retained = samples[start:]
        batch_count = len(retained)
        if not batch_count:
            continue
        batch_mean = retained.mean(axis=0)
        centered = retained - batch_mean
        delta = batch_mean - mean
        combined_count = count + batch_count
        sum_squared_deviations += centered.T @ centered
        sum_squared_deviations += np.outer(delta, delta) * (count * batch_count / combined_count)
        mean += delta * (batch_count / combined_count)
        count = combined_count
    if not count:
        raise ValueError('At least one retained draw is required to summarize samples')
    covariance = sum_squared_deviations / count

    if debug:
        print('done!')
        if transformations:
            print("Transforming results back to original parameter space...", end="")

    for chain, samples in zip(chain_results, sample_arrays):
        original_samples = samples.copy()
        for index, transform in transformations.items():
            original_samples[:, index] = transform(original_samples[:, index])
        chain[key] = original_samples

    if debug and transformations:
        print('done!')
    return covariance, mean


# Historical name pointed in the opposite direction. Keep existing callers
# working while the samplers use the explicit original-space name.
convert_samples_to_unbounded_space = convert_samples_to_original_space
