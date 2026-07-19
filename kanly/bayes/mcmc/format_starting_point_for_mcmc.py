"""Normalize and validate MCMC initialization inputs."""

from __future__ import absolute_import, print_function

import numpy as np
import time


def format_starting_point_for_mcmc(x0, x0_is_original_scale, n_chains, transformations=None, fix_params=None,
                                   param_names=None, debug=False):
    """Prepare per-chain initial points plus transformed/fixed-parameter metadata.

    Args:
        x0: Initial point vector or matrix of initial points.
        x0_is_original_scale: Whether ``x0`` is on original bounded model scale.
        n_chains: Number of chains to initialize.
        transformations: Optional transform mapping keyed by name or index.
        fix_params: Optional fixed-parameter mapping keyed by name or index.
        param_names: Optional ordered parameter names.
        debug: If True, print normalization/validation timing.
    """
    __t = time.time()
    if debug:
        print("Converting starting point and transformations, fixed parameters etc to be consistent...", end='')

    if len(np.shape(x0)) > 1:
        num_params = len(x0[0])
    else:
        num_params = len(x0)

    x0 = np.array(x0, dtype=float, copy=True)
    x0 = x0.reshape((-1, num_params))

    # Broadcast a single starting vector to all chains.
    if np.shape(x0)[0] == 1:  # np.ndim(x0) == 1 or (np.ndim(x0) == 2 and np.shape(x0)[1] == 1):
        x0 = np.array(x0, dtype=float).flatten()
        x0s = [x0.copy() for _ in range(n_chains)]
    else:
        x0s = np.array(x0).copy()
        if len(x0s) != n_chains:
            raise Exception(f"If supplying multiple starting points, must supply as many as the number of chains. "
                            f"\nYou supplied {len(x0s)}, need to supply {n_chains}.")

    if param_names is None:
        param_names = [f'<x{d}>' for d in range(num_params)]

    if len(param_names) != len(x0s[0]):
        raise ValueError(
            f"\n{x0s}\n{x0}There are {len(param_names)} many parameters but starting point has length {len(x0s[0])}")

    if transformations is None:
        transformations = dict()
        transformation_function = lambda x: x
    else:
        transformations = _convert_dict_from_str_key_to_int_key(transformations, param_names)

        transformation_function = lambda x: np.array(
            [transformations.get(idx, lambda z: z)(x) for idx, x in enumerate(x)])

    fix_params_transformed = None
    if fix_params is not None:

        fix_params = _convert_dict_from_str_key_to_int_key(fix_params, param_names)

        for i, v in fix_params.items():
            for x0 in x0s:
                x0[i] = v
        fix_params_transformed = fix_params.copy()

    if transformations and x0_is_original_scale:
        for x0 in x0s:
            for i, f in transformations.items():
                x0[i] = f.inv_transform(x0[i])

        if fix_params is not None:
            fix_params_transformed.update(
                {i: f.inv_transform(fix_params_transformed[i]) for i, f in transformations.items()
                 if i in fix_params_transformed})

    if np.any(np.isnan(x0s)):
        raise Exception("Starting point has nan values.  Please check the supplied starting point, "
                        "as well as any transformations or fixed parameters.")

    if debug:
        print(f'done! ({time.time() - __t:.2f}s)\n')

    return x0s, transformations, fix_params, fix_params_transformed, param_names, num_params, transformation_function


def _convert_dict_from_str_key_to_int_key(dict_object, param_names):
    """Accept either integer indices or parameter-name keys and convert to integer indexing.

    Args:
        dict_object: Mapping keyed by parameter index and/or parameter name.
        param_names: Ordered parameter names used to resolve string keys.
    """
    v = {i: dict_object[i] for i in dict_object if isinstance(i, int) and i < len(param_names)}
    v.update({i: dict_object[k] for i, k in enumerate(param_names) if k in dict_object})
    return v
