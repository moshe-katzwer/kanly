"""Elementary scalar functions and symbolic derivative templates for the AD graph.

:mod:`kanly.automatic_differentiation.graph` looks up unary function names in
:data:`DERIV_FUNC_NAME_DICT`; each value is a callable ``z ->`` Python source for the
derivative of that elementary with respect to its argument ``z``.
"""
from __future__ import absolute_import, print_function

# don't delete any imports
import numpy as np
from numpy import sin, cos, tan, arcsin, arccos, arctan, cbrt, sqrt, log, log2, log10, exp
from numba import njit

for c in ['np', 'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
          'cbrt', 'sqrt', 'log', 'log2', 'log10', 'exp', 'njit']:
    if c not in locals():
        raise Exception


@njit(cache=True)
def expit(x):
    """Evaluate the logistic sigmoid ``1 / (1 + exp(-x))`` (Numba-compiled)."""
    return 1.0 / (1.0 + exp(-x))


@njit(cache=True)
def logit(x):
    """Evaluate the log-odds transform ``log(x / (1 - x))`` for ``x`` in ``(0, 1)`` (Numba)."""
    return log(x / (1.0 - x))


# Keys must match ``AutoDiffGraphNode.func_name`` after parsing (e.g. ``np.log``, ``sin``).
DERIV_FUNC_NAME_DICT = {

    # expit
    'expit': lambda z: f'(np.exp(-{z}) / (1 + np.exp(-{z}))**2)',
    'logit': lambda z: f'1.0/(({z})*(1-({z})))',

    # numpy
    'np.exp': lambda z: f'np.exp({z})',

    'np.log': lambda z: f'(1.0 / ({z}))',
    'np.log10': lambda z: f'(1.0 / (({z})*np.log(10)))',
    'np.log2': lambda z: f'(1.0 / (({z})*np.log(2)))',

    'np.sqrt': lambda z: f'(0.5 / np.sqrt({z}))',
    'np.cbrt': lambda z: f'((1.0/3.0) / ({z}) ** (2.0/3.0))',

    'np.sin': lambda z: f'np.cos({z})',
    'np.cos': lambda z: f'(-np.sin({z}))',
    'np.tan': lambda z: f'(1.0/(np.cos({z})**2))',

    'np.arcsin': lambda z: f'(1.0/np.sqrt(1-({z})**2))',
    'np.arccos': lambda z: f'(-1.0/np.sqrt(1-({z})**2))',
    'np.arctan': lambda z: f'(1.0/(1 + ({z})**2))',

    # no `np.` prefix
    'exp': lambda z: f'exp({z})',

    'log': lambda z: f'(1.0 / ({z}))',
    'log10': lambda z: f'(1.0 / (({z})*log(10)))',
    'log2': lambda z: f'(1.0 / (({z})*log(2)))',

    'sqrt': lambda z: f'(0.5 / sqrt({z}))',
    'cbrt': lambda z: f'((1.0/3.0) / ({z}) ** (2.0/3.0))',

    'sin': lambda z: f'cos({z})',
    'cos': lambda z: f'(-sin({z}))',
    'tan': lambda z: f'(1.0/(cos({z})**2))',

    'arcsin': lambda z: f'(1.0/sqrt(1-({z})**2))',
    'arccos': lambda z: f'(-1.0/sqrt(1-({z})**2))',
    'arctan': lambda z: f'(1.0/(1 + ({z})**2))',

}


def get_derivative_callable(func_name):
    """Return a Numba-JIT unary callable for the derivative of an elementary function.

    Builds source ``@jit ndef ___temp___(x): return <d/dx>`` using
    :data:`DERIV_FUNC_NAME_DICT`, then ``exec`` s it. ``jit`` must exist in globals
    (as when this module is imported under :mod:`numba`).

    Parameters
    ----------
    func_name : str
        Key present in :data:`DERIV_FUNC_NAME_DICT`, e.g. ``\"np.exp\"`` or ``\"sin\"``.

    Returns
    -------
    callable
        Scalar function ``x ->`` derivative of the named elementary at ``x``.
    """
    if func_name not in DERIV_FUNC_NAME_DICT:
        raise Exception(f'"{func_name}" not in {DERIV_FUNC_NAME_DICT.keys()}')
    func_str = (
        f'\n@jit'
        f'\ndef ___temp___(x): return {DERIV_FUNC_NAME_DICT[func_name]("x")}'
    )
    exect_dict = dict(globals())
    exec(func_str, exect_dict)
    func = exect_dict['___temp___']
    return func
