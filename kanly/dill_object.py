from __future__ import absolute_import, print_function

import time
from abc import ABC

import dill
from dill import load, dump


# dill.settings['recurse'] = True

def save(object_instance, filename, debug=False):
    """Serialise ``object_instance`` to ``filename`` using ``dill``.

    ``dill`` is a more permissive alternative to ``pickle`` that supports
    almost any Python object — including local functions, lambdas, fit
    results with attached models, and objects holding live JIT-compiled
    callables. Use it to persist any kanly fit or model object.

    Args:
        object_instance: Any Python object to serialise.
        filename (str or path-like): Output file path.
        debug (bool): If ``True``, print save time.

    Returns:
        bool: ``True`` on success.

    Examples
    --------
    Save an OLS fit to disk and load it back:

    >>> import numpy as np, pandas as pd
    >>> from kanly.api import lm, save, read
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({'x': rng.normal(size=100)})
    >>> df['y'] = 1.0 + 2.0 * df['x'] + rng.normal(size=100)
    >>> fit = lm('y ~ x', df)
    >>> save(fit, '/tmp/my_fit.dill')                       # doctest: +SKIP
    True
    >>> reloaded = read('/tmp/my_fit.dill')                 # doctest: +SKIP
    >>> reloaded.params['x'].round(2) == fit.params['x'].round(2)  # doctest: +SKIP
    True

    See Also
    --------
    :func:`read` : counterpart loader.
    """
    _t = time.time()
    if debug:
        print("Loading from file...", end="")
    with open(filename, "wb") as dill_file:
        dump(object_instance, dill_file)
    if debug:
        print(f'done! ({time.time() - _t:.2f}s)')
    return True


def read(filename, debug=False):
    """Load an object previously serialised by :func:`save`.

    Args:
        filename (str or path-like): Path to a dill-serialised file.
        debug (bool): If ``True``, print load time.

    Returns:
        The deserialised Python object.

    Examples
    --------
    Load a previously saved fit:

    >>> from kanly.api import read
    >>> fit = read('/tmp/my_fit.dill')                     # doctest: +SKIP
    >>> fit.params                                          # doctest: +SKIP

    See Also
    --------
    :func:`save` : counterpart writer.
    """
    _t = time.time()
    if debug:
        print(f"Loading {filename} from file...", end="")
    with open(filename, 'rb') as dill_file:
        obj = load(dill_file)
    if debug:
        print(f'done! ({time.time() - _t:.2f}s)')
    return obj


class DillObject(ABC):
    """Convenience class used for saving objects to text and loading from text"""

    def save(self, filename, debug=False):
        return save(self, filename, debug)
