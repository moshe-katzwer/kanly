from __future__ import absolute_import, print_function

import sys
import traceback
import warnings

__version__ = '0.0.1051'

top_bar = '┏' + '━' * (31 + len(__version__) - 2) + '┓'
bottom_bar = '┗' + '━' * (31 + len(__version__) - 2) + '┛'

print("\n" + top_bar)
print(f"┃ Your version of `kanly` is {__version__} ┃")
print(bottom_bar)
print()


# def loaded_in_notebook():
#     try:
#         try:
#             from IPython import get_ipython
#         except ImportError as e:
#             raise e
#         shell = get_ipython().__class__.__name__
#         if shell == "ZMQInteractiveShell":
#             return True  # Jupyter notebook or qtconsole
#         elif shell == 'TerminalInteractiveShell':
#             return False  # Terminal running IPython
#         else:
#             return False  # some other type (???)
#
#     except NameError:
#         return False  # Probably a standard Python interpreter?

_IN_NOTEBOOK: bool | None = None

def loaded_in_notebook() -> bool:
    global _IN_NOTEBOOK
    if _IN_NOTEBOOK is None:
        if 'IPython' not in sys.modules:
            return False
        try:
            from IPython import get_ipython
            shell = get_ipython()
            _IN_NOTEBOOK = shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
        except ImportError:
            _IN_NOTEBOOK = False
    return _IN_NOTEBOOK

_IN_NOTEBOOK = loaded_in_notebook()


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, 'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


warnings.showwarning = warn_with_traceback

# TODO hacky fix -- need to close tqdm bars explicitly
import tqdm
tqdm.std.tqdm.__del__ = lambda self: None

# TODO silence RAY error
import os
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
