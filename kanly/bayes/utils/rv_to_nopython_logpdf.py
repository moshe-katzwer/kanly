"""Convert scipy RV definitions to fast log-pdf callables used in priors."""

from __future__ import absolute_import, print_function

import warnings

# DO NOT DELETE ANY OF THESE #
from scipy.stats import norm, beta, cauchy, laplace, expon, t, gamma, invgamma, \
    lognorm, logistic, gennorm, chi2, multivariate_normal, halfnorm, truncnorm, \
    pareto, halfcauchy, multivariate_t, uniform, genextreme, loguniform
from scipy.stats._distn_infrastructure import rv_frozen
from scipy.stats._multivariate import multi_rv_frozen

# DO NOT DELETE ANY OF THESE #
import numpy as np
from kanly.stats.distributions.nopython_frozen_logpdf import rv_name_2_getter, flat


def convert_str_to_scipy_rv(rv):
    """
    Executes a string fragment into a scipy random variable, e.g.
    `conver_str_to_scipy_rv('beta(1, b=2)')`

    Args:
        rv: Distribution expression string (for example ``"norm(0, 1)"``).
    """
    assert isinstance(rv, str)
    try:
        split = rv.split('(')
        split[0] = ''.join([s for s in split[0] if s.isalnum()]).lower()

        # TODO DELETE
        # if split[0][-6:] == 'normal':
        #     split[0] = split[0].replace('normal', 'norm')
        # if split[0][-11:] == 'exponential':
        #     split[0] = split[0].replace('exponential', 'expon')

        rv = '('.join(split)

        rv = rv.replace('multivariatenormal', 'multivariate_normal')
        rv = rv.replace('multivariatet', 'multivariate_t')

        # Evaluate the sanitized scipy expression in module scope where rv names are imported.
        exec_dict = globals()
        exec(f'temp = {rv}', exec_dict)
        rv_obj = exec_dict['temp']

        return rv_obj

    except Exception as e:
        warnings.warn(f"Could not convert '{rv}' to a scipy random variable!")
        raise e


def get_nopython_logpdf(rv, nopython=False, logpdf_only=False):
    """Return a distribution-specific log-pdf callable (optionally numba-compatible).

    Args:
        rv: SciPy frozen RV instance or distribution expression string.
        nopython: Whether to request a nopython-friendly compiled implementation.
        logpdf_only: If True return only callable; else return ``(callable, rv)``.
    """
    if isinstance(rv, str):
        rv = convert_str_to_scipy_rv(rv)

    assert isinstance(rv, rv_frozen)
    name = rv.dist.name

    logpdf_gettr = rv_name_2_getter(name)
    logpdf = logpdf_gettr(*rv.args, **rv.kwds, nopython=nopython)

    if logpdf_only:
        return logpdf
    else:
        return logpdf, rv
