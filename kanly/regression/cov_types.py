"""Covariance-type string constants and keyword validation helpers.

Defines canonical string names for each supported variance-covariance
estimator (HC-robust, cluster-robust, HAC, bootstrap, and nonrobust OLS)
and provides helpers that normalise and validate the ``cov_kwds`` dict that
accompanies each estimator type.
"""

from __future__ import absolute_import, print_function

import re

# Cluster-robust sandwich (Liang-Zeger style).
CLUSTER = 'CLUSTER'

# Classical (homoskedastic) OLS covariance variants.
NONROBUST = 'NONROBUST'
OLS_SMALL = 'OLS_SMALL'   # finite-sample correction applied

# Heteroskedasticity-consistent (HC) sandwich variants.
HC0 = 'HC0'   # plain sandwich, no df correction
HC1 = 'HC1'   # n/(n-k) correction
HC2 = 'HC2'   # leveraged-adjusted
HC3 = 'HC3'   # jackknife-approximation (most conservative)

# Heteroskedasticity-and-autocorrelation-consistent (HAC/Newey-West).
HAC = 'HAC'
HAC_PANEL = 'HAC_PANEL'  # TODO: panel HAC not yet fully implemented

# Parametric bootstrap covariance.
BOOTSTRAP = 'BOOTSTRAP'

# Master tuple used for validation messages.
COV_TYPES = (CLUSTER, HAC, HC0, HC1, HC2, HC3, BOOTSTRAP, NONROBUST, OLS_SMALL, HAC_PANEL)


def format_cov_kwds(cov_kwds):
    """Normalise ``cov_kwds`` to a dict with lower-cased keys.

    Args:
        cov_kwds: Optional dict of keyword arguments for a covariance
            estimator, or ``None``.

    Returns:
        Dict with all keys converted to lower-case.  Returns an empty
        dict when ``cov_kwds`` is ``None``.
    """
    if cov_kwds is None:
        cov_kwds = dict()
    return {k.lower(): v for k, v in cov_kwds.items()}


def _test_acceptable_cov_kwds(cov_kwds, acceptable_cov_kwds):
    """Assert that every key in ``cov_kwds`` belongs to ``acceptable_cov_kwds``.

    Args:
        cov_kwds: Optional dict of keyword arguments to validate.
        acceptable_cov_kwds: Set of lower-cased key names that are allowed
            for the relevant covariance type.

    Raises:
        Exception: If any key in ``cov_kwds`` is not in ``acceptable_cov_kwds``.
    """
    if cov_kwds is None:
        cov_kwds = dict()
    keys = {x.lower() for x in cov_kwds.keys()}
    try:
        assert keys <= acceptable_cov_kwds
    except:
        raise Exception(f"Acceptable `cov_kwds` are {acceptable_cov_kwds},\n\tnot {set(cov_kwds.keys())}!")


def _parse_bootstrap(cov_type, cov_kwds):
    cov_type = cov_type.upper()
    if 'BOOTSTRAP' in cov_type and cov_type != 'BOOTSTRAP':
        pattern = r'^BOOTSTRAP\(-?\d+\)$'
        if not bool(re.match(pattern, cov_type)):
            raise Exception("bootstrap cov_type must be 'bootstrap' or 'bootstrap(n)' for some int n")
        n_samples = int(cov_type.split('(')[-1].replace(')', ''))
        if cov_kwds is None:
            cov_kwds = dict()
        cov_kwds['n_samples'] = n_samples
        cov_type = 'BOOTSTRAP'
    return cov_type, cov_kwds



def check_cov_kwds(cov_type, cov_kwds):
    """Validate that ``cov_kwds`` only contains keys supported by ``cov_type``.

    Each covariance estimator accepts a specific set of keyword arguments.
    This function looks up the allowed key set for the given ``cov_type``
    and raises if ``cov_kwds`` contains any unsupported key.

    Args:
        cov_type: String covariance type, matched case-insensitively against
            the constants defined in this module (e.g. ``'HC1'``, ``'CLUSTER'``).
        cov_kwds: Optional dict of keyword arguments accompanying the
            estimator.

    Raises:
        Exception: If ``cov_type`` is not one of the recognised types in
            ``COV_TYPES``, or if ``cov_kwds`` contains an unsupported key.
    """

    cov_type = cov_type.upper()

    # Check if bootstrap encodes n_samples
    if 'BOOTSTRAP' in cov_type:
        cov_type, cov_kwds = _parse_bootstrap(cov_type, cov_kwds)

    if cov_type is None or cov_type in [NONROBUST, OLS_SMALL, HC0, HC1, HC2, HC3]:
        acceptable = set()
    elif BOOTSTRAP in cov_type:
        acceptable = {'n_samples', 'method', 'seed', 'alpha', 'use_correction', 'max_processes', 'groups'}
    elif cov_type == CLUSTER:
        acceptable = {'groups', 'use_correction'}
    elif cov_type == HAC:
        acceptable = {'use_correction', 'kernel', 'maxlags', 'df_correction'}
    elif cov_type == HAC_PANEL:
        acceptable = {'use_correction', 'kernel', 'maxlags',  'df_correction', 'groups'}
    else:
        raise Exception(f"Acceptable `cov_type` are in {COV_TYPES}!")

    _test_acceptable_cov_kwds(cov_kwds, acceptable)
