from __future__ import absolute_import, print_function

from kanly.regression.linear_models.model import SparseLinearModel
from copy import deepcopy
from kanly.utils.util import none_copy


def fit_general_model_callable(formula, data, model_fit_callable, keep_model=True, drop_1_for_FE=True) -> dict:
    """Pass in a function that trains on y, X,
     where y and X are constructed according to a formula

    Parses the formula via kanly's sparse Patsy pipeline, then hands the
    resulting ``(y, X, weights)`` to your ``model_fit_callable`` — useful for
    plugging non-kanly estimators (sklearn, xgboost, etc.) into the same
    formula-driven design matrix construction kanly uses for OLS/IV/GLM.

    Args:
        formula (str): Patsy-style formula (``y ~ x + C(g)``, optionally
            ``$ w`` for weights).
        data (DataFrame or dict): Source data.
        model_fit_callable (callable): A function ``f(y, X, w) -> fit_obj``.
        keep_model (bool): Whether to keep references to the parsed model
            and arrays on the returned dict (vs. dropping them to free memory).
        drop_1_for_FE (bool): Whether to drop one dummy per categorical
            absorbed/encoded group.

    Returns:
        dict: ``{'fit': ..., 'model': ..., 'endog_name': ...,
        'exog_names': ..., 'weights_name': ...}``.

    Examples
    --------
    Use a sklearn ``RandomForestRegressor`` with kanly's formula machinery
    so categorical encodings and constant-column handling come for free:

    >>> import numpy as np, pandas as pd
    >>> from sklearn.ensemble import RandomForestRegressor          # doctest: +SKIP
    >>> from kanly.api import fit_general_model_callable             # doctest: +SKIP
    >>> rng = np.random.default_rng(0)
    >>> n = 1_000
    >>> df = pd.DataFrame({                                          # doctest: +SKIP
    ...     'x': rng.normal(size=n),
    ...     'g': rng.integers(0, 50, n),
    ...     'w': rng.uniform(0.5, 1.5, n),
    ... })
    >>> df['y'] = np.sin(df['x'] * 4) + 0.5 * rng.normal(size=n)    # doctest: +SKIP
    >>> rfr = RandomForestRegressor(max_depth=5, n_estimators=50,   # doctest: +SKIP
    ...                              random_state=0)
    >>> result = fit_general_model_callable(                        # doctest: +SKIP
    ...     'y ~ x + C(g) - 1 $ w', df,
    ...     lambda y, X, w: rfr.fit(X, y.toarray().ravel()),
    ...     drop_1_for_FE=False)
    >>> result['fit']                                                # doctest: +SKIP
    RandomForestRegressor(max_depth=5, n_estimators=50, random_state=0)
    """

    model = SparseLinearModel.build_model_from_formula(formula, data, drop_1_for_FE=drop_1_for_FE)

    endog_name, exog_names, weights_name \
        = deepcopy(model.endog_name), deepcopy(model.exog_names), none_copy(model.weights_name)

    if keep_model:
        y, X, w = model.endog, model.exog, model.weights
    else:
        y, X, w = model.endog.copy(), model.exog.copy(), none_copy(model.weights)
        model = None

    return {'fit': model_fit_callable(y, X, w),
            'model': model,
            'keep_model': keep_model,
            'formula': formula,
            'endog': y,
            'exog': X,
            'weights': w,
            'data': data,
            'model_fit_callable': model_fit_callable,
            'endog_name':endog_name,
            'exog_names': exog_names,
            'weights_name': weights_name,
            'drop_1_for_FE': drop_1_for_FE
            }

# if __name__ == '__main__':
#
#     import pandas as pd
#     from sklearn.ensemble import RandomForestRegressor
#     import numpy as np
#
#     n = 1000
#
#     df = pd.DataFrame()
#     df['x'] = np.random.randn(n)
#     df['y'] = np.sin(df.x * 4) + .5 * np.random.randn(n)
#     df['g'] = np.random.randint(0, 200, n)
#     df['w'] = np.random.rand(n)+1
#
#     rfr = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=3)
#     result = fit_general_model_callable('y~x+C(g)-1 $ w', df,
#                                         lambda y, X, w: rfr.fit(X, y.toarray().ravel()), drop_1_for_FE=False)
#     print(result['exog_names'])