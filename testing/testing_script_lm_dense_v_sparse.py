import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix

from kanly.api import LM, lm, compare_results
from numpy.testing import assert_array_almost_equal

np.random.seed(0)

n = 200

w = np.random.rand(n)
grp = np.random.randint(0, 10, n)

for cov_type, cov_kwds in zip(
    ['cluster', 'nonrobust', 'ols_small', 'hc1', 'hac', 'hc2', 'hc3'],
    [{'groups': 'g'}, None, None, None, None, None, None]
):

    if cov_kwds is None:
        cov_kwds_REG = None
    else:
        cov_kwds_REG = {'groups': grp}

    for add_const in [False, True]:
        for instr_dim in [3, 1]:

            X = np.random.randn(n, instr_dim)
            Z = X + np.random.randn(n, instr_dim) * .05
            if instr_dim != 1:
                Z = np.hstack((X, np.random.randn(n, 1)))
            y = 3 + np.dot(X, range(X.shape[1])) + np.random.randn(n)

            df = pd.DataFrame(X, columns=[f'x{j}' for j in range(X.shape[1])])
            for j in range(Z.shape[1]):
                df[f'z{j}'] = Z[:, j]
            df['y'] = y
            df['w'] = w
            df['g'] = grp

            for iv in [False, True]:
                for weighted in [False, True]:
                    fits = dict()

                    if add_const:
                        _X = np.hstack([np.ones((n, 1)), X])
                        _Z = np.hstack([Z, np.ones((n, 1))])
                    else:
                        _X, _Z = X, Z

                    # by hand
                    W = np.diag(w ** 2) if weighted else np.eye(n)
                    if iv:
                        pi = np.linalg.inv(_Z.T.dot(W).dot(_Z)).dot(_Z.T.dot(W).dot(_X))
                        X_hat = _Z.dot(pi)
                    else:
                        X_hat = _X
                    beta = np.linalg.inv(X_hat.T.dot(W).dot(X_hat)).dot(X_hat.T.dot(W).dot(y))

                    for scale in [True, False]:

                        formula = (
                                f'y ~ {"+".join([f"x{k}" for k in range(X.shape[1])])}' + ('' if add_const else ' -1')
                                + (f' | {"+".join([f"z{k}" for k in range(Z.shape[1])])}' + ('' if add_const else ' -1') if iv else '')
                                + (f' $ I(w**2)' if weighted else '')
                        )

                        print(formula)

                        fits[f'kanly_scale={scale}'] = lm(
                            formula,
                            df,
                            cov_type=cov_type, cov_kwds=cov_kwds,
                            scale_design_matrix=scale, specification_name=f'kanly_scale={scale}')

                        for dense in [True, False]:

                            if dense:
                                __X, __Z = _X, _Z
                            else:
                                __X, __Z = csc_matrix(_X), csc_matrix(_Z)

                            fits[f'kanly_dense={dense}_scale={scale}'] = LM(
                                y, __X,
                                instruments=__Z if iv else None,
                                weights=w ** 2 if weighted else None,
                                exog_names=(['Intercept'] if add_const else []) + [f'x{k}' for k in range(X.shape[1])],
                                scale_design_matrix=scale,
                                specification_name=f'kanly_dense={dense}_scale={scale}',
                                has_constant=False,
                                cov_type=cov_type, cov_kwds=cov_kwds_REG
                            )

                    print(compare_results(
                        fits.values(),
                        suptitle={'cov_type': cov_type, 'const': add_const, 'wtd': weighted, 'iv': iv, 'instr_dim': instr_dim}))
                    fits_vec = list(fits.values())

                    for f in fits_vec:
                        assert_array_almost_equal(f.params, fits_vec[0].params)
                        assert_array_almost_equal(f.bse, fits_vec[0].bse)
                    assert_array_almost_equal(beta, fits_vec[0].params)
