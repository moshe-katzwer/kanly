from patsy import dmatrix
from scipy.interpolate import BSpline
from kanly.nonparametric.bspline import bspline_design_matrix
import numpy as np
from numpy.testing import assert_allclose

np.random.seed(0)
x = np.random.randn(100)

failed = []
for incl_intercept in [True, False]:
    for degree in range(4):
        for df in range(degree + 1, 20):
            try:
                ka = bspline_design_matrix(
                    x, degree=degree, n_bases=df, include_intercept=incl_intercept,
                    return_dense=True)[0]
                ka_sp = bspline_design_matrix(
                    x, degree=degree, n_bases=df, include_intercept=incl_intercept,
                    return_dense=False)[0].toarray()
                pa = np.array(dmatrix(f'bs(x,degree={degree},df={df},include_intercept={incl_intercept})-1'))
                assert_allclose(ka, ka_sp, rtol=1e-4, atol=1e-4)
                assert_allclose(ka, pa, rtol=1e-4, atol=1e-4)
            except Exception:
                failed.append(dict(degree=degree, n_bases=df, include_intercept=incl_intercept))

if len(failed):
    print("FAILED = ")
    print(failed)
else:
    print("ALL PASSED!")
