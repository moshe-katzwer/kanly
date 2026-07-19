from kanly.api import nlls, lm, nlls_en, elastic_net, compare_results
from kanly.bayes.bayesian_regression_model import BayesianNonlinearLeastSquaresModel
import pandas as pd
import numpy as np
from numpy.testing import assert_allclose

n = 1000
np.random.seed(0)
x = np.random.rand(n)
y = 1.2 * np.random.randn(n) + 3 * x - 2.5
df = pd.DataFrame({
    'x': x, 'y': y
})

for penalty in [1.4, 10]:

    for mean_x in [0, -4]:

        fit_ridge = lm('y ~ x', df, ridge_kwds={'alpha': {'x': penalty}, 'normalize': False},
                       specification_name='lm (ridge)')

        fit_en = elastic_net('y~x', df, alpha={'x': penalty},
                             l1_ratio=0,
                             regularize_to_values={'x': mean_x},
                             normalize=False,
                             specification_name='elastic_net',
                             xtol=1e-8, ftol=1e-8, gtol=1e-8,
                             max_iter=1000,
                             )

        fit_nlls = nlls('[y]~{Intercept}+{x}*[x]', df,
                        l2_penalties={'x': penalty},
                        regularize_to_values={'x': mean_x},
                        scale_l2_penalties=True,
                        jac_method='analytic',
                        specification_name='nlls',
                        xtol=1e-8, ftol=1e-8, gtol=1e-8
                        )

        fit_nlls_en = nlls_en('[y]~{Intercept}+{x}*[x]', df,
                              alpha={'x': penalty},
                              l1_ratio=0,
                              normalize=False,
                              regularize_to_values={'x': mean_x},
                              specification_name='nlls_en',
                              xtol=1e-8, ftol=1e-8, gtol=1e-8
                              )

        #####
        # Note, when mean_x != 0, the `lm` result should not match!!
        #####
        print(compare_results(
            [fit_ridge, fit_en, fit_nlls, fit_nlls_en]
        ))

        for f in [fit_nlls, fit_nlls_en] + ([fit_ridge] if mean_x == 0 else []):
            assert_allclose(f['x'], fit_en['x'], rtol=1e-4, atol=1e-4)
