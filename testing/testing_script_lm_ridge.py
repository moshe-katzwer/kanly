import numpy as np
import pandas as pd

from kanly.api import lm, LM, nlls
from numpy.testing import assert_array_almost_equal
from kanly.api import elastic_net, compare_results

n = 100
np.random.seed(0)
df = pd.DataFrame({
    'x': np.random.randn(n) + 2,
    'z': 5 * np.random.randn(n) + 2,
    'grp': np.random.randint(0, 12, n),
    'w': 3 + np.exp(np.random.randn(n)),
})
df['y'] = 1.2 - 0.3 * df['x'] + .2 * np.random.randn(n)

for alpha in [0., .33, 2.0]:
    for normalize in [False, True]:
        for wtd in ['', ' $ w']:
            fit_en = elastic_net('y ~ x + z' + wtd, df, debug=False,
                                 alpha=alpha, normalize=normalize,
                                 l1_ratio=0, xtol=1e-10)

            fit_lm = lm('y ~ x + z' + wtd, df, debug=False,
                        ridge_kwds={'alpha': alpha, 'normalize': normalize},
                        inverse_method=np.linalg.pinv)

            fit_LM = LM(fit_lm.model.endog, fit_lm.model.exog, weights=fit_lm.model.weights,
                        ridge_kwds={'alpha': alpha, 'normalize': normalize}, exog_names=fit_lm.exog_names)

            print(">>>> ", wtd, ' $ [w]' if wtd == ' $ w' else '', alpha, normalize)
            if normalize:
                if wtd == '':
                    scales = {
                        'x': np.sum((df.x - np.average(df.x)) ** 2),
                        'z': np.sum((df.z - np.average(df.z)) ** 2),
                    }
                else:
                    scales = {
                        'x': np.sum((df.x - np.average(df.x, weights=df.w))**2),
                        'z': np.sum((df.z - np.average(df.z, weights=df.w))**2),
                    }
            else:
                scales = {'x': 1., 'z': 1.}

            nlls_fit = nlls('[y] ~ {Intercept} +{x}*[x]+{z}*[z]' + (' $ [w]' if wtd == ' $ w' else ''), df,
                            l2_penalties={'x': alpha/scales['x'], 'z': alpha/scales['z']},
                            scale_l2_penalties=True)

            print('\n' * 4 + '#' * 100)
            print(f'alpha = {np.round(alpha, 3)}, normalize = {normalize}, wts = {wtd}')
            print(compare_results([fit_en, fit_lm, fit_LM, nlls_fit], show_bse=False))
            assert_array_almost_equal(fit_en.params, fit_lm.params)
            assert_array_almost_equal(fit_en.params, fit_LM.params)
            if not normalize:
                assert_array_almost_equal(fit_en.params, nlls_fit.params)
