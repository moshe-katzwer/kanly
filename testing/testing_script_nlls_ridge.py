
import pandas as pd
import numpy as np
from kanly.api import nlls, nlls_en, compare_results, lm, elastic_net
from numpy.testing import assert_array_almost_equal

np.random.seed(0)
n = 120
df = pd.DataFrame()
df['x'] = 1.5 + 2 * np.random.randn(n)
df['y'] = np.random.randn(n) + 1.2 * df['x']
df['wts'] = np.exp(np.random.randn(n))

for alpha in [.2, 1.5, 10]:

    for is_wtd in [False, True]:

        print(alpha, is_wtd)
        lm_suffix = ' $ wts' if is_wtd else ''
        nlls_suffix = ' $ [wts]' if is_wtd else ''

        wt_scale = sum(df.wts) if is_wtd else n

        g0 = lm('y ~ x' + lm_suffix, df)
        g1 = lm('y ~ x' + lm_suffix, df, ridge_kwds={'alpha': {'x': alpha}, 'normalize': False})
        h0 = elastic_net('y ~ x' + lm_suffix, df, alpha={'x': alpha}, l1_ratio=0.0, normalize=False, xtol=1e-8)

        a0 = nlls_en('[y] ~ {Intercept} + {x}*[x]' + nlls_suffix, df, alpha=0.0, l1_ratio=0.0, xtol=1e-8)
        a1 = nlls_en('[y] ~ {Intercept} + {x}*[x]' + nlls_suffix, df, alpha=0.0, l1_ratio=0.0, xtol=1e-8,
                     scale_penalties=False)
        a2 = nlls_en('[y] ~ {Intercept} + {x}*[x]' + nlls_suffix, df, alpha={'x': alpha}, l1_ratio=0.0, xtol=1e-8)
        a3 = nlls_en('[y] ~ {Intercept} + {x}*[x]' + nlls_suffix, df, alpha={'x': wt_scale * alpha}, l1_ratio=0.0,
                     xtol=1e-8, scale_penalties=False)

        f2 = nlls('[y] ~ {Intercept} + {x}*[x]' + nlls_suffix, df, xtol=1e-8)
        f4 = nlls('[y] ~ {Intercept} + {x}*[x]' + nlls_suffix, df, l2_penalties={'x': alpha}, xtol=1e-8)
        f5 = nlls('[y] ~ {Intercept} + {x}*[x]' + nlls_suffix, df, l2_penalties={'x': alpha}, xtol=1e-8, x_scale='jac')
        f6 = nlls('[y] ~ {Intercept} + {x}*[x]' + nlls_suffix, df, l2_penalties={'x': wt_scale * alpha}, xtol=1e-8,
                  x_scale='jac', scale_l2_penalties=False)

        print(compare_results([g0, g1, h0, a0, a1, a2, a3, f2, f4, f5, f6]))

        # Ridge params
        for fit in [g1, h0, a2, a3, f4, f5, f6]:
            assert_array_almost_equal(fit.params, g1.params, decimal=4)

        # Ridge std
        for fit in [g1, f4, f5, f6]:
            assert_array_almost_equal(fit.bse, g1.bse, decimal=4)
