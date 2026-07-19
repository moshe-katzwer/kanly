import numpy as np
import pandas as pd

from kanly.api import lm, compare_results, glm, elastic_net, qr, rlm, nlls, gmm
from numpy.testing import assert_array_almost_equal

n = 100
np.random.seed(0)
df = pd.DataFrame({
    'x': np.random.randn(n),
    'z': np.random.rand(n),
    'grp': np.random.randint(0, 12, n),
})
df['y'] = 1.2 - 0.3 * df['x'] + .2 * np.random.randn(n)

# GMM
for method, kwargs in [(gmm, dict())]:

    moments = ['[y] - {alpha} + {beta}*[x]',
               ('[y] - {alpha} + {beta}*[x]', '[x]'),
               ]
    fit1 = method(moments, df[df.x > 0], debug=False, **kwargs)
    fit2 = method(moments, df, debug=False, index=np.arange(len(df))[df.x > 0], **kwargs)
    fit3 = method(moments, df, debug=False, index=df.x > 0, **kwargs)

    print(compare_results([fit1, fit2, fit3]))
    for f in [fit1, fit2, fit3]:
        assert_array_almost_equal(fit1.params, f.params)

# LINEAR PREDICTOR
for method, kwargs in [(lm, dict()),
                       (glm, dict()),
                       (elastic_net, {'alpha': .001}),
                       (qr, {'tau': .8,
                             #'cov_type': 'bootstrap'
                             }),
                       (rlm, dict())]:

    fit1 = method('y ~ x + C(grp)', df[df.x > 0], debug=False, **kwargs)
    fit2 = method('y ~ x + C(grp)', df, debug=False, index=np.arange(len(df))[df.x > 0], **kwargs)
    fit3 = method('y ~ x + C(grp)', df, debug=False, index=df.x > 0, **kwargs)

    print(compare_results([fit1, fit2, fit3]))
    for f in [fit1, fit2, fit3]:
        assert_array_almost_equal(fit1.params, f.params)

# NLLS
for method, kwargs in [(nlls, dict())]:

    fit1 = method('[y] ~ [x] + [C(grp)] + [poly(z,2)]', df[df.x > 0], debug=False, **kwargs)
    fit2 = method('[y] ~ [x] + [C(grp)] + [poly(z,2)]', df, debug=False, index=np.arange(len(df))[df.x > 0], **kwargs)
    fit3 = method('[y] ~ [x] + [C(grp)] + [poly(z,2)]', df, debug=False, index=df.x > 0, **kwargs)

    print(compare_results([fit1, fit2, fit3]))
    for f in [fit1, fit2, fit3]:
        assert_array_almost_equal(fit1.params, f.params)
