import numpy as np
import pandas as pd
from kanly.api import lm, nlls, gmm, glm
from numpy.testing import assert_allclose

n = 1_500
np.random.seed(0)
z = 1 + np.random.randn(n)
x = [z[0]]
for i in range(1, n):
    x.append(.2 + .9 * x[i - 1] + z[i])
df = pd.DataFrame({'x': x})
df['g'] = np.repeat(range(100), n // 100)

fits = [lm('x ~ L(x,2,g)', df),
        glm('x ~ L(x,2,g)', df),
        nlls('[x] ~ {a} + {rho}*[L(x,2,g)]', df, Delta=1.4, gtol=1e-20),
        gmm(
            [
                '[x] - ({a} + {rho}*[L(x,2,g)])',
                ('[x] - ({a} + {rho}*[L(x,2,g)])', '[L(x,2,g)]')
            ],
            df, debug=False)
        ]

for f in fits:
    print("\n\n", f)
    assert_allclose(f.params, fits[0].params)
    print(f.model.valid_obs_rows)
