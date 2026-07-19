import numpy as np
from kanly.api import nlls

import numpy as np
import pandas as pd
from kanly.automatic_differentiation.elementary_functions import expit


n = 250_000
np.random.seed(0)
df = pd.DataFrame({
    'x': np.random.randn(n),
    'g': np.random.randint(0,100,n)
})
df['p'] = expit(-1.2 + .5 * df.x)
df['y'] = (np.random.rand(n) < df['p']).astype(float)

fit = nlls('[y] ~ expit({a} + {b} * [x] + [C(g,-1)])',
           df,
           max_iter=100,
           debug=True,
           jac_method='mid',
           )

print(fit)


fit = nlls('[y] ~ expit({a} + {b} * [x] + [C(g,-1)])',
           df,
           max_iter=100,
           jac_method='analytic',
           do_analytic_jac_jit=False,
           debug=True,
           )

print(fit)