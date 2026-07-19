import numpy as np
import pandas as pd

from kanly.api import lm, compare_results

n = 100
np.random.seed(0)
df = pd.DataFrame({
    'x': np.random.randn(n),
    'grp': np.random.randint(0, 12, n),
})
df['y'] = 1.2 - 0.3 * df['x'] + .2 * np.random.randn(n)

fit1 = lm('y ~ x + C(grp)', df[df.x > 0], debug=False)
fit2 = lm('y ~ x + C(grp)', df, debug=False, index=np.arange(len(df))[df.x > 0])
fit3 = lm('y ~ x + C(grp)', df, debug=False, index=df.x > 0)

print(compare_results([fit1, fit2, fit3]))
