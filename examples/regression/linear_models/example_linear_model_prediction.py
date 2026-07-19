import pandas as pd
import numpy as np
from kanly.api import lm
import matplotlib.pyplot as plt

n = 200
np.random.seed(0)
df = pd.DataFrame({
    'x': np.random.randn(n),
    'w': np.random.randn(n),
})
df['y'] = 1.2 - 0.3 * df['x'] + .1 * df['x'] * df['w'] + .2 * np.random.randn(n)

fit = lm('y ~ x*w', df.iloc[:100])
print(fit)

plt.scatter(df['x'], fit.predict(df), label='y_hat extrapolated\nto full data', alpha=.5)
plt.scatter(df['x'].iloc[:100], fit.fittedvalues, label='y_hat from estimation\non half data', marker='x', alpha=.5)
plt.scatter(df['x'].iloc[:100], fit.predict(params=[1.2, -1, .02328, .09366]), label='y_hat setting "x" coef to -2',
            marker='_', alpha=.5)
plt.legend()
plt.show()
