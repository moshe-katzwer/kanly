import numpy as np
import pandas as pd
from scipy.sparse import isspmatrix
from kanly.api import lm, nlls

np.random.seed(0)
n = 25
df = pd.DataFrame({'x': np.random.randn(n)})
df['Lx'] = df.x.shift(1)
df['LLx'] = df.x.shift(2)
df['y'] = -4 + .8 * df.x + .15 * df.Lx + .3 * np.random.randn(n)

print(lm('y ~ x + Lx + LLx', df))
print(lm('y ~ x + Lag(x) + Lag(x, 2)', df))
print(nlls('[y] ~ {a} + {b0}*[x] + {b1}*[Lag(x)]', df))
