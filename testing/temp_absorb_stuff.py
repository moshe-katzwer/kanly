from kanly.api import lm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

n = 25
np.random.seed(0)

df = pd.DataFrame()
df['x'] = np.random.randn(n)
df['w'] = np.random.randn(n)
df['g'] = np.random.randint(0, 3, n)
df['g2'] = np.random.randint(0, 2, n)
df['y'] = 5 - 2 * df.x + np.random.randn(n) + 1.5 * df.g

fit1 = lm('y~x*w', df, absorb=('g', 'g2'))
fit1 = lm('y~x*w', df, absorb=('g:g2'))
fit1

print(fit1)
print(fit1.model.build_model(df).fit())
