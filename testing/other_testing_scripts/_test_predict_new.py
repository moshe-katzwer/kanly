import numpy as np
import pandas as pd
from kanly.api import glm, lm, nlls, elastic_net
import matplotlib.pyplot as plt

n = 20
np.random.seed(0)
df = pd.DataFrame()
df['x'] = np.random.rand(n)
df['y'] = 5.2 + 0.5 * np.exp(df.x) + .6 * np.random.randn(n)

integer_index = [5, 7, 3]

for f in [
        nlls('[y] ~ {b}*[np.exp(x)] + {a}', df),
        glm('y ~ x', df, family='poisson'),
        lm('y ~ np.exp(x)', df),
        elastic_net('y ~ x', df, alpha=.00001, l1_ratio=.99, fit_intercept=False, tol=1e-12, debug=True)]:
    plt.figure()
    plt.scatter(df.x, df.y, s=100, color='cyan', alpha=.4)
    plt.scatter(df.x, f.fittedvalues, s=100, color='b')
    plt.scatter(df.x, f.predict(), marker='x', color='r', s=50)
    plt.scatter(df.x, f.predict(data=df), marker='*', color='g', s=20)
    plt.scatter(df.x, f.predict(data=df, params=f.params), marker='s', color='y', s=10)
    plt.scatter(df.x.loc[integer_index], f.predict(data=df, params=f.params, index=integer_index),
                marker='s', color='k', s=10)
    plt.title(f.__class__.__name__)
    print(f)
    print(f.params)
    plt.show()

from sklearn.linear_model import ElasticNet

print(ElasticNet(alpha=.000001, fit_intercept=True).fit(np.exp(df[['x']]), df.y).__dict__)