import numpy as np
import pandas as pd

from kanly.api import glm, lm, rlm, qr, elastic_net, gmm, nlls, sarimax

n = 5_000
np.random.seed(0)
df = pd.DataFrame({
     'x': np.random.randn(n),
     'z': np.random.randn(n),
     'e': np.random.randn(n),
})
df['y'] = np.exp(-1.5 + 2 * df.x + df.e)

for func in [lm, glm, qr, elastic_net, rlm]:
    kwargs = dict()
    if func == glm:
        kwargs = dict(family='poisson')
    elif func == qr:
        kwargs = dict(tau=.5)
    fit = func('y ~ x', df, **kwargs)
    print(fit)

fit = nlls('[y] ~ {a}+{b}*[x]', df)
print(fit)

gmm([
    '[y] - ({a}+{b}*[x])',
    ('[y] - ({a}+{b}*[x])', '[x]')
], df)
print(fit)

for func in [lm, glm, qr]:
    kwargs = dict()
    if func == glm:
        kwargs = dict(family='poisson')
    elif func == qr:
        kwargs = dict(tau=.5)
    fit = func('y ~ x | z', df, **kwargs)
    print(fit)


fit = sarimax('y ~ x', df, order=(2,0,0))
print(fit)

print("ALL RAN!")