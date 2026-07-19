import pandas as pd
from kanly.api import nlls, lm, glm, rlm, qr, elastic_net
import numpy as np

n = 100
np.random.seed(0)
df = pd.DataFrame({'y': np.random.randn(n)})

fit = lm('y ~ 1', df)
lm('y ~ 1', df, cov_type='bootstrap')
print(fit)

fit = glm('y ~ 1', df)
glm('y ~ 1', df, cov_type='bootstrap')
print(fit)

fit = qr('y ~ 1', df, .3)
qr('y ~ 1', df, .3, cov_type='bootstrap')
print(fit)

fit = rlm('y ~ 1', df)
rlm('y ~ 1', df, cov_type='bootstrap')
print(fit)

fit = nlls('[y] ~ {Intercept} + [0*y]', df)
nlls('[y] ~ {Intercept} + [0*y]', df, cov_type='bootstrap')
print(fit)
