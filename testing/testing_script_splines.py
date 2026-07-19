from kanly.api import build_data_model, lm, nlls, glm

import numpy as np

np.random.seed(0)
n = 1000

z = 4 * np.random.randn(n)
x = 4 * np.random.randn(n)
g1 = np.random.randint(0, 2, n)
g2 = np.random.randint(0, 8, n)

y = .3 * np.random.randn(n) + 4 * (z > 4)

data = {'y': y, 'g1': g1, 'g2': g2, 'z': z, 'x': x}


from statsmodels.formula.api import ols
print(ols('y ~ x + bs(z,df=3)', data).fit().summary())

print(lm('y ~ x + bs(z,df=3)', data))

print(glm('y ~ x + bs(z,df=3)', data))

print(nlls('[y] ~ {a} + {x}*[x] + [bs(z,df=3)]', data))

data_code = f'''
self.y = `y`
self.z = `z`
self.x = `x`
'''

model_code = '''
return logpdf_norm(y - $Intercept$ - $x$ * x - $_bs[z, df=3]$, 0, 1).sum()
'''

model = build_data_model(data_code, model_code, data).to_bayesian_model()
fit = model.amha(np.ones(5), n_burnin=1000, n_samples=5000)
print(fit)
