from kanly.api import DataModel, lm
import pandas as pd
import numpy as np

n = 500
np.random.seed(1)
df = pd.DataFrame({'x': np.random.randn(n)})
df['y'] = 1.32 + 3.2 * df['x'] + np.random.randn(n) * .43

data_code_block = '''
self.y = `y`
self.x = `x`
self.x2 = self.x**2
'''

model_code_block = """
return (
    # logpdf_norm($b$, 1.7, 1.5) +
    logpdf_norm(y, $a$ + $b$ * x + $c$ * x2, $sigma2$**.5).sum()
)
"""

model = DataModel.build_data_model(
    data_code_block,
    model_code_block,
    df
).to_bayesian_model(
    bounds={'b': [.2, 100000], 'sigma2': [0, np.inf]},
    priors={'b': 'norm(1,2)'}
)

print(model)

fit = model.sample({'b': 1, 'sigma2': .3}, thinning=5,
                   n_burnin=5_000, n_samples=10_000,
                   do_mala_cd_warmup=True, )
#fit.amha(4000)
print(fit)

fit.diagnostic_plot('b', show=True)
