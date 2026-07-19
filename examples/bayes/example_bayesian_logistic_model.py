from kanly.api import bfgs_pqn, build_data_model, glm, compare_results, lm
import pandas as pd
import numpy as np

np.random.seed(0)
n = 60
x = np.random.randn(n)
pr = 1.0 / (1.0 + np.exp(-(.4 - .16 * x)))
y = (np.random.rand(n) < pr).astype(int)
df = pd.DataFrame({'x': x, 'y': y})

model = build_data_model(
    data_code_block='self.x = `x`; self.y = `y`;',
    # model_code_block='pr = 1.0 / (1.0 + np.exp(-($a$ - $b$ * x)));'
    #                  'return np.sum((y - pr)**2)',
    model_code_block='pr = 1.0 / (1.0 + np.exp(-($Intercept$ + $x$ * x)));'
                     'llf = y * np.log(pr) + (1-y) * np.log(1-pr);'
                     'return llf.sum()',
    data=df
)

fit = bfgs_pqn(model, [0,0], maximize=True)
#
# print(fit)

fit_glm = glm('y ~ x', df, family='binomial')

fit_bayes = model.to_bayesian_model().amha([0,0], thinning=5,
                                           n_chains=6,
                                           n_samples=10_000)

print(compare_results([fit_glm, fit_bayes]))

print(fit_bayes)