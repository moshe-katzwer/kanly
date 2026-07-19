import numpy as np
from kanly.bayes.bayesian_regression_model import BayesianLinearModel
import matplotlib.pyplot as plt


n = 33
np.random.seed(0)
x = np.random.randn(n)
y = 3 + 10 * x + np.random.randn(n) * 4.5
data = {'x': x, 'y': y, 'wts': np.random.rand(n) * 20}

bmodel = BayesianLinearModel.build_model_from_formula('y ~ x $ wts',
                                                      data, priors={'x': 'truncnorm(1, 4, 2, 12)'},
                                                      bounds={'x': [1, 12]})

print(bmodel)
fit = bmodel.amha([0, 15, 1], thinning=2,
                  n_samples=10_000, debug=False,
                  max_subchain_draws_sample=50_000)
print()
print(fit)

fit.diagnostic_plot('x')

plt.show()
