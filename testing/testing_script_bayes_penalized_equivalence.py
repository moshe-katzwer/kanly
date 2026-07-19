import numpy as np
from numpy.testing import assert_allclose

from kanly.api import elastic_net
from kanly.bayes.bayesian_regression_model import BayesianNonlinearLeastSquaresModel

n = 15_000
x = np.random.randn(n)
sigma = .5
y = 1.4 + 2 * x + sigma * np.random.randn(n)
data = {'x': x, 'y': y, 'const': np.ones(n)}

for prior, exponent, l1_ratio, sigma_xs in [
    ('norm', 2, 0, [.1, .01, .005, .001]),
    ('laplace', 1, 1, [.01, .0001, .00005]),
]:

    for sigma_x in sigma_xs:
        bmodel = BayesianNonlinearLeastSquaresModel.build_model_from_formula(
            '[y] ~ {const} + {x}*[x]',
            data,
            priors={'x': f'{prior}(0.0, {sigma_x})'},
        )

        bayes_fit = bmodel.amha([1.2, 1, 1])
        sigma_bayes = bayes_fit.mean_params['__sigma2'] ** .5

        fit_en = elastic_net('y ~ const + x', data,
                             alpha={'x': sigma_bayes ** 2 / sigma_x ** exponent / n},
                             l1_ratio={'x': l1_ratio},
                             normalize=False, fit_intercept=False)

        print('\n\n'+'-'*100)
        print(f'prior = {prior}, sigma_x = {sigma_x}')
        print('bayes: ', [bayes_fit['const'], bayes_fit['x']])
        print('en:    ', [fit_en['const'], fit_en['x']])
        assert_allclose(
            [fit_en['const'], fit_en['x']],
            [bayes_fit['const'], bayes_fit['x']],
            atol=1e-3, rtol=1
        )
        # print(bmodel.maximize_posterior(bayes_fit.params)['x'])
        # print(bayes_fit.map_params)
