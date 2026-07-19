import pandas as pd

pd.set_option('display.max_rows', 60)

from kanly.bayes.bayesian_regression_model import BayesianNonlinearLeastSquaresModel
from scipy.stats import norm
import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal, assert_almost_equal

n = 150
np.random.seed(0)
df = pd.DataFrame({'x': np.random.randn(n)})
df['p'] = 1 / (1 + np.exp(-(.4 + .9 * df.x)))
df['y'] = (np.random.rand(n) < df['p']).astype(float)

alpha, sigma = 0, .2

for priors in [
    None,
    {'beta': lambda z: 0.0},
    {'beta': lambda z: norm.logpdf(z, loc=-.7, scale=.02)},
]:
    print('\n' * 5)
    print('#' * 100)
    print('PRIORS = ', priors)

    interval_low, interval_high = -.89, -.41
    for min_beta, max_beta in [(-np.inf, np.inf),
                               (-.9, -.4),
                               (-np.inf, -.4),
                               (-.9, np.inf)
                               ]:

        bounds = {'beta': [min_beta, max_beta]}
        print('\n\n\n' + '~' * 100)
        print(bounds)

        bmodel1 = BayesianNonlinearLeastSquaresModel.build_model_from_formula(
            '[y] ~ 1.0 / (1.0 + np.exp({alpha} + {beta} * [x]))',
            df,
            priors=priors,
            bounds=bounds, do_bounded_transform=False)

        bmodel2 = BayesianNonlinearLeastSquaresModel.build_model_from_formula(
            '[y] ~ 1.0 / (1.0 + np.exp({alpha} + {beta} * [x]))',
            df,
            priors=priors,
            bounds=bounds, do_bounded_transform=True)

        print(bmodel1.base_model.formula)
        print(bmodel2.base_model.formula)

        lp1 = bmodel1.log_posterior([alpha, (interval_low + interval_high) / 2, sigma])
        # lp2 = bmodel2.log_posterior(bmodel2.inv_transform([alpha, (interval_low + interval_high) / 2, sigma]))
        lp2 = bmodel2.log_posterior_transformed(bmodel2.inv_transform([alpha, (interval_low + interval_high) / 2, sigma]))

        print('\nlog posterior:')
        print('\tno transform: ', lp1)
        print('\ttransformed:  ', lp2)

        assert_almost_equal(lp1, lp2, decimal=4)

        print('\nexpectation')
        x_rng = np.linspace(interval_low, interval_high, 1500)
        x0 = [alpha, -.45, sigma]

        # original-scale lp from transformed model expected beta
        _sum = 0.0
        denom = 0.0
        func = lambda z: bmodel2.log_posterior_transformed(bmodel2.inv_transform([alpha, z, sigma]))
        for x in x_rng:
            _sum += x * np.exp(func(x))
            denom += np.exp(func(x))
        val1 = _sum/denom
        print('\toriginal scale LP on transfomed beta:  ', val1)

        # original-scale lp from untransformed model expected beta
        _sum = 0.0
        denom = 0.0
        func = lambda z: bmodel1.log_posterior([x0[0], z, x0[2]])
        for x in x_rng:
            _sum += x * np.exp(func(x))
            denom += np.exp(func(x))
        val2 = _sum / denom
        print('\toriginal scale:                        ', val2)

        # lp on transformed scale from transformed model
        tr = bmodel2.transformations['beta']
        x_rng2 = np.linspace(tr.inv_transform(interval_low), tr.inv_transform(interval_high), 1500)

        _sum = 0.0
        denom = 0.0

        x0_tr = bmodel2.inv_transform([alpha, (min_beta + max_beta) / 2, sigma])

        func = lambda z: bmodel2.log_posterior_transformed([x0_tr[0], z, x0_tr[2]]) + bmodel2.log_posterior_jacobian_adjustment(
            [x0_tr[0], z, x0_tr[2]])
        for x in x_rng2:
            _sum += tr.transform(x) * np.exp(func(x))
            denom += np.exp(func(x))
        val3 = _sum/denom

        print('\ttransformed scale with lp adjustment:  ', val3)

        assert_almost_equal(val1, val2, decimal=3)
        assert_almost_equal(val1, val3, decimal=3)
