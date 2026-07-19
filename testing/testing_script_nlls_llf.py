import numpy as np
import pandas as pd

from kanly.api import nlls
from scipy.stats import norm
from numpy.testing import assert_almost_equal, assert_allclose
from scipy.stats import t as t_dist

"""
Log Likelihood

\log\mathcal{L}&=-\frac{n}{2}\log\left\{ 2\pi\sigma^{2}\right\} +\frac{1}{2}\sum_{i}\log\left\{ w_{i}\right\} -\frac{1}{2\sigma^{2}}\sum_{i}w_{i}r_{i}^{2}(\beta)
"""


n = 1_000
np.random.seed(0)
df = pd.DataFrame({'x': np.random.randn(n)})

print("Test log likelihood functions")

for t_df in [np.inf, 5]:
    for is_weighted in [False, True]:

        print('\tis_weighted = ', is_weighted, '\tt_df = ', t_df)

        df['p'] = 1 / (1 + np.exp(-(.4 + .9 * df.x)))
        df['w'] = 1 + np.random.rand(n)
        df['y'] = (np.random.rand(n) < df['p']).astype(float)

        fit = nlls('[y] ~ 1.0 / (1.0 + np.exp({alpha} + {beta} * [x]))' + (' $ [w] ' if is_weighted else ''),
                   df,
                   # cov_type='hc1',
                   debug=False,
                   jac_method='mid',
                   #cov_type='bootstrap',
                   specification_name='logistic regression')

        #print(fit)

        resid = fit.model.residual_function_callable(fit.params)

        scale = fit.scale_mle

        llf_func = fit.model.get_log_likelihood_function(t_df=t_df)
        llf_obs_func = fit.model.get_log_likelihood_function_obs(t_df=t_df)

        if t_df == np.inf:
            llf_obs_by_hand = norm.logpdf(resid, loc=0.0,
                                          scale=(scale / (df.w if is_weighted else 1.0)) ** .5)
        else:
            llf_obs_by_hand = t_dist.logpdf(resid, df=t_df, loc=0.0,
                                            scale=(scale / (df.w if is_weighted else 1.0)) ** .5)

        if t_df == np.inf:
            assert_almost_equal(fit.llf, llf_func(np.hstack([fit.params, scale])))
            assert_almost_equal(fit.llf, np.sum(llf_obs_by_hand))

        assert_almost_equal(np.sum(llf_obs_by_hand), llf_func(np.hstack([fit.params, [scale]])))
        print('\t\t llf')
        assert_allclose(llf_obs_by_hand, llf_obs_func(np.hstack([fit.params, [scale]])))
        print('\t\t llf obs level')

print('all passed')

print('\nTest analytical jacobian')

for transform_scale in [False, True]:
    for is_weighted in [False, True]:

        print('\tis_weighted = ', is_weighted, ' transform_scale = ', transform_scale)

        df['p'] = 1 / (1 + np.exp(-(.4 + .9 * df.x)))
        df['w'] = 1 + np.random.rand(n)
        df['y'] = (np.random.rand(n) < df['p']).astype(float)

        fit = nlls('[y] ~ 1.0 / (1.0 + np.exp({alpha} + {beta} * [x]))' + (' $ [w] ' if is_weighted else ''),
                   df,
                   # cov_type='hc1',
                   debug=False,
                   jac_method='mid',
                   #cov_type='bootstrap',
                   specification_name='logistic regression')

        scale = np.log(fit.scale_mle) if transform_scale else fit.scale_mle

        llf_func = fit.model.get_log_likelihood_function(transform_scale=transform_scale)
        jac_func = fit.model.get_log_likelihood_function_analytical_gradient(transform_scale=transform_scale)

        x0 = np.hstack([fit.params, scale]) * 1.1
        Jac = jac_func(x0)

        dx = 1e-6
        finite_diff = []
        for i in range(len(x0)):
            x0_h = x0.copy()
            x0_h[i] += dx
            x0_l = x0.copy()
            x0_l[i] -= dx
            df_i = (llf_func(x0_h) - llf_func(x0_l)) / (2 * dx)
            finite_diff.append(df_i)

        assert_allclose(Jac, finite_diff)

print('all passed')