import contextlib
import io
import unittest

import numpy as np
from scipy import stats

from kanly.stats.distributions.fit_distributions_mle import (
    DISTRIBUTIONS,
    get_mle_distribution,
    get_mle_x_y,
)


EXPECTED_DISTRIBUTIONS = {
    'norm', 'lognorm', 'expon', 't', 'gamma', 'weibull_min', 'beta',
    'logistic', 'laplace', 'cauchy', 'pareto', 'uniform',
}


class TestFitDistributionsMle(unittest.TestCase):

    def test_supported_distributions(self):
        self.assertEqual(set(DISTRIBUTIONS), EXPECTED_DISTRIBUTIONS)

    def test_normal_fit_is_mle_and_has_no_console_output(self):
        data = np.array([-2.0, 0.0, 1.0, 5.0])
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            fitted, params = get_mle_distribution(data, dist='NORM')

        self.assertAlmostEqual(params['loc'], data.mean())
        self.assertAlmostEqual(params['scale'], data.std(ddof=0))
        self.assertAlmostEqual(fitted.mean(), data.mean())
        self.assertEqual(stdout.getvalue(), '')

    def test_shifted_exponential_fit_includes_location_parameter(self):
        data = np.array([3.0, 3.5, 4.0, 7.5])

        _, params = get_mle_distribution(data, dist='expon')

        self.assertAlmostEqual(params['loc'], data.min())
        self.assertAlmostEqual(params['scale'], np.mean(data - data.min()))

    def test_added_distributions_fit_and_plot(self):
        cases = [
            ('lognorm', stats.lognorm, (0.6,), 2, {'floc': 0}),
            ('t', stats.t, (6.0,), 2, {}),
            ('gamma', stats.gamma, (2.5,), 2, {'floc': 0}),
            ('weibull_min', stats.weibull_min, (1.8,), 2, {'floc': 0}),
            ('beta', stats.beta, (2.0, 5.0), 1, {'floc': 0, 'fscale': 1}),
            ('logistic', stats.logistic, (), 2, {}),
            ('laplace', stats.laplace, (), 2, {}),
            ('cauchy', stats.cauchy, (), 2, {}),
            ('pareto', stats.pareto, (3.0,), 2, {'floc': 0}),
            ('uniform', stats.uniform, (), 2, {}),
        ]

        for dist, scipy_dist, args, sample_scale, fit_kwargs in cases:
            with self.subTest(dist=dist):
                rng = np.random.default_rng(5741)
                data = scipy_dist.rvs(
                    *args,
                    loc=0,
                    scale=sample_scale,
                    size=400,
                    random_state=rng,
                )

                x, y, fitted, params = get_mle_x_y(
                    data,
                    dist=dist,
                    curve='pdf',
                    num_points=51,
                    return_dist_obj=True,
                    fit_kwargs=fit_kwargs,
                )

                self.assertEqual(x.shape, (51,))
                self.assertEqual(y.shape, (51,))
                self.assertTrue(np.all(np.diff(x) > 0))
                self.assertTrue(np.all(np.isfinite(x)))
                self.assertTrue(np.all(np.isfinite(y)))
                self.assertGreater(params['scale'], 0)
                self.assertEqual(fitted.dist.name, dist)

    def test_invalid_samples_raise_clear_errors(self):
        cases = [
            ([], 'at least two observations'),
            ([1.0], 'at least two observations'),
            ([1.0, 1.0], 'at least two distinct values'),
            ([1.0, np.nan], 'only finite observations'),
            ([1.0, np.inf], 'only finite observations'),
            ([[1.0, 2.0]], 'one-dimensional'),
        ]

        for data, message in cases:
            with self.subTest(data=data):
                with self.assertRaisesRegex(ValueError, message):
                    get_mle_distribution(data)

    def test_invalid_options_raise_clear_errors(self):
        data = [1.0, 2.0, 3.0]

        with self.assertRaisesRegex(ValueError, 'dist must be one of'):
            get_mle_distribution(data, dist='not-a-distribution')
        with self.assertRaisesRegex(TypeError, 'fit_kwargs must be a mapping'):
            get_mle_distribution(data, fit_kwargs=['floc', 0])
        with self.assertRaisesRegex(ValueError, 'curve must be one of'):
            get_mle_x_y(data, curve='survival')
        with self.assertRaisesRegex(ValueError, 'num_points'):
            get_mle_x_y(data, num_points=1)
        with self.assertRaisesRegex(ValueError, 'quantiles must satisfy'):
            get_mle_x_y(data, left_quantile=0.9, right_quantile=0.1)
        with self.assertRaisesRegex(ValueError, 'quantiles must satisfy'):
            get_mle_x_y(data, left_quantile='low')


if __name__ == '__main__':
    unittest.main()
