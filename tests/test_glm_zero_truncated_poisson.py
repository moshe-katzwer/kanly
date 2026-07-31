"""Tests for the zero-truncated Poisson GLM family and canonical link."""

import unittest

import numpy as np
import pandas as pd
from scipy.special import gammaln

from kanly.api import GLM, glm
from kanly.regression.generalized_linear_models.families import (
    Gamma,
    NegativeBinomial,
    ZeroTruncatedPoisson,
    _get_family,
    _get_family_and_link,
)
from kanly.regression.generalized_linear_models.links import (
    ZeroTruncatedPoissonLink,
    _get_link,
    _ztp_mean_from_rate,
    _ztp_rate_from_mean,
)
from kanly.regression.generalized_linear_models.sparse_glm_internal import (
    _estimate_scale,
)


class TestZeroTruncatedPoisson(unittest.TestCase):
    """Exercise distribution algebra, registration, validation, and fitting."""

    def test_registry_and_default_link(self):
        """Resolve normalized family/link strings and select the default link."""
        family = _get_family('zerotruncatedpoisson')
        family2, link = _get_family_and_link(
            'zero_truncated_poisson', None
        )
        family_from_class = _get_family(ZeroTruncatedPoisson)
        family_instance = ZeroTruncatedPoisson()
        family_from_instance = _get_family(family_instance)
        explicit_link = _get_link('zerotruncatedpoissonlink')
        link_from_class = _get_link(ZeroTruncatedPoissonLink)
        link_instance = ZeroTruncatedPoissonLink()

        self.assertIsInstance(family, ZeroTruncatedPoisson)
        self.assertIsInstance(family2, ZeroTruncatedPoisson)
        self.assertIsInstance(family_from_class, ZeroTruncatedPoisson)
        self.assertIs(family_from_instance, family_instance)
        self.assertIsInstance(link, ZeroTruncatedPoissonLink)
        self.assertIsInstance(explicit_link, ZeroTruncatedPoissonLink)
        self.assertIsInstance(link_from_class, ZeroTruncatedPoissonLink)
        self.assertIs(_get_link(link_instance), link_instance)
        self.assertTrue(family.is_canonical(link))
        self.assertTrue(family.is_fixed_dispersion())
        with self.assertRaisesRegex(Exception, 'Unsafe link'):
            _get_family_and_link('zero_truncated_poisson', 'log')

    def test_link_round_trip_and_derivatives(self):
        """Check the canonical link and its analytical derivatives."""
        link = ZeroTruncatedPoissonLink()
        eta = np.array([-6.0, -2.0, 0.0, 1.5, 4.0])
        mu = link.inverse_link(eta)

        np.testing.assert_allclose(link.link(mu), eta, rtol=1e-7, atol=1e-7)
        self.assertTrue(np.all(mu > 1.0))

        h = 1e-5
        numerical_first = (
            link.inverse_link(eta + h) - link.inverse_link(eta - h)
        ) / (2.0 * h)
        numerical_second = (
            link.inverse_link(eta + h)
            - 2.0 * link.inverse_link(eta)
            + link.inverse_link(eta - h)
        ) / h ** 2
        np.testing.assert_allclose(
            link.deriv_inverse_link(eta), numerical_first,
            rtol=2e-5, atol=2e-7,
        )
        np.testing.assert_allclose(
            link.deriv2_inverse_link(eta), numerical_second,
            rtol=2e-3, atol=2e-5,
        )

    def test_rate_mean_round_trip_near_boundary(self):
        """Recover rates accurately even when the conditional mean nears one."""
        rates = np.array([1e-10, 1e-6, 1e-3, 0.1, 1.0, 10.0, 100.0])
        recovered = _ztp_rate_from_mean(_ztp_mean_from_rate(rates))
        np.testing.assert_allclose(
            recovered, rates, rtol=2e-6, atol=2e-12
        )

    def test_likelihood_matches_conditional_pmf(self):
        """Match the exponential-family implementation to the direct PMF."""
        family = ZeroTruncatedPoisson()
        endog = np.array([1.0, 2.0, 5.0])
        rate = np.array([0.2, 1.5, 4.0])
        theta = np.log(rate)

        direct = (
            -rate + endog * np.log(rate) - gammaln(endog + 1.0)
            - np.log1p(-np.exp(-rate))
        )
        np.testing.assert_allclose(
            family.log_likelihood_obs(endog, theta), direct
        )

        fitted_rate = np.array([0.5, 1.0, 3.0])
        fitted_theta = np.log(fitted_rate)
        fitted_mu = family.b_deriv(fitted_theta)
        fitted_ll = family.log_likelihood_obs(endog, fitted_theta)
        saturated_ll = np.zeros_like(endog)
        above_one = endog > 1.0
        saturated_theta = family.b_deriv_inv(endog[above_one])
        saturated_ll[above_one] = family.log_likelihood_obs(
            endog[above_one], saturated_theta
        )
        np.testing.assert_allclose(
            family.deviance(endog, fitted_mu),
            2.0 * np.sum(saturated_ll - fitted_ll),
        )

    def test_support_validation_and_fixed_scale(self):
        """Reject zero/fractional outcomes and retain unit dispersion."""
        family = ZeroTruncatedPoisson()
        valid = family.check_valid_range(
            np.array([-1.0, 0.0, 0.5, 1.0, 2.0, np.inf])
        )
        np.testing.assert_array_equal(
            valid, [False, False, False, True, True, False]
        )
        self.assertEqual(
            _estimate_scale(
                np.array([1.0, 2.0]), np.array([1.2, 1.8]),
                np.ones(2), family, df_resid=1,
            ),
            1.0,
        )
        self.assertEqual(
            _estimate_scale(
                np.array([1.0, 2.0]), np.array([1.2, 1.8]),
                np.ones(2), NegativeBinomial(alpha=0.5), df_resid=1,
            ),
            1.0,
        )

        exog = np.column_stack((np.ones(3), [-1.0, 0.0, 1.0]))
        for endog in (
                np.array([0.0, 1.0, 2.0]),
                np.array([0.5, 1.0, 2.0]),
                np.array([-1.0, 1.0, 2.0])):
            with self.subTest(endog=endog), self.assertRaisesRegex(
                    ValueError, 'ZERO_TRUNCATED_POISSON'):
                GLM(
                    endog, exog, family='zero_truncated_poisson',
                    fit_intercept=False,
                )

        with self.assertRaisesRegex(ValueError, 'GAMMA'):
            GLM(
                np.array([0.0, 1.0, 2.0]), exog,
                family=Gamma, fit_intercept=False,
            )

    def test_formula_fit_recovers_parameters(self):
        """Recover a simulated log-rate regression through the formula API."""
        rng = np.random.default_rng(731)
        nobs = 2500
        x = rng.normal(size=nobs)
        true_params = np.array([0.25, 0.45])
        rate = np.exp(true_params[0] + true_params[1] * x)

        endog = rng.poisson(rate)
        is_zero = endog == 0
        while np.any(is_zero):
            endog[is_zero] = rng.poisson(rate[is_zero])
            is_zero = endog == 0
        weights = rng.uniform(0.5, 1.5, size=nobs)

        fit = glm(
            'y ~ x $ w',
            pd.DataFrame({'y': endog, 'x': x, 'w': weights}),
            family='zero_truncated_poisson', cov_type='nonrobust',
        )

        self.assertTrue(fit.converged)
        self.assertEqual(fit.scale, 1.0)
        self.assertIsInstance(fit.family, ZeroTruncatedPoisson)
        self.assertIsInstance(fit.link, ZeroTruncatedPoissonLink)
        self.assertTrue(np.all(fit.endog_predicted > 1.0))
        np.testing.assert_allclose(
            np.asarray(fit.params), true_params, atol=0.1
        )


if __name__ == '__main__':
    unittest.main()
