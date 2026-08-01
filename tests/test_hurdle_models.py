"""Tests for GLM-composed Poisson and Gamma hurdle models."""

import unittest

import numpy as np
import pandas as pd

from kanly.count_models.hurdle_models import GammaHurdle, PoissonHurdle
from kanly.regression.generalized_linear_models.families import (
    Bernoulli,
    Gamma,
    ZeroTruncatedPoisson,
)
from kanly.regression.generalized_linear_models.links import Log, Logit


def sample_zero_truncated_poisson(rng, rate):
    """Draw independent Poisson counts conditional on being positive."""
    counts = rng.poisson(rate)
    is_zero = counts == 0
    while np.any(is_zero):
        counts[is_zero] = rng.poisson(rate[is_zero])
        is_zero = counts == 0
    return counts


def make_designs(rng, nobs):
    """Create positive and hurdle designs with distinct regressors."""
    x = rng.normal(size=nobs)
    z = rng.normal(size=nobs)
    exog = np.column_stack((np.ones(nobs), x))
    exog_infl = np.column_stack((np.ones(nobs), z))
    return x, z, exog, exog_infl


class TestHurdleModels(unittest.TestCase):
    """Exercise component fitting, combination, formulas, and validation."""

    def test_poisson_hurdle_recovers_components_and_blocks_covariance(self):
        """Recover both regressions and preserve each native covariance block."""
        rng = np.random.default_rng(410)
        nobs = 4000
        _, _, exog, exog_infl = make_designs(rng, nobs)
        positive_params = np.array([0.2, 0.4])
        hurdle_params = np.array([-0.5, 0.7])

        zero_probability = 1.0 / (
            1.0 + np.exp(-(exog_infl @ hurdle_params))
        )
        is_zero = rng.random(nobs) < zero_probability
        endog = sample_zero_truncated_poisson(
            rng, np.exp(exog @ positive_params)
        )
        endog[is_zero] = 0

        model = PoissonHurdle(
            endog,
            exog,
            exog_infl=exog_infl,
            exog_names=['Intercept', 'x'],
            exog_infl_names=['Intercept', 'z'],
        )
        fit = model.fit(np.zeros(4), cov_type='NONROBUST')

        self.assertTrue(fit.converged)
        self.assertIsInstance(fit.positive_fit.family, ZeroTruncatedPoisson)
        self.assertIsInstance(fit.hurdle_fit.family, Bernoulli)
        self.assertIsInstance(fit.hurdle_fit.link, Logit)
        self.assertEqual(
            fit.param_names,
            ['Intercept', 'x', 'hurdle_Intercept', 'hurdle_z'],
        )
        np.testing.assert_allclose(
            fit.params[:2], positive_params, atol=0.08
        )
        np.testing.assert_allclose(
            fit.params[2:], hurdle_params, atol=0.1
        )

        positive_cov = np.asarray(fit.positive_fit.cov_params())
        hurdle_cov = np.asarray(fit.hurdle_fit.cov_params())
        np.testing.assert_allclose(fit.cov_params[:2, :2], positive_cov)
        np.testing.assert_allclose(fit.cov_params[2:, 2:], hurdle_cov)
        np.testing.assert_array_equal(fit.cov_params[:2, 2:], 0.0)
        np.testing.assert_array_equal(fit.cov_params[2:, :2], 0.0)

        np.testing.assert_allclose(
            fit.llf,
            fit.positive_fit.llf + fit.hurdle_fit.llf,
            rtol=1e-12,
            atol=1e-10,
        )
        self.assertLess(np.max(np.abs(fit.score())), 1e-6)

    def test_gamma_hurdle_uses_gamma_log_glm(self):
        """Fit positive Gamma outcomes through the existing Gamma GLM family."""
        rng = np.random.default_rng(918)
        nobs = 4000
        _, _, exog, exog_infl = make_designs(rng, nobs)
        positive_params = np.array([0.3, -0.35])
        hurdle_params = np.array([-0.25, 0.55])
        shape = 4.0

        zero_probability = 1.0 / (
            1.0 + np.exp(-(exog_infl @ hurdle_params))
        )
        is_zero = rng.random(nobs) < zero_probability
        positive_mean = np.exp(exog @ positive_params)
        endog = rng.gamma(shape, positive_mean / shape)
        endog[is_zero] = 0.0
        weights = rng.uniform(0.5, 1.5, size=nobs)

        model = GammaHurdle(
            endog,
            exog,
            exog_infl=exog_infl,
            weights=weights,
            exog_names=['Intercept', 'x'],
            exog_infl_names=['Intercept', 'z'],
        )
        fit = model.fit(np.zeros(4), cov_type='SANDWICH')

        self.assertTrue(fit.converged)
        self.assertEqual(fit.cov_type, 'SANDWICH')
        self.assertEqual(fit.component_cov_type, 'HC1')
        self.assertIsInstance(fit.positive_fit.family, Gamma)
        self.assertIsInstance(fit.positive_fit.link, Log)
        np.testing.assert_allclose(
            fit.params[:2], positive_params, atol=0.06
        )
        np.testing.assert_allclose(
            fit.params[2:], hurdle_params, atol=0.1
        )
        self.assertAlmostEqual(fit.scale, 1.0 / shape, delta=0.04)
        np.testing.assert_array_equal(fit.cov_params[:2, 2:], 0.0)

        expected_mean = (
            (1.0 - fit.zero_probability) * fit.positive_mean
        )
        np.testing.assert_allclose(fit.fittedvalues, expected_mean)
        np.testing.assert_allclose(
            fit.llf,
            np.dot(weights, fit.loglike_obs()),
            rtol=1e-12,
            atol=1e-10,
        )
        self.assertLess(np.max(np.abs(fit.score())), 1e-5)

    def test_weighted_loglike_and_score_are_consistent(self):
        """Match the analytical weighted score to likelihood differences."""
        rng = np.random.default_rng(77)
        nobs = 600
        _, _, exog, exog_infl = make_designs(rng, nobs)
        zero_probability = 1.0 / (
            1.0 + np.exp(-(-0.4 + 0.3 * exog_infl[:, 1]))
        )
        is_zero = rng.random(nobs) < zero_probability
        endog = sample_zero_truncated_poisson(
            rng, np.exp(0.1 + 0.25 * exog[:, 1])
        )
        endog[is_zero] = 0
        weights = rng.uniform(0.25, 2.0, size=nobs)

        model = PoissonHurdle(
            endog, exog, exog_infl=exog_infl, weights=weights
        )
        params = np.array([0.15, 0.2, -0.35, 0.25])
        score = model.score(params)
        numerical_score = np.empty_like(params)
        step = 1e-6
        for column in range(len(params)):
            low = params.copy()
            high = params.copy()
            low[column] -= step
            high[column] += step
            numerical_score[column] = (
                model.loglike(high) - model.loglike(low)
            ) / (2.0 * step)

        np.testing.assert_allclose(
            score, numerical_score, rtol=2e-6, atol=2e-5
        )
        np.testing.assert_allclose(
            model.loglike(params),
            np.dot(weights, model.loglike_obs(params)),
        )
        self.assertEqual(model.score_obs(params).shape, (nobs, 4))

    def test_formula_builder_names_alignment_and_default_hurdle(self):
        """Build explicit and constant hurdle designs through formula syntax."""
        rng = np.random.default_rng(22)
        nobs = 500
        x, z, _, _ = make_designs(rng, nobs)
        is_zero = rng.random(nobs) < 0.35
        endog = sample_zero_truncated_poisson(
            rng, np.exp(0.1 + 0.2 * x)
        )
        endog[is_zero] = 0
        weights = rng.uniform(0.5, 1.5, nobs)
        data = pd.DataFrame(
            {'y': endog, 'x': x, 'z': z, 'weight': weights}
        )

        model = PoissonHurdle.build_model_from_formula(
            'y ~ x $ weight', data, exog_infl='z'
        )
        self.assertEqual(
            model.param_names,
            ['Intercept', 'x', 'hurdle_Intercept', 'hurdle_z'],
        )
        self.assertEqual(model.weights_name, 'weight')
        self.assertEqual(model.exog_infl_formula, 'z')
        fit = model.fit(cov_type='NONROBUST')
        self.assertTrue(fit.converged)

        constant_model = PoissonHurdle.build_model_from_formula(
            'y ~ x', data
        )
        self.assertEqual(constant_model.param_names[-1], 'hurdle_const')
        np.testing.assert_array_equal(constant_model.exog_infl, 1.0)

        with self.assertRaises(Exception):
            PoissonHurdle.build_model_from_formula('y ~ x | z', data)

    def test_support_and_identification_validation(self):
        """Reject invalid counts and samples lacking one side of the hurdle."""
        exog = np.column_stack((np.ones(4), np.arange(4.0)))
        with self.assertRaisesRegex(ValueError, 'non-negative integers'):
            PoissonHurdle(np.array([0.0, 1.0, 1.5, 2.0]), exog)
        with self.assertRaisesRegex(ValueError, 'at least one zero'):
            GammaHurdle(np.array([1.0, 2.0, 3.0, 4.0]), exog)
        with self.assertRaisesRegex(ValueError, 'at least one zero'):
            GammaHurdle(np.zeros(4), exog)


if __name__ == '__main__':
    unittest.main()
