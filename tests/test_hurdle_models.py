"""Tests for GLM-composed Poisson and Gamma hurdle models."""

from contextlib import redirect_stderr, redirect_stdout
import io
import unittest

import numpy as np
import pandas as pd

from kanly.distributional_models import (
    DistributionalModelResults,
    GammaHurdle,
    NegativeBinomialPHurdle,
    PoissonHurdle,
    TwoPartModel,
)
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


def sample_zero_truncated_negative_binomial_p(rng, mean, alpha, p):
    """Draw NB-P counts conditional on positivity for ``p`` equal to 1 or 2."""
    size = mean ** (2 - p) / alpha
    probability = size / (size + mean)
    counts = rng.negative_binomial(size, probability)
    is_zero = counts == 0
    while np.any(is_zero):
        counts[is_zero] = rng.negative_binomial(
            size[is_zero], probability[is_zero]
        )
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

        self.assertIsInstance(fit, DistributionalModelResults)
        self.assertEqual(fit.get_result_type(), 'HURDLE')
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
        covariance = np.asarray(fit.cov_params())
        np.testing.assert_allclose(covariance[:2, :2], positive_cov)
        np.testing.assert_allclose(covariance[2:, 2:], hurdle_cov)
        np.testing.assert_array_equal(covariance[:2, 2:], 0.0)
        np.testing.assert_array_equal(covariance[2:, :2], 0.0)
        self.assertIn('PoissonHurdle Results', fit.summary())
        self.assertIn('block diagonal', fit.summary())

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

        self.assertIsInstance(fit, DistributionalModelResults)
        self.assertTrue(fit.converged)
        self.assertEqual(fit.cov_type, 'SANDWICH')
        self.assertEqual(fit.component_cov_type, 'NONROBUST')
        self.assertIsInstance(fit.positive_fit.family, Gamma)
        self.assertIsInstance(fit.positive_fit.link, Log)
        np.testing.assert_allclose(
            fit.params[:2], positive_params, atol=0.06
        )
        np.testing.assert_allclose(
            fit.params[2:], hurdle_params, atol=0.1
        )
        self.assertAlmostEqual(fit.scale, 1.0 / shape, delta=0.04)
        self.assertGreater(
            np.linalg.norm(np.asarray(fit.cov_params())[:2, 2:]), 0.0
        )
        self.assertTrue(fit.is_quasi_likelihood)
        self.assertIsNone(fit.aic)
        self.assertIsNone(fit.bic)
        self.assertIn('Quasi-Likelihood:', fit.summary())
        self.assertNotIn('AIC:', fit.summary())

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

    def test_negative_binomial_p_hurdle_recovers_nb1_and_nb2(self):
        """Estimate beta, dispersion, and logit coefficients for both NB-P forms."""
        positive_params = np.array([0.35, -0.25])
        hurdle_params = np.array([-0.5, 0.45])
        alpha = 0.7

        for p, seed in ((1, 931), (2, 932)):
            with self.subTest(p=p):
                rng = np.random.default_rng(seed)
                nobs = 3000
                _, _, exog, exog_infl = make_designs(rng, nobs)
                underlying_mean = np.exp(exog @ positive_params)
                endog = sample_zero_truncated_negative_binomial_p(
                    rng, underlying_mean, alpha, p
                )
                zero_probability = 1.0 / (
                    1.0 + np.exp(-(exog_infl @ hurdle_params))
                )
                endog[rng.random(nobs) < zero_probability] = 0

                model = NegativeBinomialPHurdle(
                    endog,
                    exog,
                    exog_infl=exog_infl,
                    p=p,
                    exog_names=['Intercept', 'x'],
                    exog_infl_names=['Intercept', 'z'],
                )
                fit = model.fit(cov_type='NONROBUST')

                self.assertIsInstance(fit, DistributionalModelResults)
                self.assertTrue(fit.converged)
                self.assertTrue(fit.inference_valid)
                self.assertEqual(fit.negative_binomial_p, p)
                self.assertEqual(
                    fit.param_names,
                    [
                        'Intercept', 'x', 'log_alpha',
                        'hurdle_Intercept', 'hurdle_z',
                    ],
                )
                np.testing.assert_allclose(
                    fit.params.iloc[:2], positive_params, atol=0.09
                )
                self.assertAlmostEqual(fit.dispersion, alpha, delta=0.12)
                np.testing.assert_allclose(
                    fit.params.iloc[3:], hurdle_params, atol=0.1
                )
                self.assertEqual(model.k_positive, 3)
                self.assertEqual(fit.score_obs().shape, (nobs, 5))
                self.assertLess(
                    np.max(np.abs(fit.score())) / nobs, 1e-5
                )

                covariance = np.asarray(fit.cov_params())
                np.testing.assert_array_equal(covariance[:3, 3:], 0.0)
                np.testing.assert_array_equal(covariance[3:, :3], 0.0)
                np.testing.assert_allclose(
                    fit.llf,
                    fit.positive_fit.llf + fit.hurdle_fit.llf,
                    rtol=1e-12,
                    atol=1e-10,
                )

                fitted_underlying = fit.predict(which='underlying_mean')
                fitted_positive = fit.predict(which='positive_mean')
                self.assertTrue(np.all(fitted_positive > fitted_underlying))
                np.testing.assert_allclose(
                    fit.fittedvalues,
                    fit.positive_probability * fitted_positive,
                )
                footer = fit.get_footer_info()
                self.assertIn(f'ZERO-TRUNCATED NB{p}', footer)
                self.assertIn('alpha=exp(log_alpha)', footer)

    def test_negative_binomial_p_hurdle_score_and_validation(self):
        """Match the weighted exact score and reject invalid p or outcomes."""
        rng = np.random.default_rng(133)
        nobs = 200
        _, _, exog, exog_infl = make_designs(rng, nobs)
        endog = sample_zero_truncated_negative_binomial_p(
            rng, np.exp(0.2 - 0.1 * exog[:, 1]), 0.5, 2
        )
        endog[:50] = 0
        weights = rng.uniform(0.3, 1.8, nobs)
        model = NegativeBinomialPHurdle(
            endog, exog, exog_infl=exog_infl, weights=weights
        )
        self.assertEqual(model.p, 2)

        params = np.array([0.2, -0.1, np.log(0.5), -0.8, 0.2])
        analytical = model.score(params)
        numerical = np.empty_like(params)
        step = 1e-5
        for column in range(len(params)):
            low = params.copy()
            high = params.copy()
            low[column] -= step
            high[column] += step
            numerical[column] = (
                model.loglike(high) - model.loglike(low)
            ) / (2.0 * step)
        np.testing.assert_allclose(
            analytical, numerical, rtol=2e-7, atol=2e-7
        )
        np.testing.assert_allclose(
            model.loglike(params),
            np.dot(weights, model.loglike_obs(params)),
        )

        with self.assertRaisesRegex(ValueError, 'p must be either 1'):
            NegativeBinomialPHurdle(endog, exog, p=3)
        with self.assertRaisesRegex(ValueError, 'non-negative integers'):
            NegativeBinomialPHurdle(
                np.array([0.0, 1.0, 1.5, 2.0]), exog[:4]
            )

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
        self.assertIsInstance(model, TwoPartModel)
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

        starts = model.get_start_params()
        weighted_zero_fraction = np.dot(
            weights, endog == 0.0
        ) / weights.sum()
        expected_hurdle_intercept = (
            np.log(weighted_zero_fraction)
            - np.log1p(-weighted_zero_fraction)
        )
        self.assertAlmostEqual(starts[2], expected_hurdle_intercept)
        self.assertEqual(starts[3], 0.0)

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

        debug_output = io.StringIO()
        with redirect_stdout(debug_output), redirect_stderr(debug_output):
            model = PoissonHurdle.build_model_from_formula(
                'y ~ x $ weight', data, exog_infl='z', debug=True
            )
        debug_text = debug_output.getvalue()
        self.assertIn('DISTRIBUTIONAL FORMULA MODEL', debug_text)
        self.assertIn('Zero-process exog shape:', debug_text)
        self.assertIn('Zero-process formula: z', debug_text)
        self.assertEqual(
            model.param_names,
            ['Intercept', 'x', 'hurdle_Intercept', 'hurdle_z'],
        )
        self.assertEqual(model.weights_name, 'weight')
        self.assertEqual(model.exog_infl_formula, 'z')
        fit = model.fit(cov_type='NONROBUST')
        self.assertTrue(fit.converged)
        prediction_data = pd.DataFrame(
            {'x': [-0.4, 0.8], 'z': [0.2, -0.5]}
        )
        numeric_exog = np.column_stack((
            np.ones(2), prediction_data['x']
        ))
        numeric_exog_infl = np.column_stack((
            np.ones(2), prediction_data['z']
        ))
        np.testing.assert_allclose(
            fit.predict(data=prediction_data),
            fit.predict(
                exog=numeric_exog, exog_infl=numeric_exog_infl
            ),
        )

        constant_model = PoissonHurdle.build_model_from_formula(
            'y ~ x', data
        )
        self.assertEqual(constant_model.param_names[-1], 'hurdle_const')
        np.testing.assert_array_equal(constant_model.exog_infl, 1.0)

        nb1_model = NegativeBinomialPHurdle.build_model_from_formula(
            'y ~ x $ weight', data, exog_infl='z', p=1
        )
        self.assertEqual(nb1_model.p, 1)
        self.assertEqual(
            nb1_model.param_names,
            [
                'Intercept', 'x', 'log_alpha',
                'hurdle_Intercept', 'hurdle_z',
            ],
        )

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
        with self.assertRaisesRegex(ValueError, 'finite and non-negative'):
            GammaHurdle(np.array([0.0, -1.0, 2.0, 3.0]), exog)

    def test_full_sandwich_and_coherent_hurdle_bootstrap(self):
        """Retain robust cross blocks and joint full-sample bootstrap draws."""
        rng = np.random.default_rng(812)
        nobs = 300
        _, _, exog, exog_infl = make_designs(rng, nobs)
        zero_probability = 1.0 / (
            1.0 + np.exp(-(-0.35 + 0.5 * exog_infl[:, 1]))
        )
        is_zero = rng.random(nobs) < zero_probability
        endog = sample_zero_truncated_poisson(
            rng, np.exp(0.2 + 0.25 * exog[:, 1])
        )
        endog[is_zero] = 0
        model = PoissonHurdle(endog, exog, exog_infl=exog_infl)

        sandwich = model.fit(cov_type='SANDWICH')
        covariance = np.asarray(sandwich.cov_params())
        self.assertTrue(sandwich.inference_valid)
        self.assertGreater(np.linalg.norm(covariance[:2, 2:]), 0.0)
        self.assertIsNotNone(sandwich.bread)
        self.assertIsNotNone(sandwich.meat)

        debug_output = io.StringIO()
        with redirect_stdout(debug_output), redirect_stderr(debug_output):
            bootstrap = model.fit(
                cov_type='BOOTSTRAP',
                cov_kwds={
                    'n_samples': 4,
                    'seed': 44,
                    'min_success_rate': 0.5,
                },
                debug=True,
            )
        debug_text = debug_output.getvalue()
        self.assertIn('HURDLE MODEL FIT', debug_text)
        self.assertIn('COHERENT HURDLE BAYESIAN BOOTSTRAP', debug_text)
        self.assertIn('Per-draw GLM output is suppressed', debug_text)
        self.assertIn('Final hurdle-fit diagnostics:', debug_text)
        self.assertTrue(bootstrap.inference_valid)
        self.assertEqual(bootstrap.bootstrapped_params.shape, (4, 4))
        np.testing.assert_array_equal(
            bootstrap.positive_bootstrapped_params,
            bootstrap.bootstrapped_params[:, :model.k_positive],
        )
        np.testing.assert_array_equal(
            bootstrap.hurdle_bootstrapped_params,
            bootstrap.bootstrapped_params[:, model.k_positive:],
        )
        self.assertTrue(bootstrap.cov_kwds['joint_draws'])
        np.testing.assert_allclose(
            np.asarray(bootstrap.cov_params()),
            np.cov(bootstrap.bootstrapped_params, rowvar=False, ddof=1),
        )


if __name__ == '__main__':
    unittest.main()
