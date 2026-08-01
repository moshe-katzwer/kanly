"""Tests for the shared distributional regression-results object."""

import unittest

import numpy as np
import pandas as pd

from kanly.distributional_models import (
    DistributionalModelResults,
    DistributionalModel,
    Gamma,
    GeneralizedPoisson,
    NegativeBinomial1,
    NegativeBinomial2,
    Poisson,
    ZeroInflatedNegativeBinomial,
    ZeroInflatedPoisson,
)


class TestDistributionalModelResults(unittest.TestCase):
    """Exercise direct-MLE result metadata, inference, and prediction."""

    def test_poisson_returns_full_regression_results(self):
        """Expose named inference, summaries, predictions, and raw optimizer."""
        rng = np.random.default_rng(130)
        nobs = 800
        x = rng.normal(size=nobs)
        mean = np.exp(0.25 + 0.35 * x)
        endog = rng.poisson(mean)
        data = pd.DataFrame({'y': endog, 'x': x})

        model = Poisson.build_model_from_formula('y ~ x', data)
        fit = model.fit(np.zeros(2), cov_type='NONROBUST')

        self.assertIsInstance(fit, DistributionalModelResults)
        self.assertIsInstance(fit.params, pd.Series)
        self.assertIsInstance(fit.cov_params(), pd.DataFrame)
        self.assertEqual(fit.get_result_type(), 'DISTRIBUTIONAL')
        self.assertIsInstance(model, DistributionalModel)
        self.assertEqual(fit.param_names, ['Intercept', 'x'])
        self.assertIsNotNone(fit.optimization_result)
        self.assertIs(fit.hess, fit.optimization_result.hess)
        self.assertEqual(fit.ferr, fit.optimization_result.ferr)
        self.assertIs(fit.model, model)
        self.assertTrue(fit.converged)
        np.testing.assert_allclose(fit.x, np.asarray(fit.params))
        np.testing.assert_allclose(
            fit.loglike(), model.loglike(np.asarray(fit.params))
        )
        np.testing.assert_allclose(
            fit.loglike(), fit.loglike_obs().sum()
        )
        np.testing.assert_allclose(fit.predict(), fit.fittedvalues)
        np.testing.assert_allclose(
            fit.fittedvalues,
            np.exp(model.exog @ np.asarray(fit.params)),
        )

        summary_df = fit.summary_df()
        self.assertIn('z', summary_df)
        self.assertIn('p>|z|', summary_df)
        self.assertNotIn('t', summary_df)
        summary = fit.summary()
        self.assertIn('Poisson Results', summary)
        self.assertIn('Log-Likelihood:', summary)
        self.assertIn('AIC:', summary)
        self.assertIn('BIC:', summary)
        self.assertIn('inverse observed information', summary)
        self.assertIn('asymptotic Normal inference', summary)
        self.assertIn('formula:', summary)

    def test_dispersion_models_report_transformed_alpha(self):
        """Report positive dispersion without conflating it with penalties."""
        rng = np.random.default_rng(215)
        nobs = 1000
        x = rng.normal(size=nobs)
        exog = np.column_stack((np.ones(nobs), x))

        gamma_mean = np.exp(0.2 - 0.25 * x)
        gamma_shape = 3.0
        gamma_endog = rng.gamma(
            gamma_shape, gamma_mean / gamma_shape
        )
        gamma_model = Gamma(
            gamma_endog,
            exog,
            exog_names=['Intercept', 'x'],
            endog_name='amount',
        )
        gamma_fit = gamma_model.fit(
            np.array([0.0, 0.0, -1.0]), cov_type='SANDWICH'
        )

        self.assertIsInstance(gamma_fit, DistributionalModelResults)
        self.assertAlmostEqual(
            gamma_fit.dispersion,
            np.exp(gamma_fit.params['log_alpha']),
        )
        self.assertIn('alpha = exp(log_alpha)', gamma_fit.summary())
        np.testing.assert_allclose(
            gamma_fit.predict(),
            np.exp(exog @ np.asarray(gamma_fit.params)[:2]),
        )

        alpha = 0.5
        nb_mean = np.exp(0.15 + 0.2 * x)
        size = 1.0 / alpha
        probability = size / (size + nb_mean)
        nb_endog = rng.negative_binomial(size, probability)
        nb_model = NegativeBinomial2(
            nb_endog,
            exog,
            exog_names=['Intercept', 'x'],
        )
        nb_fit = nb_model.fit(
            np.array([0.0, 0.0, np.log(alpha)]),
            cov_type='NONROBUST',
        )
        self.assertIsInstance(nb_fit, DistributionalModelResults)
        self.assertAlmostEqual(
            nb_fit.dispersion, np.exp(nb_fit.params['log_alpha'])
        )

    def test_zero_inflated_results_have_mixture_predictions(self):
        """Attach structural-zero and count-mean predictions to ZIP results."""
        rng = np.random.default_rng(330)
        nobs = 1200
        x = rng.normal(size=nobs)
        z = rng.normal(size=nobs)
        exog = np.column_stack((np.ones(nobs), x))
        exog_infl = np.column_stack((np.ones(nobs), z))
        count_mean = np.exp(0.15 + 0.3 * x)
        inflation_probability = 1.0 / (
            1.0 + np.exp(-(-0.8 + 0.5 * z))
        )
        endog = rng.poisson(count_mean)
        endog[rng.random(nobs) < inflation_probability] = 0

        model = ZeroInflatedPoisson(
            endog,
            exog,
            exog_infl=exog_infl,
            exog_names=['Intercept', 'x'],
            exog_infl_names=['Intercept', 'z'],
        )
        fit = model.fit(
            np.array([0.0, 0.0, -0.5, 0.0]),
            cov_type='NONROBUST',
        )

        self.assertIsInstance(fit, DistributionalModelResults)
        self.assertTrue(fit.is_zero_inflated)
        self.assertEqual(len(fit.inflation_probability), nobs)
        self.assertEqual(len(fit.count_mean), nobs)
        np.testing.assert_allclose(
            fit.fittedvalues,
            (1.0 - fit.inflation_probability) * fit.count_mean,
        )
        np.testing.assert_allclose(
            fit.predict(which='inflation_probability'),
            fit.inflation_probability,
        )
        self.assertIn('structural-zero probability', fit.summary())

    def test_remaining_count_subclasses_return_shared_results(self):
        """Cover GP, NB-1, and zero-inflated NB through the common fit path."""
        rng = np.random.default_rng(441)
        nobs = 500
        x = rng.normal(size=nobs)
        exog = np.column_stack((np.ones(nobs), x))
        mean = np.exp(0.15 + 0.25 * x)
        alpha = 0.6

        nb1_size = mean / alpha
        nb1_endog = rng.negative_binomial(nb1_size, 1.0 / (1.0 + alpha))
        nb2_size = 1.0 / alpha
        nb2_probability = nb2_size / (nb2_size + mean)
        zinb_endog = rng.negative_binomial(nb2_size, nb2_probability)
        zinb_endog[rng.random(nobs) < 0.2] = 0

        cases = [
            (
                GeneralizedPoisson(rng.poisson(mean), exog),
                np.array([0.0, 0.0, 0.05]),
            ),
            (
                NegativeBinomial1(nb1_endog, exog),
                np.array([0.0, 0.0, np.log(alpha)]),
            ),
            (
                ZeroInflatedNegativeBinomial(zinb_endog, exog),
                np.array([0.0, 0.0, -1.0, np.log(alpha)]),
            ),
        ]
        for model, start_params in cases:
            with self.subTest(model=model.__class__.__name__):
                fit = model.fit(start_params, cov_type='NONROBUST')
                self.assertIsInstance(fit, DistributionalModelResults)
                self.assertEqual(fit.cov_params().shape, (
                    len(fit.params), len(fit.params)
                ))
                self.assertTrue(np.isfinite(fit.llf))
                self.assertIn('z', fit.summary_df())

    def test_bootstrap_results_retain_draws_and_summary_metadata(self):
        """Expose bootstrap draws through DistributionalModelResults."""
        rng = np.random.default_rng(492)
        nobs = 250
        x = rng.normal(size=nobs)
        exog = np.column_stack((np.ones(nobs), x))
        endog = rng.poisson(np.exp(0.1 + 0.2 * x))
        model = Poisson(endog, exog)

        fit = model.fit(
            np.zeros(2),
            cov_type='BOOTSTRAP',
            cov_kwds={'n_samples': 8, 'seed': 9},
        )

        self.assertIsInstance(fit, DistributionalModelResults)
        self.assertEqual(fit.bootstrapped_params.shape, (8, 2))
        self.assertEqual(fit.cov_kwds['n_successful'], 8)
        self.assertIn('Bayesian bootstrap', fit.summary())
        self.assertIsInstance(fit.cov_params(), pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
