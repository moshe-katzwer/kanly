"""Tests for the shared distributional regression-results object."""

from contextlib import redirect_stderr, redirect_stdout
import io
import unittest
import warnings

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

    def test_debug_diagnostics_cover_formula_fit_and_bootstrap(self):
        """Propagate debug output through parsing, BFGS, and bootstrap."""
        rng = np.random.default_rng(810)
        x = rng.normal(size=100)
        endog = rng.poisson(np.exp(0.1 + 0.2 * x))
        output = io.StringIO()

        with redirect_stdout(output), redirect_stderr(output):
            model = Poisson.build_model_from_formula(
                'y ~ x', {'y': endog, 'x': x}, debug=True
            )
            fit = model.fit(
                cov_type='BOOTSTRAP',
                cov_kwds={
                    'n_samples': 3,
                    'seed': 7,
                    'min_success_rate': 0.5,
                },
                debug=True,
            )

        debug_output = output.getvalue()
        self.assertIn('DISTRIBUTIONAL FORMULA MODEL', debug_output)
        self.assertIn('DISTRIBUTIONAL MODEL FIT', debug_output)
        self.assertIn('BFGS OPTIONS', debug_output)
        self.assertIn('BAYESIAN BOOTSTRAP', debug_output)
        self.assertIn('Bootstrap diagnostics:', debug_output)
        self.assertIn('Final fit diagnostics:', debug_output)
        self.assertTrue(fit.converged)
        self.assertTrue(fit.inference_valid)

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
        prediction_data = pd.DataFrame({'x': [-0.5, 0.75]})
        np.testing.assert_allclose(
            fit.predict(data=prediction_data),
            np.exp(
                np.column_stack((np.ones(2), prediction_data['x']))
                @ np.asarray(fit.params)
            ),
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

    def test_all_direct_models_supply_default_start_params(self):
        """Fit every concrete direct model without explicit starting values."""
        rng = np.random.default_rng(987)
        nobs = 500
        x = rng.normal(size=nobs)
        exog = np.column_stack((np.ones(nobs), x))
        mean = np.exp(0.2 + 0.25 * x)
        alpha = 0.6

        nb1_size = mean / alpha
        nb1_endog = rng.negative_binomial(
            nb1_size, 1.0 / (1.0 + alpha)
        )
        nb2_size = 1.0 / alpha
        nb2_probability = nb2_size / (nb2_size + mean)
        nb2_endog = rng.negative_binomial(nb2_size, nb2_probability)
        zip_endog = rng.poisson(mean)
        zip_endog[rng.random(nobs) < 0.2] = 0
        zinb_endog = nb2_endog.copy()
        zinb_endog[rng.random(nobs) < 0.2] = 0

        models = [
            Poisson(rng.poisson(mean), exog),
            Gamma(rng.gamma(3.0, mean / 3.0), exog),
            GeneralizedPoisson(rng.poisson(mean), exog),
            NegativeBinomial1(nb1_endog, exog),
            NegativeBinomial2(nb2_endog, exog),
            ZeroInflatedPoisson(zip_endog, exog),
            ZeroInflatedNegativeBinomial(zinb_endog, exog),
        ]

        for model in models:
            with self.subTest(model=model.__class__.__name__):
                start_params = model.get_start_params()
                np.testing.assert_allclose(
                    model.default_start_params, start_params
                )
                self.assertEqual(len(start_params), len(model.param_names))
                self.assertTrue(np.all(np.isfinite(start_params)))

                fit = model.fit(cov_type='NONROBUST')
                self.assertIsInstance(fit, DistributionalModelResults)
                self.assertTrue(fit.converged)
                np.testing.assert_allclose(
                    fit.optimization_result.options['x0'], start_params
                )

        weighted_model = Poisson(
            np.array([0.0, 10.0]),
            np.ones((2, 1)),
            weights=np.array([9.0, 1.0]),
        )
        self.assertAlmostEqual(
            weighted_model.get_start_params()[0], np.log(1.0)
        )

    def test_explicit_start_params_override_defaults(self):
        """Retain user-provided starting values exactly."""
        rng = np.random.default_rng(779)
        x = rng.normal(size=300)
        exog = np.column_stack((np.ones(len(x)), x))
        endog = rng.poisson(np.exp(0.1 + 0.2 * x))
        model = Poisson(endog, exog)
        explicit_start = np.array([-0.2, 0.1])

        fit = model.fit(explicit_start, cov_type='NONROBUST')

        np.testing.assert_allclose(
            fit.optimization_result.options['x0'], explicit_start
        )

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
        np.testing.assert_allclose(
            np.asarray(fit.cov_params()),
            np.atleast_2d(np.cov(
                fit.bootstrapped_params, rowvar=False, ddof=1
            )),
        )
        self.assertIn('Bayesian bootstrap', fit.summary())
        self.assertIsInstance(fit.cov_params(), pd.DataFrame)

    def test_outcome_support_is_rejected_during_construction(self):
        """Reject non-finite or negative outcomes before likelihood work."""
        exog = np.ones((3, 1))
        count_models = [
            Poisson,
            GeneralizedPoisson,
            NegativeBinomial1,
            NegativeBinomial2,
            ZeroInflatedPoisson,
            ZeroInflatedNegativeBinomial,
        ]
        invalid_outcomes = [
            np.array([-0.1, 1.0, 2.0]),
            np.array([np.nan, 1.0, 2.0]),
            np.array([np.inf, 1.0, 2.0]),
        ]
        for model_class in count_models:
            for endog in invalid_outcomes:
                with self.subTest(
                        model=model_class.__name__, endog=endog[0]):
                    with self.assertRaisesRegex(
                            ValueError, 'finite and non-negative'):
                        model_class(endog, exog)

        for endog in (
                np.array([0.0, 1.0, 2.0]),
                np.array([-0.1, 1.0, 2.0]),
                np.array([np.nan, 1.0, 2.0])):
            with self.subTest(gamma_endog=endog[0]):
                with self.assertRaisesRegex(
                        ValueError, 'finite and strictly positive'):
                    Gamma(endog, exog)

        # Fractional non-negative count outcomes remain an intentional
        # likelihood-like extension rather than a normalized count PMF.
        fractional = np.array([0.0, 0.5, 1.5])
        for model_class in count_models:
            model = model_class(fractional, exog)
            self.assertTrue(model._is_quasi_likelihood())

    def test_common_data_validation_fails_immediately(self):
        """Reject invalid designs, empty samples, and zero total weight."""
        with self.assertRaisesRegex(ValueError, 'finite values'):
            Poisson(
                np.array([0.0, 1.0]),
                np.array([[1.0], [np.nan]]),
            )
        with self.assertRaisesRegex(ValueError, 'at least one row'):
            Poisson(np.array([]), np.empty((0, 1)))
        with self.assertRaisesRegex(ValueError, 'at least one column'):
            Poisson(np.array([0.0, 1.0]), np.empty((2, 0)))
        with self.assertRaisesRegex(ValueError, 'positive total'):
            Poisson(
                np.array([0.0, 1.0]),
                np.ones((2, 1)),
                weights=np.zeros(2),
            )

    def test_negative_binomial_poisson_limit_is_stable(self):
        """Recover finite Poisson limits for vanishing NB dispersion."""
        endog = np.array([0.0, 1.0, 2.0, 3.0])
        exog = np.column_stack((np.ones(4), np.linspace(-1.0, 1.0, 4)))
        beta = np.array([0.1, -0.2])
        poisson_llf = Poisson(endog, exog).loglike(beta)
        for model_class in (NegativeBinomial1, NegativeBinomial2):
            model = model_class(endog, exog)
            params = np.r_[beta, -1000.0]
            self.assertAlmostEqual(model.loglike(params), poisson_llf)
            self.assertTrue(np.all(np.isfinite(model.score(params))))
            self.assertEqual(model.score(params)[-1], 0.0)

        zinb = ZeroInflatedNegativeBinomial(endog, exog)
        params = np.r_[beta, -1.0, -1000.0]
        self.assertTrue(np.all(np.isfinite(
            zinb.predict(params, which='zero_probability')
        )))
        self.assertTrue(np.all(np.isfinite(zinb.score(params))))

    def test_invalid_optimum_and_singular_information_suppress_inference(self):
        """Do not attach standard errors to boundary or unidentified fits."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            gp = GeneralizedPoisson(
                np.full(40, 2.0), np.ones((40, 1)), p=1
            ).fit(cov_type='NONROBUST')
        self.assertTrue(gp.optimizer_converged)
        self.assertFalse(gp.converged)
        self.assertFalse(gp.first_order_valid)
        self.assertFalse(gp.inference_valid)
        self.assertFalse(gp.did_compute_var_covar())

        duplicated_exog = np.ones((8, 2))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            singular = Poisson(
                np.array([0.0, 1.0, 2.0, 1.0, 3.0, 0.0, 2.0, 1.0]),
                duplicated_exog,
            ).fit(np.zeros(2), cov_type='NONROBUST')
        self.assertTrue(singular.converged)
        self.assertFalse(singular.inference_valid)
        self.assertFalse(singular.did_compute_var_covar())
        self.assertEqual(singular.information_rank, 1)

    def test_zero_inflation_identification_is_checked(self):
        """Reject all-zero mixtures and flag no-zero boundary estimates."""
        exog = np.ones((6, 1))
        with self.assertRaisesRegex(ValueError, 'unidentified'):
            ZeroInflatedPoisson(np.zeros(6), exog)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model = ZeroInflatedPoisson(np.arange(1.0, 7.0), exog)
        self.assertTrue(any('boundary parameter' in str(w.message)
                            for w in caught))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            fit = model.fit(cov_type='NONROBUST')
        self.assertFalse(fit.inference_valid)
        self.assertFalse(fit.did_compute_var_covar())

    def test_fractional_and_importance_weighted_objectives_withhold_ic(self):
        """Do not present ordinary likelihood IC for pseudo objectives."""
        fractional = Poisson(
            np.array([0.25, 1.5, 2.25]), np.ones((3, 1))
        ).fit(cov_type='NONROBUST')
        self.assertTrue(fractional.is_quasi_likelihood)
        self.assertIsNone(fractional.aic)
        self.assertIsNone(fractional.bic)

        weighted = Poisson(
            np.array([0.0, 1.0, 2.0]),
            np.ones((3, 1)),
            weights=np.array([0.5, 1.0, 2.0]),
        ).fit(cov_type='SANDWICH')
        self.assertFalse(weighted.is_quasi_likelihood)
        self.assertIsNone(weighted.aic)
        self.assertIsNone(weighted.bic)
        score_obs = weighted.model.score_obs(np.asarray(weighted.params))
        weighted_scores = score_obs * weighted.model.weights[:, None]
        np.testing.assert_allclose(
            weighted.meat, weighted_scores.T @ weighted_scores
        )


if __name__ == '__main__':
    unittest.main()
