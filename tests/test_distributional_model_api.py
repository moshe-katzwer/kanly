"""Tests for one-shot formula and array distributional-model APIs."""

from contextlib import redirect_stderr, redirect_stdout
import io
import unittest
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit

from kanly.api import (
    DISTRIBUTIONAL_MODEL as ROOT_DISTRIBUTIONAL_MODEL,
    distributional_model as root_distributional_model,
)
from kanly.distributional_models import (
    DISTRIBUTIONAL_MODEL,
    DISTRIBUTIONAL_MODEL_ALIASES,
    DistributionalModelResults,
    distributional_model,
)
from kanly.distributional_models.continuous_models import Gamma
from kanly.distributional_models.count_models import (
    GeneralizedPoisson,
    NegativeBinomial1,
    NegativeBinomial2,
    Poisson,
    ZeroInflatedNegativeBinomial,
    ZeroInflatedPoisson,
)
from kanly.distributional_models.hurdle_models import (
    GammaHurdle,
    NegativeBinomialPHurdle,
    PoissonHurdle,
)
from kanly.regression.generalized_linear_models.links import CLogLog


def _sample_zero_truncated_poisson(rng, mean):
    """Draw independent Poisson counts conditional on positivity."""
    values = rng.poisson(mean)
    is_zero = values == 0
    while np.any(is_zero):
        values[is_zero] = rng.poisson(mean[is_zero])
        is_zero = values == 0
    return values


def _sample_zero_truncated_nb2(rng, mean, alpha):
    """Draw independent NB2 counts conditional on positivity."""
    size = 1.0 / alpha
    probability = size / (size + mean)
    values = rng.negative_binomial(size, probability)
    is_zero = values == 0
    while np.any(is_zero):
        values[is_zero] = rng.negative_binomial(
            size, probability[is_zero]
        )
        is_zero = values == 0
    return values


class TestDistributionalModelAPI(unittest.TestCase):
    """Exercise dispatch, aliases, argument filtering, and both data paths."""

    @classmethod
    def setUpClass(cls):
        """Create deterministic data suitable for all supported model types."""
        rng = np.random.default_rng(20260803)
        cls.nobs = 700
        cls.x = rng.normal(size=cls.nobs)
        cls.z = rng.normal(size=cls.nobs)
        cls.exog = np.column_stack((np.ones(cls.nobs), cls.x))
        cls.exog_infl = np.column_stack((np.ones(cls.nobs), cls.z))
        cls.mean = np.exp(0.25 + 0.2 * cls.x)
        cls.zero_probability = expit(-0.65 + 0.45 * cls.z)

        cls.poisson = rng.poisson(cls.mean)

        alpha = 0.55
        nb2_size = 1.0 / alpha
        nb2_probability = nb2_size / (nb2_size + cls.mean)
        cls.nb2 = rng.negative_binomial(nb2_size, nb2_probability)

        nb1_size = cls.mean / alpha
        nb1_probability = 1.0 / (1.0 + alpha)
        cls.nb1 = rng.negative_binomial(nb1_size, nb1_probability)

        cls.zip = cls.poisson.copy()
        cls.zip[rng.random(cls.nobs) < cls.zero_probability] = 0
        cls.zinb = cls.nb2.copy()
        cls.zinb[rng.random(cls.nobs) < cls.zero_probability] = 0

        cls.poisson_hurdle = _sample_zero_truncated_poisson(
            rng, cls.mean
        )
        cls.poisson_hurdle[
            rng.random(cls.nobs) < cls.zero_probability
        ] = 0
        cls.nb_hurdle = _sample_zero_truncated_nb2(
            rng, cls.mean, alpha
        )
        cls.nb_hurdle[
            rng.random(cls.nobs) < cls.zero_probability
        ] = 0

        gamma_shape = 4.0
        cls.gamma = rng.gamma(
            gamma_shape, cls.mean / gamma_shape
        )
        cls.gamma_hurdle = cls.gamma.copy()
        cls.gamma_hurdle[
            rng.random(cls.nobs) < cls.zero_probability
        ] = 0.0

    def _array_fit(self, endog, model_name, **kwargs):
        """Fit an array model while keeping expected warnings out of output."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return DISTRIBUTIONAL_MODEL(
                endog,
                self.exog,
                model_name=model_name,
                exog_infl=self.exog_infl,
                exog_names=['Intercept', 'x'],
                exog_infl_names=['Intercept', 'z'],
                cov_type='NONROBUST',
                **kwargs,
            )

    def test_array_api_dispatches_every_supported_model(self):
        """Build and fit all ten public concrete models through aliases."""
        cases = (
            (self.poisson, 'POISSON', Poisson, {}),
            (self.zip, 'zero_inflated poisson', ZeroInflatedPoisson, {}),
            (self.zinb, 'ZINB', ZeroInflatedNegativeBinomial, {}),
            (self.nb1, 'negative-binomial-one', NegativeBinomial1, {}),
            (self.nb2, 'NB_2', NegativeBinomial2, {}),
            (self.poisson, 'generalized poisson', GeneralizedPoisson, {}),
            (self.poisson_hurdle, 'hurdle_poisson', PoissonHurdle, {}),
            (
                self.nb_hurdle, 'negative binomial p hurdle',
                NegativeBinomialPHurdle, {'p': 2},
            ),
            (self.gamma_hurdle, 'GAMMA-HURDLE', GammaHurdle, {}),
            (self.gamma, 'gamma', Gamma, {}),
        )
        for endog, name, expected_class, kwargs in cases:
            with self.subTest(model_name=name):
                fit = self._array_fit(endog, name, **kwargs)
                self.assertIsInstance(fit, DistributionalModelResults)
                self.assertIsInstance(fit.model, expected_class)

    def test_formula_api_and_irrelevant_arguments(self):
        """Fit formula models and silently filter options that do not apply."""
        poisson_data = pd.DataFrame({
            'y': self.poisson, 'x': self.x, 'z': self.z,
        })
        poisson_fit = distributional_model(
            'y ~ x',
            poisson_data,
            model_name=' pOiSsOn ',
            exog_infl='z',
            p=999,
            positive_link='identity',
            positive_fit_kwargs={'ignored': True},
            hurdle_fit_kwargs={'ignored': True},
            cov_type='NONROBUST',
        )
        self.assertIsInstance(poisson_fit.model, Poisson)
        self.assertEqual(poisson_fit.param_names, ['Intercept', 'x'])

        zip_data = pd.DataFrame({
            'y': self.zip, 'x': self.x, 'z': self.z,
        })
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            zip_fit = distributional_model(
                'y ~ x', zip_data, model_name='ZIP', exog_infl='z',
                cov_type='NONROBUST',
            )
        self.assertIsInstance(zip_fit.model, ZeroInflatedPoisson)
        self.assertEqual(
            zip_fit.param_names[-2:], ['inflate_Intercept', 'inflate_z']
        )

        gp_fit = distributional_model(
            'y ~ x', poisson_data, model_name='GP_2',
            cov_type='NONROBUST',
        )
        self.assertIsInstance(gp_fit.model, GeneralizedPoisson)
        self.assertEqual(gp_fit.model.p, 2)

    def test_hurdle_zero_model_option_reaches_array_and_formula_models(self):
        """Route the censored-Poisson zero process through both public APIs."""
        array_fit = self._array_fit(
            self.poisson_hurdle,
            'poisson hurdle',
            zero_model='poisson',
        )
        self.assertEqual(array_fit.model.zero_model, 'poisson')
        self.assertEqual(array_fit.zero_model, 'poisson')
        self.assertIsInstance(array_fit.hurdle_fit.link, CLogLog)

        data = pd.DataFrame({
            'y': self.poisson_hurdle,
            'x': self.x,
            'z': self.z,
        })
        formula_fit = distributional_model(
            'y ~ x',
            data,
            model_name='poisson hurdle',
            exog_infl='z',
            zero_model='poisson',
            cov_type='NONROBUST',
        )
        self.assertEqual(formula_fit.model.zero_model, 'poisson')
        self.assertIsInstance(formula_fit.hurdle_fit.link, CLogLog)

    def test_array_add_constant_names_and_root_exports(self):
        """Add an intercept and expose identical entry points on ``kanly.api``."""
        fit = DISTRIBUTIONAL_MODEL(
            self.poisson,
            self.x,
            model_name='pois',
            add_constant=True,
            exog_names=['x'],
            exog_infl=self.exog_infl,
            exog_infl_names=['ignored', 'ignored_too'],
            p=-10,
            positive_link='ignored',
            cov_type='NONROBUST',
        )
        self.assertEqual(fit.param_names, ['Intercept', 'x'])
        self.assertTrue(fit.model.has_intercept)
        self.assertIs(root_distributional_model, distributional_model)
        self.assertIs(ROOT_DISTRIBUTIONAL_MODEL, DISTRIBUTIONAL_MODEL)

    def test_parameterization_aliases_and_conflicts(self):
        """Apply implied p values and reject contradictory explicit values."""
        gp2 = self._array_fit(self.poisson, 'gp2')
        self.assertEqual(gp2.model.p, 2)

        nb1_hurdle = self._array_fit(self.nb_hurdle, 'NB1_HURDLE')
        self.assertEqual(nb1_hurdle.model.p, 1)

        with self.assertRaisesRegex(ValueError, 'implies p=2'):
            self._array_fit(self.poisson, 'gp2', p=1)
        with self.assertRaisesRegex(ValueError, 'Unknown distributional'):
            self._array_fit(self.poisson, 'negative binomial seventeen')
        with self.assertRaisesRegex(TypeError, 'model_name must be a string'):
            self._array_fit(self.poisson, 2)

    def test_alias_table_and_debug_dispatch(self):
        """Publish every model in the lookup and describe normalized dispatch."""
        canonical_names = {row[0] for row in DISTRIBUTIONAL_MODEL_ALIASES}
        self.assertEqual(
            canonical_names,
            {
                'Poisson', 'ZeroInflatedPoisson',
                'ZeroInflatedNegativeBinomial', 'NegativeBinomial1',
                'NegativeBinomial2', 'GeneralizedPoisson',
                'PoissonHurdle', 'NegativeBinomialPHurdle',
                'GammaHurdle', 'GaussianHurdle', 'LognormalHurdle',
                'InverseGaussianHurdle', 'Gamma',
            },
        )

        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            DISTRIBUTIONAL_MODEL(
                self.poisson,
                self.exog,
                model_name='Poisson',
                exog_infl=self.exog_infl,
                debug=True,
                cov_type='NONROBUST',
            )
        text = output.getvalue()
        self.assertIn('DISTRIBUTIONAL MODEL API DISPATCH', text)
        self.assertIn("Normalized key: poisson", text)
        self.assertIn('Selected class: Poisson', text)
        self.assertIn('exog_infl: ignored for this one-part model', text)


if __name__ == '__main__':
    unittest.main()
