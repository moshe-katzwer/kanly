"""Dense/sparse parity tests for distributional regression models."""

import unittest
import warnings

import numpy as np
from scipy.sparse import csc_matrix, isspmatrix

from kanly.distributional_models import DISTRIBUTIONAL_MODEL
from kanly.distributional_models.base import (
    _build_score_obs,
    _weight_score_obs,
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
    GaussianHurdle,
    InverseGaussianHurdle,
    LognormalHurdle,
    NegativeBinomialPHurdle,
    PoissonHurdle,
)


def _dense(values):
    """Convert a sparse result to an ndarray for numerical comparisons."""
    return values.toarray() if isspmatrix(values) else np.asarray(values)


class TestSparseDistributionalModels(unittest.TestCase):
    """Require storage-type fidelity and dense/sparse numerical parity."""

    @classmethod
    def setUpClass(cls):
        cls.x = np.linspace(-1.0, 1.0, 10)
        cls.z = np.linspace(1.0, -1.0, 10) ** 3
        cls.exog = np.column_stack((np.ones(10), cls.x))
        cls.exog_infl = np.column_stack((np.ones(10), cls.z))
        cls.count_endog = np.array(
            [0.0, 1.0, 2.0, 0.0, 3.0, 1.0, 4.0, 0.0, 2.0, 1.0]
        )
        cls.positive_endog = np.array(
            [0.7, 1.1, 1.8, 0.9, 2.6, 1.4, 3.2, 0.8, 2.1, 1.3]
        )
        cls.hurdle_count_endog = np.array(
            [0.0, 1.0, 2.0, 0.0, 3.0, 1.0, 4.0, 0.0, 2.0, 1.0]
        )
        cls.hurdle_positive_endog = np.array(
            [0.0, 1.1, 1.8, 0.0, 2.6, 1.4, 3.2, 0.0, 2.1, 1.3]
        )

    def _assert_model_parity(self, dense_model, sparse_model, params):
        """Compare core evaluations while requiring sparse score storage."""
        params = np.asarray(params, dtype=float)
        self.assertFalse(isspmatrix(dense_model.exog))
        self.assertTrue(isspmatrix(sparse_model.exog))
        self.assertFalse(dense_model.is_sparse_model)
        self.assertTrue(sparse_model.is_sparse_model)

        dense_score_obs = dense_model.score_obs(params)
        sparse_score_obs = sparse_model.score_obs(params)
        self.assertIsInstance(dense_score_obs, np.ndarray)
        self.assertTrue(isspmatrix(sparse_score_obs))
        self.assertEqual(dense_score_obs.shape, sparse_score_obs.shape)
        np.testing.assert_allclose(
            _dense(sparse_score_obs), dense_score_obs,
            rtol=1e-11, atol=1e-11,
        )
        np.testing.assert_allclose(
            sparse_model.loglike_obs(params),
            dense_model.loglike_obs(params),
            rtol=1e-12, atol=1e-12,
        )
        np.testing.assert_allclose(
            sparse_model.score(params), dense_model.score(params),
            rtol=1e-11, atol=1e-11,
        )
        np.testing.assert_allclose(
            sparse_model.predict(params), dense_model.predict(params),
            rtol=1e-12, atol=1e-12,
        )

    def test_one_part_models_have_dense_sparse_parity(self):
        """Cover every one-part likelihood and each extra score column."""
        cases = (
            (Poisson, self.count_endog, np.array([0.1, -0.2]), {}),
            (
                Gamma, self.positive_endog,
                np.array([0.1, -0.2, np.log(0.4)]), {},
            ),
            (
                GeneralizedPoisson, self.count_endog,
                np.array([0.1, -0.2, 0.05]), {'p': 2},
            ),
            (
                NegativeBinomial1, self.count_endog,
                np.array([0.1, -0.2, np.log(0.4)]), {},
            ),
            (
                NegativeBinomial2, self.count_endog,
                np.array([0.1, -0.2, np.log(0.4)]), {},
            ),
        )
        for model_class, endog, params, kwargs in cases:
            with self.subTest(model=model_class.__name__):
                dense_model = model_class(endog, self.exog, **kwargs)
                sparse_model = model_class(
                    endog, csc_matrix(self.exog), **kwargs
                )
                self._assert_model_parity(
                    dense_model, sparse_model, params
                )

    def test_zero_inflated_models_have_dense_sparse_parity(self):
        """Cover both count/inflation designs and dispersion score stacking."""
        cases = (
            (ZeroInflatedPoisson, np.array([0.1, -0.2, -0.5, 0.3])),
            (
                ZeroInflatedNegativeBinomial,
                np.array([0.1, -0.2, -0.5, 0.3, np.log(0.4)]),
            ),
        )
        for model_class, params in cases:
            with self.subTest(model=model_class.__name__):
                dense_model = model_class(
                    self.count_endog,
                    self.exog,
                    exog_infl=self.exog_infl,
                )
                sparse_model = model_class(
                    self.count_endog,
                    csc_matrix(self.exog),
                    exog_infl=csc_matrix(self.exog_infl),
                )
                self._assert_model_parity(
                    dense_model, sparse_model, params
                )

    def test_hurdle_models_have_dense_sparse_parity(self):
        """Cover GLM, OLS, and exact-likelihood hurdle components."""
        cases = (
            (GaussianHurdle, self.hurdle_positive_endog, {}),
            (LognormalHurdle, self.hurdle_positive_endog, {}),
            (PoissonHurdle, self.hurdle_count_endog, {}),
            (GammaHurdle, self.hurdle_positive_endog, {}),
            (InverseGaussianHurdle, self.hurdle_positive_endog, {}),
            (NegativeBinomialPHurdle, self.hurdle_count_endog, {'p': 2}),
        )
        for model_class, endog, kwargs in cases:
            with self.subTest(model=model_class.__name__):
                dense_model = model_class(
                    endog,
                    self.exog,
                    exog_infl=self.exog_infl,
                    **kwargs,
                )
                sparse_model = model_class(
                    endog,
                    csc_matrix(self.exog),
                    exog_infl=csc_matrix(self.exog_infl),
                    **kwargs,
                )
                self._assert_model_parity(
                    dense_model, sparse_model, dense_model.get_start_params()
                )

    def test_either_two_part_design_makes_score_sparse(self):
        """Return a sparse combined Jacobian for either mixed input ordering."""
        params = np.array([0.1, -0.2, -0.5, 0.3])
        models = (
            ZeroInflatedPoisson(
                self.count_endog,
                self.exog,
                exog_infl=csc_matrix(self.exog_infl),
            ),
            ZeroInflatedPoisson(
                self.count_endog,
                csc_matrix(self.exog),
                exog_infl=self.exog_infl,
            ),
        )
        for model in models:
            with self.subTest(
                    main_sparse=isspmatrix(model.exog),
                    inflation_sparse=isspmatrix(model.exog_infl)):
                self.assertTrue(model.is_sparse_model)
                self.assertTrue(isspmatrix(model.score_obs(params)))

        hurdle = PoissonHurdle(
            self.hurdle_count_endog,
            self.exog,
            exog_infl=csc_matrix(self.exog_infl),
        )
        self.assertTrue(isspmatrix(
            hurdle.score_obs(hurdle.get_start_params())
        ))

    def test_formula_dense_threshold_controls_both_designs(self):
        """Keep sparse-first formula matrices sparse only above the threshold."""
        data = {
            'y': self.count_endog,
            'x': self.x,
            'z': self.z,
        }
        sparse_model = ZeroInflatedPoisson.build_model_from_formula(
            'y ~ x', data, exog_infl='z', dense_threshold_mb=0
        )
        dense_model = ZeroInflatedPoisson.build_model_from_formula(
            'y ~ x', data, exog_infl='z', dense_threshold_mb=1024
        )
        self.assertTrue(isspmatrix(sparse_model.exog))
        self.assertTrue(isspmatrix(sparse_model.exog_infl))
        self.assertTrue(isspmatrix(
            sparse_model.score_obs(np.zeros(4))
        ))
        self.assertIsInstance(dense_model.exog, np.ndarray)
        self.assertIsInstance(dense_model.exog_infl, np.ndarray)
        self.assertIsInstance(dense_model.score_obs(np.zeros(4)), np.ndarray)

    def test_sparse_fit_keeps_score_sparse_and_covariance_dense(self):
        """Exercise optimization and sandwich inference without densifying X."""
        model = Poisson(self.count_endog, csc_matrix(self.exog))
        fit = model.fit(start_params=np.zeros(2), cov_type='SANDWICH')
        self.assertTrue(isspmatrix(fit.model.exog))
        self.assertTrue(isspmatrix(fit.score_obs()))
        self.assertIsInstance(fit.score(), np.ndarray)
        self.assertIsInstance(fit.meat, np.ndarray)
        self.assertEqual(fit.meat.shape, (2, 2))

    def test_sparse_score_builder_fills_one_csc_matrix_by_column(self):
        """Drop score zeros while mapping subset blocks into final CSC rows."""
        sparse_values = csc_matrix(np.array([
            [1.0, 0.0],
            [2.0, 3.0],
            [0.0, 4.0],
        ]))
        score = _build_score_obs(
            (
                (sparse_values, np.array([2.0, 0.0, -1.0]),
                 np.array([0, 2, 4])),
                (np.array([1.0, 0.0, 2.0, 0.0, 3.0]), None),
            ),
            nobs=5,
        )
        expected = np.array([
            [2.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0],
            [0.0, -4.0, 3.0],
        ])
        self.assertTrue(isspmatrix(score))
        self.assertEqual(score.format, 'csc')
        self.assertEqual(score.nnz, np.count_nonzero(expected))
        self.assertFalse(np.any(score.data == 0.0))
        np.testing.assert_array_equal(score.toarray(), expected)

        weights = np.array([1.0, 2.0, 3.0, 4.0, 0.0])
        weighted = _weight_score_obs(score, weights)
        self.assertIs(weighted, score)
        np.testing.assert_array_equal(
            weighted.toarray(), expected * weights[:, None]
        )
        self.assertFalse(np.any(weighted.data == 0.0))

    def test_hurdle_component_fits_reuse_cached_sparse_data(self):
        """Pass cached parent subsets directly to native LM, GLM, and logit."""
        rng = np.random.default_rng(903)
        nobs = 240
        x = rng.normal(size=nobs)
        z = rng.normal(size=nobs)
        exog = csc_matrix(np.column_stack((np.ones(nobs), x)))
        exog_infl = csc_matrix(np.column_stack((np.ones(nobs), z)))
        is_zero = rng.random(nobs) < 0.35

        gaussian_y = 3.0 + 0.2 * x + rng.normal(scale=0.3, size=nobs)
        gaussian_y[is_zero] = 0.0
        gaussian = GaussianHurdle(
            gaussian_y, exog, exog_infl=exog_infl
        )
        self.assertIs(gaussian.exog, exog)
        self.assertIs(gaussian.exog_infl, exog_infl)
        self.assertTrue(np.shares_memory(gaussian.endog, gaussian_y))
        gaussian_fit = gaussian.fit(cov_type='NONROBUST')
        self.assertIs(
            gaussian_fit.positive_fit.model.exog,
            gaussian._positive_exog,
        )
        self.assertIs(
            gaussian_fit.positive_fit.model.endog,
            gaussian._positive_linear_component_endog,
        )
        self.assertIs(
            gaussian_fit.hurdle_fit.model.exog,
            gaussian.exog_infl,
        )
        self.assertIs(
            gaussian_fit.hurdle_fit.model.endog,
            gaussian._hurdle_component_endog,
        )

        rate = np.exp(0.1 + 0.2 * x)
        count_y = rng.poisson(rate)
        redraw = count_y == 0
        while np.any(redraw):
            count_y[redraw] = rng.poisson(rate[redraw])
            redraw = count_y == 0
        count_y[is_zero] = 0
        poisson = PoissonHurdle(count_y, exog, exog_infl=exog_infl)
        poisson_fit = poisson.fit(cov_type='NONROBUST')
        self.assertIs(
            poisson_fit.positive_fit.model.exog,
            poisson._positive_exog,
        )
        self.assertIs(
            poisson_fit.positive_fit.model.endog,
            poisson._positive_component_endog,
        )
        self.assertIs(
            poisson_fit.hurdle_fit.model.exog,
            poisson.exog_infl,
        )
        self.assertIs(
            poisson_fit.hurdle_fit.model.endog,
            poisson._hurdle_component_endog,
        )

    def test_array_api_add_constant_retains_sparse_input(self):
        """Prepend the array-API intercept without densifying sparse exog."""
        rng = np.random.default_rng(902)
        x = rng.normal(size=120)
        endog = rng.poisson(np.exp(0.1 + 0.2 * x))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            fit = DISTRIBUTIONAL_MODEL(
                endog,
                csc_matrix(x[:, None]),
                model_name='Poisson',
                add_constant=True,
                cov_type='NONROBUST',
            )
        self.assertTrue(isspmatrix(fit.model.exog))
        self.assertTrue(isspmatrix(fit.score_obs()))
        self.assertEqual(fit.model.exog.shape, (len(x), 2))

    def test_sparse_design_validation_rejects_nonfinite_values(self):
        """Apply dense finite-value validation to sparse matrix data arrays."""
        invalid = csc_matrix(np.array([[1.0], [np.nan]]))
        with self.assertRaisesRegex(ValueError, 'finite values'):
            Poisson(np.array([0.0, 1.0]), invalid)
        with self.assertRaisesRegex(ValueError, 'finite values'):
            ZeroInflatedPoisson(
                np.array([0.0, 1.0]), np.ones((2, 1)),
                exog_infl=invalid,
            )


if __name__ == '__main__':
    unittest.main()
