"""Unit tests for the numerical helpers in ``lm_internal``.

These tests deliberately use small, deterministic inputs so they validate the
normal-equation calculations without depending on the formula interface.
"""

import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.sparse import csc_matrix, isspmatrix

from kanly.regression.linear_models.lm_internal import (
    LinearModelRegressionResultsRaw,
    check_types,
    fgls_internal,
    fit_least_squares_model_internal,
    get_exog_dot_endog,
    get_fitted_values,
    get_fit_summary_stats,
    get_gls_sum_squares,
    lin_mod_fit_predicted_values_and_residuals,
    lin_mod_get_mean_exog_columns,
    lin_mod_get_method,
    lin_mod_get_rsquared,
    lin_mod_get_sum_of_squares,
    lin_mod_get_sum_of_squares_gls,
    lm_internal,
    loglike_internal,
    loglike_internal_gls,
)


class TestLmInternal(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        self.y = np.array([1.2, 2.8, 5.1, 6.9])
        self.weights = np.array([1.0, 2.0, 0.5, 3.0])

    @staticmethod
    def _dense(value):
        return value.toarray() if isspmatrix(value) else np.asarray(value)

    def test_fitted_values_and_cross_products_for_dense_and_sparse_inputs(self):
        params = np.array([1.0, 2.0])
        fitted, resid = get_fitted_values(self.x, self.y, params)
        assert_allclose(fitted, [1.0, 3.0, 5.0, 7.0])
        assert_allclose(resid, [.2, -.2, .1, -.1])

        expected_wls = self.x.T @ (self.weights * self.y)
        assert_allclose(get_exog_dot_endog(self.x, self.y, weights=self.weights), expected_wls)
        assert_allclose(
            self._dense(get_exog_dot_endog(csc_matrix(self.x), self.y, weights=self.weights)).ravel(),
            expected_wls,
        )

    def test_cross_product_and_least_squares_support_gls(self):
        sigma = np.array([
            [2.0, .2, .0, .0],
            [.2, 1.0, .1, .0],
            [.0, .1, 1.5, .3],
            [.0, .0, .3, 1.2],
        ])
        sigma_inv = np.linalg.inv(sigma)
        expected_cross_product = self.x.T @ sigma_inv @ self.y
        expected_params = np.linalg.inv(self.x.T @ sigma_inv @ self.x) @ expected_cross_product

        assert_allclose(get_exog_dot_endog(self.x, self.y, sigma_inv=sigma_inv), expected_cross_product)
        params, covariance, eigenvals, condition_number = fit_least_squares_model_internal(
            self.x, self.y, sigma_inv=sigma_inv, normalize_XpX=False, compute_eigenvalues=True,
        )
        assert_allclose(params, expected_params)
        assert_allclose(covariance, np.linalg.inv(self.x.T @ sigma_inv @ self.x))
        self.assertEqual(len(eigenvals), 2)
        self.assertGreater(condition_number, 1)

    def test_lm_internal_matches_ols_and_wls_for_scaled_and_sparse_designs(self):
        expected_ols = np.linalg.lstsq(self.x, self.y, rcond=None)[0]
        expected_wls = np.linalg.inv(self.x.T @ np.diag(self.weights) @ self.x) @ self.x.T @ np.diag(self.weights) @ self.y

        for exog in (self.x, csc_matrix(self.x)):
            for scale_design_matrix in (False, True):
                result = lm_internal(
                    self.y, exog, scale_design_matrix=scale_design_matrix, compute_eigenvalues=False,
                )
                assert_allclose(self._dense(result.params).ravel(), expected_ols)
                assert_allclose(self._dense(result.resid_raw).ravel(), self.y - self.x @ expected_ols)
                self.assertIsNone(result.absorb_info)
                self.assertIsNone(result.instrument_info)

        weighted_result = lm_internal(
            self.y, self.x, weights=self.weights, scale_design_matrix=True, compute_eigenvalues=False,
        )
        assert_allclose(self._dense(weighted_result.params).ravel(), expected_wls)

    def test_type_normalisation_promotes_related_inputs_to_sparse(self):
        endog, exog, weights, instruments, absorb, sigma, sigma_inv = check_types(
            self.y, csc_matrix(self.x), csc_matrix(self.weights).T, self.x[:, :1], None, None, None,
        )
        self.assertTrue(hasattr(endog, "tocsc"))
        self.assertTrue(hasattr(exog, "tocsc"))
        self.assertTrue(hasattr(instruments, "tocsc"))
        assert_allclose(weights, self.weights)
        self.assertIsNone(absorb)
        self.assertIsNone(sigma)
        self.assertIsNone(sigma_inv)

    def test_summary_and_prediction_helpers(self):
        params = np.array([1.0, 2.0])
        y_hat, y_hat_instrumented, resid, resid_instrumented, ssr, wssr = \
            lin_mod_fit_predicted_values_and_residuals(params, self.y, self.x, self.x, self.weights)
        assert_allclose(y_hat, y_hat_instrumented)
        assert_allclose(resid, resid_instrumented)
        self.assertAlmostEqual(ssr, np.sum(resid ** 2))
        self.assertAlmostEqual(wssr, np.sum(self.weights * resid ** 2))

        summary = get_fit_summary_stats(
            nobs=len(self.y), num_absorbed=0, params=params, endog=self.y, exog=self.x,
            exog_absorb_instrumented=self.x, rsquared_within_raw=.0, weights=None,
            do_fgls=False, is_weighted=False, has_implicit_constant=False, has_intercept=True,
            is_absorb=False, absorb_info=None, sigma=None, sigma_inv=None,
        )
        self.assertEqual(len(summary), 16)
        self.assertEqual(summary[0], 2)  # df_resid
        assert_allclose(summary[9], resid)
        assert_allclose(summary[11], y_hat)
        self.assertIsNone(summary[13])  # no absorbed fixed effects

    def test_sum_of_squares_likelihood_and_method_helpers(self):
        y = np.array([1.0, 2.0, 3.0])
        weights = np.array([1.0, 2.0, 3.0])
        sst, wsst, uncentered_tss = lin_mod_get_sum_of_squares(y, weights)
        self.assertAlmostEqual(sst, np.var(y))
        self.assertAlmostEqual(wsst, np.sum(weights * (y - np.average(y, weights=weights)) ** 2))
        self.assertAlmostEqual(uncentered_tss, np.sum(weights * y ** 2))

        rsquared, rsquared_adj = lin_mod_get_rsquared(2.0, 8.0, 10.0, True, 10, 8)
        self.assertAlmostEqual(rsquared, .75)
        self.assertAlmostEqual(rsquared_adj, .71875)
        self.assertEqual(lin_mod_get_method(), "OLS")
        self.assertEqual(lin_mod_get_method(is_weighted=True), "WLS")
        self.assertEqual(lin_mod_get_method(is_iv=True, ridge_kwds={"alpha": 1.0}), "IV (2SLS)-RIDGE")

        residuals = np.array([1.0, -2.0])
        sigma_inv = np.array([[2.0, .5], [.5, 1.0]])
        gls_ssr = residuals @ sigma_inv @ residuals
        self.assertAlmostEqual(get_gls_sum_squares(residuals, None, sigma_inv), gls_ssr)
        _, gls_tss, gls_uncentered_tss = lin_mod_get_sum_of_squares_gls(y[:2], None, sigma_inv)
        self.assertLess(gls_tss, gls_uncentered_tss)
        self.assertTrue(np.isfinite(loglike_internal(residuals, 2, weights=np.ones(2))))
        self.assertTrue(np.isfinite(loglike_internal_gls(gls_ssr, 2, sigma_inv=sigma_inv)))

    def test_column_means_result_container_and_fgls_guards(self):
        dense_means, dense_count = lin_mod_get_mean_exog_columns(False, None, self.x)
        sparse_means, sparse_weight_sum = lin_mod_get_mean_exog_columns(True, self.weights, csc_matrix(self.x))
        assert_allclose(dense_means, self.x.mean(axis=0))
        self.assertEqual(dense_count, len(self.x))
        assert_allclose(sparse_means, np.average(self.x, axis=0, weights=self.weights))
        self.assertEqual(sparse_weight_sum, self.weights.sum())

        raw = LinearModelRegressionResultsRaw(
            np.array([1.0]), np.eye(1), np.ones((3, 1)), None, None,
            np.ones(3), np.zeros(3), .5, 1.0, np.ones(1),
        )
        self.assertEqual(raw.nobs, 3)
        self.assertEqual(raw.num_params, 1)
        self.assertIs(raw.final_design_matrix, raw.exog_absorb_instrumented)
        self.assertIn("params", repr(raw))

        with self.assertRaisesRegex(Exception, "GLS and FGLS"):
            fgls_internal(4, self.y, self.x, do_fgls=True, sigma=np.eye(4))

        result = fgls_internal(4, self.y, self.x, do_fgls=False, compute_eigenvalues=False)
        assert_allclose(result["result"].params, np.linalg.lstsq(self.x, self.y, rcond=None)[0])
        self.assertEqual(result["fgls_info"], {})


if __name__ == "__main__":
    unittest.main()
