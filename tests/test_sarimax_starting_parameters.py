"""Tests for SARIMAX starting-parameter stabilization."""

import unittest

import numpy as np

from kanly.time_series.sarimax.handle_params import stabilize_arma_starts
from kanly.time_series.sarimax.hannan_rissanen import hannan_rissanen


def polynomial_roots(params, lags, *, moving_average):
    """Return roots for an explicitly lagged AR or MA polynomial."""
    polynomial = np.zeros(max(lags, default=0) + 1)
    polynomial[0] = 1.0
    if lags:
        polynomial[lags] = params if moving_average else -np.asarray(params)
    return np.roots(polynomial[::-1])


class TestSarimaxStartingParameters(unittest.TestCase):

    def test_valid_model_coefficients_are_not_transformed(self):
        ar, ma, sar, sma = stabilize_arma_starts(
            [0.8], [0.6], [-0.4], [-0.3],
            ar_lags=[1], ma_lags=[1], sar_lags=[1], sma_lags=[1],
            enforce_stationarity=True,
            enforce_invertibility=True,
        )

        np.testing.assert_array_equal(ar, [0.8])
        np.testing.assert_array_equal(ma, [0.6])
        np.testing.assert_array_equal(sar, [-0.4])
        np.testing.assert_array_equal(sma, [-0.3])

    def test_invalid_sparse_blocks_are_repaired_using_their_real_lags(self):
        ar, ma, _, _ = stabilize_arma_starts(
            [0.0, 1.2], [0.0, -1.4], [], [],
            ar_lags=[1, 3], ma_lags=[2, 5],
            sar_lags=[], sma_lags=[],
            enforce_stationarity=True,
            enforce_invertibility=True,
        )

        self.assertAlmostEqual(np.sum(np.abs(ar)), 0.98)
        self.assertAlmostEqual(np.sum(np.abs(ma)), 0.98)
        self.assertTrue(np.all(
            np.abs(polynomial_roots(ar, [1, 3], moving_average=False)) > 1
        ))
        self.assertTrue(np.all(
            np.abs(polynomial_roots(ma, [2, 5], moving_average=True)) > 1
        ))

    def test_disabled_repairs_leave_even_invalid_values_alone(self):
        values = stabilize_arma_starts(
            [1.2], [-1.3], [1.4], [-1.5],
            ar_lags=[1], ma_lags=[1], sar_lags=[1], sma_lags=[1],
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        for actual, expected in zip(values, ([1.2], [-1.3], [1.4], [-1.5])):
            np.testing.assert_array_equal(actual, expected)

    def test_pure_ar_hannan_rissanen_starts_are_stabilized(self):
        # This branch previously bypassed the stationarity helper entirely.
        endog = 1.2 ** np.arange(30)

        params = hannan_rissanen(
            endog,
            ar_lags=[1],
            enforce_stationarity=True,
            concentrate_scale=True,
        )

        np.testing.assert_allclose(params, [0.98])


if __name__ == "__main__":
    unittest.main()
