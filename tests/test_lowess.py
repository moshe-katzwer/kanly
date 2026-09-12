"""Regression tests for LOWESS neighborhood selection."""

import unittest

import numpy as np

from kanly.api import LOWESS


class TestLowessNeighborhoodBounds(unittest.TestCase):
    """Keep half-open neighborhood bounds valid at the right endpoint."""

    def test_rightmost_observations_influence_endpoint_fit(self):
        x = np.arange(44, dtype=float)

        for do_njit in (False, True):
            endpoint_fits = {}
            for position in (0, -1, -2):
                with self.subTest(do_njit=do_njit, position=position):
                    y = np.zeros_like(x)
                    y[position] = 100.0

                    x_smooth, y_smooth = LOWESS(
                        y,
                        x,
                        xvals=x,
                        frac=0.5,
                        it=0,
                        degree=1,
                        delta=0,
                        do_njit=do_njit,
                    )

                    np.testing.assert_array_equal(x_smooth, x)
                    endpoint = 0 if position == 0 else -1
                    self.assertGreater(y_smooth[endpoint], 0.0)
                    endpoint_fits[position] = y_smooth[endpoint]

            np.testing.assert_allclose(
                endpoint_fits[-1], endpoint_fits[0], rtol=1e-12, atol=1e-12
            )

    def test_full_sample_neighborhood_recovers_linear_data(self):
        x = np.array([0.0, 0.5, 2.0, 4.5, 8.0, 13.0])
        y = 2.0 + 3.0 * x
        fitted = []

        for do_njit in (False, True):
            for grid_name, xvals in (("default", None), ("explicit", x)):
                with self.subTest(do_njit=do_njit, grid=grid_name):
                    x_smooth, y_smooth = LOWESS(
                        y,
                        x,
                        xvals=xvals,
                        frac=len(x),
                        it=0,
                        degree=1,
                        delta=0,
                        do_njit=do_njit,
                    )

                    np.testing.assert_array_equal(x_smooth, x)
                    np.testing.assert_allclose(
                        y_smooth, y, rtol=1e-9, atol=1e-9
                    )
                    if grid_name == "explicit":
                        fitted.append(y_smooth)

        np.testing.assert_allclose(fitted[0], fitted[1], rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
