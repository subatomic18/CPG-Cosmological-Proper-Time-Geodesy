#!/usr/bin/env python3
import unittest
import numpy as np

from cpg_gp import Matern32GP


class TestMatern32GP(unittest.TestCase):
    def test_linear_signal_recovery(self):
        x = np.array([0.1, 0.3, 0.55, 0.8, 1.1, 1.4, 1.8])
        y = 70.0 + 55.0 * x
        C = np.diag(np.full(len(x), 2.0**2))
        xe = np.array([0.2, 0.7, 1.3, 1.6])
        p = Matern32GP(x, y, C).fit_predict(xe)
        self.assertTrue(np.allclose(p.mean, 70.0 + 55.0 * xe, atol=1.0))
        self.assertEqual(p.covariance.shape, (4, 4))
        self.assertGreaterEqual(np.linalg.eigvalsh(p.covariance).min(), -1e-9)

    def test_full_covariance(self):
        x = np.array([0.2, 0.5, 0.9, 1.3])
        y = np.array([80.0, 92.0, 118.0, 145.0])
        C = np.array([[9, 2, 0, 0], [2, 9, 1, 0], [0, 1, 16, 3], [0, 0, 3, 16]], float)
        p = Matern32GP(x, y, C).fit_predict(np.array([0.4, 1.0]))
        self.assertTrue(np.all(np.isfinite(p.mean)))
        self.assertTrue(np.all(np.diag(p.covariance) > 0))

    def test_no_extrapolation(self):
        x = np.array([0.2, 0.5, 0.9])
        y = np.array([80.0, 95.0, 120.0])
        C = np.eye(3)
        with self.assertRaises(ValueError):
            Matern32GP(x, y, C).fit_predict(np.array([0.1]))


if __name__ == "__main__":
    unittest.main()
