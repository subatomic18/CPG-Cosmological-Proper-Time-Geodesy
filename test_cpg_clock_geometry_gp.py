#!/usr/bin/env python3
import unittest

from cpg_chronometers import CCPoint
from cpg_clock_geometry_closure import BAOPoint
from cpg_clock_geometry_gp import GPClockGeometryClosureAuditor


class TestGPClockGeometryClosure(unittest.TestCase):
    def test_constant_q_recovery(self):
        q_true = 147.5
        c = 299792.458
        cc = [
            CCPoint(0.2, 75.0, 1.0),
            CCPoint(0.5, 90.0, 1.0),
            CCPoint(0.8, 105.0, 1.0),
            CCPoint(1.1, 120.0, 1.0),
            CCPoint(1.4, 135.0, 1.0),
            CCPoint(1.7, 150.0, 1.0),
        ]
        bao = []
        for z in [0.35, 0.65, 0.95, 1.25, 1.55]:
            H = 65.0 + 50.0 * z
            dh = c / (q_true * H)
            bao.append(BAOPoint(z, dh, 0.003 * dh))

        out = GPClockGeometryClosureAuditor(cc, bao).reconstruct(
            n_draws=2500, seed=9, method="gp"
        )
        self.assertEqual(out["reconstruction"]["method"], "matern32-gaussian-process")
        self.assertAlmostEqual(out["constant_fit"]["Q0_mpc"], q_true, delta=1.5)
        self.assertGreater(out["constant_fit"]["p_value"], 0.05)
        for row in out["points"]:
            self.assertAlmostEqual(row["Q_over_Q0"], 1.0, delta=0.03)

    def test_pchip_still_available(self):
        c = 299792.458
        q = 147.5
        cc = [CCPoint(0.2, 80, 1), CCPoint(0.8, 110, 1), CCPoint(1.4, 140, 1)]
        bao = [
            BAOPoint(0.5, c / (q * 95.0), 0.01),
            BAOPoint(1.1, c / (q * 125.0), 0.01),
        ]
        out = GPClockGeometryClosureAuditor(cc, bao).reconstruct(
            n_draws=1000, seed=4, method="pchip"
        )
        self.assertEqual(out["reconstruction"]["method"], "pchip-monte-carlo")


if __name__ == "__main__":
    unittest.main()
