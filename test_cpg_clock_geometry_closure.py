#!/usr/bin/env python3
import unittest

import numpy as np

from cpg_chronometers import CCPoint
from cpg_clock_geometry_closure import BAOPoint, ClockGeometryClosureAuditor


class TestClockGeometryClosure(unittest.TestCase):
    def test_constant_q_recovery(self):
        # Construct a synthetic universe with H_CC(z) linear in z and constant
        # Q = Gamma*r_d = 147.5 Mpc. BAO is then generated exactly from
        # D_H/r_d = c/[Q H_CC].
        q_true = 147.5
        c = 299792.458

        cc = [
            CCPoint(0.2, 75.0, 0.3),
            CCPoint(0.6, 95.0, 0.3),
            CCPoint(1.0, 115.0, 0.3),
            CCPoint(1.4, 135.0, 0.3),
            CCPoint(1.8, 155.0, 0.3),
        ]
        bao_z = [0.4, 0.8, 1.2, 1.6]
        bao = []
        for z in bao_z:
            H = 65.0 + 50.0 * z
            dh_over_rd = c / (q_true * H)
            bao.append(BAOPoint(z, dh_over_rd, 0.002 * dh_over_rd))

        auditor = ClockGeometryClosureAuditor(cc, bao)
        out = auditor.reconstruct(n_draws=2500, seed=7)

        self.assertAlmostEqual(out["constant_fit"]["Q0_mpc"], q_true, delta=0.8)
        self.assertGreater(out["constant_fit"]["p_value"], 0.05)
        for row in out["points"]:
            self.assertAlmostEqual(row["Q_over_Q0"], 1.0, delta=0.02)

    def test_rd_conversion(self):
        q_true = 147.5
        c = 299792.458
        cc = [
            CCPoint(0.2, 80.0, 0.5),
            CCPoint(0.8, 110.0, 0.5),
            CCPoint(1.4, 140.0, 0.5),
        ]
        bao = []
        for z, H in [(0.5, 95.0), (1.1, 125.0)]:
            bao.append(BAOPoint(z, c / (q_true * H), 0.01))

        out = ClockGeometryClosureAuditor(cc, bao).reconstruct(
            n_draws=2000, seed=3, rd_mpc=147.5, rd_sigma_mpc=0.2
        )
        self.assertIsNotNone(out["rd_calibration"])
        self.assertAlmostEqual(out["rd_calibration"]["Gamma0"], 1.0, delta=0.02)
        for row in out["points"]:
            self.assertIn("Gamma", row)

    def test_rejects_extrapolation(self):
        cc = [
            CCPoint(0.2, 80.0, 2.0),
            CCPoint(0.8, 100.0, 2.0),
            CCPoint(1.4, 130.0, 2.0),
        ]
        bao = [BAOPoint(0.1, 20.0, 0.5), BAOPoint(1.0, 15.0, 0.5)]
        with self.assertRaises(ValueError):
            ClockGeometryClosureAuditor(cc, bao)

    def test_correlated_covariance(self):
        cc = [
            CCPoint(0.2, 80.0, 2.0),
            CCPoint(0.8, 100.0, 2.0),
            CCPoint(1.4, 130.0, 2.0),
        ]
        bao = [BAOPoint(0.5, 20.0, 0.4), BAOPoint(1.1, 15.0, 0.3)]
        cc_C = np.array(
            [[4.0, 1.0, 0.5], [1.0, 4.0, 1.0], [0.5, 1.0, 4.0]]
        )
        bao_C = np.array([[0.16, 0.03], [0.03, 0.09]])
        auditor = ClockGeometryClosureAuditor(
            cc, bao, cc_covariance=cc_C, bao_covariance=bao_C
        )
        out = auditor.reconstruct(n_draws=1200, seed=11)
        self.assertEqual(len(out["points"]), 2)
        self.assertEqual(np.asarray(out["Q_covariance_mpc2"]).shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
