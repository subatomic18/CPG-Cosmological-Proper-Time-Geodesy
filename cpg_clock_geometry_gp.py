#!/usr/bin/env python3
"""GP-first clock-geometry closure diagnostic for CPG.

Recommended real-data front end for the clock-geometry test. The default
reconstruction is a covariance-aware Matern-3/2 Gaussian process. The legacy
Monte-Carlo PCHIP calculation remains available as a robustness check.
"""
from __future__ import annotations

import argparse
import json
from typing import Optional

import numpy as np
from scipy.stats import chi2

from cpg_chronometers import build_covariance, load_cc_csv, load_systematics_csv
from cpg_clock_geometry_closure import (
    C_LIGHT_KM_S,
    ClockGeometryClosureAuditor,
    load_bao_csv,
    load_covariance_csv,
)
from cpg_gp import Matern32GP


class GPClockGeometryClosureAuditor(ClockGeometryClosureAuditor):
    """Clock-geometry auditor using a covariance-aware Matern-3/2 GP."""

    def reconstruct(
        self,
        n_draws: int = 8000,
        seed: int = 42,
        rd_mpc: Optional[float] = None,
        rd_sigma_mpc: float = 0.0,
        method: str = "gp",
    ) -> dict:
        method = str(method).lower()
        if method == "pchip":
            out = super().reconstruct(
                n_draws=n_draws,
                seed=seed,
                rd_mpc=rd_mpc,
                rd_sigma_mpc=rd_sigma_mpc,
            )
            out["reconstruction"] = {"method": "pchip-monte-carlo"}
            return out
        if method != "gp":
            raise ValueError("method must be 'gp' or 'pchip'")
        if int(n_draws) < 500:
            raise ValueError("n_draws must be at least 500")
        if rd_mpc is not None and rd_mpc <= 0:
            raise ValueError("rd_mpc must be positive")
        if rd_sigma_mpc < 0:
            raise ValueError("rd_sigma_mpc must be non-negative")
        if rd_mpc is None and rd_sigma_mpc != 0:
            raise ValueError("rd_sigma_mpc requires rd_mpc")

        gp = Matern32GP(self.cc_z, self.cc_H, self.cc_C)
        pred = gp.fit_predict(self.bao_z)

        rng = np.random.default_rng(seed)
        h_draws = rng.multivariate_normal(pred.mean, pred.covariance, size=int(n_draws))
        bao_draws = rng.multivariate_normal(self.bao_DHrd, self.bao_C, size=int(n_draws))
        good = np.all(h_draws > 0, axis=1) & np.all(bao_draws > 0, axis=1)
        h_draws = h_draws[good]
        bao_draws = bao_draws[good]
        if len(h_draws) < max(250, int(0.5 * n_draws)):
            raise RuntimeError("Too many nonphysical GP/BAO Monte-Carlo draws")

        q_draws = C_LIGHT_KM_S / (bao_draws * h_draws)
        h16, h50, h84 = np.percentile(h_draws, [16, 50, 84], axis=0)
        q16, q50, q84 = np.percentile(q_draws, [16, 50, 84], axis=0)
        q_mean = np.mean(q_draws, axis=0)
        q_cov = np.atleast_2d(np.cov(q_draws, rowvar=False, ddof=1))
        q0, q0_sigma, chisq = self._gls_constant(q_mean, q_cov)
        dof = len(self.bao_points) - 1
        p_const = float(chi2.sf(chisq, dof))

        one = np.ones(len(self.bao_points), dtype=float)
        W = np.linalg.pinv(q_cov, hermitian=True)
        denom = float(one @ W @ one)
        q0_draws = (q_draws @ W @ one) / denom
        shape_draws = q_draws / q0_draws[:, None]
        s16, s50, s84 = np.percentile(shape_draws, [16, 50, 84], axis=0)
        q0_d16, q0_d50, q0_d84 = np.percentile(q0_draws, [16, 50, 84])

        points = []
        for j, p in enumerate(self.bao_points):
            points.append({
                "z": float(p.z),
                "H_CC": float(h50[j]),
                "H_CC_lo": float(h16[j]),
                "H_CC_hi": float(h84[j]),
                "Q_mpc": float(q50[j]),
                "Q_lo_mpc": float(q16[j]),
                "Q_hi_mpc": float(q84[j]),
                "Q_over_Q0": float(s50[j]),
                "Q_over_Q0_lo": float(s16[j]),
                "Q_over_Q0_hi": float(s84[j]),
            })

        result = {
            "observable": "Q(z) = Gamma(z) r_d",
            "null_hypothesis": "Q(z) is constant in redshift (dGamma/dz = 0)",
            "reconstruction": {
                "method": "matern32-gaussian-process",
                "amplitude_km_s_Mpc": pred.amplitude,
                "length_scale_z": pred.length_scale,
                "linear_mean_intercept_km_s_Mpc": pred.mean_intercept,
                "linear_mean_slope_km_s_Mpc_per_z": pred.mean_slope,
                "negative_log_likelihood": pred.negative_log_likelihood,
                "hyperparameter_uncertainty_propagated": False,
            },
            "points": points,
            "constant_fit": {
                "Q0_mpc": q0,
                "Q0_sigma_mpc": q0_sigma,
                "Q0_draw_median_mpc": float(q0_d50),
                "Q0_draw_lo_mpc": float(q0_d16),
                "Q0_draw_hi_mpc": float(q0_d84),
                "chi2": chisq,
                "dof": int(dof),
                "p_value": p_const,
            },
            "H_CC_covariance": pred.covariance.tolist(),
            "Q_covariance_mpc2": q_cov.tolist(),
            "accepted_draws": int(len(q_draws)),
            "requested_draws": int(n_draws),
            "seed": int(seed),
            "rd_calibration": None,
        }

        if rd_mpc is not None:
            if rd_sigma_mpc == 0:
                rd_draws = np.full(len(q_draws), float(rd_mpc))
                q_for_gamma = q_draws
                q0_for_gamma = q0_draws
            else:
                rd_all = rng.normal(float(rd_mpc), float(rd_sigma_mpc), len(q_draws))
                positive = rd_all > 0
                rd_draws = rd_all[positive]
                q_for_gamma = q_draws[positive]
                q0_for_gamma = q0_draws[positive]
            gamma_draws = q_for_gamma / rd_draws[:, None]
            g16, g50, g84 = np.percentile(gamma_draws, [16, 50, 84], axis=0)
            for j, row in enumerate(result["points"]):
                row["Gamma"] = float(g50[j])
                row["Gamma_lo"] = float(g16[j])
                row["Gamma_hi"] = float(g84[j])
            gamma0_draws = q0_for_gamma / rd_draws
            g0_16, g0_50, g0_84 = np.percentile(gamma0_draws, [16, 50, 84])
            result["rd_calibration"] = {
                "rd_mpc": float(rd_mpc),
                "rd_sigma_mpc": float(rd_sigma_mpc),
                "Gamma0": float(g0_50),
                "Gamma0_lo": float(g0_16),
                "Gamma0_hi": float(g0_84),
            }
        return result


def main():
    parser = argparse.ArgumentParser(description="CPG GP-first clock-geometry closure")
    parser.add_argument("--cc", required=True, help="cosmic-chronometer CSV")
    parser.add_argument("--bao", required=True, help="radial BAO CSV")
    parser.add_argument("--cc-covariance", default=None)
    parser.add_argument("--bao-covariance", default=None)
    parser.add_argument("--cc-systematics", default=None)
    parser.add_argument("--method", choices=("gp", "pchip"), default="gp")
    parser.add_argument("--draws", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rd", type=float, default=None)
    parser.add_argument("--rd-sigma", type=float, default=0.0)
    args = parser.parse_args()

    cc = load_cc_csv(args.cc)
    bao = load_bao_csv(args.bao)
    if args.cc_covariance:
        cc_cov = load_covariance_csv(args.cc_covariance, len(cc))
    elif args.cc_systematics:
        cc_cov = build_covariance(cc, load_systematics_csv(args.cc_systematics))
    else:
        cc_cov = None
    bao_cov = load_covariance_csv(args.bao_covariance, len(bao)) if args.bao_covariance else None

    out = GPClockGeometryClosureAuditor(cc, bao, cc_cov, bao_cov).reconstruct(
        n_draws=args.draws,
        seed=args.seed,
        rd_mpc=args.rd,
        rd_sigma_mpc=args.rd_sigma,
        method=args.method,
    )
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
