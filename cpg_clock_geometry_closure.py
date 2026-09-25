#!/usr/bin/env python3
"""Clock-geometry closure diagnostic for Cosmological Proper-Time Geodesy (CPG).

This module compares radial BAO geometry with cosmic-chronometer clock
measurements without requiring an absolute sound-horizon calibration.

Radial BAO measures D_H/r_d, where D_H = c/H_geom. Cosmic chronometers measure
H_CC = -(1+z)^(-1) dz/dtau. For the CPG mapping

    d tau / dz = -Gamma(z) / [(1+z) H_geom(z)],

the directly reconstructed quantity is

    Q(z) = Gamma(z) r_d
         = c / { [D_H(z)/r_d] H_CC(z) }.

A redshift-independent Q is therefore the sound-horizon-degenerate null test
for dGamma/dz = 0. Supplying an external r_d is optional and only converts Q
to an absolute Gamma. A closure residual is a diagnostic and is not, by itself,
evidence for RTD-EU or any non-standard cosmology.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy import interpolate
from scipy.stats import chi2

from cpg_chronometers import (
    CCPoint,
    build_covariance,
    load_cc_csv,
    load_systematics_csv,
)

C_LIGHT_KM_S = 299792.458


@dataclass(frozen=True)
class BAOPoint:
    z: float
    DH_over_rd: float
    sigma_DH_over_rd: float
    reference: str = ""


def load_bao_csv(path: str | Path) -> list[BAOPoint]:
    """Load radial BAO D_H/r_d measurements from CSV.

    Required columns are ``z,DH_over_rd,sigma_DH_over_rd``. An optional
    ``reference`` column is retained as metadata.
    """
    out: list[BAOPoint] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(
                BAOPoint(
                    z=float(row["z"]),
                    DH_over_rd=float(row["DH_over_rd"]),
                    sigma_DH_over_rd=float(row["sigma_DH_over_rd"]),
                    reference=row.get("reference", ""),
                )
            )
    if len(out) < 2:
        raise ValueError("Need at least two radial BAO points")
    z = np.array([p.z for p in out], dtype=float)
    if np.any(np.diff(z) <= 0):
        raise ValueError("BAO redshifts must be strictly increasing")
    if any(p.DH_over_rd <= 0 or p.sigma_DH_over_rd <= 0 for p in out):
        raise ValueError("D_H/r_d and its uncertainty must be positive")
    return out


def load_covariance_csv(path: str | Path, expected_size: int) -> np.ndarray:
    """Load a plain numeric covariance matrix from CSV."""
    C = np.loadtxt(path, delimiter=",", dtype=float)
    C = np.atleast_2d(C)
    if C.shape != (expected_size, expected_size):
        raise ValueError(
            f"Covariance shape {C.shape} does not match "
            f"({expected_size}, {expected_size})"
        )
    return C


def _validate_covariance(C: np.ndarray, n: int, label: str) -> np.ndarray:
    C = np.asarray(C, dtype=float)
    if C.shape != (n, n):
        raise ValueError(f"{label} covariance shape mismatch")
    if not np.allclose(C, C.T, atol=1e-10, rtol=1e-10):
        raise ValueError(f"{label} covariance must be symmetric")
    if np.linalg.eigvalsh(C).min() < -1e-8:
        raise ValueError(f"{label} covariance must be positive semidefinite")
    return C


class ClockGeometryClosureAuditor:
    """Covariance-aware CPG clock-versus-geometry closure test.

    Cosmic-chronometer H(z) is reconstructed at the radial-BAO redshifts with
    Monte-Carlo PCHIP interpolation. The supplied CC covariance is propagated
    by multivariate Gaussian draws. BAO covariance is propagated in the same
    draw-level calculation.

    The primary observable is Q(z)=Gamma(z) r_d, so the shape test does not
    require an external value of r_d.
    """

    def __init__(
        self,
        cc_points: Sequence[CCPoint],
        bao_points: Sequence[BAOPoint],
        cc_covariance: Optional[np.ndarray] = None,
        bao_covariance: Optional[np.ndarray] = None,
    ):
        self.cc_points = list(cc_points)
        self.bao_points = list(bao_points)
        if len(self.cc_points) < 3:
            raise ValueError("Need at least three cosmic-chronometer points")
        if len(self.bao_points) < 2:
            raise ValueError("Need at least two radial BAO points")

        self.cc_z = np.array([p.z for p in self.cc_points], dtype=float)
        self.cc_H = np.array([p.H for p in self.cc_points], dtype=float)
        self.bao_z = np.array([p.z for p in self.bao_points], dtype=float)
        self.bao_DHrd = np.array(
            [p.DH_over_rd for p in self.bao_points], dtype=float
        )

        if np.any(np.diff(self.cc_z) <= 0):
            raise ValueError("CC redshifts must be strictly increasing")
        if np.any(np.diff(self.bao_z) <= 0):
            raise ValueError("BAO redshifts must be strictly increasing")
        if self.bao_z[0] < self.cc_z[0] or self.bao_z[-1] > self.cc_z[-1]:
            raise ValueError(
                "BAO redshifts must lie inside the measured CC redshift range; "
                "the closure diagnostic does not extrapolate H_CC(z)"
            )

        if cc_covariance is None:
            cc_sig = np.array(
                [p.sigma_H for p in self.cc_points], dtype=float
            )
            cc_covariance = np.diag(cc_sig**2)
        if bao_covariance is None:
            bao_sig = np.array(
                [p.sigma_DH_over_rd for p in self.bao_points], dtype=float
            )
            bao_covariance = np.diag(bao_sig**2)

        self.cc_C = _validate_covariance(
            cc_covariance, len(self.cc_points), "CC"
        )
        self.bao_C = _validate_covariance(
            bao_covariance, len(self.bao_points), "BAO"
        )

    @staticmethod
    def _gls_constant(
        y: np.ndarray, C: np.ndarray
    ) -> tuple[float, float, float]:
        """Return GLS constant, one-sigma uncertainty, and chi-square."""
        one = np.ones(len(y), dtype=float)
        W = np.linalg.pinv(C, hermitian=True)
        denom = float(one @ W @ one)
        if denom <= 0:
            raise ValueError("Cannot fit constant: non-positive GLS information")
        y0 = float((one @ W @ y) / denom)
        sigma0 = float(np.sqrt(1.0 / denom))
        d = y - y0
        chisq = float(d @ W @ d)
        return y0, sigma0, chisq

    def reconstruct(
        self,
        n_draws: int = 8000,
        seed: int = 42,
        rd_mpc: Optional[float] = None,
        rd_sigma_mpc: float = 0.0,
    ) -> dict:
        """Reconstruct Q(z)=Gamma(z)r_d and test its redshift constancy."""
        if int(n_draws) < 500:
            raise ValueError("n_draws must be at least 500")
        if rd_mpc is not None and rd_mpc <= 0:
            raise ValueError("rd_mpc must be positive")
        if rd_sigma_mpc < 0:
            raise ValueError("rd_sigma_mpc must be non-negative")
        if rd_mpc is None and rd_sigma_mpc != 0:
            raise ValueError("rd_sigma_mpc requires rd_mpc")

        rng = np.random.default_rng(seed)
        cc_draws = rng.multivariate_normal(
            self.cc_H, self.cc_C, size=int(n_draws)
        )
        bao_draws = rng.multivariate_normal(
            self.bao_DHrd, self.bao_C, size=int(n_draws)
        )

        h_eval = []
        dh_eval = []
        for cc_h, bao_dh in zip(cc_draws, bao_draws):
            if np.any(cc_h <= 0) or np.any(bao_dh <= 0):
                continue
            spl = interpolate.PchipInterpolator(
                self.cc_z, cc_h, extrapolate=False
            )
            h = np.asarray(spl(self.bao_z), dtype=float)
            if np.any(~np.isfinite(h)) or np.any(h <= 0):
                continue
            h_eval.append(h)
            dh_eval.append(bao_dh)

        if len(h_eval) < max(250, int(0.5 * n_draws)):
            raise RuntimeError("Too many nonphysical Monte-Carlo draws")

        h_eval = np.asarray(h_eval, dtype=float)
        dh_eval = np.asarray(dh_eval, dtype=float)
        q_draws = C_LIGHT_KM_S / (dh_eval * h_eval)

        h16, h50, h84 = np.percentile(h_eval, [16, 50, 84], axis=0)
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
        s16, s50, s84 = np.percentile(
            shape_draws, [16, 50, 84], axis=0
        )
        q0_d16, q0_d50, q0_d84 = np.percentile(
            q0_draws, [16, 50, 84]
        )

        points = []
        for j, p in enumerate(self.bao_points):
            points.append(
                {
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
                }
            )

        result = {
            "observable": "Q(z) = Gamma(z) r_d",
            "null_hypothesis": (
                "Q(z) is constant in redshift (dGamma/dz = 0)"
            ),
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
                rd_all = rng.normal(
                    float(rd_mpc), float(rd_sigma_mpc), len(q_draws)
                )
                good = rd_all > 0
                rd_draws = rd_all[good]
                q_for_gamma = q_draws[good]
                q0_for_gamma = q0_draws[good]

            gamma_draws = q_for_gamma / rd_draws[:, None]
            g16, g50, g84 = np.percentile(
                gamma_draws, [16, 50, 84], axis=0
            )
            for j, row in enumerate(result["points"]):
                row["Gamma"] = float(g50[j])
                row["Gamma_lo"] = float(g16[j])
                row["Gamma_hi"] = float(g84[j])

            gamma0_draws = q0_for_gamma / rd_draws
            g0_16, g0_50, g0_84 = np.percentile(
                gamma0_draws, [16, 50, 84]
            )
            result["rd_calibration"] = {
                "rd_mpc": float(rd_mpc),
                "rd_sigma_mpc": float(rd_sigma_mpc),
                "Gamma0": float(g0_50),
                "Gamma0_lo": float(g0_16),
                "Gamma0_hi": float(g0_84),
            }

        return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "CPG clock-geometry closure: reconstruct Q(z)=Gamma(z) r_d "
            "from radial BAO and cosmic chronometers"
        )
    )
    parser.add_argument("--cc", required=True, help="cosmic-chronometer CSV")
    parser.add_argument("--bao", required=True, help="radial BAO CSV")
    parser.add_argument(
        "--cc-covariance",
        default=None,
        help="optional full numeric CC covariance CSV",
    )
    parser.add_argument(
        "--bao-covariance",
        default=None,
        help="optional full numeric BAO covariance CSV",
    )
    parser.add_argument(
        "--cc-systematics",
        default=None,
        help=(
            "optional Moresco-style systematics CSV used with "
            "build_covariance; ignored when --cc-covariance is supplied"
        ),
    )
    parser.add_argument("--draws", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rd",
        type=float,
        default=None,
        help="optional sound horizon r_d [Mpc] for absolute Gamma",
    )
    parser.add_argument("--rd-sigma", type=float, default=0.0)
    args = parser.parse_args()

    cc = load_cc_csv(args.cc)
    bao = load_bao_csv(args.bao)

    if args.cc_covariance:
        cc_cov = load_covariance_csv(args.cc_covariance, len(cc))
    elif args.cc_systematics:
        sys = load_systematics_csv(args.cc_systematics)
        cc_cov = build_covariance(cc, sys)
    else:
        cc_cov = None

    if args.bao_covariance:
        bao_cov = load_covariance_csv(args.bao_covariance, len(bao))
    else:
        bao_cov = None

    auditor = ClockGeometryClosureAuditor(
        cc,
        bao,
        cc_covariance=cc_cov,
        bao_covariance=bao_cov,
    )
    result = auditor.reconstruct(
        n_draws=args.draws,
        seed=args.seed,
        rd_mpc=args.rd,
        rd_sigma_mpc=args.rd_sigma,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
