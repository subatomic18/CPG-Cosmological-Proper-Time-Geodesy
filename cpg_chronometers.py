#!/usr/bin/env python3
"""Cosmic-chronometer closure utilities for CPG.

Reconstructs elapsed time from tabulated CC H(z) measurements and compares
it with the CPG fiducial reference. A residual is a diagnostic, not by itself
evidence for RTD-EU.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy import integrate, interpolate

from cpg_v0_3 import FiducialCosmology, HUBBLE_CONVERSION_GYR


@dataclass(frozen=True)
class CCPoint:
    z: float
    H: float
    sigma_H: float
    reference: str = ""
    covariance_group: str = "independent"


def load_cc_csv(path: str | Path) -> list[CCPoint]:
    out: list[CCPoint] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(CCPoint(
                float(row["z"]),
                float(row["H_km_s_Mpc"]),
                float(row["sigma_H"]),
                row.get("reference", ""),
                row.get("covariance_group", "independent"),
            ))
    if len(out) < 2:
        raise ValueError("Need at least two cosmic-chronometer points")
    z = np.array([p.z for p in out])
    if np.any(np.diff(z) <= 0):
        raise ValueError("CC redshifts must be strictly increasing")
    if any(p.H <= 0 or p.sigma_H <= 0 for p in out):
        raise ValueError("H and sigma_H must be positive")
    return out


def load_systematics_csv(path: str | Path):
    a = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    required = {"z", "IMF_pct", "stlib_pct", "model_ooo_pct"}
    if not required.issubset(a.dtype.names or ()):
        raise ValueError(f"Systematics CSV requires columns {sorted(required)}")
    return a


def build_covariance(points: Sequence[CCPoint], systematics=None,
                     correlated_group="moresco_bc03") -> np.ndarray:
    """Build a CC covariance from quoted errors plus correlated model terms.

    The diagonal starts from the quoted sigma_H values. For points tagged with
    ``correlated_group``, optional redshift-dependent IMF, stellar-library and
    outlier-omitted SPS/model percentages are interpolated from the supplied
    Moresco-2020-style table and represented as fully correlated nuisance
    components (outer products).

    This is a transparent component-level reconstruction for CPG diagnostics;
    it should not be described as an exact reproduction of every covariance
    contribution in the original heterogeneous literature compilation.
    """
    H = np.array([p.H for p in points], dtype=float)
    sig = np.array([p.sigma_H for p in points], dtype=float)
    C = np.diag(sig**2)
    if systematics is None:
        return C

    idx = np.array([i for i, p in enumerate(points)
                    if p.covariance_group == correlated_group])
    if len(idx) == 0:
        return C

    zsys = np.asarray(systematics["z"], float)
    zpts = np.array([points[i].z for i in idx])
    for name in ("IMF_pct", "stlib_pct", "model_ooo_pct"):
        pct = np.interp(
            zpts, zsys, np.asarray(systematics[name], float),
            left=float(systematics[name][0]),
            right=float(systematics[name][-1]),
        )
        amp = H[idx] * pct / 100.0
        C[np.ix_(idx, idx)] += np.outer(amp, amp)
    return C


class ChronometerClosureAuditor:
    def __init__(self, points: Sequence[CCPoint], covariance=None, cosmo=None):
        self.points = list(points)
        self.z = np.array([p.z for p in points], dtype=float)
        self.H = np.array([p.H for p in points], dtype=float)
        self.cosmo = cosmo or FiducialCosmology()

        if covariance is None:
            covariance = np.diag(np.array([p.sigma_H for p in points], dtype=float)**2)
        self.C = np.asarray(covariance, dtype=float)
        if self.C.shape != (len(points), len(points)):
            raise ValueError("Covariance shape mismatch")
        if not np.allclose(self.C, self.C.T, atol=1e-10):
            raise ValueError("Covariance must be symmetric")
        if np.linalg.eigvalsh(self.C).min() < -1e-8:
            raise ValueError("Covariance must be positive semidefinite")

    def reference_elapsed(self, z_end: float, z_start: Optional[float] = None) -> float:
        z0 = self.z[0] if z_start is None else float(z_start)
        if z_end <= z0:
            raise ValueError("Require z_end > z_start")
        f = lambda z: HUBBLE_CONVERSION_GYR / ((1.0 + z) * self.cosmo.H(z))
        return float(integrate.quad(f, z0, z_end, epsabs=1e-11, epsrel=1e-11)[0])

    @staticmethod
    def _elapsed_from_curve(zgrid, Hgrid, z_start, z_end):
        if np.any(Hgrid <= 0):
            return np.nan
        spl = interpolate.PchipInterpolator(zgrid, Hgrid, extrapolate=False)
        grid = np.linspace(z_start, z_end, 1200)
        y = HUBBLE_CONVERSION_GYR / ((1.0 + grid) * spl(grid))
        return float(integrate.trapezoid(y, grid))

    def reconstruct(self, z_eval, n_draws=4000, seed=42):
        """Monte-Carlo PCHIP reconstruction using the supplied H(z) covariance.

        Returns medians and central 68% intervals for elapsed time, residual
        DeltaT = t_CC - t_ref, and closure C = t_CC/t_ref. No extrapolation is
        allowed outside the measured CC redshift range.
        """
        ze = np.asarray(z_eval, dtype=float)
        z0 = self.z[0]
        if np.any(ze <= z0) or np.any(ze > self.z[-1]):
            raise ValueError(f"z_eval must satisfy {z0} < z <= {self.z[-1]}")

        rng = np.random.default_rng(seed)
        draws = rng.multivariate_normal(self.H, self.C, size=int(n_draws))
        vals = np.full((len(draws), len(ze)), np.nan)
        for i, hd in enumerate(draws):
            if np.any(hd <= 0):
                continue
            for j, z1 in enumerate(ze):
                vals[i, j] = self._elapsed_from_curve(self.z, hd, z0, float(z1))

        vals = vals[np.all(np.isfinite(vals), axis=1)]
        if len(vals) < max(100, int(0.5 * n_draws)):
            raise RuntimeError("Too many nonphysical H(z) Monte-Carlo draws")

        q16, q50, q84 = np.percentile(vals, [16, 50, 84], axis=0)
        tref = np.array([self.reference_elapsed(float(z)) for z in ze])
        r16, r50, r84 = np.percentile(vals - tref[None, :], [16, 50, 84], axis=0)
        c16, c50, c84 = np.percentile(vals / tref[None, :], [16, 50, 84], axis=0)

        return [{
            "z": float(ze[j]),
            "t_cc_gyr": float(q50[j]),
            "t_cc_lo_gyr": float(q16[j]),
            "t_cc_hi_gyr": float(q84[j]),
            "t_ref_gyr": float(tref[j]),
            "deltaT_gyr": float(r50[j]),
            "deltaT_lo_gyr": float(r16[j]),
            "deltaT_hi_gyr": float(r84[j]),
            "C": float(c50[j]),
            "C_lo": float(c16[j]),
            "C_hi": float(c84[j]),
            "accepted_draws": int(len(vals)),
            "requested_draws": int(n_draws),
        } for j in range(len(ze))]

    def chi2_reference(self):
        model = np.array([self.cosmo.H(z) for z in self.z])
        d = self.H - model
        W = np.linalg.pinv(self.C, hermitian=True)
        return float(d @ W @ d)
