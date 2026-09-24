"""CPG compensated accumulated-redshift stress test.

This module tests the next phenomenological possibility after a purely
positive late-onset mapping failed to close exactly at high redshift.

The differential mapping is

    d ln(1+z_obs) = [1 + epsilon(z)] d ln(1+z_geom)

with

    epsilon(z) = epsilon_pos(z) - A * G[ln(1+z)],

where epsilon_pos is the previously tested sharp late-onset positive profile
and G is a broad Gaussian compensation term in x = ln(1+z).  For every trial
Gaussian centre and width, A is solved so that the total accumulated shift at
recombination is exactly zero:

    integral_0^{z*} epsilon(z) d ln(1+z) = 0.

This is deliberately a mathematical closure experiment, not a derivation of
RTD-EU or a claim that such a sign-changing field exists physically.  The goal
is to determine what a compensating profile would have to look like if the
local H0 normalization is to be changed while the CMB redshift mapping closes
again by recombination.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


DEFAULT_HUBBLE_FLOW_Z = (0.0233, 0.05, 0.10, 0.15)
DEFAULT_POSITIVE_ZT = 0.112964
DEFAULT_POSITIVE_N = 24.0


@dataclass(frozen=True)
class CompensatedCandidate:
    compensation_center_z: float
    compensation_sigma_log1pz: float
    compensation_amplitude: float
    epsilon0: float
    target_ratio: float
    hubble_ratios: tuple[float, ...]
    hubble_rms_fractional_error: float
    hubble_max_fractional_error: float
    cmb_log_shift: float
    cmb_fractional_1plusz_shift: float
    maximum_accumulated_log_shift: float
    minimum_epsilon: float
    z1_geometric: float
    z2_geometric: float
    z6_geometric: float
    z14_geometric: float

    def to_dict(self) -> dict:
        return {
            "compensation_center_z": self.compensation_center_z,
            "compensation_sigma_log1pz": self.compensation_sigma_log1pz,
            "compensation_amplitude": self.compensation_amplitude,
            "compensation_amplitude_percent": 100.0 * self.compensation_amplitude,
            "epsilon0": self.epsilon0,
            "epsilon0_percent": 100.0 * self.epsilon0,
            "target_H0_ratio": self.target_ratio,
            "hubble_flow_ratios": list(self.hubble_ratios),
            "hubble_rms_fractional_error": self.hubble_rms_fractional_error,
            "hubble_max_fractional_error": self.hubble_max_fractional_error,
            "cmb_log_shift": self.cmb_log_shift,
            "cmb_fractional_1plusz_shift": self.cmb_fractional_1plusz_shift,
            "maximum_accumulated_log_shift": self.maximum_accumulated_log_shift,
            "maximum_accumulated_percent_approx": 100.0 * self.maximum_accumulated_log_shift,
            "minimum_epsilon": self.minimum_epsilon,
            "minimum_epsilon_percent": 100.0 * self.minimum_epsilon,
            "z1_geometric": self.z1_geometric,
            "z2_geometric": self.z2_geometric,
            "z6_geometric": self.z6_geometric,
            "z14_32_geometric": self.z14_geometric,
        }


@dataclass(frozen=True)
class CompensatedScanResult:
    h0_late: float
    h0_early: float
    epsilon0: float
    target_ratio: float
    positive_transition_z: float
    positive_sharpness: float
    allowed_hubble_max_error: float
    best: CompensatedCandidate | None

    def to_dict(self) -> dict:
        return {
            "H0_late_km_s_Mpc": self.h0_late,
            "H0_early_km_s_Mpc": self.h0_early,
            "epsilon0": self.epsilon0,
            "epsilon0_percent": 100.0 * self.epsilon0,
            "target_H0_ratio": self.target_ratio,
            "positive_transition_z": self.positive_transition_z,
            "positive_sharpness_n": self.positive_sharpness,
            "allowed_hubble_max_fractional_error": self.allowed_hubble_max_error,
            "best_candidate": None if self.best is None else self.best.to_dict(),
            "scope_note": (
                "Phenomenological sign-changing accumulated-redshift closure test only. "
                "The compensating term is imposed mathematically and is not derived from "
                "Einstein equations, matter dynamics, or RTD-EU."
            ),
        }


def epsilon0_from_h0(h0_late: float, h0_early: float) -> float:
    if h0_late <= 0.0 or h0_early <= 0.0:
        raise ValueError("H0 values must be positive")
    return float(h0_late / h0_early - 1.0)


def _positive_profile(z: float, epsilon0: float, zt: float, n: float) -> float:
    if z < 0.0 or zt <= 0.0 or n <= 0.0:
        raise ValueError("invalid profile input")
    if z == 0.0:
        return float(epsilon0)
    exponent = n * math.log(z / zt)
    if exponent >= 50.0:
        return float(epsilon0 * math.exp(-exponent))
    if exponent <= -50.0:
        return float(epsilon0)
    return float(epsilon0 / (1.0 + math.exp(exponent)))


def _gaussian_integral_x(xmax: float, mu: float, sigma: float) -> float:
    root2 = math.sqrt(2.0)
    pref = sigma * math.sqrt(math.pi / 2.0)
    return float(
        pref
        * (
            math.erf((xmax - mu) / (root2 * sigma))
            - math.erf((0.0 - mu) / (root2 * sigma))
        )
    )


def _positive_integral_x(
    xmax: float, epsilon0: float, zt: float, n: float
) -> float:
    value, _ = quad(
        lambda x: _positive_profile(math.expm1(x), epsilon0, zt, n),
        0.0,
        xmax,
        epsabs=2e-10,
        epsrel=2e-8,
        limit=150,
    )
    return float(value)


def compensation_amplitude_for_closure(
    epsilon0: float,
    center_z: float,
    sigma_log1pz: float,
    z_star: float = 1089.938,
    positive_zt: float = DEFAULT_POSITIVE_ZT,
    positive_n: float = DEFAULT_POSITIVE_N,
) -> float:
    if center_z <= 0.0 or sigma_log1pz <= 0.0:
        raise ValueError("compensation center and width must be positive")
    xstar = math.log1p(z_star)
    ipos = _positive_integral_x(xstar, epsilon0, positive_zt, positive_n)
    igauss = _gaussian_integral_x(
        xstar, math.log1p(center_z), sigma_log1pz
    )
    if igauss <= 0.0:
        raise ValueError("compensation Gaussian has zero support in integration range")
    return float(ipos / igauss)


def accumulated_log_shift(
    z_geom: float,
    epsilon0: float,
    center_z: float,
    sigma_log1pz: float,
    amplitude: float,
    positive_zt: float = DEFAULT_POSITIVE_ZT,
    positive_n: float = DEFAULT_POSITIVE_N,
) -> float:
    x = math.log1p(float(z_geom))
    ipos = _positive_integral_x(x, epsilon0, positive_zt, positive_n)
    igauss = _gaussian_integral_x(x, math.log1p(center_z), sigma_log1pz)
    return float(ipos - amplitude * igauss)


def epsilon_profile(
    z: float,
    epsilon0: float,
    center_z: float,
    sigma_log1pz: float,
    amplitude: float,
    positive_zt: float = DEFAULT_POSITIVE_ZT,
    positive_n: float = DEFAULT_POSITIVE_N,
) -> float:
    x = math.log1p(float(z))
    mu = math.log1p(center_z)
    g = math.exp(-0.5 * ((x - mu) / sigma_log1pz) ** 2)
    return float(
        _positive_profile(z, epsilon0, positive_zt, positive_n)
        - amplitude * g
    )


def observed_redshift(
    z_geom: float,
    epsilon0: float,
    center_z: float,
    sigma_log1pz: float,
    amplitude: float,
    positive_zt: float = DEFAULT_POSITIVE_ZT,
    positive_n: float = DEFAULT_POSITIVE_N,
) -> float:
    shift = accumulated_log_shift(
        z_geom,
        epsilon0,
        center_z,
        sigma_log1pz,
        amplitude,
        positive_zt,
        positive_n,
    )
    return float(math.expm1(math.log1p(z_geom) + shift))


def geometric_redshift(
    z_obs: float,
    epsilon0: float,
    center_z: float,
    sigma_log1pz: float,
    amplitude: float,
    positive_zt: float = DEFAULT_POSITIVE_ZT,
    positive_n: float = DEFAULT_POSITIVE_N,
) -> float:
    z_obs = float(z_obs)
    if z_obs == 0.0:
        return 0.0
    upper = max(z_obs * 1.25, z_obs + 0.1)
    return float(
        brentq(
            lambda zg: observed_redshift(
                zg,
                epsilon0,
                center_z,
                sigma_log1pz,
                amplitude,
                positive_zt,
                positive_n,
            )
            - z_obs,
            0.0,
            upper,
            xtol=1e-10,
            rtol=1e-9,
            maxiter=100,
        )
    )


def evaluate_candidate(
    h0_late: float,
    h0_early: float,
    center_z: float,
    sigma_log1pz: float,
    hubble_flow_redshifts: Iterable[float] = DEFAULT_HUBBLE_FLOW_Z,
    z_star: float = 1089.938,
    positive_zt: float = DEFAULT_POSITIVE_ZT,
    positive_n: float = DEFAULT_POSITIVE_N,
) -> CompensatedCandidate:
    ratio = float(h0_late / h0_early)
    epsilon0 = ratio - 1.0
    amplitude = compensation_amplitude_for_closure(
        epsilon0,
        center_z,
        sigma_log1pz,
        z_star=z_star,
        positive_zt=positive_zt,
        positive_n=positive_n,
    )

    zs = tuple(float(z) for z in hubble_flow_redshifts)
    ratios = []
    for zobs in zs:
        zg = geometric_redshift(
            zobs,
            epsilon0,
            center_z,
            sigma_log1pz,
            amplitude,
            positive_zt,
            positive_n,
        )
        ratios.append(zobs / zg)
    rarr = np.asarray(ratios, dtype=float)
    ferr = rarr / ratio - 1.0

    cmb_log = accumulated_log_shift(
        z_star,
        epsilon0,
        center_z,
        sigma_log1pz,
        amplitude,
        positive_zt,
        positive_n,
    )
    cmb_frac = math.expm1(cmb_log)

    grid_z = np.geomspace(1e-4, z_star, 180)
    shifts = [
        accumulated_log_shift(
            float(z),
            epsilon0,
            center_z,
            sigma_log1pz,
            amplitude,
            positive_zt,
            positive_n,
        )
        for z in grid_z
    ]
    eps_values = [
        epsilon_profile(
            float(z),
            epsilon0,
            center_z,
            sigma_log1pz,
            amplitude,
            positive_zt,
            positive_n,
        )
        for z in grid_z
    ]

    def zg(zo: float) -> float:
        return geometric_redshift(
            zo,
            epsilon0,
            center_z,
            sigma_log1pz,
            amplitude,
            positive_zt,
            positive_n,
        )

    return CompensatedCandidate(
        compensation_center_z=float(center_z),
        compensation_sigma_log1pz=float(sigma_log1pz),
        compensation_amplitude=float(amplitude),
        epsilon0=float(epsilon0),
        target_ratio=ratio,
        hubble_ratios=tuple(float(x) for x in ratios),
        hubble_rms_fractional_error=float(np.sqrt(np.mean(ferr * ferr))),
        hubble_max_fractional_error=float(np.max(np.abs(ferr))),
        cmb_log_shift=float(cmb_log),
        cmb_fractional_1plusz_shift=float(cmb_frac),
        maximum_accumulated_log_shift=float(np.max(np.abs(shifts))),
        minimum_epsilon=float(np.min(eps_values)),
        z1_geometric=zg(1.0),
        z2_geometric=zg(2.0),
        z6_geometric=zg(6.0),
        z14_geometric=zg(14.32),
    )


def scan_compensated_mapping(
    h0_late: float,
    h0_early: float,
    allowed_hubble_max_error: float = 0.01,
    hubble_flow_redshifts: Iterable[float] = DEFAULT_HUBBLE_FLOW_Z,
    z_star: float = 1089.938,
    positive_zt: float = DEFAULT_POSITIVE_ZT,
    positive_n: float = DEFAULT_POSITIVE_N,
    center_grid: Sequence[float] | None = None,
    sigma_grid: Sequence[float] | None = None,
) -> CompensatedScanResult:
    if center_grid is None:
        center_grid = np.geomspace(0.20, 6.0, 28)
    if sigma_grid is None:
        sigma_grid = np.linspace(0.08, 1.0, 20)

    best: CompensatedCandidate | None = None
    for center in center_grid:
        for sigma in sigma_grid:
            candidate = evaluate_candidate(
                h0_late,
                h0_early,
                float(center),
                float(sigma),
                hubble_flow_redshifts=hubble_flow_redshifts,
                z_star=z_star,
                positive_zt=positive_zt,
                positive_n=positive_n,
            )
            if candidate.hubble_max_fractional_error > allowed_hubble_max_error:
                continue
            # Primary objective: smallest compensation amplitude. Secondary:
            # smallest maximum accumulated distortion.
            if best is None or (
                candidate.compensation_amplitude,
                candidate.maximum_accumulated_log_shift,
            ) < (
                best.compensation_amplitude,
                best.maximum_accumulated_log_shift,
            ):
                best = candidate

    return CompensatedScanResult(
        h0_late=float(h0_late),
        h0_early=float(h0_early),
        epsilon0=epsilon0_from_h0(h0_late, h0_early),
        target_ratio=float(h0_late / h0_early),
        positive_transition_z=float(positive_zt),
        positive_sharpness=float(positive_n),
        allowed_hubble_max_error=float(allowed_hubble_max_error),
        best=best,
    )


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CPG compensated accumulated-redshift closure scan."
    )
    p.add_argument("--h0-late", type=float, required=True)
    p.add_argument("--h0-early", type=float, required=True)
    p.add_argument("--max-hubble-error", type=float, default=0.01)
    p.add_argument("--z-star", type=float, default=1089.938)
    p.add_argument("--json", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = scan_compensated_mapping(
        h0_late=args.h0_late,
        h0_early=args.h0_early,
        allowed_hubble_max_error=args.max_hubble_error,
        z_star=args.z_star,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    print("CPG compensated accumulated-redshift closure test")
    print(f"H0 late / early       : {result.target_ratio:.8f}")
    print(f"epsilon0               : {100*result.epsilon0:.3f}%")
    print(f"positive transition z  : {result.positive_transition_z:.6f}")
    print(f"positive sharpness n   : {result.positive_sharpness:.1f}")
    if result.best is None:
        print("No candidate satisfied the Hubble-flow tolerance.")
        return 2

    b = result.best
    print("\nBest compensation profile in scan")
    print(f"compensation center z  : {b.compensation_center_z:.5f}")
    print(f"sigma ln(1+z)          : {b.compensation_sigma_log1pz:.5f}")
    print(f"negative peak amplitude: {100*b.compensation_amplitude:.4f}%")
    print(f"minimum epsilon        : {100*b.minimum_epsilon:.4f}%")
    print(f"Hubble RMS error       : {100*b.hubble_rms_fractional_error:.3f}%")
    print(f"Hubble max error       : {100*b.hubble_max_fractional_error:.3f}%")
    for z, r in zip(DEFAULT_HUBBLE_FLOW_Z, b.hubble_ratios):
        print(f"  z_obs={z:.4f}: effective z_obs/z_geom={r:.7f}")
    print(f"maximum |log shift|    : {100*b.maximum_accumulated_log_shift:.3f}%")
    print(f"CMB log shift          : {b.cmb_log_shift:+.3e}")
    print(f"CMB (1+z) shift       : {100*b.cmb_fractional_1plusz_shift:+.6f}%")
    print("\nObserved -> geometric redshift")
    print(f"z=1.00  -> {b.z1_geometric:.5f}")
    print(f"z=2.00  -> {b.z2_geometric:.5f}")
    print(f"z=6.00  -> {b.z6_geometric:.5f}")
    print(f"z=14.32 -> {b.z14_geometric:.5f}")
    print("\nInterpretation:")
    print(
        "A sign-changing phenomenological profile can be constructed so the local "
        "normalization shift survives while the accumulated redshift correction "
        "closes again by recombination. The scientific question then becomes whether "
        "any physical metric/congruence dynamics can generate such a profile without "
        "violating intermediate-redshift SN, BAO, chronometer, lensing, and CMB data."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
