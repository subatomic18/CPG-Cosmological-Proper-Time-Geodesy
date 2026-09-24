"""CPG late-onset redshift-remapping closure test.

This module follows the constant-redshift stress test with a redshift-dependent
phenomenological profile

    epsilon(z) = epsilon0 / [1 + (z / z_t)^n]

but applies it to the *differential accumulated redshift* rather than simply
substituting epsilon(z) into a constant-exponent formula:

    d ln(1+z_obs) = [1 + epsilon(z_geom)] d ln(1+z_geom).

Therefore

    ln[(1+z_obs)/(1+z_geom)]
      = integral_0^z_geom epsilon(z) d ln(1+z).

This distinction matters.  Even when epsilon(z) tends to zero at high redshift,
a positive low-redshift contribution leaves a finite accumulated offset at all
higher redshifts unless a later contribution of opposite sign cancels it.

The module is deliberately phenomenological.  It does not derive epsilon(z)
from Einstein's equations or from RTD-EU.  It asks a falsification-oriented
question: can a smooth positive late-onset mapping be large enough across the
SH0ES Hubble-flow range to account for the late/early H0 normalization split
while leaving the high-redshift/CMB mapping essentially unchanged?
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
from typing import Iterable, Sequence

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


DEFAULT_HUBBLE_FLOW_Z = (0.0233, 0.05, 0.10, 0.15)


@dataclass(frozen=True)
class LateOnsetCandidate:
    z_transition: float
    sharpness: float
    epsilon0: float
    target_ratio: float
    hubble_ratios: tuple[float, ...]
    hubble_rms_fractional_error: float
    hubble_max_fractional_error: float
    cmb_fractional_1plusz_shift: float
    z14_geometric: float
    zstar_observed_if_geometric_standard: float
    zstar_geometric_if_observed_standard: float

    def to_dict(self) -> dict:
        return {
            "z_transition": self.z_transition,
            "sharpness_n": self.sharpness,
            "epsilon0": self.epsilon0,
            "epsilon0_percent": 100.0 * self.epsilon0,
            "target_H0_ratio": self.target_ratio,
            "hubble_flow_ratios": list(self.hubble_ratios),
            "hubble_rms_fractional_error": self.hubble_rms_fractional_error,
            "hubble_max_fractional_error": self.hubble_max_fractional_error,
            "cmb_fractional_1plusz_shift": self.cmb_fractional_1plusz_shift,
            "cmb_percent_1plusz_shift": 100.0 * self.cmb_fractional_1plusz_shift,
            "z14_32_geometric": self.z14_geometric,
            "zstar_observed_if_geometric_standard": self.zstar_observed_if_geometric_standard,
            "zstar_geometric_if_observed_standard": self.zstar_geometric_if_observed_standard,
        }


@dataclass(frozen=True)
class LateOnsetScanResult:
    h0_late: float
    h0_early: float
    epsilon0: float
    target_ratio: float
    hubble_flow_redshifts: tuple[float, ...]
    allowed_hubble_max_error: float
    positive_mapping_high_z_floor: float
    positive_mapping_floor_pivot_z: float
    best: LateOnsetCandidate | None

    def to_dict(self) -> dict:
        return {
            "H0_late_km_s_Mpc": self.h0_late,
            "H0_early_km_s_Mpc": self.h0_early,
            "epsilon0": self.epsilon0,
            "epsilon0_percent": 100.0 * self.epsilon0,
            "target_H0_ratio": self.target_ratio,
            "hubble_flow_redshifts": list(self.hubble_flow_redshifts),
            "allowed_hubble_max_fractional_error": self.allowed_hubble_max_error,
            "positive_mapping_high_z_floor": self.positive_mapping_high_z_floor,
            "positive_mapping_high_z_floor_percent": 100.0 * self.positive_mapping_high_z_floor,
            "positive_mapping_floor_pivot_z": self.positive_mapping_floor_pivot_z,
            "best_candidate": None if self.best is None else self.best.to_dict(),
            "scope_note": (
                "Phenomenological positive late-onset accumulated-redshift stress test. "
                "Not a physical RTD-EU derivation and not a full SH0ES/CMB joint likelihood."
            ),
        }


def epsilon0_from_h0(h0_late: float, h0_early: float) -> float:
    h0_late = float(h0_late)
    h0_early = float(h0_early)
    if h0_late <= 0.0 or h0_early <= 0.0:
        raise ValueError("H0 values must be positive")
    return h0_late / h0_early - 1.0


def epsilon_profile(z: float, epsilon0: float, z_transition: float, sharpness: float) -> float:
    """Smooth positive late-onset profile, evaluated stably for large powers."""
    z = float(z)
    epsilon0 = float(epsilon0)
    z_transition = float(z_transition)
    sharpness = float(sharpness)
    if z < 0.0:
        raise ValueError("z must be non-negative")
    if z_transition <= 0.0 or sharpness <= 0.0:
        raise ValueError("z_transition and sharpness must be positive")
    if z == 0.0:
        return epsilon0
    exponent = sharpness * math.log(z / z_transition)
    if exponent >= 50.0:
        return epsilon0 * math.exp(-exponent)
    if exponent <= -50.0:
        return epsilon0
    return epsilon0 / (1.0 + math.exp(exponent))


def accumulated_log_shift(
    z_geom: float,
    epsilon0: float,
    z_transition: float,
    sharpness: float,
) -> float:
    """Return integral epsilon(z) d ln(1+z) from 0 to z_geom."""
    z_geom = float(z_geom)
    if z_geom < 0.0:
        raise ValueError("z_geom must be non-negative")
    xmax = math.log1p(z_geom)
    value, _ = quad(
        lambda x: epsilon_profile(
            math.expm1(x), epsilon0, z_transition, sharpness
        ),
        0.0,
        xmax,
        epsabs=1e-11,
        epsrel=1e-9,
        limit=200,
    )
    return float(value)


def observed_redshift(
    z_geom: float,
    epsilon0: float,
    z_transition: float,
    sharpness: float,
) -> float:
    log1p_obs = math.log1p(float(z_geom)) + accumulated_log_shift(
        z_geom, epsilon0, z_transition, sharpness
    )
    return float(math.expm1(log1p_obs))


def geometric_redshift(
    z_obs: float,
    epsilon0: float,
    z_transition: float,
    sharpness: float,
) -> float:
    z_obs = float(z_obs)
    if z_obs < 0.0:
        raise ValueError("z_obs must be non-negative")
    if z_obs == 0.0:
        return 0.0
    return float(
        brentq(
            lambda z: observed_redshift(
                z, epsilon0, z_transition, sharpness
            )
            - z_obs,
            0.0,
            z_obs,
            xtol=1e-11,
            rtol=1e-10,
            maxiter=100,
        )
    )


def positive_mapping_high_z_floor(target_ratio: float, pivot_z_obs: float) -> float:
    """General lower bound for a positive accumulated remapping.

    If the mapping has already produced z_obs/z_geom = target_ratio at a
    finite pivot redshift, and the accumulated correction never decreases
    afterwards, then every higher redshift retains at least

        (1+z_obs)/(1+z_geom) - 1

    with z_geom = z_obs / target_ratio at the pivot.
    """
    target_ratio = float(target_ratio)
    pivot_z_obs = float(pivot_z_obs)
    if target_ratio <= 0.0 or pivot_z_obs < 0.0:
        raise ValueError("invalid ratio or pivot redshift")
    z_geom = pivot_z_obs / target_ratio
    return float((1.0 + pivot_z_obs) / (1.0 + z_geom) - 1.0)


def evaluate_candidate(
    h0_late: float,
    h0_early: float,
    z_transition: float,
    sharpness: float,
    hubble_flow_redshifts: Iterable[float] = DEFAULT_HUBBLE_FLOW_Z,
    z_star: float = 1089.938,
) -> LateOnsetCandidate:
    ratio = float(h0_late) / float(h0_early)
    epsilon0 = ratio - 1.0
    zs = tuple(float(z) for z in hubble_flow_redshifts)

    hubble_ratios = []
    for z_obs in zs:
        z_geom = geometric_redshift(
            z_obs, epsilon0, z_transition, sharpness
        )
        hubble_ratios.append(z_obs / z_geom)

    arr = np.asarray(hubble_ratios, dtype=float)
    frac = arr / ratio - 1.0
    rms = float(np.sqrt(np.mean(frac * frac)))
    maxerr = float(np.max(np.abs(frac)))

    zstar_obs = observed_redshift(
        z_star, epsilon0, z_transition, sharpness
    )
    cmb_shift = float((1.0 + zstar_obs) / (1.0 + z_star) - 1.0)
    zstar_geom = geometric_redshift(
        z_star, epsilon0, z_transition, sharpness
    )
    z14_geom = geometric_redshift(
        14.32, epsilon0, z_transition, sharpness
    )

    return LateOnsetCandidate(
        z_transition=float(z_transition),
        sharpness=float(sharpness),
        epsilon0=epsilon0,
        target_ratio=ratio,
        hubble_ratios=tuple(float(x) for x in hubble_ratios),
        hubble_rms_fractional_error=rms,
        hubble_max_fractional_error=maxerr,
        cmb_fractional_1plusz_shift=cmb_shift,
        z14_geometric=z14_geom,
        zstar_observed_if_geometric_standard=zstar_obs,
        zstar_geometric_if_observed_standard=zstar_geom,
    )


def scan_late_onset(
    h0_late: float,
    h0_early: float,
    hubble_flow_redshifts: Iterable[float] = DEFAULT_HUBBLE_FLOW_Z,
    allowed_hubble_max_error: float = 0.01,
    z_star: float = 1089.938,
    z_transition_grid: Sequence[float] | None = None,
    sharpness_grid: Sequence[float] = (1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0),
) -> LateOnsetScanResult:
    """Find the smallest CMB imprint among profiles that preserve the H0 shift.

    A candidate is retained only if its effective z_obs/z_geom ratio stays
    within ``allowed_hubble_max_error`` of H0_late/H0_early at every supplied
    Hubble-flow test redshift.
    """
    zs = tuple(float(z) for z in hubble_flow_redshifts)
    if not zs:
        raise ValueError("at least one Hubble-flow redshift is required")
    if z_transition_grid is None:
        z_transition_grid = np.logspace(-2.2, -0.1, 120)

    best: LateOnsetCandidate | None = None
    for n in sharpness_grid:
        for zt in z_transition_grid:
            candidate = evaluate_candidate(
                h0_late=h0_late,
                h0_early=h0_early,
                z_transition=float(zt),
                sharpness=float(n),
                hubble_flow_redshifts=zs,
                z_star=z_star,
            )
            if candidate.hubble_max_fractional_error > allowed_hubble_max_error:
                continue
            if best is None or (
                candidate.cmb_fractional_1plusz_shift
                < best.cmb_fractional_1plusz_shift
            ):
                best = candidate

    ratio = float(h0_late) / float(h0_early)
    pivot = max(zs)
    floor = positive_mapping_high_z_floor(ratio, pivot)
    return LateOnsetScanResult(
        h0_late=float(h0_late),
        h0_early=float(h0_early),
        epsilon0=ratio - 1.0,
        target_ratio=ratio,
        hubble_flow_redshifts=zs,
        allowed_hubble_max_error=float(allowed_hubble_max_error),
        positive_mapping_high_z_floor=floor,
        positive_mapping_floor_pivot_z=pivot,
        best=best,
    )


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CPG positive late-onset accumulated-redshift closure scan."
    )
    p.add_argument("--h0-late", type=float, required=True)
    p.add_argument("--h0-early", type=float, required=True)
    p.add_argument("--max-hubble-error", type=float, default=0.01)
    p.add_argument("--z-star", type=float, default=1089.938)
    p.add_argument("--json", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = scan_late_onset(
        h0_late=args.h0_late,
        h0_early=args.h0_early,
        allowed_hubble_max_error=args.max_hubble_error,
        z_star=args.z_star,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    print("CPG late-onset accumulated-redshift closure test")
    print(f"H0 late / early      : {result.target_ratio:.8f}")
    print(f"epsilon0              : {result.epsilon0:.8f} ({100*result.epsilon0:.3f}%)")
    print(
        f"positive high-z floor : {100*result.positive_mapping_high_z_floor:.3f}% "
        f"in (1+z), if the full H0 ratio persists through z={result.positive_mapping_floor_pivot_z:.3f}"
    )
    print(
        f"scan requirement      : max Hubble-flow ratio error <= "
        f"{100*result.allowed_hubble_max_error:.2f}%"
    )

    if result.best is None:
        print("No candidate in the scan satisfied the Hubble-flow requirement.")
        return 2

    b = result.best
    print("\nBest positive late-onset candidate in scan")
    print(f"z_transition          : {b.z_transition:.6f}")
    print(f"sharpness n            : {b.sharpness:.1f}")
    print(f"Hubble RMS error       : {100*b.hubble_rms_fractional_error:.3f}%")
    print(f"Hubble max error       : {100*b.hubble_max_fractional_error:.3f}%")
    for z, r in zip(result.hubble_flow_redshifts, b.hubble_ratios):
        print(f"  z_obs={z:.4f}: effective z_obs/z_geom={r:.7f}")
    print(f"CMB (1+z) shift        : {100*b.cmb_fractional_1plusz_shift:.3f}%")
    print(f"z=14.32 -> z_geom      : {b.z14_geometric:.5f}")
    print(
        f"z*_geom={args.z_star:.3f} -> z*_obs={b.zstar_observed_if_geometric_standard:.3f}"
    )
    print(
        f"z*_obs={args.z_star:.3f} -> z*_geom={b.zstar_geometric_if_observed_standard:.3f}"
    )
    print("\nInterpretation:")
    print(
        "Making epsilon(z) switch off at high redshift removes the runaway constant-"
        "epsilon behavior, but a positive accumulated late-time correction leaves a "
        "finite high-z offset. To remove that offset completely requires cancellation "
        "from an opposite-sign contribution or a different, non-accumulated mechanism."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
