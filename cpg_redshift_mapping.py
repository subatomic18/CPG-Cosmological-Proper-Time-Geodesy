"""CPG one-parameter redshift-mapping stress test.

This module tests a deliberately simple phenomenological mapping

    ln(1 + z_obs) = (1 + epsilon) ln(1 + z_geom)

or equivalently

    1 + z_obs = (1 + z_geom)**(1 + epsilon).

It is *not* a derivation of RTD-EU physics.  It is a controlled stress test:
if one coherent redshift-accumulation factor is large enough to turn the
standard late-universe SH0ES H0 into the standard early-universe Planck H0,
what else would that same mapping imply?

Near z=0,

    z_obs ~= (1 + epsilon) z_geom,

so an H0 inferred directly from observed redshift would be larger than the
underlying geometric H0 by the same factor.  The module therefore solves

    epsilon = H0_late / H0_early - 1

and propagates that epsilon to selected redshifts, including recombination.

The point is falsification-oriented: a constant epsilon that fixes one part of
the Hubble tension may create an unacceptable distortion elsewhere.  A
failure of this constant model does not rule out a redshift-dependent or
metric-derived mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
from typing import Iterable, Sequence


@dataclass(frozen=True)
class RedshiftMappingResult:
    h0_late: float
    h0_early: float
    epsilon: float
    z_star_standard: float
    z_star_observed_if_geometric_standard: float
    z_star_geometric_if_observed_standard: float
    sample_rows: tuple[dict[str, float], ...]

    @property
    def local_scale_factor(self) -> float:
        """z_geom/z_obs in the z -> 0 limit."""
        return 1.0 / (1.0 + self.epsilon)

    @property
    def cmb_observed_ratio(self) -> float:
        """(1+z_obs,pred)/(1+z_star,standard) for standard geometric z*."""
        return (
            1.0 + self.z_star_observed_if_geometric_standard
        ) / (1.0 + self.z_star_standard)

    def to_dict(self) -> dict:
        return {
            "model": "ln(1+z_obs)=(1+epsilon)ln(1+z_geom)",
            "H0_late_km_s_Mpc": self.h0_late,
            "H0_early_km_s_Mpc": self.h0_early,
            "epsilon": self.epsilon,
            "epsilon_percent": 100.0 * self.epsilon,
            "local_zgeom_over_zobs": self.local_scale_factor,
            "z_star_standard": self.z_star_standard,
            "z_star_observed_if_geometric_standard": (
                self.z_star_observed_if_geometric_standard
            ),
            "z_star_geometric_if_observed_standard": (
                self.z_star_geometric_if_observed_standard
            ),
            "cmb_1plusz_ratio": self.cmb_observed_ratio,
            "samples": list(self.sample_rows),
            "scope_note": (
                "Phenomenological constant-epsilon stress test only; not a physical "
                "RTD-EU derivation and not a replacement for a full light-cone fit."
            ),
        }


def observed_redshift(z_geom: float, epsilon: float) -> float:
    """Map geometric/scale-factor redshift to observed redshift."""
    z_geom = float(z_geom)
    epsilon = float(epsilon)
    if z_geom < 0.0:
        raise ValueError("redshift must be non-negative")
    if 1.0 + epsilon <= 0.0:
        raise ValueError("epsilon must satisfy 1 + epsilon > 0")
    return math.expm1((1.0 + epsilon) * math.log1p(z_geom))


def geometric_redshift(z_obs: float, epsilon: float) -> float:
    """Invert the mapping and return geometric/scale-factor redshift."""
    z_obs = float(z_obs)
    epsilon = float(epsilon)
    if z_obs < 0.0:
        raise ValueError("redshift must be non-negative")
    if 1.0 + epsilon <= 0.0:
        raise ValueError("epsilon must satisfy 1 + epsilon > 0")
    return math.expm1(math.log1p(z_obs) / (1.0 + epsilon))


def epsilon_from_h0(h0_late: float, h0_early: float) -> float:
    """Constant epsilon required by the local Hubble-law limit."""
    h0_late = float(h0_late)
    h0_early = float(h0_early)
    if h0_late <= 0.0 or h0_early <= 0.0:
        raise ValueError("H0 values must be positive")
    return h0_late / h0_early - 1.0


def run_stress_test(
    h0_late: float,
    h0_early: float,
    z_star: float = 1089.938,
    sample_redshifts: Iterable[float] = (0.0233, 0.05, 0.10, 0.15, 14.32),
) -> RedshiftMappingResult:
    """Solve epsilon from the H0 split and propagate it across redshift."""
    epsilon = epsilon_from_h0(h0_late, h0_early)
    z_star_obs = observed_redshift(z_star, epsilon)
    z_star_geom = geometric_redshift(z_star, epsilon)

    rows: list[dict[str, float]] = []
    for z_obs in sample_redshifts:
        z_obs = float(z_obs)
        z_geom = geometric_redshift(z_obs, epsilon)
        rows.append(
            {
                "z_obs": z_obs,
                "z_geom": z_geom,
                "z_geom_over_z_obs": (z_geom / z_obs) if z_obs != 0.0 else 1.0,
                "fractional_reduction": ((z_obs - z_geom) / z_obs)
                if z_obs != 0.0
                else 0.0,
            }
        )

    return RedshiftMappingResult(
        h0_late=float(h0_late),
        h0_early=float(h0_early),
        epsilon=epsilon,
        z_star_standard=float(z_star),
        z_star_observed_if_geometric_standard=z_star_obs,
        z_star_geometric_if_observed_standard=z_star_geom,
        sample_rows=tuple(rows),
    )


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CPG constant log-redshift mapping stress test."
    )
    p.add_argument("--h0-late", type=float, required=True)
    p.add_argument("--h0-early", type=float, required=True)
    p.add_argument("--z-star", type=float, default=1089.938)
    p.add_argument(
        "--sample-z",
        type=float,
        nargs="*",
        default=[0.0233, 0.05, 0.10, 0.15, 14.32],
    )
    p.add_argument("--json", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_stress_test(
        h0_late=args.h0_late,
        h0_early=args.h0_early,
        z_star=args.z_star,
        sample_redshifts=args.sample_z,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    print("CPG constant log-redshift mapping stress test")
    print("Model: ln(1+z_obs) = (1+epsilon) ln(1+z_geom)")
    print(f"H0 late            : {result.h0_late:.6f} km s^-1 Mpc^-1")
    print(f"H0 early           : {result.h0_early:.6f} km s^-1 Mpc^-1")
    print(f"epsilon             : {result.epsilon:.8f} ({100*result.epsilon:.3f}%)")
    print(f"local zgeom/zobs    : {result.local_scale_factor:.8f}")
    print("\nSelected observed redshifts mapped to geometric redshift")
    for row in result.sample_rows:
        print(
            f"z_obs={row['z_obs']:.5f} -> z_geom={row['z_geom']:.5f} "
            f"(reduction={100*row['fractional_reduction']:.3f}%)"
        )
    print("\nCMB/recombination stress")
    print(f"standard z*         : {result.z_star_standard:.3f}")
    print(
        "predicted z_obs if z*_geom stays standard: "
        f"{result.z_star_observed_if_geometric_standard:.3f}"
    )
    print(
        "required z*_geom if z*_obs stays standard: "
        f"{result.z_star_geometric_if_observed_standard:.3f}"
    )
    print(f"(1+z) CMB ratio     : {result.cmb_observed_ratio:.6f}")
    print("\nInterpretation:")
    print(
        "A constant epsilon large enough to reconcile the local H0 normalization "
        "also produces a very large high-redshift remapping. This is a stress "
        "test, not a physical RTD-EU solution."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
