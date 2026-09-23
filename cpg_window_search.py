#!/usr/bin/env python3
"""Smooth early-time window search for CPG.

Searches a simple smooth family of localized H(z) suppressions that produce a
requested extra elapsed time while minimizing the induced change in comoving
distance to last scattering. This is a phenomenological consistency test, not
an Einstein-equation solver and not evidence for RTD-EU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import optimize

from cpg_early_window import EarlyTimeWindowAuditor
from cpg_v0_3 import C_KM_S, HUBBLE_CONVERSION_GYR


@dataclass(frozen=True)
class SmoothWindowCandidate:
    a: float
    b: float
    amplitude: float
    peak_z: float
    min_h_factor: float
    extra_elapsed_gyr: float
    delta_distance_mpc: float
    fractional_distance_shift: float


def beta_bump(z: float, z_low: float, z_high: float, a: float, b: float) -> float:
    """Compact beta-shaped bump normalized to unity at its maximum.

    For a,b >= 2 the profile and its first derivative vanish at both window
    boundaries, so q(z)=1-A*bump joins smoothly onto the fiducial H(z).
    """
    if z <= z_low or z >= z_high:
        return 0.0
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive")
    x = (z - z_low) / (z_high - z_low)
    xp = a / (a + b)
    return float((x / xp) ** a * ((1.0 - x) / (1.0 - xp)) ** b)


def make_h_factor(z_low: float, z_high: float, a: float, b: float, amplitude: float):
    if not 0.0 <= amplitude < 1.0:
        raise ValueError("Require 0 <= amplitude < 1")

    def q(z: float) -> float:
        return 1.0 - amplitude * beta_bump(z, z_low, z_high, a, b)

    return q


def solve_amplitude(auditor: EarlyTimeWindowAuditor, z_low: float, z_high: float,
                    extra_gyr: float, a: float, b: float,
                    max_amplitude: float = 0.95) -> float | None:
    if extra_gyr <= 0:
        raise ValueError("extra_gyr must be positive")
    if not 0 < max_amplitude < 1:
        raise ValueError("max_amplitude must satisfy 0 < max_amplitude < 1")

    tref = auditor.elapsed_time(z_low, z_high)
    target = tref + extra_gyr

    def residual(A: float) -> float:
        q = make_h_factor(z_low, z_high, a, b, A)
        return auditor.elapsed_time(z_low, z_high, q) - target

    if residual(max_amplitude) < 0:
        return None
    return float(optimize.brentq(residual, 0.0, max_amplitude, xtol=1e-12, rtol=1e-12))


def theoretical_distance_lower_bound_mpc(z_low: float, extra_gyr: float) -> float:
    """Lower bound for positive H suppression supported entirely at z>=z_low.

    From dD/dt_extra = (c/HubbleConversion)*(1+z), the smallest possible
    distance penalty is approached by concentrating the extra elapsed time as
    close as possible to the lower boundary of the allowed redshift window.
    """
    return float((C_KM_S / HUBBLE_CONVERSION_GYR) * (1.0 + z_low) * extra_gyr)


def search_smooth_windows(z_low: float = 14.32, z_high: float = 26.7,
                          extra_gyr: float = 0.020, z_star: float = 1089.8,
                          a_values: Iterable[float] = (2.0,),
                          b_values: Iterable[float] = tuple(np.linspace(2.0, 60.0, 117)),
                          max_amplitude: float = 0.35,
                          auditor: EarlyTimeWindowAuditor | None = None):
    auditor = auditor or EarlyTimeWindowAuditor()
    dref = auditor.comoving_distance(z_star)
    candidates: list[SmoothWindowCandidate] = []

    for a in a_values:
        for b in b_values:
            a = float(a); b = float(b)
            if a < 2 or b < 2:
                raise ValueError("Use a,b >= 2 for C1 boundary matching")
            A = solve_amplitude(auditor, z_low, z_high, extra_gyr, a, b,
                                max_amplitude=max_amplitude)
            if A is None:
                continue
            q = make_h_factor(z_low, z_high, a, b, A)
            tmod = auditor.elapsed_time(z_low, z_high, q)
            tref = auditor.elapsed_time(z_low, z_high)
            dmod = auditor.comoving_distance(z_star, q, breakpoints=(z_low, z_high))
            peak_x = a / (a + b)
            peak_z = z_low + peak_x * (z_high - z_low)
            candidates.append(SmoothWindowCandidate(
                a=a, b=b, amplitude=A, peak_z=float(peak_z),
                min_h_factor=1.0 - A, extra_elapsed_gyr=tmod - tref,
                delta_distance_mpc=dmod - dref,
                fractional_distance_shift=dmod / dref - 1.0,
            ))

    candidates.sort(key=lambda c: c.delta_distance_mpc)
    return candidates


def main():
    z_low, z_high, extra, z_star = 14.32, 26.7, 0.020, 1089.8
    auditor = EarlyTimeWindowAuditor()
    candidates = search_smooth_windows(
        z_low=z_low, z_high=z_high, extra_gyr=extra, z_star=z_star,
        a_values=(2.0,), b_values=np.linspace(2.0, 60.0, 117),
        max_amplitude=0.35, auditor=auditor,
    )
    if not candidates:
        raise SystemExit("No candidate satisfies the requested extra time and amplitude bound")

    best = candidates[0]
    dref = auditor.comoving_distance(z_star)
    bound = theoretical_distance_lower_bound_mpc(z_low, extra)
    print("CPG SMOOTH EARLY-WINDOW SEARCH")
    print("=" * 72)
    print(f"target extra time: {extra*1e3:.3f} Myr")
    print(f"window: z=[{z_low:.3f}, {z_high:.3f}]")
    print(f"best beta shape: a={best.a:.3f}, b={best.b:.3f}")
    print(f"peak z: {best.peak_z:.4f}")
    print(f"peak H suppression: {best.amplitude*100:.3f}%")
    print(f"minimum q=H_mod/H_ref: {best.min_h_factor:.6f}")
    print(f"distance penalty: {best.delta_distance_mpc:.3f} Mpc "
          f"({best.fractional_distance_shift*100:.4f}%)")
    print(f"mathematical lower bound: {bound:.3f} Mpc ({bound/dref*100:.4f}%)")
    print("NOTE: the optimum is conditional on this profile family and amplitude cap.")


if __name__ == "__main__":
    main()
