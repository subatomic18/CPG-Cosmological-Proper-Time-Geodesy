#!/usr/bin/env python3
"""Early-time expansion-window auditor for Cosmological Proper-Time Geodesy (CPG).

This module is a fast consistency diagnostic. It does not solve Einstein's
field equations or establish a physical RTD-EU model. Given a multiplicative
modification q(z) to the fiducial H(z), it reports the change in elapsed cosmic
time and the resulting shift in the comoving distance to last scattering.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy import integrate

from cpg_v0_3 import C_KM_S, FiducialCosmology, HUBBLE_CONVERSION_GYR


@dataclass(frozen=True)
class EarlyWindowResult:
    z_low: float
    z_high: float
    z_star: float
    h_factor: float
    reference_elapsed_gyr: float
    modified_elapsed_gyr: float
    extra_elapsed_gyr: float
    reference_comoving_distance_mpc: float
    modified_comoving_distance_mpc: float
    delta_distance_mpc: float
    fractional_distance_shift: float
    theta_ratio_fixed_rs: float
    required_rs_ratio_for_fixed_theta: float


class EarlyTimeWindowAuditor:
    """Fast age-versus-distance consistency test for localized H(z) changes."""

    def __init__(self, cosmo: Optional[FiducialCosmology] = None):
        self.cosmo = cosmo or FiducialCosmology()

    def elapsed_time(self, z_low: float, z_high: float,
                     h_factor_fn: Optional[Callable[[float], float]] = None) -> float:
        if z_high <= z_low or z_low < 0:
            raise ValueError("Require 0 <= z_low < z_high")

        def integrand(z: float) -> float:
            q = 1.0 if h_factor_fn is None else float(h_factor_fn(z))
            if q <= 0:
                raise ValueError("H(z) multiplicative factor must be positive")
            return HUBBLE_CONVERSION_GYR / ((1.0 + z) * self.cosmo.H(z) * q)

        return float(integrate.quad(
            integrand, z_low, z_high, epsabs=1e-12, epsrel=1e-12, limit=600
        )[0])

    def comoving_distance(self, z: float,
                          h_factor_fn: Optional[Callable[[float], float]] = None,
                          breakpoints=()) -> float:
        if z <= 0:
            raise ValueError("Require z > 0")

        def integrand(zz: float) -> float:
            q = 1.0 if h_factor_fn is None else float(h_factor_fn(zz))
            if q <= 0:
                raise ValueError("H(z) multiplicative factor must be positive")
            return C_KM_S / (self.cosmo.H(zz) * q)

        points = [0.0] + sorted(float(x) for x in breakpoints if 0.0 < x < z) + [float(z)]
        total = 0.0
        for lo, hi in zip(points[:-1], points[1:]):
            total += integrate.quad(
                integrand, lo, hi, epsabs=1e-9, epsrel=1e-11, limit=600
            )[0]
        return float(total)

    @staticmethod
    def constant_window_factor(q: float, z_low: float, z_high: float):
        q = float(q)
        if q <= 0:
            raise ValueError("q must be positive")
        if z_high <= z_low or z_low < 0:
            raise ValueError("Require 0 <= z_low < z_high")

        def factor(z: float) -> float:
            return q if z_low <= z <= z_high else 1.0

        return factor

    def solve_constant_factor_for_extra_time(self, z_low: float, z_high: float,
                                             extra_gyr: float) -> float:
        if extra_gyr <= 0:
            raise ValueError("extra_gyr must be positive")
        ref = self.elapsed_time(z_low, z_high)
        target = ref + float(extra_gyr)

        # For a constant multiplicative H factor in the interval, elapsed time
        # scales exactly as 1/q. Use the analytic solution and independently
        # verify it numerically.
        q = ref / target
        check = self.elapsed_time(
            z_low, z_high, self.constant_window_factor(q, z_low, z_high)
        )
        if not np.isclose(check, target, rtol=1e-10, atol=1e-12):
            raise RuntimeError("Failed to recover requested elapsed-time target")
        return float(q)

    def audit_constant_window(self, z_low: float, z_high: float, q: float,
                              z_star: float = 1089.8) -> EarlyWindowResult:
        if z_star <= z_high:
            raise ValueError("z_star must lie above the modified window")
        factor = self.constant_window_factor(q, z_low, z_high)

        tref = self.elapsed_time(z_low, z_high)
        tmod = self.elapsed_time(z_low, z_high, factor)
        dref = self.comoving_distance(z_star)
        dmod = self.comoving_distance(
            z_star, factor, breakpoints=(z_low, z_high)
        )

        frac_d = dmod / dref - 1.0
        # For fixed sound horizon, theta_* is inversely proportional to D_A;
        # at the same z_star the (1+z_star) conversion cancels in the ratio.
        theta_ratio = dref / dmod
        # To keep theta_* fixed, r_s must scale in the same way as D_A.
        required_rs_ratio = dmod / dref

        return EarlyWindowResult(
            z_low=float(z_low), z_high=float(z_high), z_star=float(z_star),
            h_factor=float(q), reference_elapsed_gyr=tref,
            modified_elapsed_gyr=tmod, extra_elapsed_gyr=tmod - tref,
            reference_comoving_distance_mpc=dref,
            modified_comoving_distance_mpc=dmod,
            delta_distance_mpc=dmod - dref,
            fractional_distance_shift=frac_d,
            theta_ratio_fixed_rs=theta_ratio,
            required_rs_ratio_for_fixed_theta=required_rs_ratio,
        )

    def audit_target_extra_time(self, z_low: float, z_high: float,
                                extra_gyr: float, z_star: float = 1089.8) -> EarlyWindowResult:
        q = self.solve_constant_factor_for_extra_time(z_low, z_high, extra_gyr)
        return self.audit_constant_window(z_low, z_high, q, z_star=z_star)


def format_result(r: EarlyWindowResult) -> str:
    return (
        "CPG EARLY-TIME WINDOW AUDIT\n"
        + "=" * 72 + "\n"
        f"window: z=[{r.z_low:.3f}, {r.z_high:.3f}] | z_star={r.z_star:.1f}\n"
        f"H factor q: {r.h_factor:.9f} ({(r.h_factor-1.0)*100:+.3f}%)\n"
        f"elapsed reference: {r.reference_elapsed_gyr*1e3:.3f} Myr\n"
        f"elapsed modified:  {r.modified_elapsed_gyr*1e3:.3f} Myr\n"
        f"extra elapsed:     {r.extra_elapsed_gyr*1e3:+.3f} Myr\n"
        f"D_C(z_star) ref:   {r.reference_comoving_distance_mpc:.3f} Mpc\n"
        f"D_C(z_star) mod:   {r.modified_comoving_distance_mpc:.3f} Mpc\n"
        f"distance shift:    {r.delta_distance_mpc:+.3f} Mpc "
        f"({r.fractional_distance_shift*100:+.4f}%)\n"
        f"theta*/theta*_ref if r_s fixed: {r.theta_ratio_fixed_rs:.9f} "
        f"({(r.theta_ratio_fixed_rs-1.0)*100:+.4f}%)\n"
        f"required r_s/r_s_ref for fixed theta*: {r.required_rs_ratio_for_fixed_theta:.9f} "
        f"({(r.required_rs_ratio_for_fixed_theta-1.0)*100:+.4f}%)"
    )


if __name__ == "__main__":
    auditor = EarlyTimeWindowAuditor()
    result = auditor.audit_target_extra_time(14.32, 26.7, 0.020, z_star=1089.8)
    print(format_result(result))
