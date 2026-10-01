#!/usr/bin/env python3
"""Controlled end-to-end clock/light-cone closure demo for CPG.

This is a validation experiment, not a cosmological model.  It uses the same
metric for source/observer clock rates and for the photon null geodesic, then
checks whether the numerical redshift closes against the analytic result.

Metric family:

    ds^2 = -alpha(x)^2 dt^2 + a(t)^2 (dx^2+dy^2+dz^2)

with a linear lapse profile between source x=0 and observer x=L,

    alpha(0)=1-delta,  alpha(L)=1+delta,

and exponential scale factor a(t)=exp(H t).  Static coordinate observers have
proper-time rates d tau/dt=alpha.  For this separable benchmark the exact
source-to-observer frequency ratio is

    1+z = [a(t_obs)/a(t_emit)] [alpha_obs/alpha_emit].

Therefore the combined closure statistic

    C = (1+z) (alpha_emit/alpha_obs) / [a_obs/a_emit]

must equal unity.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import argparse
import json
import math

import numpy as np

from cpg_lightcone import FiniteDifferenceSpacetime, endpoint_redshift
from cpg_lightcone_shooting import shoot_to_observer_worldline


@dataclass(frozen=True)
class ClosureDemoResult:
    H: float
    lapse_half_contrast: float
    separation: float
    arrival_time: float
    numerical_redshift: float
    analytic_redshift: float
    redshift_error: float
    proper_time_rate_ratio_emit_over_obs: float
    expansion_factor: float
    closure_ratio: float
    spatial_miss_distance: float
    max_abs_null_residual: float
    optimizer_nfev: int


def run_demo(H: float = 0.08, delta: float = 5e-5,
             separation: float = 1.0) -> ClosureDemoResult:
    if H < 0:
        raise ValueError("H must be non-negative")
    if not 0 <= delta < 0.25:
        raise ValueError("delta must satisfy 0 <= delta < 0.25")
    if separation <= 0:
        raise ValueError("separation must be positive")

    L = float(separation)

    def lapse(x: float) -> float:
        return 1.0 - delta + 2.0 * delta * float(x) / L

    def metric_fn(event):
        t, x = float(event[0]), float(event[1])
        alpha = lapse(x)
        a = math.exp(H * t)
        return np.diag([-alpha * alpha, a * a, a * a, a * a])

    # Bounds are intentionally generous relative to the unit separation.
    spacetime = FiniteDifferenceSpacetime(
        metric_fn,
        spatial_step=1e-5 * max(L, 1.0),
        time_step=1e-5,
        t_bounds=(-1.0, 5.0 * max(L, 1.0) + 5.0),
    )
    emit = np.array([0.0, 0.0, 0.0, 0.0])
    observer = lambda _t: np.array([L, 0.0, 0.0])

    result = shoot_to_observer_worldline(
        spacetime,
        emit,
        observer,
        (0.5 * L, 2.5 * L + 0.5),
        initial_arrival_time=1.05 * L,
        position_tolerance=1e-9,
        ray_rtol=1e-10,
        ray_atol=1e-12,
        ray_max_step=0.02 * max(L, 1.0),
        max_nfev=80,
    )

    alpha_emit = lapse(0.0)
    alpha_obs = lapse(L)
    u_emit = np.array([1.0 / alpha_emit, 0.0, 0.0, 0.0])
    u_obs = np.array([1.0 / alpha_obs, 0.0, 0.0, 0.0])
    z_num = endpoint_redshift(
        spacetime.metric(result.ray.emit_event),
        u_emit,
        result.ray.k_emit,
        spacetime.metric(result.ray.obs_event),
        u_obs,
        result.ray.k_obs,
    )

    expansion = math.exp(H * result.arrival_time)
    rate_ratio = alpha_emit / alpha_obs
    z_exact = expansion * alpha_obs / alpha_emit - 1.0
    closure = (1.0 + z_num) * rate_ratio / expansion

    return ClosureDemoResult(
        H=float(H),
        lapse_half_contrast=float(delta),
        separation=L,
        arrival_time=float(result.arrival_time),
        numerical_redshift=float(z_num),
        analytic_redshift=float(z_exact),
        redshift_error=float(z_num - z_exact),
        proper_time_rate_ratio_emit_over_obs=float(rate_ratio),
        expansion_factor=float(expansion),
        closure_ratio=float(closure),
        spatial_miss_distance=float(result.spatial_miss_distance),
        max_abs_null_residual=float(result.ray.max_abs_null_residual),
        optimizer_nfev=int(result.optimizer_nfev),
    )


def main() -> int:
    p = argparse.ArgumentParser(description="CPG end-to-end clock/light-cone closure demo")
    p.add_argument("--H", type=float, default=0.08)
    p.add_argument("--delta", type=float, default=5e-5)
    p.add_argument("--separation", type=float, default=1.0)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    r = run_demo(args.H, args.delta, args.separation)
    if args.json:
        print(json.dumps(asdict(r), indent=2, sort_keys=True))
        return 0
    print("CPG END-TO-END CLOCK / LIGHT-CONE CLOSURE DEMO")
    print("=" * 72)
    print(f"H                         : {r.H:.12g}")
    print(f"lapse half-contrast delta : {r.lapse_half_contrast:.12g}")
    print(f"arrival time              : {r.arrival_time:.12g}")
    print(f"numerical z               : {r.numerical_redshift:.12g}")
    print(f"analytic z                : {r.analytic_redshift:.12g}")
    print(f"z numerical-analytic      : {r.redshift_error:+.6e}")
    print(f"tau-rate emit/obs         : {r.proper_time_rate_ratio_emit_over_obs:.12g}")
    print(f"expansion factor          : {r.expansion_factor:.12g}")
    print(f"closure ratio C           : {r.closure_ratio:.12g}")
    print(f"spatial miss              : {r.spatial_miss_distance:.6e}")
    print(f"max |k.g.k|               : {r.max_abs_null_residual:.6e}")
    print("NOTE: controlled validation metric only; not evidence for RTD-EU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
