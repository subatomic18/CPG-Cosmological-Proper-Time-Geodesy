#!/usr/bin/env python3
"""Boundary-value shooting utilities for the experimental CPG light cone.

The initial ``cpg_lightcone`` module traces a null ray from an emission event to
a chosen coordinate time.  This module solves the next problem: choose the ray
launch direction *and* arrival time so that the null geodesic intersects a
specified observer worldline.

The solver uses four unknowns,

    (k_x, k_y, k_z, t_arrival),

and four residuals: the three spatial miss components at the observer plus a
unit-norm constraint on the initial spatial wavevector.  The overall magnitude
of a null wavevector is an affine-parameter freedom, so fixing its spatial norm
removes that numerical degeneracy without changing the ray path.

This is still an experimental numerical-relativity bridge.  A converged
coordinate intersection is not by itself a cosmological observable: emitter
and observer histories must be physically specified, endpoint events must be
interpreted invariantly, and the result requires resolution/convergence tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import least_squares

from cpg_lightcone import (
    FiniteDifferenceSpacetime,
    MescalineSpacetime,
    NullGeodesicResult,
    trace_null_geodesic,
)
from cpg_mescaline_adapter import MescalineSnapshot, track_fluid_worldline

Array = np.ndarray


def minimum_image_displacement(a: Array, b: Array, box_size: float | Array) -> Array:
    """Return the periodic minimum-image displacement ``a-b``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    box = np.asarray(box_size, dtype=float)
    if a.shape != (3,) or b.shape != (3,):
        raise ValueError("a and b must have shape (3,)")
    if box.ndim == 0:
        box = np.full(3, float(box))
    if box.shape != (3,) or np.any(box <= 0) or not np.all(np.isfinite(box)):
        raise ValueError("box_size must be positive scalar or shape (3,)")
    d = a - b
    return d - box * np.round(d / box)


def _unwrap_periodic_positions(positions: Array, box_size: float | Array) -> Array:
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("positions must have shape (n,3)")
    box = np.asarray(box_size, dtype=float)
    if box.ndim == 0:
        box = np.full(3, float(box))
    if box.shape != (3,) or np.any(box <= 0):
        raise ValueError("box_size must be positive scalar or shape (3,)")
    out = np.empty_like(pos)
    out[0] = pos[0]
    for i in range(1, len(pos)):
        step = minimum_image_displacement(pos[i], pos[i - 1], box)
        out[i] = out[i - 1] + step
    return out


@dataclass(frozen=True)
class TrackedSpatialWorldline:
    """Time-sampled spatial worldline with linear interpolation."""

    times: Array
    positions: Array

    def __post_init__(self) -> None:
        t = np.asarray(self.times, dtype=float)
        p = np.asarray(self.positions, dtype=float)
        if t.ndim != 1 or p.shape != (len(t), 3):
            raise ValueError("times must be (n,) and positions (n,3)")
        if len(t) < 2 or np.any(np.diff(t) <= 0):
            raise ValueError("worldline times must be strictly increasing")
        if not np.all(np.isfinite(t)) or not np.all(np.isfinite(p)):
            raise ValueError("worldline samples must be finite")
        object.__setattr__(self, "times", t)
        object.__setattr__(self, "positions", p)

    @classmethod
    def from_periodic_samples(cls, times: Sequence[float], positions: Array,
                              box_size: float | Array) -> "TrackedSpatialWorldline":
        return cls(np.asarray(times, dtype=float),
                   _unwrap_periodic_positions(np.asarray(positions, dtype=float), box_size))

    def position(self, t: float) -> Array:
        t = float(t)
        if t < self.times[0] or t > self.times[-1]:
            raise ValueError(
                f"time {t} outside tracked range [{self.times[0]}, {self.times[-1]}]"
            )
        return np.array([
            np.interp(t, self.times, self.positions[:, j]) for j in range(3)
        ], dtype=float)

    def shifted(self, offset: Array) -> "TrackedSpatialWorldline":
        off = np.asarray(offset, dtype=float)
        if off.shape != (3,):
            raise ValueError("offset must have shape (3,)")
        return TrackedSpatialWorldline(self.times, self.positions + off[None, :])

    def align_periodic_image(self, reference_position: Array, t_ref: float,
                             box_size: float | Array) -> "TrackedSpatialWorldline":
        """Shift the unwrapped path to the periodic image nearest a reference."""
        ref = np.asarray(reference_position, dtype=float)
        if ref.shape != (3,):
            raise ValueError("reference_position must have shape (3,)")
        box = np.asarray(box_size, dtype=float)
        if box.ndim == 0:
            box = np.full(3, float(box))
        here = self.position(float(t_ref))
        shift = box * np.round((ref - here) / box)
        return self.shifted(shift)


@dataclass(frozen=True)
class ObserverShootingResult:
    ray: NullGeodesicResult
    arrival_time: float
    observer_position: Array
    initial_spatial_k: Array
    spatial_miss: Array
    spatial_miss_distance: float
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    optimizer_cost: float
    optimizer_nfev: int
    converged: bool


def shoot_to_observer_worldline(
    spacetime: FiniteDifferenceSpacetime,
    emit_event: Array,
    observer_position_fn: Callable[[float], Array],
    arrival_time_bounds: tuple[float, float],
    *,
    initial_arrival_time: float | None = None,
    initial_direction: Array | None = None,
    displacement_fn: Callable[[Array, Array], Array] | None = None,
    position_scale: float | None = None,
    position_tolerance: float = 1e-6,
    max_nfev: int = 80,
    ray_rtol: float = 1e-8,
    ray_atol: float = 1e-10,
    ray_max_step: float = np.inf,
    require_convergence: bool = True,
) -> ObserverShootingResult:
    """Shoot a null ray until it intersects a spatial observer worldline.

    ``observer_position_fn(t)`` returns the observer's coordinate position.
    ``arrival_time_bounds`` supplies the allowed coordinate-time window.  The
    solver varies the initial spatial wavevector and arrival time together.

    The default spatial residual is ordinary coordinate subtraction.  For a
    periodic box, pass a minimum-image ``displacement_fn`` or align an unwrapped
    tracked worldline to the emitter's periodic image first.
    """
    x0 = np.asarray(emit_event, dtype=float)
    if x0.shape != (4,):
        raise ValueError("emit_event must have shape (4,)")
    t_lo, t_hi = map(float, arrival_time_bounds)
    if not (x0[0] < t_lo < t_hi):
        raise ValueError("require emit_time < arrival lower bound < upper bound")
    if initial_arrival_time is None:
        initial_arrival_time = 0.5 * (t_lo + t_hi)
    t_guess = float(initial_arrival_time)
    if not t_lo <= t_guess <= t_hi:
        raise ValueError("initial_arrival_time must lie inside arrival_time_bounds")

    def obs_pos(t: float) -> Array:
        p = np.asarray(observer_position_fn(float(t)), dtype=float)
        if p.shape != (3,) or not np.all(np.isfinite(p)):
            raise ValueError("observer_position_fn must return finite shape (3,)")
        return p

    displacement = displacement_fn or (lambda a, b: np.asarray(a) - np.asarray(b))
    p_guess = obs_pos(t_guess)
    if initial_direction is None:
        direction0 = np.asarray(displacement(p_guess, x0[1:]), dtype=float)
    else:
        direction0 = np.asarray(initial_direction, dtype=float)
    if direction0.shape != (3,) or not np.all(np.isfinite(direction0)):
        raise ValueError("initial_direction must be finite shape (3,)")
    n0 = float(np.linalg.norm(direction0))
    if n0 <= 0:
        raise ValueError("initial direction cannot be zero")
    direction0 = direction0 / n0

    if position_scale is None:
        position_scale = max(float(np.linalg.norm(displacement(p_guess, x0[1:]))), 1.0)
    position_scale = float(position_scale)
    if position_scale <= 0 or not np.isfinite(position_scale):
        raise ValueError("position_scale must be finite and positive")
    if position_tolerance <= 0:
        raise ValueError("position_tolerance must be positive")

    x_init = np.concatenate((direction0, [t_guess]))
    lower = np.array([-2.0, -2.0, -2.0, t_lo], dtype=float)
    upper = np.array([2.0, 2.0, 2.0, t_hi], dtype=float)

    def residual(params: Array) -> Array:
        k_spatial = np.asarray(params[:3], dtype=float)
        knorm = float(np.linalg.norm(k_spatial))
        if knorm < 1e-8 or not np.isfinite(knorm):
            return np.array([1e3, 1e3, 1e3, 1e3], dtype=float)
        t_arr = float(params[3])
        try:
            ray = trace_null_geodesic(
                spacetime,
                x0,
                k_spatial,
                t_arr,
                rtol=ray_rtol,
                atol=ray_atol,
                max_step=ray_max_step,
            )
            if not ray.reached_target_time:
                return np.array([1e3, 1e3, 1e3, knorm - 1.0], dtype=float)
            miss = np.asarray(displacement(ray.obs_event[1:], obs_pos(t_arr)), dtype=float)
            if miss.shape != (3,) or not np.all(np.isfinite(miss)):
                raise ValueError("displacement_fn must return finite shape (3,)")
            return np.concatenate((miss / position_scale, [knorm - 1.0]))
        except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError):
            return np.array([1e3, 1e3, 1e3, knorm - 1.0], dtype=float)

    opt = least_squares(
        residual,
        x_init,
        bounds=(lower, upper),
        method="trf",
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=int(max_nfev),
    )

    k_best = np.asarray(opt.x[:3], dtype=float)
    k_best /= np.linalg.norm(k_best)
    t_best = float(opt.x[3])
    ray = trace_null_geodesic(
        spacetime,
        x0,
        k_best,
        t_best,
        rtol=ray_rtol,
        atol=ray_atol,
        max_step=ray_max_step,
    )
    observer_best = obs_pos(t_best)
    miss = np.asarray(displacement(ray.obs_event[1:], observer_best), dtype=float)
    miss_distance = float(np.linalg.norm(miss))
    converged = bool(opt.success and ray.reached_target_time and miss_distance <= position_tolerance)

    result = ObserverShootingResult(
        ray=ray,
        arrival_time=t_best,
        observer_position=observer_best,
        initial_spatial_k=k_best,
        spatial_miss=miss,
        spatial_miss_distance=miss_distance,
        optimizer_success=bool(opt.success),
        optimizer_status=int(opt.status),
        optimizer_message=str(opt.message),
        optimizer_cost=float(opt.cost),
        optimizer_nfev=int(opt.nfev),
        converged=converged,
    )
    if require_convergence and not converged:
        raise RuntimeError(
            "observer shooting did not converge: "
            f"optimizer_success={opt.success}, miss={miss_distance:.6g}, "
            f"t={t_best:.6g}, message={opt.message}"
        )
    return result


def mescaline_fluid_worldline(
    snapshots: Sequence[MescalineSnapshot],
    start_position: Array,
    dx: float,
    origin=(0.0, 0.0, 0.0),
) -> TrackedSpatialWorldline:
    """Track a Mescaline fluid element and return an unwrapped spatial path."""
    snaps = list(snapshots)
    if len(snaps) < 2:
        raise ValueError("need at least two snapshots")
    positions, samples = track_fluid_worldline(snaps, start_position, dx, origin)
    times = np.array([s.t for s in samples], dtype=float)
    box_size = float(snaps[0].nx) * float(dx)
    return TrackedSpatialWorldline.from_periodic_samples(times, positions, box_size)


def shoot_to_mescaline_fluid(
    spacetime: MescalineSpacetime,
    emit_event: Array,
    observer_start_position: Array,
    arrival_time_bounds: tuple[float, float],
    *,
    initial_arrival_time: float | None = None,
    position_tolerance: float | None = None,
    max_nfev: int = 80,
    ray_rtol: float = 1e-8,
    ray_atol: float = 1e-10,
    ray_max_step: float = np.inf,
) -> tuple[ObserverShootingResult, TrackedSpatialWorldline]:
    """Target a tracked Mescaline fluid observer with a null geodesic.

    The observer path is unwrapped across periodic boundaries and shifted to the
    periodic image nearest the emitter at the initial arrival-time guess.
    """
    if initial_arrival_time is None:
        initial_arrival_time = 0.5 * sum(map(float, arrival_time_bounds))
    worldline = mescaline_fluid_worldline(
        spacetime.snapshots,
        np.asarray(observer_start_position, dtype=float),
        spacetime.dx,
        spacetime.origin,
    )
    box_size = float(spacetime.snapshots[0].nx) * spacetime.dx
    worldline = worldline.align_periodic_image(
        np.asarray(emit_event, dtype=float)[1:],
        float(initial_arrival_time),
        box_size,
    )
    if position_tolerance is None:
        position_tolerance = max(1e-6, 1e-3 * spacetime.dx)
    result = shoot_to_observer_worldline(
        spacetime,
        emit_event,
        worldline.position,
        arrival_time_bounds,
        initial_arrival_time=initial_arrival_time,
        position_tolerance=float(position_tolerance),
        max_nfev=max_nfev,
        ray_rtol=ray_rtol,
        ray_atol=ray_atol,
        ray_max_step=ray_max_step,
    )
    return result, worldline
