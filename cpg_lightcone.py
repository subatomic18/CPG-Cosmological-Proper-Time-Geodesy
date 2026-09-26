#!/usr/bin/env python3
"""Null-geodesic and endpoint-redshift utilities for CPG.

This module is the first light-cone bridge between CPG worldline proper-time
calculations and photon observables. It integrates the null geodesic equation

    dx^mu/dlambda = k^mu
    dk^mu/dlambda = -Gamma^mu_ab k^a k^b

and evaluates the observed photon frequency

    omega = -u_mu k^mu

so that

    1 + z = omega_emit / omega_obs.

The core is metric-agnostic. A finite-difference spacetime wrapper accepts any
callable g_mn(x), and ``MescalineSpacetime`` supplies g_mn from the same
Mescaline/Einstein-Toolkit snapshots used by CPG's fluid-worldline adapter.

This is an experimental numerical-relativity bridge, not a complete
cosmological likelihood. A physical source-to-observer prediction requires
physically specified emitter/observer worldlines, invariant endpoint events or
hypersurfaces, sufficient simulation resolution, and convergence tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from cpg_mescaline_adapter import MescalineSnapshot, _periodic_trilinear

Array = np.ndarray


def adm_metric(alpha: float, beta: Array, gamma: Array) -> Array:
    """Return the 4-metric for the ADM convention used by CPG."""
    alpha = float(alpha)
    beta = np.asarray(beta, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    if alpha <= 0 or not np.isfinite(alpha):
        raise ValueError("alpha must be finite and positive")
    if beta.shape != (3,) or gamma.shape != (3, 3):
        raise ValueError("beta must have shape (3,) and gamma shape (3,3)")
    if not np.all(np.isfinite(beta)) or not np.all(np.isfinite(gamma)):
        raise ValueError("ADM fields must be finite")
    if not np.allclose(gamma, gamma.T, atol=1e-12, rtol=1e-10):
        raise ValueError("gamma must be symmetric")
    if np.linalg.eigvalsh(gamma).min() <= 0:
        raise ValueError("gamma must be positive definite")

    g = np.empty((4, 4), dtype=float)
    gbeta = gamma @ beta
    g[0, 0] = -alpha * alpha + float(beta @ gbeta)
    g[0, 1:] = gbeta
    g[1:, 0] = gbeta
    g[1:, 1:] = gamma
    return g


def eulerian_four_velocity(alpha: float, beta: Array, gamma: Array, v_eulerian: Array) -> Array:
    """Convert an ADM Eulerian three-velocity to contravariant u^mu."""
    alpha = float(alpha)
    beta = np.asarray(beta, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    v = np.asarray(v_eulerian, dtype=float)
    if alpha <= 0 or beta.shape != (3,) or gamma.shape != (3, 3) or v.shape != (3,):
        raise ValueError("invalid ADM velocity inputs")
    v2 = float(v @ gamma @ v)
    if not 0.0 <= v2 < 1.0:
        raise ValueError("Eulerian velocity must be timelike: 0 <= v^2 < 1")
    lorentz = 1.0 / np.sqrt(1.0 - v2)
    u = np.empty(4, dtype=float)
    u[0] = lorentz / alpha
    u[1:] = lorentz * (v - beta / alpha)

    g = adm_metric(alpha, beta, gamma)
    norm = float(u @ g @ u)
    if not np.isclose(norm, -1.0, atol=1e-10, rtol=1e-9):
        raise RuntimeError(f"constructed four-velocity has norm {norm}, expected -1")
    return u


def photon_frequency(metric: Array, u_contravariant: Array, k_contravariant: Array) -> float:
    """Return omega = -u_mu k^mu for a future-directed photon."""
    g = np.asarray(metric, dtype=float)
    u = np.asarray(u_contravariant, dtype=float)
    k = np.asarray(k_contravariant, dtype=float)
    if g.shape != (4, 4) or u.shape != (4,) or k.shape != (4,):
        raise ValueError("metric must be 4x4 and u,k must be four-vectors")
    omega = -float(u @ g @ k)
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError("photon frequency must be finite and positive")
    return omega


def endpoint_redshift(metric_emit: Array, u_emit: Array, k_emit: Array,
                      metric_obs: Array, u_obs: Array, k_obs: Array) -> float:
    """Return z from endpoint observer contractions."""
    w_emit = photon_frequency(metric_emit, u_emit, k_emit)
    w_obs = photon_frequency(metric_obs, u_obs, k_obs)
    return float(w_emit / w_obs - 1.0)


def solve_future_null_k0(metric: Array, spatial_k: Array) -> float:
    """Solve the null condition for the future-directed k^0."""
    g = np.asarray(metric, dtype=float)
    ki = np.asarray(spatial_k, dtype=float)
    if g.shape != (4, 4) or ki.shape != (3,):
        raise ValueError("metric must be 4x4 and spatial_k shape (3,)")
    if not np.all(np.isfinite(ki)) or np.linalg.norm(ki) == 0:
        raise ValueError("spatial_k must be finite and non-zero")

    A = float(g[0, 0])
    B = 2.0 * float(g[0, 1:] @ ki)
    C = float(ki @ g[1:, 1:] @ ki)
    disc = B * B - 4.0 * A * C
    if disc <= 0 or A == 0:
        raise ValueError("metric/direction does not yield a real null k0")
    root = np.sqrt(disc)
    roots = [(-B + root) / (2.0 * A), (-B - root) / (2.0 * A)]
    positive = [r for r in roots if r > 0 and np.isfinite(r)]
    if not positive:
        raise ValueError("no future-directed null root")
    return float(max(positive))


class FiniteDifferenceSpacetime:
    """Numerical metric and Christoffel evaluator for g_mu_nu(x)."""

    def __init__(self, metric_fn: Callable[[Array], Array], spatial_step: float,
                 time_step: float, t_bounds: tuple[float, float] | None = None):
        if spatial_step <= 0 or time_step <= 0:
            raise ValueError("finite-difference steps must be positive")
        self.metric_fn = metric_fn
        self.spatial_step = float(spatial_step)
        self.time_step = float(time_step)
        self.t_bounds = t_bounds

    def metric(self, event: Array) -> Array:
        x = np.asarray(event, dtype=float)
        if x.shape != (4,):
            raise ValueError("event must have shape (4,)")
        g = np.asarray(self.metric_fn(x), dtype=float)
        if g.shape != (4, 4):
            raise ValueError("metric_fn must return shape (4,4)")
        if not np.all(np.isfinite(g)):
            raise ValueError("metric contains non-finite values")
        if not np.allclose(g, g.T, atol=1e-10, rtol=1e-10):
            raise ValueError("metric must be symmetric")
        return g

    def metric_derivatives(self, event: Array) -> Array:
        """Return dg[coord, mu, nu] = partial_coord g_mu_nu."""
        x = np.asarray(event, dtype=float)
        deriv = np.empty((4, 4, 4), dtype=float)
        for a in range(4):
            h = self.time_step if a == 0 else self.spatial_step
            xp = x.copy()
            xm = x.copy()
            if a == 0 and self.t_bounds is not None:
                lo, hi = self.t_bounds
                if x[0] - h < lo:
                    xp[0] = min(x[0] + h, hi)
                    dx = xp[0] - x[0]
                    if dx <= 0:
                        raise ValueError("cannot time-difference at upper spacetime boundary")
                    deriv[a] = (self.metric(xp) - self.metric(x)) / dx
                    continue
                if x[0] + h > hi:
                    xm[0] = max(x[0] - h, lo)
                    dx = x[0] - xm[0]
                    if dx <= 0:
                        raise ValueError("cannot time-difference at lower spacetime boundary")
                    deriv[a] = (self.metric(x) - self.metric(xm)) / dx
                    continue
            xp[a] += h
            xm[a] -= h
            deriv[a] = (self.metric(xp) - self.metric(xm)) / (2.0 * h)
        return deriv

    def christoffel(self, event: Array) -> Array:
        g = self.metric(event)
        ginv = np.linalg.inv(g)
        dg = self.metric_derivatives(event)
        Gamma = np.zeros((4, 4, 4), dtype=float)
        for mu in range(4):
            for a in range(4):
                for b in range(4):
                    s = 0.0
                    for nu in range(4):
                        s += ginv[mu, nu] * (
                            dg[a, nu, b] + dg[b, nu, a] - dg[nu, a, b]
                        )
                    Gamma[mu, a, b] = 0.5 * s
        return Gamma


@dataclass(frozen=True)
class NullGeodesicResult:
    events: Array
    wavevectors: Array
    affine_parameter: Array
    reached_target_time: bool
    max_abs_null_residual: float

    @property
    def emit_event(self) -> Array:
        return self.events[0]

    @property
    def obs_event(self) -> Array:
        return self.events[-1]

    @property
    def k_emit(self) -> Array:
        return self.wavevectors[0]

    @property
    def k_obs(self) -> Array:
        return self.wavevectors[-1]


def trace_null_geodesic(spacetime: FiniteDifferenceSpacetime, emit_event: Array,
                        spatial_k: Array, target_time: float, *, rtol: float = 1e-8,
                        atol: float = 1e-10, max_step: float = np.inf,
                        max_affine: float | None = None) -> NullGeodesicResult:
    """Trace a future-directed null ray until coordinate time ``target_time``.

    This initial version is an initial-value ray tracer. It does not yet solve a
    boundary-value shooting problem to a prescribed observer position.
    """
    x0 = np.asarray(emit_event, dtype=float)
    ki = np.asarray(spatial_k, dtype=float)
    if x0.shape != (4,) or ki.shape != (3,):
        raise ValueError("emit_event must be shape (4,), spatial_k shape (3,)")
    if target_time <= x0[0]:
        raise ValueError("target_time must exceed emission coordinate time")

    g0 = spacetime.metric(x0)
    k0 = solve_future_null_k0(g0, ki)
    y0 = np.concatenate((x0, np.array([k0, *ki], dtype=float)))
    if max_affine is None:
        max_affine = 20.0 * (target_time - x0[0]) / max(k0, 1e-12)

    def rhs(_lam: float, y: Array) -> Array:
        x = y[:4]
        k = y[4:]
        Gamma = spacetime.christoffel(x)
        dk = -np.einsum("mab,a,b->m", Gamma, k, k)
        return np.concatenate((k, dk))

    def hit_time(_lam: float, y: Array) -> float:
        return float(y[0] - target_time)

    hit_time.terminal = True
    hit_time.direction = 1.0

    sol = solve_ivp(rhs, (0.0, float(max_affine)), y0, method="DOP853",
                    rtol=rtol, atol=atol, max_step=max_step,
                    events=hit_time, dense_output=False)
    if not sol.success:
        raise RuntimeError(f"null geodesic integration failed: {sol.message}")

    events = sol.y[:4].T
    ks = sol.y[4:].T
    residuals = np.array([float(k @ spacetime.metric(x) @ k)
                          for x, k in zip(events, ks)])
    reached = bool(sol.t_events and len(sol.t_events[0]) > 0)
    return NullGeodesicResult(events, ks, sol.t, reached,
                              float(np.max(np.abs(residuals))))


class MescalineSpacetime(FiniteDifferenceSpacetime):
    """Continuous-in-time ADM metric built from Mescaline snapshots."""

    def __init__(self, snapshots: Sequence[MescalineSnapshot], dx: float,
                 origin=(0.0, 0.0, 0.0), *, spatial_fd_fraction: float = 0.25,
                 time_fd_fraction: float = 0.25):
        snaps = list(snapshots)
        if len(snaps) < 2:
            raise ValueError("need at least two Mescaline snapshots")
        snaps.sort(key=lambda s: s.time)
        times = np.array([s.time for s in snaps], dtype=float)
        if np.any(np.diff(times) <= 0):
            raise ValueError("snapshot times must be strictly increasing")
        if dx <= 0:
            raise ValueError("dx must be positive")
        self.snapshots = snaps
        self.times = times
        self.dx = float(dx)
        self.origin = np.asarray(origin, dtype=float)
        if self.origin.shape != (3,):
            raise ValueError("origin must have shape (3,)")
        dt_min = float(np.min(np.diff(times)))
        super().__init__(self._metric_from_event,
                         max(self.dx * float(spatial_fd_fraction), 1e-12),
                         max(dt_min * float(time_fd_fraction), 1e-12),
                         (float(times[0]), float(times[-1])))

    def _bracket(self, t: float) -> tuple[int, int, float]:
        if t < self.times[0] or t > self.times[-1]:
            raise ValueError(f"time {t} outside Mescaline range [{self.times[0]}, {self.times[-1]}]")
        if t == self.times[-1]:
            i0 = len(self.times) - 2
            return i0, i0 + 1, 1.0
        i1 = int(np.searchsorted(self.times, t, side="right"))
        i0 = max(0, i1 - 1)
        dt = self.times[i1] - self.times[i0]
        return i0, i1, float((t - self.times[i0]) / dt)

    def _sample_snapshot(self, snap: MescalineSnapshot, pos: Array):
        alpha = float(_periodic_trilinear(snap.alpha, pos, self.origin, self.dx))
        gamma = np.asarray(_periodic_trilinear(snap.gamma, pos, self.origin, self.dx), dtype=float)
        gamma = 0.5 * (gamma + gamma.T)
        velocity = np.asarray(_periodic_trilinear(snap.velocity, pos, self.origin, self.dx), dtype=float)
        return alpha, gamma, velocity

    def sample_adm(self, event: Array):
        x = np.asarray(event, dtype=float)
        i0, i1, w = self._bracket(float(x[0]))
        a0, g0, v0 = self._sample_snapshot(self.snapshots[i0], x[1:])
        a1, g1, v1 = self._sample_snapshot(self.snapshots[i1], x[1:])
        alpha = (1.0 - w) * a0 + w * a1
        gamma = (1.0 - w) * g0 + w * g1
        velocity = (1.0 - w) * v0 + w * v1
        gamma = 0.5 * (gamma + gamma.T)
        if np.linalg.eigvalsh(gamma).min() <= 0:
            raise ValueError("interpolated spatial metric is not positive definite")
        if float(velocity @ gamma @ velocity) >= 1.0:
            raise ValueError("interpolated Eulerian fluid velocity is non-timelike")
        return float(alpha), np.zeros(3), gamma, velocity

    def _metric_from_event(self, event: Array) -> Array:
        alpha, beta, gamma, _ = self.sample_adm(event)
        return adm_metric(alpha, beta, gamma)

    def fluid_four_velocity(self, event: Array) -> Array:
        alpha, beta, gamma, velocity = self.sample_adm(event)
        return eulerian_four_velocity(alpha, beta, gamma, velocity)

    def fluid_endpoint_redshift(self, ray: NullGeodesicResult) -> float:
        if not ray.reached_target_time:
            raise ValueError("ray did not reach requested target time")
        return endpoint_redshift(
            self.metric(ray.emit_event), self.fluid_four_velocity(ray.emit_event), ray.k_emit,
            self.metric(ray.obs_event), self.fluid_four_velocity(ray.obs_event), ray.k_obs,
        )
