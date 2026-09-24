#!/usr/bin/env python3
"""ADM-metric worldline proper-time integrator for CPG.

This module is the bridge from numerical-relativity output to chronometric
closure.  Given lapse, shift, spatial metric, and a coordinate worldline as
functions of simulation coordinate time, it computes

    d tau / dt = sqrt(alpha^2 - gamma_ij (v^i + beta^i)(v^j + beta^j))

in geometrized units c=1 for the ADM convention

    ds^2 = -alpha^2 dt^2
           + gamma_ij (dx^i + beta^i dt)(dx^j + beta^j dt).

The result is invariant proper time along the supplied timelike path, although
intermediate lapse/shift values are gauge dependent.  Physical comparisons
must therefore use physically specified worldlines and common invariant
boundary events or hypersurfaces.

The CSV loader expects samples already interpolated onto a worldline.  This
keeps the CPG core independent of any one simulation code or HDF5 layout; an
Einstein-Toolkit/Carpet adapter can be added on top once raw simulation output
is available.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from cpg_hubble_closure import HubbleClosureAuditor, GammaTest


_REQUIRED_COLUMNS = (
    "t", "alpha",
    "betax", "betay", "betaz",
    "gxx", "gxy", "gxz", "gyy", "gyz", "gzz",
    "vx", "vy", "vz",
)


@dataclass(frozen=True)
class ADMWorldlineSample:
    t: float
    alpha: float
    beta: np.ndarray
    gamma: np.ndarray
    velocity: np.ndarray

    def validate(self) -> None:
        if not np.isfinite(self.t):
            raise ValueError("sample time must be finite")
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("lapse alpha must be finite and positive")
        if self.beta.shape != (3,) or self.velocity.shape != (3,):
            raise ValueError("beta and velocity must have shape (3,)")
        if self.gamma.shape != (3, 3):
            raise ValueError("gamma must have shape (3,3)")
        if not np.all(np.isfinite(self.beta)) or not np.all(np.isfinite(self.velocity)):
            raise ValueError("beta and velocity must be finite")
        if not np.all(np.isfinite(self.gamma)):
            raise ValueError("gamma must be finite")
        if not np.allclose(self.gamma, self.gamma.T, atol=1e-12, rtol=1e-10):
            raise ValueError("spatial metric gamma must be symmetric")
        if np.linalg.eigvalsh(self.gamma).min() <= 0:
            raise ValueError("spatial metric gamma must be positive definite")

    def dtau_dt(self) -> float:
        self.validate()
        w = self.velocity + self.beta
        radicand = self.alpha**2 - float(w @ self.gamma @ w)
        if radicand <= 0:
            raise ValueError(
                "worldline sample is null/spacelike or inconsistent with ADM data"
            )
        return float(np.sqrt(radicand))


@dataclass(frozen=True)
class WorldlineIntegral:
    t_start: float
    t_end: float
    coordinate_elapsed: float
    proper_elapsed: float
    mean_dtau_dt: float
    final_dtau_dt: float
    n_samples: int


@dataclass(frozen=True)
class TwoWorldlineComparison:
    a: WorldlineIntegral
    b: WorldlineIntegral
    delta_tau_a_minus_b: float
    accumulated_ratio_a_over_b: float
    final_gamma_a_over_b: float


@dataclass(frozen=True)
class WorldlineHubbleResult:
    gamma0_a_over_b: float
    hubble_test: GammaTest


def load_worldline_csv(path: str | Path) -> list[ADMWorldlineSample]:
    """Load an ADM worldline sampled at monotonically increasing t.

    Units must be mutually consistent and use c=1.  The coordinate velocities
    are dx^i/dt in the same coordinate units used by the ADM metric.
    """
    samples: list[ADMWorldlineSample] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        missing = [c for c in _REQUIRED_COLUMNS if c not in fields]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        for row in reader:
            gamma = np.array([
                [float(row["gxx"]), float(row["gxy"]), float(row["gxz"])],
                [float(row["gxy"]), float(row["gyy"]), float(row["gyz"])],
                [float(row["gxz"]), float(row["gyz"]), float(row["gzz"])],
            ], dtype=float)
            sample = ADMWorldlineSample(
                t=float(row["t"]),
                alpha=float(row["alpha"]),
                beta=np.array([
                    float(row["betax"]), float(row["betay"]), float(row["betaz"])
                ], dtype=float),
                gamma=gamma,
                velocity=np.array([
                    float(row["vx"]), float(row["vy"]), float(row["vz"])
                ], dtype=float),
            )
            sample.validate()
            samples.append(sample)
    _validate_sequence(samples)
    return samples


def _validate_sequence(samples: Sequence[ADMWorldlineSample]) -> None:
    if len(samples) < 2:
        raise ValueError("Need at least two worldline samples")
    times = np.array([s.t for s in samples], dtype=float)
    if np.any(np.diff(times) <= 0):
        raise ValueError("worldline sample times must be strictly increasing")
    for s in samples:
        s.validate()


def integrate_worldline(samples: Sequence[ADMWorldlineSample]) -> WorldlineIntegral:
    _validate_sequence(samples)
    t = np.array([s.t for s in samples], dtype=float)
    rates = np.array([s.dtau_dt() for s in samples], dtype=float)
    tau = float(np.trapezoid(rates, t))
    elapsed = float(t[-1] - t[0])
    return WorldlineIntegral(
        t_start=float(t[0]),
        t_end=float(t[-1]),
        coordinate_elapsed=elapsed,
        proper_elapsed=tau,
        mean_dtau_dt=float(tau / elapsed),
        final_dtau_dt=float(rates[-1]),
        n_samples=len(samples),
    )


def compare_worldlines(
    samples_a: Sequence[ADMWorldlineSample],
    samples_b: Sequence[ADMWorldlineSample],
    endpoint_tolerance: float = 1e-10,
) -> TwoWorldlineComparison:
    """Compare two paths sampled between the same coordinate-time endpoints.

    Equal coordinate endpoints are only a numerical synchronization check; a
    physical analysis should additionally ensure that those endpoints lie on
    common invariantly defined boundary hypersurfaces.
    """
    ia = integrate_worldline(samples_a)
    ib = integrate_worldline(samples_b)
    if abs(ia.t_start - ib.t_start) > endpoint_tolerance:
        raise ValueError("worldlines do not share the same starting t")
    if abs(ia.t_end - ib.t_end) > endpoint_tolerance:
        raise ValueError("worldlines do not share the same ending t")
    gamma_final = ia.final_dtau_dt / ib.final_dtau_dt
    return TwoWorldlineComparison(
        a=ia,
        b=ib,
        delta_tau_a_minus_b=float(ia.proper_elapsed - ib.proper_elapsed),
        accumulated_ratio_a_over_b=float(ia.proper_elapsed / ib.proper_elapsed),
        final_gamma_a_over_b=float(gamma_final),
    )


def couple_comparison_to_hubble(
    comparison: TwoWorldlineComparison,
    hubble_auditor: HubbleClosureAuditor,
) -> WorldlineHubbleResult:
    """Feed the final simulation-derived relative clock rate into H0 closure.

    This retains the same phenomenological H_local = H_ref/Gamma0 mapping used
    by cpg_hubble_closure.py.  It must not be interpreted as a complete
    cosmological observable calculation until the same spacetime also supplies
    the redshift/light-cone map.
    """
    gamma0 = comparison.final_gamma_a_over_b
    return WorldlineHubbleResult(
        gamma0_a_over_b=gamma0,
        hubble_test=hubble_auditor.evaluate_gamma(gamma0),
    )


def make_uniform_worldline(
    times: Iterable[float],
    alpha: float = 1.0,
    beta=(0.0, 0.0, 0.0),
    gamma=None,
    velocity=(0.0, 0.0, 0.0),
) -> list[ADMWorldlineSample]:
    """Convenience generator for tests and analytic validation fixtures."""
    gamma_arr = np.eye(3) if gamma is None else np.asarray(gamma, dtype=float)
    beta_arr = np.asarray(beta, dtype=float)
    velocity_arr = np.asarray(velocity, dtype=float)
    return [
        ADMWorldlineSample(
            t=float(t), alpha=float(alpha), beta=beta_arr.copy(),
            gamma=gamma_arr.copy(), velocity=velocity_arr.copy(),
        )
        for t in times
    ]
