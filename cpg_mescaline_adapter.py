#!/usr/bin/env python3
"""Adapter from Mescaline/Einstein-Toolkit HDF5 snapshots to CPG worldlines.

This module targets the public ``mescaline-1.0`` trip-report format used for
post-processing Einstein Toolkit cosmological simulations.  Mescaline writes
one HDF5 file per simulation time with the datasets

    vel[0], vel[1], vel[2]
    g_xx, g_xy, g_xz, g_yy, g_yz, g_zz
    alpha

and assumes a vanishing shift vector.  The velocity fields are HydroBase
Eulerian three-velocities, NOT coordinate velocities.  HydroBase defines

    v^i = u^i/(alpha u^0) + beta^i/alpha,

so for beta^i = 0 the coordinate velocity is

    dx^i/dt = alpha v^i.

For a fluid worldline the corresponding proper-time rate is therefore

    d tau/dt = alpha * sqrt(1 - gamma_ij v^i v^j).

The adapter tracks a fluid element through a time-ordered sequence of uniform,
periodic Mescaline snapshots using trilinear interpolation and a Heun
(predictor-corrector) update.  It returns CPG ADMWorldlineSample objects so the
same proper-time and Hubble-closure machinery can be used downstream.

Important limitations
---------------------
* Mescaline trip reports do not store spatial grid spacing/origin, so these
  must be supplied from the simulation metadata or parameter file.
* The public Mescaline implementation assumes zero shift; this adapter does
  the same.  Raw ET data with non-zero shift should be handled by a dedicated
  raw-output adapter instead.
* Equal coordinate-time endpoints are only a numerical synchronization
  condition.  A physical RTD-EU comparison still requires invariantly defined
  endpoint hypersurfaces.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

from cpg_adm_worldlines import ADMWorldlineSample, compare_worldlines


_DATASETS = (
    "vel[0]", "vel[1]", "vel[2]",
    "g_xx", "g_xy", "g_xz", "g_yy", "g_yz", "g_zz",
    "alpha",
)


@dataclass(frozen=True)
class MescalineSnapshot:
    path: Path
    time: float
    nx: int
    alpha: np.ndarray
    velocity: np.ndarray          # shape (3,nx,nx,nx), Eulerian v^i
    gamma: np.ndarray             # shape (3,3,nx,nx,nx)

    def validate(self) -> None:
        if self.nx < 2:
            raise ValueError("snapshot nx must be >= 2")
        shape3 = (self.nx, self.nx, self.nx)
        if self.alpha.shape != shape3:
            raise ValueError(f"alpha shape {self.alpha.shape} != {shape3}")
        if self.velocity.shape != (3,) + shape3:
            raise ValueError("velocity must have shape (3,nx,nx,nx)")
        if self.gamma.shape != (3, 3) + shape3:
            raise ValueError("gamma must have shape (3,3,nx,nx,nx)")
        if not np.isfinite(self.time):
            raise ValueError("snapshot time must be finite")
        if not np.all(np.isfinite(self.alpha)) or np.any(self.alpha <= 0):
            raise ValueError("alpha must be finite and positive")
        if not np.all(np.isfinite(self.velocity)) or not np.all(np.isfinite(self.gamma)):
            raise ValueError("snapshot fields must be finite")


def _scalar_attr(value) -> float:
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError("expected scalar HDF5 attribute")
    return float(arr.reshape(-1)[0])


def load_mescaline_snapshot(path: str | Path) -> MescalineSnapshot:
    """Load one Mescaline ``trip_report_3D_itXXXXXX.h5`` snapshot."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        missing = [name for name in _DATASETS if name not in f]
        if missing:
            raise ValueError(f"{path}: missing Mescaline datasets {missing}")
        if "Global Attributes" not in f:
            raise ValueError(f"{path}: missing 'Global Attributes' dataset")
        attrs = f["Global Attributes"].attrs
        if "time" not in attrs or "nx" not in attrs:
            raise ValueError(f"{path}: global attributes must contain time and nx")
        time = _scalar_attr(attrs["time"])
        nx = int(round(_scalar_attr(attrs["nx"])))

        alpha = np.asarray(f["alpha"], dtype=float)
        vel = np.stack([
            np.asarray(f["vel[0]"], dtype=float),
            np.asarray(f["vel[1]"], dtype=float),
            np.asarray(f["vel[2]"], dtype=float),
        ])
        gamma = np.empty((3, 3, nx, nx, nx), dtype=float)
        gamma[0, 0] = np.asarray(f["g_xx"], dtype=float)
        gamma[0, 1] = gamma[1, 0] = np.asarray(f["g_xy"], dtype=float)
        gamma[0, 2] = gamma[2, 0] = np.asarray(f["g_xz"], dtype=float)
        gamma[1, 1] = np.asarray(f["g_yy"], dtype=float)
        gamma[1, 2] = gamma[2, 1] = np.asarray(f["g_yz"], dtype=float)
        gamma[2, 2] = np.asarray(f["g_zz"], dtype=float)

    snap = MescalineSnapshot(path, time, nx, alpha, vel, gamma)
    snap.validate()
    return snap


def load_mescaline_series(paths: Sequence[str | Path]) -> list[MescalineSnapshot]:
    snaps = [load_mescaline_snapshot(p) for p in paths]
    snaps.sort(key=lambda s: s.time)
    if len(snaps) < 2:
        raise ValueError("need at least two Mescaline snapshots")
    times = np.array([s.time for s in snaps])
    if np.any(np.diff(times) <= 0):
        raise ValueError("snapshot times must be strictly increasing")
    nx = snaps[0].nx
    if any(s.nx != nx for s in snaps):
        raise ValueError("all snapshots must have the same nx")
    return snaps


def _periodic_trilinear(field: np.ndarray, pos: np.ndarray,
                        origin: np.ndarray, dx: float) -> np.ndarray:
    """Trilinearly interpolate scalar/vector/tensor field on periodic cube.

    ``field`` must end in the three spatial axes ``(...,nx,nx,nx)``.
    ``pos`` and ``origin`` are coordinate vectors in the same units as ``dx``.
    """
    if dx <= 0:
        raise ValueError("dx must be positive")
    nx = field.shape[-1]
    if field.shape[-3:] != (nx, nx, nx):
        raise ValueError("field spatial axes must be cubic")
    q = (np.asarray(pos, dtype=float) - origin) / dx
    q = np.mod(q, nx)
    i0 = np.floor(q).astype(int)
    f = q - i0
    i1 = (i0 + 1) % nx

    out = np.zeros(field.shape[:-3], dtype=float)
    for ax in (0, 1):
        wx = (1.0 - f[0]) if ax == 0 else f[0]
        ix = i0[0] if ax == 0 else i1[0]
        for ay in (0, 1):
            wy = (1.0 - f[1]) if ay == 0 else f[1]
            iy = i0[1] if ay == 0 else i1[1]
            for az in (0, 1):
                wz = (1.0 - f[2]) if az == 0 else f[2]
                iz = i0[2] if az == 0 else i1[2]
                out = out + wx * wy * wz * field[..., ix, iy, iz]
    return out


def sample_fluid_state(snapshot: MescalineSnapshot, pos, origin, dx):
    """Interpolate alpha, gamma_ij and HydroBase Eulerian velocity at ``pos``."""
    origin = np.asarray(origin, dtype=float)
    pos = np.asarray(pos, dtype=float)
    alpha = float(_periodic_trilinear(snapshot.alpha, pos, origin, dx))
    velocity = np.asarray(_periodic_trilinear(snapshot.velocity, pos, origin, dx))
    gamma = np.asarray(_periodic_trilinear(snapshot.gamma, pos, origin, dx))
    gamma = 0.5 * (gamma + gamma.T)
    if np.linalg.eigvalsh(gamma).min() <= 0:
        raise ValueError("interpolated spatial metric is not positive definite")
    v2 = float(velocity @ gamma @ velocity)
    if v2 >= 1.0:
        raise ValueError("interpolated Eulerian fluid velocity is non-timelike")
    if alpha <= 0:
        raise ValueError("interpolated lapse is non-positive")
    return alpha, gamma, velocity


def coordinate_velocity_zero_shift(alpha: float, eulerian_velocity: np.ndarray) -> np.ndarray:
    """HydroBase v^i -> dx^i/dt for the Mescaline beta^i=0 assumption."""
    return float(alpha) * np.asarray(eulerian_velocity, dtype=float)


def track_fluid_worldline(snaps: Sequence[MescalineSnapshot], start_pos,
                          dx: float, origin=(0.0, 0.0, 0.0)):
    """Track one fluid element and return positions plus CPG ADM samples.

    The integration uses a Heun predictor-corrector across each snapshot pair.
    Metric quantities recorded in the returned ADM sample are interpolated at
    the tracked position at each snapshot.  Because Mescaline assumes zero
    shift, the ADM beta vector is set to zero and the coordinate velocity is
    alpha*v_Eulerian.
    """
    if len(snaps) < 2:
        raise ValueError("need at least two snapshots")
    origin = np.asarray(origin, dtype=float)
    if origin.shape != (3,):
        raise ValueError("origin must have shape (3,)")
    pos = np.asarray(start_pos, dtype=float).copy()
    if pos.shape != (3,):
        raise ValueError("start_pos must have shape (3,)")
    nx = snaps[0].nx
    box = nx * float(dx)
    if box <= 0:
        raise ValueError("dx must be positive")

    positions = [pos.copy()]
    samples: list[ADMWorldlineSample] = []

    for n, snap in enumerate(snaps):
        alpha, gamma, v_eul = sample_fluid_state(snap, pos, origin, dx)
        v_coord = coordinate_velocity_zero_shift(alpha, v_eul)
        samples.append(ADMWorldlineSample(
            t=float(snap.time), alpha=alpha, beta=np.zeros(3),
            gamma=gamma, velocity=v_coord,
        ))
        if n == len(snaps) - 1:
            break

        nxt = snaps[n + 1]
        dt = float(nxt.time - snap.time)
        if dt <= 0:
            raise ValueError("snapshot times must be strictly increasing")

        # Predictor with fields at the current snapshot.
        pred = pos + dt * v_coord
        pred = origin + np.mod(pred - origin, box)

        # Corrector with the next snapshot at the predicted position.
        alpha2, _, v_eul2 = sample_fluid_state(nxt, pred, origin, dx)
        v_coord2 = coordinate_velocity_zero_shift(alpha2, v_eul2)
        pos = pos + 0.5 * dt * (v_coord + v_coord2)
        pos = origin + np.mod(pos - origin, box)
        positions.append(pos.copy())

    return np.asarray(positions), samples


def compare_mescaline_fluid_worldlines(snaps: Sequence[MescalineSnapshot],
                                        start_a, start_b, dx: float,
                                        origin=(0.0, 0.0, 0.0)):
    """Track and compare two fluid worldlines through the same snapshot series."""
    pos_a, samples_a = track_fluid_worldline(snaps, start_a, dx, origin)
    pos_b, samples_b = track_fluid_worldline(snaps, start_b, dx, origin)
    comparison = compare_worldlines(samples_a, samples_b)
    return {
        "positions_a": pos_a,
        "positions_b": pos_b,
        "samples_a": samples_a,
        "samples_b": samples_b,
        "comparison": comparison,
    }
