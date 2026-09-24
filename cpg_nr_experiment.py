#!/usr/bin/env python3
"""Run a two-worldline CPG experiment on Mescaline/Einstein-Toolkit snapshots.

The first snapshot is used to choose an underdense (void-like) and overdense
(dense-like) starting cell from its ``rho`` field.  Both fluid elements are
then tracked through the same relativistic spacetime using the Mescaline
adapter and their proper times are compared.

This script reports dimensionless chronometric quantities by default.  It does
not claim that a final relative clock rate is by itself an observable Hubble
shift; the redshift/light-cone calculation must come from the same spacetime.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import h5py
import numpy as np

from cpg_mescaline_adapter import (
    load_mescaline_series,
    compare_mescaline_fluid_worldlines,
)


def _parse_vec(text: str) -> np.ndarray:
    vals = [float(x) for x in text.split(",")]
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    return np.asarray(vals, dtype=float)


def _load_initial_density(path: str | Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if "rho" not in f:
            raise ValueError(f"{path}: Mescaline file does not contain rho")
        rho = np.asarray(f["rho"], dtype=float)
    if rho.ndim != 3:
        raise ValueError("rho must be a 3-D field")
    if not np.any(np.isfinite(rho)):
        raise ValueError("rho has no finite values")
    return rho


def _pick_index(rho: np.ndarray, percentile: float) -> tuple[int, int, int]:
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must lie in [0,100]")
    finite = np.isfinite(rho)
    values = rho[finite]
    target = float(np.percentile(values, percentile))
    score = np.where(finite, np.abs(rho - target), np.inf)
    return tuple(int(i) for i in np.unravel_index(np.argmin(score), rho.shape))


def main() -> None:
    p = argparse.ArgumentParser(
        description="CPG numerical-relativity two-worldline proper-time experiment"
    )
    p.add_argument("--snapshots", required=True,
                   help="glob for Mescaline trip_report_3D_*.h5 files")
    p.add_argument("--dx", type=float, required=True,
                   help="uniform grid spacing used by Mescaline trip reports")
    p.add_argument("--origin", type=_parse_vec, default=np.zeros(3),
                   help="coordinate of grid index (0,0,0), as x,y,z; default 0,0,0")
    p.add_argument("--void-percentile", type=float, default=1.0,
                   help="initial density percentile for void-like path; default 1")
    p.add_argument("--dense-percentile", type=float, default=99.0,
                   help="initial density percentile for dense-like path; default 99")
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = p.parse_args()

    paths = sorted(glob.glob(args.snapshots))
    if len(paths) < 2:
        raise SystemExit("Need at least two snapshot files matching --snapshots")

    snaps = load_mescaline_series(paths)
    rho0 = _load_initial_density(snaps[0].path)
    if rho0.shape != (snaps[0].nx,) * 3:
        raise ValueError("rho shape does not match snapshot nx")

    iv = _pick_index(rho0, args.void_percentile)
    idn = _pick_index(rho0, args.dense_percentile)
    origin = np.asarray(args.origin, dtype=float)
    pos_void = origin + args.dx * np.asarray(iv, dtype=float)
    pos_dense = origin + args.dx * np.asarray(idn, dtype=float)

    result = compare_mescaline_fluid_worldlines(
        snaps, pos_void, pos_dense, args.dx, origin
    )
    comp = result["comparison"]

    frac_dense = comp.delta_tau_a_minus_b / comp.b.proper_elapsed
    mean_gamma = comp.a.mean_dtau_dt / comp.b.mean_dtau_dt

    out = {
        "n_snapshots": len(snaps),
        "t_start": comp.a.t_start,
        "t_end": comp.a.t_end,
        "coordinate_elapsed": comp.a.coordinate_elapsed,
        "void_percentile": args.void_percentile,
        "dense_percentile": args.dense_percentile,
        "void_index": list(iv),
        "dense_index": list(idn),
        "void_start_position": pos_void.tolist(),
        "dense_start_position": pos_dense.tolist(),
        "void_initial_rho": float(rho0[iv]),
        "dense_initial_rho": float(rho0[idn]),
        "tau_void": comp.a.proper_elapsed,
        "tau_dense": comp.b.proper_elapsed,
        "delta_tau_void_minus_dense": comp.delta_tau_a_minus_b,
        "delta_tau_over_tau_dense": float(frac_dense),
        "mean_gamma_void_over_dense": float(mean_gamma),
        "final_gamma_void_over_dense": comp.final_gamma_a_over_b,
        "void_final_dtau_dt": comp.a.final_dtau_dt,
        "dense_final_dtau_dt": comp.b.final_dtau_dt,
        "note": (
            "Proper-time comparison from one simulated spacetime. Interpret only "
            "with physically specified endpoint hypersurfaces; H0 mapping requires "
            "the same spacetime's light-cone/redshift calculation."
        ),
    }

    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
        return

    print("CPG NUMERICAL-RELATIVITY TWO-WORLDLINE EXPERIMENT")
    print("=" * 72)
    print(f"snapshots: {out['n_snapshots']}")
    print(f"coordinate interval: {out['coordinate_elapsed']:.12g}")
    print(f"void cell:  {iv}, rho0={out['void_initial_rho']:.12g}")
    print(f"dense cell: {idn}, rho0={out['dense_initial_rho']:.12g}")
    print(f"tau_void:  {out['tau_void']:.12g}")
    print(f"tau_dense: {out['tau_dense']:.12g}")
    print(f"Delta tau (void-dense): {out['delta_tau_void_minus_dense']:.12g}")
    print(f"Delta tau / tau_dense: {100.0*frac_dense:.8g}%")
    print(f"mean Gamma_void/dense:  {mean_gamma:.12g}")
    print(f"final Gamma_void/dense: {comp.final_gamma_a_over_b:.12g}")
    print("NOTE: this is a proper-time diagnostic, not by itself an H0 prediction.")


if __name__ == "__main__":
    main()
