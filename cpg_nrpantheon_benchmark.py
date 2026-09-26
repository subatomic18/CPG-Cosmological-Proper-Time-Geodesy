#!/usr/bin/env python3
"""Benchmark observer/sky-sampling H0 variation with public NRPantheon data.

NRPantheon catalogs are generated from a fully nonlinear numerical-relativity
large-scale-structure simulation with general-relativistic ray tracing.  Each
public file contains columns

    z, mu, dL, theta, phi

where dL is in Mpc/h.  For the Einstein-de Sitter background used by the
fiducial simulation, the exact luminosity-distance normalization is

    dL(z; H) = 2 c / H * [1 + z - sqrt(1+z)],

where H here is H0/h in km s^-1 Mpc^-1.

This module fits only that common normalization.  The default estimator is the
least-squares mean in log-distance (equivalently equal-weight distance modulus)
for each catalog.  It is deliberately a transparent CPG benchmark and is NOT
an exact reproduction of Macpherson (2024), which uses the Pantheon covariance
and a specified cosmographic fit.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import glob
import json
import math
import re

import numpy as np

C_LIGHT_KM_S = 299792.458
_FILENAME = re.compile(r"NRPantheon_obs(?P<obs>\d+)_rot(?P<rot>\d+)\.dat$")


@dataclass(frozen=True)
class CatalogFit:
    path: str
    observer: int
    rotation: int
    n: int
    h0_over_h: float
    rms_mu_residual_mag: float


@dataclass(frozen=True)
class NRPantheonBenchmark:
    n_catalogs: int
    n_observers: int
    rotations_per_observer: tuple[int, ...]
    truth_h0_over_h: float
    mean_catalog_h0_over_h: float
    std_catalog_h0_over_h: float
    min_catalog_h0_over_h: float
    max_catalog_h0_over_h: float
    max_catalog_span_percent_of_truth: float
    mean_observer_h0_over_h: float
    std_observer_mean_h0_over_h: float
    min_observer_mean_h0_over_h: float
    max_observer_mean_h0_over_h: float
    observer_mean_span_percent_of_truth: float
    mean_within_observer_rotation_std: float
    max_within_observer_rotation_span: float
    max_within_observer_rotation_span_percent_of_truth: float
    catalog_fits: tuple[CatalogFit, ...]
    observer_summaries: tuple[dict, ...]

    def to_dict(self) -> dict:
        out = asdict(self)
        out["scope_note"] = (
            "Equal-weight log-distance EdS normalization benchmark on public "
            "NRPantheon catalogs. Not an exact reproduction of the published "
            "Pantheon-covariance likelihood analysis and not a proper-time test."
        )
        return out


def eds_shape(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if np.any(z <= 0):
        raise ValueError("all redshifts must be positive")
    return 1.0 + z - np.sqrt(1.0 + z)


def eds_luminosity_distance(z: np.ndarray, h0_over_h: float) -> np.ndarray:
    if h0_over_h <= 0:
        raise ValueError("h0_over_h must be positive")
    return 2.0 * C_LIGHT_KM_S * eds_shape(z) / float(h0_over_h)


def fit_catalog(path: str | Path) -> CatalogFit:
    path = Path(path)
    m = _FILENAME.search(path.name)
    if m is None:
        raise ValueError(f"unrecognized NRPantheon filename: {path.name}")
    data = np.loadtxt(path, comments="#", dtype=float)
    if data.ndim != 2 or data.shape[1] < 3 or len(data) < 2:
        raise ValueError(f"{path}: expected >=2 rows and >=3 columns")
    z = data[:, 0]
    mu = data[:, 1]
    dl = data[:, 2]
    if np.any(z <= 0) or np.any(dl <= 0):
        raise ValueError(f"{path}: z and dL must be positive")

    # ln dL_model = ln[2 c f(z)] - ln H.  Equal-weight least squares
    # in ln(dL) therefore has an analytic normalization solution.
    log_h = float(np.mean(np.log(2.0 * C_LIGHT_KM_S * eds_shape(z)) - np.log(dl)))
    hfit = math.exp(log_h)
    dl_model = eds_luminosity_distance(z, hfit)
    mu_model = 5.0 * np.log10(dl_model) + 25.0
    rms_mu = float(np.sqrt(np.mean((mu - mu_model) ** 2)))
    return CatalogFit(
        path=str(path),
        observer=int(m.group("obs")),
        rotation=int(m.group("rot")),
        n=int(len(z)),
        h0_over_h=float(hfit),
        rms_mu_residual_mag=rms_mu,
    )


def benchmark(paths, truth_h0_over_h: float = 100.001) -> NRPantheonBenchmark:
    fits = tuple(sorted((fit_catalog(p) for p in paths),
                        key=lambda f: (f.observer, f.rotation)))
    if not fits:
        raise ValueError("no NRPantheon catalogs supplied")
    by_obs: dict[int, list[CatalogFit]] = {}
    for f in fits:
        by_obs.setdefault(f.observer, []).append(f)

    obs_rows = []
    within_stds = []
    within_spans = []
    for obs in sorted(by_obs):
        values = np.array([f.h0_over_h for f in by_obs[obs]], dtype=float)
        span = float(np.ptp(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        within_stds.append(std)
        within_spans.append(span)
        obs_rows.append({
            "observer": int(obs),
            "n_rotations": int(len(values)),
            "mean_h0_over_h": float(np.mean(values)),
            "std_rotation_h0_over_h": std,
            "min_rotation_h0_over_h": float(np.min(values)),
            "max_rotation_h0_over_h": float(np.max(values)),
            "rotation_span_h0_over_h": span,
            "rotation_span_percent_of_truth": 100.0 * span / truth_h0_over_h,
        })

    cat = np.array([f.h0_over_h for f in fits], dtype=float)
    obsmeans = np.array([r["mean_h0_over_h"] for r in obs_rows], dtype=float)
    rotations = tuple(sorted({len(v) for v in by_obs.values()}))
    return NRPantheonBenchmark(
        n_catalogs=len(fits),
        n_observers=len(by_obs),
        rotations_per_observer=rotations,
        truth_h0_over_h=float(truth_h0_over_h),
        mean_catalog_h0_over_h=float(np.mean(cat)),
        std_catalog_h0_over_h=float(np.std(cat, ddof=1)) if len(cat) > 1 else 0.0,
        min_catalog_h0_over_h=float(np.min(cat)),
        max_catalog_h0_over_h=float(np.max(cat)),
        max_catalog_span_percent_of_truth=100.0 * float(np.ptp(cat)) / truth_h0_over_h,
        mean_observer_h0_over_h=float(np.mean(obsmeans)),
        std_observer_mean_h0_over_h=float(np.std(obsmeans, ddof=1)) if len(obsmeans) > 1 else 0.0,
        min_observer_mean_h0_over_h=float(np.min(obsmeans)),
        max_observer_mean_h0_over_h=float(np.max(obsmeans)),
        observer_mean_span_percent_of_truth=100.0 * float(np.ptp(obsmeans)) / truth_h0_over_h,
        mean_within_observer_rotation_std=float(np.mean(within_stds)),
        max_within_observer_rotation_span=float(np.max(within_spans)),
        max_within_observer_rotation_span_percent_of_truth=(
            100.0 * float(np.max(within_spans)) / truth_h0_over_h
        ),
        catalog_fits=fits,
        observer_summaries=tuple(obs_rows),
    )


def main() -> int:
    p = argparse.ArgumentParser(description="CPG public NRPantheon effective-H0 benchmark")
    p.add_argument("patterns", nargs="+", help="catalog paths or glob patterns")
    p.add_argument("--truth", type=float, default=100.001,
                   help="global simulation H0/h reference [km/s/Mpc]")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    paths = []
    for pattern in args.patterns:
        matches = glob.glob(pattern)
        paths.extend(matches if matches else [pattern])
    result = benchmark(paths, truth_h0_over_h=args.truth)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0

    print("CPG NRPANTHEON EFFECTIVE-H0 BENCHMARK")
    print("=" * 72)
    print(f"catalogs / observers     : {result.n_catalogs} / {result.n_observers}")
    print(f"rotations per observer   : {result.rotations_per_observer}")
    print(f"truth H0/h               : {result.truth_h0_over_h:.6f}")
    print(f"mean catalog H0/h        : {result.mean_catalog_h0_over_h:.6f}")
    print(f"catalog std              : {result.std_catalog_h0_over_h:.6f}")
    print(f"catalog min..max         : {result.min_catalog_h0_over_h:.6f} .. {result.max_catalog_h0_over_h:.6f}")
    print(f"full catalog span/truth  : {result.max_catalog_span_percent_of_truth:.3f}%")
    print(f"observer-mean std        : {result.std_observer_mean_h0_over_h:.6f}")
    print(f"observer means min..max  : {result.min_observer_mean_h0_over_h:.6f} .. {result.max_observer_mean_h0_over_h:.6f}")
    print(f"observer span/truth      : {result.observer_mean_span_percent_of_truth:.3f}%")
    print(f"mean within-obs rot std  : {result.mean_within_observer_rotation_std:.6f}")
    print(f"max within-obs rot span  : {result.max_within_observer_rotation_span:.6f} ({result.max_within_observer_rotation_span_percent_of_truth:.3f}%)")
    print("NOTE: equal-weight EdS normalization benchmark, not published covariance fit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
