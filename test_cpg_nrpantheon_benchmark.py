import math
from pathlib import Path

import numpy as np

from cpg_nrpantheon_benchmark import (
    C_LIGHT_KM_S,
    benchmark,
    eds_luminosity_distance,
    fit_catalog,
)


def _write_catalog(path: Path, observer: int, rotation: int, h0: float):
    z = np.array([0.03, 0.05, 0.08, 0.12, 0.15])
    dl = eds_luminosity_distance(z, h0)
    mu = 5.0 * np.log10(dl) + 25.0
    theta = np.linspace(0.2, 2.2, len(z))
    phi = np.linspace(0.1, 5.0, len(z))
    arr = np.column_stack((z, mu, dl, theta, phi))
    np.savetxt(path / f"NRPantheon_obs{observer:04d}_rot{rotation:02d}.dat", arr,
               header="z, mu, dL, theta, phi")


def test_exact_eds_catalog_recovers_normalization(tmp_path):
    _write_catalog(tmp_path, 1, 1, 101.25)
    f = fit_catalog(tmp_path / "NRPantheon_obs0001_rot01.dat")
    assert math.isclose(f.h0_over_h, 101.25, rel_tol=1e-12)
    assert f.rms_mu_residual_mag < 1e-12


def test_benchmark_summarizes_observer_and_rotation_scatter(tmp_path):
    for obs, vals in {1: [99.0, 101.0], 2: [102.0, 104.0]}.items():
        for rot, h0 in enumerate(vals, 1):
            _write_catalog(tmp_path, obs, rot, h0)
    paths = sorted(tmp_path.glob("NRPantheon_*.dat"))
    out = benchmark(paths, truth_h0_over_h=100.0)
    assert out.n_catalogs == 4
    assert out.n_observers == 2
    assert out.rotations_per_observer == (2,)
    assert math.isclose(out.mean_catalog_h0_over_h, 101.5, rel_tol=1e-12)
    assert math.isclose(out.min_observer_mean_h0_over_h, 100.0, rel_tol=1e-12)
    assert math.isclose(out.max_observer_mean_h0_over_h, 103.0, rel_tol=1e-12)
    assert math.isclose(out.observer_mean_span_percent_of_truth, 3.0, rel_tol=1e-12)
    assert math.isclose(out.max_within_observer_rotation_span_percent_of_truth, 2.0, rel_tol=1e-12)
