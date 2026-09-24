import numpy as np
import pytest

from cpg_cmb_closure import (
    CMBAcousticClosureResult,
    CMBCompressedInputs,
    published_central_value_check,
)


def test_published_central_value_check_passes_small_offset():
    result = CMBAcousticClosureResult(
        h0=67.31,
        theta_star_100_target=1.04109,
        theta_star_100_model=1.04109,
        z_star=1089.9,
        r_star_Mpc=144.4,
        DA_star_Mpc=12.7,
        age_Gyr=13.8,
        ombh2=0.02236,
        omch2=0.1202,
        reference_h0=67.27,
        reference_sigma=0.60,
    )
    check = published_central_value_check(result, tolerance_km_s_Mpc=0.5)
    assert check["within_tolerance"] is True
    assert np.isclose(check["delta_h0"], 0.04)


def test_input_dataclass_keeps_physical_densities():
    x = CMBCompressedInputs(
        ombh2=0.02236,
        omch2=0.1202,
        theta_star_100=1.04109,
    )
    assert np.isclose(x.ombh2, 0.02236)
    assert np.isclose(x.omch2, 0.1202)
    assert np.isclose(x.theta_star_100, 1.04109)


def test_camb_solver_optional_dependency_message():
    # This test only checks that the public solver exists and can be imported.
    # The GitHub Actions integration workflow performs the actual CAMB run.
    from cpg_cmb_closure import solve_h0_from_acoustic_scale

    assert callable(solve_h0_from_acoustic_scale)
