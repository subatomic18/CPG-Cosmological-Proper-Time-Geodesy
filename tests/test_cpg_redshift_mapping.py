import numpy as np

from cpg_redshift_mapping import (
    epsilon_from_h0,
    geometric_redshift,
    observed_redshift,
    run_stress_test,
)


def test_mapping_round_trip():
    epsilon = 0.05
    for z in (0.01, 0.1, 1.0, 10.0, 1000.0):
        z_obs = observed_redshift(z, epsilon)
        z_back = geometric_redshift(z_obs, epsilon)
        assert np.isclose(z_back, z, rtol=0, atol=1e-12)


def test_epsilon_matches_local_hubble_ratio():
    late = 73.04342157543183
    early = 67.26899612716719
    epsilon = epsilon_from_h0(late, early)
    assert np.isclose(1.0 + epsilon, late / early)


def test_stress_result_has_large_cmb_distortion():
    result = run_stress_test(
        h0_late=73.04342157543183,
        h0_early=67.26899612716719,
        z_star=1089.938,
    )
    assert result.epsilon > 0.08
    assert result.z_star_observed_if_geometric_standard > 1900.0
    assert result.z_star_geometric_if_observed_standard < 700.0
