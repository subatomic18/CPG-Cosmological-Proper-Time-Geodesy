import numpy as np

from cpg_late_onset_mapping import (
    epsilon0_from_h0,
    epsilon_profile,
    geometric_redshift,
    observed_redshift,
    positive_mapping_high_z_floor,
)


def test_epsilon0_matches_h0_ratio():
    hlate = 73.04342157543183
    hearly = 67.26899612716719
    e0 = epsilon0_from_h0(hlate, hearly)
    assert np.isclose(1.0 + e0, hlate / hearly)


def test_mapping_round_trip():
    e0 = 0.08584081494762485
    zt = 0.12
    n = 8.0
    for z in (0.01, 0.05, 0.1, 1.0, 14.32):
        zo = observed_redshift(z, e0, zt, n)
        zg = geometric_redshift(zo, e0, zt, n)
        assert np.isclose(zg, z, rtol=0.0, atol=1e-8)


def test_positive_floor_is_nonzero():
    ratio = 73.04342157543183 / 67.26899612716719
    floor = positive_mapping_high_z_floor(ratio, 0.15)
    assert floor > 0.0
    assert np.isclose(floor, 0.010418916486982566, rtol=0.0, atol=1e-12)


def test_profile_decays_with_redshift():
    e0 = 0.08
    low = epsilon_profile(0.0, e0, 0.1, 4.0)
    mid = epsilon_profile(0.1, e0, 0.1, 4.0)
    high = epsilon_profile(10.0, e0, 0.1, 4.0)
    assert low > mid > high >= 0.0
