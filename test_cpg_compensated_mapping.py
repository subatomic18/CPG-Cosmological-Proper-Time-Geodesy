import math

from cpg_compensated_mapping import (
    compensation_amplitude_for_closure,
    epsilon0_from_h0,
    evaluate_candidate,
)


def test_epsilon0_matches_h0_ratio():
    h0_late = 73.04342157543183
    h0_early = 67.26899612716719
    eps = epsilon0_from_h0(h0_late, h0_early)
    assert math.isclose(1.0 + eps, h0_late / h0_early, rel_tol=1e-12)


def test_compensation_amplitude_is_positive():
    eps = epsilon0_from_h0(73.04342157543183, 67.26899612716719)
    amp = compensation_amplitude_for_closure(
        eps,
        center_z=2.0,
        sigma_log1pz=0.5,
    )
    assert amp > 0.0


def test_candidate_closes_at_cmb():
    c = evaluate_candidate(
        73.04342157543183,
        67.26899612716719,
        center_z=2.0,
        sigma_log1pz=0.5,
    )
    assert abs(c.cmb_log_shift) < 1e-8
    assert abs(c.cmb_fractional_1plusz_shift) < 1e-8
