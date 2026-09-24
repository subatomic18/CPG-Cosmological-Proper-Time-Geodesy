import math

from cpg_hubble_closure import H0Measurement, HubbleClosureAuditor
from cpg_two_congruence import (
    WeakFieldState,
    TwoCongruenceAuditor,
    weak_field_clock_rate,
    relative_gamma,
)

c = 299792.458

# Identical histories must close exactly.
same = WeakFieldState("same", 0.0, 0.0)
assert math.isclose(relative_gamma(same, same), 1.0, abs_tol=1e-15)

# Weak-field formula check.
v = 220.0
dense_phi = -(v / c) ** 2
dense = WeakFieldState("galaxy", dense_phi, v)
expected_rate = 1.0 + dense_phi - 0.5 * (v / c) ** 2
assert math.isclose(weak_field_clock_rate(dense), expected_rate, rel_tol=0, abs_tol=1e-15)

# A deliberately favorable large-void benchmark.
void = WeakFieldState("void", 3.2e-5, 0.0)
gamma = relative_gamma(dense, void)
assert 0.99996 < gamma < 0.99998

aud = TwoCongruenceAuditor()
acc = aud.integrate_scaled_present_states(dense, void, z_start=2.0)
assert acc.delta_tau_a_minus_b_years < 0
assert 1.5e5 < abs(acc.delta_tau_a_minus_b_years) < 2.0e5

# Connect the present physical clock ratio to the Hubble-closure diagnostic.
hubble = HubbleClosureAuditor(
    H0Measurement("reference", 67.4, 0.5),
    H0Measurement("local", 73.18, 0.88),
)
out = aud.couple_present_to_hubble(dense, void, hubble)
assert math.isclose(out.gamma0_a_over_b, gamma, rel_tol=1e-15)
assert 67.401 < out.hubble_test.h0_predicted_local < 67.404
assert out.hubble_test.gap_closed_fraction < 0.001

# Rich-cluster benchmark should be a larger effect but still tiny.
vc = 1000.0
cluster = WeakFieldState("cluster", -(vc / c) ** 2, vc)
cluster_out = aud.couple_present_to_hubble(cluster, void, hubble)
assert cluster_out.gamma0_a_over_b < out.gamma0_a_over_b
assert cluster_out.hubble_test.h0_predicted_local > out.hubble_test.h0_predicted_local
assert cluster_out.hubble_test.gap_closed_fraction < 0.001

print("TWO-CONGRUENCE TESTS PASS")
print("galaxy vs void:", acc)
print("galaxy Hubble:", out)
print("cluster Hubble:", cluster_out)
