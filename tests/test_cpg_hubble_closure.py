import math

from cpg_hubble_closure import H0Measurement, HubbleClosureAuditor

ref = H0Measurement("reference", 67.4, 0.5)
local = H0Measurement("local", 73.18, 0.88)
aud = HubbleClosureAuditor(ref, local)

req = aud.required_closure()
assert math.isclose(req.gamma0_required, 67.4 / 73.18, rel_tol=0, abs_tol=1e-12)
assert math.isclose(req.gamma0_sigma, 0.013013307183664268, rel_tol=1e-12)
assert math.isclose(req.tension_sigma, 5.710748751966136, rel_tol=1e-12)
assert math.isclose(req.clock_departure_percent, 7.898332877835479, rel_tol=1e-12)

null = aud.evaluate_gamma(1.0)
assert math.isclose(null.h0_predicted_local, 67.4, rel_tol=0, abs_tol=1e-12)
assert math.isclose(null.residual_tension_sigma, req.tension_sigma, rel_tol=1e-12)
assert math.isclose(null.gap_closed_fraction, 0.0, abs_tol=1e-12)

five = aud.evaluate_gamma(0.95)
assert math.isclose(five.h0_predicted_local, 70.94736842105263, rel_tol=1e-12)
assert 2.1 < five.residual_tension_sigma < 2.3
assert 0.61 < five.gap_closed_fraction < 0.62

exact = aud.evaluate_gamma(req.gamma0_required)
assert math.isclose(exact.h0_predicted_local, 73.18, rel_tol=0, abs_tol=1e-12)
assert math.isclose(exact.residual_h0, 0.0, abs_tol=1e-12)
assert math.isclose(exact.gap_closed_fraction, 1.0, abs_tol=1e-12)

try:
    aud.evaluate_gamma(0.0)
    raise AssertionError("gamma0=0 should fail")
except ValueError:
    pass

print("HUBBLE CLOSURE TESTS PASS")
print(req)
print(five)
