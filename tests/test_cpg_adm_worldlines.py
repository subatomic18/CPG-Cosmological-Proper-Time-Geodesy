import math
import numpy as np

from cpg_adm_worldlines import (
    ADMWorldlineSample,
    compare_worldlines,
    integrate_worldline,
    make_uniform_worldline,
)

# Minkowski rest clock: d tau / dt = 1.
times = np.linspace(0.0, 10.0, 101)
rest = make_uniform_worldline(times)
out = integrate_worldline(rest)
assert math.isclose(out.proper_elapsed, 10.0, rel_tol=0, abs_tol=1e-12)
assert math.isclose(out.mean_dtau_dt, 1.0, rel_tol=0, abs_tol=1e-12)

# Constant coordinate speed in Minkowski space should reproduce SR gamma factor.
v = 0.3
moving = make_uniform_worldline(times, velocity=(v, 0.0, 0.0))
out_m = integrate_worldline(moving)
expected_rate = math.sqrt(1.0 - v*v)
assert math.isclose(out_m.mean_dtau_dt, expected_rate, rel_tol=1e-12)
assert math.isclose(out_m.proper_elapsed, 10.0 * expected_rate, rel_tol=1e-12)

cmp = compare_worldlines(moving, rest)
assert math.isclose(cmp.final_gamma_a_over_b, expected_rate, rel_tol=1e-12)
assert math.isclose(cmp.delta_tau_a_minus_b,
                    10.0 * (expected_rate - 1.0), rel_tol=1e-12)

# Constant lapse with zero shift/velocity: d tau / dt = alpha.
slow = make_uniform_worldline(times, alpha=0.95)
out_s = integrate_worldline(slow)
assert math.isclose(out_s.proper_elapsed, 9.5, rel_tol=1e-12)

# Shift contribution follows the documented ADM convention.
s = ADMWorldlineSample(
    t=0.0,
    alpha=1.0,
    beta=np.array([0.1, 0.0, 0.0]),
    gamma=np.eye(3),
    velocity=np.zeros(3),
)
assert math.isclose(s.dtau_dt(), math.sqrt(0.99), rel_tol=1e-12)

# Non-timelike samples must fail.
bad = make_uniform_worldline(times, velocity=(1.0, 0.0, 0.0))
try:
    integrate_worldline(bad)
    raise AssertionError("null worldline should fail")
except ValueError:
    pass

print("ADM WORLDLINE TESTS PASS")
print(out)
print(out_m)
print(cmp)
