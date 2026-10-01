import math
from cpg_early_window import EarlyTimeWindowAuditor

A = EarlyTimeWindowAuditor()
r = A.audit_target_extra_time(14.32, 26.7, 0.020, z_star=1089.8)

assert math.isclose(r.reference_elapsed_gyr, 0.168657299682101, rel_tol=2e-10)
assert math.isclose(r.extra_elapsed_gyr, 0.020, rel_tol=2e-10, abs_tol=1e-12)
assert math.isclose(r.h_factor, 0.8939876695272263, rel_tol=2e-10)
assert math.isclose(r.delta_distance_mpc, 122.69803833820879, rel_tol=2e-10)
assert math.isclose(r.fractional_distance_shift, 0.008849913427349865, rel_tol=2e-10)
assert math.isclose(r.theta_ratio_fixed_rs, 0.9912277204869016, rel_tol=2e-10)

print("EARLY-WINDOW TESTS PASS")
print(r)
