from cpg_v0_3 import CPGEngine, HighZTransientAuditor, generate_synthetic_catalog

engine = CPGEngine()
closure = engine.verify_numerical_closure()
assert closure["passed"], closure

null_events = generate_synthetic_catalog(n=600, seed=7, b_low=1.0, b_high=1.0,
                                         transition_z=2.5, fractional_measurement_error=0.04)
null_fit = HighZTransientAuditor(null_events).fit_high_z(z_threshold=2.5, fit_intercept=False)
assert abs(null_fit.params[0] - 1.0) < 4*null_fit.stderr()[0]

inj_events = generate_synthetic_catalog(n=600, seed=7, b_low=1.0, b_high=1.08,
                                        transition_z=2.5, fractional_measurement_error=0.04)
inj_fit = HighZTransientAuditor(inj_events).fit_high_z(z_threshold=2.5, fit_intercept=False)
assert abs(inj_fit.params[0] - 1.08) < 4*inj_fit.stderr()[0]

print("ALL TESTS PASS")
print("numerical closure delta Gyr =", closure["delta_num_gyr"])
print("null high-z b =", null_fit.params[0], "+/-", null_fit.stderr()[0])
print("injected high-z b =", inj_fit.params[0], "+/-", inj_fit.stderr()[0])
