from pathlib import Path
import numpy as np

from cpg_chronometers import (
    load_cc_csv,
    load_systematics_csv,
    build_covariance,
    ChronometerClosureAuditor,
)

root = Path(__file__).resolve().parent
points = load_cc_csv(root / "data" / "cosmic_chronometers_32.csv")
sys = load_systematics_csv(root / "data" / "moresco_mm20_systematics.csv")
C = build_covariance(points, sys)

assert C.shape == (32, 32)
assert np.allclose(C, C.T)
assert np.linalg.eigvalsh(C).min() > -1e-8
assert np.count_nonzero(C - np.diag(np.diag(C))) > 0

aud = ChronometerClosureAuditor(points, C)
out = aud.reconstruct([0.5, 1.0, 1.5, 1.965], n_draws=500, seed=7)

assert len(out) == 4
assert all(r["t_cc_gyr"] > 0 and r["C"] > 0 for r in out)
assert np.isfinite(aud.chi2_reference())

print("CHRONOMETER TESTS PASS")
for r in out:
    print(r)
print("chi2_ref =", aud.chi2_reference())
