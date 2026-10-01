# Changelog

## Unreleased

### Scientific and numerical development after v0.3.1

- Added Hubble chronometric-closure diagnostics.
- Added a weak-field two-congruence chronometric diagnostic and regression tests.
- Added ADM worldline proper-time integration, Mescaline/HDF5 adaptation, and a numerical-relativity experiment runner.
- Added SH0ES distance-ladder reproduction tooling and an automated public-data benchmark.
- Added a compressed Planck CMB acoustic-scale closure module and automated central-value reproduction benchmark.
- Added constant, late-onset, and compensated redshift-mapping closure/stress tests.
- Added a covariance-aware clock-geometry closure diagnostic using radial BAO and cosmic chronometers.
- Added the sound-horizon-degenerate observable `Q(z) = Gamma(z) r_d` with full covariance propagation, normalized shape output, optional `r_d` calibration, and generalized-least-squares constant-closure testing.
- Added a covariance-aware Matern-3/2 Gaussian-process reconstruction while retaining PCHIP as a robustness option.
- Added versioned DESI DR2 radial-BAO inputs and covariance used by the public-data clock-geometry benchmark.
- Added CPG Prediction Register v1.0, frozen on 2026-09-28, with explicit null tests, prospective-data rules, failure conditions, and the weak-field environmental benchmark scale.

### Repository consolidation

- Moved regression tests into `tests/` without changing their scientific content.
- Moved the transient input template into `examples/` and technical GP notes into `docs/`.
- Consolidated seven narrowly scoped GitHub Actions files into two workflows: `Core validation` and `Scientific benchmarks`.
- Added `requirements-dev.txt`, `pytest.ini`, `.gitignore`, contributor guidance, a documentation index, and a release checklist.
- Updated the README to distinguish the archived v0.3.1 DOI from post-release development on `main`.
- Kept the `cpg_*.py` research modules at repository root to preserve existing imports and command-line interfaces during the current development line.
- No scientific result or numerical algorithm was changed by the repository-structure cleanup itself.

## v0.3.1 — 2026-09-20

- Created the Zenodo-enabled archival release of CPG.
- Assigned permanent DOI: `10.5281/zenodo.22863852`.
- Updated repository citation metadata and documentation to reference the archived release.
- No scientific or numerical behavior changed relative to v0.3.0.

## v0.3.0 — 2026-09-20

- Added high-redshift standardized-transient closure analysis.
- Added CSV catalogue ingestion.
- Added generalized least-squares fitting for the redshift-stretching exponent.
- Added configurable `z >= 2.5` high-redshift subset analysis.
- Added redshift-binned and environment-stratified fits.
- Added covariance and systematic-error support.
- Added smooth redshift-drift testing.
- Added null-closure and synthetic injection/recovery tests.
- Retained the dual Gauss-Kronrod/Radau proper-time integration engine.
- Enforced normalized default fiducial density closure (`E(0)=1`).
- Clarified that the transient chronometric ratio is an observational statistic and is not, by itself, a metric lapse.
