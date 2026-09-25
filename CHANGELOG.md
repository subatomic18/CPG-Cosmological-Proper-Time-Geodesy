# Changelog

## Unreleased

- Added a covariance-aware clock-geometry closure diagnostic.
- Added the sound-horizon-degenerate observable `Q(z) = Gamma(z) r_d` from radial BAO and cosmic chronometers.
- Added Monte-Carlo propagation of cosmic-chronometer and BAO covariance.
- Added PCHIP reconstruction of `H_CC(z)` at BAO redshifts without extrapolation.
- Added generalized-least-squares constant-`Q` closure testing with `chi2`, degrees of freedom, and p-value.
- Added normalized shape output `Q(z)/Q0` and optional conversion to absolute `Gamma(z)` when an external `r_d` is supplied.
- Added synthetic regression tests for constant-`Q` recovery, optional `r_d` conversion, covariance handling, and extrapolation rejection.

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
