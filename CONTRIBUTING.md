# Contributing to CPG

Cosmological Proper-Time Geodesy is research software. Changes should preserve both numerical reproducibility and clear separation between measured diagnostics and physical interpretation.

## Development workflow

1. Create a branch from `main`.
2. Keep each change scientifically focused.
3. Add or update regression tests under `tests/`.
4. Run `python -m pytest -q` before opening a pull request.
5. Run the relevant reproducibility benchmark when a change affects a published or public-data baseline.
6. Record user-visible or scientific changes in `CHANGELOG.md`.

## Scientific guardrails

- A closure residual is a diagnostic, not automatically evidence for RTD-EU or non-standard cosmology.
- New physical claims must distinguish coordinate effects from invariant observables.
- Data inputs must have documented provenance; unpublished covariance information must not be reconstructed or invented.
- Conventional astrophysical, calibration, selection, and peculiar-velocity effects must be tested before a residual receives a new-physics interpretation.
- Negative and null results should be retained.
- The frozen `predictions/CPG_Prediction_Register_v1.0.md` must not be silently altered. Scientific revisions belong in a newly versioned prediction register.

## Dependencies

Runtime dependencies are listed in `requirements.txt`. Development/test dependencies are listed in `requirements-dev.txt`. Optional benchmark-specific dependencies, such as CAMB, are installed only in the benchmark workflow that needs them.

## Repository organization

The current `cpg_*.py` modules remain at repository root to preserve CLI and import compatibility while the post-v0.3.1 research line is active. Tests, documentation, examples, data, and prospective predictions are kept in dedicated directories. A package-level source-layout migration should be treated as a separate versioned change.
