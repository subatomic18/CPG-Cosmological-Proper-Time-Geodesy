# Cosmological Proper-Time Geodesy (CPG)

**Author:** Jeffery Barnes  
**Affiliation:** Independent Researcher  
**Code license:** MIT  
**Archived release:** v0.3.1  
**Archived release DOI:** [10.5281/zenodo.22863852](https://doi.org/10.5281/zenodo.22863852)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22863852.svg)](https://doi.org/10.5281/zenodo.22863852)
[![Core validation](https://github.com/subatomic18/CPG-Cosmological-Proper-Time-Geodesy/actions/workflows/ci.yml/badge.svg)](https://github.com/subatomic18/CPG-Cosmological-Proper-Time-Geodesy/actions/workflows/ci.yml)
[![Scientific benchmarks](https://github.com/subatomic18/CPG-Cosmological-Proper-Time-Geodesy/actions/workflows/benchmarks.yml/badge.svg)](https://github.com/subatomic18/CPG-Cosmological-Proper-Time-Geodesy/actions/workflows/benchmarks.yml)

> **Repository status:** `main` contains development work completed after the archived v0.3.1 release. The DOI above identifies the frozen v0.3.1 archive, not every later commit on `main`.

CPG is a numerical research framework for testing **cosmological chronometric closure**: whether independently specified cosmological clocks, worldlines, geometry probes, and reference baselines close consistently within a stated spacetime model.

## Scientific scope

CPG is a diagnostic framework, not by itself a physical derivation of an RTD-EU clock field. A nonzero temporal, chronometer, supernova, BAO, distance-ladder, or other closure residual is **not automatically evidence for RTD-EU or new physics**. Source evolution, calibration, selection effects, peculiar velocities, endpoint definitions, covariance, and other conventional systematics must be evaluated first.

The long-term program is to connect

```text
matter dynamics -> metric + worldlines -> chronometric mapping -> proper time -> observables
```

and test that chain against independent cosmological probes.

## Current capabilities

The repository currently includes:

- dual-pathway numerical proper-time integration and internal numerical closure checks;
- high-redshift standardized-transient time-dilation analysis;
- cosmic-chronometer diagnostics;
- covariance-aware radial-BAO / chronometer clock-geometry closure;
- Gaussian-process and PCHIP reconstructions of the clock-geometry statistic;
- compressed Planck acoustic-scale reproduction;
- SH0ES distance-ladder reproduction tooling;
- redshift-mapping, late-onset, and compensated-mapping stress tests;
- weak-field two-congruence and ADM worldline proper-time tools;
- Mescaline/HDF5 and numerical-relativity experiment adapters; and
- a frozen prospective prediction and falsification register.

## Clock-geometry closure

Radial BAO measures `D_H/r_d`, while cosmic chronometers reconstruct `H_CC(z)`. CPG combines them into the sound-horizon-degenerate observable

```text
Q(z) = Gamma(z) r_d = c / { [D_H(z)/r_d] H_CC(z) }.
```

A constant `Q(z)` is the null test for no detected redshift dependence in `Gamma(z)`. The shape test therefore does not require an assumed sound horizon. An external `r_d` may optionally be supplied to convert `Q(z)` into an absolute `Gamma(z)` reconstruction.

The covariance-aware implementation propagates chronometer and radial-BAO covariance, refuses extrapolation beyond the measured chronometer range, returns `Q(z)` and `Q(z)/Q0`, and performs generalized-least-squares constant-closure testing.

The GP-first reconstruction is documented in [`docs/CLOCK_GEOMETRY_GP.md`](docs/CLOCK_GEOMETRY_GP.md).

## Prediction register

The prospective baseline is frozen in [`predictions/CPG_Prediction_Register_v1.0.md`](predictions/CPG_Prediction_Register_v1.0.md). It records the weak-field environmental benchmark, explicit null and failure conditions, cross-probe consistency requirements, and the rule that future data must be evaluated against the frozen prediction before any refit is presented as a prediction.

## Repository layout

```text
CPG/
├── cpg_*.py                 # research modules and command-line tools
├── tests/                   # regression tests
├── data/                    # versioned public/derived input tables
├── examples/                # small input-format examples
├── predictions/             # frozen prospective prediction registers
├── docs/                    # technical and release documentation
├── .github/workflows/       # core validation + scientific benchmarks
├── requirements.txt         # runtime dependencies
├── requirements-dev.txt     # test/development dependencies
├── CITATION.cff
├── CHANGELOG.md
└── LICENSE
```

The `cpg_*.py` modules intentionally remain at repository root during the post-v0.3.1 development line so existing CLI commands and imports remain stable. A future package-layout migration should be handled as a separate versioned change.

## Installation

Python 3.10+ is recommended.

```bash
python -m pip install -r requirements.txt
```

For development and tests:

```bash
python -m pip install -r requirements-dev.txt
```

CAMB is an optional benchmark dependency and is installed only when running the compressed-CMB reproduction.

## Validation

Run the complete regression suite from repository root:

```bash
python -m pytest -q
```

Compile all research modules:

```bash
python -m py_compile cpg_*.py
```

The original v0.3 numerical engine also retains its built-in self-test:

```bash
python cpg_v0_3.py --self-test
```

## Example analyses

Clock-geometry closure using the versioned chronometer input and a radial BAO catalogue:

```bash
python cpg_clock_geometry_closure.py \
  --cc data/cosmic_chronometers_32.csv \
  --bao data/desi_dr2_radial_bao.csv \
  --cc-systematics data/moresco_mm20_systematics.csv \
  --bao-covariance data/desi_dr2_radial_bao_cov.csv \
  --draws 8000
```

GP-first reconstruction:

```bash
python cpg_clock_geometry_gp.py \
  --cc data/cosmic_chronometers_32.csv \
  --bao data/desi_dr2_radial_bao.csv \
  --cc-systematics data/moresco_mm20_systematics.csv \
  --bao-covariance data/desi_dr2_radial_bao_cov.csv \
  --method gp \
  --draws 8000
```

High-redshift transient example:

```bash
python cpg_v0_3.py \
  --catalog examples/transient_template.csv \
  --z-threshold 2.5 \
  --z-bins 0,1,2,2.5,3,4,6 \
  --systematic-fraction 0.02
```

## Development policy

Validated changes should include regression tests and, when applicable, a reproducible public-data benchmark. Experimental work should remain isolated until its assumptions and numerical convergence are sufficiently tested. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Citation

For the archived software release, cite:

**Barnes, Jeffery. (2026). _Cosmological Proper-Time Geodesy (CPG) (v0.3.1)._ Zenodo. https://doi.org/10.5281/zenodo.22863852**

Machine-readable metadata are provided in [`CITATION.cff`](CITATION.cff). Development commits after v0.3.1 should be identified by commit or by a future archived release rather than attributed retroactively to the v0.3.1 DOI.

## License

Copyright (c) 2026 Jeffery Barnes.

The source code is released under the MIT License. See [`LICENSE`](LICENSE).
