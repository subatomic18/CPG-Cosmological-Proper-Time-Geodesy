# Cosmological Proper-Time Geodesy (CPG) v0.3.1

**Author:** Jeffery Barnes  
**Affiliation:** Independent Researcher  
**Code license:** MIT  
**Archived release DOI:** [10.5281/zenodo.22863852](https://doi.org/10.5281/zenodo.22863852)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22863852.svg)](https://doi.org/10.5281/zenodo.22863852)

CPG is a numerical framework for testing **cosmological chronometric closure**. Version 0.3 extends the original dual-pathway proper-time engine with a high-redshift transient analysis layer designed for future standardized transient data, including Type Ia supernova samples extending into the poorly tested high-redshift regime.

## Scope

CPG tests whether cosmological clock observables close consistently with a specified reference cosmology. It does **not** derive a physical RTD-EU clock field or lapse directly from Einstein's equations. Any measured temporal residual must therefore be distinguished from source evolution, calibration, selection effects, progenitor physics, and other astrophysical systematics before it is interpreted physically.

## Core numerical engine

The proper-time traverse is evaluated in two independent numerical formulations:

1. adaptive Gauss-Kronrod quadrature in redshift, and
2. Radau integration after transforming to `ln(a)`.

The two results are compared as an internal numerical closure test.

For a user-supplied chronometric function `Gamma(z)`, the reference integral is

```text
Delta tau = integral Gamma(z) / [(1+z) H(z)] dz
```

The default fiducial background enforces density closure so that `E(0)=1`.

## High-redshift transient module

For standardized transient timescales, CPG fits

```text
ln(T_obs/T_rest) = alpha + b ln(1+z)
```

where standard cosmological redshift stretching corresponds to `b = 1`. The nuisance parameter `alpha` may be fitted to absorb a global calibration offset or fixed to zero when source-frame standardization is independently calibrated.

The transient closure statistic is

```text
R_SN(z) = exp(alpha) (1+z)^(b-1) - 1
```

Version 0.3 includes:

- CSV catalogue ingestion
- generalized least-squares fitting
- configurable high-redshift threshold, default `z >= 2.5`
- redshift-binned fits
- environment-stratified fits
- systematic-error floors
- optional covariance matrices
- smooth redshift-drift testing
- standard-null closure tests
- synthetic signal injection and recovery
- automated numerical validation tests

## Clock-geometry closure diagnostic

The development branch also includes a covariance-aware comparison between
radial BAO geometry and cosmic-chronometer clocks. Radial BAO measures
`D_H/r_d`, while cosmic chronometers reconstruct `H_CC(z)`. CPG combines them
into the sound-horizon-degenerate observable

```text
Q(z) = Gamma(z) r_d = c / { [D_H(z)/r_d] H_CC(z) }.
```

A constant `Q(z)` is the null test for no detected redshift dependence in
`Gamma(z)`. Because the primary statistic is `Gamma r_d`, the shape test does
not require an assumed sound horizon. An external `r_d` can optionally be
supplied to convert `Q(z)` into an absolute `Gamma(z)` reconstruction.

The implementation in `cpg_clock_geometry_closure.py`:

- propagates a full cosmic-chronometer covariance matrix when supplied,
- propagates a full radial-BAO covariance matrix when supplied,
- reconstructs `H_CC` at BAO redshifts with Monte-Carlo PCHIP interpolation,
- refuses to extrapolate beyond the measured chronometer redshift range,
- returns `Q(z)`, `Q(z)/Q0`, the reconstructed covariance of `Q`,
- fits a generalized-least-squares constant `Q0`, and
- reports `chi2`, degrees of freedom, and the constant-closure p-value.

Example:

```bash
python3 cpg_clock_geometry_closure.py \
  --cc data/cosmic_chronometers_32.csv \
  --bao path/to/radial_bao.csv \
  --cc-systematics data/moresco_mm20_systematics.csv \
  --draws 8000
```

Optional absolute sound-horizon calibration:

```bash
python3 cpg_clock_geometry_closure.py \
  --cc data/cosmic_chronometers_32.csv \
  --bao path/to/radial_bao.csv \
  --rd 147.09 \
  --rd-sigma 0.26
```

Radial BAO CSV files require:

```text
z,DH_over_rd,sigma_DH_over_rd
```

with an optional `reference` column. A full numeric BAO covariance matrix can
be supplied separately with `--bao-covariance`.

## Requirements

- Python 3.10+
- NumPy
- SciPy

Install dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

## Run the self-test

```bash
python3 cpg_v0_3.py --self-test
```

Or run the automated test script:

```bash
python3 test_cpg_v0_3.py
```

The clock-geometry diagnostic has its own regression tests:

```bash
python3 test_cpg_clock_geometry_closure.py
```

## Catalogue format

Required CSV columns:

```text
event_id,z,t_obs,t_rest,sigma_t_obs
```

Optional columns:

```text
sigma_t_rest,environment,quality
```

The timescale unit is arbitrary provided observed/rest-frame times and their uncertainties use the same unit.

## Example analysis

```bash
python3 cpg_v0_3.py \
  --catalog transient_template.csv \
  --z-threshold 2.5 \
  --z-bins 0,1,2,2.5,3,4,6 \
  --systematic-fraction 0.02
```

To fix the calibration intercept to zero:

```bash
python3 cpg_v0_3.py --catalog transient_template.csv --fix-intercept
```

## Scientific interpretation

CPG is an observational and numerical diagnostic framework. A nonzero `R_SN`, chronometer residual, clock-geometry residual, or other closure statistic is **not automatically evidence for RTD-EU**. A physical interpretation requires a self-consistent spacetime model, worldlines/congruences, invariant boundary conditions, a light-cone/redshift mapping, and control of observational systematics.

The long-term RTD-EU/CPG program is to connect

```text
matter dynamics -> g_mn, u^m -> chronometric mapping -> Delta tau -> observables
```

and test that chain against independent cosmological probes.

## Citation

The archived v0.3.1 release is permanently available from Zenodo:

**Barnes, Jeffery. (2026). _Cosmological Proper-Time Geodesy (CPG) (v0.3.1)._ Zenodo. https://doi.org/10.5281/zenodo.22863852**

Machine-readable citation metadata are provided in [`CITATION.cff`](CITATION.cff).

## License

Copyright (c) 2026 Jeffery Barnes.

The source code is released under the MIT License. See [`LICENSE`](LICENSE).
