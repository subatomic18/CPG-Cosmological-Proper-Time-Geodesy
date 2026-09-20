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

CPG is an observational and numerical diagnostic framework. A nonzero `R_SN`, chronometer residual, or other closure statistic is **not automatically evidence for RTD-EU**. A physical interpretation requires a self-consistent spacetime model, worldlines/congruences, invariant boundary conditions, a light-cone/redshift mapping, and control of observational systematics.

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
