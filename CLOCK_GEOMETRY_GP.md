# GP-first clock-geometry closure

For real cosmic-chronometer data, the recommended CPG front end is now `cpg_clock_geometry_gp.py`.

The primary observable remains

```text
Q(z) = Gamma(z) r_d = c / { [D_H(z)/r_d] H_CC(z) }.
```

The default `--method gp` reconstructs `H_CC(z)` with a covariance-aware Matern-3/2 Gaussian process using a generalized linear mean. Full non-diagonal cosmic-chronometer covariance is accepted. The posterior prediction covariance is propagated jointly with radial-BAO covariance through Monte-Carlo draws.

The earlier Monte-Carlo PCHIP reconstruction is preserved as a robustness check with `--method pchip`.

No extrapolation beyond the chronometer redshift range is allowed in either method.

## Reproducible DESI DR2 radial BAO run

The repository includes the public DESI DR2 radial BAO values and extracted radial covariance in:

```text
data/desi_dr2_radial_bao.csv
data/desi_dr2_radial_bao_cov.csv
```

Run the current public 32-point chronometer compilation with its Moresco-style correlated systematics using:

```bash
python3 cpg_clock_geometry_gp.py \
  --cc data/cosmic_chronometers_32.csv \
  --bao data/desi_dr2_radial_bao.csv \
  --cc-systematics data/moresco_mm20_systematics.csv \
  --bao-covariance data/desi_dr2_radial_bao_cov.csv \
  --method gp \
  --draws 8000
```

For the PCHIP robustness calculation, change only:

```text
--method pchip
```

## Interpreting the output

The scientific null test is whether `Q(z)` is consistent with a constant. The JSON output reports `Q0`, its uncertainty, `chi2`, degrees of freedom, the constant-closure p-value, and `Q(z)/Q0`.

An external sound-horizon value may optionally be supplied with `--rd` and `--rd-sigma` to convert `Q` into an absolute `Gamma`, but the redshift-shape test itself does not require this calibration.

The GP output also reports its fitted amplitude, redshift correlation length, linear-mean parameters, and marginal negative log likelihood. Current hyperparameter uncertainty is not marginalized; the JSON explicitly marks this with `hyperparameter_uncertainty_propagated: false`.

A closure residual is an observational diagnostic only. It is not, by itself, evidence for RTD-EU or another non-standard cosmology.

## Newer DESI chronometer covariance

The GP front end can ingest a future published full covariance directly with `--cc-covariance`. The August 2026 DESI-DR1 chronometer analysis states that its reconstructed `H(z)` array and covariance will be provided as supplementary material with the journal publication. Until those machine-readable files are publicly released, CPG should not substitute or reverse-engineer an unpublished covariance.
