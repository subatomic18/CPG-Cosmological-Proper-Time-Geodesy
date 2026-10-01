# Examples

`transient_template.csv` is a minimal catalogue showing the input schema expected by the high-redshift transient analysis in `cpg_v0_3.py`.

Run it from the repository root with:

```bash
python cpg_v0_3.py \
  --catalog examples/transient_template.csv \
  --z-threshold 2.5 \
  --z-bins 0,1,2,2.5,3,4,6 \
  --systematic-fraction 0.02
```

The template is illustrative only; it is not an observational dataset.
