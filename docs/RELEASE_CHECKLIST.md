# CPG release checklist

Use this checklist for the next archived software release (expected to be a post-v0.3.1 release such as v0.4.0).

1. Confirm `main` contains only intended validated changes.
2. Run `python -m pytest -q` and confirm the Core validation workflow passes.
3. Confirm all jobs in the Scientific benchmarks workflow pass.
4. Review public-data inputs and provenance notes.
5. Review the frozen prediction register; do not rewrite v1.0 in place.
6. Update `CHANGELOG.md` from `Unreleased` to the release version/date.
7. Update `README.md` release status and examples if interfaces changed.
8. Update `CITATION.cff` version, release date, and DOI only after the archive DOI is known.
9. Create the GitHub release/tag.
10. Verify the Zenodo archive and DOI metadata.
11. Confirm the archived source reproduces the tagged commit exactly.

Experimental light-cone/null-geodesic work should remain outside the release until its validation and convergence requirements are satisfied.
