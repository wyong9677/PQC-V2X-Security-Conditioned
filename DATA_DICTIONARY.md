# Data dictionary

## Canonical results

`04_results/` contains paper-facing result artifacts produced by the canonical
numerical pipeline.

## Final Bellman evidence on Zenodo

The Zenodo extended-evidence archive retains:
- the successful E3B FIX1 captured-history run;
- E3E common-projection analysis;
- E3F final claim-boundary closure.

The final common projected variable is `required`, with vector length 94,325.
All four matched ML/SLH terminal comparisons are elementwise equal and have
maximum absolute difference 0.

Profile-specific internal `h` arrays are structurally different and are not
treated as a common trajectory object.
