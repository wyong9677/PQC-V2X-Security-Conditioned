# PQC-V2X Security-Conditioned Viability — Reproducibility Artifact

This repository accompanies:

**Security-Conditioned Viability for Post-Quantum V2X Control: Authentication-Service State, Starvation-Robust Fallback, and Cross-Layer Service Qualification**

Authors: Yong Wang, Qiurui Liu, Eddie Shahril Ismail, Xiao Ma

## Scope

The public artifact contains canonical source code, configuration, lightweight
paper-facing results, and verification/reproducibility materials. The complete
frozen numerical evidence is archived separately on Zenodo.

## Canonical numerical claim added in the final audit

Across four matched ML-DSA-65 / SLH-DSA-SHA2-192s real-profile fixed-point
solves, the terminal common projected `required` vectors are elementwise
identical on all 94,325 physical nodes, with maximum absolute difference 0.

This does **not** establish equality of the complete Bellman trajectories:
profile-specific internal `h` states remain structurally different.

## Repository layout

- `01_config/` — canonical experiment/provenance configuration.
- `02_src/` — canonical source code.
- `04_results/` — lightweight canonical paper-facing results.
- `scripts/validated/` — validated audit/reproduction runners.
- `REPRODUCIBILITY.md` — reproduction and verification guidance.
- `DATA_DICTIONARY.md` — result/evidence organization.
- `ENVIRONMENT.md` — software environment.
- `SHA256SUMS.txt` — staged-file checksums.

## Data archive

The DOI-bearing Zenodo record should be linked here after publication.

GitHub repository: https://github.com/wyong9677/PQC-V2X-Security-Conditioned
