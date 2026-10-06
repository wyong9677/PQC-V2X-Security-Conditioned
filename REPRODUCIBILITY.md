# Reproducibility

## Canonical entry point

The principal real-profile closure implementation is:

`02_src/e5_real_profile_closure.py`

The original audited environment used Python 3.12.13 and NumPy 2.4.6.

Run experiments from the numerical-experiment root in an isolated Python
environment after installing the required dependencies described by the
configuration/provenance artifacts.

## Evidence boundaries

The public release distinguishes:
1. canonical source/configuration;
2. canonical paper-facing results;
3. final Bellman common-projection evidence;
4. private/local historical debugging material, which is not part of the
   publication artifact.

The final numerical evidence establishes terminal common projected-vector
equality on the tested finite geometry. It does not establish complete
Bellman-trajectory equality or continuous maximal-kernel equality.
