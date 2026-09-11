# v1.0.0 — reproducibility release

This archival release accompanies the manuscript *Hybrid Quantum-Classical
Algorithms for Accelerated Machine Learning Tasks: A Cost-Benefit Benchmark of
Budgeted Quantum-Kernel Acquisition*.

## Included

- reference implementation of the HQKM evaluation pipeline;
- leakage-safe nested-selection and leave-one-dataset-out protocols;
- budgeted landmark-acquisition and multi-objective screening code;
- regression tests for the critical implementation blockers identified during
  the internal audit;
- machine-readable result artifacts and the submitted manuscript snapshot;
- pinned dependencies, citation metadata, and continuous integration.

## Scope statement

The quantum component is evaluated under the model and execution regime
documented in the manuscript. This release is not evidence of demonstrated
quantum advantage or hardware-level acceleration.

Before publishing, resolve every `REPLACE_WITH_*` token, run
`python scripts/release_check.py`, tag the tested commit as `v1.0.0`, and
archive that GitHub release in Zenodo.
