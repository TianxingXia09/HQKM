# Reproducibility guide

## Supported environment

- Python 3.10–3.12
- CPU execution; no quantum hardware is required
- Linux, macOS, or Windows

Create an isolated environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
python -m pytest -q
```

## Fast verification

```bash
python run_experiments.py --profile smoke --out results/smoke_reproduction.json
```

This verifies dataset generation, kernel construction, finite-shot sampling,
Nyström reconstruction, PSD repair, execution accounting, and result logging.

## Paper-result artifacts

The `results/` directory contains machine-readable JSON artifacts used for the
development, locked external, leave-one-dataset-out, and Pareto analyses.
Figures can be rebuilt with the documented plotting scripts. Raw third-party
datasets are not redistributed; see `DATA.md`.

The expensive experimental commands and their intended outputs are listed in
`README.md`. Run them only after copying the required third-party datasets into
a local directory. All preprocessing is fitted on training rows only, and
quantum exploration cost remains charged after fallback.

## Release verification

Before creating a DOI-bearing release, replace every `REPLACE_WITH_*` token in
`CITATION.cff`, `.zenodo.json`, and `pyproject.toml`, then run:

```bash
python scripts/release_check.py
```

The check fails if author/repository metadata is unresolved, local filesystem
paths remain in published result JSON, required files are missing, or tests
fail.
