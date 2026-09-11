# Contributing

Open an issue before changing a locked experimental configuration or external
evaluation protocol. Pull requests must include regression tests, preserve
train/tune/test separation, and keep quantum acquisition costs charged even
when the final predictor falls back to a classical model.

Run `python -m pytest -q` before submitting a change. Do not commit raw
third-party datasets, credentials, local paths, or regenerated confirmatory
results produced after inspecting the locked external outcomes.
