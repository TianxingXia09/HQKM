#!/usr/bin/env python
"""Development-only screen of the budget-aware multi-objective acquisition.

External datasets are rejected by construction.  Candidate settings are fixed
in this file before outcomes are computed; any later external evaluation must
use the selected setting without modification.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from hqkm.baselines import ModelConfig
from hqkm.estimator import SketchConfig
from hqkm.local_datasets import load_uploaded_dataset
from run_uploaded_benchmarks import run_cell


DEVELOPMENT_DATASETS = (
    "adult", "diabetes", "ionosphere", "spambase",
    "phoneme", "qsar_biodeg", "sonar",
)

VARIANTS = {
    "classical": ("classical_mkl", {}),
    "residual_accuracy": ("budgeted_hybrid_mkl", {
        "acquisition_strategy": "residual", "acquisition_metric": "accuracy"}),
    "multiobjective_budgeted": ("budgeted_hybrid_mkl", {
        "acquisition_strategy": "multiobjective",
        "acquisition_metric": "multiobjective",
        "acquisition_max_executions": 600_000.0,
        "acquisition_min_efficiency": 0.0,
    }),
}


def config(shots, overrides):
    values = dict(
        mkl_iters=60, qnn_iters=20, sketch=SketchConfig(m=40, shots=shots),
        acquisition_rounds=(5, 10, 20, 40), acquisition_cost_penalty=0.01,
        acquisition_patience=2, acquisition_accuracy_weight=0.5,
        acquisition_auc_weight=0.5, acquisition_residual_weight=0.35,
        acquisition_uncertainty_weight=0.35, acquisition_diversity_weight=0.30)
    values.update(overrides)
    return ModelConfig(**values)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--datasets", nargs="+", default=list(DEVELOPMENT_DATASETS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--n-cap", type=int, default=600)
    p.add_argument("--shots", type=int, default=100)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    forbidden = sorted(set(args.datasets) - set(DEVELOPMENT_DATASETS))
    if forbidden:
        raise ValueError(f"non-development datasets are forbidden: {forbidden}")

    loaded, audits = {}, {}
    for dataset in args.datasets:
        X, y, audit = load_uploaded_dataset(args.root, dataset)
        loaded[dataset], audits[dataset] = (X, y), audit

    rows, started = [], time.perf_counter()
    for dataset in args.datasets:
        for seed in args.seeds:
            for variant, (model_name, overrides) in VARIANTS.items():
                row = run_cell(*loaded[dataset], dataset, model_name, seed,
                               config(args.shots, overrides), args.n_cap)
                row["variant"] = variant
                rows.append(row)

    summary = {}
    for dataset in args.datasets:
        summary[dataset] = {}
        base = [r for r in rows if r["dataset"] == dataset
                and r["variant"] == "classical"]
        for variant in VARIANTS:
            rr = [r for r in rows if r["dataset"] == dataset
                  and r["variant"] == variant]
            summary[dataset][variant] = {
                "accuracy_mean": float(np.mean([r["accuracy"] for r in rr])),
                "balanced_accuracy_mean": float(np.mean([r["balanced_accuracy"] for r in rr])),
                "auc_mean": float(np.mean([r["auc"] for r in rr])),
                "executions_mean": float(np.mean([r["total_executions"] for r in rr])),
                "delta_accuracy_mean": float(np.mean([
                    r["accuracy"] - b["accuracy"] for r, b in zip(rr, base)])),
                "delta_auc_mean": float(np.mean([
                    r["auc"] - b["auc"] for r, b in zip(rr, base)])),
            }

    result = {
        "protocol_status": "development-only; external datasets inaccessible by construction",
        "selection_rule": (
            "advance only if multiobjective weakly improves mean accuracy and AUC "
            "over residual_accuracy at no greater mean circuit cost"),
        "variants": VARIANTS, "datasets": args.datasets,
        "config": {"seeds": args.seeds, "n_cap": args.n_cap,
                   "shots": args.shots},
        "audits": audits, "rows": rows, "summary": summary,
        "total_wall_clock_s": time.perf_counter() - started,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
