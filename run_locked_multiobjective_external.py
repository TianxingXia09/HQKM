#!/usr/bin/env python
"""One-shot external validation of the development-selected configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np

from hqkm.baselines import ModelConfig
from hqkm.estimator import SketchConfig
from hqkm.local_datasets import load_uploaded_dataset
from run_uploaded_benchmarks import run_cell


EXTERNAL_DATASETS = (
    "credit_g_external", "heart_external", "magic_gamma_external")
LOCKED_CONFIG = {
    "mkl_iters": 60, "shots": 100, "rounds": [5, 10, 20, 40],
    "cost_penalty": 0.01, "patience": 2,
    "accuracy_weight": 0.5, "auc_weight": 0.5,
    "residual_weight": 0.35, "uncertainty_weight": 0.35,
    "diversity_weight": 0.30, "max_fit_executions": 600000.0,
    "min_efficiency": 0.0,
}


def config():
    c = LOCKED_CONFIG
    return ModelConfig(
        mkl_iters=c["mkl_iters"], qnn_iters=20,
        sketch=SketchConfig(m=40, shots=c["shots"]),
        acquisition_rounds=tuple(c["rounds"]),
        acquisition_strategy="multiobjective",
        acquisition_metric="multiobjective",
        acquisition_cost_penalty=c["cost_penalty"],
        acquisition_patience=c["patience"],
        acquisition_accuracy_weight=c["accuracy_weight"],
        acquisition_auc_weight=c["auc_weight"],
        acquisition_residual_weight=c["residual_weight"],
        acquisition_uncertainty_weight=c["uncertainty_weight"],
        acquisition_diversity_weight=c["diversity_weight"],
        acquisition_max_executions=c["max_fit_executions"],
        acquisition_min_efficiency=c["min_efficiency"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--datasets", nargs="+", default=list(EXTERNAL_DATASETS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--n-cap", type=int, default=600)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    forbidden = sorted(set(args.datasets) - set(EXTERNAL_DATASETS))
    if forbidden:
        raise ValueError(f"only locked external datasets are allowed: {forbidden}")

    loaded, audits = {}, {}
    for name in args.datasets:
        X, y, audit = load_uploaded_dataset(args.root, name)
        loaded[name], audits[name] = (X, y), audit
    rows, started = [], time.perf_counter()
    for name in args.datasets:
        for seed in args.seeds:
            for variant, model_name in (("classical", "classical_mkl"),
                                        ("multiobjective_budgeted", "budgeted_hybrid_mkl")):
                row = run_cell(*loaded[name], name, model_name, seed,
                               config(), args.n_cap)
                row["variant"] = variant
                rows.append(row)
    summary = {}
    for name in args.datasets:
        summary[name] = {}
        base = [r for r in rows if r["dataset"] == name and r["variant"] == "classical"]
        for variant in ("classical", "multiobjective_budgeted"):
            rr = [r for r in rows if r["dataset"] == name and r["variant"] == variant]
            summary[name][variant] = {
                "accuracy_mean": float(np.mean([r["accuracy"] for r in rr])),
                "balanced_accuracy_mean": float(np.mean([r["balanced_accuracy"] for r in rr])),
                "auc_mean": float(np.mean([r["auc"] for r in rr])),
                "executions_mean": float(np.mean([r["total_executions"] for r in rr])),
                "delta_accuracy_mean": float(np.mean([r["accuracy"]-b["accuracy"]
                                                       for r, b in zip(rr, base)])),
                "delta_auc_mean": float(np.mean([r["auc"]-b["auc"]
                                                  for r, b in zip(rr, base)])),
            }
    encoded = json.dumps(LOCKED_CONFIG, sort_keys=True).encode()
    result = {
        "protocol_status": "one-shot locked external validation",
        "locked_config_sha256": hashlib.sha256(encoded).hexdigest(),
        "locked_config": LOCKED_CONFIG, "datasets": args.datasets,
        "seeds": args.seeds, "n_cap": args.n_cap, "audits": audits,
        "rows": rows, "summary": summary,
        "total_wall_clock_s": time.perf_counter()-started,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
