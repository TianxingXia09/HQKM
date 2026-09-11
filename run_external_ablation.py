#!/usr/bin/env python
"""Locked external ablation: no variant is tuned from external outcomes."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from hqkm.baselines import ModelConfig
from hqkm.estimator import SketchConfig
from hqkm.local_datasets import load_uploaded_dataset
from run_uploaded_benchmarks import run_cell


VARIANTS = {
    "classical": ("classical_mkl", {}),
    "full_hybrid": ("hybrid_mkl", {}),
    "residual_accuracy": ("budgeted_hybrid_mkl",
                          {"acquisition_strategy": "residual", "acquisition_metric": "accuracy"}),
    "uncertainty_accuracy": ("budgeted_hybrid_mkl",
                             {"acquisition_strategy": "uncertainty", "acquisition_metric": "accuracy"}),
    "residual_auc": ("budgeted_hybrid_mkl",
                     {"acquisition_strategy": "residual", "acquisition_metric": "auc"}),
    "random_accuracy": ("budgeted_hybrid_mkl",
                        {"acquisition_strategy": "random", "acquisition_metric": "accuracy"}),
    "kmeans_accuracy": ("budgeted_hybrid_mkl",
                        {"acquisition_strategy": "kmeans", "acquisition_metric": "accuracy"}),
}


def config(shots, overrides):
    values = dict(mkl_iters=60, qnn_iters=20,
                  sketch=SketchConfig(m=40, shots=shots),
                  acquisition_rounds=(5, 10, 20, 40),
                  acquisition_cost_penalty=0.01, acquisition_patience=2)
    values.update(overrides)
    return ModelConfig(**values)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--datasets", nargs="+", default=[
        "credit_g_external", "heart_external", "magic_gamma_external"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--n-cap", type=int, default=600)
    p.add_argument("--shots", type=int, default=100)
    p.add_argument("--out", required=True)
    args = p.parse_args()
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
    index = {(r["dataset"], r["seed"], r["variant"]): r for r in rows}
    summary = {}
    for dataset in args.datasets:
        summary[dataset] = {}
        for variant in VARIANTS:
            rr = [index[dataset,s,variant] for s in args.seeds]
            summary[dataset][variant] = {
                "accuracy_mean": float(np.mean([r["accuracy"] for r in rr])),
                "balanced_accuracy_mean": float(np.mean([r["balanced_accuracy"] for r in rr])),
                "auc_mean": float(np.mean([r["auc"] for r in rr])),
                "executions_mean": float(np.mean([r["total_executions"] for r in rr])),
                "selected_rate": (float(np.mean([r["info"].get("selected_quantum", False)
                                                   for r in rr]))
                                  if "hybrid_mkl" in rr[0]["model"] and variant != "full_hybrid"
                                  else None),
            }
    result = {"protocol_status": "locked before external model outcomes",
              "variants": VARIANTS, "datasets": args.datasets,
              "config": {"seeds": args.seeds, "n_cap": args.n_cap,
                         "shots": args.shots},
              "audits": audits, "rows": rows, "summary": summary,
              "total_wall_clock_s": time.perf_counter()-started}
    with open(args.out, "w", encoding="utf-8") as f: json.dump(result, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
