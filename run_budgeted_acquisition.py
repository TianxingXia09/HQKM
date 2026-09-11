#!/usr/bin/env python
"""Reproducible pilot for sequential budgeted quantum landmark acquisition."""

from __future__ import annotations

import argparse
import json
import time

from hqkm.baselines import ModelConfig
from hqkm.estimator import SketchConfig
from hqkm.protocol import ProtocolConfig, run_single


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+",
                   default=["S3_entangled_anchor", "breast_cancer"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--shots", type=int, default=100)
    p.add_argument("--rounds", nargs="+", type=int, default=[5, 10, 20, 40])
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--out", default="results/budgeted_acquisition_smoke.json")
    args = p.parse_args(argv)
    cfg = ProtocolConfig(
        seeds=tuple(args.seeds), datasets=tuple(args.datasets),
        n_samples=args.n_samples, n_features=6,
        model_cfg=ModelConfig(
            mkl_iters=50, qnn_iters=20,
            sketch=SketchConfig(m=max(args.rounds), shots=args.shots),
            acquisition_rounds=tuple(args.rounds),
            acquisition_strategy="residual", acquisition_cost_penalty=0.01,
            acquisition_patience=2))
    rows, started = [], time.perf_counter()
    for dataset in args.datasets:
        for seed in args.seeds:
            for model in ("classical_mkl", "hybrid_mkl", "budgeted_hybrid_mkl"):
                rows.append(run_single(dataset, model, seed, cfg))
    summary = {}
    for dataset in args.datasets:
        selected = [r for r in rows if r["dataset"] == dataset
                    and r["model"] == "budgeted_hybrid_mkl"]
        classical = [r for r in rows if r["dataset"] == dataset
                     and r["model"] == "classical_mkl"]
        full = [r for r in rows if r["dataset"] == dataset
                and r["model"] == "hybrid_mkl"]
        summary[dataset] = {
            "mean_budgeted_minus_classical": sum(a["accuracy"]-b["accuracy"]
                for a, b in zip(selected, classical)) / len(selected),
            "mean_full_minus_classical": sum(a["accuracy"]-b["accuracy"]
                for a, b in zip(full, classical)) / len(selected),
            "mean_budgeted_executions": sum(r["executions"] for r in selected)/len(selected),
            "mean_full_executions": sum(r["executions"] for r in full)/len(full),
            "quantum_selected_rate": sum(r["info"]["selected_quantum"] for r in selected)/len(selected),
        }
    result = {"config": cfg.as_dict(), "rows": rows, "summary": summary,
              "total_wall_clock_s": time.perf_counter()-started,
              "scope": "engineering smoke test; not confirmatory evidence"}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
