#!/usr/bin/env python
"""Leave-one-dataset-out validation and accuracy--quantum-cost Pareto analysis."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import numpy as np

from hqkm.baselines import ModelConfig
from hqkm.estimator import SketchConfig
from hqkm.protocol import ProtocolConfig, run_single

DATASETS = ("breast_cancer", "digits01", "digits17", "digits38",
            "digits49", "digits56", "wine", "wine02", "wine12",
            "iris01", "iris02", "iris12")
SOURCE_GROUP = {
    "breast_cancer": "wdbc",
    **{d: "digits" for d in DATASETS if d.startswith("digits")},
    **{d: "wine" for d in DATASETS if d.startswith("wine")},
    **{d: "iris" for d in DATASETS if d.startswith("iris")},
}


def _paired(raw):
    index = {(r["dataset"], r["seed"], r["shots"], r["model"]): r for r in raw}
    rows = []
    for dataset in sorted({r["dataset"] for r in raw}):
        for seed in sorted({r["seed"] for r in raw if r["dataset"] == dataset}):
            classical = index[dataset, seed, 0, "classical_mkl"]
            for shots in sorted({r["shots"] for r in raw if r["dataset"] == dataset and r["shots"] > 0}):
                hybrid = index[dataset, seed, shots, "hybrid_mkl"]
                rows.append({
                    "dataset": dataset, "source_group": SOURCE_GROUP[dataset],
                    "seed": seed, "shots": shots,
                    "delta_tune_accuracy": hybrid["info"]["tune_accuracy"] - classical["info"]["tune_accuracy"],
                    "delta_test_accuracy": hybrid["accuracy"] - classical["accuracy"],
                    "hybrid_accuracy": hybrid["accuracy"],
                    "classical_accuracy": classical["accuracy"],
                    "executions": hybrid["executions"],
                    "quantum_wall_clock_s": max(hybrid["wall_clock_s"] - classical["wall_clock_s"], 0.0),
                })
    return rows


def choose_threshold(calibration, candidates):
    """Maximize delivered test gain on calibration tasks; ties use less quantum."""
    table = []
    for threshold in candidates:
        selected = [r for r in calibration if r["delta_tune_accuracy"] > threshold]
        gain = sum(r["delta_test_accuracy"] for r in selected) / max(len(calibration), 1)
        rate = len(selected) / max(len(calibration), 1)
        table.append({"threshold": threshold, "delivered_gain": gain,
                      "selection_rate": rate})
    best = max(table, key=lambda x: (x["delivered_gain"], -x["selection_rate"], x["threshold"]))
    return best["threshold"], table


def cross_validate_gate(pairs, leave="dataset"):
    key = "dataset" if leave == "dataset" else "source_group"
    candidates = (-1e-12, 0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10)
    predictions = []
    for shots in sorted({r["shots"] for r in pairs}):
        budget = [r for r in pairs if r["shots"] == shots]
        for held_out in sorted({r[key] for r in budget}):
            calibration = [r for r in budget if r[key] != held_out]
            test = [r for r in budget if r[key] == held_out]
            threshold, _ = choose_threshold(calibration, candidates)
            for row in test:
                selected = row["delta_tune_accuracy"] > threshold
                predictions.append({**row, "leave": leave, "held_out": held_out,
                                    "threshold": threshold, "selected": selected,
                                    "delivered_gain": row["delta_test_accuracy"] if selected else 0.0,
                                    "delivered_executions": row["executions"] if selected else 0.0,
                                    "delivered_wall_clock_s": row["quantum_wall_clock_s"] if selected else 0.0})
    return predictions


def summarize(predictions):
    out = []
    for shots in sorted({r["shots"] for r in predictions}):
        rows = [r for r in predictions if r["shots"] == shots]
        out.append({
            "shots": shots,
            "n": len(rows),
            "selection_rate": float(np.mean([r["selected"] for r in rows])),
            "mean_delivered_accuracy_gain": float(np.mean([r["delivered_gain"] for r in rows])),
            "mean_executions": float(np.mean([r["delivered_executions"] for r in rows])),
            "mean_quantum_wall_clock_s": float(np.mean([r["delivered_wall_clock_s"] for r in rows])),
            "always_hybrid_gain": float(np.mean([r["delta_test_accuracy"] for r in rows])),
            "always_hybrid_executions": float(np.mean([r["executions"] for r in rows])),
            "false_positive": int(sum(r["selected"] and r["delta_test_accuracy"] <= 0 for r in rows)),
            "false_negative": int(sum((not r["selected"]) and r["delta_test_accuracy"] > 0 for r in rows)),
        })
    return out


def pareto_front(points, cost_key="mean_executions", gain_key="mean_delivered_accuracy_gain"):
    """Return nondominated points: lower cost and higher gain are preferred."""
    unique = []
    seen = set()
    for point in points:
        signature = (point[cost_key], point[gain_key])
        if signature not in seen:
            seen.add(signature)
            unique.append(point)
    front = []
    for p in unique:
        dominated = any(
            q[cost_key] <= p[cost_key] and q[gain_key] >= p[gain_key]
            and (q[cost_key] < p[cost_key] or q[gain_key] > p[gain_key])
            for q in unique
        )
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda x: x[cost_key])


def policy_points(summary):
    """Comparable classical, cross-validated gate, and always-hybrid points."""
    points = [{"policy": "classical", "shots": 0,
               "mean_executions": 0.0, "mean_delivered_accuracy_gain": 0.0,
               "mean_quantum_wall_clock_s": 0.0}]
    for row in summary:
        points.append({"policy": "cross_validated_gate", **row})
        points.append({"policy": "always_hybrid", "shots": row["shots"],
                       "mean_executions": row["always_hybrid_executions"],
                       "mean_delivered_accuracy_gain": row["always_hybrid_gain"],
                       "mean_quantum_wall_clock_s": None})
    return points


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--shots", nargs="+", type=int, default=[100, 500, 1000])
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--out", default="loodo_results.json")
    args = parser.parse_args(argv)
    raw, started = [], time.perf_counter()
    for dataset in args.datasets:
        for seed in args.seeds:
            base_cfg = ProtocolConfig(seeds=(seed,), datasets=(dataset,), n_samples=args.n_samples,
                model_cfg=ModelConfig(mkl_iters=100, qnn_iters=40,
                                      sketch=SketchConfig(m=100, shots=args.shots[0])))
            classical = run_single(dataset, "classical_mkl", seed, base_cfg)
            raw.append({**classical, "shots": 0})
            for shots in args.shots:
                cfg = ProtocolConfig(seeds=(seed,), datasets=(dataset,), n_samples=args.n_samples,
                    model_cfg=ModelConfig(mkl_iters=100, qnn_iters=40,
                                          sketch=SketchConfig(m=100, shots=shots)))
                hybrid = run_single(dataset, "hybrid_mkl", seed, cfg)
                raw.append({**hybrid, "shots": shots})
    pairs = _paired(raw)
    loodo = cross_validate_gate(pairs, "dataset")
    logo = cross_validate_gate(pairs, "source_group")
    loodo_summary, logo_summary = summarize(loodo), summarize(logo)
    loodo_points, logo_points = policy_points(loodo_summary), policy_points(logo_summary)
    result = {"design": {"datasets": args.datasets, "source_groups": SOURCE_GROUP,
                          "seeds": args.seeds, "shots": args.shots,
                          "n_samples_cap": args.n_samples,
                          "note": "12 binary tasks from four independent source datasets"},
              "pairs": pairs, "loodo_predictions": loodo,
              "leave_source_out_predictions": logo,
              "loodo_summary": loodo_summary,
              "leave_source_out_summary": logo_summary,
              "policy_points_loodo": loodo_points,
              "policy_points_leave_source_out": logo_points,
              "pareto_loodo": pareto_front(loodo_points),
              "pareto_leave_source_out": pareto_front(logo_points),
              "total_wall_clock_s": time.perf_counter() - started}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"out": args.out, "seconds": result["total_wall_clock_s"],
                      "loodo": loodo_summary, "logo": logo_summary}, indent=2))


if __name__ == "__main__":
    main()
