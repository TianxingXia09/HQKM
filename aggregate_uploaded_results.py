#!/usr/bin/env python
"""Aggregate paired uploaded-benchmark runs without reselecting datasets."""

from __future__ import annotations

import argparse
import json

import numpy as np

from hqkm.metrics import holm_bonferroni, wilcoxon_signed_rank


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rows, audits = [], {}
    for path in args.inputs:
        blob = json.load(open(path, encoding="utf-8"))
        rows.extend(blob["rows"]); audits.update(blob.get("audits", {}))
    index = {(r["dataset"], r["seed"], r["model"]): r for r in rows}
    datasets = sorted({r["dataset"] for r in rows})
    per_dataset, raw_p = {}, {}
    for dataset in datasets:
        seeds = sorted({r["seed"] for r in rows if r["dataset"] == dataset})
        per_dataset[dataset] = {}
        for model in ("hybrid_mkl", "budgeted_hybrid_mkl"):
            entry = {}
            for metric in ("accuracy", "balanced_accuracy", "auc"):
                delta = [index[dataset,s,model][metric] -
                         index[dataset,s,"classical_mkl"][metric] for s in seeds]
                _, pvalue = wilcoxon_signed_rank(np.asarray(delta))
                entry[metric] = {"deltas": delta, "mean_delta": float(np.mean(delta)),
                                 "wilcoxon_p_raw": pvalue}
                raw_p[f"{dataset}:{model}:{metric}"] = pvalue
            costs = [index[dataset,s,model]["total_executions"] for s in seeds]
            entry["executions_mean"] = float(np.mean(costs))
            if model == "budgeted_hybrid_mkl":
                entry["selection_rate"] = float(np.mean([
                    index[dataset,s,model]["info"]["selected_quantum"] for s in seeds]))
            per_dataset[dataset][model] = entry
    adjusted = holm_bonferroni(raw_p)
    for dataset in datasets:
        for model in ("hybrid_mkl", "budgeted_hybrid_mkl"):
            for metric in ("accuracy", "balanced_accuracy", "auc"):
                key=f"{dataset}:{model}:{metric}"
                per_dataset[dataset][model][metric].update(adjusted[key])

    # Dataset is the independent unit: average seeds within each source first.
    aggregate = {}
    rng = np.random.default_rng(20260907)
    for model in ("hybrid_mkl", "budgeted_hybrid_mkl"):
        aggregate[model] = {}
        for metric in ("accuracy", "balanced_accuracy", "auc"):
            source_delta = np.array([per_dataset[d][model][metric]["mean_delta"]
                                     for d in datasets])
            boot = [np.mean(rng.choice(source_delta, len(source_delta), replace=True))
                    for _ in range(10000)]
            aggregate[model][metric] = {
                "mean_source_delta": float(source_delta.mean()),
                "cluster_bootstrap_ci95": np.quantile(boot, [.025, .975]).tolist(),
                "wins": int(np.sum(source_delta > 1e-12)),
                "ties": int(np.sum(np.abs(source_delta) <= 1e-12)),
                "losses": int(np.sum(source_delta < -1e-12)),
            }
    result = {"independence_unit": "source dataset", "datasets": datasets,
              "n_sources": len(datasets), "audits": audits,
              "per_dataset": per_dataset, "aggregate": aggregate,
              "multiplicity": "Holm-Bonferroni across all dataset-model-metric tests"}
    with open(args.out, "w", encoding="utf-8") as f: json.dump(result, f, indent=2)
    print(json.dumps({"datasets": datasets, "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
