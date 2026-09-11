#!/usr/bin/env python
"""Statistics and Pareto fronts for the locked external ablation."""

from __future__ import annotations

import argparse
import json

import numpy as np

from hqkm.metrics import holm_bonferroni, wilcoxon_signed_rank
from run_loodo import pareto_front


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("external")
    p.add_argument("--development", nargs="+", default=[])
    p.add_argument("--out", required=True)
    args = p.parse_args()
    data = json.load(open(args.external, encoding="utf-8"))
    rows = data["rows"]
    idx = {(r["dataset"],r["seed"],r["variant"]):r for r in rows}
    datasets = data["datasets"]
    variants = [v for v in data["variants"] if v != "classical"]
    metrics = ("accuracy", "balanced_accuracy", "auc")
    per_dataset, pvalues = {}, {}
    for ds in datasets:
        seeds = sorted({r["seed"] for r in rows if r["dataset"] == ds})
        per_dataset[ds] = {}
        for variant in variants:
            per_dataset[ds][variant] = {}
            for metric in metrics:
                delta=[idx[ds,s,variant][metric]-idx[ds,s,"classical"][metric] for s in seeds]
                _, pv=wilcoxon_signed_rank(np.asarray(delta))
                per_dataset[ds][variant][metric]={"deltas":delta,
                    "mean_delta":float(np.mean(delta)),"wilcoxon_p_raw":pv}
                pvalues[f"{ds}:{variant}:{metric}"]=pv
            per_dataset[ds][variant]["executions_mean"]=float(np.mean(
                [idx[ds,s,variant]["total_executions"] for s in seeds]))
    adjusted=holm_bonferroni(pvalues)
    for ds in datasets:
        for v in variants:
            for m in metrics:
                per_dataset[ds][v][m].update(adjusted[f"{ds}:{v}:{m}"])

    rng=np.random.default_rng(20260907)
    aggregate={}; points={m:[{"variant":"classical","mean_executions":0.0,
                              "mean_delta":0.0}] for m in metrics}
    for v in variants:
        source_cost=np.array([per_dataset[d][v]["executions_mean"] for d in datasets])
        aggregate[v]={"mean_executions":float(source_cost.mean())}
        for m in metrics:
            delta=np.array([per_dataset[d][v][m]["mean_delta"] for d in datasets])
            boot=[float(np.mean(rng.choice(delta,len(delta),replace=True))) for _ in range(10000)]
            aggregate[v][m]={"mean_source_delta":float(delta.mean()),
                "cluster_bootstrap_ci95":np.quantile(boot,[.025,.975]).tolist(),
                "wins":int(np.sum(delta>1e-12)),"ties":int(np.sum(abs(delta)<=1e-12)),
                "losses":int(np.sum(delta<-1e-12))}
            points[m].append({"variant":v,"mean_executions":float(source_cost.mean()),
                              "mean_delta":float(delta.mean())})
    pareto={m:pareto_front(points[m],gain_key="mean_delta") for m in metrics}

    # Frozen ten-source extension: only methods already run on development and external.
    combined={"full_hybrid":{},"residual_accuracy":{}}
    development=[]
    for path in args.development:
        development.extend(json.load(open(path,encoding="utf-8"))["rows"])
    if development:
        didx={(r["dataset"],r["seed"],r["model"]):r for r in development}
        dev_ds=sorted({r["dataset"] for r in development})
        mapping={"full_hybrid":"hybrid_mkl","residual_accuracy":"budgeted_hybrid_mkl"}
        for out_name,model in mapping.items():
            for metric in metrics:
                source=[]
                for ds in dev_ds:
                    seeds=sorted({r["seed"] for r in development if r["dataset"]==ds})
                    source.append(np.mean([didx[ds,s,model][metric]-didx[ds,s,"classical_mkl"][metric]
                                           for s in seeds]))
                source += [per_dataset[d][out_name][metric]["mean_delta"] for d in datasets]
                arr=np.asarray(source); boot=[float(np.mean(rng.choice(arr,len(arr),replace=True))) for _ in range(10000)]
                combined[out_name][metric]={"n_sources":len(arr),"mean_source_delta":float(arr.mean()),
                    "cluster_bootstrap_ci95":np.quantile(boot,[.025,.975]).tolist(),
                    "wins":int(np.sum(arr>1e-12)),"ties":int(np.sum(abs(arr)<=1e-12)),
                    "losses":int(np.sum(arr<-1e-12))}
    result={"protocol":"locked external ablation","datasets":datasets,
            "per_dataset":per_dataset,"aggregate_external":aggregate,
            "pareto_external":pareto,"combined_ten_source_frozen_methods":combined,
            "multiplicity":"Holm-Bonferroni across external dataset-variant-metric tests"}
    with open(args.out,"w",encoding="utf-8") as f:json.dump(result,f,indent=2)
    print(json.dumps({"external":aggregate,"pareto":pareto,"combined":combined},indent=2))


if __name__ == "__main__":
    main()
