#!/usr/bin/env python
"""LODO evaluation of a train/tune-only gate before quantum acquisition."""

from __future__ import annotations

import argparse
import json
import numpy as np

from hqkm.meta_gate import ClassicalPreAcquisitionGate, META_FEATURES
from run_development_multiobjective import DEVELOPMENT_DATASETS


ETA_PER_MILLION = 0.005


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    data = json.load(open(args.input, encoding="utf-8"))
    if set(data["datasets"]) != set(DEVELOPMENT_DATASETS):
        raise ValueError("gate analysis requires exactly the seven development datasets")
    rows = data["rows"]
    keyed = {(r["dataset"], r["seed"], r["variant"]): r for r in rows}
    examples = []
    for dataset in data["datasets"]:
        for seed in data["config"]["seeds"]:
            classical = keyed[dataset, seed, "classical"]
            quantum = keyed[dataset, seed, "multiobjective_budgeted"]
            dacc = quantum["accuracy"] - classical["accuracy"]
            dauc = quantum["auc"] - classical["auc"]
            cost_million = quantum["total_executions"] / 1e6
            target = 0.5*dacc + 0.5*dauc - ETA_PER_MILLION*cost_million
            examples.append({
                "dataset": dataset, "seed": seed,
                "meta": quantum["info"]["pre_acquisition_meta_features"],
                "delta_accuracy": dacc, "delta_auc": dauc,
                "cost_million": cost_million, "net_value": target,
            })

    predictions = []
    for held_out in data["datasets"]:
        train = [e for e in examples if e["dataset"] != held_out]
        test = [e for e in examples if e["dataset"] == held_out]
        gate = ClassicalPreAcquisitionGate(ridge=1.0, threshold=0.0).fit(
            [e["meta"] for e in train], [e["net_value"] for e in train])
        values = gate.predict_value([e["meta"] for e in test])
        decisions = values > 0.0
        for e, value, decision in zip(test, values, decisions):
            predictions.append({**{k:v for k,v in e.items() if k != "meta"},
                                "predicted_net_value": float(value),
                                "acquire": bool(decision)})

    actual = np.asarray([p["net_value"] > 0 for p in predictions])
    chosen = np.asarray([p["acquire"] for p in predictions])
    tp, tn = int(np.sum(actual & chosen)), int(np.sum(~actual & ~chosen))
    fp, fn = int(np.sum(~actual & chosen)), int(np.sum(actual & ~chosen))
    policy_acc = np.asarray([p["delta_accuracy"] if p["acquire"] else 0.0
                             for p in predictions])
    policy_auc = np.asarray([p["delta_auc"] if p["acquire"] else 0.0
                             for p in predictions])
    policy_cost = np.asarray([p["cost_million"] if p["acquire"] else 0.0
                              for p in predictions])
    always_cost = np.asarray([p["cost_million"] for p in predictions])
    summary = {
        "n_examples": len(predictions), "held_out_datasets": len(data["datasets"]),
        "eta_per_million": ETA_PER_MILLION,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "decision_accuracy": float((actual == chosen).mean()),
        "precision": float(tp/max(tp+fp, 1)), "recall": float(tp/max(tp+fn, 1)),
        "acquisition_rate": float(chosen.mean()),
        "mean_delta_accuracy_policy": float(policy_acc.mean()),
        "mean_delta_auc_policy": float(policy_auc.mean()),
        "mean_cost_million_policy": float(policy_cost.mean()),
        "mean_cost_million_always": float(always_cost.mean()),
        "cost_reduction_fraction": float(1-policy_cost.mean()/always_cost.mean()),
        "mean_net_value_policy": float(np.mean(
            0.5*policy_acc + 0.5*policy_auc - ETA_PER_MILLION*policy_cost)),
        "mean_net_value_always": float(np.mean([p["net_value"] for p in predictions])),
    }
    by_dataset = {}
    for dataset in data["datasets"]:
        rr = [p for p in predictions if p["dataset"] == dataset]
        by_dataset[dataset] = {
            "acquisition_rate": float(np.mean([p["acquire"] for p in rr])),
            "actual_positive_rate": float(np.mean([p["net_value"] > 0 for p in rr])),
            "mean_predicted_net_value": float(np.mean([p["predicted_net_value"] for p in rr])),
            "mean_realized_net_value_if_gated": float(np.mean([
                p["net_value"] if p["acquire"] else 0.0 for p in rr])),
        }
    result = {
        "protocol": "leave-one-dataset-out; ridge=1 and threshold=0 fixed a priori",
        "feature_names": META_FEATURES, "target_definition":
        "0.5*dAccuracy + 0.5*dAUC - 0.005*executions_in_millions",
        "summary": summary, "by_dataset": by_dataset, "predictions": predictions,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"summary": summary, "by_dataset": by_dataset}, indent=2))


if __name__ == "__main__":
    main()
