#!/usr/bin/env python
"""Audit, classical-pilot, and hybrid comparison for uploaded CSV datasets."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from hqkm.baselines import ModelConfig, build_model
from hqkm.datasets import (fit_real_preprocessor, make_splits,
                           transform_real_features)
from hqkm.estimator import SketchConfig
from hqkm.local_datasets import UPLOAD_MANIFEST, load_uploaded_dataset
from hqkm.metrics import classification_metrics


def run_cell(X, y, dataset, model_name, seed, cfg, n_cap):
    rng = np.random.default_rng(seed)
    if n_cap and len(y) > n_cap:
        pos, neg = np.flatnonzero(y > 0), np.flatnonzero(y <= 0)
        npos = max(2, round(n_cap * len(pos) / len(y)))
        idx = np.r_[rng.choice(pos, min(npos, len(pos)), replace=False),
                    rng.choice(neg, min(n_cap-npos, len(neg)), replace=False)]
        idx = rng.permutation(idx); X, y = X[idx], y[idx]
    tr, tu, te = make_splits(y, np.random.default_rng(seed))
    prep = fit_real_preprocessor(X[tr], cfg.encoding.n_qubits)
    Xt, Xu, Xe = (transform_real_features(X[i], prep) for i in (tr, tu, te))
    model = build_model(model_name, cfg, np.random.default_rng(seed + 7919))
    started = time.perf_counter()
    model.fit(Xt, y[tr], Xu, y[tu])
    after_fit_executions = model.executions
    scores = model.decision_function(Xe)
    elapsed = time.perf_counter() - started
    return {"dataset": dataset, "model": model_name, "seed": seed,
            "n_train": len(tr), "n_tune": len(tu), "n_test": len(te),
            "fit_executions": after_fit_executions,
            "total_executions": model.executions, "wall_clock_s": elapsed,
            "info": model.info, **classification_metrics(scores, y[te]).as_dict()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--mode", choices=["classical", "compare"], default="classical")
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--n-cap", type=int, default=600)
    p.add_argument("--shots", type=int, default=100)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    names = args.datasets or [n for n,s in UPLOAD_MANIFEST.items() if s["status"] == "accepted"]
    audits, loaded = {}, {}
    for name in names:
        X, y, audit = load_uploaded_dataset(args.root, name)
        audits[name], loaded[name] = audit, (X, y)
    cfg = ModelConfig(mkl_iters=60, qnn_iters=20,
                      sketch=SketchConfig(m=40, shots=args.shots),
                      acquisition_rounds=(5, 10, 20, 40),
                      acquisition_strategy="residual",
                      acquisition_cost_penalty=0.01, acquisition_patience=2)
    models = (["classical_mkl"] if args.mode == "classical" else
              ["classical_mkl", "hybrid_mkl", "budgeted_hybrid_mkl"])
    rows, failures, started = [], [], time.perf_counter()
    for name in names:
        for seed in args.seeds:
            for model in models:
                try: rows.append(run_cell(*loaded[name], name, model, seed, cfg, args.n_cap))
                except Exception as exc:
                    failures.append({"dataset": name, "seed": seed, "model": model,
                                     "error": type(exc).__name__, "message": str(exc)})
    summary = {}
    for name in names:
        summary[name] = {}
        for model in models:
            rr = [r for r in rows if r["dataset"] == name and r["model"] == model]
            summary[name][model] = ({"n": len(rr),
                "accuracy_mean": float(np.mean([r["accuracy"] for r in rr])),
                "accuracy_std": float(np.std([r["accuracy"] for r in rr])),
                "auc_mean": float(np.mean([r["auc"] for r in rr])),
                "balanced_accuracy_mean": float(np.mean([r["balanced_accuracy"] for r in rr])),
                "executions_mean": float(np.mean([r["total_executions"] for r in rr]))}
                if rr else {"n": 0})
        acc = summary[name]["classical_mkl"].get("accuracy_mean")
        summary[name]["eligible_classical_band"] = bool(acc is not None and .70 <= acc <= .97)
    result = {"mode": args.mode, "root_manifest": UPLOAD_MANIFEST,
              "audits": audits, "config": {"seeds": args.seeds, "n_cap": args.n_cap,
              "shots": args.shots, "models": models}, "rows": rows,
              "failures": failures, "summary": summary,
              "total_wall_clock_s": time.perf_counter()-started}
    with open(args.out, "w", encoding="utf-8") as f: json.dump(result, f, indent=2)
    print(json.dumps({"summary": summary, "failures": failures}, indent=2))


if __name__ == "__main__":
    main()
