#!/usr/bin/env python
"""Command-line entry point for all experiments.

Examples
--------
Full paper configuration (days on a laptop; run per study if that is too long)::

    python run_experiments.py --profile paper --out results.json

Smoke test (~1 minute)::

    python run_experiments.py --profile smoke --studies E1 E5

One real dataset that auto-downloads on first use::

    python run_experiments.py --profile paper --datasets adult magic --studies E4

Profiles set every knob that changes results; anything else lives in
``hqkm.protocol.ProtocolConfig``. Use ``--seeds``, ``--n-samples`` and
``--datasets`` to override.
"""

from __future__ import annotations

import argparse
import json
import sys

from hqkm.baselines import ModelConfig
from hqkm.estimator import SketchConfig
from hqkm.protocol import DEFAULT_MODELS, ProtocolConfig, run_all

PROFILES = {
    "smoke": dict(
        seeds=(0, 1), n_samples=150,
        datasets=("S2_in_rkhs", "S4_high_freq"),
        mixing_grid=(0.0, 1.0), noise_grid=(0.0,),
        m_grid=(40,), shots_grid=(1000, "exact"),
        scaling_sizes=(150,),
        model_overrides=dict(mkl_iters=30, qnn_iters=15,
                             sketch=SketchConfig(m=40, shots=1000)),
    ),
    "pilot": dict(
        seeds=(0, 1, 2), n_samples=600,
        datasets=("S1_mixed", "S2_in_rkhs", "S3_entangled_anchor",
                  "S4_high_freq", "S5_noise"),
        mixing_grid=(0.0, 0.25, 0.5, 0.75, 1.0), noise_grid=(0.0, 0.1),
        m_grid=(100, 200), shots_grid=(200, 1000, "exact"),
        scaling_sizes=(500, 1000, 2000),
        model_overrides=dict(mkl_iters=100, qnn_iters=80,
                             sketch=SketchConfig(m=100, shots=1000)),
    ),
    "paper": dict(
        seeds=(0, 1, 2, 3, 4), n_samples=1000,
        datasets=("S1_mixed", "S2_in_rkhs", "S3_entangled_anchor",
                  "S4_high_freq", "S5_noise", "breast_cancer", "digits01",
                  "wine", "adult", "electricity", "magic"),
        mixing_grid=(0.0, 0.25, 0.5, 0.75, 1.0), noise_grid=(0.0, 0.1, 0.2),
        m_grid=(100, 200, 400), shots_grid=(200, 1000, "exact"),
        scaling_sizes=(1000, 2000, 4000, 8000),
        model_overrides=dict(mkl_iters=200, qnn_iters=300,
                             sketch=SketchConfig(m=200, shots=1000)),
    ),
}


def build_config(args: argparse.Namespace) -> ProtocolConfig:
    prof = PROFILES[args.profile]
    cfg = ProtocolConfig(
        seeds=tuple(args.seeds) if args.seeds else prof["seeds"],
        n_samples=args.n_samples or prof["n_samples"],
        datasets=tuple(args.datasets) if args.datasets else prof["datasets"],
        models=tuple(args.models) if args.models else DEFAULT_MODELS,
        mixing_grid=prof["mixing_grid"], noise_grid=prof["noise_grid"],
        m_grid=prof["m_grid"], shots_grid=prof["shots_grid"],
        scaling_sizes=prof["scaling_sizes"],
        model_cfg=ModelConfig(**prof["model_overrides"]),
    )
    return cfg


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", choices=sorted(PROFILES), default="smoke",
                   help="preset knob set (default: smoke)")
    p.add_argument("--studies", nargs="+", default=["E1", "E2", "E3", "E4", "E5"],
                   choices=["E1", "E2", "E3", "E4", "E5"])
    p.add_argument("--datasets", nargs="+", default=None,
                   help="override the dataset list (see hqkm.list_datasets())")
    p.add_argument("--models", nargs="+", default=None,
                   help="override the model list (see hqkm.ALL_MODELS)")
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--out", default="results.json")
    p.add_argument("--list-datasets", action="store_true")
    args = p.parse_args(argv)

    if args.list_datasets:
        for name, spec in __import__("hqkm").list_datasets().items():
            print(f"{name:22s} {spec.kind:9s} {spec.purpose}")
        return 0

    cfg = build_config(args)
    print(f"profile={args.profile} studies={args.studies} "
          f"seeds={cfg.seeds} n={cfg.n_samples} datasets={cfg.datasets}",
          file=sys.stderr)
    out = run_all(cfg, studies=tuple(args.studies), out_path=args.out)
    for key, study in out["studies"].items():
        rows = len(study.get("rows", []))
        print(f"{key}: {rows} rows in {study['wall_clock_s']:.1f}s")
    print(f"wrote {args.out} (total {out['total_wall_clock_s']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
