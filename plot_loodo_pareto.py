#!/usr/bin/env python
"""Create the LOODO accuracy--execution Pareto figure from saved JSON."""

import argparse
import json

import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    args = p.parse_args()
    data = json.load(open(args.input, encoding="utf-8"))
    points = data["policy_points_loodo"]
    styles = {"classical": ("o", "#333333"),
              "cross_validated_gate": ("s", "#D55E00"),
              "always_hybrid": ("^", "#0072B2")}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for policy in styles:
        rows = [r for r in points if r["policy"] == policy]
        marker, color = styles[policy]
        ax.scatter([r["mean_executions"] for r in rows],
                   [100 * r["mean_delivered_accuracy_gain"] for r in rows],
                   marker=marker, color=color, s=55, label=policy.replace("_", " "))
        for r in rows:
            if r["shots"] and policy == "always_hybrid":
                ax.annotate(f'{r["shots"]} shots',
                            (r["mean_executions"], 100*r["mean_delivered_accuracy_gain"]),
                            xytext=(4, 5), textcoords="offset points", fontsize=8)
    front = data["pareto_loodo"]
    ax.plot([r["mean_executions"] for r in front],
            [100*r["mean_delivered_accuracy_gain"] for r in front],
            color="#009E73", linewidth=1.5, linestyle="--", label="Pareto frontier")
    ax.set_xscale("symlog", linthresh=1e5)
    ax.set_xlabel("Mean quantum circuit executions per task–seed")
    ax.set_ylabel("Mean delivered accuracy gain (percentage points)")
    ax.grid(alpha=.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output, dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
