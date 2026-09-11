#!/usr/bin/env python
"""Plot external-test accuracy and AUC Pareto points."""

import argparse
import json

import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument("input");p.add_argument("output")
    a=p.parse_args();d=json.load(open(a.input,encoding="utf-8"))
    fig,axes=plt.subplots(1,2,figsize=(10,4.1))
    labels={"full_hybrid":"Full hybrid","residual_accuracy":"Residual/Acc",
            "uncertainty_accuracy":"Uncertainty/Acc","residual_auc":"Residual/AUC",
            "random_accuracy":"Random/Acc","kmeans_accuracy":"K-means/Acc"}
    markers=["^","s","D","P","X","v"]
    for ax,metric,title in zip(axes,["accuracy","auc"],["Accuracy","AUC"]):
        points=[{"variant":"classical","mean_executions":0,"mean_delta":0}]
        points += [{"variant":v,"mean_executions":z["mean_executions"],
                    "mean_delta":z[metric]["mean_source_delta"]}
                   for v,z in d["aggregate_external"].items()]
        for i,z in enumerate(points):
            name=z["variant"]
            ax.scatter(z["mean_executions"]/1e6,100*z["mean_delta"],s=55,
                       marker="o" if name=="classical" else markers[(i-1)%len(markers)],
                       label="Classical" if name=="classical" else labels[name])
        front=d["pareto_external"][metric]
        ax.plot([z["mean_executions"]/1e6 for z in front],
                [100*z["mean_delta"] for z in front],"--",color="black",lw=1)
        ax.axhline(0,color="grey",lw=.7)
        ax.set_xlabel("Mean circuit executions (millions)");ax.set_ylabel(f"Mean Δ{title} (percentage points)")
        ax.set_title(f"External {title} frontier");ax.grid(alpha=.2)
    handles,legend=axes[1].get_legend_handles_labels()
    fig.legend(handles,legend,loc="lower center",ncol=4,frameon=False,fontsize=8)
    fig.tight_layout(rect=(0,.13,1,1));fig.savefig(a.output,dpi=220,bbox_inches="tight")


if __name__=="__main__":main()
