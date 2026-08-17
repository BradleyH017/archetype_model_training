#!/usr/bin/env python3
"""
005_aggregate.py
Aggregate per-cluster evaluation results across all clusters.

Outputs (in --outdir):
  all_clusters_metrics.csv         — full long-format metrics table
  model_summary.csv                — mean R² / RMSE per model, pooling every
                                      individual archetype/PC row uniformly
                                      (naive — a cluster with more targets
                                      counts more; see model_summary_overall.csv)
  all_clusters_overall_metrics.csv — one row per (cluster × model × label_mode):
                                      multivariate (variance-weighted) R² and
                                      aggregate RMSE across that cluster's targets
  model_summary_overall.csv        — per model, r2_variance_weighted/rmse_aggregate
                                      averaged ACROSS CLUSTERS (each cluster
                                      contributes one number, regardless of how
                                      many targets it has) — a more principled
                                      ranking than model_summary.csv's flat pooling
  {model}_r2_heatmap.pdf       — R² heatmap: clusters × archetypes
  model_r2_boxplot.pdf         — R² distribution per model
  model_r2_stripplot.pdf       — all data points, faceted by cluster
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# No helpers functions needed here beyond stdlib/seaborn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dirs", nargs="+", required=True,
                   help="Per-cluster evaluation directories")
    p.add_argument("--outdir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # ── Load per-cluster metrics ───────────────────────────────────────────────
    dfs = []
    for d in args.eval_dirs:
        f = os.path.join(d, "combined_metrics.csv")
        if os.path.exists(f):
            dfs.append(pd.read_csv(f))
    if not dfs:
        print("No combined_metrics.csv files found.")
        return

    df = pd.concat(dfs, ignore_index=True)
    if "cluster" not in df.columns:
        df["cluster"] = df["archetype"].str.extract(r'^([A-Za-z]+)')
    df.to_csv(os.path.join(args.outdir, "all_clusters_metrics.csv"), index=False)
    print(f"Loaded {len(df)} rows from {len(dfs)} clusters.")

    # ── Per-model summary ──────────────────────────────────────────────────────
    summary = (df.groupby("model")
                 .agg(
                     r2_mean      = ("r2",   "mean"),
                     r2_std       = ("r2",   "std"),
                     r2_median    = ("r2",   "median"),
                     rmse_mean    = ("rmse", "mean"),
                     rmse_std     = ("rmse", "std"),
                     n_archetypes = ("archetype", "count"),
                 )
                 .reset_index()
                 .sort_values("r2_mean", ascending=False))
    summary.to_csv(os.path.join(args.outdir, "model_summary.csv"), index=False)
    print("\nModel summary:")
    print(summary.to_string(index=False))

    # ── Overall (multivariate, variance-weighted) per-cluster metrics ─────────
    overall_dfs = []
    for d in args.eval_dirs:
        f = os.path.join(d, "combined_overall_metrics.csv")
        if os.path.exists(f):
            overall_dfs.append(pd.read_csv(f))
    if overall_dfs:
        df_overall = pd.concat(overall_dfs, ignore_index=True)
        df_overall.to_csv(os.path.join(args.outdir, "all_clusters_overall_metrics.csv"), index=False)
        print(f"\nLoaded {len(df_overall)} overall-metric rows from {len(overall_dfs)} clusters.")

        # Each row here is already one whole-cluster aggregate (not a per-target
        # row), so a plain mean across rows correctly averages "per cluster",
        # not "per archetype" — unlike model_summary.csv above.
        summary_overall = (df_overall.groupby("model")
                              .agg(
                                  r2_variance_weighted_mean = ("r2_variance_weighted", "mean"),
                                  r2_variance_weighted_std  = ("r2_variance_weighted", "std"),
                                  rmse_aggregate_mean       = ("rmse_aggregate", "mean"),
                                  rmse_aggregate_std        = ("rmse_aggregate", "std"),
                                  n_clusters                = ("cluster", "count"),
                              )
                              .reset_index()
                              .sort_values("r2_variance_weighted_mean", ascending=False))
        summary_overall.to_csv(os.path.join(args.outdir, "model_summary_overall.csv"), index=False)
        print("\nModel summary (overall, variance-weighted, averaged across clusters):")
        print(summary_overall.to_string(index=False))

    models = df["model"].unique()

    # ── R² heatmap per model ───────────────────────────────────────────────────
    for model in models:
        sub   = df[df["model"] == model].copy()
        pivot = sub.pivot_table(index="cluster", columns="archetype", values="r2")
        fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.1),
                                        max(3, len(pivot.index) * 0.8)))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn",
                    vmin=0, vmax=1, ax=ax, linewidths=0.5,
                    cbar_kws={"label": "OOF R²"})
        ax.set_title(f"{model.upper()} — OOF R² by cluster and archetype")
        ax.set_xlabel("Archetype")
        ax.set_ylabel("Cluster")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"{model}_r2_heatmap.pdf"), dpi=150)
        plt.close()

    # ── Box plot: R² distribution by model ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(5, len(models) * 2), 5))
    sns.boxplot(data=df, x="model", y="r2", ax=ax, order=sorted(models), width=0.5)
    sns.stripplot(data=df, x="model", y="r2", ax=ax, order=sorted(models),
                  color="black", alpha=0.4, size=4, jitter=True)
    ax.axhline(0, color="r", lw=0.8, ls="--")
    ax.set_title("OOF R² distribution across all clusters and archetypes")
    ax.set_xlabel("Model")
    ax.set_ylabel("R²")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "model_r2_boxplot.pdf"), dpi=150)
    plt.close()

    # ── Strip plot faceted by cluster ─────────────────────────────────────────
    clusters = sorted(df["cluster"].unique())
    n_cl     = len(clusters)
    fig, axes = plt.subplots(1, n_cl, figsize=(4 * n_cl, 5), sharey=True)
    if n_cl == 1:
        axes = [axes]
    for ax, cl in zip(axes, clusters):
        sub = df[df["cluster"] == cl]
        sns.stripplot(data=sub, x="model", y="r2", hue="archetype",
                      ax=ax, dodge=True, jitter=True, alpha=0.7, size=6)
        ax.axhline(0, color="r", lw=0.8, ls="--")
        ax.set_title(f"Cluster {cl}")
        ax.set_xlabel("")
        if ax is not axes[0]:
            ax.set_ylabel("")
        ax.legend(fontsize=7, title="Archetype", loc="lower right")
    fig.suptitle("Per-cluster OOF R² by model and archetype", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "model_r2_stripplot.pdf"), dpi=150)
    plt.close()

    print(f"\nAggregate results saved to {args.outdir}")


if __name__ == "__main__":
    main()
