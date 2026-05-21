#!/usr/bin/env python3
"""
004_evaluate.py
Per-cluster model comparison: loads OOF predictions from tree and MIL models,
produces a combined metrics table and diagnostic plots.

Outputs (in --outdir):
  combined_metrics.csv            — all models × all archetypes
  model_comparison_bars.pdf       — R² and RMSE bar charts by model
  {model}_oof_scatter.pdf         — true vs predicted scatter per archetype
  {model}_feature_importance.pdf  — top-30 features (tree models only)
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import load_oof


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cluster",  required=True)
    p.add_argument("--tree_dir", required=True)
    p.add_argument("--mil_dir",  required=True)
    p.add_argument("--targets",  required=True)
    p.add_argument("--outdir",   required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df_tgt = pd.read_csv(args.targets, index_col=0)

    # ── Collect summary metrics ───────────────────────────────────────────────
    metric_paths = [
        os.path.join(args.tree_dir, "summary_metrics.csv"),
        os.path.join(args.mil_dir,  "summary_metrics.csv"),
    ]
    dfs = [pd.read_csv(p) for p in metric_paths if os.path.exists(p)]
    if not dfs:
        print("No summary_metrics.csv files found — nothing to evaluate.")
        return
    df_metrics = pd.concat(dfs, ignore_index=True)
    df_metrics.to_csv(os.path.join(args.outdir, "combined_metrics.csv"), index=False)
    print(df_metrics.to_string(index=False))

    arch_cols = df_tgt.columns.tolist()
    models    = df_metrics["model"].unique()

    # ── Model comparison bar charts ───────────────────────────────────────────
    pivot_r2   = df_metrics.pivot_table(index="archetype", columns="model", values="r2")
    pivot_rmse = df_metrics.pivot_table(index="archetype", columns="model", values="rmse")

    fig, axes = plt.subplots(1, 2, figsize=(5 * len(models) + 4, 5))
    pivot_r2.plot.bar(ax=axes[0], rot=30, width=0.7)
    axes[0].set_title(f"Cluster {args.cluster}: OOF R² per archetype")
    axes[0].set_ylabel("R²")
    axes[0].axhline(0, color="k", lw=0.8, ls="--")
    axes[0].legend(title="Model")

    pivot_rmse.plot.bar(ax=axes[1], rot=30, width=0.7)
    axes[1].set_title(f"Cluster {args.cluster}: OOF RMSE per archetype")
    axes[1].set_ylabel("RMSE")
    axes[1].legend(title="Model")

    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "model_comparison_bars.pdf"), dpi=150)
    plt.close()

    # ── OOF scatter plots ─────────────────────────────────────────────────────
    model_sources = {
        "xgboost": args.tree_dir,
        "rf":      args.tree_dir,
        "mil":     args.mil_dir,
    }
    for model_name, mdir in model_sources.items():
        df_oof = load_oof(mdir, model_name)
        if df_oof is None:
            continue
        cols = [c for c in arch_cols if c in df_oof.columns]
        n    = len(cols)
        fig, axes = plt.subplots(1, n, figsize=(4 * n + 1, 4))
        if n == 1:
            axes = [axes]
        for ax, col in zip(axes, cols):
            common = df_tgt.index.intersection(df_oof.index)
            true   = df_tgt.loc[common, col].values
            pred   = df_oof.loc[common, col].values
            mask   = ~(np.isnan(true) | np.isnan(pred))
            true, pred = true[mask], pred[mask]
            ax.scatter(true, pred, alpha=0.5, s=15, rasterized=True)
            lo, hi = min(true.min(), pred.min()), max(true.max(), pred.max())
            ax.plot([lo, hi], [lo, hi], "r--", lw=1)
            denom = ((true - true.mean()) ** 2).sum()
            r2_c  = 1 - ((true - pred) ** 2).sum() / denom if denom > 0 else float("nan")
            ax.set_title(f"{col}\nR²={r2_c:.3f}", fontsize=9)
            ax.set_xlabel("True")
            ax.set_ylabel("Predicted")
        plt.suptitle(f"Cluster {args.cluster} — {model_name.upper()} OOF", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"{model_name}_oof_scatter.pdf"), dpi=150)
        plt.close()

    # ── Feature importance plots ──────────────────────────────────────────────
    for model_name in ("xgboost", "rf"):
        imp_path = os.path.join(args.tree_dir, f"{model_name}_feature_importances.csv")
        if not os.path.exists(imp_path):
            continue
        df_imp = pd.read_csv(imp_path).head(30)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(df_imp["feature"].iloc[::-1], df_imp["importance"].iloc[::-1])
        ax.set_title(f"Cluster {args.cluster} — {model_name.upper()} top-30 features")
        ax.set_xlabel("Mean importance")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"{model_name}_feature_importance.pdf"), dpi=150)
        plt.close()

    print(f"\nEvaluation saved to {args.outdir}")


if __name__ == "__main__":
    main()
