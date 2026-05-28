#!/usr/bin/env python3
"""
004_evaluate.py
Per-cluster model comparison: loads OOF predictions from tree and MIL models,
produces a combined metrics table and diagnostic plots.

Tree models are now split into one job per (model × label_mode), so --tree_dirs
accepts multiple directories.  MIL remains a single directory (--mil_dir).

Outputs (in --outdir):
  combined_metrics.csv                      — all models × archetypes × label_modes
  model_comparison_bars.pdf                 — R² and RMSE bar charts by model
  {model}_{label_mode}_oof_scatter.pdf      — true vs predicted scatter per archetype
  {model}_{label_mode}_feature_importance.pdf — top-30 features (tree models only)
"""
import argparse
import glob
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
    p.add_argument("--cluster",   required=True)
    p.add_argument("--tree_dirs", nargs="+", required=True,
                   help="One directory per (model × label_mode) tree job")
    p.add_argument("--mil_dir",   required=True)
    p.add_argument("--targets",   required=True)
    p.add_argument("--outdir",    required=True)
    return p.parse_args()


def _collect_sources(dirs):
    """Scan dirs for *_summary_metrics.csv and return list of (model, label_mode, dir)."""
    sources = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for sm_path in sorted(glob.glob(os.path.join(d, "*_summary_metrics.csv"))):
            df_sm = pd.read_csv(sm_path)
            if df_sm.empty:
                continue
            model_name = df_sm["model"].iloc[0]
            label_mode = df_sm["label_mode"].iloc[0] if "label_mode" in df_sm.columns else "full"
            sources.append((model_name, label_mode, d))
    return sources


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df_tgt = pd.read_csv(args.targets, index_col=0)
    arch_cols = df_tgt.columns.tolist()

    # ── Collect summary metrics ───────────────────────────────────────────────
    # Each (model × label_mode) job writes its own *_summary_metrics.csv.
    all_dirs = list(args.tree_dirs) + [args.mil_dir]
    metric_csvs = []
    for d in all_dirs:
        metric_csvs += glob.glob(os.path.join(d, "*_summary_metrics.csv"))

    dfs = [pd.read_csv(p) for p in sorted(set(metric_csvs)) if os.path.exists(p)]
    if not dfs:
        print("No *_summary_metrics.csv files found — nothing to evaluate.")
        return
    df_metrics = pd.concat(dfs, ignore_index=True)
    df_metrics.to_csv(os.path.join(args.outdir, "combined_metrics.csv"), index=False)
    print(df_metrics.to_string(index=False))

    # ── Model comparison bar charts ───────────────────────────────────────────
    # Group by model+label_mode so each bar is a distinct (model, mode) combo.
    df_metrics["model_mode"] = df_metrics["model"] + "/" + df_metrics.get("label_mode", "full")
    pivot_r2   = df_metrics.pivot_table(index="archetype", columns="model_mode", values="r2")
    pivot_rmse = df_metrics.pivot_table(index="archetype", columns="model_mode", values="rmse")
    n_cols = max(len(pivot_r2.columns), 1)

    fig, axes = plt.subplots(1, 2, figsize=(4 * n_cols + 4, 5))
    pivot_r2.plot.bar(ax=axes[0], rot=30, width=0.7)
    axes[0].set_title(f"Cluster {args.cluster}: OOF R² per archetype")
    axes[0].set_ylabel("R²")
    axes[0].axhline(0, color="k", lw=0.8, ls="--")
    axes[0].legend(title="Model/Mode", fontsize=7)

    pivot_rmse.plot.bar(ax=axes[1], rot=30, width=0.7)
    axes[1].set_title(f"Cluster {args.cluster}: OOF RMSE per archetype")
    axes[1].set_ylabel("RMSE")
    axes[1].legend(title="Model/Mode", fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "model_comparison_bars.pdf"), dpi=150)
    plt.close()

    # ── OOF scatter plots ─────────────────────────────────────────────────────
    # One plot per (model, label_mode) combination found across all tree dirs.
    tree_sources = _collect_sources(args.tree_dirs)
    mil_sources  = _collect_sources([args.mil_dir])

    for model_name, label_mode, mdir in tree_sources + mil_sources:
        df_oof = load_oof(mdir, model_name)
        if df_oof is None:
            continue
        cols = [c for c in arch_cols if c in df_oof.columns]
        n    = len(cols)
        if n == 0:
            continue
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
            lo = min(true.min(), pred.min())
            hi = max(true.max(), pred.max())
            ax.plot([lo, hi], [lo, hi], "r--", lw=1)
            denom = ((true - true.mean()) ** 2).sum()
            r2_c  = 1 - ((true - pred) ** 2).sum() / denom if denom > 0 else float("nan")
            ax.set_title(f"{col}\nR²={r2_c:.3f}", fontsize=9)
            ax.set_xlabel("True")
            ax.set_ylabel("Predicted")
        plt.suptitle(f"Cluster {args.cluster} — {model_name.upper()} [{label_mode}] OOF",
                     fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir,
                                 f"{model_name}_{label_mode}_oof_scatter.pdf"), dpi=150)
        plt.close()

    # ── Feature importance plots ──────────────────────────────────────────────
    for model_name, label_mode, mdir in tree_sources:
        imp_path = os.path.join(mdir, f"{model_name}_feature_importances.csv")
        if not os.path.exists(imp_path):
            continue
        df_imp = pd.read_csv(imp_path).head(30)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(df_imp["feature"].iloc[::-1], df_imp["importance"].iloc[::-1])
        ax.set_title(f"Cluster {args.cluster} — {model_name.upper()} [{label_mode}] top-30 features")
        ax.set_xlabel("Mean importance")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir,
                                 f"{model_name}_{label_mode}_feature_importance.pdf"), dpi=150)
        plt.close()

    print(f"\nEvaluation saved to {args.outdir}")


if __name__ == "__main__":
    main()
