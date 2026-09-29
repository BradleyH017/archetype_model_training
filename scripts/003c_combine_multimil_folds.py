#!/usr/bin/env python3
"""
003c_combine_multimil_folds.py
Stitches 003b_train_multimil.py's per-fold OOF predictions (one Nextflow
task per outer fold, see workflow/main.nf's TRAIN_MULTIMIL) back into the
same output schema 002_train_tree_models.py/003_train_mil.py already
produce, so 004_evaluate.py needs no changes to pick this model up (it
already glob-scans any directory it's given for *_summary_metrics.csv).

Runs on plain container_ml (pandas/numpy/scikit-learn only) -- deliberately
NOT the multimil/torch environment; this step is pure metric arithmetic on
already-computed predictions, no model code involved.

Outputs (in --outdir), model name = "multimil_{target_mode}":
  {cluster}_multimil_{target_mode}_oof_predictions.csv
  {cluster}_multimil_{target_mode}_summary_metrics.csv   -- per-archetype OOF R2/RMSE
  {cluster}_multimil_{target_mode}_overall_metrics.csv   -- variance-weighted multivariate R2/RMSE
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import overall_metrics, per_archetype_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cluster", required=True)
    p.add_argument("--target_mode", choices=["joint", "indep"], required=True)
    p.add_argument("--fold_oof_predictions", nargs="+", required=True,
                   help="One {cluster}_multimil_{target_mode}_fold{i}_oof_predictions.csv per outer fold")
    p.add_argument("--targets", required=True)
    p.add_argument("--label_mode", default="full")
    p.add_argument("--outdir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    model_name = f"multimil_{args.target_mode}"

    df_tgt = pd.read_csv(args.targets, index_col=0)
    arch_cols = df_tgt.columns.tolist()

    fold_dfs = [pd.read_csv(f, index_col=0) for f in args.fold_oof_predictions]
    oof_df = pd.concat(fold_dfs, axis=0)
    oof_df = oof_df.reindex(df_tgt.index)[arch_cols]

    prefix = f"{args.cluster}_{model_name}"
    oof_df.to_csv(os.path.join(args.outdir, f"{prefix}_oof_predictions.csv"))

    y = df_tgt.values
    oof_preds = oof_df.values

    pd.DataFrame(
        per_archetype_metrics(y, oof_preds, arch_cols, model_name, args.cluster, args.label_mode)
    ).to_csv(os.path.join(args.outdir, f"{prefix}_summary_metrics.csv"), index=False)

    pd.DataFrame([
        overall_metrics(y, oof_preds, model_name, args.cluster, args.label_mode)
    ]).to_csv(os.path.join(args.outdir, f"{prefix}_overall_metrics.csv"), index=False)

    n_covered = (~np.isnan(oof_preds).any(axis=1)).sum()
    print(f"Combined {len(fold_dfs)} folds -> {n_covered}/{len(df_tgt)} samples with predictions. "
          f"Results in {args.outdir}")


if __name__ == "__main__":
    main()
