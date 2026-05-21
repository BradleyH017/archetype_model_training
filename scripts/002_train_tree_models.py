#!/usr/bin/env python3
"""
002_train_tree_models.py
Nested cross-validation for XGBoost and/or RandomForest on pseudo-bulk archetype prediction.

Outputs (in --outdir):
  {model}_fold_metrics.csv         — per-fold mean R² / RMSE + best params
  {model}_oof_predictions.csv      — out-of-fold predictions (samples × archetypes)
  {model}_feature_importances.csv  — mean importance across outer folds
  summary_metrics.csv              — per-archetype OOF R² / RMSE for all models
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import (
    build_estimator,
    load_pseudobulk,
    per_archetype_metrics,
    run_nested_cv_tree,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cluster",       required=True)
    p.add_argument("--pseudobulk",    required=True)
    p.add_argument("--targets",       required=True)
    p.add_argument("--xgb_params",    required=True)
    p.add_argument("--rf_params",     required=True)
    p.add_argument("--models",        nargs="+", default=["xgboost", "rf"])
    p.add_argument("--n_outer_folds", type=int, default=5)
    p.add_argument("--n_inner_folds", type=int, default=5)
    p.add_argument("--n_iter",        type=int, default=50)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--outdir",        required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading pseudo-bulk from {args.pseudobulk}")
    X, samples, feature_names = load_pseudobulk(args.pseudobulk)

    print(f"Loading targets from {args.targets}")
    df_tgt   = pd.read_csv(args.targets, index_col=0)
    keep_idx = [i for i, s in enumerate(samples) if s in df_tgt.index]
    if len(keep_idx) < len(samples):
        print(f"  Dropping {len(samples) - len(keep_idx)} samples not in targets")
    X         = X[keep_idx]
    samples   = [samples[i] for i in keep_idx]
    y         = df_tgt.loc[samples].values.astype(float)
    arch_cols = df_tgt.columns.tolist()
    print(f"  X={X.shape}  y={y.shape}")

    all_metrics = []

    for model_name in args.models:
        print(f"\n{'='*40}\n{model_name.upper()}")
        param_path = args.xgb_params if model_name == "xgboost" else args.rf_params
        base, grid = build_estimator(model_name, param_path, args.seed)

        fold_metrics, oof_preds, all_imps = run_nested_cv_tree(
            X, y, model_name, base, grid,
            args.n_outer_folds, args.n_inner_folds, args.n_iter,
            args.seed, feature_names,
        )

        all_metrics.extend(
            per_archetype_metrics(y, oof_preds, arch_cols, model_name, args.cluster))

        pd.DataFrame(fold_metrics).to_csv(
            f"{args.outdir}/{model_name}_fold_metrics.csv", index=False)

        pd.DataFrame(oof_preds, columns=arch_cols, index=samples).to_csv(
            f"{args.outdir}/{model_name}_oof_predictions.csv")

        if all_imps:
            mean_imp = np.mean(all_imps, axis=0)
            (pd.DataFrame({"feature": feature_names, "importance": mean_imp})
               .sort_values("importance", ascending=False)
               .to_csv(f"{args.outdir}/{model_name}_feature_importances.csv", index=False))

    pd.DataFrame(all_metrics).to_csv(
        f"{args.outdir}/summary_metrics.csv", index=False)
    print(f"\nDone. Results in {args.outdir}")


if __name__ == "__main__":
    main()
