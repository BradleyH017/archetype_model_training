#!/usr/bin/env python3
"""
002_train_tree_models.py
Nested cross-validation for XGBoost and/or RandomForest on pseudo-bulk archetype prediction.

Outputs (in --outdir):
  {model}_fold_metrics.csv         — per-fold mean R² / RMSE + best params
  {model}_oof_predictions.csv      — out-of-fold predictions (samples × archetypes)
  {model}_feature_importances.csv  — mean importance across outer folds
  {model}_summary_metrics.csv      — per-archetype OOF R² / RMSE (one file per model)

Label modes (--label_mode):
  full            — all continuous archetype weights (default, original behaviour)
  argmax          — full cohort, targets converted to one-hot at dominant archetype
  argmax_filtered — cohort filtered to max archetype weight > --min_score, then one-hot
"""
# TESTING: module load HGI/softpack/users/eh19/BioAUTOMATED/1
# module load cuda-12.1.1
# module load gcc-13.1.0
import argparse
import os
import sys
import xgboost as xgb

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.append("scripts")
from helpers import (
    apply_label_mode,
    build_estimator,
    load_pseudobulk,
    per_archetype_metrics,
    run_nested_cv_tree,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cluster",          required=True)
    p.add_argument("--pseudobulk",       required=True)
    p.add_argument("--targets",          required=True)
    p.add_argument("--xgb_params",       required=True)
    p.add_argument("--rf_params",        required=True)
    p.add_argument("--models",           nargs="+", default=["xgboost", "rf"])
    p.add_argument("--n_outer_folds",    type=int, default=5)
    p.add_argument("--n_inner_folds",    type=int, default=5)
    p.add_argument("--n_iter",           type=int, default=50)
    p.add_argument("--seed",             type=int, default=42)
    p.add_argument("--outdir",           required=True)
    # Parallelism: n_jobs_outer runs outer folds in parallel (set = n_outer_folds for max);
    #              n_jobs_inner parallelises RandomizedSearchCV iterations within a fold.
    # Avoid oversubscription: if either > 1, leave estimator_n_jobs at the default of 1.
    p.add_argument("--n_jobs_outer",     type=int, default=1,
                   help="Parallel outer folds (1=sequential, -1=all cores, or n_outer_folds)")
    p.add_argument("--n_jobs_inner",     type=int, default=1,
                   help="n_jobs for RandomizedSearchCV inner HPO iterations")
    p.add_argument("--estimator_n_jobs", type=int, default=None,
                   help="n_jobs inside base estimator; auto=1 when outer/inner parallelism on")
    p.add_argument("--n_jobs_models",    type=int, default=-1,
                   help="Parallel models (-1=all models simultaneously, 1=sequential)")
    p.add_argument("--device",           default="cpu", choices=["cpu", "cuda"],
                   help="XGBoost device; 'cuda' uses gpu_hist (submit to a GPU node)")
    p.add_argument("--label_mode",       default="full",
                   choices=["full", "argmax", "argmax_filtered"],
                   help="Target encoding: 'full'=continuous weights, 'argmax'=one-hot at "
                        "dominant archetype (full cohort), 'argmax_filtered'=one-hot + "
                        "restrict to samples where max weight > --min_score")
    p.add_argument("--min_score",        type=float, default=0.5,
                   help="Minimum dominant archetype weight for argmax_filtered mode")

    # ── Interactive testing ───────────────────────────────────────────────────
    # Uncomment the block below and swap `return p.parse_args()` → `return args`
    # to run interactively without passing command-line flags.
    args = argparse.Namespace(
         cluster         = "D",
         pseudobulk      = "results/prepared_data/D.pseudobulk.npz",
         targets         = "results/prepared_data/D.targets.csv",
         xgb_params      = "scripts/param_grids/xgboost_params_small.json",
         rf_params       = "scripts/param_grids/rf_params.json",
         models          = ["xgboost"],        # one model per job; swap to ["xgboost"] as needed
         n_outer_folds   = 5,
         n_inner_folds   = 5,
         n_iter          = 50,
         seed            = 42,
         outdir          = "temp/D",
         n_jobs_outer    = 1,
         n_jobs_inner    = 1,
         estimator_n_jobs= 4,
         n_jobs_models   = 1,             # single model — no inter-model parallelism needed
         device          = "cpu",
         label_mode      = "argmax_filtered",
         min_score       = 0.5,
    )
    # ─────────────────────────────────────────────────────────────────────────

    return p.parse_args()


def _run_one_model(model_name, args, estimator_n_jobs, X, y, arch_cols, samples, feature_names):
    """Train a single model with nested CV and return its results."""
    param_path = args.xgb_params if model_name == "xgboost" else args.rf_params
    device     = args.device if model_name == "xgboost" else "cpu"
    base, grid = build_estimator(model_name, param_path, args.seed,
                                 n_jobs=estimator_n_jobs, device=device)
    print(f"\n{'='*40}\n{model_name.upper()}", flush=True)
    fold_metrics, oof_preds, all_imps = run_nested_cv_tree(
        X, y, model_name, base, grid,
        args.n_outer_folds, args.n_inner_folds, args.n_iter,
        args.seed, feature_names,
        n_jobs_outer=args.n_jobs_outer,
        n_jobs_inner=args.n_jobs_inner,
    )
    return model_name, fold_metrics, oof_preds, all_imps


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Auto-set estimator n_jobs: use 1 when outer/inner parallelism is active to
    # avoid thread oversubscription; otherwise allow the estimator to use more cores.
    estimator_n_jobs = args.estimator_n_jobs
    if estimator_n_jobs is None:
        estimator_n_jobs = 1 if (args.n_jobs_outer != 1 or args.n_jobs_inner != 1) else 4

    print(f"Loading pseudo-bulk from {args.pseudobulk}")
    X, samples, feature_names = load_pseudobulk(args.pseudobulk)

    print(f"Loading targets from {args.targets}")
    df_tgt = pd.read_csv(args.targets, index_col=0)
    keep_idx = [i for i, s in enumerate(samples) if s in df_tgt.index]
    if len(keep_idx) < len(samples):
        print(f"  Dropping {len(samples) - len(keep_idx)} samples not in targets")
    X = X[keep_idx]
    samples = [samples[i] for i in keep_idx]
    y = df_tgt.loc[samples].values.astype(float)
    arch_cols = df_tgt.columns.tolist()

    y, keep_mask = apply_label_mode(y, args.label_mode, args.min_score)
    if not keep_mask.all():
        n_kept = keep_mask.sum()
        print(f"  label_mode='{args.label_mode}': {(~keep_mask).sum()} samples dropped "
              f"(max weight ≤ {args.min_score}), {n_kept} remain")
        X       = X[keep_mask]
        samples = [s for s, k in zip(samples, keep_mask) if k]
        y       = y[keep_mask]

    print(f"X={X.shape}  y={y.shape}  label_mode={args.label_mode}")
    print(f"n_jobs_outer={args.n_jobs_outer}  n_jobs_inner={args.n_jobs_inner}"
          f"  estimator_n_jobs={estimator_n_jobs}  device={args.device}")

    n_jobs_models = (len(args.models) if args.n_jobs_models < 0
                     else min(args.n_jobs_models, len(args.models)))
    print(f"Running {len(args.models)} model(s) with n_jobs_models={n_jobs_models}")

    model_results = Parallel(n_jobs=n_jobs_models, prefer="processes")(
        delayed(_run_one_model)(
            model_name, args, estimator_n_jobs, X, y, arch_cols, samples, feature_names
        )
        for model_name in args.models
    )

    for model_name, fold_metrics, oof_preds, all_imps in model_results:
        metrics = per_archetype_metrics(y, oof_preds, arch_cols, model_name, args.cluster, args.label_mode)

        pd.DataFrame(fold_metrics).to_csv(
            f"{args.outdir}/{model_name}_fold_metrics.csv", index=False)

        pd.DataFrame(oof_preds, columns=arch_cols, index=samples).to_csv(
            f"{args.outdir}/{model_name}_oof_predictions.csv")

        pd.DataFrame(metrics).to_csv(
            f"{args.outdir}/{model_name}_summary_metrics.csv", index=False)

        if all_imps:
            mean_imp = np.mean(all_imps, axis=0)
            (pd.DataFrame({"feature": feature_names, "importance": mean_imp})
               .sort_values("importance", ascending=False)
               .to_csv(f"{args.outdir}/{model_name}_feature_importances.csv", index=False))

    print(f"\nDone. Results in {args.outdir}")


if __name__ == "__main__":
    main()
