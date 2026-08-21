#!/usr/bin/env python3
"""
006_finalize_model.py
Refit ONE final, deployable model for a chosen cluster/model/label_mode, using
ALL currently available samples — no outer holdout.

This is NOT part of the automatic Nextflow DAG (main.nf) and is not meant to
run automatically. 002_train_tree_models.py's nested CV exists purely to
ESTIMATE generalization performance honestly — it deliberately trains and
discards 5 different models per fold, and never produces anything you could
apply to a genuinely new sample. Run this script manually, AFTER using nested
CV (and the overall_metrics()/model_summary_overall.csv comparison) to decide
which cluster/model/label_mode is actually worth finalizing.

What it does:
  1. Loads a cluster's full pseudobulk feature matrix and targets (same inputs
     002_train_tree_models.py uses).
  2. Selects HVGs ONCE from the full dataset (no fold split left to protect —
     every sample here is "training" from this point on).
  3. Fits a StandardScaler on the full (HVG-subset) feature matrix.
  4. Runs RandomizedSearchCV with plain (non-nested) k-fold CV — this performs
     hyperparameter selection AND refits on the full data with the winning
     combination in one call (`refit=True`), which is exactly what's needed
     here since there is no further held-out set to protect.
  5. Persists everything needed to score a new sample later — the fitted
     estimator(s), scaler, exact feature names/order, and metadata — to one
     joblib bundle.

Technical covariate regression (--meta_file / --train_only_covs /
--train_test_covs), if used, is applied once to the full dataset AFTER HVG
selection (selection always runs on raw, pre-regression values — see
helpers._fit_outer_fold's docstring; there's no held-out fold here to protect
either way — see step 2 above). Only the train_test_covs adjustment is
persisted in the bundle (as bundle["covariate_model"]) for reapplication to
genuinely new samples at scoring time — see helpers.regress_out_covariates for
why train_only_covs is fit-and-discarded instead. bundle["train_only_covs"]
records which covariates were used for provenance only; they are never
reapplied.

Does NOT support "mil": MIL's persistence needs a meaningfully different
format (torch model state dict, attention weights) and is out of scope here —
running this script with --model mil raises a clear error rather than doing
something silently wrong.

Example (using paths PREPARE_DATA already publishes):
    python scripts/006_finalize_model.py \\
        --cluster B \\
        --pseudobulk results/prepared_data/B.pseudobulk.npz \\
        --targets    results/prepared_data/B.targets.csv \\
        --model      elasticnet_indep \\
        --elasticnet_params scripts/param_grids/elasticnet_params.json \\
        --hvg_pct 0.20 --hvg_max 3000 \\
        --label_mode full \\
        --outdir results/final_models
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

sys.path.append("scripts")
from helpers import (
    apply_label_mode,
    build_estimator,
    load_meta_file,
    load_pseudobulk,
    regress_out_covariates,
    select_hvgs,
    select_hvgs_confound_aware,
    split_covariate_names,
    validate_covariate_lists,
    VARIANCE_WEIGHTED_R2_SCORER,
)

_INDEP_MODELS = ("elasticnet_indep", "lasso_indep", "xgboost_indep")
# xgboost_indep is fit per-target here (not via MultiOutputRegressor — see
# helpers.run_nested_cv_tree), same reasoning as elasticnet_indep/lasso_indep.
# xgboost_joint natively accepts 2D y (multi_strategy="multi_output_tree",
# requires XGBoost>=2.0 — see helpers.build_estimator), so it belongs with the
# other genuinely joint fits below despite the name similarity to xgboost_indep.
_JOINT_MODELS = ("xgboost_joint", "rf", "elasticnet", "lasso")
_SUPPORTED    = _JOINT_MODELS + _INDEP_MODELS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cluster",          required=True)
    p.add_argument("--pseudobulk",       required=True)
    p.add_argument("--targets",          required=True)
    p.add_argument("--model",            required=True,
                   choices=_SUPPORTED + ("mil",),
                   help="Model to finalize. 'mil' is not supported by this script — "
                        "see module docstring.")
    p.add_argument("--xgb_params",       default=None)
    p.add_argument("--rf_params",        default=None)
    p.add_argument("--elasticnet_params", default=None)
    p.add_argument("--lasso_params",     default=None)
    p.add_argument("--hvg_pct",          type=float, default=0.20)
    p.add_argument("--hvg_max",          type=int,   default=2000)
    p.add_argument("--label_mode",       default="full",
                   choices=["full", "argmax", "argmax_filtered"])
    p.add_argument("--min_score",        type=float, default=0.5,
                   help="Minimum dominant archetype weight for argmax_filtered mode")
    p.add_argument("--n_folds",          type=int, default=5,
                   help="Plain (non-nested) k-fold CV used only to pick hyperparameters "
                        "before the final refit on all data")
    p.add_argument("--n_iter",           type=int, default=50)
    p.add_argument("--seed",             type=int, default=42)
    p.add_argument("--n_jobs",           type=int, default=1,
                   help="n_jobs for RandomizedSearchCV")
    p.add_argument("--device",           default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--meta_file",        default=None,
                   help="Per-sample metadata CSV — must match whatever the real "
                        "002_train_tree_models.py run used, if it used covariate regression.")
    p.add_argument("--train_only_covs",  nargs="*", default=[])
    p.add_argument("--train_test_covs",  nargs="*", default=[])
    p.add_argument("--hvg_select_method", default="default",
                   choices=["default", "sq_target_vs_technical"],
                   help="Should match whatever the real 002_train_tree_models.py run used — "
                        "see helpers.select_hvgs_confound_aware and temp/hvg_selection.txt. "
                        "Uses one shared panel (against the full target matrix) here "
                        "regardless of model, unlike 002's per-target panels for "
                        "elasticnet_indep/lasso_indep — this script already uses one shared "
                        "panel for every model (no fold-specific reselection at all, since "
                        "there's no fold), so that simplification is kept for this method too.")
    p.add_argument("--outdir",           required=True)
    return p.parse_args()


def _param_path_for(model_name, args):
    return {
        "xgboost_indep":    args.xgb_params,
        "xgboost_joint":    args.xgb_params,
        "rf":               args.rf_params,
        "elasticnet":       args.elasticnet_params,
        "lasso":            args.lasso_params,
        "elasticnet_indep": args.elasticnet_params,
        "lasso_indep":      args.lasso_params,
    }[model_name]


def main():
    args = parse_args()
    args.train_only_covs = split_covariate_names(args.train_only_covs)
    args.train_test_covs = split_covariate_names(args.train_test_covs)
    if args.hvg_select_method == "sq_target_vs_technical" and not (args.train_only_covs or args.train_test_covs):
        raise ValueError(
            "--hvg_select_method sq_target_vs_technical requires a non-empty "
            "--train_only_covs or --train_test_covs (it ranks genes by target "
            "signal relative to covariate confound exposure — scored against "
            "--train_only_covs if given, else --train_test_covs — which is "
            "undefined without at least one)."
        )

    if args.model == "mil":
        raise ValueError(
            "006_finalize_model.py does not support MIL — its persistence needs a "
            "different format (torch model state, attention weights) and is out of "
            "scope for this script. Use one of: " + ", ".join(_SUPPORTED)
        )

    param_path = _param_path_for(args.model, args)
    if param_path is None:
        raise ValueError(f"--model {args.model} requires the matching --*_params grid path")

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading pseudo-bulk from {args.pseudobulk}")
    X, samples, feature_names, n_gene_features = load_pseudobulk(args.pseudobulk)
    print(f"  {n_gene_features} candidate genes + {len(feature_names) - n_gene_features} "
          f"abundance features")

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
              f"(max weight <= {args.min_score}), {n_kept} remain")
        X       = X[keep_mask]
        samples = [s for s, k in zip(samples, keep_mask) if k]
        y       = y[keep_mask]

    df_meta = None
    if args.train_only_covs or args.train_test_covs:
        if not args.meta_file:
            raise ValueError("--train_only_covs/--train_test_covs given but --meta_file is not set")
        print(f"Loading covariate metadata from {args.meta_file}")
        df_meta = load_meta_file(args.meta_file)
        validate_covariate_lists(df_meta, args.train_only_covs, args.train_test_covs)
        keep_idx = [i for i, s in enumerate(samples) if s in df_meta.index]
        if len(keep_idx) < len(samples):
            print(f"  Dropping {len(samples) - len(keep_idx)} samples not in meta_file")
        X       = X[keep_idx]
        samples = [samples[i] for i in keep_idx]
        y       = y[keep_idx]

    print(f"X={X.shape}  y={y.shape}  label_mode={args.label_mode}")

    # ── HVG selection: once, on the full RAW dataset (before any covariate
    # regression) — no held-out fold to protect from this point on, since
    # this fit is meant to be applied to genuinely new samples, not evaluated
    # on any of the current ones. Selection always precedes regression (see
    # helpers._fit_outer_fold's docstring): "sq_target_vs_technical" needs
    # covariate_r2 computed on unadjusted values, and regressing only the
    # selected columns afterward is cheaper than regressing every candidate
    # gene (regression is column-independent, so this gives identical
    # results either way). One shared panel for every model (see
    # --hvg_select_method help), even elasticnet_indep/lasso_indep/
    # xgboost_indep — unlike 002_train_tree_models.py's per-target panels for
    # the former two, matching this script's existing simpler
    # (non-fold-specific) architecture.
    if args.hvg_select_method == "sq_target_vs_technical":
        confound_covs = args.train_only_covs if args.train_only_covs else args.train_test_covs
        hvg_idx_local = select_hvgs_confound_aware(
            X[:, :n_gene_features], y, df_meta, samples, confound_covs, args.hvg_pct, args.hvg_max,
        )
    else:
        hvg_idx_local = select_hvgs(X[:, :n_gene_features], args.hvg_pct, args.hvg_max)
    abund_idx     = np.arange(n_gene_features, X.shape[1])
    sel_idx       = np.concatenate([hvg_idx_local, abund_idx])
    X             = X[:, sel_idx]
    final_feature_names = [feature_names[i] for i in sel_idx]
    print(f"  Selected {len(hvg_idx_local)} HVGs + {len(abund_idx)} abundance features "
          f"= {X.shape[1]} final features")

    covariate_model = None  # (ColumnTransformer, LinearRegression) for train_test_covs, or None
    if args.train_only_covs or args.train_test_covs:
        print(f"  Regressing out covariates (fit on all {len(samples)} samples — no held-out "
              f"fold here): train_only_covs={args.train_only_covs}  train_test_covs={args.train_test_covs}")
        # No held-out fold in this script — every sample is "training" from
        # here on, so X_te == X_tr; only the train_test_covs adjustment
        # (covariate_model) is meaningful afterwards, for reapplication to
        # genuinely new samples at scoring time.
        X, _, covariate_model = regress_out_covariates(
            X, X, df_meta, samples, samples, args.train_only_covs, args.train_test_covs,
        )

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_targets = y.shape[1]
    cv = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    def _fit_one(model_name, y_col, target_label):
        base, grid = build_estimator(model_name, param_path, args.seed,
                                     n_jobs=1, device=args.device)
        search = RandomizedSearchCV(
            base, grid,
            n_iter      = args.n_iter,
            cv          = cv,
            scoring     = VARIANCE_WEIGHTED_R2_SCORER,
            random_state= args.seed,
            n_jobs      = args.n_jobs,
            refit       = True,
            error_score = "raise",
        )
        search.fit(X_scaled, y_col)
        print(f"  [{target_label}] best CV R² (variance-weighted) = "
              f"{search.best_score_:.3f}  best_params = {search.best_params_}")
        return search.best_estimator_, search.best_params_, search.best_score_

    print(f"\nFitting {args.model} ...")
    if args.model in _INDEP_MODELS:
        estimators, best_params, best_scores = [], [], []
        for j, col in enumerate(arch_cols):
            est, params, score = _fit_one(args.model, y[:, j], col)
            estimators.append(est)
            best_params.append(params)
            best_scores.append(score)
        bundle = {
            "model_name":     args.model,
            "cluster":        args.cluster,
            "label_mode":     args.label_mode,
            "estimators":     estimators,
            "target_names":   arch_cols,
            "scaler":         scaler,
            "feature_names":  final_feature_names,
            "n_gene_features": len(hvg_idx_local),
            "hvg_pct":        args.hvg_pct,
            "hvg_max":        args.hvg_max,
            "best_params":    best_params,
            "best_cv_scores": best_scores,
            "covariate_model": covariate_model,
            "train_test_covs": args.train_test_covs,
            "train_only_covs": args.train_only_covs,
            "hvg_select_method": args.hvg_select_method,
        }
    else:
        y_fit = y if n_targets > 1 else y.ravel()
        est, params, score = _fit_one(args.model, y_fit, "joint")
        bundle = {
            "model_name":     args.model,
            "cluster":        args.cluster,
            "label_mode":     args.label_mode,
            "estimator":      est,
            "target_names":   arch_cols,
            "scaler":         scaler,
            "feature_names":  final_feature_names,
            "n_gene_features": len(hvg_idx_local),
            "hvg_pct":        args.hvg_pct,
            "hvg_max":        args.hvg_max,
            "best_params":    params,
            "best_cv_scores": score,
            "covariate_model": covariate_model,
            "train_test_covs": args.train_test_covs,
            "train_only_covs": args.train_only_covs,
            "hvg_select_method": args.hvg_select_method,
        }

    outpath = os.path.join(args.outdir, f"{args.cluster}_{args.model}_{args.label_mode}_final_model.joblib")
    joblib.dump(bundle, outpath)
    print(f"\nSaved final model bundle to {outpath}")


if __name__ == "__main__":
    main()
