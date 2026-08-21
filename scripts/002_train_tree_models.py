#!/usr/bin/env python3
"""
002_train_tree_models.py
Nested cross-validation for XGBoost, RandomForest, ElasticNet and/or Lasso on
pseudo-bulk archetype prediction. ElasticNet/Lasso use the joint multi-task
variants (MultiTaskElasticNet/MultiTaskLasso): one regularized fit per cluster
across all its archetype targets, sharing feature selection but keeping a
per-target coefficient.

"elasticnet_indep"/"lasso_indep" (only added to --models when
raw_or_clr_pca == 'pca' and run_independent_pca_fits is enabled) fit an
independent single-output ElasticNet/Lasso per target instead — no shared
feature support across targets — to compare against the joint fits above.
This matters because PCA targets are orthogonal by construction, which
weakens the multi-task assumption that targets share a common feature set.

XGBoost has the same indep/joint split, via two distinct model names rather
than a flag, since they need different XGBoost versions:
  "xgboost_indep"  — one independent single-output XGBRegressor per target,
                      via sklearn's MultiOutputRegressor. This was the
                      pipeline's original (and, until now, only) "xgboost"
                      behavior; renamed to make the independence explicit and
                      consistent with elasticnet_indep/lasso_indep. Its
                      feature_importances_ is a single vector averaged across
                      the per-target estimators_, but each target's own fitted
                      model is still a normal single-output XGBRegressor, so
                      SHAP/TreeExplainer works per archetype if computed
                      directly against those individual estimators.
  "xgboost_joint"  — XGBoost's native multi_strategy="multi_output_tree"
                      (requires XGBoost>=2.0): one shared ensemble of trees
                      per cluster, each leaf holding a vector across all
                      archetype targets. Typically run in a separate
                      environment/container with a matching XGBoost version
                      (params.xgboost2_env in the Nextflow workflow, since
                      container_ml's pinned XGBoost predates this feature).
                      Unlike xgboost_indep, feature_importances_ here is
                      genuinely a single vector shared across all targets
                      (not per-archetype) — same as rf — and upstream does
                      not yet support Tree SHAP for this strategy at all.

HVGs are reselected independently for each outer CV fold, from that fold's
training samples only (--hvg_pct/--hvg_max here, not in 001_prepare_data.py)
— see helpers.select_hvgs / _fit_outer_fold. This avoids ranking genes by
variance across samples that later become test data.

Outputs (in --outdir):
  {model}_fold_metrics.csv         — per-fold mean R² / RMSE + best params
  {model}_oof_predictions.csv      — out-of-fold predictions (samples × archetypes)
  {model}_feature_importances.csv  — mean importance across outer folds;
                                      for elasticnet/lasso this is a per-archetype
                                      signed coefficient table (feature × archetype)
                                      rather than a single importance column
  {model}_summary_metrics.csv      — per-archetype OOF R² / RMSE (one file per model)
  {model}_overall_metrics.csv      — one aggregate row per model: multivariate
                                      (variance-weighted) R² and aggregate RMSE
                                      across ALL targets at once — see
                                      helpers.overall_metrics

Hyperparameter search and per-fold R² both use multioutput='variance_weighted'
scoring (not 'uniform_average'): each target column's contribution is weighted
by its own variance rather than averaged uniformly. For CLR-PCA targets this
is equivalent to a proper multivariate R² based on total squared Euclidean
(Aitchison) distance, since PCA preserves distance.

Label modes (--label_mode):
  full            — all continuous archetype weights (default, original behaviour)
  argmax          — full cohort, targets converted to one-hot at dominant archetype
  argmax_filtered — cohort filtered to max archetype weight > --min_score, then one-hot

Technical covariate regression (--meta_file / --train_only_covs / --train_test_covs):
  Optional. Removes technical-covariate effects from the pseudobulk features
  BEFORE HVG selection, fit fresh per outer fold on that fold's training
  samples only — see helpers.regress_out_covariates and the section 2b module
  comment in helpers.py for the train_only_covs vs train_test_covs
  distinction (whether the fitted adjustment is also applied to the held-out
  fold, vs. train-only and discarded). Both default to empty (no adjustment).

HVG selection method (--hvg_select_method): selection always runs on RAW,
pre-covariate-regression values now, with covariates only regressed out of
whichever columns get selected afterward (see helpers._fit_outer_fold's
docstring — regression is column-independent, so this is equivalent to
regressing everything first and subsetting after, just cheaper).
  "default"               — rank candidate genes by raw variance, via
                             helpers.select_hvgs (original behaviour, though
                             note it now sees raw rather than
                             covariate-residualized values — see above).
  "sq_target_vs_technical" — rank by target_r2^2/covariate_r2 instead, via
                             helpers.select_hvgs_confound_aware. Scored
                             against --train_only_covs if non-empty, else
                             --train_test_covs — requires at least one of the
                             two to be non-empty. See temp/hvg_selection.txt
                             for the full derivation, the alternatives
                             considered and rejected, and validation across
                             every cluster/archetype available. For
                             elasticnet_indep/lasso_indep this naturally
                             gives each archetype its own gene panel
                             (target_r2 is computed against that target's own
                             column); for every other model, one shared panel
                             (computed against the full target matrix, or via
                             MultiOutputRegressor for xgboost_indep).
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
    combine_fold_importances,
    load_meta_file,
    load_pseudobulk,
    overall_metrics,
    per_archetype_metrics,
    run_nested_cv_tree,
    run_nested_cv_independent,
    split_covariate_names,
    validate_covariate_lists,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cluster",          required=True)
    p.add_argument("--pseudobulk",       required=True)
    p.add_argument("--targets",          required=True)
    p.add_argument("--xgb_params",       required=True)
    p.add_argument("--rf_params",        required=True)
    p.add_argument("--elasticnet_params", required=True)
    p.add_argument("--lasso_params",     required=True)
    p.add_argument("--models",           nargs="+", default=["xgboost_indep", "rf"])
    p.add_argument("--hvg_pct",          type=float, default=0.20,
                   help="Per-outer-fold HVG selection: fraction of candidate genes to keep")
    p.add_argument("--hvg_max",          type=int,   default=2000,
                   help="Per-outer-fold HVG selection: hard cap on number of HVGs")
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
    p.add_argument("--meta_file",        default=None,
                   help="Per-sample metadata CSV (first column = sample ID, matching "
                        "--sample_col elsewhere in the pipeline). Required if either "
                        "--train_only_covs or --train_test_covs is non-empty.")
    p.add_argument("--train_only_covs",  nargs="*", default=[],
                   help="Covariate column(s) in --meta_file to regress out of the "
                        "TRAINING fold only (not applied to the held-out fold) — for "
                        "covariates unmeasurable/inapplicable in external data, e.g. a "
                        "batch/pooling identifier. See helpers.regress_out_covariates.")
    p.add_argument("--train_test_covs",  nargs="*", default=[],
                   help="Covariate column(s) in --meta_file to regress out of both the "
                        "training AND held-out fold (fit on train, applied to test) — "
                        "for covariates measurable identically in any dataset, e.g. "
                        "total_counts/n_genes_expressed. See helpers.regress_out_covariates.")
    p.add_argument("--hvg_select_method", default="default",
                   choices=["default", "sq_target_vs_technical"],
                   help="'default': rank HVG candidates by raw variance. "
                        "'sq_target_vs_technical': rank by target_r2^2/covariate_r2 "
                        "instead, scored against --train_only_covs if non-empty else "
                        "--train_test_covs (requires at least one non-empty) — see "
                        "helpers.select_hvgs_confound_aware and temp/hvg_selection.txt.")

    # ── Interactive testing ───────────────────────────────────────────────────
    # Uncomment the block below and swap `return p.parse_args()` → `return args`
    # to run interactively without passing command-line flags.
    args = argparse.Namespace(
         cluster         = "D",
         pseudobulk      = "results/prepared_data/D.pseudobulk.npz",
         targets         = "results/prepared_data/D.targets.csv",
         xgb_params      = "scripts/param_grids/xgboost_params_small.json",
         rf_params       = "scripts/param_grids/rf_params.json",
         elasticnet_params = "scripts/param_grids/elasticnet_params.json",
         lasso_params    = "scripts/param_grids/lasso_params.json",
         models          = ["xgboost_indep"],  # one model per job; swap as needed
         hvg_pct         = 0.20,
         hvg_max         = 2000,
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
         meta_file       = None,
         train_only_covs = [],
         train_test_covs = [],
         hvg_select_method = "default",
    )
    # ─────────────────────────────────────────────────────────────────────────

    return p.parse_args()


def _run_one_model(model_name, args, estimator_n_jobs, X, y, arch_cols, samples, feature_names,
                    n_gene_features, df_meta):
    """Train a single model with nested CV and return its results."""
    param_path = {
        "xgboost_indep":    args.xgb_params,
        "xgboost_joint":    args.xgb_params,
        "rf":               args.rf_params,
        "elasticnet":       args.elasticnet_params,
        "lasso":            args.lasso_params,
        "elasticnet_indep": args.elasticnet_params,
        "lasso_indep":      args.lasso_params,
    }[model_name]
    # xgboost_joint always runs CPU/hist regardless of --device — see
    # build_estimator's comment on why GPU isn't wired up for multi_output_tree.
    device     = args.device if model_name == "xgboost_indep" else "cpu"
    base, grid = build_estimator(model_name, param_path, args.seed,
                                 n_jobs=estimator_n_jobs, device=device)
    print(f"\n{'='*40}\n{model_name.upper()}", flush=True)
    if model_name in ("elasticnet_indep", "lasso_indep"):
        fold_metrics, oof_preds, per_target_imps = run_nested_cv_independent(
            X, y, model_name, base, grid,
            args.n_outer_folds, args.n_inner_folds, args.n_iter,
            args.seed, feature_names, target_names=arch_cols,
            n_jobs_outer=args.n_jobs_outer,
            n_jobs_inner=args.n_jobs_inner,
            n_gene_features=n_gene_features, hvg_pct=args.hvg_pct, hvg_max=args.hvg_max,
            samples=samples, covariate_df=df_meta,
            train_only_covs=args.train_only_covs, train_test_covs=args.train_test_covs,
            hvg_select_method=args.hvg_select_method,
        )
        # per_target_imps: list of length n_targets, each its own (sel_idx, imp)-
        # per-fold list — NOT pre-stacked, since hvg_select_method=
        # "sq_target_vs_technical" can select a different gene panel per target
        # (see run_nested_cv_independent's docstring). Caller must combine each
        # target separately, not assume a shared sel_idx.
        return model_name, fold_metrics, oof_preds, per_target_imps
    else:
        fold_metrics, oof_preds, all_imps = run_nested_cv_tree(
            X, y, model_name, base, grid,
            args.n_outer_folds, args.n_inner_folds, args.n_iter,
            args.seed, feature_names,
            n_jobs_outer=args.n_jobs_outer,
            n_jobs_inner=args.n_jobs_inner,
            n_gene_features=n_gene_features, hvg_pct=args.hvg_pct, hvg_max=args.hvg_max,
            samples=samples, covariate_df=df_meta,
            train_only_covs=args.train_only_covs, train_test_covs=args.train_test_covs,
            hvg_select_method=args.hvg_select_method,
        )
        return model_name, fold_metrics, oof_preds, all_imps


def main():
    args = parse_args()
    args.train_only_covs = split_covariate_names(args.train_only_covs)
    args.train_test_covs = split_covariate_names(args.train_test_covs)
    os.makedirs(args.outdir, exist_ok=True)

    # Auto-set estimator n_jobs: use 1 when outer/inner parallelism is active to
    # avoid thread oversubscription; otherwise allow the estimator to use more cores.
    estimator_n_jobs = args.estimator_n_jobs
    if estimator_n_jobs is None:
        estimator_n_jobs = 1 if (args.n_jobs_outer != 1 or args.n_jobs_inner != 1) else 4

    print(f"Loading pseudo-bulk from {args.pseudobulk}")
    X, samples, feature_names, n_gene_features = load_pseudobulk(args.pseudobulk)
    print(f"  {n_gene_features} candidate genes + {len(feature_names) - n_gene_features} "
          f"abundance features; HVGs reselected per outer fold "
          f"(hvg_pct={args.hvg_pct}, hvg_max={args.hvg_max})")

    print(f"Loading targets from {args.targets}")
    df_tgt = pd.read_csv(args.targets, index_col=0)
    keep_idx = [i for i, s in enumerate(samples) if s in df_tgt.index]
    if len(keep_idx) < len(samples):
        print(f"  Dropping {len(samples) - len(keep_idx)} samples not in targets")
    X = X[keep_idx]
    samples = [samples[i] for i in keep_idx]
    y = df_tgt.loc[samples].values.astype(float)
    arch_cols = df_tgt.columns.tolist()

    if args.hvg_select_method == "sq_target_vs_technical" and not (args.train_only_covs or args.train_test_covs):
        raise ValueError(
            "--hvg_select_method sq_target_vs_technical requires a non-empty "
            "--train_only_covs or --train_test_covs (it ranks genes by target "
            "signal relative to covariate confound exposure — scored against "
            "--train_only_covs if given, else --train_test_covs — which is "
            "undefined without at least one)."
        )

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
        print(f"  train_only_covs={args.train_only_covs}  train_test_covs={args.train_test_covs}")

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
            model_name, args, estimator_n_jobs, X, y, arch_cols, samples, feature_names,
            n_gene_features, df_meta,
        )
        for model_name in args.models
    )

    for model_name, fold_metrics, oof_preds, all_imps in model_results:
        metrics = per_archetype_metrics(y, oof_preds, arch_cols, model_name, args.cluster, args.label_mode)
        overall = overall_metrics(y, oof_preds, model_name, args.cluster, args.label_mode)

        pd.DataFrame(fold_metrics).to_csv(
            f"{args.outdir}/{model_name}_fold_metrics.csv", index=False)

        pd.DataFrame(oof_preds, columns=arch_cols, index=samples).to_csv(
            f"{args.outdir}/{model_name}_oof_predictions.csv")

        pd.DataFrame(metrics).to_csv(
            f"{args.outdir}/{model_name}_summary_metrics.csv", index=False)

        pd.DataFrame([overall]).to_csv(
            f"{args.outdir}/{model_name}_overall_metrics.csv", index=False)

        if model_name in ("elasticnet_indep", "lasso_indep"):
            # all_imps here is actually per_target_imps (see run_nested_cv_independent):
            # one (sel_idx, imp)-per-fold list PER TARGET, not pre-stacked, because
            # hvg_select_method="sq_target_vs_technical" can select a different gene
            # panel per target. Combine each target separately, then stack — never
            # assume a shared sel_idx across targets the way the joint-model branch
            # below does.
            per_target_mean_imp, per_target_n_sel = [], []
            for imps_j in all_imps:
                mean_imp_j, n_sel_j = combine_fold_importances(imps_j, len(feature_names))
                per_target_mean_imp.append(mean_imp_j)
                per_target_n_sel.append(n_sel_j)
            if all(m is not None for m in per_target_mean_imp):
                mean_imp_filled = np.nan_to_num(np.vstack(per_target_mean_imp))  # (n_targets, n_features)
                df_imp = (pd.DataFrame(mean_imp_filled.T, index=feature_names, columns=arch_cols)
                            .rename_axis("feature"))
                for j, col in enumerate(arch_cols):
                    df_imp[f"n_folds_selected_{col}"] = per_target_n_sel[j]
                df_imp.to_csv(f"{args.outdir}/{model_name}_feature_importances.csv")
        elif all_imps:
            # Fold-specific HVG reselection means different outer folds may have
            # kept different gene columns; combine_fold_importances scatters each
            # fold's importances onto the full feature axis (NaN where a gene
            # wasn't selected that fold) before averaging, and also reports how
            # many folds selected each feature — a useful stability diagnostic.
            mean_imp, n_folds_selected = combine_fold_importances(all_imps, len(feature_names))
            mean_imp_filled = np.nan_to_num(mean_imp)
            if mean_imp.ndim == 1:
                (pd.DataFrame({"feature": feature_names,
                               "importance": mean_imp_filled,
                               "n_folds_selected": n_folds_selected})
                   .sort_values("importance", ascending=False)
                   .to_csv(f"{args.outdir}/{model_name}_feature_importances.csv", index=False))
            else:
                # Joint multi-task linear models produce mean_imp as
                # (n_targets, n_features), one shared sel_idx across targets
                # (they fit all targets from one shared feature panel, unlike
                # elasticnet_indep/lasso_indep above). Keep one signed
                # coefficient column per archetype so individual clr_pca targets
                # can still be traced back to informative genes.
                df_imp = (pd.DataFrame(mean_imp_filled.T, index=feature_names, columns=arch_cols)
                            .rename_axis("feature"))
                df_imp["n_folds_selected"] = n_folds_selected
                df_imp.to_csv(f"{args.outdir}/{model_name}_feature_importances.csv")

    print(f"\nDone. Results in {args.outdir}")


if __name__ == "__main__":
    main()
