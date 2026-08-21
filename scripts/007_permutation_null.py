#!/usr/bin/env python3
"""
007_permutation_null.py
Per-archetype permutation null test for a (cluster, model, label_mode)
combination already trained by 002_train_tree_models.py.

Answers a narrower, more direct question than a full re-tuning permutation
null would: "given the hyperparameters this model actually settled on across
its real nested-CV folds, does it explain more variance in each archetype PC
than chance, when applied to data with no real signal?" NOT "could some
hyperparameter search on noise ever produce this R²?".

Design:
  - Hyperparameters are FIXED to the most common ("mode") choice recorded
    across the real analysis's outer folds, read directly from that job's
    {model}_fold_metrics.csv — no RandomizedSearchCV inner search is rerun
    here, which is what makes this ~50-250x cheaper than a full nested-CV
    permutation null and lets you afford many more permutation draws. This
    script does NOT depend on 006_finalize_model.py having been run — it
    only needs 002_train_tree_models.py's own output, so it can run
    automatically for every model right after training, before you've
    decided which one to finalize/deploy.
  - HVGs are still reselected fresh per outer fold, per permutation, from that
    fold's training data only (helpers.run_fixed_cv / select_hvgs) — this is
    NOT part of what gets fixed, it's cheap and remains part of the procedure
    being tested. --hvg_select_method must match whatever the real
    002_train_tree_models.py run used (see helpers.select_hvgs_confound_aware),
    or this isn't testing the same procedure. Technical covariate regression
    (--meta_file / --train_only_covs / --train_test_covs), if enabled for the
    real analysis, is refit fresh per permutation the same way — see
    helpers.regress_out_covariates.
  - y is permuted row-wise as ONE consistent shuffle across all target columns
    (not independently per column), breaking the sample-level X<->y
    correspondence while preserving whatever structure exists across targets.
  - The "observed" statistic is computed with the exact same reduced
    procedure (fixed hyperparameters, same fold structure, fresh per-fold HVG
    reselection) on the UNPERMUTED data — NOT read from 002_train_tree_models.py's
    original summary metrics, since those came from the full nested-search
    procedure and aren't a valid like-for-like comparison against the null
    draws. The same KFold split (same seed) is used for the observed run and
    every permutation run; only y's row order varies.

Runs automatically in the Nextflow DAG (workflow/main.nf's PERMUTATION_NULL
process) for the elasticnet/lasso family (elasticnet, lasso, elasticnet_indep,
lasso_indep) and xgboost_indep, right after TRAIN_TREE_MODELS, so results are
available before EVALUATE_MODELS/AGGREGATE_RESULTS.

xgboost_indep is a hybrid case, handled via run_fixed_cv's `per_target_fit`
flag: its hyperparameter *search* is joint in the real pipeline (one shared
setting picked via the aggregate variance-weighted score — see
run_nested_cv_tree — so its fold_metrics.csv has no per-target archetype
breakdown, and derive_best_params returns a single shared dict, same as
elasticnet/lasso), but its *fitting* is always independent per target
regardless, because the real pipeline wraps it in MultiOutputRegressor (plain
XGBRegressor doesn't support multi-output y directly without XGBoost>=2.0's
multi_strategy — see xgboost_joint below). _PER_TARGET_FIT_MODELS below forces
the correct per-target fitting even though best_params is a shared dict, not
a list.

Not run for rf (RandomForestRegressor is natively, genuinely joint — no
independent-fitting edge case to work around, but also no clean way to force
per-target fitting without changing what "rf" actually means), xgboost_joint
(natively joint the same way rf is, but also needs the separate XGBoost>=2.0
environment set via params.xgboost2_env — wiring that through this script's
container_ml-based process is out of scope here), or MIL (needs a different
persistence/fitting approach entirely; out of scope here).

Example:
    python scripts/007_permutation_null.py \\
        --cluster B \\
        --pseudobulk results/prepared_data/B.pseudobulk.npz \\
        --targets    results/prepared_data/B.targets.csv \\
        --model      elasticnet_indep \\
        --fold_metrics results/tree_models/B/elasticnet_indep/full/B_elasticnet_indep_full_tree/elasticnet_indep_fold_metrics.csv \\
        --elasticnet_params scripts/param_grids/elasticnet_params.json \\
        --hvg_pct 0.20 --hvg_max 3000 \\
        --n_permutations 100 \\
        --outdir results/permutation_null
"""
import argparse
import ast
import collections
import os
import sys

import numpy as np
import pandas as pd

sys.path.append("scripts")
from helpers import (
    apply_label_mode,
    load_meta_file,
    load_pseudobulk,
    run_fixed_cv,
    split_covariate_names,
    validate_covariate_lists,
)

_INDEP_MODELS = ("elasticnet_indep", "lasso_indep")
# xgboost_indep's hyperparameter SEARCH is joint (one shared setting, so its
# fold_metrics.csv has no per-target archetype breakdown — same parsing as
# elasticnet/lasso in derive_best_params), but its FITTING is always
# independent per target regardless (see run_fixed_cv's per_target_fit
# docstring) — keep this separate from _INDEP_MODELS, which governs
# fold_metrics parsing, not fitting. xgboost_joint is NOT included: it's
# natively joint in both search and fitting (like rf), and isn't run through
# this script at all — see the module docstring.
_PER_TARGET_FIT_MODELS = _INDEP_MODELS + ("xgboost_indep",)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cluster",             required=True)
    p.add_argument("--pseudobulk",          required=True)
    p.add_argument("--targets",             required=True)
    p.add_argument("--model",               required=True,
                   help="Model name matching the fold_metrics file, e.g. elasticnet_indep")
    p.add_argument("--fold_metrics",        required=True,
                   help="{model}_fold_metrics.csv from the real 002_train_tree_models.py run "
                        "for this (cluster, model, label_mode) — used to derive fixed "
                        "hyperparameters (the mode across outer folds), not to search anew")
    p.add_argument("--xgb_params",          default=None)
    p.add_argument("--rf_params",           default=None)
    p.add_argument("--elasticnet_params",   default=None)
    p.add_argument("--lasso_params",        default=None)
    p.add_argument("--hvg_pct",             type=float, default=0.20,
                   help="Per-fold HVG selection, same setting used for the real analysis")
    p.add_argument("--hvg_max",             type=int,   default=2000)
    p.add_argument("--n_permutations",      type=int,   default=100)
    p.add_argument("--n_folds",             type=int,   default=5,
                   help="Plain K-fold CV per permutation (no inner search)")
    p.add_argument("--label_mode",          default="full",
                   choices=["full", "argmax", "argmax_filtered"])
    p.add_argument("--min_score",           type=float, default=0.5)
    p.add_argument("--seed",                type=int,   default=42,
                   help="Base seed: shared KFold split for observed + every "
                        "permutation; permutation i's row-shuffle uses seed+1+i")
    p.add_argument("--meta_file",           default=None,
                   help="Per-sample metadata CSV — must match whatever the real "
                        "002_train_tree_models.py run used, if it used covariate regression.")
    p.add_argument("--train_only_covs",     nargs="*", default=[])
    p.add_argument("--train_test_covs",     nargs="*", default=[])
    p.add_argument("--hvg_select_method",   default="default",
                   choices=["default", "sq_target_vs_technical"],
                   help="Must match whatever the real 002_train_tree_models.py run used — "
                        "see helpers.select_hvgs_confound_aware and temp/hvg_selection.txt.")
    p.add_argument("--outdir",              required=True)
    return p.parse_args()


def _mode_params(param_strings):
    """Most common hyperparameter dict among a list of str(dict)-repr strings."""
    parsed = [tuple(sorted(ast.literal_eval(s).items())) for s in param_strings]
    best, _ = collections.Counter(parsed).most_common(1)[0]
    return dict(best)


def derive_best_params(fold_metrics_path, model_name, arch_cols):
    """Fixed hyperparameters for run_fixed_cv, derived from the real analysis's
    own {model}_fold_metrics.csv (the mode of best_params across outer folds)
    — no new search, just reading what the real nested CV already chose.

    Returns a single dict (joint models) or a list of dicts ordered to match
    arch_cols (independent models) — the same contract run_fixed_cv expects.
    """
    df = pd.read_csv(fold_metrics_path)
    if model_name in _INDEP_MODELS:
        if "archetype" not in df.columns:
            raise ValueError(
                f"Expected an 'archetype' column in {fold_metrics_path} for "
                f"independent model '{model_name}', but none was found."
            )
        # Group by archetype and take each target's own mode independently —
        # do not rely on row order or group-iteration order matching arch_cols.
        by_arch = {col: _mode_params(sub["best_params"].tolist())
                   for col, sub in df.groupby("archetype")}
        missing = [c for c in arch_cols if c not in by_arch]
        if missing:
            raise ValueError(f"fold_metrics is missing archetype(s) {missing} "
                              f"found in targets: {fold_metrics_path}")
        return [by_arch[col] for col in arch_cols]
    else:
        if "archetype" in df.columns:
            raise ValueError(
                f"Did not expect an 'archetype' column in {fold_metrics_path} "
                f"for joint model '{model_name}' — got one anyway; wrong file?"
            )
        return _mode_params(df["best_params"].tolist())


def _param_path_for(model_name, args):
    return {
        "xgboost_indep":    args.xgb_params,
        "rf":               args.rf_params,
        "elasticnet":       args.elasticnet_params,
        "lasso":            args.lasso_params,
        "elasticnet_indep": args.elasticnet_params,
        "lasso_indep":      args.lasso_params,
    }[model_name]


def _per_pc_r2(y, oof_preds):
    """Per-column R², masking any NaN rows (mirrors helpers.per_archetype_metrics
    but returns a plain array rather than a list of report-ready dicts)."""
    from sklearn.metrics import r2_score
    n_targets = y.shape[1]
    out = np.full(n_targets, np.nan)
    for j in range(n_targets):
        mask = ~np.isnan(oof_preds[:, j])
        out[j] = r2_score(y[mask, j], oof_preds[mask, j])
    return out


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
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading pseudo-bulk from {args.pseudobulk}")
    X, samples, feature_names, n_gene_features = load_pseudobulk(args.pseudobulk)

    print(f"Loading targets from {args.targets}")
    df_tgt = pd.read_csv(args.targets, index_col=0)
    keep_idx = [i for i, s in enumerate(samples) if s in df_tgt.index]
    X = X[keep_idx]
    samples = [samples[i] for i in keep_idx]
    y = df_tgt.loc[samples].values.astype(float)
    arch_cols = df_tgt.columns.tolist()

    y, keep_mask = apply_label_mode(y, args.label_mode, args.min_score)
    if not keep_mask.all():
        X = X[keep_mask]
        y = y[keep_mask]
        samples = [s for s, k in zip(samples, keep_mask) if k]

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

    print(f"X={X.shape}  y={y.shape}  label_mode={args.label_mode}")

    model_name = args.model
    if model_name == "mil":
        raise ValueError("007_permutation_null.py does not support MIL.")

    print(f"Deriving fixed hyperparameters from {args.fold_metrics} "
          f"(mode across the real analysis's outer folds)")
    best_params = derive_best_params(args.fold_metrics, model_name, arch_cols)
    print(f"  model={model_name}  cluster={args.cluster}  label_mode={args.label_mode}")
    print(f"  best_params={best_params}")

    param_grid_path = _param_path_for(model_name, args)
    if param_grid_path is None:
        raise ValueError(f"model '{model_name}' requires the matching --*_params grid path "
                          "(contents are unused, but build_estimator needs a valid file)")

    n_samples = X.shape[0]

    print(f"\nRunning observed (unpermuted) fixed-CV fit ({args.n_folds} folds, "
          f"hyperparameters fixed from fold_metrics mode)...")
    oof_observed = run_fixed_cv(
        X, y, model_name, best_params, param_grid_path, args.n_folds, args.seed,
        n_gene_features=n_gene_features, hvg_pct=args.hvg_pct, hvg_max=args.hvg_max,
        per_target_fit=(model_name in _PER_TARGET_FIT_MODELS),
        samples=samples, covariate_df=df_meta,
        train_only_covs=args.train_only_covs, train_test_covs=args.train_test_covs,
        hvg_select_method=args.hvg_select_method,
    )
    observed_r2 = _per_pc_r2(y, oof_observed)
    for col, r2 in zip(arch_cols, observed_r2):
        print(f"  observed R² [{col}] = {r2:.4f}")

    print(f"\nRunning {args.n_permutations} permutations...")
    null_r2 = np.full((args.n_permutations, len(arch_cols)), np.nan)
    raw_rows = [{"permutation_id": -1, "archetype": col, "r2": r2}
                for col, r2 in zip(arch_cols, observed_r2)]

    for i in range(args.n_permutations):
        rng      = np.random.default_rng(args.seed + 1 + i)
        perm_idx = rng.permutation(n_samples)
        y_perm   = y[perm_idx]

        oof_perm = run_fixed_cv(
            X, y_perm, model_name, best_params, param_grid_path, args.n_folds, args.seed,
            n_gene_features=n_gene_features, hvg_pct=args.hvg_pct, hvg_max=args.hvg_max,
            per_target_fit=(model_name in _PER_TARGET_FIT_MODELS),
            samples=samples, covariate_df=df_meta,
            train_only_covs=args.train_only_covs, train_test_covs=args.train_test_covs,
            hvg_select_method=args.hvg_select_method,
        )
        r2_i = _per_pc_r2(y_perm, oof_perm)
        null_r2[i] = r2_i
        raw_rows.extend({"permutation_id": i, "archetype": col, "r2": r2}
                        for col, r2 in zip(arch_cols, r2_i))
        if (i + 1) % max(1, args.n_permutations // 10) == 0:
            print(f"  {i + 1}/{args.n_permutations} permutations done")

    summary_rows = []
    for j, col in enumerate(arch_cols):
        obs      = observed_r2[j]
        null_col = null_r2[:, j]
        p_value  = (1 + (null_col >= obs).sum()) / (args.n_permutations + 1)
        summary_rows.append({
            "cluster": args.cluster, "model": model_name, "label_mode": args.label_mode,
            "archetype": col, "observed_r2": obs,
            "null_mean": float(np.mean(null_col)), "null_std": float(np.std(null_col)),
            "null_p95": float(np.percentile(null_col, 95)),
            "n_permutations": args.n_permutations, "p_value": p_value,
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(
        os.path.join(args.outdir, f"{args.cluster}_{model_name}_{args.label_mode}_permutation_summary.csv"),
        index=False)
    pd.DataFrame(raw_rows).to_csv(
        os.path.join(args.outdir, f"{args.cluster}_{model_name}_{args.label_mode}_permutation_null_raw.csv"),
        index=False)

    print("\n=== Permutation test summary ===")
    print(df_summary.to_string(index=False))
    print(f"\nDone. Results in {args.outdir}")


if __name__ == "__main__":
    main()
