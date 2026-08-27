#!/usr/bin/env python3
"""
003_train_mil.py
Attention-based Multiple Instance Learning (MIL) with nested cross-validation.

Architecture (Ilse et al., 2018):
  - Per-cell encoder (2-layer MLP)
  - Gated attention pooling → single bag embedding
  - Bag-level predictor with softmax (simplex output)

HVGs are reselected independently for each outer CV fold, from that fold's
training samples only (--hvg_pct/--hvg_max here, not in 001_prepare_data.py):
each training bag's cells are pseudobulked (--pseudobulk_agg), then ranked by
variance across those per-sample profiles — the same statistic used for the
tree/linear models in 002_train_tree_models.py — rather than from a fixed
pre-selected panel. This avoids ranking genes using cells from samples that
end up in the held-out test fold.

Hyperparameter comparison and per-fold R² both use multioutput='variance_weighted'
scoring (not 'uniform_average'): each target column's contribution is weighted
by its own variance rather than averaged uniformly. For CLR-PCA targets this
is equivalent to a proper multivariate R² based on total squared Euclidean
(Aitchison) distance, since PCA preserves distance.

Outputs (in --outdir):
  mil_fold_metrics.csv       — per-fold R² / RMSE + best hyperparams
  mil_oof_predictions.csv    — out-of-fold predictions (samples × archetypes)
  mil_summary_metrics.csv    — per-archetype OOF metrics
  mil_overall_metrics.csv    — one aggregate row: multivariate (variance-weighted)
                                R² and aggregate RMSE across ALL targets at once

Label modes (--label_mode):
  full            — all continuous archetype weights (default, original behaviour)
  argmax          — full cohort, targets converted to one-hot at dominant archetype
  argmax_filtered — cohort filtered to max archetype weight > --min_score, then one-hot
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import (
    HAS_TORCH,
    apply_label_mode,
    load_meta_file,
    load_mil_bags,
    nested_cv_mil,
    overall_metrics,
    per_archetype_metrics,
    split_covariate_names,
    validate_covariate_lists,
)

if not HAS_TORCH:
    raise RuntimeError("PyTorch is required for MIL training but could not be imported.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cluster",           required=True)
    p.add_argument("--mil_bags",          required=True)
    p.add_argument("--targets",           required=True)
    p.add_argument("--mil_params",        required=True)
    p.add_argument("--hvg_pct",           type=float, default=0.20,
                   help="Per-outer-fold HVG selection: fraction of candidate genes to keep")
    p.add_argument("--hvg_max",           type=int,   default=2000,
                   help="Per-outer-fold HVG selection: hard cap on number of HVGs")
    p.add_argument("--pseudobulk_agg",    choices=["mean", "sum"], default="mean",
                   help="How to aggregate a fold's training bags to per-sample profiles "
                        "for HVG ranking (should match 001_prepare_data.py's setting)")
    p.add_argument("--n_outer_folds",     type=int, default=5)
    p.add_argument("--n_inner_folds",     type=int, default=5)
    p.add_argument("--n_iter",            type=int, default=20)
    p.add_argument("--max_cells_per_bag", type=int, default=2000)
    p.add_argument("--seed",              type=int, default=42)
    p.add_argument("--outdir",            required=True)
    p.add_argument("--label_mode",        default="full",
                   choices=["full", "argmax", "argmax_filtered"],
                   help="Target encoding: 'full'=continuous weights, 'argmax'=one-hot at "
                        "dominant archetype (full cohort), 'argmax_filtered'=one-hot + "
                        "restrict to samples where max weight > --min_score")
    p.add_argument("--min_score",         type=float, default=0.5,
                   help="Minimum dominant archetype weight for argmax_filtered mode")
    p.add_argument("--meta_file",         default=None,
                   help="Per-sample metadata CSV (first column = sample ID). Required if "
                        "either --train_only_covs or --train_test_covs is non-empty.")
    p.add_argument("--train_only_covs",   nargs="*", default=[],
                   help="Covariate column(s) regressed out of training CELLS only (not "
                        "applied to held-out cells) — see helpers.regress_out_covariates.")
    p.add_argument("--train_test_covs",   nargs="*", default=[],
                   help="Covariate column(s) regressed out of both training and held-out "
                        "CELLS (fit on train, applied to test) — see helpers.regress_out_covariates.")
    p.add_argument("--hvg_select_method", default="default",
                   choices=["default", "sq_target_vs_technical"],
                   help="'default': rank HVG candidates (on pseudobulked training bags) by "
                        "raw variance. 'sq_target_vs_technical': rank by target_r2^2/covariate_r2 "
                        "instead, scored against --train_only_covs if non-empty else "
                        "--train_test_covs (requires at least one non-empty) — see "
                        "helpers.select_hvgs_confound_aware.")
    return p.parse_args()


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading bags from {args.mil_bags}")
    bags, samples, n_gene_features = load_mil_bags(args.mil_bags)
    print(f"  {len(bags)} bags  |  cells/bag: "
          f"min={min(b.shape[0] for b in bags)} "
          f"max={max(b.shape[0] for b in bags)}")
    print(f"  {n_gene_features} candidate genes + {bags[0].shape[1] - n_gene_features} "
          f"abundance features; HVGs reselected per outer fold "
          f"(hvg_pct={args.hvg_pct}, hvg_max={args.hvg_max})")

    print(f"Loading targets from {args.targets}")
    df_tgt  = pd.read_csv(args.targets, index_col=0)
    keep    = [i for i, s in enumerate(samples) if s in df_tgt.index]
    bags    = [bags[i] for i in keep]
    samples = [samples[i] for i in keep]
    y       = df_tgt.loc[samples].values.astype(float)
    arch_cols = df_tgt.columns.tolist()

    y, keep_mask = apply_label_mode(y, args.label_mode, args.min_score)
    if not keep_mask.all():
        n_kept = keep_mask.sum()
        print(f"  label_mode='{args.label_mode}': {(~keep_mask).sum()} samples dropped "
              f"(max weight ≤ {args.min_score}), {n_kept} remain")
        bags    = [bags[i]    for i, k in enumerate(keep_mask) if k]
        samples = [samples[i] for i, k in enumerate(keep_mask) if k]
        y       = y[keep_mask]

    df_meta = None
    if args.train_only_covs or args.train_test_covs:
        if not args.meta_file:
            raise ValueError("--train_only_covs/--train_test_covs given but --meta_file is not set")
        print(f"Loading covariate metadata from {args.meta_file}")
        df_meta = load_meta_file(args.meta_file)
        validate_covariate_lists(df_meta, args.train_only_covs, args.train_test_covs)
        keep = [i for i, s in enumerate(samples) if s in df_meta.index]
        if len(keep) < len(samples):
            print(f"  Dropping {len(samples) - len(keep)} samples not in meta_file")
        bags    = [bags[i]    for i in keep]
        samples = [samples[i] for i in keep]
        y       = y[keep]
        print(f"  train_only_covs={args.train_only_covs}  train_test_covs={args.train_test_covs}")

    print(f"  {len(bags)} bags × {y.shape[1]} targets  label_mode={args.label_mode}")

    with open(args.mil_params) as f:
        hp_grid = json.load(f)

    torch.manual_seed(args.seed)
    fold_metrics, oof_preds = nested_cv_mil(
        bags, y, hp_grid,
        args.n_outer_folds, args.n_inner_folds, args.n_iter,
        args.max_cells_per_bag, args.seed, device,
        n_gene_features=n_gene_features, hvg_pct=args.hvg_pct, hvg_max=args.hvg_max,
        pseudobulk_agg=args.pseudobulk_agg,
        samples=samples, covariate_df=df_meta,
        train_only_covs=args.train_only_covs, train_test_covs=args.train_test_covs,
        hvg_select_method=args.hvg_select_method,
    )

    pd.DataFrame(fold_metrics).to_csv(
        f"{args.outdir}/mil_fold_metrics.csv", index=False)

    pd.DataFrame(oof_preds, columns=arch_cols, index=samples).to_csv(
        f"{args.outdir}/mil_oof_predictions.csv")

    pd.DataFrame(
        per_archetype_metrics(y, oof_preds, arch_cols, "mil", args.cluster, args.label_mode)
    ).to_csv(f"{args.outdir}/mil_summary_metrics.csv", index=False)

    pd.DataFrame([
        overall_metrics(y, oof_preds, "mil", args.cluster, args.label_mode)
    ]).to_csv(f"{args.outdir}/mil_overall_metrics.csv", index=False)

    print(f"\nDone. Results in {args.outdir}")


if __name__ == "__main__":
    main()
