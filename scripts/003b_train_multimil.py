#!/usr/bin/env python3
"""
003b_train_multimil.py
Trains ONE outer CV fold of a MultiMIL (https://github.com/theislab/multimil)
model for one cluster — a deliberate departure from 002_train_tree_models.py/
003_train_mil.py, which each loop over every outer fold inside one process.
MultiMIL training is GPU-bound and heavy enough per fold that each fold is
its own Nextflow task instead (see workflow/main.nf's TRAIN_MULTIMIL), fanned
out over cluster x target_mode x fold_idx. 003c_combine_multimil_folds.py
stitches the resulting per-fold OOF predictions back into the same
{model}_summary_metrics.csv / {model}_overall_metrics.csv schema
002/003 already produce, so 004_evaluate.py needs no changes.

Requires Python >=3.12 + the multimil/torch/scvi-tools stack from
installs/multimil_download.sh -- NOT this pipeline's container_ml/
container_torch. See workflow/main.nf's TRAIN_MULTIMIL (params.multimil_env).

Comparability with 002/003, by construction rather than a shared artifact:
  - Reads {cluster}.mil_bags.h5 (same prevalence-filtered candidate gene
    panel + abundance features as everything else), not the raw per-cluster
    AnnData.
  - Reconstructs THIS fold's train/held-out sample split via the same
    KFold(n_splits, shuffle=True, random_state=seed) call, on the same
    (filtered, ordered) sample list 002/003 use -- confirmed by direct code
    inspection to currently produce identical folds, not assumed. See
    temp/ (or ask) for the fold-consistency check this relies on; there is
    no shared fold-assignment file yet, so this only holds as long as
    002/003's own filtering logic doesn't diverge from what's replicated
    here.
  - Reselects HVGs from THIS fold's training bags only, via the same
    helpers.select_hvgs / select_hvgs_confound_aware functions and the same
    --hvg_pct/--hvg_max/--hvg_select_method 002/003 use -- a pure function
    of (training sample set, params), so it reproduces the identical panel
    without needing 002's fold to have run first or persisted anything.
  - Applies the same --train_only_covs/--train_test_covs regression
    (helpers.regress_out_covariates), on a per-CELL matrix with each cell's
    sample ID repeated per its own cell count -- the same trick
    003_train_mil.py's nested_cv_mil uses for MIL bags.
  - Fits a StandardScaler on training cells only, matching 002/003's own
    scaling convention.

target_mode:
  joint -- one MILClassifier, regression=all 3 archetype targets at once
           (shared attention/embedding trunk, multi-head output) -- the MIL
           analogue of xgboost_joint/MultiTaskElasticNet.
  indep -- three separate MILClassifier fits, one target each -- the MIL
           analogue of xgboost_indep/elasticnet_indep. Three full trainings
           per fold, so ~3x the cost of joint.

No inner-fold hyperparameter search (unlike 002/003's nested CV): MultiMIL
training is expensive enough on GPU that a full nested search wasn't judged
worth it for a first pass -- hyperparameters are fixed via CLI flags
(sensible MultiMIL defaults). Revisit if joint/indep results look promising
enough to justify the added cost.

Ground truth for this fold's held-out samples comes directly from
--targets (already loaded here), not from MultiMIL's own bag_true_* outputs
-- simpler, and sidesteps needing to trust get_sample_representations-style
aggregation code for something we already have from disk. Predictions are
read from adata.obs[f"predicted_{target}"] after get_model_output(), grouped
by sample (every cell of a sample gets the identical bag-level prediction
broadcast onto it by MultiMIL's own save_predictions_in_adata, so
groupby(sample_key).first() recovers the one per-sample value without any
assumption about bag ordering).

Outputs (in --outdir):
  {cluster}_multimil_{target_mode}_fold{fold_idx}_oof_predictions.csv
      Held-out-fold predictions for this fold's test samples only (rows =
      this fold's test samples, columns = archetype targets) -- combined
      across all folds by 003c_combine_multimil_folds.py.
  {cluster}_multimil_{target_mode}_fold{fold_idx}_hvg_panel.txt
      The gene panel this fold actually trained on, for provenance.
"""
import argparse
import os
import sys

# Must come before any third-party import below -- everything here (anndata,
# multimil, and even numpy/pandas/sklearn depending on the container) only
# resolves from the multimil install directory, not this container's ambient
# site-packages. A bare relative "multimil_packages_1.0.0" only works when
# run by hand from the repo root (e.g. 003a) -- under Nextflow each task
# runs in its own isolated work/xx/yyyy/ directory, so that path silently
# resolved to nothing there. MULTIMIL_PKG_DIR (set by workflow/main.nf's
# TRAIN_MULTIMIL to ${launchDir}/multimil_packages_1.0.0, an absolute path)
# overrides the relative default for that case; env var rather than a CLI
# flag because this has to happen before argparse ever runs.
sys.path.insert(0, os.environ.get("MULTIMIL_PKG_DIR", "multimil_packages_1.0.0"))

# anndata directly, not scanpy: `import scanpy` eagerly imports essentially
# every scanpy submodule at package-init time, and one of them
# (scanpy.get._kernels) applies numba's @njit at IMPORT time -- numba then
# tries to set up an on-disk compile cache next to that source file, which
# fails hard ("no locator available") when multimil_packages_1.0.0 sits on a
# read-only mount inside the container, confirmed by direct traceback. We
# only ever use AnnData() itself here, which lives in anndata regardless --
# no reason to pay for scanpy's fragile-in-this-container import chain for
# that.
import anndata as ad
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import (
    load_meta_file,
    load_mil_bags,
    regress_out_covariates,
    select_hvgs,
    select_hvgs_confound_aware,
    split_covariate_names,
    validate_covariate_lists,
)

import multimil as mtm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cluster",     required=True)
    p.add_argument("--mil_bags",    required=True)
    p.add_argument("--targets",     required=True)
    p.add_argument("--sample_col",  required=True,
                   help="Per-sample identifier column name (matches params.sample_col)")
    p.add_argument("--fold_idx",      type=int, required=True,
                   help="Which outer fold (0-indexed) to train/evaluate")
    p.add_argument("--n_outer_folds", type=int, default=5)
    p.add_argument("--target_mode",   choices=["joint", "indep"], required=True)
    p.add_argument("--hvg_pct",       type=float, default=0.20)
    p.add_argument("--hvg_max",       type=int,   default=2000)
    p.add_argument("--hvg_select_method", default="default",
                   choices=["default", "sq_target_vs_technical"])
    p.add_argument("--pseudobulk_agg", choices=["mean", "sum"], default="mean",
                   help="How to aggregate this fold's training bags to per-sample "
                        "profiles for HVG ranking (should match 001_prepare_data.py)")
    p.add_argument("--max_cells_per_bag", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_epochs", type=int, default=200,
                   help="Passed to mil.train() -- 200 matches MILClassifier.train()'s "
                        "own default; unlike some other scvi-tools models, it does not "
                        "auto-compute a value from dataset size when given None (confirmed "
                        "by direct TypeError: 'max_epochs // 3' inside its own train(), "
                        "which assumes an int unconditionally)")
    p.add_argument("--meta_file", default=None)
    p.add_argument("--train_only_covs", nargs="*", default=[])
    p.add_argument("--train_test_covs", nargs="*", default=[])
    p.add_argument("--outdir", required=True)
    return p.parse_args()


def _pseudobulk(bags, agg):
    agg_fn = np.mean if agg == "mean" else np.sum
    return np.array([agg_fn(b, axis=0) for b in bags], dtype=np.float32)


def _subsample_bag(bag, max_cells, seed):
    if bag.shape[0] <= max_cells:
        return bag
    rng = np.random.default_rng(seed)
    idx = rng.choice(bag.shape[0], max_cells, replace=False)
    return bag[idx]


def _build_adata(bags, samples, targets_df, arch_cols, sample_col):
    """Stack per-sample cell x feature bags into one AnnData, one row per
    cell, with each cell's obs carrying its sample ID and (for ground-truth
    bookkeeping / MultiMIL's own registration) that sample's archetype
    values broadcast across all its cells. Sorted by sample_col -- MultiMIL's
    own tutorials note this is required for training to work correctly."""
    X_parts, obs_rows = [], []
    for s, bag in zip(samples, bags):
        X_parts.append(bag)
        row = {sample_col: s}
        for col in arch_cols:
            row[col] = targets_df.loc[s, col]
        obs_rows.extend([row] * bag.shape[0])
    X = np.vstack(X_parts).astype(np.float32)
    obs = pd.DataFrame(obs_rows)
    adata = ad.AnnData(X=X, obs=obs)
    idx = adata.obs[sample_col].sort_values().index
    return adata[idx].copy()


def _fit_one_multimil(ref_adata, query_adata, regression_keys, sample_col, seed, max_epochs):
    mtm.model.MILClassifier.setup_anndata(
        ref_adata,
        categorical_covariate_keys=[sample_col],
        continuous_covariate_keys=regression_keys,
    )
    mil = mtm.model.MILClassifier(
        ref_adata,
        regression=regression_keys,
        sample_key=sample_col,
        class_loss_coef=0.1,
    )
    mil.train(max_epochs=max_epochs)
    mil.get_model_output()

    new_model = mtm.model.MILClassifier.load_query_data(query_adata, mil)
    new_model.get_model_output()

    preds = {}
    for key in regression_keys:
        preds[key] = query_adata.obs.groupby(sample_col)[f"predicted_{key}"].first()
    return pd.DataFrame(preds)


def main():
    args = parse_args()
    args.train_only_covs = split_covariate_names(args.train_only_covs)
    args.train_test_covs = split_covariate_names(args.train_test_covs)
    if args.hvg_select_method == "sq_target_vs_technical" and not (args.train_only_covs or args.train_test_covs):
        raise ValueError(
            "--hvg_select_method sq_target_vs_technical requires a non-empty "
            "--train_only_covs or --train_test_covs."
        )
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading bags from {args.mil_bags}")
    bags, samples, n_gene_features = load_mil_bags(args.mil_bags)

    df_tgt = pd.read_csv(args.targets, index_col=0)
    keep = [i for i, s in enumerate(samples) if s in df_tgt.index]
    bags = [bags[i] for i in keep]
    samples = [samples[i] for i in keep]
    arch_cols = df_tgt.columns.tolist()

    df_meta = None
    if args.train_only_covs or args.train_test_covs:
        if not args.meta_file:
            raise ValueError("--train_only_covs/--train_test_covs given but --meta_file is not set")
        df_meta = load_meta_file(args.meta_file)
        validate_covariate_lists(df_meta, args.train_only_covs, args.train_test_covs)
        keep = [i for i, s in enumerate(samples) if s in df_meta.index]
        bags = [bags[i] for i in keep]
        samples = [samples[i] for i in keep]

    # Same KFold construction as 002_train_tree_models.py/003_train_mil.py,
    # on the same (filtered, ordered) sample list -- reproduces identical
    # fold membership, confirmed by direct inspection, not a shared artifact.
    outer_cv = KFold(n_splits=args.n_outer_folds, shuffle=True, random_state=args.seed)
    splits = list(outer_cv.split(samples))
    if args.fold_idx >= len(splits):
        raise ValueError(f"--fold_idx {args.fold_idx} out of range for --n_outer_folds {args.n_outer_folds}")
    train_idx, test_idx = splits[args.fold_idx]
    samples_tr = [samples[i] for i in train_idx]
    samples_te = [samples[i] for i in test_idx]
    bags_tr_full = [bags[i] for i in train_idx]
    bags_te_full = [bags[i] for i in test_idx]
    print(f"Fold {args.fold_idx}/{args.n_outer_folds}: {len(samples_tr)} train, {len(samples_te)} test samples")

    # HVG reselection, from this fold's training bags only.
    pb_tr = _pseudobulk([b[:, :n_gene_features] for b in bags_tr_full], args.pseudobulk_agg)
    if args.hvg_select_method == "sq_target_vs_technical":
        confound_covs = args.train_only_covs if args.train_only_covs else args.train_test_covs
        y_tr = df_tgt.loc[samples_tr].values
        hvg_idx_local = select_hvgs_confound_aware(
            pb_tr, y_tr, df_meta, samples_tr, confound_covs, args.hvg_pct, args.hvg_max)
    else:
        hvg_idx_local = select_hvgs(pb_tr, args.hvg_pct, args.hvg_max)
    total_cols = bags_tr_full[0].shape[1]
    abund_idx = np.arange(n_gene_features, total_cols)
    sel_idx = np.concatenate([hvg_idx_local, abund_idx])
    bags_tr = [b[:, sel_idx] for b in bags_tr_full]
    bags_te = [b[:, sel_idx] for b in bags_te_full]

    with open(os.path.join(args.outdir, f"{args.cluster}_multimil_{args.target_mode}_fold{args.fold_idx}_hvg_panel.txt"), "w") as f:
        f.write("\n".join(str(i) for i in sel_idx))

    # Covariate regression, on a per-CELL matrix (each cell's sample ID
    # repeated per its own cell count) -- same trick nested_cv_mil uses.
    if df_meta is not None and (args.train_only_covs or args.train_test_covs):
        cells_tr = np.vstack(bags_tr)
        cells_te = np.vstack(bags_te) if bags_te else np.zeros((0, cells_tr.shape[1]), dtype=np.float32)
        rep_samples_tr = [s for s, b in zip(samples_tr, bags_tr) for _ in range(b.shape[0])]
        rep_samples_te = [s for s, b in zip(samples_te, bags_te) for _ in range(b.shape[0])]
        cells_tr, cells_te, _ = regress_out_covariates(
            cells_tr, cells_te, df_meta, rep_samples_tr, rep_samples_te,
            args.train_only_covs, args.train_test_covs)
        offset = 0
        for i, b in enumerate(bags_tr):
            bags_tr[i] = cells_tr[offset:offset + b.shape[0]]
            offset += b.shape[0]
        offset = 0
        for i, b in enumerate(bags_te):
            bags_te[i] = cells_te[offset:offset + b.shape[0]]
            offset += b.shape[0]

    # Subsample cells per bag, then scale -- scaler fit on training cells
    # only, matching 002/003's convention.
    bags_tr = [_subsample_bag(b, args.max_cells_per_bag, args.seed + i) for i, b in enumerate(bags_tr)]
    bags_te = [_subsample_bag(b, args.max_cells_per_bag, args.seed + 10_000 + i) for i, b in enumerate(bags_te)]
    scaler = StandardScaler().fit(np.vstack(bags_tr))
    bags_tr = [scaler.transform(b).astype(np.float32) for b in bags_tr]
    bags_te = [scaler.transform(b).astype(np.float32) for b in bags_te]

    ref = _build_adata(bags_tr, samples_tr, df_tgt, arch_cols, args.sample_col)
    query = _build_adata(bags_te, samples_te, df_tgt, arch_cols, args.sample_col)

    if args.target_mode == "joint":
        oof = _fit_one_multimil(ref, query, arch_cols, args.sample_col, args.seed, args.max_epochs)
    else:  # indep: one full fit per target, columns assembled from separate models
        oof = pd.DataFrame(index=pd.Index(sorted(set(samples_te)), name=args.sample_col))
        for key in arch_cols:
            ref_i = ref.copy()
            query_i = query.copy()
            oof_i = _fit_one_multimil(ref_i, query_i, [key], args.sample_col, args.seed, args.max_epochs)
            oof[key] = oof_i[key]

    oof = oof.reindex(samples_te)
    oof.index.name = "sample"
    oof.to_csv(os.path.join(
        args.outdir, f"{args.cluster}_multimil_{args.target_mode}_fold{args.fold_idx}_oof_predictions.csv"))
    print(f"\nDone. Fold {args.fold_idx} results in {args.outdir}")


if __name__ == "__main__":
    main()
