#!/usr/bin/env python3
"""
001_prepare_data.py
Compute pseudo-bulk features and MIL bags for ONE cluster (already split out
by 000_split_by_cluster.py), then extract that cluster's archetype score
targets.

Gene features here are only prevalence-filtered, NOT yet narrowed to HVGs —
the actual HVG (top-variance) selection is deferred to the outer CV loop in
002_train_tree_models.py / 003_train_mil.py, computed from each fold's
training samples only. Doing it here (before any train/test split exists)
would rank genes by variance across ALL samples, including future test folds
— a mild but real data leak.

Outputs (for cluster "A"):
  A.pseudobulk.npz  — samples × (prevalence-filtered genes + abundance) sparse
                       matrix, gene/sample names, and n_gene_features (the
                       gene/abundance column boundary)
  A.targets.csv     — samples × archetype_weight columns
  A.mil_bags.h5     — HDF5: one group per sample, dataset = cells × (genes + abundance)
"""
import argparse
import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.append("scripts")
from helpers import (
    filter_lowly_expressed_genes,
    pseudobulk_aggregate,
    compute_abundance,
    get_archetype_weight_cols,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--anndata",           required=True,
                   help="Per-cluster AnnData written by 000_split_by_cluster.py")
    p.add_argument("--cluster",           required=True)
    p.add_argument("--archetype_scores",  required=True)
    p.add_argument("--target_mode",       choices=["raw", "pca"], default="raw",
                   help="'raw' uses simplex weight columns (e.g. B1, B2); "
                        "'pca' uses CLR-PCA transformed columns (e.g. B1_clr_pca, B2_clr_pca).")
    p.add_argument("--sample_col",        default="donor_id")
    p.add_argument("--celltype_col",      default="cell_type")
    p.add_argument("--pseudobulk_agg",    choices=["mean", "sum"], default="mean")
    p.add_argument("--include_abundance",    default="true")
    p.add_argument("--abundance_type",       default="within_cluster")
    p.add_argument("--full_obs_celltypes",   default=None,
                   help="TSV (sample_col, celltype_col) across ALL clusters, from "
                        "000_split_by_cluster.py. Required when --abundance_type is "
                        "'global' or 'both'.")
    p.add_argument("--min_sample_pct",       type=float, default=None,
                   help="Keep only genes expressed (> 0) in at least this fraction of pseudo-bulk "
                        "samples, e.g. 0.2 keeps genes expressed in >=20%% of samples. This is the "
                        "only gene filtering done here; HVG selection happens downstream, per CV fold.")

    # ── Interactive testing ───────────────────────────────────────────────────
    # Uncomment the block below and swap `return p.parse_args()` → `return args`
    # to run interactively without passing command-line flags.
    #args = argparse.Namespace(
    #     anndata              = "A.cluster_input.h5ad",
    #     cluster              = "A",
    #     archetype_scores     = "/lustre/scratch125/humgen/projects_v2/ibdresponse/analysis/bradley_analysis/archetypal_analysis_from_mrvi/results/tables/sample_archetype_scores.txt.gz",
    #     target_mode          = "raw",
    #     sample_col           = "pool_participant",
    #     celltype_col         = "Celltypist:IBDverse_eqtl:predicted_labels",
    #     pseudobulk_agg       = "mean",
    #     include_abundance    = "true",
    #     abundance_type       = "global",
    #     full_obs_celltypes   = "full_obs_celltypes.tsv.gz",
    #     min_sample_pct       = 0.2
    #)
    # ─────────────────────────────────────────────────────────────────────────

    return p.parse_args()


def main():
    args = parse_args()
    include_abundance = args.include_abundance.lower() == "true"
    cluster = args.cluster

    print("Loading archetype scores...")
    df_scores = pd.read_csv(args.archetype_scores, sep='\t', index_col='sample')

    obs_full = None
    if args.abundance_type in ("global", "both"):
        if args.full_obs_celltypes is None:
            raise ValueError("--full_obs_celltypes is required when --abundance_type is 'global' or 'both'")
        obs_full = pd.read_csv(args.full_obs_celltypes, sep='\t')

    print(f"Loading per-cluster AnnData for cluster {cluster}...")
    adata_cl = ad.read_h5ad(args.anndata)

    print(f"\n=== Cluster {cluster} ===")

    # ── Pseudo-bulk ───────────────────────────────────────────────────────
    pb_matrix, samples = pseudobulk_aggregate(
        adata_cl, args.sample_col, agg=args.pseudobulk_agg)
    print(f"  Pseudo-bulk ({args.pseudobulk_agg}): {pb_matrix.shape}")

    # ── Prevalence filter ───────────────────────────────────────────────────
    # The only gene filtering done here. HVG (top-variance) selection is
    # deferred to the outer CV loop downstream (see module docstring).
    if args.min_sample_pct is not None:
        gene_keep_mask = filter_lowly_expressed_genes(pb_matrix, args.min_sample_pct)
        n_removed      = (~gene_keep_mask).sum()
        pb_matrix      = pb_matrix[:, gene_keep_mask]
        orig_gene_idx  = np.where(gene_keep_mask)[0]
        print(f"  Prevalence filter (>={args.min_sample_pct * 100:.0f}% of samples): "
              f"{n_removed} genes removed, {pb_matrix.shape[1]} remaining")
    else:
        orig_gene_idx = np.arange(pb_matrix.shape[1])

    gene_names      = adata_cl.var_names[orig_gene_idx].tolist()
    n_gene_features = len(gene_names)
    print(f"  Candidate genes for downstream fold-specific HVG selection: {n_gene_features}")

    if include_abundance:
        abund, abund_names = compute_abundance(
            obs_full, adata_cl, args.sample_col, args.celltype_col,
            args.abundance_type, samples)
        cl_celltypes = set(adata_cl.obs[args.celltype_col].unique())
        keep = [i for i, n in enumerate(abund_names) if n.split("_", 2)[-1] in cl_celltypes]
        abund, abund_names = abund[:, keep], [abund_names[i] for i in keep]

        pb_final      = np.hstack([pb_matrix, abund])
        feature_names = gene_names + abund_names
    else:
        pb_final      = pb_matrix
        feature_names = gene_names
        abund         = None

    # Save pseudo-bulk NPZ (genes first, then abundance features — n_gene_features
    # marks the boundary so downstream fold-specific HVG selection only reranks
    # the gene block and always keeps the abundance block intact).
    csr = sp.csr_matrix(pb_final)
    np.savez(f"{cluster}.pseudobulk.npz",
             data=csr.data, indices=csr.indices,
             indptr=csr.indptr, shape=np.array(csr.shape),
             samples=np.array(samples, dtype=object),
             features=np.array(feature_names, dtype=object),
             n_gene_features=np.array(n_gene_features))
    print(f"  Saved {cluster}.pseudobulk.npz")

    # ── Targets ───────────────────────────────────────────────────────────
    arch_cols = get_archetype_weight_cols(df_scores, cluster, target_mode=args.target_mode)
    if not arch_cols:
        print(f"  WARNING: no '{args.target_mode}' archetype columns found for prefix '{cluster}' — skipping targets/MIL bags")
        print("\nDone.")
        return
    print(f"  Archetype cols: {arch_cols}")

    df_tgt = (df_scores
              .loc[df_scores.index.intersection(samples), arch_cols]
              .reindex(samples))
    df_tgt.index.name = 'sample'
    df_tgt.to_csv(f"{cluster}.targets.csv")
    print(f"  Saved {cluster}.targets.csv  shape={df_tgt.shape}")

    # ── MIL bags ──────────────────────────────────────────────────────────
    with h5py.File(f"{cluster}.mil_bags.h5", "w") as hf:
        # feature_names now covers ALL prevalence-passing genes (thousands, not
        # just a few thousand HVGs), which is too large for an HDF5 attribute
        # (hard ~64KB object-header cap) — store it as a dataset instead.
        hf.create_dataset('feature_names',
                          data=np.array(feature_names, dtype=h5py.special_dtype(vlen=str)))
        hf.attrs.create('samples',
                        data=np.array(samples, dtype=h5py.special_dtype(vlen=str)))
        hf.attrs.create('n_gene_features', data=n_gene_features)
        for i, s in enumerate(samples):
            mask_s   = (adata_cl.obs[args.sample_col] == s).values
            X_s      = adata_cl.X[mask_s, :]
            X_s_gene = (np.asarray(X_s[:, orig_gene_idx].todense())
                        if sp.issparse(X_s)
                        else np.asarray(X_s[:, orig_gene_idx])).astype(np.float32)
            if include_abundance:
                abund_row = np.tile(abund[i:i+1, :], (X_s_gene.shape[0], 1))
                X_s_full  = np.hstack([X_s_gene, abund_row])
            else:
                X_s_full  = X_s_gene
            safe_key = s.replace("/", "_")
            hf.create_dataset(safe_key, data=X_s_full, compression='gzip')

    print(f"  Saved {cluster}.mil_bags.h5")
    print("\nDone.")


if __name__ == "__main__":
    main()
