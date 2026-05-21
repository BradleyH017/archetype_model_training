#!/usr/bin/env python3
"""
001_prepare_data.py
Compute per-cluster pseudo-bulk features and MIL bags from AnnData,
then extract per-cluster archetype score targets.

Outputs (one set per cluster, e.g. cluster "A"):
  A.pseudobulk.npz  — samples × HVGs sparse matrix + gene/sample names
  A.targets.csv     — samples × archetype_weight columns
  A.mil_bags.h5     — HDF5: one group per sample, dataset = cells × HVGs
"""
import argparse
import os
import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.append("scripts")
from helpers import (
    filter_lowly_expressed_genes,
    select_hvgs,
    pseudobulk_aggregate,
    compute_abundance,
    get_archetype_weight_cols,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--anndata",           required=True)
    p.add_argument("--archetype_scores",  required=True)
    p.add_argument("--cluster_col",       default="mrvi_merged_cluster")
    p.add_argument("--sample_col",        default="donor_id")
    p.add_argument("--celltype_col",      default="cell_type")
    p.add_argument("--rawcelltype_col",   default="cell_type")
    p.add_argument("--hvg_pct",           type=float, default=0.20)
    p.add_argument("--hvg_max",           type=int,   default=2000)
    p.add_argument("--pseudobulk_agg",    choices=["mean", "sum"], default="mean")
    p.add_argument("--include_abundance",    default="true")
    p.add_argument("--abundance_type",       default="within_cluster")
    p.add_argument("--celltype_cluster_map", default=None,
                   help="TSV (celltype, clustered_celltypes). If provided, cells are "
                        "additionally filtered to those whose celltype maps to each cluster.")
    p.add_argument("--min_sample_pct",       type=float, default=None,
                   help="Keep only genes expressed (> 0) in at least this fraction of pseudo-bulk "
                        "samples before HVG selection, e.g. 0.2 keeps genes expressed in >=20%% of samples.")

    # ── Interactive testing ───────────────────────────────────────────────────
    # Uncomment the block below and swap `return p.parse_args()` → `return args`
    # to run interactively without passing command-line flags.
    args = argparse.Namespace(
         anndata              = "/lustre/scratch125/humgen/projects_v2/ibdresponse/analysis/bradley_analysis/IBDR_prep/processed_data/anndata_ready-batch1_8-meta_format_2025-09-22-IBDverse_conf_gt0pt5-immune-baseline-nobadsamps.h5ad",
         archetype_scores     = "/lustre/scratch125/humgen/projects_v2/ibdresponse/analysis/bradley_analysis/archetypal_analysis_from_mrvi/results/tables/sample_archetype_scores.txt.gz",
         celltype_cluster_map = "/lustre/scratch125/humgen/projects_v2/ibdresponse/analysis/bradley_analysis/mrVI/results/tables/celltype_to_cluster_map-from_final_celltypes.txt.gz",
         cluster_col          = "clustered_celltypes",
         sample_col           = "pool_participant",
         rawcelltype_col      = "Celltypist:IBDverse_eqtl:predicted_labels",
         hvg_pct              = 0.20,
         hvg_max              = 2000,
         pseudobulk_agg       = "mean",
         include_abundance    = "true",
         abundance_type       = "global",
         min_sample_pct       = 0.2
    )
    # ─────────────────────────────────────────────────────────────────────────

    return args


def main():
    args = parse_args()
    include_abundance = args.include_abundance.lower() == "true"

    print("Loading archetype scores...")
    df_scores = pd.read_csv(args.archetype_scores, sep='\t', index_col='sample')

    # Build {cluster → set(celltypes)} lookup from the map file, if provided
    cluster_celltypes = None
    if args.celltype_cluster_map is not None:
        df_map = pd.read_csv(args.celltype_cluster_map, sep='\t')
        cluster_celltypes = (
            df_map.groupby(args.cluster_col)["celltype"]
                  .apply(set)
                  .to_dict()
        )
        print(f"Loaded celltype→cluster map: {sum(len(v) for v in cluster_celltypes.values())} "
              f"celltypes across {len(cluster_celltypes)} clusters")

    print("Loading AnnData...")
    adata = ad.read_h5ad(args.anndata)
    # Use log counts
    adata.X = adata.layers['log1p_cp10k'].copy()
    # Merge
    adata.obs = adata.obs.reset_index().merge(df_map, how='left', left_on=args.rawcelltype_col, right_on='celltype').set_index(adata.obs.index)
    clusters = df_map['clustered_celltypes'].unique()
    
    for cluster in clusters:
        print(f"\n=== Cluster {cluster} ===")
        cl_mask = adata.obs[args.cluster_col] == cluster
        adata_cl = adata[cl_mask].copy()

        # ── Pseudo-bulk ───────────────────────────────────────────────────────
        pb_matrix, samples = pseudobulk_aggregate(
            adata_cl, args.sample_col, agg=args.pseudobulk_agg)
        print(f"  Pseudo-bulk ({args.pseudobulk_agg}): {pb_matrix.shape}")

        # ── Prevalence filter (applied before HVG selection) ──────────────────
        if args.min_sample_pct is not None:
            gene_keep_mask = filter_lowly_expressed_genes(pb_matrix, args.min_sample_pct)
            n_removed      = (~gene_keep_mask).sum()
            pb_matrix      = pb_matrix[:, gene_keep_mask]
            orig_gene_idx  = np.where(gene_keep_mask)[0]
            print(f"  Prevalence filter (>={args.min_sample_pct * 100:.0f}% of samples): "
                  f"{n_removed} genes removed, {pb_matrix.shape[1]} remaining")
        else:
            orig_gene_idx = np.arange(pb_matrix.shape[1])

        # hvg_idx is local to pb_matrix; orig_hvg_idx maps back to adata.X columns
        hvg_idx = select_hvgs(pb_matrix, args.hvg_pct, args.hvg_max)
        orig_hvg_idx = orig_gene_idx[hvg_idx]
        pb_hvg       = pb_matrix[:, hvg_idx]
        hvg_names    = adata.var_names[orig_hvg_idx].tolist()
        print(f"  HVGs: {len(hvg_names)}")

        if include_abundance:
            abund, abund_names = compute_abundance(
                adata, adata_cl, args.sample_col, args.rawcelltype_col,
                args.abundance_type, samples)
            cl_celltypes = set(adata_cl.obs[args.rawcelltype_col].unique())
            keep = [i for i, n in enumerate(abund_names) if n.split("_", 2)[-1] in cl_celltypes]
            abund, abund_names = abund[:, keep], [abund_names[i] for i in keep]

            pb_final      = np.hstack([pb_hvg, abund])
            feature_names = hvg_names + abund_names
        else:
            pb_final      = pb_hvg
            feature_names = hvg_names
            abund         = None

        # Save pseudo-bulk NPZ
        csr = sp.csr_matrix(pb_final)
        np.savez(f"{cluster}.pseudobulk.npz",
                 data=csr.data, indices=csr.indices,
                 indptr=csr.indptr, shape=np.array(csr.shape),
                 samples=np.array(samples, dtype=object),
                 features=np.array(feature_names, dtype=object))
        print(f"  Saved {cluster}.pseudobulk.npz")

        # ── Targets ───────────────────────────────────────────────────────────
        arch_cols = get_archetype_weight_cols(df_scores, cluster)
        if not arch_cols:
            print(f"  WARNING: no archetype columns found for prefix '{cluster}' — skipping")
            continue
        print(f"  Archetype cols: {arch_cols}")

        df_tgt = (df_scores
                  .loc[df_scores.index.intersection(samples), arch_cols]
                  .reindex(samples))
        df_tgt.index.name = 'sample'
        df_tgt.to_csv(f"{cluster}.targets.csv")
        print(f"  Saved {cluster}.targets.csv  shape={df_tgt.shape}")

        # ── MIL bags ──────────────────────────────────────────────────────────
        with h5py.File(f"{cluster}.mil_bags.h5", "w") as hf:
            hf.attrs.create('feature_names',
                            data=np.array(feature_names, dtype=h5py.special_dtype(vlen=str)))
            hf.attrs.create('samples',
                            data=np.array(samples, dtype=h5py.special_dtype(vlen=str)))
            for i, s in enumerate(samples):
                mask_s  = (adata_cl.obs[args.sample_col] == s).values
                X_s     = adata_cl.X[mask_s, :]
                X_s_hvg = (np.asarray(X_s[:, orig_hvg_idx].todense())
                            if sp.issparse(X_s)
                            else np.asarray(X_s[:, orig_hvg_idx])).astype(np.float32)
                if include_abundance:
                    abund_row = np.tile(abund[i:i+1, :], (X_s_hvg.shape[0], 1))
                    X_s_full  = np.hstack([X_s_hvg, abund_row])
                else:
                    X_s_full  = X_s_hvg
                safe_key = s.replace("/", "_")
                hf.create_dataset(safe_key, data=X_s_full, compression='gzip')
                
        print(f"  Saved {cluster}.mil_bags.h5")

    print("\nDone.")


if __name__ == "__main__":
    main()
