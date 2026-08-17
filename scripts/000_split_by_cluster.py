#!/usr/bin/env python3
"""
000_split_by_cluster.py
Load the full AnnData once, derive each cell's cluster via the celltype→cluster
map (adata.obs contains celltype_col but not cluster_col), and write out one
lightweight per-cluster AnnData plus a slim full-dataset (sample, celltype)
table — so that 001_prepare_data.py can run as one parallel job per cluster
instead of a single sequential loop over all clusters.

Outputs:
  {cluster}.cluster_input.h5ad   — per-cluster AnnData subset (one per cluster)
  full_obs_celltypes.tsv.gz      — sample_col, celltype_col across ALL cells
                                    (only read downstream when abundance_type
                                    is 'global' or 'both')
"""
import argparse

import anndata as ad
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--anndata",             required=True)
    p.add_argument("--celltype_cluster_map", required=True,
                   help="TSV (celltype, clustered_celltypes) used to derive cluster_col, "
                        "since adata.obs only contains celltype_col.")
    p.add_argument("--cluster_col",         default="clustered_celltypes")
    p.add_argument("--sample_col",          default="donor_id")
    p.add_argument("--celltype_col",        default="cell_type")
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading celltype→cluster map...")
    df_map = pd.read_csv(args.celltype_cluster_map, sep='\t')

    print("Loading AnnData...")
    adata = ad.read_h5ad(args.anndata)
    # Use log counts
    adata.X = adata.layers['log1p_cp10k'].copy()
    adata.layers.clear()
    # Merge in cluster_col via the celltype→cluster map (adata.obs has celltype_col, not cluster_col)
    adata.obs = (adata.obs.reset_index()
                 .merge(df_map, how='left', left_on=args.celltype_col, right_on='celltype')
                 .set_index(adata.obs.index))

    print("Writing full-dataset (sample, celltype) table for global abundance features...")
    (adata.obs[[args.sample_col, args.celltype_col]]
     .to_csv("full_obs_celltypes.tsv.gz", sep='\t', index=False))

    clusters = df_map['clustered_celltypes'].unique()
    print(f"Splitting into {len(clusters)} clusters...")
    for cluster in clusters:
        cl_mask  = adata.obs[args.cluster_col] == cluster
        adata_cl = adata[cl_mask].copy()
        outpath  = f"{cluster}.cluster_input.h5ad"
        adata_cl.write_h5ad(outpath)
        print(f"  {cluster}: {adata_cl.shape[0]} cells -> {outpath}")

    print("Done.")


if __name__ == "__main__":
    main()
