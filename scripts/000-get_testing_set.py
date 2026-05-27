import os
import scanpy as sc
import numpy as np

ORIG_INPUT = "/lustre/scratch125/humgen/projects_v2/ibdresponse/analysis/bradley_analysis/IBDR_prep/processed_data/anndata_ready-batch1_8-meta_format_2025-09-22-IBDverse_conf_gt0pt5-immune-baseline-nobadsamps.h5ad"
temp_input_dir = "temp"

SEED = 42

os.makedirs(temp_input_dir, exist_ok=True)

adata = sc.read_h5ad(ORIG_INPUT)

rng = np.random.default_rng(SEED)
all_samples = adata.obs["pool_participant"].unique()
selected_samples = rng.choice(all_samples, size=40, replace=False)
adata = adata[adata.obs["pool_participant"].isin(selected_samples)].copy()

adata.write_h5ad(f"{temp_input_dir}/testing_set.h5ad")
