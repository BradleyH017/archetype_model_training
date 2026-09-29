# module load HGI/softpack/users/sb84/bayesian_inference/1
# farm-gpu0203
import sys
import os
sys.path.insert(0, "multimil_packages_1.0.0")
import torch
torch.cuda.is_available()
import multimil as mtm
import numpy as np
import scanpy as sc
import pandas as pd
import scvi
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

# options
h5ad = "results/split_by_cluster/B.cluster_input.h5ad"
sample_key = "pool_participant"
# Continuous archetype CLR-PCA scores -- regression targets (MSE loss head),
# not classification (cross-entropy) -- see MILClassifier() below.
regression_keys = ["B1_clr_pca", "B2_clr_pca", "B3_clr_pca"]

# Load
adata = sc.read_h5ad(h5ad)
samples = adata.obs[sample_key].values
unique_samples = np.unique(samples)

# TESTING: Susbet to 50 samples
unique_samples = unique_samples[:20]

# Set up
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

for i, (train_index, val_index) in enumerate(kf.split(unique_samples)):
    train_samples = unique_samples[train_index]
    val_samples = unique_samples[val_index]
    adata.obs.loc[adata.obs[sample_key].isin(train_samples), f"split{i}"] = "train"
    adata.obs.loc[adata.obs[sample_key].isin(val_samples), f"split{i}"] = "val"
    adata_train = adata[adata.obs[f"split{i}"] == "train"].copy()
    adata_val = adata[adata.obs[f"split{i}"] == "val"].copy()
    # check that all regression keys in validation are present in training
    for regression_key in regression_keys:
        train_conditions = set(adata_train.obs[regression_key].unique())
        val_conditions = set(adata_val.obs[regression_key].unique())

R2_sum={}
for i in range(n_splits):
    print(f"-- Training on split {i} --")
    # Get split
    query = adata[adata.obs[f"split{i}"] == "val"].copy()
    ref = adata[adata.obs[f"split{i}"] == "train"].copy()
    query.obs["ref"] = "query"
    ref.obs["ref"] = "reference"
    idx = ref.obs[sample_key].sort_values().index
    ref = ref[idx].copy()
    mtm.model.MILClassifier.setup_anndata(
        ref,
        categorical_covariate_keys=[sample_key],
        continuous_covariate_keys=regression_keys,
    )
    mil = mtm.model.MILClassifier(
        ref,
        regression=regression_keys,
        sample_key=sample_key,
        class_loss_coef=0.1,
    )
    # Train
    mil.train(max_epochs=20)
    mil.save(f"temp/multimil_model_{i}", overwrite=True)
    mil.get_model_output()
    # Predict on the query
    new_model = mtm.model.MILClassifier.load_query_data(query, mil)
    new_model.get_model_output()
    cell_attn_i = pd.concat([ref.obs["cell_attn"], query.obs["cell_attn"]])
    cell_attn_i.name = f"cell_attn_{i}"
    adata.obs = adata.obs.join(cell_attn_i)
    R2_sum_per_fold = {}
    for regression_key in regression_keys:
        r2 = r2_score(query.obs[regression_key], query.obs[f"predicted_{regression_key}"])
        R2_sum_per_fold[regression_key] = r2
    R2_sum[f"fold{i}"] = R2_sum_per_fold
    del R2_sum_per_fold, r2, query, ref
    
# Parallelise the above. Either it will run back to back or in parallel
# Then when these are all complete, run the below: 

# Iteratively load all the models
cell_atn = {}
for i in range(n_splits):
    print(f"-- Loading model for split {i} --")
    query = adata[adata.obs[f"split{i}"] == "val"].copy()
    ref = adata[adata.obs[f"split{i}"] == "train"].copy()
    query.obs["ref"] = "query"
    ref.obs["ref"] = "reference"
    idx = ref.obs[sample_key].sort_values().index
    ref = ref[idx].copy()
    mil = mtm.model.MILClassifier.load(f"temp/multimil_model_{i}", adata = ref)
    mil.get_model_output()
    new_model = mtm.model.MILClassifier.load_query_data(query, mil)
    new_model.get_model_output()
    cell_attn_i = pd.concat([ref.obs["cell_attn"], query.obs["cell_attn"]])
    cell_attn_i.name = f"cell_attn_{i}"
    cell_atn[f"cell_attn_{i}"] = cell_attn_i

# Get the mean cell attention scores
adata = adata[adata.obs[sample_key].isin(unique_samples)]
to_add = pd.concat(cell_atn.values(), axis=1)
to_add['cell_attn'] = np.mean(to_add.values, axis=1)
adata.obs = adata.obs.drop(columns=[col for col in adata.obs.columns if col.startswith("cell_attn")])
adata.obs = adata.obs.join(to_add)

# Get the sample representations (weighted sum of cell representations)
sample_reps = mtm.utils.get_sample_representations(adata, sample_key=sample_key, covs_to_keep=regression_keys)
sample_reps