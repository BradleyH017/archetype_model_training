# module load HGI/softpack/users/sb84/bayesian_inference/1
import sys
sys.path.insert(0, "multimil_packages_1.0.0")
import torch
torch.cuda.is_available()
import multimil as mtm
import numpy as np
import scanpy as sc
import pandas as pd
import scvi
import warnings
from sklearn.model_selection import KFold

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
unique_samples = unique_samples[:50]

# Set up
n_splits = 3
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

query = adata[adata.obs["split1"] == "val"].copy()
ref = adata[adata.obs["split1"] == "train"].copy()
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
mil.train()

