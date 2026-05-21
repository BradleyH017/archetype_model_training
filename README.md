# Archetype Classifier — Model Training Pipeline

Trains machine learning models to predict per-sample archetype simplex weights from single-cell RNA-seq data. Three model families are supported: **XGBoost**, **Random Forest**, and a **gated-attention Multiple Instance Learning (MIL)** network. All models are evaluated with nested cross-validation to give unbiased out-of-fold (OOF) performance estimates.

---

## Overview

Each cell cluster is treated independently. For each cluster the pipeline:

1. Aggregates cells to pseudo-bulk expression profiles per sample (mean or sum).
2. Filters lowly-expressed genes and selects highly variable genes (HVGs).
3. Optionally appends cell-type abundance features (within-cluster or global frequencies).
4. Trains XGBoost, Random Forest, and/or attention-MIL models with nested cross-validation.
5. Evaluates and compares models, then aggregates results across all clusters.

Targets are the archetype simplex weights produced by the upstream archetypal analysis (MRVI).

---

## Directory structure

```
model_training/
├── workflow/
│   └── main.nf                  # Nextflow DSL2 pipeline
├── conf/
│   └── training.config          # All pipeline parameters
├── nextflow.config              # Executor / resource / container settings
├── scripts/
│   ├── helpers.py               # Shared library (feature prep, models, eval)
│   ├── 001_prepare_data.py      # Pseudo-bulk features + MIL bags
│   ├── 002_train_tree_models.py # XGBoost / RF nested CV
│   ├── 003_train_mil.py         # Attention MIL nested CV
│   ├── 004_evaluate.py          # Per-cluster evaluation & plots
│   ├── 005_aggregate.py         # Cross-cluster summary & plots
│   └── param_grids/
│       ├── xgboost_params.json
│       ├── rf_params.json
│       └── mil_params.json
└── results/                     # Pipeline outputs (generated at runtime)
```

---

## Running the pipeline

```bash
nextflow run workflow/main.nf -profile sanger -params-file conf/training.config
```

The `sanger` profile uses an LSF executor and Singularity containers. Resource limits are defined in `nextflow.config` (up to 700 GB RAM, 32 CPUs, 720 h per job).

---

## Pipeline steps

### 1 · Prepare data (`001_prepare_data.py`)

For each cluster:

- Filters the AnnData to cells belonging to that cluster (optionally using a celltype→cluster mapping file).
- Computes pseudo-bulk profiles (mean or sum per sample).
- Applies a prevalence filter (`min_sample_pct`): removes genes expressed in fewer than the given fraction of samples.
- Selects the top HVGs by variance (`hvg_pct`, capped at `hvg_max`).
- Optionally computes cell-type abundance features normalised to a simplex, restricted to cell types present within that cluster.

**Outputs** (one set per cluster, e.g. cluster `A`):

| File | Contents |
|---|---|
| `A.pseudobulk.npz` | Sparse CSR matrix — samples × (HVGs + abundance features) |
| `A.targets.csv` | Samples × archetype weights (regression targets) |
| `A.mil_bags.h5` | HDF5 — one dataset per sample, shape cells × features |

### 2 · Train tree models (`002_train_tree_models.py`)

Runs nested cross-validation (default 5 outer × 5 inner folds) for XGBoost and Random Forest. Hyperparameters are searched with `RandomizedSearchCV` (50 iterations per outer fold). `StandardScaler` is fitted on the training split only.

**Outputs** under `{cluster}_tree/`:

| File | Contents |
|---|---|
| `{model}_fold_metrics.csv` | R² and RMSE per outer fold |
| `{model}_oof_predictions.csv` | OOF predicted archetype weights |
| `{model}_feature_importances.csv` | Mean importance across folds |
| `summary_metrics.csv` | Per-archetype OOF R² and RMSE |

### 3 · Train MIL (`003_train_mil.py`)

Trains a **gated-attention MIL** model (Ilse et al., 2018). Each sample is a *bag* of cells; the model learns an attention-weighted pooling over cells before making a simplex prediction.

Architecture:
- Per-cell encoder: two-layer MLP with LayerNorm
- Attention gate: `tanh(Vh) ⊙ σ(Uh)` → softmax pooling
- Bag predictor: linear layers → softmax output

Training uses Adam + cosine-annealing LR over up to 200 epochs. Hyperparameters (hidden dim, attention dim, dropout, LR, weight decay, batch size, epochs) are randomly sampled (20 combinations per outer fold).

**Outputs** under `{cluster}_mil/` — same structure as tree outputs.

### 4 · Evaluate (`004_evaluate.py`)

Combines metrics from all models for a single cluster and generates:

- `combined_metrics.csv` — all models × archetypes
- `model_comparison_bars.pdf` — R² and RMSE bar charts by archetype
- `{model}_oof_scatter.pdf` — true vs predicted scatter per archetype
- `{model}_feature_importance.pdf` — top-30 features (tree models)

### 5 · Aggregate (`005_aggregate.py`)

Collects evaluation outputs across all clusters and generates:

- `all_clusters_metrics.csv` — full long-format table
- `model_summary.csv` — mean / median / std R² and RMSE per model
- `{model}_r2_heatmap.pdf` — R² heatmap (clusters × archetypes)
- `model_r2_boxplot.pdf` — R² distribution per model
- `model_r2_stripplot.pdf` — per-archetype R², faceted by cluster

---

## Key parameters (`conf/training.config`)

| Parameter | Default | Description |
|---|---|---|
| `cluster_col` | `mrvi_merged_cluster` | AnnData column with cluster labels |
| `sample_col` | `donor_id` | Per-sample identifier column |
| `hvg_pct` | `0.20` | Fraction of genes to keep as HVGs |
| `hvg_max` | `2000` | Hard cap on number of HVGs |
| `pseudobulk_agg` | `mean` | Aggregation method (`mean` or `sum`) |
| `min_sample_pct` | `0.2` | Minimum fraction of samples a gene must be expressed in |
| `include_abundance` | `true` | Include cell-type frequency features |
| `abundance_type` | `within_cluster` | `within_cluster`, `global`, or `both` |
| `n_outer_folds` | `5` | Outer CV folds |
| `n_inner_folds` | `5` | Inner CV folds (HPO) |
| `n_hpo_iter_tree` | `50` | RandomizedSearch iterations (tree models) |
| `n_hpo_iter_mil` | `20` | Random HPO draws (MIL) |
| `max_cells_per_bag` | `2000` | Cells subsampled per sample for MIL |
| `random_seed` | `42` | Global random seed |

---

## Dependencies

Tree models require `scikit-learn`, `xgboost`, and `anndata`/`scanpy`. MIL training additionally requires `torch`. The Nextflow pipeline pulls the appropriate Singularity containers automatically on the Sanger cluster.
