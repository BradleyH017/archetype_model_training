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

See [hyperparameters.md](hyperparameters.md) for full details of the hyperparameter search grids and optimisation strategy for each model.

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
│   ├── 000_split_by_cluster.py  # Splits full AnnData into one per-cluster AnnData
│   ├── 001_prepare_data.py      # Pseudo-bulk features + MIL bags (one job per cluster)
│   ├── 002_train_tree_models.py # XGBoost / RF nested CV
│   ├── 003_train_mil.py         # Attention MIL nested CV
│   ├── 004_evaluate.py          # Per-cluster evaluation & plots
│   ├── 005_aggregate.py         # Cross-cluster summary & plots
│   ├── 006_finalize_model.py    # Manual: refit one deployable model on all data
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

### 0 · Split by cluster (`000_split_by_cluster.py`)

Runs once. Loads the full input AnnData, derives each cell's cluster via the celltype→cluster mapping file (`adata.obs` only contains `celltype_col`, not `cluster_col`), and writes:

- One lightweight per-cluster AnnData (`{cluster}.cluster_input.h5ad`).
- A slim `(sample_col, celltype_col)` table across all cells (`full_obs_celltypes.tsv.gz`), used downstream only if `abundance_type` is `global` or `both`.

This lets `001_prepare_data.py` below run as one parallel job per cluster instead of looping over clusters sequentially in a single large-memory job.

### 1 · Prepare data (`001_prepare_data.py`)

Runs once per cluster (fanned out from step 0's output), each job independently:

- Computes pseudo-bulk profiles (mean or sum per sample).
- Applies a prevalence filter (`min_sample_pct`): removes genes expressed in fewer than the given fraction of samples. This is the **only** gene filtering done here.
- Optionally computes cell-type abundance features normalised to a simplex, restricted to cell types present within that cluster.

Note this step does **not** select HVGs — see step 2/3 below for why, and where that happens instead.

**Outputs** (one set per cluster, e.g. cluster `A`):

| File | Contents |
|---|---|
| `A.pseudobulk.npz` | Sparse CSR matrix — samples × (prevalence-filtered genes + abundance features), plus `n_gene_features` marking the gene/abundance column boundary |
| `A.targets.csv` | Samples × archetype weights (regression targets) |
| `A.mil_bags.h5` | HDF5 — one dataset per sample, shape cells × (prevalence-filtered genes + abundance features), with `n_gene_features` stored as an attribute |

### 2 · Train tree models (`002_train_tree_models.py`)

Runs nested cross-validation (default 5 outer × 5 inner folds) for XGBoost, Random Forest, and the joint multi-task ElasticNet/Lasso (`MultiTaskElasticNet`/`MultiTaskLasso` — one shared feature support across all of a cluster's archetype targets). Hyperparameters are searched with `RandomizedSearchCV` (50 iterations per outer fold). `StandardScaler` is fitted on the training split only.

Hyperparameter search and per-fold R² use `multioutput='variance_weighted'` scoring, not `'uniform_average'`: each target column's contribution is weighted by its own variance rather than averaged uniformly across columns. For CLR-PCA targets this is equivalent to a genuine multivariate R² based on total squared Euclidean (Aitchison) distance, since CLR→PCA preserves distance — a model that nails a high-variance PC now correctly counts for more than one that only does well on a near-zero-variance PC.

**HVGs are reselected independently for each outer CV fold**, from that fold's training samples only (`hvg_pct`/`hvg_max`, consumed here rather than in step 1). Ranking genes by variance across *all* samples before any train/test split exists — including future test-fold samples — is a mild but real data leak; computing it per fold avoids that, at the cost of each fold potentially keeping a different gene panel (feature importances are combined across folds accordingly — see `n_folds_selected` below).

When `raw_or_clr_pca == "pca"` and `run_independent_pca_fits` is enabled, two extra models (`elasticnet_indep`, `lasso_indep`) also run: independent single-output ElasticNet/Lasso fit separately per archetype target, with no shared feature support. This exists because PCA targets are orthogonal by construction, which weakens the multi-task assumption that all targets share a common predictive feature set — comparing the joint vs. independent fits' OOF R² lets you check empirically whether that assumption holds for this data.

**Outputs** under `{cluster}_tree/`:

| File | Contents |
|---|---|
| `{model}_fold_metrics.csv` | R² and RMSE per outer fold |
| `{model}_oof_predictions.csv` | OOF predicted archetype weights |
| `{model}_feature_importances.csv` | Mean importance across folds that selected each feature (NaN-aware; abundance features are in every fold), plus `n_folds_selected` — how many of the outer folds' fold-specific HVG reselection kept that gene, a stability diagnostic |
| `{model}_summary_metrics.csv` | Per-archetype OOF R² and RMSE |
| `{model}_overall_metrics.csv` | One aggregate row: multivariate (variance-weighted) R² and aggregate RMSE across ALL of the cluster's targets at once |

### 3 · Train MIL (`003_train_mil.py`)

Trains a **gated-attention MIL** model (Ilse et al., 2018). Each sample is a *bag* of cells; the model learns an attention-weighted pooling over cells before making a simplex prediction.

Architecture:
- Per-cell encoder: two-layer MLP with LayerNorm
- Attention gate: `tanh(Vh) ⊙ σ(Uh)` → softmax pooling
- Bag predictor: linear layers → softmax output

Training uses Adam + cosine-annealing LR over up to 200 epochs. Hyperparameters (hidden dim, attention dim, dropout, LR, weight decay, batch size, epochs) are randomly sampled (20 combinations per outer fold).

Like the tree/linear models, **HVGs are reselected per outer fold** (`hvg_pct`/`hvg_max`) rather than fixed at data-prep time: each fold's training bags are first pseudobulked (`pseudobulk_agg` — cells within a bag mean/summed to one profile per sample), then ranked by variance across those per-sample profiles, the same statistic used in step 2. This keeps the selection criterion consistent between MIL and the tree/linear models and avoids the same test-fold leakage. R² (both for inner-loop hyperparameter comparison and outer-fold reporting) also uses the same `variance_weighted` scoring as step 2.

**Outputs** under `{cluster}_mil/` — same structure as tree outputs, including `mil_overall_metrics.csv` (MIL doesn't produce a feature-importance file, since attention weights aren't a per-gene importance score in the same sense).

### 4 · Evaluate (`004_evaluate.py`)

Combines metrics from all models for a single cluster and generates:

- `combined_metrics.csv` — all models × archetypes
- `combined_overall_metrics.csv` — one row per (model × label_mode): multivariate (variance-weighted) R² and aggregate RMSE
- `model_comparison_bars.pdf` — R² and RMSE bar charts by archetype
- `{model}_oof_scatter.pdf` — true vs predicted scatter per archetype
- `{model}_feature_importance.pdf` — top-30 features (tree models)

### 5 · Aggregate (`005_aggregate.py`)

Collects evaluation outputs across all clusters and generates:

- `all_clusters_metrics.csv` — full long-format table
- `model_summary.csv` — mean / median / std R² and RMSE per model, pooled uniformly across every individual archetype/PC row (a cluster with more targets counts more)
- `all_clusters_overall_metrics.csv` — one row per (cluster × model × label_mode) multivariate metric
- `model_summary_overall.csv` — per model, `r2_variance_weighted`/`rmse_aggregate` averaged **across clusters** (each cluster contributes one number regardless of how many targets it has) — a more principled ranking than `model_summary.csv`'s flat pooling, worth checking alongside it rather than instead of it
- `{model}_r2_heatmap.pdf` — R² heatmap (clusters × archetypes)
- `model_r2_boxplot.pdf` — R² distribution per model
- `model_r2_stripplot.pdf` — per-archetype R², faceted by cluster

---

## Finalizing a model for deployment (`006_finalize_model.py`)

Nested CV (steps 2–3) exists purely to **estimate generalization performance honestly** — it deliberately trains and discards 5 different models per outer fold, and produces nothing you could apply to a genuinely new, unseen sample. `006_finalize_model.py` is a separate, manual script (not part of `main.nf`'s automatic DAG) for that: given one chosen cluster/model/label_mode, it refits a single model on **all** currently available samples and persists everything needed to score a new sample later.

Run it only after using nested CV — and `model_summary_overall.csv` in particular — to decide which cluster/model/label_mode combination is actually worth finalizing:

```bash
python scripts/006_finalize_model.py \
    --cluster B \
    --pseudobulk results/prepared_data/B.pseudobulk.npz \
    --targets    results/prepared_data/B.targets.csv \
    --model      elasticnet_indep \
    --elasticnet_params scripts/param_grids/elasticnet_params.json \
    --hvg_pct 0.20 --hvg_max 3000 \
    --label_mode full \
    --outdir results/final_models
```

It selects HVGs once (from all samples — there's no held-out fold left to protect), fits a `StandardScaler`, then runs plain (non-nested) `RandomizedSearchCV` — which performs hyperparameter selection via k-fold CV and refits on the full data with the winning combination in one call. The result is saved as a single `joblib` bundle (`{cluster}_{model}_{label_mode}_final_model.joblib`) containing the fitted estimator(s), scaler, exact feature names/order, and metadata (chosen hyperparameters, CV scores) — everything needed to transform and score a new sample's pseudobulk profile later.

Supports `xgboost`, `rf`, `elasticnet`, `lasso`, `elasticnet_indep`, `lasso_indep`. **Does not support `mil`** — MIL's persistence needs a meaningfully different format (torch model state, attention weights); passing `--model mil` raises a clear error.

---

## Key parameters (`conf/training.config`)

| Parameter | Default | Description |
|---|---|---|
| `cluster_col` | `mrvi_merged_cluster` | AnnData column with cluster labels |
| `sample_col` | `donor_id` | Per-sample identifier column |
| `raw_or_clr_pca` | `raw` | `raw` uses simplex weight targets (e.g. `B1`); `pca` uses CLR-PCA transformed targets (e.g. `B1_clr_pca`) |
| `run_independent_pca_fits` | `true` | Only used when `raw_or_clr_pca == "pca"`: also fit independent per-target ElasticNet/Lasso (`elasticnet_indep`, `lasso_indep`) alongside the joint multi-task fits |
| `hvg_pct` | `0.20` | Fraction of genes to keep as HVGs — reselected per outer CV fold in `002`/`003`, not at data-prep time |
| `hvg_max` | `2000` | Hard cap on number of HVGs — same per-fold reselection as `hvg_pct` |
| `pseudobulk_agg` | `mean` | Aggregation method (`mean` or `sum`); also used by `003_train_mil.py` to pseudobulk each fold's training bags for HVG ranking |
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
