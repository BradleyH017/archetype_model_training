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
│   ├── 007_permutation_null.py  # Automatic: per-PC permutation null test (ElasticNet/Lasso family)
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

XGBoost itself comes in two variants, run as genuinely different model names rather than a flag, because they need different XGBoost versions:
- `xgboost_indep` (`run_xgboost`) — one independent single-output `XGBRegressor` per archetype target, via sklearn's `MultiOutputRegressor`. This is the pipeline's original XGBoost behavior.
- `xgboost_joint` (`run_xgboost_joint`, off by default) — XGBoost's native `multi_strategy="multi_output_tree"` (requires XGBoost≥2.0): one shared ensemble of trees per cluster, each leaf holding a vector across all archetype targets — the direct XGBoost analogue of the joint `MultiTaskElasticNet`/`MultiTaskLasso` fits above. It runs through a separate Nextflow process/container (`params.xgboost2_env`), since `container_ml`'s pinned XGBoost predates `multi_strategy`. It's still upstream-experimental: no Tree SHAP support at all, and `feature_importances_` collapses to a single vector shared across all targets (like `rf`) rather than per-archetype.

Hyperparameter search and per-fold R² use `multioutput='variance_weighted'` scoring, not `'uniform_average'`: each target column's contribution is weighted by its own variance rather than averaged uniformly across columns. For CLR-PCA targets this is equivalent to a genuine multivariate R² based on total squared Euclidean (Aitchison) distance, since CLR→PCA preserves distance — a model that nails a high-variance PC now correctly counts for more than one that only does well on a near-zero-variance PC.

**HVGs are reselected independently for each outer CV fold**, from that fold's training samples only (`hvg_pct`/`hvg_max`, consumed here rather than in step 1). Ranking genes by variance across *all* samples before any train/test split exists — including future test-fold samples — is a mild but real data leak; computing it per fold avoids that, at the cost of each fold potentially keeping a different gene panel (feature importances are combined across folds accordingly — see `n_folds_selected` below).

When `raw_or_clr_pca == "pca"` and `run_independent_pca_fits` is enabled, two extra models (`elasticnet_indep`, `lasso_indep`) also run: independent single-output ElasticNet/Lasso fit separately per archetype target, with no shared feature support. This exists because PCA targets are orthogonal by construction, which weakens the multi-task assumption that all targets share a common predictive feature set — comparing the joint vs. independent fits' OOF R² lets you check empirically whether that assumption holds for this data.

**Technical covariate regression** (`meta_file` / `train_only_covs` / `train_test_covs`, all optional, off by default): removes technical-covariate effects from the pseudobulk features — not the per-cell MIL bags — from whichever genes HVG selection has already picked (selection itself always runs first, on raw pre-regression values — see the HVG selection method paragraph below for why). Fit fresh per outer fold, on that fold's training samples only (categorical covariate columns are one-hot encoded, continuous ones passed through; see `helpers.fit_covariate_design`). The two covariate lists get different treatment across the train/test boundary:
- `train_only_covs` — regressed out of the training fold only; the held-out fold is left untouched, and the fit is discarded. For covariates specific to this dataset's own processing that are unmeasurable or inapplicable for genuinely external data (e.g. a batch/pooling identifier) — adjusting the held-out fold too would assume a correction capability you won't have at deployment, inflating the OOF estimate.
- `train_test_covs` — fit on the training fold, then the same fitted adjustment is applied to the held-out fold too (standard fit-on-train/apply-to-test, like `StandardScaler`). For covariates measurable identically in any dataset (e.g. `total_counts`/`n_genes_expressed`). This one is also persisted in `006_finalize_model.py`'s bundle (`covariate_model`) for reapplication to genuinely new external samples at scoring time.

`meta_file` is a per-sample metadata CSV (first column = sample ID, matching `sample_col`); required only if either covariate list is non-empty. See `helpers.regress_out_covariates` for the full implementation.

**HVG selection method** (`hvg_select_method`, `"default"` or `"sq_target_vs_technical"`) **always runs before covariate regression, on each fold's raw training values** — technical covariates are then regressed out only of whichever columns got selected (equivalent to regressing every candidate gene first and subsetting after, since regression is column-independent, but cheaper). Originally `train_only_covs` was regressed out of the training fold only, never the held-out fold — asymmetric by design, for covariates unmeasurable/inapplicable on genuinely external data — and a plain covariate-residualized variance ranking only *softly* downweighted genes whose expression tracked those covariates, so a gene could still rank highly on inflated raw variance despite that inflation being invisible to the (untouched) held-out fold. That combination showed up concretely as a pathologically negative OOF R² for one archetype whose top genes turned out to be driven largely by a technical covariate rather than the archetype itself. `"sq_target_vs_technical"` ranks each gene instead by `target_r2² / covariate_r2` — the fraction of that gene's raw variance explained by the archetype targets (squared, to penalize genes that are merely unconfounded but uninformative) divided by the fraction explained by a covariate list — computed fresh per outer fold from that fold's training samples' raw values (`helpers.select_hvgs_confound_aware`). That covariate list is `train_only_covs` if non-empty, else `train_test_covs` — originally scoped only to `train_only_covs`, since only those covariates carried the asymmetric miscalibration risk above; once a covariate is symmetrically regressed out (fit-on-train/apply-to-test, no asymmetry), the remaining reason to rank against it at all is just not wasting a necessarily limited HVG budget on genes whose apparent importance is more attributable to a covariate than to the phenotype, which applies just as well to `train_test_covs`. It **requires a non-empty `train_only_covs` or `train_test_covs`** (raises `ValueError` if both are empty) and is only used by `002_train_tree_models.py`/`006_finalize_model.py`/`007_permutation_null.py` — **not** `003_train_mil.py`, whose own per-fold HVG reselection on pseudobulked bags is untouched. For `elasticnet_indep`/`lasso_indep`, each archetype target gets its own HVG panel under this method (feature importances are combined and reported per-target accordingly, see `n_folds_selected_{archetype}` below); every other model (including `xgboost_indep`, for which `MultiOutputRegressor` requires one shared feature matrix across targets) keeps one shared panel. See `temp/hvg_selection.txt` for the full derivation, alternatives considered and rejected, and the empirical validation across every cluster/archetype available.

**Outputs** under `{cluster}_tree/`:

| File | Contents |
|---|---|
| `{model}_fold_metrics.csv` | R² and RMSE per outer fold |
| `{model}_oof_predictions.csv` | OOF predicted archetype weights |
| `{model}_feature_importances.csv` | Mean importance across folds that selected each feature (NaN-aware; abundance features are in every fold), plus `n_folds_selected` — how many of the outer folds' fold-specific HVG reselection kept that gene, a stability diagnostic. For `elasticnet_indep`/`lasso_indep` under `hvg_select_method == "sq_target_vs_technical"`, this is instead one `n_folds_selected_{archetype}` column per target, since each target can get its own HVG panel |
| `{model}_summary_metrics.csv` | Per-archetype OOF R² and RMSE |
| `{model}_overall_metrics.csv` | One aggregate row: multivariate (variance-weighted) R² and aggregate RMSE across ALL of the cluster's targets at once |

**What "importance" actually means differs by model family.** For `elasticnet`/`lasso` (and their `_indep` variants) it's the fitted regression coefficient — signed, additive, and a rigorous per-feature effect size with no caveats beyond the usual ones for penalized linear models. For `xgboost_indep`/`xgboost_joint`/`rf` it's each model's **native, built-in importance** — Mean Decrease in Impurity for RF, gain for XGBoost (`best.feature_importances_` in `helpers._fit_outer_fold`) — **not SHAP**. These native importances are cheap (computed automatically during training, no extra pass) but have well-known biases: they favor high-cardinality/continuous features (more candidate split points inflates apparent importance) and split "credit" somewhat arbitrarily among correlated features rather than attributing it fairly. If RF/XGBoost's gene rankings matter for a real conclusion, treat them as a rough guide rather than a rigorous effect size — TreeSHAP (exact and reasonably cheap for tree ensembles specifically, unlike the much more expensive KernelSHAP needed for arbitrary models) would be the natural upgrade if more rigor is needed here — though note this is currently only possible for `xgboost_indep` (each per-target `estimators_[j]` is a normal single-output tree model); upstream does not yet support Tree SHAP for `xgboost_joint`'s vector-leaf trees at all, and `rf`/`xgboost_joint`'s `feature_importances_` is a single vector shared across all targets, not per-archetype, unlike `xgboost_indep`'s per-target `estimators_`.

### 2a · Permutation null test (`007_permutation_null.py`)

Runs automatically for the ElasticNet/Lasso family (`elasticnet`, `lasso`, `elasticnet_indep`, `lasso_indep`) and `xgboost_indep` — not `xgboost_joint`/`rf`/`mil` — right after step 2, one job per (cluster × model × label_mode). Answers a more targeted question than "is this R² impressive": *does this specific model, with the hyperparameters its own nested CV actually settled on, explain more variance in each archetype PC than you'd get by chance?*

Hyperparameters are **not re-searched** — they're fixed to the mode (most common choice) across that model's own `{model}_fold_metrics.csv` rows from step 2, so each permutation is just one plain (non-nested) K-fold CV pass with no inner `RandomizedSearchCV` loop. That's what makes it cheap enough to run automatically for every model before you've picked a winner, rather than only against a single already-chosen one — this script has no dependency on `006_finalize_model.py`.

For each archetype PC: an "observed" R² is computed under the fixed-hyperparameter K-fold CV (same fold split used for every run). Sample labels (`y`, all target columns shuffled together as one consistent row permutation, preserving cross-target structure while breaking the X↔y link) are then shuffled `n_permutations` times, refitting the same fixed-hyperparameter procedure each time — including fresh per-fold HVG reselection, since that's part of the procedure being tested, not something to hold fixed — to build a null R² distribution. An empirical p-value per PC is `(1 + count(null_r2 >= observed_r2)) / (n_permutations + 1)`.

**Outputs** under `results/permutation_null/{cluster}/{model}/{label_mode}/`:

| File | Contents |
|---|---|
| `{cluster}_{model}_{label_mode}_permutation_summary.csv` | One row per archetype: observed R², null mean/std/95th percentile, p-value |
| `{cluster}_{model}_{label_mode}_permutation_null_raw.csv` | Every individual permutation's per-PC R² (long format), plus the observed run tagged `permutation_id = -1`, for plotting a null histogram |

Only the summary CSV is threaded into `EVALUATE_MODELS`/`AGGREGATE_RESULTS` below (see `combined_permutation_null.csv` / `all_clusters_permutation_null.csv`); the raw per-permutation draws stay published here for direct inspection if you want to plot a histogram yourself.

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
- `combined_permutation_null.csv` — ElasticNet/Lasso family + `xgboost_indep` only (see step 2a): observed R², null distribution stats, and p-value per archetype — absent/empty if none of those models succeeded for this cluster
- `model_comparison_bars.pdf` — R² and RMSE bar charts by archetype
- `{model}_oof_scatter.pdf` — true vs predicted scatter per archetype
- `{model}_feature_importance.pdf` — top-30 features (tree models)

### 5 · Aggregate (`005_aggregate.py`)

Collects evaluation outputs across all clusters and generates:

- `all_clusters_metrics.csv` — full long-format table
- `model_summary.csv` — mean / median / std R² and RMSE per model, pooled uniformly across every individual archetype/PC row (a cluster with more targets counts more)
- `all_clusters_overall_metrics.csv` — one row per (cluster × model × label_mode) multivariate metric
- `model_summary_overall.csv` — per model, `r2_variance_weighted`/`rmse_aggregate` averaged **across clusters** (each cluster contributes one number regardless of how many targets it has) — a more principled ranking than `model_summary.csv`'s flat pooling, worth checking alongside it rather than instead of it
- `all_clusters_permutation_null.csv` — every cluster's `combined_permutation_null.csv` rows pooled together (ElasticNet/Lasso family + `xgboost_indep`). Kept as its own table rather than merged into `model_summary_overall.csv` — an R² and a p-value answer different questions, and conflating them into one ranking would misrepresent both
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

HVGs are selected once (from all samples, on raw values) — via `--hvg_select_method`, same `"default"`/`"sq_target_vs_technical"` choice and covariate-list requirement as step 2, but always as **one shared panel for every model**, including `elasticnet_indep`/`lasso_indep` — unlike step 2's genuinely per-target panels for those two, this is a deliberate simplification consistent with `006`'s existing non-fold-specific design. If `train_only_covs`/`train_test_covs` are set (same `meta_file` convention as step 2), technical covariates are then regressed out once, on the full dataset, from the selected columns — there's no held-out fold here to protect either way. It then fits a `StandardScaler`, then runs plain (non-nested) `RandomizedSearchCV` — which performs hyperparameter selection via k-fold CV and refits on the full data with the winning combination in one call. The result is saved as a single `joblib` bundle (`{cluster}_{model}_{label_mode}_final_model.joblib`) containing the fitted estimator(s), scaler, exact feature names/order, and metadata (chosen hyperparameters, CV scores) — everything needed to transform and score a new sample's pseudobulk profile later. If `train_test_covs` was used, the bundle also includes `covariate_model` (the fitted `ColumnTransformer`/`LinearRegression` pair) to reapply that same adjustment to a genuinely new sample's covariates before scoring; `train_only_covs`' adjustment is fit-and-discarded and never reapplied (see step 2's covariate section), so `train_only_covs` is recorded in the bundle for provenance only.

Supports `xgboost_indep`, `xgboost_joint`, `rf`, `elasticnet`, `lasso`, `elasticnet_indep`, `lasso_indep`. **Does not support `mil`** — MIL's persistence needs a meaningfully different format (torch model state, attention weights); passing `--model mil` raises a clear error.

Note: the permutation-null significance test (`007_permutation_null.py`) is a separate concern from finalizing — it runs automatically in the main pipeline for the ElasticNet/Lasso family and `xgboost_indep`, from each model's own real nested-CV fold_metrics, with no dependency on this script. See step 2a above.

---

## Key parameters (`conf/training.config`)

| Parameter | Default | Description |
|---|---|---|
| `cluster_col` | `mrvi_merged_cluster` | AnnData column with cluster labels |
| `sample_col` | `donor_id` | Per-sample identifier column |
| `raw_or_clr_pca` | `raw` | `raw` uses simplex weight targets (e.g. `B1`); `pca` uses CLR-PCA transformed targets (e.g. `B1_clr_pca`) |
| `run_independent_pca_fits` | `true` | Only used when `raw_or_clr_pca == "pca"`: also fit independent per-target ElasticNet/Lasso (`elasticnet_indep`, `lasso_indep`) alongside the joint multi-task fits |
| `run_xgboost_joint` | `false` | Also fit `xgboost_joint` (native `multi_strategy="multi_output_tree"`, requires XGBoost≥2.0) alongside the default `xgboost_indep`. Runs via `xgboost2_env` below, not `container_ml` |
| `meta_file` | `""` | Per-sample metadata CSV for technical covariate regression (first column = sample ID, matching `sample_col`). Required only if either covariate list below is non-empty |
| `train_only_covs` | `""` | Space-separated covariate column(s) regressed out of the training fold only — not applied to the held-out fold, never persisted. For covariates unmeasurable/inapplicable in external data (e.g. a batch/pooling identifier) |
| `train_test_covs` | `""` | Space-separated covariate column(s) regressed out of both the training and held-out fold (fit-on-train/apply-to-test) — persisted in `006_finalize_model.py`'s bundle. For covariates measurable identically in any dataset (e.g. `total_counts`/`n_genes_expressed`) |
| `xgboost2_env` | `""` | Container/environment with XGBoost≥2.0, used only by `xgboost_joint`. Required when `run_xgboost_joint = "true"` |
| `hvg_pct` | `0.20` | Fraction of genes to keep as HVGs — reselected per outer CV fold in `002`/`003`, not at data-prep time |
| `hvg_max` | `2000` | Hard cap on number of HVGs — same per-fold reselection as `hvg_pct` |
| `hvg_select_method` | `default` | Selection always runs before covariate regression, on raw values. `default` ranks HVGs by raw variance; `sq_target_vs_technical` ranks by `target_r2²/covariate_r2` instead, scored against `train_only_covs` if non-empty else `train_test_covs` (requires at least one of the two non-empty). Used by `002`/`006`/`007`, not `003_train_mil.py`. See `temp/hvg_selection.txt` for the full rationale |
| `pseudobulk_agg` | `mean` | Aggregation method (`mean` or `sum`); also used by `003_train_mil.py` to pseudobulk each fold's training bags for HVG ranking |
| `min_sample_pct` | `0.2` | Minimum fraction of samples a gene must be expressed in |
| `include_abundance` | `true` | Include cell-type frequency features |
| `abundance_type` | `within_cluster` | `within_cluster`, `global`, or `both` |
| `n_outer_folds` | `5` | Outer CV folds |
| `n_inner_folds` | `5` | Inner CV folds (HPO) |
| `n_hpo_iter_tree` | `50` | RandomizedSearch iterations (tree models) |
| `n_hpo_iter_mil` | `20` | Random HPO draws (MIL) |
| `n_permutations` | `100` | Permutation-null draws per (cluster × ElasticNet/Lasso-family model); uses `n_outer_folds` for its fold count, no separate hyperparameter search |
| `max_cells_per_bag` | `2000` | Cells subsampled per sample for MIL |
| `random_seed` | `42` | Global random seed |

---

## Dependencies

Tree models require `scikit-learn`, `xgboost`, and `anndata`/`scanpy`. MIL training additionally requires `torch`. The Nextflow pipeline pulls the appropriate Singularity containers automatically on the Sanger cluster — except `xgboost_joint`, which needs XGBoost≥2.0 and runs in whatever container/environment you point `xgboost2_env` at, separately from `container_ml`.
