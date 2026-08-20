"""
helpers.py
Shared utility functions and classes for the archetype classification pipeline.

Sections:
  1. Data I/O          — load_pseudobulk, load_mil_bags
  2. Feature prep      — filter_lowly_expressed_genes, select_hvgs, pseudobulk_aggregate,
                         compute_abundance, get_archetype_weight_cols
  2b. Covariate regression — split_covariate_names, load_meta_file,
                         validate_covariate_lists, fit_covariate_design,
                         transform_covariate_design, regress_out_covariates
  3. Tree/linear models — build_estimator, run_nested_cv_tree, run_fixed_cv
                         (xgboost_indep, xgboost_joint, rf, elasticnet, lasso)
  4. MIL               — AttentionMIL, MILDataset, mil_collate, train_epoch,
                         evaluate_bags, build_mil_model, fit_mil_model,
                         nested_cv_mil
  5. Evaluation I/O    — load_oof, per_archetype_metrics, overall_metrics
"""

import itertools
import json
import re
import warnings

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, MultiTaskElasticNet, MultiTaskLasso
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# import xgboost as xgb

warnings.filterwarnings("ignore")

# Variance-weighted R² scorer for hyperparameter search: for multi-output fits,
# this weights each target column's contribution by its own variance rather
# than averaging per-column R² uniformly. For CLR-PCA targets this is
# equivalent to a genuine multivariate R² based on total squared Euclidean
# (Aitchison) distance, since PCA preserves distance — see overall_metrics().
# For single-target fits it's identical to plain R² (no aggregation to do).
VARIANCE_WEIGHTED_R2_SCORER = make_scorer(r2_score, multioutput='variance_weighted')

# Torch is only available in the MIL container; import gracefully.
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ─── 1. Data I/O ──────────────────────────────────────────────────────────────

def load_pseudobulk(path):
    """Load a pseudo-bulk NPZ written by 001_prepare_data.py.

    Columns are ordered [gene features] + [abundance features]. Gene features
    are NOT yet narrowed to HVGs — that selection is fold-specific and happens
    downstream (see select_hvgs / run_nested_cv_tree), computed from each
    outer fold's training samples only, to avoid ranking genes by variance
    across samples that later become test data.

    Returns X (n_samples × n_features dense float32), sample list, feature
    list, and n_gene_features (the number of leading gene-feature columns;
    columns after this are abundance features, always kept intact).
    """
    d        = np.load(path, allow_pickle=True)
    X        = sp.csr_matrix(
                   (d['data'], d['indices'], d['indptr']),
                   shape=tuple(d['shape'])
               ).toarray()
    samples         = d['samples'].tolist()
    features        = d['features'].tolist()
    n_gene_features = int(d['n_gene_features'])
    return X, samples, features, n_gene_features


def load_mil_bags(path):
    """Load MIL bags HDF5 written by 001_prepare_data.py.

    Each bag's columns are ordered [gene features] + [abundance features],
    matching load_pseudobulk. Returns list of np.arrays (one per sample),
    corresponding sample list, and n_gene_features (see load_pseudobulk).
    """
    bags, samples = [], []
    with h5py.File(path, "r") as hf:
        n_gene_features = int(hf.attrs['n_gene_features'])
        all_samples = list(hf.attrs['samples'])
        for s in all_samples:
            safe_key = s.replace("/", "_")
            if safe_key in hf:
                bags.append(hf[safe_key][:])
                samples.append(s)
    return bags, samples, n_gene_features


# ─── 2. Feature preparation ───────────────────────────────────────────────────

def filter_lowly_expressed_genes(pb_matrix, min_pct):
    """Return a boolean mask of genes expressed (> 0) in >= min_pct of pseudo-bulk samples.

    Genes expressed in fewer than min_pct of samples are removed before HVG selection
    to reduce noise from rarely detected genes.
    min_pct is a fraction in [0, 1], e.g. 0.2 keeps genes expressed in >=20% of samples.
    """
    frac_expressed = (pb_matrix > 0).mean(axis=0)
    return frac_expressed >= min_pct


def select_hvgs(pb_matrix, hvg_pct, hvg_max):
    """Return sorted gene indices for the top HVGs by variance across pseudo-bulk samples."""
    variances = np.asarray(pb_matrix.var(axis=0)).flatten()
    n_hvg     = min(hvg_max, max(1, int(pb_matrix.shape[1] * hvg_pct)))
    idx       = np.argsort(variances)[::-1][:n_hvg]
    return np.sort(idx)


def pseudobulk_aggregate(adata_cluster, sample_col, agg="mean"):
    """Aggregate per-cell expression to per-sample pseudo-bulk.

    agg : 'mean' | 'sum'
    Returns (n_samples × n_genes) float32 matrix and ordered sample list.
    """
    if agg not in ("mean", "sum"):
        raise ValueError(f"agg must be 'mean' or 'sum', got '{agg}'")
    samples = sorted(adata_cluster.obs[sample_col].unique())
    rows = []
    for s in samples:
        mask = (adata_cluster.obs[sample_col] == s).values
        X_s  = adata_cluster.X[mask, :]
        if sp.issparse(X_s):
            vec = np.asarray(X_s.sum(axis=0)).flatten()
            if agg == "mean":
                vec = vec / max(mask.sum(), 1)
        else:
            vec = np.asarray(X_s).sum(axis=0)
            if agg == "mean":
                vec = vec / max(mask.sum(), 1)
        rows.append(vec)
    return np.array(rows, dtype=np.float32), samples


def compute_abundance(obs_full, adata_cl, sample_col, celltype_col,
                      abundance_type, samples):
    """Cell-type frequency features per sample.

    obs_full : pd.DataFrame with [sample_col, celltype_col] across ALL clusters,
               or None — only required when abundance_type is 'global' or 'both'.
    abundance_type : 'within_cluster' | 'global' | 'both'
    Returns (n_samples × n_celltypes) float32 array and list of column names.
    """
    if abundance_type in ("within_cluster", "both"):
        ct_wc = (adata_cl.obs
                 .groupby([sample_col, celltype_col]).size()
                 .unstack(fill_value=0))
        ct_wc = ct_wc.div(ct_wc.sum(axis=1), axis=0).reindex(samples).fillna(0)
        wc_names = [f"abund_wc_{c}" for c in ct_wc.columns]

    if abundance_type in ("global", "both"):
        ct_gl = (obs_full
                 .groupby([sample_col, celltype_col]).size()
                 .unstack(fill_value=0))
        ct_gl = ct_gl.div(ct_gl.sum(axis=1), axis=0).reindex(samples).fillna(0)
        gl_names = [f"abund_gl_{c}" for c in ct_gl.columns]

    if abundance_type == "within_cluster":
        return ct_wc.values.astype(np.float32), wc_names
    if abundance_type == "global":
        return ct_gl.values.astype(np.float32), gl_names
    return (np.hstack([ct_wc.values, ct_gl.values]).astype(np.float32),
            wc_names + gl_names)


def get_archetype_weight_cols(df_scores, cluster_prefix, target_mode="raw"):
    """Return simplex-weight columns for a cluster.

    target_mode : 'raw' → e.g. 'A' → ['A1','A2']
                  'pca' → CLR-PCA transformed columns, e.g. 'A' → ['A1_clr_pca','A2_clr_pca']
    """
    if target_mode == "raw":
        pat = re.compile(rf'^{re.escape(cluster_prefix)}\d+$')
    elif target_mode == "pca":
        pat = re.compile(rf'^{re.escape(cluster_prefix)}\d+_clr_pca$')
    else:
        raise ValueError(f"target_mode must be 'raw' or 'pca', got '{target_mode}'")
    return [c for c in df_scores.columns if pat.match(c)]


def apply_label_mode(y, mode, min_score=0.5):
    """Transform targets according to label mode.

    Returns (y_out, keep_mask):
      'full'            → y unchanged, keep_mask all-True
      'argmax'          → one-hot at argmax col, keep_mask all-True
      'argmax_filtered' → one-hot at argmax col, keep_mask where max weight > min_score
    """
    if mode == "full":
        return y.copy(), np.ones(len(y), dtype=bool)
    argmax_idx = np.argmax(y, axis=1)
    y_onehot = np.zeros_like(y)
    y_onehot[np.arange(len(y)), argmax_idx] = 1.0
    if mode == "argmax_filtered":
        keep_mask = y.max(axis=1) > min_score
    else:
        keep_mask = np.ones(len(y), dtype=bool)
    return y_onehot, keep_mask


# ─── 2b. Technical covariate regression ──────────────────────────────────────
# Removes technical-covariate effects from pseudobulk features BEFORE HVG
# selection, since select_hvgs ranks genes by raw variance — a covariate that
# inflates variance in biologically uninformative genes can otherwise crowd
# out real signal from the HVG panel. Two distinct regimes, matching whether a
# covariate will be available/meaningful for genuinely external data:
#   train_only_covs — regressed out of the TRAINING rows only; the held-out
#     fold (or, in 006_finalize_model.py, any future external sample) is left
#     untouched, and the fitted model is never persisted. For covariates that
#     are specific to this dataset's own processing (e.g. a batch/pooling
#     identifier) and therefore uncomputable/meaningless for external data —
#     adjusting the held-out side too would assume a correction capability
#     you won't actually have at deployment, inflating the OOF estimate.
#   train_test_covs — fit on training rows, then the SAME fitted adjustment is
#     applied to the held-out fold (or persisted and applied to new external
#     samples). For covariates that are measurable identically in any dataset
#     (e.g. per-sample QC metrics like total_counts/n_genes) — the standard
#     leakage-safe fit-on-train/apply-to-test pattern, same as StandardScaler.

def split_covariate_names(raw_names):
    """Normalize a --train_only_covs/--train_test_covs argparse value (nargs='*')
    into a clean list of individual covariate names.

    Handles comma-separated, space-separated, and mixed input the same way:
    argparse's nargs='*' only splits on shell word boundaries, so a config
    value with no spaces (e.g. Nextflow's params.train_only_covs interpolated
    unquoted into the command line) arrives as ONE token containing every
    name joined by commas, not as separate tokens — this splits each token on
    commas too and strips whitespace, so "a,b,c", "a b c", and "a, b, c" all
    normalize to the same ["a", "b", "c"].
    """
    out = []
    for token in raw_names:
        out.extend(t.strip() for t in token.split(",") if t.strip())
    return out


def load_meta_file(meta_file, sample_col=None):
    """Load a per-sample metadata CSV, indexed by its first column (matching
    the convention already used for targets.csv elsewhere in this pipeline).
    sample_col is accepted only for a clearer error message if the file looks
    like it wasn't indexed as expected; the index itself is always column 0.
    """
    df_meta = pd.read_csv(meta_file, index_col=0)
    if df_meta.index.duplicated().any():
        dupes = df_meta.index[df_meta.index.duplicated()].unique().tolist()
        raise ValueError(f"meta_file has duplicate sample IDs in its first column: {dupes}")
    return df_meta


def validate_covariate_lists(df_meta, train_only_covs, train_test_covs):
    overlap = set(train_only_covs) & set(train_test_covs)
    if overlap:
        raise ValueError(
            f"Covariate(s) {sorted(overlap)} appear in both train_only_covs and "
            "train_test_covs — each covariate must be regressed out one way or the other, not both."
        )
    missing = [c for c in list(train_only_covs) + list(train_test_covs) if c not in df_meta.columns]
    if missing:
        raise ValueError(f"Covariate(s) {missing} not found in meta_file columns: "
                          f"{df_meta.columns.tolist()}")


def fit_covariate_design(df_meta, samples, cov_names):
    """Build a covariate design matrix for `samples`, fitting the encoding.

    Categorical vs continuous is inferred from dtype: non-numeric (e.g. str)
    columns are one-hot encoded (dropping one reference level; unseen
    categories at transform time become an all-zero row rather than an
    error); numeric columns are passed through unchanged.

    Returns (design_matrix, fitted_ColumnTransformer), or (None, None) if
    cov_names is empty.
    """
    if not cov_names:
        return None, None
    sub = df_meta.loc[samples, list(cov_names)]
    cat_cols  = [c for c in cov_names if not pd.api.types.is_numeric_dtype(sub[c])]
    cont_cols = [c for c in cov_names if c not in cat_cols]
    transformers = []
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols))
    ct = ColumnTransformer(transformers, remainder="passthrough" if cont_cols else "drop")
    design = np.asarray(ct.fit_transform(sub), dtype=float)
    if design.shape[1] >= design.shape[0]:
        print(f"WARNING: covariate design matrix for {cov_names} has "
              f"{design.shape[1]} columns for only {design.shape[0]} training "
              "samples — the fitted adjustment may be poorly determined.", flush=True)
    return design, ct


def transform_covariate_design(df_meta, samples, cov_names, fitted_ct):
    """Apply an already-fitted ColumnTransformer (from fit_covariate_design)
    to a different set of samples — e.g. a held-out fold, or genuinely new
    external samples at deployment time."""
    sub = df_meta.loc[samples, list(cov_names)]
    return np.asarray(fitted_ct.transform(sub), dtype=float)


def regress_out_covariates(X_tr, X_te, df_meta, samples_tr, samples_te,
                           train_only_covs, train_test_covs):
    """Remove technical-covariate effects from pseudobulk features X_tr/X_te
    (samples × features), fitting only on the training rows. See the module
    comment above section 2b for the train_only_covs vs train_test_covs
    distinction. Either list may be empty/None to skip that regime entirely.

    Sequential, not joint: train_only_covs is regressed out of X_tr first, then
    train_test_covs is regressed out of the (now train-only-adjusted) X_tr,
    with the same fitted model applied to X_te.

    Returns (X_tr_adj, X_te_adj, test_cov_model) — test_cov_model is the
    (fitted ColumnTransformer, fitted LinearRegression) pair for
    train_test_covs, needed by 006_finalize_model.py to persist and later
    reapply this adjustment to new external samples; None if train_test_covs
    is empty.
    """
    X_tr_adj, X_te_adj = X_tr, X_te

    if train_only_covs:
        design_tr, _ = fit_covariate_design(df_meta, samples_tr, train_only_covs)
        reg = LinearRegression().fit(design_tr, X_tr_adj)
        X_tr_adj = X_tr_adj - design_tr @ reg.coef_.T

    test_cov_model = None
    if train_test_covs:
        design_tr, ct = fit_covariate_design(df_meta, samples_tr, train_test_covs)
        reg = LinearRegression().fit(design_tr, X_tr_adj)
        X_tr_adj = X_tr_adj - design_tr @ reg.coef_.T
        design_te = transform_covariate_design(df_meta, samples_te, train_test_covs, ct)
        X_te_adj  = X_te_adj - design_te @ reg.coef_.T
        test_cov_model = (ct, reg)

    return X_tr_adj, X_te_adj, test_cov_model


# ─── 3. Tree models ───────────────────────────────────────────────────────────

def build_estimator(model_name, param_grid_path, seed, n_jobs=1, device="cpu"):
    import xgboost as xgb
    import json as json
    """Instantiate a base estimator and load its hyperparameter grid from JSON."""
    with open(param_grid_path) as f:
        grid = json.load(f)
    if model_name in ("xgboost_indep", "xgboost_joint"):
        xgb_kwargs = dict(
            objective   = "reg:squarederror",
            tree_method = "hist",
            random_state= seed,
            n_jobs      = n_jobs,
            verbosity   = 0,
        )
        if model_name == "xgboost_joint":
            # Native joint multi-output fit: ONE shared ensemble of trees per
            # cluster, each leaf holding a vector across all archetype targets
            # (requires XGBoost>=2.0 — see version check below). Upstream still
            # calls this strategy experimental with limited feature support
            # (no Tree SHAP, single shared feature_importances_ rather than
            # per-target) and only guarantees tree_method="hist" is feature
            # complete for it, so GPU is deliberately not wired up here
            # regardless of --device.
            major = int(xgb.__version__.split(".")[0])
            if major < 2:
                raise RuntimeError(
                    f"xgboost_joint requires XGBoost>=2.0.0 for "
                    f"multi_strategy='multi_output_tree' (found {xgb.__version__} "
                    "in this environment). Run xgboost_joint in the environment set "
                    "via params.xgboost2_env, not params.container_ml."
                )
            xgb_kwargs["multi_strategy"] = "multi_output_tree"
        elif device == "cuda":
            xgb_kwargs["tree_method"] = "gpu_hist"
            # Verify this XGBoost build was compiled with GPU support
            try:
                _probe = xgb.XGBRegressor(tree_method="gpu_hist", n_estimators=1, verbosity=0)
                _probe.fit(np.zeros((2, 2), dtype=np.float32), np.zeros(2, dtype=np.float32))
                del _probe
            except xgb.core.XGBoostError:
                import sys
                print(
                    "WARNING: XGBoost was not compiled with GPU support in this environment; "
                    "falling back to tree_method='hist' (CPU).",
                    file=sys.stderr, flush=True,
                )
                xgb_kwargs["tree_method"] = "hist"
        base = xgb.XGBRegressor(**xgb_kwargs)
    elif model_name == "rf":
        base = RandomForestRegressor(random_state=seed, n_jobs=n_jobs)
    elif model_name == "elasticnet":
        # Joint multi-task fit: one shared feature support across all archetype
        # targets, with a per-target coefficient matrix (see coef_ below).
        base = MultiTaskElasticNet(random_state=seed, max_iter=50000)
    elif model_name == "lasso":
        base = MultiTaskLasso(random_state=seed, max_iter=50000)
    elif model_name == "elasticnet_indep":
        # Independent single-output fit per target — no shared feature support
        # across targets (see run_nested_cv_independent). Same alpha/l1_ratio
        # grid as the joint "elasticnet" model.
        base = ElasticNet(random_state=seed, max_iter=50000)
    elif model_name == "lasso_indep":
        base = Lasso(random_state=seed, max_iter=50000)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return base, grid


def _fit_outer_fold(fold_i, tr_idx, te_idx, X, y, estimator, search_grid,
                    n_inner, n_iter, seed, n_jobs_inner, n_targets,
                    n_gene_features=None, hvg_pct=None, hvg_max=None,
                    samples=None, covariate_df=None,
                    train_only_covs=None, train_test_covs=None):
    """Run one outer CV fold: inner HPO search + refit + predict. Used by run_nested_cv_tree.

    If n_gene_features is given, HVGs are reselected here from X_tr's gene
    block ONLY (columns [:n_gene_features]) — fold-specific, so genes are
    never ranked using samples that end up in the held-out test fold. Columns
    at/after n_gene_features (abundance features) are always kept as-is.

    If covariate_df is given (with samples aligned to X's row order), technical
    covariates are regressed out BEFORE HVG selection — see regress_out_covariates
    and the section 2b module comment for the train_only_covs/train_test_covs
    distinction. Fit-and-discarded per fold, same lifecycle as the scaler below.
    """
    X_tr_full, X_te_full = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    if covariate_df is not None and (train_only_covs or train_test_covs):
        samples_tr = [samples[i] for i in tr_idx]
        samples_te = [samples[i] for i in te_idx]
        X_tr_full, X_te_full, _ = regress_out_covariates(
            X_tr_full, X_te_full, covariate_df, samples_tr, samples_te,
            train_only_covs, train_test_covs,
        )

    if n_gene_features is not None:
        hvg_idx_local = select_hvgs(X_tr_full[:, :n_gene_features], hvg_pct, hvg_max)
        abund_idx     = np.arange(n_gene_features, X_tr_full.shape[1])
        sel_idx       = np.concatenate([hvg_idx_local, abund_idx])
    else:
        sel_idx = np.arange(X_tr_full.shape[1])

    X_tr, X_te = X_tr_full[:, sel_idx], X_te_full[:, sel_idx]

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

    inner_cv = KFold(n_splits=n_inner, shuffle=True, random_state=seed + 1)
    search   = RandomizedSearchCV(
        clone(estimator),
        search_grid,
        n_iter      = n_iter,
        cv          = inner_cv,
        scoring     = VARIANCE_WEIGHTED_R2_SCORER,
        random_state= seed,
        n_jobs      = n_jobs_inner,
        refit       = True,
        error_score = "raise",
    )
    search.fit(X_tr_sc, y_tr if n_targets > 1 else y_tr.ravel())
    best = search.best_estimator_

    y_pred = best.predict(X_te_sc)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    r2   = r2_score(y_te, y_pred, multioutput='variance_weighted')
    rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
    metrics = {"fold": fold_i, "r2_mean": r2, "rmse_mean": rmse,
               "best_params": str(search.best_params_)}
    print(f"    fold {fold_i+1}: R²={r2:.3f}  RMSE={rmse:.4f}", flush=True)

    if hasattr(best, "estimators_"):
        imp = np.mean([e.feature_importances_ for e in best.estimators_], axis=0)
    elif hasattr(best, "feature_importances_"):
        imp = best.feature_importances_
    elif hasattr(best, "coef_"):
        # Linear models: coef_ is (n_features,) for single-target, or
        # (n_targets, n_features) for MultiTask*  — kept per-target so that
        # signed, per-archetype coefficients survive averaging across folds.
        imp = best.coef_
    else:
        imp = None

    return fold_i, te_idx, y_pred, metrics, imp, sel_idx


def run_nested_cv_tree(X, y, model_name, base, grid,
                       n_outer, n_inner, n_iter, seed, feature_names,
                       n_jobs_outer=1, n_jobs_inner=1,
                       n_gene_features=None, hvg_pct=None, hvg_max=None,
                       samples=None, covariate_df=None,
                       train_only_covs=None, train_test_covs=None):
    """Nested CV for XGBoost (indep/joint) / RF / ElasticNet / Lasso with RandomizedSearchCV inner loop.

    n_jobs_outer : parallel outer folds (set to n_outer for max parallelism;
                   use n_jobs_inner=1 and estimator n_jobs=1 to avoid oversubscription)
    n_jobs_inner : n_jobs passed to RandomizedSearchCV for parallel HPO iterations
    n_gene_features, hvg_pct, hvg_max : if n_gene_features is given, HVGs are
        reselected per outer fold from that fold's training samples only (see
        _fit_outer_fold) instead of using a fixed pre-selected panel.
    samples, covariate_df, train_only_covs, train_test_covs : if covariate_df
        is given (samples aligned to X's row order), technical covariates are
        regressed out per outer fold before HVG selection — see
        regress_out_covariates and the section 2b module comment.
    Returns fold_metrics list, oof_preds (n_samples × n_targets), and all_imps
    — a list of (sel_idx, imp) tuples, one per outer fold, since fold-specific
    HVG reselection means different folds may keep different gene columns.
    """
    outer_cv  = KFold(n_splits=n_outer, shuffle=True, random_state=seed)
    n_targets = y.shape[1]

    # Wrap XGBoost in MultiOutputRegressor for multi-target consistency — only
    # "xgboost_indep" needs this. "xgboost_joint" natively accepts 2D y
    # (multi_strategy="multi_output_tree", set in build_estimator), same as
    # rf and the MultiTask* linear models below.
    if n_targets > 1 and model_name == "xgboost_indep":
        estimator   = MultiOutputRegressor(base, n_jobs=1)
        search_grid = {f"estimator__{k}": v for k, v in grid.items()}
    else:
        estimator   = base
        search_grid = grid

    results = Parallel(n_jobs=n_jobs_outer)(
        delayed(_fit_outer_fold)(
            fold_i, tr_idx, te_idx, X, y, estimator, search_grid,
            n_inner, n_iter, seed, n_jobs_inner, n_targets,
            n_gene_features=n_gene_features, hvg_pct=hvg_pct, hvg_max=hvg_max,
            samples=samples, covariate_df=covariate_df,
            train_only_covs=train_only_covs, train_test_covs=train_test_covs,
        )
        for fold_i, (tr_idx, te_idx) in enumerate(outer_cv.split(X))
    )

    oof_preds    = np.full_like(y, np.nan, dtype=float)
    fold_metrics = []
    all_imps     = []

    for fold_i, te_idx, y_pred, metrics, imp, sel_idx in sorted(results, key=lambda r: r[0]):
        oof_preds[te_idx] = y_pred
        fold_metrics.append(metrics)
        if imp is not None:
            all_imps.append((sel_idx, imp))

    return fold_metrics, oof_preds, all_imps


def run_fixed_cv(X, y, model_name, best_params, param_grid_path, n_folds, seed,
                 n_gene_features=None, hvg_pct=None, hvg_max=None, per_target_fit=None,
                 samples=None, covariate_df=None, train_only_covs=None, train_test_covs=None):
    """Plain (non-nested) K-fold CV with FIXED hyperparameters — no inner search.

    Used for permutation-null testing (007_permutation_null.py): the expensive
    part of nested CV (the RandomizedSearchCV inner loop) is replaced by
    directly building an estimator with the given best_params. HVGs are still
    reselected fresh per fold from that fold's training data only (see
    select_hvgs / _fit_outer_fold), exactly as in the real nested-CV pipeline
    — only the hyperparameter *search* is skipped, not the feature reselection.

    best_params : dict, applied to every target — OR a list of dicts, one per
        target column, in the same order as y's columns (independent-search
        models: elasticnet_indep/lasso_indep).
    per_target_fit : if True, fit one independent estimator per target column
        even though best_params is a single SHARED dict — this is
        xgboost_indep's case: its hyperparameter *search* is joint (one shared
        setting picked via the aggregate variance-weighted score, so
        best_params is a plain dict, same as elasticnet/lasso/rf), but its
        *fitting* is always independent per target regardless, because the
        real pipeline wraps it in MultiOutputRegressor (run_nested_cv_tree) —
        plain XGBRegressor doesn't support multi-output y directly without
        XGBoost>=2.0's multi_strategy (see xgboost_joint in build_estimator).
        Defaults to isinstance(best_params, list) if not given, which
        preserves existing elasticnet_indep/lasso_indep/joint behavior exactly.
    param_grid_path : passed through to build_estimator purely to construct
        the base estimator object; its grid contents are irrelevant here since
        hyperparameters are immediately overridden via set_params().
    samples, covariate_df, train_only_covs, train_test_covs : same covariate
        regression as run_nested_cv_tree/_fit_outer_fold (see the section 2b
        module comment) — refit fresh every call, same as HVG reselection,
        since 007_permutation_null.py calls this once per permutation and the
        adjustment is part of the procedure being tested, not held fixed.

    Returns oof_preds (n_samples × n_targets).
    """
    n_targets = y.shape[1]
    outer_cv  = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_preds = np.full_like(y, np.nan, dtype=float)

    if per_target_fit is None:
        per_target_fit = isinstance(best_params, list)

    for tr_idx, te_idx in outer_cv.split(X):
        X_tr_full, X_te_full = X[tr_idx], X[te_idx]
        y_tr = y[tr_idx]

        if covariate_df is not None and (train_only_covs or train_test_covs):
            samples_tr = [samples[i] for i in tr_idx]
            samples_te = [samples[i] for i in te_idx]
            X_tr_full, X_te_full, _ = regress_out_covariates(
                X_tr_full, X_te_full, covariate_df, samples_tr, samples_te,
                train_only_covs, train_test_covs,
            )

        if n_gene_features is not None:
            hvg_idx_local = select_hvgs(X_tr_full[:, :n_gene_features], hvg_pct, hvg_max)
            abund_idx     = np.arange(n_gene_features, X_tr_full.shape[1])
            sel_idx       = np.concatenate([hvg_idx_local, abund_idx])
        else:
            sel_idx = np.arange(X_tr_full.shape[1])

        X_tr, X_te = X_tr_full[:, sel_idx], X_te_full[:, sel_idx]

        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        if per_target_fit:
            for j in range(n_targets):
                base, _   = build_estimator(model_name, param_grid_path, seed)
                params_j  = best_params[j] if isinstance(best_params, list) else best_params
                base.set_params(**params_j)
                base.fit(X_tr_sc, y_tr[:, j])
                oof_preds[te_idx, j] = base.predict(X_te_sc)
        else:
            base, _ = build_estimator(model_name, param_grid_path, seed)
            base.set_params(**best_params)
            y_fit = y_tr if n_targets > 1 else y_tr.ravel()
            base.fit(X_tr_sc, y_fit)
            y_pred = base.predict(X_te_sc)
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)
            oof_preds[te_idx] = y_pred

    return oof_preds


def combine_fold_importances(all_imps, n_total_features):
    """Combine per-fold (sel_idx, imp) importances into full-width arrays.

    Fold-specific HVG reselection means different folds may have kept
    different gene columns, so each fold's importance vector is scattered
    into a full (n_total_features,) or (n_targets, n_total_features) slot
    before averaging with a NaN-aware mean over folds that selected each
    feature (abundance features are selected by every fold; genes may not be).

    Returns (mean_imp, n_folds_selected) — n_folds_selected is a 1D
    (n_total_features,) count of how many folds included each feature,
    independent of the target axis (HVG selection doesn't depend on y).
    """
    if not all_imps:
        return None, None
    first_imp = all_imps[0][1]
    n_folds   = len(all_imps)

    if first_imp.ndim == 1:
        acc = np.full((n_folds, n_total_features), np.nan)
        for i, (sel_idx, imp) in enumerate(all_imps):
            acc[i, sel_idx] = imp
        n_folds_selected = (~np.isnan(acc)).sum(axis=0)
    else:
        n_targets = first_imp.shape[0]
        acc = np.full((n_folds, n_targets, n_total_features), np.nan)
        for i, (sel_idx, imp) in enumerate(all_imps):
            acc[i][:, sel_idx] = imp
        n_folds_selected = (~np.isnan(acc[:, 0, :])).sum(axis=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slice for never-selected genes
        mean_imp = np.nanmean(acc, axis=0)

    return mean_imp, n_folds_selected


def run_nested_cv_independent(X, y, model_name, base, grid,
                              n_outer, n_inner, n_iter, seed, feature_names,
                              target_names=None, n_jobs_outer=1, n_jobs_inner=1,
                              n_gene_features=None, hvg_pct=None, hvg_max=None,
                              samples=None, covariate_df=None,
                              train_only_covs=None, train_test_covs=None):
    """Fit one independent single-output model per target column, instead of a
    joint multi-task fit — no shared feature support is enforced across targets.

    Reuses run_nested_cv_tree once per column. KFold splitting (and, if
    n_gene_features is given, the per-fold HVG reselection, and any covariate
    regression) in run_nested_cv_tree depends only on X, `seed`, samples, and
    covariate_df (not y), so outer-fold membership AND each fold's selected
    gene columns/covariate adjustment are identical across targets — letting
    per-fold importances recombine into the same (n_targets, n_features) shape
    the joint MultiTaskElasticNet/MultiTaskLasso models produce.

    Returns fold_metrics (each dict tagged with 'archetype'), oof_preds
    (n_samples × n_targets), and all_imps — a list of (sel_idx, imp) tuples,
    one per outer fold, matching run_nested_cv_tree's contract.
    """
    n_targets = y.shape[1]
    if target_names is None:
        target_names = [str(j) for j in range(n_targets)]

    oof_preds       = np.full_like(y, np.nan, dtype=float)
    fold_metrics    = []
    per_target_imps = []

    for j in range(n_targets):
        fm_j, oof_j, imps_j = run_nested_cv_tree(
            X, y[:, j:j + 1], model_name, base, grid,
            n_outer, n_inner, n_iter, seed, feature_names,
            n_jobs_outer=n_jobs_outer, n_jobs_inner=n_jobs_inner,
            n_gene_features=n_gene_features, hvg_pct=hvg_pct, hvg_max=hvg_max,
            samples=samples, covariate_df=covariate_df,
            train_only_covs=train_only_covs, train_test_covs=train_test_covs,
        )
        for m in fm_j:
            m["archetype"] = target_names[j]
        fold_metrics.extend(fm_j)
        oof_preds[:, j] = oof_j.ravel()
        per_target_imps.append(imps_j)  # list of (sel_idx, imp) per outer fold

    all_imps = []
    if per_target_imps and all(len(imps) == n_outer for imps in per_target_imps):
        for fold_i in range(n_outer):
            sel_idx_0, _ = per_target_imps[0][fold_i]
            stacked = np.vstack([per_target_imps[j][fold_i][1].ravel() for j in range(n_targets)])
            all_imps.append((sel_idx_0, stacked))

    return fold_metrics, oof_preds, all_imps


# ─── 4. MIL ───────────────────────────────────────────────────────────────────

if HAS_TORCH:
    class AttentionMIL(nn.Module):
        """Gated attention-pooling MIL (Ilse et al., 2018) with simplex output."""

        def __init__(self, input_dim, hidden_dim, attention_dim, n_outputs, dropout):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.att_V = nn.Linear(hidden_dim, attention_dim)
            self.att_U = nn.Linear(hidden_dim, attention_dim)
            self.att_w = nn.Linear(attention_dim, 1, bias=False)
            self.predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, n_outputs),
                nn.Softmax(dim=-1),
            )

        def forward(self, x):
            """x: (n_cells, input_dim) → (n_outputs,)"""
            h = self.encoder(x)
            A = torch.tanh(self.att_V(h)) * torch.sigmoid(self.att_U(h))
            A = torch.softmax(self.att_w(A), dim=0)
            z = (A * h).sum(dim=0, keepdim=True)
            return self.predictor(z).squeeze(0)

    class MILDataset(Dataset):
        def __init__(self, bags, targets, max_cells, scaler=None, seed=0):
            self.bags      = bags
            self.targets   = targets
            self.max_cells = max_cells
            self.scaler    = scaler
            self.seed      = seed

        def __len__(self):
            return len(self.bags)

        def __getitem__(self, idx):
            x = self.bags[idx].astype(np.float32)
            if self.scaler is not None:
                x = self.scaler.transform(x)
            if x.shape[0] > self.max_cells:
                rng  = np.random.default_rng(self.seed + idx)
                cidx = rng.choice(x.shape[0], self.max_cells, replace=False)
                x    = x[cidx]
            return torch.tensor(x), torch.tensor(self.targets[idx].astype(np.float32))

    def mil_collate(batch):
        bags, targets = zip(*batch)
        return list(bags), torch.stack(targets)

    def train_epoch(model, loader, optimizer, device):
        model.train()
        total = 0.0
        for bags, targets in loader:
            targets = targets.to(device)
            optimizer.zero_grad()
            preds = torch.stack([model(b.to(device)) for b in bags])
            loss  = nn.functional.mse_loss(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item()
        return total / len(loader)

    @torch.no_grad()
    def evaluate_bags(model, bags, y, device, max_cells, scaler, seed=0):
        model.eval()
        preds = []
        for i, bag in enumerate(bags):
            x = bag.astype(np.float32)
            if scaler is not None:
                x = scaler.transform(x)
            if x.shape[0] > max_cells:
                rng  = np.random.default_rng(seed + i)
                cidx = rng.choice(x.shape[0], max_cells, replace=False)
                x    = x[cidx]
            p = model(torch.tensor(x).to(device)).cpu().numpy()
            preds.append(p)
        preds = np.array(preds)
        r2   = r2_score(y, preds, multioutput='variance_weighted')
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        return preds, r2, rmse

    def build_mil_model(input_dim, hp, n_outputs, seed, device):
        torch.manual_seed(seed)
        return AttentionMIL(
            input_dim    = input_dim,
            hidden_dim   = hp["hidden_dim"],
            attention_dim= hp["attention_dim"],
            n_outputs    = n_outputs,
            dropout      = hp["dropout"],
        ).to(device)

    def fit_mil_model(model, bags_train, y_train, hp, max_cells, scaler, device, seed):
        ds        = MILDataset(bags_train, y_train, max_cells, scaler, seed)
        dl        = DataLoader(ds, batch_size=hp["batch_size"],
                               shuffle=True, collate_fn=mil_collate)
        opt       = torch.optim.Adam(model.parameters(),
                                     lr=hp["lr"],
                                     weight_decay=hp.get("weight_decay", 1e-4))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["epochs"])
        for _ in range(hp["epochs"]):
            train_epoch(model, dl, opt, device)
            scheduler.step()

    def nested_cv_mil(bags, y, hp_grid, n_outer, n_inner, n_iter, max_cells, seed, device,
                      n_gene_features=None, hvg_pct=None, hvg_max=None, pseudobulk_agg="mean"):
        """Nested CV for attention-based MIL with random hyperparameter search.

        n_gene_features, hvg_pct, hvg_max : if n_gene_features is given, HVGs
            are reselected per outer fold from that fold's training samples
            only — each training bag's cells are first pseudobulked
            (aggregated via pseudobulk_agg), then ranked by variance across
            those per-sample pseudobulk profiles (columns [:n_gene_features]
            only), the same statistic used for the tree/linear models. This
            avoids ranking genes using cells from samples that end up in the
            held-out test fold. Columns at/after n_gene_features (abundance
            features) are always kept as-is.
        """
        n_outputs = y.shape[1]

        outer_cv = KFold(n_splits=n_outer, shuffle=True, random_state=seed)
        inner_cv = KFold(n_splits=n_inner, shuffle=True, random_state=seed + 1)

        all_combos = list(itertools.product(*hp_grid.values()))
        rng        = np.random.default_rng(seed)
        rng.shuffle(all_combos)
        combos  = all_combos[:n_iter]
        hp_keys = list(hp_grid.keys())

        oof_preds    = np.full_like(y, np.nan, dtype=float)
        fold_metrics = []

        for fold_i, (tr_idx, te_idx) in enumerate(outer_cv.split(bags)):
            print(f"  Outer fold {fold_i+1}/{n_outer}")
            bags_tr_full = [bags[i] for i in tr_idx]
            bags_te_full = [bags[i] for i in te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            if n_gene_features is not None:
                agg_fn = np.mean if pseudobulk_agg == "mean" else np.sum
                pb_tr  = np.array(
                    [agg_fn(b[:, :n_gene_features], axis=0) for b in bags_tr_full],
                    dtype=np.float32,
                )
                hvg_idx_local = select_hvgs(pb_tr, hvg_pct, hvg_max)
                abund_idx     = np.arange(n_gene_features, bags_tr_full[0].shape[1])
                sel_idx       = np.concatenate([hvg_idx_local, abund_idx])
                bags_tr = [b[:, sel_idx] for b in bags_tr_full]
                bags_te = [b[:, sel_idx] for b in bags_te_full]
            else:
                bags_tr, bags_te = bags_tr_full, bags_te_full

            input_dim = bags_tr[0].shape[1]

            scaler = StandardScaler()
            scaler.fit(np.vstack(bags_tr))

            best_val_r2 = -np.inf
            best_hp     = None

            for combo in combos:
                hp      = dict(zip(hp_keys, combo))
                val_r2s = []
                for _, (in_tr, in_val) in enumerate(inner_cv.split(bags_tr)):
                    bt = [bags_tr[i] for i in in_tr]
                    bv = [bags_tr[i] for i in in_val]
                    yt = y_tr[in_tr]
                    yv = y_tr[in_val]
                    m  = build_mil_model(input_dim, hp, n_outputs, seed, device)
                    fit_mil_model(m, bt, yt, hp, max_cells, scaler, device, seed)
                    _, vr2, _ = evaluate_bags(m, bv, yv, device, max_cells, scaler, seed)
                    val_r2s.append(vr2)
                    del m

                mean_vr2 = float(np.mean(val_r2s))
                if mean_vr2 > best_val_r2:
                    best_val_r2 = mean_vr2
                    best_hp     = hp

            print(f"    Best inner R²={best_val_r2:.3f}  hp={best_hp}")

            final = build_mil_model(input_dim, best_hp, n_outputs, seed, device)
            fit_mil_model(final, bags_tr, y_tr, best_hp, max_cells, scaler, device, seed)
            preds, r2, rmse = evaluate_bags(final, bags_te, y_te, device,
                                            max_cells, scaler, seed)
            oof_preds[te_idx] = preds
            del final

            fold_metrics.append({
                "fold": fold_i, "r2_mean": r2, "rmse_mean": rmse,
                "best_params": str(best_hp),
            })
            print(f"    Outer R²={r2:.3f}  RMSE={rmse:.4f}")

        return fold_metrics, oof_preds


# ─── 5. Evaluation I/O ────────────────────────────────────────────────────────

def load_oof(directory, prefix):
    """Load out-of-fold predictions CSV, or return None if absent."""
    import os
    path = os.path.join(directory, f"{prefix}_oof_predictions.csv")
    return pd.read_csv(path, index_col=0) if os.path.exists(path) else None


def per_archetype_metrics(y, oof_preds, arch_cols, model_name, cluster, label_mode="full"):
    """Return a list of dicts with per-archetype OOF R² and RMSE."""
    rows = []
    for j, col in enumerate(arch_cols):
        mask = ~np.isnan(oof_preds[:, j])
        r2_c   = r2_score(y[mask, j], oof_preds[mask, j])
        rmse_c = float(np.sqrt(mean_squared_error(y[mask, j], oof_preds[mask, j])))
        rows.append({"model": model_name, "cluster": cluster,
                     "archetype": col, "r2": r2_c, "rmse": rmse_c,
                     "label_mode": label_mode})
    return rows


def overall_metrics(y, oof_preds, model_name, cluster, label_mode="full"):
    """Multivariate accuracy across ALL targets at once, using their native
    (unstandardized) scale — for CLR-PCA targets this equals the Aitchison
    distance-based accuracy, since PCA preserves Euclidean distance.

    r2_variance_weighted : 1 - (total SS residual across all targets) /
                            (total SS from the mean across all targets) —
                            equivalent to sklearn's
                            r2_score(multioutput='variance_weighted').
    rmse_aggregate : sqrt(mean per-sample squared Euclidean distance across
                          all target columns) — i.e. RMS of the total
                          multi-target miss distance per sample, NOT an
                          average of per-column RMSE values.

    Returns a single dict (one aggregate row for the whole
    model/cluster/label_mode combination), not a per-archetype list.
    """
    mask = ~np.isnan(oof_preds).any(axis=1)
    y_m, pred_m = y[mask], oof_preds[mask]
    r2_vw    = r2_score(y_m, pred_m, multioutput='variance_weighted')
    rmse_agg = float(np.sqrt(np.mean(np.sum((y_m - pred_m) ** 2, axis=1))))
    return {"model": model_name, "cluster": cluster,
            "r2_variance_weighted": r2_vw, "rmse_aggregate": rmse_agg,
            "label_mode": label_mode}
