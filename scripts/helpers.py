"""
helpers.py
Shared utility functions and classes for the archetype classification pipeline.

Sections:
  1. Data I/O          — load_pseudobulk, load_mil_bags
  2. Feature prep      — filter_lowly_expressed_genes, select_hvgs, pseudobulk_aggregate,
                         compute_abundance, get_archetype_weight_cols
  3. Tree models       — build_estimator, run_nested_cv_tree
  4. MIL               — AttentionMIL, MILDataset, mil_collate, train_epoch,
                         evaluate_bags, build_mil_model, fit_mil_model,
                         nested_cv_mil
  5. Evaluation I/O    — load_oof, per_archetype_metrics
"""

import itertools
import json
import re
import warnings

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
# import xgboost as xgb

warnings.filterwarnings("ignore")

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

    Returns X (n_samples × n_features dense float32), sample list, feature list.
    """
    d        = np.load(path, allow_pickle=True)
    X        = sp.csr_matrix(
                   (d['data'], d['indices'], d['indptr']),
                   shape=tuple(d['shape'])
               ).toarray()
    samples  = d['samples'].tolist()
    features = d['features'].tolist()
    return X, samples, features


def load_mil_bags(path):
    """Load MIL bags HDF5 written by 001_prepare_data.py.

    Returns list of np.arrays (one per sample) and corresponding sample list.
    """
    bags, samples = [], []
    with h5py.File(path, "r") as hf:
        all_samples = list(hf.attrs['samples'])
        for s in all_samples:
            safe_key = s.replace("/", "_")
            if safe_key in hf:
                bags.append(hf[safe_key][:])
                samples.append(s)
    return bags, samples


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


def compute_abundance(adata_full, adata_cl, sample_col, celltype_col,
                      abundance_type, samples):
    """Cell-type frequency features per sample.

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
        ct_gl = (adata_full.obs
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


def get_archetype_weight_cols(df_scores, cluster_prefix):
    """Return raw simplex-weight columns for a cluster (e.g. 'A' → ['A1','A2'])."""
    pat = re.compile(rf'^{re.escape(cluster_prefix)}\d+$')
    return [c for c in df_scores.columns if pat.match(c)]


# ─── 3. Tree models ───────────────────────────────────────────────────────────

def build_estimator(model_name, param_grid_path, seed):
    """Instantiate a base estimator and load its hyperparameter grid from JSON."""
    with open(param_grid_path) as f:
        grid = json.load(f)
    if model_name == "xgboost":
        base = xgb.XGBRegressor(
            objective   = "reg:squarederror",
            tree_method = "hist",
            random_state= seed,
            n_jobs      = 4,
            verbosity   = 0,
        )
    elif model_name == "rf":
        base = RandomForestRegressor(random_state=seed, n_jobs=4)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return base, grid


def run_nested_cv_tree(X, y, model_name, base, grid,
                       n_outer, n_inner, n_iter, seed, feature_names):
    """Nested CV for XGBoost / RF with RandomizedSearchCV inner loop.

    Returns fold_metrics list, oof_preds (n_samples × n_targets), importances list.
    """
    outer_cv  = KFold(n_splits=n_outer, shuffle=True, random_state=seed)
    inner_cv  = KFold(n_splits=n_inner, shuffle=True, random_state=seed + 1)
    n_targets = y.shape[1]

    oof_preds    = np.full_like(y, np.nan, dtype=float)
    fold_metrics = []
    all_imps     = []

    # Wrap XGBoost in MultiOutputRegressor for multi-target consistency
    if n_targets > 1 and model_name == "xgboost":
        estimator   = MultiOutputRegressor(base, n_jobs=1)
        search_grid = {f"estimator__{k}": v for k, v in grid.items()}
    else:
        estimator   = base
        search_grid = grid

    for fold_i, (tr_idx, te_idx) in enumerate(outer_cv.split(X)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        search = RandomizedSearchCV(
            estimator,
            search_grid,
            n_iter      = n_iter,
            cv          = inner_cv,
            scoring     = "r2",
            random_state= seed,
            n_jobs      = 1,
            refit       = True,
            error_score = "raise",
        )
        search.fit(X_tr_sc, y_tr if n_targets > 1 else y_tr.ravel())
        best = search.best_estimator_

        y_pred = best.predict(X_te_sc)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        oof_preds[te_idx] = y_pred

        r2   = r2_score(y_te, y_pred, multioutput='uniform_average')
        rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
        fold_metrics.append({
            "fold":        fold_i,
            "r2_mean":     r2,
            "rmse_mean":   rmse,
            "best_params": str(search.best_params_),
        })
        print(f"    fold {fold_i+1}: R²={r2:.3f}  RMSE={rmse:.4f}")

        if hasattr(best, "estimators_"):
            imp = np.mean([e.feature_importances_ for e in best.estimators_], axis=0)
        elif hasattr(best, "feature_importances_"):
            imp = best.feature_importances_
        else:
            imp = None
        if imp is not None:
            all_imps.append(imp)

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
        r2   = r2_score(y, preds, multioutput='uniform_average')
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

    def nested_cv_mil(bags, y, hp_grid, n_outer, n_inner, n_iter, max_cells, seed, device):
        """Nested CV for attention-based MIL with random hyperparameter search."""
        input_dim = bags[0].shape[1]
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
            bags_tr    = [bags[i] for i in tr_idx]
            bags_te    = [bags[i] for i in te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

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


def per_archetype_metrics(y, oof_preds, arch_cols, model_name, cluster):
    """Return a list of dicts with per-archetype OOF R² and RMSE."""
    rows = []
    for j, col in enumerate(arch_cols):
        mask = ~np.isnan(oof_preds[:, j])
        r2_c   = r2_score(y[mask, j], oof_preds[mask, j])
        rmse_c = float(np.sqrt(mean_squared_error(y[mask, j], oof_preds[mask, j])))
        rows.append({"model": model_name, "cluster": cluster,
                     "archetype": col, "r2": r2_c, "rmse": rmse_c})
    return rows
