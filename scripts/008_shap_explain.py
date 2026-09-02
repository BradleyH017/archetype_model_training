#!/usr/bin/env python3
"""
008_shap_explain.py
Manual, standalone script (like 006_finalize_model.py — not part of the
Nextflow DAG): computes TreeSHAP feature attributions for an already-
finalized model bundle (results/final_models/*.joblib, from
006_finalize_model.py).

Only meaningful for tree-based models:
  xgboost_indep — bundle["estimators"] is a list of independently-fit, plain
                  single-output XGBRegressors; TreeSHAP applies directly.
  rf            — bundle["estimator"] is one shared RandomForestRegressor
                  with native multi-output support; TreeSHAP-eligible too,
                  and can give genuinely per-archetype attribution where
                  rf's own feature_importances_ collapses to one vector
                  shared across all targets.
NOT xgboost_joint (upstream has no Tree SHAP support at all for
multi_strategy="multi_output_tree") and NOT elasticnet/lasso/
elasticnet_indep/lasso_indep (a linear model's own coefficient is already
the exact attribution — SHAP has nothing to add there). This script refuses
to run on any other model name.

IMPORTANT — what "SHAP value" actually means here: for each archetype
target, SHAP produces ONE SIGNED VALUE PER SAMPLE PER GENE — a full
(n_samples x n_features) attribution matrix, not a single global number per
gene. This script saves that full matrix (one per target, in the .npz), a
mean(|SHAP|)-per-gene summary table shaped like {model}_feature_importances.csv
for quick comparison against the model's own gain-based importances, and a
SIGNED mean(SHAP)-per-gene summary (mean(|SHAP|) with the abs() removed is
NOT the same thing -- it collapses direction away by construction) -- but
the per-sample matrix is the richer artifact. Keep it if you want beeswarm/
dependence plots, or to check whether a gene's effect direction differs
across sample subgroups, which neither summary number can tell you.

Also writes {cluster}_gene_universe.txt: the full prevalence-filtered
candidate gene set for this cluster (NOT restricted to the HVG panel the
model was fit on) -- the GO over-representation universe for
009_go_enrichment.R, which also consumes both summary CSVs above (signed
mean(SHAP) > 0 for its signed test; unsigned mean(|SHAP|) > 0 for its
unsigned one).

Usage:
    python scripts/008_shap_explain.py \\
        --bundle results/final_models/B_xgboost_indep_full_final_model.joblib \\
        --pseudobulk results/prepared_data/B.pseudobulk.npz \\
        --meta_file /path/to/meta_df.csv \\
        --outdir results/final_models/shap

--meta_file is only required if the bundle used train_test_covs (i.e.
bundle["covariate_model"] is not None) — 006_finalize_model.py records this
regardless, so check that first if you're unsure.
"""
import argparse
import os
import sys
import xgboost

import joblib
import numpy as np
import pandas as pd
sys.path.append("pip_packages")
import shap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import load_meta_file, load_pseudobulk, transform_covariate_design

_TREE_SHAP_MODELS = ("xgboost_indep", "rf")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle",     required=True,
                   help="Path to a *_final_model.joblib from 006_finalize_model.py")
    p.add_argument("--pseudobulk", required=True,
                   help="results/prepared_data/{cluster}.pseudobulk.npz — must match the "
                        "bundle's own cluster")
    p.add_argument("--meta_file",  default=None,
                   help="Required only if bundle['covariate_model'] is not None "
                        "(i.e. train_test_covs was used when the bundle was created)")
    p.add_argument("--outdir",     required=True)
    return p.parse_args()


def build_input_matrix(bundle, pseudobulk_path, meta_file):
    """Reconstruct the exact matrix the bundle's model was fit on: subset/
    reorder to bundle['feature_names'], reapply bundle['covariate_model'] if
    present (never refit — reuses the exact fitted correction 006 already
    computed, same as scoring a genuinely new sample would), then
    bundle['scaler'].transform().

    Also returns features/n_gene_features from this same load_pseudobulk call
    (not just the HVG-restricted subset) -- features[:n_gene_features] is the
    prevalence-filtered candidate gene set for this cluster, used as the GO
    over-representation universe in 009_go_enrichment.R. Deliberately NOT
    restricted to bundle['feature_names'] (the HVG panel): HVG selection here
    is itself target-informed (sq_target_vs_technical), not a neutral
    detection filter, so treating "not HVG-selected" as "outside the
    universe" would compound two target-informed filtering steps into one
    enrichment number and throw away real signal from genes the panel cap
    excluded.
    """
    X, samples, features, n_gene_features = load_pseudobulk(pseudobulk_path)
    feature_idx = {f: i for i, f in enumerate(features)}
    missing = [f for f in bundle["feature_names"] if f not in feature_idx]
    if missing:
        raise ValueError(f"{len(missing)} of the bundle's feature_names are not present in "
                          f"this pseudobulk file (first few: {missing[:5]}) — wrong cluster/file?")
    sel_idx = [feature_idx[f] for f in bundle["feature_names"]]
    X = X[:, sel_idx]
    if bundle.get("covariate_model") is not None:
        if not meta_file:
            raise ValueError("bundle['covariate_model'] is set (train_test_covs was used) — "
                              "--meta_file is required to reapply it.")
        ct, reg = bundle["covariate_model"]
        df_meta = load_meta_file(meta_file)
        design = transform_covariate_design(df_meta, samples, bundle["train_test_covs"], ct)
        X = X - design @ reg.coef_.T
    X_scaled = bundle["scaler"].transform(X)
    return X_scaled, samples, features, n_gene_features


def explain_indep(bundle, X_scaled):
    """xgboost_indep: one TreeSHAP call per target's independently-fit,
    plain single-output XGBRegressor."""
    shap_by_target = {}
    for target, est in zip(bundle["target_names"], bundle["estimators"]):
        explainer = shap.TreeExplainer(est)
        shap_by_target[target] = explainer.shap_values(X_scaled)
    return shap_by_target


def explain_rf(bundle, X_scaled):
    """rf: one shared RandomForestRegressor with native multi-output support.
    shap.TreeExplainer handles multi-output natively too, but the exact
    return shape has varied across shap versions -- handle a list-of-arrays
    or a stacked 3D array either way rather than assuming one."""
    explainer = shap.TreeExplainer(bundle["estimator"])
    raw = explainer.shap_values(X_scaled)
    n_targets = len(bundle["target_names"])
    shap_by_target = {}
    if isinstance(raw, list):
        for target, arr in zip(bundle["target_names"], raw):
            shap_by_target[target] = arr
    else:
        raw = np.asarray(raw)
        if raw.ndim == 3 and raw.shape[-1] == n_targets:
            for j, target in enumerate(bundle["target_names"]):
                shap_by_target[target] = raw[:, :, j]
        elif raw.ndim == 3 and raw.shape[0] == n_targets:
            for j, target in enumerate(bundle["target_names"]):
                shap_by_target[target] = raw[j]
        else:
            raise ValueError(f"Unexpected shap_values() return shape {raw.shape} for "
                              f"{n_targets} targets -- check this shap version's multi-output "
                              f"convention by hand.")
    return shap_by_target


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    bundle = joblib.load(args.bundle)
    model_name = bundle["model_name"]
    if model_name not in _TREE_SHAP_MODELS:
        raise ValueError(
            f"--bundle is a '{model_name}' model. TreeSHAP is only meaningful for "
            f"{_TREE_SHAP_MODELS} here — xgboost_joint has no upstream Tree SHAP support for "
            "multi_output_tree, and elasticnet/lasso/elasticnet_indep/lasso_indep's own "
            "coefficients are already the exact attribution, so SHAP has nothing to add there."
        )

    print(f"Loading bundle: cluster={bundle['cluster']}  model={model_name}  "
          f"label_mode={bundle['label_mode']}")
    X_scaled, samples, features, n_gene_features = build_input_matrix(
        bundle, args.pseudobulk, args.meta_file)
    print(f"Input matrix: {X_scaled.shape}  ({len(bundle['feature_names'])} features, "
          f"{len(samples)} samples)")

    if model_name == "xgboost_indep":
        shap_by_target = explain_indep(bundle, X_scaled)
    else:
        shap_by_target = explain_rf(bundle, X_scaled)

    prefix = f"{bundle['cluster']}_{model_name}_{bundle['label_mode']}"

    # Full per-sample, per-gene, signed SHAP matrices -- one array per
    # target, NOT collapsed. This is the richer artifact; the summary table
    # below is just one common way to compress it to a single global ranking
    # number per gene.
    npz_path = os.path.join(args.outdir, f"{prefix}_shap_values.npz")
    np.savez(npz_path, samples=np.array(samples),
             feature_names=np.array(bundle["feature_names"]),
             **{f"shap_{t}": arr for t, arr in shap_by_target.items()})
    print(f"Saved full per-sample SHAP matrices to {npz_path}")

    # Summary: mean(|SHAP|) per gene per archetype, shaped like
    # {model}_feature_importances.csv for direct comparison against the
    # model's own gain-based importances.
    summary = pd.DataFrame({"feature": bundle["feature_names"]})
    for target, arr in shap_by_target.items():
        summary[target] = np.abs(arr).mean(axis=0)
    csv_path = os.path.join(args.outdir, f"{prefix}_shap_summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"Saved mean(|SHAP|)-per-gene summary to {csv_path}")

    # Signed mean(SHAP) per gene per archetype -- NOT the same thing as the
    # unsigned summary above with the abs() removed: mean(|SHAP|) collapses
    # direction away by construction (a gene with +5/-5 across two samples
    # and a gene with +5/+5 both average to |SHAP|=5, but only the second is
    # "positive on net"). Needed for a directional hit criterion
    # (SHAP > 0) -- see 009_go_enrichment.R, which uses this for its signed
    # GO test and the unsigned summary above (mean(|SHAP|) > 0, i.e.
    # "nonzero regardless of direction") for its unsigned one.
    summary_signed = pd.DataFrame({"feature": bundle["feature_names"]})
    for target, arr in shap_by_target.items():
        summary_signed[target] = arr.mean(axis=0)
    csv_signed_path = os.path.join(args.outdir, f"{prefix}_shap_summary_signed.csv")
    summary_signed.to_csv(csv_signed_path, index=False)
    print(f"Saved mean(SHAP)-per-gene (signed) summary to {csv_signed_path}")

    # GO over-representation universe (009_go_enrichment.R): the full
    # prevalence-filtered candidate gene set for this cluster, not just the
    # HVG panel bundle['feature_names'] the model was actually fit on -- see
    # build_input_matrix's docstring for why. One file per cluster
    # (independent of which model got explained), safely overwritten
    # identically if 008 is re-run for a different model on the same cluster.
    universe_path = os.path.join(args.outdir, f"{bundle['cluster']}_gene_universe.txt")
    with open(universe_path, "w") as f:
        f.write("\n".join(features[:n_gene_features]))
    print(f"Saved prevalence-filtered gene universe ({n_gene_features} genes) to {universe_path}")

    print("\nTop 10 genes by mean(|SHAP|), per archetype:")
    for target in bundle["target_names"]:
        top = summary.sort_values(target, ascending=False).head(10)
        print(f"\n  {target}:")
        for _, row in top.iterrows():
            print(f"    {row['feature']:20s} {row[target]:.4f}")


if __name__ == "__main__":
    main()
