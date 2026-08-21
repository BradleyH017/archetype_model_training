#!/usr/bin/env python3
"""
999-check_hvg_selection.py
Diagnostic, not part of the Nextflow DAG: visualizes how "default" (raw
variance) vs "sq_target_vs_technical" (confound-aware, target_r2^2/covariate_r2)
HVG selection differ, per cluster and archetype.

Computed on the FULL dataset (not per outer-CV fold) purely for illustration
— this is a descriptive check of the selection criteria themselves, not a
substitute for the fold-specific reselection that actually happens in
002_train_tree_models.py/helpers.select_hvgs/select_hvgs_confound_aware.

For every requested (cluster, archetype) combination:
  - fits gene ~ covariates and gene ~ archetype (the same variance-decomposition
    machinery as helpers.select_hvgs_confound_aware) for every candidate gene
  - determines each method's top-hvg_max panel
  - renders a 2x3-panel figure per page:
      top row    — a covariate_r2-vs-target_r2 scatter colored by panel
                   membership (both / default only / confound-aware only /
                   neither), plus boxplots of covariate_r2 and target_r2 for
                   each panel
      bottom row — whether each ranking CRITERION itself tracks mean
                   expression (a scatter of the criterion vs log1p(mean
                   expression), with Pearson r annotated, for both default's
                   raw variance and the confound-aware ratio), plus a boxplot
                   of the SELECTED panel's own expression-level distribution.
                   Raw variance is well known to correlate with mean
                   expression (highly expressed genes accrue more absolute
                   variance "for free"); the confound-aware ratio divides two
                   similarly expression-biased quantities, which cancels most
                   of that bias as a side effect — see temp/hvg_selection.txt,
                   point 8, for the original LOESS-based check this reproduces.

The covariate list used for the confound-aware ranking follows the same
fallback as the real pipeline (see helpers.select_hvgs_confound_aware):
--train_only_covs if non-empty, else --train_test_covs.

Outputs (in --outdir, default temp/):
  {out_prefix}.pdf   — one page per (cluster, archetype)
  {out_prefix}.csv   — summary stats per combination (mean covariate_r2/
                        target_r2 per panel, plus each criterion's Pearson r
                        against log1p(mean expression)) — the per-combination
                        analogue of the all-cluster validation table in
                        temp/hvg_selection.txt

Example (defaults match the project's current conf/training.config):
    python scripts/999-check_hvg_selection.py
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import (
    load_pseudobulk,
    load_meta_file,
    fit_covariate_design,
    split_covariate_names,
    _explained_variance_ratio,  # reused directly so this stays in lockstep
                                 # with select_hvgs_confound_aware's own math
)

# Defaults mirror this project's conf/training.config as of the current
# hvg_select_method="sq_target_vs_technical" setup (train_only_covs empty,
# train_test_covs holding all nine covariates) — override via flags for a
# different setup.
_DEFAULT_META_FILE = ("/lustre/scratch125/humgen/projects_v2/ibdresponse/analysis/"
                       "bradley_analysis/mrVI/results/tables/meta_df.csv")
_DEFAULT_TRAIN_TEST_COVS = [
    "num_samples_in_pool", "is_pooled", "receipt_to_processing",
    "shipment_to_wetlab", "collection_to_wetlab", "collection_to_shipment",
    "samp_median_mt_perc", "nCells", "samp_median_total_counts",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared_data_dir", default="results/prepared_data",
                    help="Directory with {cluster}.pseudobulk.npz / {cluster}.targets.csv "
                         "from 001_prepare_data.py")
    p.add_argument("--clusters", nargs="*", default=None,
                    help="Clusters to check; default: every cluster with a .pseudobulk.npz "
                         "in --prepared_data_dir")
    p.add_argument("--meta_file", default=_DEFAULT_META_FILE)
    p.add_argument("--train_only_covs", nargs="*", default=[])
    p.add_argument("--train_test_covs", nargs="*", default=_DEFAULT_TRAIN_TEST_COVS)
    p.add_argument("--hvg_pct", type=float, default=1.0)
    p.add_argument("--hvg_max", type=int, default=3000)
    p.add_argument("--eps", type=float, default=1e-4,
                    help="Same epsilon as select_hvgs_confound_aware's denominator")
    p.add_argument("--outdir", default="temp")
    p.add_argument("--out_prefix", default="hvg_selection_comparison")
    return p.parse_args()


def discover_clusters(prepared_data_dir):
    paths = sorted(glob.glob(os.path.join(prepared_data_dir, "*.pseudobulk.npz")))
    return [os.path.basename(p).split(".")[0] for p in paths]


def _plot_one(cluster, arch, covariate_r2, target_r2, variance, score, log_mean_expr,
              cat, sel_default, sel_confound, confound_covs):
    colors = {"both": "#7f7f7f", "default only": "#d62728",
              "confound-aware only": "#1f77b4", "neither": "#e0e0e0"}
    sizes  = {"both": 6, "default only": 10, "confound-aware only": 10, "neither": 3}
    zorder = {"neither": 1, "both": 2, "default only": 3, "confound-aware only": 4}
    order  = ["neither", "both", "default only", "confound-aware only"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    ax = axes[0, 0]
    for c in order:
        m = cat == c
        ax.scatter(covariate_r2[m], target_r2[m], s=sizes[c], c=colors[c],
                   alpha=0.6 if c != "neither" else 0.25, label=f"{c} (n={m.sum()})",
                   zorder=zorder[c], rasterized=True)
    ax.set_xlabel("covariate_r2")
    ax.set_ylabel(f"target_r2 ({arch})")
    ax.set_title(f"Cluster {cluster}: every candidate gene\n(top-{len(sel_default)} panel membership)")
    ax.legend(fontsize=8, markerscale=2, loc="upper right")

    panels = [
        (axes[0, 1], (covariate_r2[list(sel_default)], covariate_r2[list(sel_confound)]),
         "covariate_r2 of selected genes", "Confound exposure of the selected panel\n(lower = better)"),
        (axes[0, 2], (target_r2[list(sel_default)], target_r2[list(sel_confound)]),
         f"target_r2 of selected genes ({arch})", "Phenotype signal of the selected panel\n(higher = better)"),
    ]
    for ax, values, ylabel, title in panels:
        bp = ax.boxplot(values, labels=["default", "sq_target_vs_technical"],
                         showfliers=False, patch_artist=True)
        for patch, c in zip(bp["boxes"], ["#d62728", "#1f77b4"]):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        for i, d in enumerate(values):
            ax.text(i + 1, ax.get_ylim()[1] * 0.92, f"mean={np.mean(d):.3f}", ha="center", fontsize=9)

    # ── Bottom row: does either criterion track mean expression? ──────────
    # Default ranks by raw variance, which is well known (scanpy/Seurat's own
    # motivation for mean-expression-binned dispersion ranking) to correlate
    # with mean expression — highly expressed genes accrue more absolute
    # variance "for free". The confound-aware score divides two similarly
    # expression-biased quantities (target_r2, covariate_r2), which cancels
    # most of that bias as a side effect (see temp/hvg_selection.txt, point 8)
    # — these panels show that directly, with Pearson r annotated, plus a
    # boxplot of each panel's own mean-expression distribution.
    r_variance = np.corrcoef(log_mean_expr, variance)[0, 1]
    r_score    = np.corrcoef(log_mean_expr, score)[0, 1]

    ax = axes[1, 0]
    for c in order:
        m = cat == c
        ax.scatter(log_mean_expr[m], variance[m], s=sizes[c], c=colors[c],
                   alpha=0.6 if c != "neither" else 0.25, zorder=zorder[c], rasterized=True)
    ax.set_yscale("log")
    ax.set_xlabel("log1p(mean expression)")
    ax.set_ylabel("raw variance  (default's ranking criterion)")
    ax.set_title(f"Default's criterion vs mean expression\nPearson r = {r_variance:.2f}")

    ax = axes[1, 1]
    for c in order:
        m = cat == c
        ax.scatter(log_mean_expr[m], score[m], s=sizes[c], c=colors[c],
                   alpha=0.6 if c != "neither" else 0.25, zorder=zorder[c], rasterized=True)
    ax.set_yscale("log")
    ax.set_xlabel("log1p(mean expression)")
    ax.set_ylabel("target_r2² / covariate_r2  (confound-aware criterion)")
    ax.set_title(f"Confound-aware criterion vs mean expression\nPearson r = {r_score:.2f}  "
                 f"(cancels most of default's bias)")

    ax = axes[1, 2]
    values = [log_mean_expr[list(sel_default)], log_mean_expr[list(sel_confound)]]
    bp = ax.boxplot(values, labels=["default", "sq_target_vs_technical"],
                     showfliers=False, patch_artist=True)
    for patch, c in zip(bp["boxes"], ["#d62728", "#1f77b4"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
    ax.set_ylabel("log1p(mean expression) of selected genes")
    ax.set_title("Expression-level distribution of the\nselected panel itself")
    for i, d in enumerate(values):
        ax.text(i + 1, ax.get_ylim()[1] * 0.92, f"mean={np.mean(d):.2f}", ha="center", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(f"Cluster {cluster} / {arch} — default vs confound-aware HVG selection\n"
                 f"(illustrative: full dataset, not per-fold; ranking covariates = {confound_covs})",
                 fontsize=10)
    return fig


def main():
    args = parse_args()
    args.train_only_covs = split_covariate_names(args.train_only_covs)
    args.train_test_covs = split_covariate_names(args.train_test_covs)
    confound_covs = args.train_only_covs if args.train_only_covs else args.train_test_covs
    if not confound_covs:
        raise ValueError("--train_only_covs or --train_test_covs must be non-empty — the "
                          "confound-aware ranking is undefined without at least one.")

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Loading covariate metadata from {args.meta_file}")
    meta = load_meta_file(args.meta_file)

    clusters = args.clusters or discover_clusters(args.prepared_data_dir)
    print(f"Clusters: {clusters}")
    print(f"Ranking covariate list (train_only_covs if non-empty, else train_test_covs): {confound_covs}")

    pdf_path = os.path.join(args.outdir, f"{args.out_prefix}.pdf")
    csv_rows = []

    with PdfPages(pdf_path) as pdf:
        for cluster in clusters:
            pb_path  = os.path.join(args.prepared_data_dir, f"{cluster}.pseudobulk.npz")
            tgt_path = os.path.join(args.prepared_data_dir, f"{cluster}.targets.csv")
            if not (os.path.exists(pb_path) and os.path.exists(tgt_path)):
                print(f"  [{cluster}] missing pseudobulk/targets, skipping")
                continue

            X, samples, features, n_gene_features = load_pseudobulk(pb_path)
            gene_names = features[:n_gene_features]
            df_tgt = pd.read_csv(tgt_path, index_col=0)

            common = [s for s in samples if s in df_tgt.index and s in meta.index]
            if len(common) < len(samples):
                print(f"  [{cluster}] {len(samples) - len(common)} samples dropped "
                      f"(missing targets/meta)")
            keep_idx = [samples.index(s) for s in common]
            genes_c  = X[keep_idx, :n_gene_features]
            meta_c   = meta.reindex(common)

            design, _    = fit_covariate_design(meta_c, common, confound_covs)
            reg_cov      = LinearRegression().fit(design, genes_c)
            covariate_r2 = _explained_variance_ratio(design, reg_cov.coef_, genes_c)
            variance     = genes_c.var(axis=0)
            log_mean_expr = np.log1p(genes_c.mean(axis=0))
            n_hvg        = min(args.hvg_max, max(1, int(genes_c.shape[1] * args.hvg_pct)))
            sel_default  = set(np.argsort(variance)[::-1][:n_hvg])

            for arch in df_tgt.columns:
                y  = df_tgt.loc[common, arch].values.astype(float)
                y2 = y.reshape(-1, 1)
                reg_tgt   = LinearRegression().fit(y2, genes_c)
                target_r2 = _explained_variance_ratio(y2, reg_tgt.coef_, genes_c)
                score        = target_r2 ** 2 / (covariate_r2 + args.eps)
                sel_confound = set(np.argsort(score)[::-1][:n_hvg])

                cat = np.full(len(gene_names), "neither", dtype=object)
                for i in range(len(gene_names)):
                    in_d, in_c = i in sel_default, i in sel_confound
                    if in_d and in_c:
                        cat[i] = "both"
                    elif in_d:
                        cat[i] = "default only"
                    elif in_c:
                        cat[i] = "confound-aware only"

                fig = _plot_one(cluster, arch, covariate_r2, target_r2, variance, score,
                                 log_mean_expr, cat, sel_default, sel_confound, confound_covs)
                pdf.savefig(fig)
                plt.close(fig)

                mean_cov_default   = covariate_r2[list(sel_default)].mean()
                mean_cov_confound  = covariate_r2[list(sel_confound)].mean()
                mean_tgt_default   = target_r2[list(sel_default)].mean()
                mean_tgt_confound  = target_r2[list(sel_confound)].mean()
                r_variance = np.corrcoef(log_mean_expr, variance)[0, 1]
                r_score    = np.corrcoef(log_mean_expr, score)[0, 1]
                csv_rows.append({
                    "cluster": cluster, "archetype": arch, "n_hvg": n_hvg,
                    "mean_covariate_r2_default": mean_cov_default,
                    "mean_covariate_r2_confound_aware": mean_cov_confound,
                    "mean_target_r2_default": mean_tgt_default,
                    "mean_target_r2_confound_aware": mean_tgt_confound,
                    "n_both": int((cat == "both").sum()),
                    "n_default_only": int((cat == "default only").sum()),
                    "n_confound_aware_only": int((cat == "confound-aware only").sum()),
                    # How strongly each ranking criterion tracks mean expression
                    # (Pearson r vs log1p(mean_expr), all candidate genes) — see
                    # temp/hvg_selection.txt point 8: the confound-aware ratio
                    # should show a much weaker |r| than raw variance.
                    "corr_default_criterion_vs_mean_expr": r_variance,
                    "corr_confound_aware_criterion_vs_mean_expr": r_score,
                })
                print(f"  [{cluster}/{arch}] mean covariate_r2: default={mean_cov_default:.3f} "
                      f"confound-aware={mean_cov_confound:.3f}  |  "
                      f"mean target_r2: default={mean_tgt_default:.3f} "
                      f"confound-aware={mean_tgt_confound:.3f}  |  "
                      f"criterion-vs-mean-expr r: default={r_variance:.3f} "
                      f"confound-aware={r_score:.3f}")

    df_summary = pd.DataFrame(csv_rows)
    csv_path = os.path.join(args.outdir, f"{args.out_prefix}.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSaved {pdf_path} ({len(csv_rows)} pages) and {csv_path}")

    print("\nOverall means across all combinations:")
    print(df_summary[[
        "mean_covariate_r2_default", "mean_covariate_r2_confound_aware",
        "mean_target_r2_default", "mean_target_r2_confound_aware",
        "corr_default_criterion_vs_mean_expr", "corr_confound_aware_criterion_vs_mean_expr",
    ]].mean().to_string())


if __name__ == "__main__":
    main()
