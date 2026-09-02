#!/usr/bin/env Rscript
# 009_go_enrichment.R
# Manual, standalone script (like 006_finalize_model.py/008_shap_explain.py --
# not part of the Nextflow DAG): GO over-representation analysis (ORA) on the
# SHAP summaries produced by 008_shap_explain.py.
#
# ORA (clusterProfiler::enrichGO), not ranked GSEA (contrast with the sibling
# pipeline's ../../sc_nf_diffexpression/bin/015-run_fgsea.R, which ranks genes
# and runs fgsea::fgseaMultilevel): SHAP values have many exact zeros, so a
# ranked ordering would be statistically degenerate for the tied block. ORA
# just needs a fixed hit list vs. a universe, which zeros don't complicate.
#
# Runs TWO tests per archetype:
#   signed   -- hits = genes with mean(SHAP) > 0 (from *_shap_summary_signed.csv)
#   unsigned -- hits = genes with mean(|SHAP|) > 0, i.e. any nonzero attribution
#               regardless of direction (from *_shap_summary.csv)
# A signed (positive-only) hit list tested against unsigned GO terms is
# standard practice -- exactly analogous to testing a DE analysis's
# log2FC > 0 "upregulated" gene list against unsigned GO/MSigDB terms, which
# 015-run_fgsea.R's own ranked test does too (its --unsigned_ranking flag
# only controls the ranking statistic, never the gene sets).
#
# Universe: --gene_universe, i.e. 008_shap_explain.py's
# {cluster}_gene_universe.txt -- the prevalence-filtered candidate gene set
# for this cluster, NOT the HVG panel the model was actually fit on. HVG
# selection here (sq_target_vs_technical) is itself target-informed, not a
# neutral detection filter, so restricting the universe to the HVG panel
# would compound two target-informed filters into one enrichment number and
# discard real signal from genes the panel's capacity cap (hvg_max) excluded.
#
# Usage:
#   Rscript scripts/009_go_enrichment.R \
#       --shap_summary results/final_models/shap/B_xgboost_indep_full_shap_summary.csv \
#       --shap_summary_signed results/final_models/shap/B_xgboost_indep_full_shap_summary_signed.csv \
#       --gene_universe results/final_models/shap/B_gene_universe.txt \
#       --cluster B \
#       --outdir results/final_models/go_enrichment
#
# Requires: optparse, clusterProfiler, org.Hs.eg.db (human, SYMBOL keyType --
# matches this pipeline's gene-symbol convention throughout).
suppressPackageStartupMessages({
  library(optparse)
  library(clusterProfiler)
  library(org.Hs.eg.db)
})

option_list <- list(
  make_option("--shap_summary", type = "character",
              help = "{prefix}_shap_summary.csv from 008_shap_explain.py (mean(|SHAP|) per gene per archetype) -- drives the unsigned test"),
  make_option("--shap_summary_signed", type = "character",
              help = "{prefix}_shap_summary_signed.csv from 008_shap_explain.py (mean(SHAP) per gene per archetype) -- drives the signed test"),
  make_option("--gene_universe", type = "character",
              help = "{cluster}_gene_universe.txt from 008_shap_explain.py -- one gene SYMBOL per line, the prevalence-filtered (pre-HVG) candidate set"),
  make_option("--cluster", type = "character", help = "Cluster label, recorded in the output and used to name it"),
  make_option("--ont", type = "character", default = "BP",
              help = "GO ontology: BP, CC, MF, or ALL [default %default]"),
  make_option("--pvalue_cutoff", type = "double", default = 0.05,
              help = "enrichGO pvalueCutoff [default %default]"),
  make_option("--qvalue_cutoff", type = "double", default = 0.2,
              help = "enrichGO qvalueCutoff [default %default]"),
  make_option("--outdir", type = "character", help = "Output directory")
)
opt <- parse_args(OptionParser(option_list = option_list))

for (required in c("shap_summary", "shap_summary_signed", "gene_universe", "cluster", "outdir")) {
  if (is.null(opt[[required]])) stop(sprintf("--%s is required", required))
}

dir.create(opt$outdir, recursive = TRUE, showWarnings = FALSE)

universe <- readLines(opt$gene_universe)
summary_unsigned <- read.csv(opt$shap_summary, check.names = FALSE)
summary_signed <- read.csv(opt$shap_summary_signed, check.names = FALSE)

archetypes <- intersect(setdiff(colnames(summary_unsigned), "feature"),
                         setdiff(colnames(summary_signed), "feature"))
if (length(archetypes) == 0) {
  stop("No shared archetype columns between --shap_summary and --shap_summary_signed")
}

run_ora <- function(hits, universe, ont, pvalue_cutoff, qvalue_cutoff) {
  if (length(hits) == 0) return(NULL)
  res <- tryCatch(
    enrichGO(gene = hits, universe = universe, OrgDb = org.Hs.eg.db,
             keyType = "SYMBOL", ont = ont, pAdjustMethod = "BH",
             pvalueCutoff = pvalue_cutoff, qvalueCutoff = qvalue_cutoff),
    error = function(e) {
      message("  enrichGO failed: ", conditionMessage(e))
      NULL
    }
  )
  if (is.null(res)) return(NULL)
  res_df <- as.data.frame(res)
  if (nrow(res_df) == 0) return(NULL)
  res_df
}

all_results <- list()
for (archetype in archetypes) {
  signed_hits <- summary_signed$feature[summary_signed[[archetype]] > 0]
  unsigned_hits <- summary_unsigned$feature[summary_unsigned[[archetype]] > 0]

  for (sign in c("signed", "unsigned")) {
    hits <- if (sign == "signed") signed_hits else unsigned_hits
    message(sprintf("cluster=%s archetype=%s sign=%s: %d hits / %d universe genes",
                     opt$cluster, archetype, sign, length(hits), length(universe)))
    res_df <- run_ora(hits, universe, opt$ont, opt$pvalue_cutoff, opt$qvalue_cutoff)
    if (is.null(res_df)) {
      message("  no significant terms (or no hits) -- skipping")
      next
    }
    res_df$cluster <- opt$cluster
    res_df$archetype <- archetype
    res_df$sign <- sign
    all_results[[paste(archetype, sign)]] <- res_df
  }
}

out_path <- file.path(opt$outdir, sprintf("%s_go_enrichment.tsv", opt$cluster))
if (length(all_results) == 0) {
  message("No enriched terms found for any archetype/sign combination -- writing header-only output")
  write.table(data.frame(), out_path, sep = "\t", row.names = FALSE, quote = FALSE)
} else {
  combined <- do.call(rbind, all_results)
  write.table(combined, out_path, sep = "\t", row.names = FALSE, quote = FALSE)
}
message(sprintf("Saved combined GO enrichment results to %s", out_path))
