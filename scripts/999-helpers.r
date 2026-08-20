#' Load and combine model output files from nested directory structure
#'
#' @param resdir Path to results directory containing model outputs
#' @return List with three data frames: summary_metrics, fold_metrics, feature_importance
#' @details Each file is annotated with cluster (from parent directory) and indep_model flag
load_model_files = function(resdir, permute_dir){
    # Helper function to reduce code duplication
    load_and_combine = function(resdir, pattern, label) {
        files = list.files(resdir, pattern = pattern, full.names = TRUE, recursive = TRUE)
        
        if (length(files) == 0) {
            warning(sprintf("No files found matching pattern '%s' in %s", pattern, resdir))
            return(data.frame())
        }
        
        lapply(files, function(filepath) {
            # Extract cluster from directory structure (parent of file)
            relative_path = sub(paste0("^", resdir, "/?"), "", filepath)
            cluster = unlist(strsplit(relative_path, "/"))[1]
            
            read.csv(filepath, stringsAsFactors = FALSE) %>%
                mutate(cluster = cluster)
        }) %>%
            bind_rows() %>%
            {
                if ("model" %in% names(.)) {
                    mutate(., indep_model = ifelse(grepl("indep", model), "Independent", "Joint"))
                } else {
                    .
                }
            }
    }
    
    # Load each file type
    summary_metrics = load_and_combine(resdir, "summary_metrics.csv", "summary")
    fold_metrics = load_and_combine(resdir, "fold_metrics.csv", "fold")
    feature_importance = load_and_combine(resdir, "feature_importances.csv", "feature")
    overall = load_and_combine(resdir, "overall_metrics.csv", "overall")
    permute = load_and_combine(permute_dir, "permutation_summary.csv", "permutation")

    return(list(
        sum = summary_metrics,
        fold = fold_metrics,
        feature = feature_importance,
        overall = overall,
        permute = permute
    ))
}

#' Per-PC variance of the TRUE archetype target values, computed on the fly
#'
#' This is exactly the per-column weight sklearn's
#' r2_score(multioutput='variance_weighted') uses internally (np.var(y_true,
#' axis=0)) — it is NOT read from any stored/precomputed PCA eigenvalue
#' (e.g. nothing from archetypal_analysis_from_mrvi's clr_pca_loadings files),
#' it is recomputed directly from each cluster's own targets.csv every time.
#'
#' @param prepared_data_dir Path to results/prepared_data/ (has {cluster}.targets.csv)
#' @param clusters Character vector of cluster labels to load
#' @return data.frame with columns: cluster, archetype, variance
get_pc_variance = function(prepared_data_dir, clusters){
    lapply(clusters, function(cl){
        f = file.path(prepared_data_dir, paste0(cl, ".targets.csv"))
        if (!file.exists(f)) {
            warning(sprintf("targets file not found: %s", f))
            return(data.frame())
        }
        df = read.csv(f, row.names = 1)
        data.frame(
            cluster   = cl,
            archetype = colnames(df),
            variance  = sapply(df, var, use = "complete.obs")
        )
    }) %>% bind_rows()
}