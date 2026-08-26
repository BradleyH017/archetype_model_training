# Bradley August 2026
# module load $uber
library(tidyverse)
library(ggplot2)
library(ggrepel)

# options
resdir = "results/tree_models/"
permute_dir = "results/permutation_null/"
prepared_data_dir = "results/prepared_data/"
outdir = "OUT/"
source("scripts/999-helpers.r")
if(!dir.exists(outdir)) dir.create(outdir, recursive = TRUE)

# load results
res = load_model_files(resdir, permute_dir)

# Plot the overall metrics 
tile_data = res$overall %>%
    group_by(cluster, model) %>%
    summarise(r2_variance_weighted = mean(r2_variance_weighted, na.rm = TRUE), .groups = "drop") %>%
    complete(cluster, model) %>%
    mutate(
        cluster = factor(cluster, levels = sort(unique(cluster))),
        model = factor(model, levels = sort(unique(model)))
    ) %>%
    left_join(
        # 007_permutation_null.py writes one row per (cluster, model, archetype) —
        # take the LEAST significant (max) p-value across a model's archetypes,
        # so a tile is only starred if every PC it predicts clears that bar.
        res$permute %>%
            group_by(cluster, model) %>%
            summarise(p_value = max(p_value, na.rm = TRUE), .groups = "drop"),
        by = c("cluster", "model")
    ) %>%
    mutate(
        sig_label = case_when(
            is.na(p_value) ~ "",
            p_value < 0.01  ~ "**",
            p_value < 0.05  ~ "*",
            TRUE            ~ ""
        )
    )

ggplot(tile_data, aes(x = model, y = cluster, fill = r2_variance_weighted)) +
    geom_tile(color = "white", linewidth = 0.5) +
    geom_text(aes(label = sig_label), color = "red", size = 5, fontface = "bold", vjust = 0.75) +
    scale_fill_gradient(low = "white", high = "darkblue",
                       limits = c(0, max(tile_data$r2_variance_weighted, na.rm = TRUE)),
                       na.value = "grey80",
                       name = "R² Variance\nWeighted") +
    theme_classic() +
    theme(
        axis.text.x = element_text(angle = 45, hjust = 1),
        panel.grid = element_blank(),
        axis.title = element_text(face = "bold")
    ) +
    labs(
        title = "R² Variance Weighted by Cluster and Model",
        subtitle = "* = all PCs p<0.05 (permutation null)   ** = all PCs p<0.01",
        x = "Model",
        y = "Cluster"
    ) +
    coord_fixed(ratio = 1)

ggsave(filename = file.path(outdir, "overall_metrics_tile_plot.png"), width = 8, height = 6)    

# Create row-scaled version (scales R² within each cluster)
tile_data_scaled = tile_data %>%
    group_by(cluster) %>%
    mutate(
        r2_scaled = (r2_variance_weighted - min(r2_variance_weighted, na.rm = TRUE)) / 
                    (max(r2_variance_weighted, na.rm = TRUE) - min(r2_variance_weighted, na.rm = TRUE))
    ) %>%
    ungroup()

ggplot(tile_data_scaled, aes(x = model, y = cluster, fill = r2_scaled)) +
    geom_tile(color = "white", linewidth = 0.5) +
    geom_text(aes(label = sig_label), color = "red", size = 5, fontface = "bold", vjust = 0.75) +
    scale_fill_gradient(low = "white", high = "darkblue",
                       limits = c(0, 1),
                       na.value = "grey80",
                       name = "R² Scaled\n(by cluster)") +
    theme_classic() +
    theme(
        axis.text.x = element_text(angle = 45, hjust = 1),
        panel.grid = element_blank(),
        axis.title = element_text(face = "bold")
    ) +
    labs(
        title = "R² Variance Weighted by Cluster and Model (Row-Scaled)",
        subtitle = "* = all PCs p<0.05 (permutation null)   ** = all PCs p<0.01",
        x = "Model",
        y = "Cluster"
    ) +
    coord_fixed(ratio = 1)

ggsave(filename = file.path(outdir, "overall_metrics_tile_plot_row_scaled.png"), width = 8, height = 6)

# Save a summary table of the overall results
sumdf = res$overall %>% 
    group_by(model, indep_model) %>% 
    summarise(
        med_r2 = median(r2_variance_weighted), 
        min_r2 = min(r2_variance_weighted), 
        max_r2 = max(r2_variance_weighted)
    ) %>% 
    ungroup() %>% 
    arrange(desc(med_r2))

write.table(sumdf, file = file.path(outdir, "overall_metrics_summary.txt"), sep = "\t", row.names = FALSE, quote = FALSE)

# Plot the R2 from the summary metrics
ggplot(res$sum, aes(x=cluster, y=r2)) + 
    geom_boxplot() + 
    facet_wrap(~indep_model, scales="free") +
    theme_classic() + 
    ggtitle("R2 from summary metrics") + 
    xlab("Cluster") + 
    ylab("R2") 

ggsave(filename = file.path(outdir, "summary_metrics_boxplot.png"), width = 8, height = 6)

###########
# Investigate the best performing model based on variance-normalised r2
###########
best_model = res$overall %>% 
    filter(r2_variance_weighted == max(r2_variance_weighted))

# Plot the per-PC r2 for this model
per_cluster_weighted_r2 = res$sum %>%
    filter(model == best_model$model) %>%
    group_by(cluster) %>%
    summarise(weighted_r2 = mean(r2), .groups = "drop")

best_model_sum = res$sum %>%
    filter(model == best_model$model)

# Variance explained per PC, computed on the fly from the TRUE archetype
# scores (results/prepared_data/{cluster}.targets.csv) — this is exactly the
# per-column weight sklearn's r2_score(multioutput='variance_weighted') uses
# internally (np.var(y_true, axis=0)), not a stored/precomputed PCA eigenvalue.
# See get_pc_variance() in 999-helpers.r.
pc_variance = get_pc_variance(prepared_data_dir, unique(best_model_sum$cluster))

best_model_sum %>%
    left_join(per_cluster_weighted_r2, by = "cluster") %>%
    left_join(pc_variance, by = c("cluster", "archetype")) %>%
    ggplot(aes(x=variance, y=r2)) +
    geom_point() +
    geom_smooth(method = "lm", se = FALSE, color = "blue", linetype = "dotted") +
    geom_text_repel(aes(label = archetype), size = 3, max.overlaps = 20) +
    geom_line(aes(y=weighted_r2), color="red", linetype="dashed") +
    facet_wrap(~cluster, scales = "free_x") +
    theme_classic() +
    ggtitle(paste("Per-PC R2 vs. variance explained for best model:", best_model$model)) +
    xlab("Variance of true archetype PC score (across samples)") +
    ylab("R2") +
    ylim(0, 1)

ggsave(filename = file.path(outdir, "best_model_per_pc_r2.png"), width = 8, height = 6)

# Plot the per cluster overlap of non-0 features
