# Bradley August 2026
# module load $uber
library(tidyverse)
library(ggplot2)

# options
resdir = "temp/results/tree_models/"
#models = c("lasso")
#models_exp = paste0(models, "_indep")
source("scripts/999-helpers.r")

# load results
res = load_model_files(resdir)

# Plot the overall metrics 
tile_data = res$overall %>%
    group_by(cluster, model) %>%
    summarise(r2_variance_weighted = mean(r2_variance_weighted, na.rm = TRUE), .groups = "drop") %>%
    complete(cluster, model) %>%
    mutate(
        cluster = factor(cluster, levels = sort(unique(cluster))),
        model = factor(model, levels = sort(unique(model)))
    )

ggplot(tile_data, aes(x = model, y = cluster, fill = r2_variance_weighted)) +
    geom_tile(color = "white", linewidth = 0.5) +
    scale_fill_gradient(low = "white", high = "darkblue", 
                       limits = c(0, 1), 
                       na.value = "grey80",
                       name = "R² Variance\nWeighted") +
    theme_minimal() +
    theme(
        axis.text.x = element_text(angle = 45, hjust = 1),
        panel.grid = element_blank(),
        axis.title = element_text(face = "bold")
    ) +
    labs(
        title = "R² Variance Weighted by Cluster and Model",
        x = "Model",
        y = "Cluster"
    ) +
    coord_fixed(ratio = 1)


# Plot the R2 from the summary metrics
ggplot(res$sum, aes(x=cluster, y=r2)) + 
    geom_boxplot() + 
    facet_wrap(~indep_model, scales="free") +
    theme_classic() + 
    ggtitle("R2 from summary metrics") + 
    xlab("Cluster") + 
    ylab("R2") 
