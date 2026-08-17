nextflow.enable.dsl = 2

// ─── 0. Split the full AnnData into one lightweight AnnData per cluster ──────
// Runs once: loads the full (potentially very large) input AnnData, derives
// cluster_col via the celltype→cluster map (adata.obs only has celltype_col),
// and writes one subset AnnData per cluster so PREPARE_DATA below can fan out
// and run in parallel instead of looping over clusters in a single big job.
process SPLIT_BY_CLUSTER {
    label 'mem_100gb'
    container params.container_ml
    publishDir "${params.outdir}/split_by_cluster", mode: 'copy'

    input:
    path anndata
    path celltype_cluster_map

    output:
    path "*.cluster_input.h5ad",      emit: cluster_adatas
    path "full_obs_celltypes.tsv.gz", emit: full_obs

    script:
    """
    python ${launchDir}/scripts/000_split_by_cluster.py \\
        --anndata              ${anndata} \\
        --celltype_cluster_map ${celltype_cluster_map} \\
        --cluster_col          "${params.cluster_col}" \\
        --sample_col           "${params.sample_col}" \\
        --celltype_col         "${params.celltype_col}"
    """
}

// ─── 1. Extract pseudo-bulk features and MIL bags for one cluster ────────────
// Submitted as one job per cluster (fans out from SPLIT_BY_CLUSTER's output).
process PREPARE_DATA {
    label 'mem_32gb'
    container params.container_ml
    tag "${cluster}"
    publishDir "${params.outdir}/prepared_data", mode: 'copy'

    input:
    tuple val(cluster), path(cluster_adata)
    path archetype_scores
    path full_obs_celltypes

    output:
    tuple val(cluster), path("*.pseudobulk.npz"),                 emit: pseudobulk
    tuple val(cluster), path("*.targets.csv"),   optional: true,  emit: targets
    tuple val(cluster), path("*.mil_bags.h5"),   optional: true,  emit: mil_bags

    script:
    """
    python ${launchDir}/scripts/001_prepare_data.py \\
        --anndata              ${cluster_adata} \\
        --cluster              "${cluster}" \\
        --archetype_scores     ${archetype_scores} \\
        --target_mode          "${params.raw_or_clr_pca}" \\
        --sample_col           "${params.sample_col}" \\
        --celltype_col         "${params.celltype_col}" \\
        --pseudobulk_agg       "${params.pseudobulk_agg}" \\
        --min_sample_pct       ${params.min_sample_pct} \\
        --include_abundance    ${params.include_abundance} \\
        --abundance_type       "${params.abundance_type}" \\
        --full_obs_celltypes   ${full_obs_celltypes}
    """
}

// ─── 2a. Train one tree/linear model for one label mode ──────────────────────
// Submitted as one job per (cluster × model × label_mode).
// 4 models × 1 label mode = 4 jobs per cluster (xgboost, rf, elasticnet, lasso).
process TRAIN_TREE_MODELS {
    label 'mem_32gb'
    container params.container_ml
    tag "${cluster}_${model}_${label_mode}"
    publishDir "${params.outdir}/tree_models/${cluster}/${model}/${label_mode}", mode: 'copy'

    input:
    tuple val(cluster), path(pseudobulk), path(targets), val(model), val(label_mode)
    path xgb_param_grid
    path rf_param_grid
    path elasticnet_param_grid
    path lasso_param_grid

    output:
    tuple val(cluster), path("${cluster}_${model}_${label_mode}_tree/"), emit: results
    // Same data as `results`, additionally tagged with model/label_mode — lets
    // PERMUTATION_NULL below filter to specific models without needing to
    // parse them back out of the directory name (label_mode can itself
    // contain underscores, e.g. "argmax_filtered", making that parsing
    // ambiguous). `results` is untouched so nothing downstream of it changes.
    tuple val(cluster), val(model), val(label_mode), path("${cluster}_${model}_${label_mode}_tree/"), emit: results_tagged

    script:
    // Only xgboost gets extra inner-search parallelism (it's the one hitting
    // TERM_RUNLIMIT). This value is baked into the rendered command line, so
    // keeping it at 1 for every other model means their already-completed
    // -resume cache entries stay byte-identical and won't be needlessly
    // re-run when this changes.
    def n_jobs_inner_this = (model == 'xgboost') ? params.n_jobs_inner_xgboost : 1
    """
    mkdir -p ${cluster}_${model}_${label_mode}_tree
    python ${launchDir}/scripts/002_train_tree_models.py \\
        --cluster          ${cluster} \\
        --pseudobulk       ${pseudobulk} \\
        --targets          ${targets} \\
        --xgb_params       ${xgb_param_grid} \\
        --rf_params        ${rf_param_grid} \\
        --elasticnet_params ${elasticnet_param_grid} \\
        --lasso_params     ${lasso_param_grid} \\
        --models           ${model} \\
        --hvg_pct          ${params.hvg_pct} \\
        --hvg_max          ${params.hvg_max} \\
        --label_mode       ${label_mode} \\
        --min_score        ${params.min_score} \\
        --n_outer_folds    ${params.n_outer_folds} \\
        --n_inner_folds    ${params.n_inner_folds} \\
        --n_iter           ${params.n_hpo_iter_tree} \\
        --seed             ${params.random_seed} \\
        --n_jobs_outer     1 \\
        --n_jobs_inner     ${n_jobs_inner_this} \\
        --n_jobs_models    1 \\
        --outdir           ${cluster}_${model}_${label_mode}_tree
    """
}

// ─── 2a-null. Per-PC permutation null test (ElasticNet/Lasso family only) ────
// Submitted as one job per (cluster × model × label_mode), for whichever of
// elasticnet/lasso/elasticnet_indep/lasso_indep actually ran. Fixes
// hyperparameters to the mode already chosen across TRAIN_TREE_MODELS' own
// outer folds (its {model}_fold_metrics.csv) — no new hyperparameter search —
// so this is cheap enough to run automatically for every model, before you've
// decided which one to use, rather than only against a single finalized pick.
process PERMUTATION_NULL {
    label 'mem_32gb'
    container params.container_ml
    tag "${cluster}_${model}_${label_mode}"
    publishDir "${params.outdir}/permutation_null/${cluster}/${model}/${label_mode}", mode: 'copy'

    input:
    tuple val(cluster), path(tree_dir), val(model), val(label_mode), path(pseudobulk), path(targets)
    path xgb_param_grid
    path rf_param_grid
    path elasticnet_param_grid
    path lasso_param_grid

    output:
    tuple val(cluster), path("*_permutation_summary.csv"), path("*_permutation_null_raw.csv"), emit: results

    script:
    """
    python ${launchDir}/scripts/007_permutation_null.py \\
        --cluster        ${cluster} \\
        --pseudobulk     ${pseudobulk} \\
        --targets        ${targets} \\
        --model          ${model} \\
        --fold_metrics   ${tree_dir}/${model}_fold_metrics.csv \\
        --xgb_params     ${xgb_param_grid} \\
        --rf_params      ${rf_param_grid} \\
        --elasticnet_params ${elasticnet_param_grid} \\
        --lasso_params   ${lasso_param_grid} \\
        --hvg_pct        ${params.hvg_pct} \\
        --hvg_max        ${params.hvg_max} \\
        --n_permutations ${params.n_permutations} \\
        --n_folds        ${params.n_outer_folds} \\
        --label_mode     ${label_mode} \\
        --min_score      ${params.min_score} \\
        --seed           ${params.random_seed} \\
        --outdir         .
    """
}

// ─── 2b. Train Attention-based MIL with nested CV ────────────────────────────
// Submitted as one job per (cluster × label_mode).
process TRAIN_MIL {
    label 'mem_64gb'
    container params.container_torch
    tag "${cluster}_${label_mode}"
    publishDir "${params.outdir}/mil_models/${cluster}/${label_mode}", mode: 'copy'

    input:
    tuple val(cluster), path(mil_bags), path(targets), val(label_mode)
    path mil_param_grid

    output:
    tuple val(cluster), path("${cluster}_${label_mode}_mil/"), emit: results

    script:
    """
    mkdir -p ${cluster}_${label_mode}_mil
    python ${launchDir}/scripts/003_train_mil.py \\
        --cluster           ${cluster} \\
        --mil_bags          ${mil_bags} \\
        --targets           ${targets} \\
        --mil_params        ${mil_param_grid} \\
        --hvg_pct           ${params.hvg_pct} \\
        --hvg_max           ${params.hvg_max} \\
        --pseudobulk_agg    "${params.pseudobulk_agg}" \\
        --label_mode        ${label_mode} \\
        --min_score         ${params.min_score} \\
        --n_outer_folds     ${params.n_outer_folds} \\
        --n_inner_folds     ${params.n_inner_folds} \\
        --n_iter            ${params.n_hpo_iter_mil} \\
        --max_cells_per_bag ${params.max_cells_per_bag} \\
        --seed              ${params.random_seed} \\
        --outdir            ${cluster}_${label_mode}_mil
    """
}

// ─── 3. Per-cluster evaluation: compare all models ───────────────────────────
process EVALUATE_MODELS {
    label 'mem_8gb'
    container params.container_ml
    tag "${cluster}"
    publishDir "${params.outdir}/evaluation/${cluster}", mode: 'copy'

    input:
    tuple val(cluster), path(tree_dirs), path(mil_dirs), path(targets), path(perm_null_summaries)

    output:
    path "${cluster}_eval/", emit: eval_dir

    script:
    """
    mkdir -p ${cluster}_eval
    python ${launchDir}/scripts/004_evaluate.py \\
        --cluster   ${cluster} \\
        --tree_dirs ${tree_dirs} \\
        --mil_dir   ${mil_dirs} \\
        --targets   ${targets} \\
        --permutation_null_summaries ${perm_null_summaries} \\
        --outdir    ${cluster}_eval
    """
}

// ─── 4. Aggregate results across all clusters ────────────────────────────────
process AGGREGATE_RESULTS {
    label 'mem_8gb'
    container params.container_ml
    publishDir "${params.outdir}/summary", mode: 'copy'

    input:
    path eval_dirs

    output:
    path "summary/"

    script:
    """
    mkdir -p summary
    python ${launchDir}/scripts/005_aggregate.py \\
        --eval_dirs ${eval_dirs} \\
        --outdir    summary
    """
}

// ─── Workflow ─────────────────────────────────────────────────────────────────
workflow {

    // Input channels
    anndata_ch           = Channel.fromPath(params.anndata,             checkIfExists: true)
    celltype_map_ch      = Channel.fromPath(params.celltype_cluster_map, checkIfExists: true)
    // Reused across every parallel PREPARE_DATA / TRAIN_TREE_MODELS job, so these
    // must be value channels (Channel.value), not plain queue channels — otherwise
    // they'd only be available to the first job and exhausted after that.
    archetype_scores_ch  = Channel.value(file(params.archetype_scores, checkIfExists: true))
    xgb_params_ch        = Channel.value(file("${launchDir}/scripts/param_grids/xgboost_params.json"))
    rf_params_ch         = Channel.value(file("${launchDir}/scripts/param_grids/rf_params.json"))
    elasticnet_params_ch = Channel.value(file("${launchDir}/scripts/param_grids/elasticnet_params.json"))
    lasso_params_ch      = Channel.value(file("${launchDir}/scripts/param_grids/lasso_params.json"))
    mil_params_ch        = Channel.value(file("${launchDir}/scripts/param_grids/mil_params.json"))

    // Build model and label_mode lists from config flags
    def model_list = []
    if (params.run_xgboost    == "true") model_list << "xgboost"
    if (params.run_rf         == "true") model_list << "rf"
    if (params.run_elasticnet == "true") model_list << "elasticnet"
    if (params.run_lasso      == "true") model_list << "lasso"
    // Independent single-output ElasticNet/Lasso per target (no shared feature
    // support), only meaningful to compare against the joint fits when targets
    // are the orthogonal PCA archetype scores.
    if (params.raw_or_clr_pca == "pca" && params.run_independent_pca_fits == "true") {
        if (params.run_elasticnet == "true") model_list << "elasticnet_indep"
        if (params.run_lasso      == "true") model_list << "lasso_indep"
    }
    models_ch = Channel.fromList(model_list)

    def label_mode_list = []
    if (params.run_label_full            == "true") label_mode_list << "full"
    if (params.run_label_argmax          == "true") label_mode_list << "argmax"
    if (params.run_label_argmax_filtered == "true") label_mode_list << "argmax_filtered"
    label_modes_ch = Channel.fromList(label_mode_list)

    // Step 0: Split the full AnnData into one lightweight per-cluster AnnData —
    // runs once, so PREPARE_DATA below can fan out and run in parallel per cluster.
    SPLIT_BY_CLUSTER(anndata_ch, celltype_map_ch)

    // Fan out per cluster: derive cluster key from filename prefix (e.g. "A" from A.cluster_input.h5ad)
    cluster_adata_ch = SPLIT_BY_CLUSTER.out.cluster_adatas
        .flatten()
        .map { f -> [f.simpleName, f] }

    // SPLIT_BY_CLUSTER runs once but its full_obs output is needed by every
    // parallel PREPARE_DATA job below — .first() converts it to a value channel.
    full_obs_ch = SPLIT_BY_CLUSTER.out.full_obs.first()

    // Step 1: Prepare features — one parallel job per cluster
    PREPARE_DATA(cluster_adata_ch, archetype_scores_ch, full_obs_ch)

    pseudobulk_ch = PREPARE_DATA.out.pseudobulk
    targets_ch    = PREPARE_DATA.out.targets
    mil_bags_ch   = PREPARE_DATA.out.mil_bags

    // Step 2a: Tree models — fan out to one job per (cluster × model × label_mode)
    tree_input_ch = pseudobulk_ch
        .join(targets_ch)
        .combine(models_ch)
        .combine(label_modes_ch)
    // tree_input_ch: [cluster, pseudobulk, targets, model, label_mode]
    TRAIN_TREE_MODELS(tree_input_ch, xgb_params_ch, rf_params_ch, elasticnet_params_ch, lasso_params_ch)

    // Collect all tree result dirs for the same cluster into a list
    tree_results_ch = TRAIN_TREE_MODELS.out.results.groupTuple()
    // tree_results_ch: [cluster, [dir_xgboost_full, dir_xgboost_argmax, ..., dir_rf_argmax_filtered]]

    // Step 2a-null: per-PC permutation null test, ElasticNet/Lasso family only —
    // fixes hyperparameters from each job's own {model}_fold_metrics.csv (no new
    // search), so it's cheap enough to run automatically for every one of these
    // models/clusters before you've picked a winner, feeding into EVALUATE_MODELS
    // below rather than being a manual follow-up on a single chosen model.
    def NULL_TEST_MODELS = ['elasticnet', 'lasso', 'elasticnet_indep', 'lasso_indep']
    // Up to 4 models share the same cluster key here. Nextflow's join() silently
    // drops all but the first match when a key repeats on one side (see
    // https://github.com/nextflow-io/nextflow/issues/654) — with 4
    // elasticnet/lasso variants all keyed by the same cluster, join() would only
    // ever pass through whichever model's result happened to arrive first,
    // silently dropping the other 3. Broadcast via a cluster->value Map lookup
    // instead (.combine() against a single-emission value channel), which can't
    // drop anything regardless of how many models share a key.
    pseudobulk_map_ch = pseudobulk_ch.toList().map { it.collectEntries { c, pb  -> [(c): pb] } }
    targets_map_ch    = targets_ch.toList().map    { it.collectEntries { c, tgt -> [(c): tgt] } }
    perm_null_input_ch = TRAIN_TREE_MODELS.out.results_tagged
        .filter { cluster, model, label_mode, dir -> model in NULL_TEST_MODELS }
        .combine(pseudobulk_map_ch)
        .combine(targets_map_ch)
        .map { cluster, model, label_mode, dir, pbMap, tgtMap ->
            [cluster, dir, model, label_mode, pbMap[cluster], tgtMap[cluster]]
        }
    // perm_null_input_ch: [cluster, tree_dir, model, label_mode, pseudobulk, targets]
    PERMUTATION_NULL(perm_null_input_ch, xgb_params_ch, rf_params_ch, elasticnet_params_ch, lasso_params_ch)

    // Only thread the summary CSV into EVALUATE_MODELS (the raw per-permutation
    // draws stay published under results/permutation_null/ for direct inspection
    // but don't need to flow through the eval/aggregate chain).
    perm_null_results_ch = PERMUTATION_NULL.out.results
        .map { cluster, summary, raw -> [cluster, summary] }
        .groupTuple()
    // perm_null_results_ch: [cluster, [summary_elasticnet, summary_lasso, ...]]
    // — however many of the 4 target models actually succeeded for that cluster,
    // not necessarily all 4.

    // Step 2b: MIL — fan out to one job per (cluster × label_mode)
    mil_input_ch = mil_bags_ch
        .join(targets_ch)
        .combine(label_modes_ch)
    // mil_input_ch: [cluster, mil_bags, targets, label_mode]
    if (params.run_mil == "true") {
        TRAIN_MIL(mil_input_ch, mil_params_ch)
        mil_results_ch = TRAIN_MIL.out.results.groupTuple()
    } else {
        // Emit an empty grouped placeholder per cluster so EVALUATE_MODELS can still run
        mil_results_ch = pseudobulk_ch
            .map { cluster, pb -> [cluster, file("${params.outdir}/mil_models/${cluster}")] }
            .groupTuple()
    }
    // mil_results_ch: [cluster, [mil_dir_full, mil_dir_argmax, mil_dir_argmax_filtered]]

    // Step 3: Evaluate — join tree + MIL results + targets + permutation-null by cluster.
    // The permutation-null join uses remainder:true so a cluster is never dropped
    // from evaluation entirely in the (unlikely) case none of the 4 target models'
    // null tests succeeded for it — missing entries become null, mapped to [] below.
    eval_input_ch = tree_results_ch
        .join(mil_results_ch)
        .join(targets_ch)
        .join(perm_null_results_ch, remainder: true)
        .map { cluster, tree_dirs, mil_dirs, targets, perm_null_summaries ->
            [cluster, tree_dirs, mil_dirs, targets, perm_null_summaries ?: []]
        }
    // eval_input_ch: [cluster, [tree_dirs], [mil_dirs], targets, [perm_null_summaries]]
    EVALUATE_MODELS(eval_input_ch)

    // Step 4: Aggregate all cluster evaluations
    AGGREGATE_RESULTS(EVALUATE_MODELS.out.eval_dir.collect())
}
