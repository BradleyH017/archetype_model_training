nextflow.enable.dsl = 2

// ─── 1. Extract per-cluster pseudo-bulk features and MIL bags ────────────────
process PREPARE_DATA {
    label 'mem_100gb'
    container params.container_ml
    publishDir "${params.outdir}/prepared_data", mode: 'copy'

    input:
    path anndata
    path archetype_scores
    path celltype_cluster_map

    output:
    path "*.pseudobulk.npz", emit: pseudobulk
    path "*.targets.csv",    emit: targets
    path "*.mil_bags.h5",    emit: mil_bags

    script:
    def map_arg = celltype_cluster_map.name != "NO_FILE" ? "--celltype_cluster_map ${celltype_cluster_map}" : ""
    """
    python ${launchDir}/scripts/001_prepare_data.py \\
        --anndata              ${anndata} \\
        --archetype_scores     ${archetype_scores} \\
        --cluster_col          "${params.cluster_col}" \\
        --sample_col           "${params.sample_col}" \\
        --celltype_col         "${params.celltype_col}" \\
        --rawcelltype_col      "${params.rawcelltype_col}" \\
        --hvg_pct              ${params.hvg_pct} \\
        --hvg_max              ${params.hvg_max} \\
        --pseudobulk_agg       "${params.pseudobulk_agg}" \\
        --min_sample_pct       ${params.min_sample_pct} \\
        --include_abundance    ${params.include_abundance} \\
        --abundance_type       "${params.abundance_type}" \\
        ${map_arg}
    """
}

// ─── 2a. Train XGBoost and Random Forest with nested CV ──────────────────────
process TRAIN_TREE_MODELS {
    label 'mem_32gb'
    container params.container_ml
    tag "${cluster}"
    publishDir "${params.outdir}/tree_models/${cluster}", mode: 'copy'

    input:
    tuple val(cluster), path(pseudobulk), path(targets)
    path xgb_param_grid
    path rf_param_grid

    output:
    tuple val(cluster), path("${cluster}_tree/"), emit: results

    script:
    def models = []
    if (params.run_xgboost == "true") models << "xgboost"
    if (params.run_rf      == "true") models << "rf"
    def models_str = models.join(" ")
    """
    mkdir -p ${cluster}_tree
    python ${launchDir}/scripts/002_train_tree_models.py \\
        --cluster        ${cluster} \\
        --pseudobulk     ${pseudobulk} \\
        --targets        ${targets} \\
        --xgb_params     ${xgb_param_grid} \\
        --rf_params      ${rf_param_grid} \\
        --models         ${models_str} \\
        --n_outer_folds  ${params.n_outer_folds} \\
        --n_inner_folds  ${params.n_inner_folds} \\
        --n_iter         ${params.n_hpo_iter_tree} \\
        --seed           ${params.random_seed} \\
        --n_jobs_outer   ${params.n_jobs_outer ?: 1} \\
        --n_jobs_inner   ${params.n_jobs_inner ?: 1} \\
        --outdir         ${cluster}_tree
    """
}

// ─── 2b. Train Attention-based MIL with nested CV ────────────────────────────
process TRAIN_MIL {
    label 'mem_64gb'
    container params.container_torch
    tag "${cluster}"
    publishDir "${params.outdir}/mil_models/${cluster}", mode: 'copy'

    input:
    tuple val(cluster), path(mil_bags), path(targets)
    path mil_param_grid

    output:
    tuple val(cluster), path("${cluster}_mil/"), emit: results

    script:
    """
    mkdir -p ${cluster}_mil
    python ${launchDir}/scripts/003_train_mil.py \\
        --cluster           ${cluster} \\
        --mil_bags          ${mil_bags} \\
        --targets           ${targets} \\
        --mil_params        ${mil_param_grid} \\
        --n_outer_folds     ${params.n_outer_folds} \\
        --n_inner_folds     ${params.n_inner_folds} \\
        --n_iter            ${params.n_hpo_iter_mil} \\
        --max_cells_per_bag ${params.max_cells_per_bag} \\
        --seed              ${params.random_seed} \\
        --outdir            ${cluster}_mil
    """
}

// ─── 3. Per-cluster evaluation: compare all models ───────────────────────────
process EVALUATE_MODELS {
    label 'mem_8gb'
    container params.container_ml
    tag "${cluster}"
    publishDir "${params.outdir}/evaluation/${cluster}", mode: 'copy'

    input:
    tuple val(cluster), path(tree_dir), path(mil_dir), path(targets)

    output:
    path "${cluster}_eval/", emit: eval_dir

    script:
    """
    mkdir -p ${cluster}_eval
    python ${launchDir}/scripts/004_evaluate.py \\
        --cluster     ${cluster} \\
        --tree_dir    ${tree_dir} \\
        --mil_dir     ${mil_dir} \\
        --targets     ${targets} \\
        --outdir      ${cluster}_eval
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
    anndata_ch           = Channel.fromPath(params.anndata,          checkIfExists: true)
    archetype_scores_ch  = Channel.fromPath(params.archetype_scores, checkIfExists: true)
    xgb_params_ch        = Channel.value(file("${launchDir}/scripts/param_grids/xgboost_params.json"))
    rf_params_ch         = Channel.value(file("${launchDir}/scripts/param_grids/rf_params.json"))
    mil_params_ch        = Channel.value(file("${launchDir}/scripts/param_grids/mil_params.json"))

    // Celltype→cluster map is optional; pass a sentinel file when not set
    celltype_map_ch = (params.celltype_cluster_map && params.celltype_cluster_map != "")
        ? Channel.fromPath(params.celltype_cluster_map, checkIfExists: true)
        : Channel.fromPath("NO_FILE", checkIfExists: false).ifEmpty(file("NO_FILE"))

    // Step 1: Prepare features — runs once, outputs one file set per cluster
    PREPARE_DATA(anndata_ch, archetype_scores_ch, celltype_map_ch)

    // Fan out per cluster: derive cluster key from filename prefix (e.g. "A" from A_pseudobulk.npz)
    pseudobulk_ch = PREPARE_DATA.out.pseudobulk
        .flatten()
        .map { f -> [f.simpleName.replaceAll('_pseudobulk', ''), f] }

    targets_ch = PREPARE_DATA.out.targets
        .flatten()
        .map { f -> [f.simpleName.replaceAll('_targets', ''), f] }

    mil_bags_ch = PREPARE_DATA.out.mil_bags
        .flatten()
        .map { f -> [f.simpleName.replaceAll('_mil_bags', ''), f] }

    // Step 2a: Tree models — [cluster, pseudobulk, targets]
    tree_input_ch = pseudobulk_ch.join(targets_ch)
    TRAIN_TREE_MODELS(tree_input_ch, xgb_params_ch, rf_params_ch)

    // Step 2b: MIL — [cluster, mil_bags, targets]
    mil_input_ch = mil_bags_ch.join(targets_ch)
    if (params.run_mil == "true") {
        TRAIN_MIL(mil_input_ch, mil_params_ch)
        mil_results_ch = TRAIN_MIL.out.results
    } else {
        // Emit empty placeholder dirs so EVALUATE_MODELS can still run
        mil_results_ch = tree_input_ch
            .map { cluster, pb, tgt -> [cluster, file("${params.outdir}/mil_models/${cluster}/${cluster}_mil")] }
    }

    // Step 3: Evaluate — join tree + MIL results + targets by cluster
    eval_input_ch = TRAIN_TREE_MODELS.out.results
        .join(mil_results_ch)
        .join(targets_ch)
    // eval_input_ch: [cluster, tree_dir, mil_dir, targets]
    EVALUATE_MODELS(eval_input_ch)

    // Step 4: Aggregate all cluster evaluations
    AGGREGATE_RESULTS(EVALUATE_MODELS.out.eval_dir.collect())
}
