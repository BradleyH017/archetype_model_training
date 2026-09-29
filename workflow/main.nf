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
// Submitted as one job per (cluster × model × label_mode): xgboost_indep, rf,
// elasticnet, lasso — plus elasticnet_indep/lasso_indep when enabled.
// xgboost_joint runs separately (different XGBoost version) — see
// TRAIN_XGBOOST_JOINT below.
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
    path meta_file

    output:
    tuple val(cluster), path("${cluster}_${model}_${label_mode}_tree/"), emit: results
    // Same data as `results`, additionally tagged with model/label_mode — lets
    // PERMUTATION_NULL below filter to specific models without needing to
    // parse them back out of the directory name (label_mode can itself
    // contain underscores, e.g. "argmax_filtered", making that parsing
    // ambiguous). `results` is untouched so nothing downstream of it changes.
    tuple val(cluster), val(model), val(label_mode), path("${cluster}_${model}_${label_mode}_tree/"), emit: results_tagged

    script:
    // Only xgboost_indep gets extra inner-search parallelism (it's the one
    // hitting TERM_RUNLIMIT). This value is baked into the rendered command
    // line, so keeping it at 1 for every other model means their
    // already-completed -resume cache entries stay byte-identical and won't
    // be needlessly re-run when this changes.
    def n_jobs_inner_this = (model == 'xgboost_indep') ? params.n_jobs_inner_xgboost : 1
    // meta_file is optional (sentinel "NO_FILE" when unset) — omit --meta_file
    // entirely rather than pass a nonexistent path through to argparse.
    def meta_arg = meta_file.name != "NO_FILE" ? "--meta_file ${meta_file}" : ""
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
        ${meta_arg} \\
        --train_only_covs  ${params.train_only_covs} \\
        --train_test_covs  ${params.train_test_covs} \\
        --hvg_select_method ${params.hvg_select_method} \\
        --outdir           ${cluster}_${model}_${label_mode}_tree
    """
}

// ─── 2a-joint. XGBoost native joint multi-output fit ──────────────────────────
// Separate process (and container) from TRAIN_TREE_MODELS: multi_strategy=
// "multi_output_tree" requires XGBoost>=2.0, and container_ml's pinned
// XGBoost predates it (see helpers.build_estimator's version check). Point
// params.xgboost2_env at an environment with a current XGBoost to use this.
// Guarded by params.run_xgboost_joint (off by default) — see workflow below.
process TRAIN_XGBOOST_JOINT {
    // Native multi_strategy="multi_output_tree" fits ran heavier than
    // TRAIN_TREE_MODELS' other models on the same 32GB/12h envelope — LSF
    // TERM_MEMLIMIT'd clusters A/B/D at 33.9-35.2GB (against a 32GB request).
    // mem_64gb gives ample headroom (64GB, 24h, 'long' queue from the first
    // attempt — no need to wait for a retry to get a workable time budget);
    // cpus=8 is added via the withName override in conf/training.config to
    // match --n_jobs_inner_xgboost below (mem_64gb itself doesn't set cpus,
    // since MIL — its other consumer — doesn't need this many).
    label 'mem_64gb'
    container params.xgboost2_env
    tag "xgboost_joint_${cluster}_${label_mode}"
    publishDir "${params.outdir}/tree_models/${cluster}/xgboost_joint/${label_mode}", mode: 'copy'

    input:
    tuple val(cluster), path(pseudobulk), path(targets), val(label_mode)
    path xgb_param_grid
    path rf_param_grid
    path elasticnet_param_grid
    path lasso_param_grid
    path meta_file

    output:
    tuple val(cluster), path("${cluster}_xgboost_joint_${label_mode}_tree/"), emit: results

    script:
    def meta_arg = meta_file.name != "NO_FILE" ? "--meta_file ${meta_file}" : ""
    """
    mkdir -p ${cluster}_xgboost_joint_${label_mode}_tree
    python ${launchDir}/scripts/002_train_tree_models.py \\
        --cluster          ${cluster} \\
        --pseudobulk       ${pseudobulk} \\
        --targets          ${targets} \\
        --xgb_params       ${xgb_param_grid} \\
        --rf_params        ${rf_param_grid} \\
        --elasticnet_params ${elasticnet_param_grid} \\
        --lasso_params     ${lasso_param_grid} \\
        --models           xgboost_joint \\
        --hvg_pct          ${params.hvg_pct} \\
        --hvg_max          ${params.hvg_max} \\
        --label_mode       ${label_mode} \\
        --min_score        ${params.min_score} \\
        --n_outer_folds    ${params.n_outer_folds} \\
        --n_inner_folds    ${params.n_inner_folds} \\
        --n_iter           ${params.n_hpo_iter_tree} \\
        --seed             ${params.random_seed} \\
        --n_jobs_outer     1 \\
        --n_jobs_inner     ${params.n_jobs_inner_xgboost} \\
        ${meta_arg} \\
        --train_only_covs  ${params.train_only_covs} \\
        --train_test_covs  ${params.train_test_covs} \\
        --hvg_select_method ${params.hvg_select_method} \\
        --n_jobs_models    1 \\
        --outdir           ${cluster}_xgboost_joint_${label_mode}_tree
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
    path meta_file

    output:
    tuple val(cluster), path("*_permutation_summary.csv"), path("*_permutation_null_raw.csv"), emit: results

    script:
    def meta_arg = meta_file.name != "NO_FILE" ? "--meta_file ${meta_file}" : ""
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
        ${meta_arg} \\
        --train_only_covs ${params.train_only_covs} \\
        --train_test_covs ${params.train_test_covs} \\
        --hvg_select_method ${params.hvg_select_method} \\
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
    path meta_file

    output:
    tuple val(cluster), path("${cluster}_${label_mode}_mil/"), emit: results

    script:
    def meta_arg = meta_file.name != "NO_FILE" ? "--meta_file ${meta_file}" : ""
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
        ${meta_arg} \\
        --train_only_covs   ${params.train_only_covs} \\
        --train_test_covs   ${params.train_test_covs} \\
        --hvg_select_method ${params.hvg_select_method} \\
        --outdir            ${cluster}_${label_mode}_mil
    """
}

// ─── 2c. MultiMIL — one Nextflow task PER OUTER FOLD ─────────────────────────
// Deliberately different from TRAIN_TREE_MODELS/TRAIN_MIL, which each loop
// over every outer fold inside one process: MultiMIL training is GPU-bound
// and heavy enough per fold that each fold gets its own task/GPU allocation
// instead of serializing 5 folds on one GPU. Fanned out over
// cluster x target_mode(joint|indep) x fold_idx. See
// scripts/003b_train_multimil.py's module docstring for the fold/HVG-panel
// comparability argument with TRAIN_TREE_MODELS/TRAIN_MIL.
process TRAIN_MULTIMIL {
    label 'gpu_multimil'
    container params.multimil_env
    tag "${cluster}_${target_mode}_fold${fold_idx}"
    publishDir "${params.outdir}/multimil/${cluster}/${target_mode}", mode: 'copy'

    input:
    tuple val(cluster), path(mil_bags), path(targets), val(target_mode), val(fold_idx)
    path meta_file

    output:
    tuple val(cluster), val(target_mode), path("*_fold${fold_idx}_oof_predictions.csv"), emit: fold_oof

    script:
    def meta_arg = meta_file.name != "NO_FILE" ? "--meta_file ${meta_file}" : ""
    """
    export MULTIMIL_PKG_DIR="${launchDir}/multimil_packages_1.0.0"
    # Defensive: multimil_packages_1.0.0 is read-only from inside this
    # container, and numba (used by scvi-tools/anndata's own optional
    # accelerated code paths, not just scanpy) hard-fails rather than
    # falling back gracefully when it can't write its on-disk compile
    # cache next to a JIT'd function's source file -- confirmed by direct
    # traceback once already (scanpy's own numba usage, since removed from
    # 003b_train_multimil.py, but this task's own work dir is always
    # writable, so pointing both caches there costs nothing either way).
    export NUMBA_CACHE_DIR="\$PWD"
    export MPLCONFIGDIR="\$PWD"
    python -u ${launchDir}/scripts/003b_train_multimil.py \\
        --cluster           ${cluster} \\
        --mil_bags          ${mil_bags} \\
        --targets           ${targets} \\
        --sample_col        "${params.sample_col}" \\
        --fold_idx          ${fold_idx} \\
        --n_outer_folds     ${params.n_outer_folds} \\
        --target_mode       ${target_mode} \\
        --hvg_pct           ${params.hvg_pct} \\
        --hvg_max           ${params.hvg_max} \\
        --hvg_select_method ${params.hvg_select_method} \\
        --pseudobulk_agg    "${params.pseudobulk_agg}" \\
        --max_cells_per_bag ${params.max_cells_per_bag} \\
        --seed              ${params.random_seed} \\
        ${meta_arg} \\
        --train_only_covs   ${params.train_only_covs} \\
        --train_test_covs   ${params.train_test_covs} \\
        --outdir            .
    """
}

// ─── 2c-combine. Stitch MultiMIL's per-fold OOF predictions back together ───
// One task per (cluster x target_mode), gathering that combination's 5
// TRAIN_MULTIMIL fold outputs. Produces the exact {model}_summary_metrics.csv/
// {model}_overall_metrics.csv schema TRAIN_TREE_MODELS/TRAIN_MIL already use
// (model name = "multimil_joint"/"multimil_indep"), so EVALUATE_MODELS below
// needs no changes at all — its --tree_dirs is already a generic glob over
// any directory for *_summary_metrics.csv.
process COMBINE_MULTIMIL_FOLDS {
    label 'mem_8gb'
    container params.container_ml
    tag "${cluster}_${target_mode}"
    publishDir "${params.outdir}/multimil_combined/${cluster}/${target_mode}", mode: 'copy'

    input:
    tuple val(cluster), val(target_mode), path(fold_oof_csvs), path(targets)

    output:
    tuple val(cluster), path("${cluster}_multimil_${target_mode}/"), emit: results

    script:
    """
    mkdir -p ${cluster}_multimil_${target_mode}
    python ${launchDir}/scripts/003c_combine_multimil_folds.py \\
        --cluster ${cluster} \\
        --target_mode ${target_mode} \\
        --fold_oof_predictions ${fold_oof_csvs} \\
        --targets ${targets} \\
        --outdir ${cluster}_multimil_${target_mode}
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

    // Per-sample metadata for technical covariate regression is optional; pass
    // a sentinel file when not set (params.train_only_covs/train_test_covs are
    // both empty by default, so this is a no-op unless you actually list covariates).
    // Reused across many TRAIN_TREE_MODELS/TRAIN_XGBOOST_JOINT/PERMUTATION_NULL
    // tasks, so this must be a single value channel regardless of branch.
    def meta_file_path = (params.meta_file && params.meta_file != "")
        ? file(params.meta_file, checkIfExists: true)
        : file("NO_FILE")
    meta_file_ch = Channel.value(meta_file_path)

    // Build model and label_mode lists from config flags
    // xgboost_joint is NOT added here — it needs a different container
    // (params.xgboost2_env) so it runs through its own dedicated process,
    // TRAIN_XGBOOST_JOINT, guarded by params.run_xgboost_joint below.
    def model_list = []
    if (params.run_xgboost    == "true") model_list << "xgboost_indep"
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
    TRAIN_TREE_MODELS(tree_input_ch, xgb_params_ch, rf_params_ch, elasticnet_params_ch, lasso_params_ch, meta_file_ch)

    // xgboost_joint (multi_strategy="multi_output_tree", XGBoost>=2.0) runs in
    // its own process/container (params.xgboost2_env) rather than through
    // TRAIN_TREE_MODELS/model_list above, since container_ml's pinned
    // XGBoost predates this feature — see TRAIN_XGBOOST_JOINT.
    all_tree_results_ch = TRAIN_TREE_MODELS.out.results
    if (params.run_xgboost_joint == "true") {
        xgb_joint_input_ch = pseudobulk_ch
            .join(targets_ch)
            .combine(label_modes_ch)
        // xgb_joint_input_ch: [cluster, pseudobulk, targets, label_mode]
        TRAIN_XGBOOST_JOINT(xgb_joint_input_ch, xgb_params_ch, rf_params_ch, elasticnet_params_ch, lasso_params_ch, meta_file_ch)
        all_tree_results_ch = all_tree_results_ch.mix(TRAIN_XGBOOST_JOINT.out.results)
    }

    // Collect all tree result dirs for the same cluster into a list
    tree_results_ch = all_tree_results_ch.groupTuple()
    // tree_results_ch: [cluster, [dir_xgboost_indep_full, ..., dir_rf_full, dir_xgboost_joint_full (if enabled)]]

    // Cluster->targets broadcast Map (see the permutation-null block below
    // for why a Map-lookup is used instead of a plain join) — built here,
    // unconditionally, since both the permutation-null block and the
    // MultiMIL block need it.
    targets_map_ch = targets_ch.toList().map { it.collectEntries { c, tgt -> [(c): tgt] } }

    // Step 2a-null: per-PC permutation null test — fixes hyperparameters from
    // each job's own {model}_fold_metrics.csv (no new search), so it's cheap
    // enough to run automatically for every one of these models/clusters
    // before you've picked a winner, feeding into EVALUATE_MODELS below
    // rather than being a manual follow-up on a single chosen model.
    // xgboost_indep included: its hyperparameter search is joint (single
    // shared setting, parsed the same way as elasticnet/lasso) but its
    // fitting is always independent per target regardless — see
    // 007_permutation_null.py's module docstring and _PER_TARGET_FIT_MODELS.
    // rf/mil are not included — rf is natively, genuinely joint (nothing to
    // work around), and MIL needs a different approach entirely. xgboost_joint
    // is also not included: it's natively joint the same way rf is, but this
    // process runs in container_ml, which doesn't have the XGBoost>=2.0 that
    // xgboost_joint requires (see TRAIN_XGBOOST_JOINT) — wiring a second
    // container-specific permutation-null process wasn't asked for here.
    def NULL_TEST_MODELS = ['elasticnet', 'lasso', 'elasticnet_indep', 'lasso_indep', 'xgboost_indep']
    if (params.run_permutation_null == "true") {
        // Up to 4 models share the same cluster key here. Nextflow's join() silently
        // drops all but the first match when a key repeats on one side (see
        // https://github.com/nextflow-io/nextflow/issues/654) — with 4
        // elasticnet/lasso variants all keyed by the same cluster, join() would only
        // ever pass through whichever model's result happened to arrive first,
        // silently dropping the other 3. Broadcast via a cluster->value Map lookup
        // instead (.combine() against a single-emission value channel), which can't
        // drop anything regardless of how many models share a key.
        pseudobulk_map_ch = pseudobulk_ch.toList().map { it.collectEntries { c, pb  -> [(c): pb] } }
        perm_null_input_ch = TRAIN_TREE_MODELS.out.results_tagged
            .filter { cluster, model, label_mode, dir -> model in NULL_TEST_MODELS }
            .combine(pseudobulk_map_ch)
            .combine(targets_map_ch)
            .map { cluster, model, label_mode, dir, pbMap, tgtMap ->
                [cluster, dir, model, label_mode, pbMap[cluster], tgtMap[cluster]]
            }
        // perm_null_input_ch: [cluster, tree_dir, model, label_mode, pseudobulk, targets]
        PERMUTATION_NULL(perm_null_input_ch, xgb_params_ch, rf_params_ch, elasticnet_params_ch, lasso_params_ch, meta_file_ch)

        // Only thread the summary CSV into EVALUATE_MODELS (the raw per-permutation
        // draws stay published under results/permutation_null/ for direct inspection
        // but don't need to flow through the eval/aggregate chain).
        perm_null_results_ch = PERMUTATION_NULL.out.results
            .map { cluster, summary, raw -> [cluster, summary] }
            .groupTuple()
        // perm_null_results_ch: [cluster, [summary_elasticnet, summary_lasso, ...]]
        // — however many of the 4 target models actually succeeded for that cluster,
        // not necessarily all 4.
    } else {
        // eval_input_ch's join below already uses remainder:true + `?: []`
        // specifically to tolerate clusters with no permutation-null result at
        // all, so Channel.empty() here (rather than a per-cluster placeholder
        // like mil_results_ch's else-branch needs) is sufficient.
        perm_null_results_ch = Channel.empty()
    }

    // Step 2b: MIL — fan out to one job per (cluster × label_mode)
    mil_input_ch = mil_bags_ch
        .join(targets_ch)
        .combine(label_modes_ch)
    // mil_input_ch: [cluster, mil_bags, targets, label_mode]
    if (params.run_mil == "true") {
        TRAIN_MIL(mil_input_ch, mil_params_ch, meta_file_ch)
        mil_results_ch = TRAIN_MIL.out.results.groupTuple()
    } else {
        // Emit an empty grouped placeholder per cluster so EVALUATE_MODELS can still run
        mil_results_ch = pseudobulk_ch
            .map { cluster, pb -> [cluster, file("${params.outdir}/mil_models/${cluster}")] }
            .groupTuple()
    }
    // mil_results_ch: [cluster, [mil_dir_full, mil_dir_argmax, mil_dir_argmax_filtered]]

    // Step 2c: MultiMIL — fan out to one job per (cluster × target_mode × fold_idx)
    def multimil_modes = []
    if (params.run_multiMIL_joint == "true") multimil_modes << "joint"
    if (params.run_multiMIL_indep == "true") multimil_modes << "indep"
    if (multimil_modes) {
        multimil_modes_ch = Channel.fromList(multimil_modes)
        fold_idx_ch        = Channel.of(0..(params.n_outer_folds - 1))
        multimil_input_ch  = mil_bags_ch
            .join(targets_ch)
            .combine(multimil_modes_ch)
            .combine(fold_idx_ch)
        // multimil_input_ch: [cluster, mil_bags, targets, target_mode, fold_idx]
        TRAIN_MULTIMIL(multimil_input_ch, meta_file_ch)

        // Broadcast targets by cluster (same Map-lookup trick as the
        // permutation-null join above) rather than a plain join, since
        // grouping by (cluster, target_mode) still leaves a many-cluster
        // key on one side that a plain join could handle wrong.
        multimil_fold_grouped_ch = TRAIN_MULTIMIL.out.fold_oof
            .groupTuple(by: [0, 1])
            .combine(targets_map_ch)
            .map { cluster, target_mode, fold_csvs, tgtMap ->
                [cluster, target_mode, fold_csvs, tgtMap[cluster]]
            }
        // multimil_fold_grouped_ch: [cluster, target_mode, [fold0_oof, ..., fold4_oof], targets]
        COMBINE_MULTIMIL_FOLDS(multimil_fold_grouped_ch)

        multimil_results_ch = COMBINE_MULTIMIL_FOLDS.out.results.groupTuple()
        // multimil_results_ch: [cluster, [multimil_joint_dir, multimil_indep_dir]] (however many ran)

        // Fold MultiMIL's combined result dirs into the SAME list already
        // used as EVALUATE_MODELS' --tree_dirs, rather than adding a new
        // input to 004_evaluate.py — it already glob-scans any directory
        // it's given for *_summary_metrics.csv.
        tree_results_ch = tree_results_ch
            .join(multimil_results_ch, remainder: true)
            .map { cluster, treeDirs, mmDirs -> [cluster, treeDirs + (mmDirs ?: [])] }
    }

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
