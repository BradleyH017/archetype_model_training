#!/bin/bash
# run_finalize_model.sh
# Wrapper for 006_finalize_model.py (manual, not part of the Nextflow DAG):
# refits ONE final, deployable model on ALL currently available samples for a
# given cluster, using the same hvg_select_method/covariate/hvg_pct/hvg_max
# settings currently live in conf/training.config. Loads Singularity the same
# way run_training.sh does, then runs 006 directly inside the container
# (006 is never invoked through Nextflow, so it has to be wrapped by hand).
#
# Usage:
#   ./run_finalize_model.sh --cluster B --model elasticnet_indep [--config conf/training.config]
#
# --model must be one of: xgboost_indep, xgboost_joint, rf, elasticnet,
# lasso, elasticnet_indep, lasso_indep. ("mil" is not supported by
# 006_finalize_model.py itself -- see its module docstring for why.)
#
# hvg_pct/hvg_max/hvg_select_method/meta_file/train_only_covs/train_test_covs/
# n_outer_folds/random_seed/label_mode and the actual TRAIN_TREE_MODELS
# container are all read directly from --config (default
# conf/training.config) via scripts/999-parse_training_config.py, so they
# only ever need to be specified in that one file -- see that script's
# docstring for what it can and can't parse. The hyperparameter grid paths
# (xgb_params/rf_params/elasticnet_params/lasso_params) are NOT config
# params at all -- they're hardcoded paths in workflow/main.nf itself -- so
# they stay hardcoded below to match that.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# ── Parse args ───────────────────────────────────────────────────────────────
CLUSTER=""
MODEL=""
CONFIG_FILE="conf/training.config"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cluster) CLUSTER="$2";     shift 2 ;;
        --model)   MODEL="$2";       shift 2 ;;
        --config)  CONFIG_FILE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$CLUSTER" || -z "$MODEL" ]]; then
    echo "Usage: $0 --cluster <A|B|C|D|E> --model <model_name> [--config <path>]" >&2
    exit 1
fi

case "$MODEL" in
    xgboost_indep|xgboost_joint|rf|elasticnet|lasso|elasticnet_indep|lasso_indep) ;;
    *)
        echo "Unsupported --model '${MODEL}'. 006_finalize_model.py supports:" >&2
        echo "  xgboost_indep  xgboost_joint  rf  elasticnet  lasso  elasticnet_indep  lasso_indep" >&2
        echo "(\"mil\" is not supported -- see 006_finalize_model.py's module docstring)" >&2
        exit 1
        ;;
esac

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config file not found: ${CONFIG_FILE}" >&2
    exit 1
fi

# ── Settings read from $CONFIG_FILE ─────────────────────────────────────────
eval "$(python3 scripts/999-parse_training_config.py --config "$CONFIG_FILE")"
# Populates: HVG_PCT, HVG_MAX, HVG_SELECT_METHOD, META_FILE, TRAIN_ONLY_COVS,
# TRAIN_TEST_COVS, N_OUTER_FOLDS, RANDOM_SEED, LABEL_MODE, CONTAINER

# 006_finalize_model.py only uses whichever grid its own
# _param_path_for(--model) maps to -- passing all four every time is
# harmless, the unused ones are simply ignored. NOT config params -- these
# paths are hardcoded in workflow/main.nf itself, so they're hardcoded here
# to match, not read from $CONFIG_FILE.
XGB_PARAMS="scripts/param_grids/xgboost_params.json"
RF_PARAMS="scripts/param_grids/rf_params.json"
ELASTICNET_PARAMS="scripts/param_grids/elasticnet_params.json"
LASSO_PARAMS="scripts/param_grids/lasso_params.json"

PSEUDOBULK="results/prepared_data/${CLUSTER}.pseudobulk.npz"
TARGETS="results/prepared_data/${CLUSTER}.targets.csv"
OUTDIR="results/final_models"

mkdir -p "$OUTDIR"

module load cellgen/singularity

echo "Finalizing: cluster=${CLUSTER}  model=${MODEL}  label_mode=${LABEL_MODE}  "
echo "  hvg_select_method=${HVG_SELECT_METHOD}  config=${CONFIG_FILE}  container=${CONTAINER}"

# TRAIN_TEST_COVS/TRAIN_ONLY_COVS are comma-joined single strings (matching
# how conf/training.config itself stores them) -- 006's own
# split_covariate_names() handles that directly, same as every other script
# in this pipeline, so they're passed quoted as ONE token each, not split.
singularity exec -B /lustre -B /software --cleanenv "$CONTAINER" \
    python scripts/006_finalize_model.py \
        --cluster "$CLUSTER" \
        --pseudobulk "$PSEUDOBULK" \
        --targets "$TARGETS" \
        --model "$MODEL" \
        --xgb_params "$XGB_PARAMS" \
        --rf_params "$RF_PARAMS" \
        --elasticnet_params "$ELASTICNET_PARAMS" \
        --lasso_params "$LASSO_PARAMS" \
        --hvg_pct "$HVG_PCT" \
        --hvg_max "$HVG_MAX" \
        --label_mode "$LABEL_MODE" \
        --n_folds "$N_OUTER_FOLDS" \
        --seed "$RANDOM_SEED" \
        --meta_file "$META_FILE" \
        --train_only_covs "$TRAIN_ONLY_COVS" \
        --train_test_covs "$TRAIN_TEST_COVS" \
        --hvg_select_method "$HVG_SELECT_METHOD" \
        --outdir "$OUTDIR"

echo "Done. Bundle saved to ${OUTDIR}/${CLUSTER}_${MODEL}_${LABEL_MODE}_final_model.joblib"
