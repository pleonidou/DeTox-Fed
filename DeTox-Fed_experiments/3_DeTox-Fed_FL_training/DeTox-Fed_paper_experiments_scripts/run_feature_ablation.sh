#!/usr/bin/env bash
# Contribution of each feature block.
#
# Usage (from the repository root):
#   DeTox-Fed_experiments/3_DeTox-Fed_FL_training/DeTox-Fed_paper_experiments_scripts/run_feature_ablation.sh <pleroma|mastodon>

source "$(dirname "$0")/common.sh"

ABLATIONS=(
    "deepwalk_only|--use_deepwalk"
    "deepwalk_author|--use_deepwalk --use_author"
    "deepwalk_author_sentiment|--use_deepwalk --use_author --use_sentiment"
    "author_sentiment_conversation|--use_author --use_sentiment --use_conv_length --use_num_user --use_reaction_time"
    "all_features|${ALL_FEATURES}"
)

for ABLATION in "${ABLATIONS[@]}"; do
    NAME="${ABLATION%%|*}"
    FLAGS="${ABLATION#*|}"
    for SEED in "${SEEDS[@]}"; do
        run ${FLAGS} --random_seed "${SEED}" \
            --tag "${NAME}" --results_csv "${RESULTS_DIR}/feature_ablation_experiment.csv"
    done
done
