#!/usr/bin/env bash
# Learning rate and number of local epochs.
#
# Usage (from the repository root):
#   DeTox-Fed_experiments/3_DeTox-Fed_FL_training/DeTox-Fed_paper_experiments_scripts/run_hyperparameters.sh <pleroma|mastodon>

source "$(dirname "$0")/common.sh"

for LR in 0.1 0.01 0.001 0.0001; do
    for SEED in "${SEEDS[@]}"; do
        run ${ALL_FEATURES} --learning_rate "${LR}" --random_seed "${SEED}" \
            --tag "lr=${LR}" --results_csv "${RESULTS_DIR}/learning_rate_experiment.csv"
    done
done

for EPOCHS in 1 3 5 10 20; do
    for SEED in "${SEEDS[@]}"; do
        run ${ALL_FEATURES} --local_epochs "${EPOCHS}" --random_seed "${SEED}" \
            --tag "local_epochs=${EPOCHS}" --results_csv "${RESULTS_DIR}/local_epochs_experiment.csv"
    done
done
