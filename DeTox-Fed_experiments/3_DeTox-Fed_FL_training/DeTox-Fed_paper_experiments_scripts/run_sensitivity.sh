#!/usr/bin/env bash
# Sensitivity of DeTox-Fed to the label budget, the toxicity threshold, the
# number of clients per round and the minimum conversation length.
#
# Usage (from the repository root):
#   DeTox-Fed_experiments/3_DeTox-Fed_FL_training/DeTox-Fed_paper_experiments_scripts/run_sensitivity.sh <pleroma|mastodon>

source "$(dirname "$0")/common.sh"

if [[ "${DATASET}" == "pleroma" ]]; then
    CLIENTS_PER_ROUND=(10 25 50 100)
else
    CLIENTS_PER_ROUND=(5 10 15 20)
fi

for LABELS in 20 50 100 200 500; do
    for SEED in "${SEEDS[@]}"; do
        run ${ALL_FEATURES} --max_labels_per_client "${LABELS}" --random_seed "${SEED}" \
            --tag "labels=${LABELS}" --results_csv "${RESULTS_DIR}/max_labels_per_client_experiment.csv"
    done
done

for THRESHOLD in 0.4 0.5 0.6 0.7 0.8; do
    for SEED in "${SEEDS[@]}"; do
        run ${ALL_FEATURES} --toxicity_binary_threshold "${THRESHOLD}" --random_seed "${SEED}" \
            --tag "threshold=${THRESHOLD}" --results_csv "${RESULTS_DIR}/toxicity_threshold_experiment.csv"
    done
done

for CLIENTS in "${CLIENTS_PER_ROUND[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        run ${ALL_FEATURES} --client_num_per_round "${CLIENTS}" --random_seed "${SEED}" \
            --tag "clients=${CLIENTS}" --results_csv "${RESULTS_DIR}/clients_per_round_experiment.csv"
    done
done

# Conversations of any length, more than 5 toots, more than 10 toots.
for LENGTH in all length_5 length_10; do
    for SEED in "${SEEDS[@]}"; do
        run ${ALL_FEATURES} \
            --features_npy "${DATA_DIR}/${DATASET}_conversation_features_${LENGTH}.npy" \
            --features_index "${DATA_DIR}/${DATASET}_index_conversation_features_${LENGTH}.parquet" \
            --random_seed "${SEED}" \
            --tag "conversations=${LENGTH}" --results_csv "${RESULTS_DIR}/conversation_length_experiment.csv"
    done
done
