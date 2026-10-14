#!/usr/bin/env bash
# Train the federated GraphNLI baseline once per seed.
# Each run writes to results/graphnli/<dataset>/run_<seed>/, the layout read by
# 3_conversation-level-labels-evaluation/conversation_level_evaluation.ipynb.
#
# Usage (from the repository root):
#   Graphnli_baseline_experiments/2_FL_GraphNLI_model_training/GraphNLI_experiments_scripts/run_seeds.sh <pleroma|mastodon>

set -euo pipefail

DATASET="${1:-}"
if [[ "${DATASET}" != "pleroma" && "${DATASET}" != "mastodon" ]]; then
    echo "Usage: $0 <pleroma|mastodon>" >&2
    exit 1
fi

# Run from the repository root, where data/, models/ and results/ live.
cd "$(dirname "$0")/../../.."

TRAINING_DIR="Graphnli_baseline_experiments/2_FL_GraphNLI_model_training"
SEEDS=(4000 400 40)  # seeds of the three paper runs

for SEED in "${SEEDS[@]}"; do
    echo "=== ${DATASET}, seed ${SEED}"
    python -u "${TRAINING_DIR}/train.py" "@${TRAINING_DIR}/args_for_training/${DATASET}.args" \
        --random_seed "${SEED}" \
        --log_folder "results/graphnli/${DATASET}/run_${SEED}"
done
