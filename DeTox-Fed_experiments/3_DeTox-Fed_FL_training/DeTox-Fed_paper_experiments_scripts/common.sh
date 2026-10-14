# Shared settings for the experiment scripts. Source this file; do not run it.

set -euo pipefail

DATASET="${1:-}"
if [[ "${DATASET}" != "pleroma" && "${DATASET}" != "mastodon" ]]; then
    echo "Usage: $0 <pleroma|mastodon>" >&2
    exit 1
fi

# Run from the repository root, where data/ and results/ live.
cd "$(dirname "$0")/../../.."

TRAINING_DIR="DeTox-Fed_experiments/3_DeTox-Fed_FL_training"
ARGS_FILE="${TRAINING_DIR}/args_for_training/${DATASET}.args"
RESULTS_DIR="results/detoxfed/${DATASET}"
DATA_DIR="data/${DATASET}"
SEEDS=(40 400 4000 10 100 1000)
ALL_FEATURES="--use_deepwalk --use_author --use_sentiment --use_conv_length --use_num_user --use_reaction_time"

mkdir -p "${RESULTS_DIR}"

run() {
    echo "=== $*"
    python "${TRAINING_DIR}/train_DeTox-Fed.py" "@${ARGS_FILE}" "$@"
}
