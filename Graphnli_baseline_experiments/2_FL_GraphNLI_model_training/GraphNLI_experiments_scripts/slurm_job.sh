#!/bin/bash
# Example SLURM job for one training run. Adjust the partition, resources and
# environment activation to your cluster.
#
# Usage (from the repository root):
#   sbatch Graphnli_baseline_experiments/2_FL_GraphNLI_model_training/GraphNLI_experiments_scripts/slurm_job.sh <pleroma|mastodon>

#SBATCH --job-name=fl_graphnli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/fl_graphnli_%j.out
#SBATCH --error=logs/fl_graphnli_%j.err

set -euo pipefail

DATASET="${1:-mastodon}"
TRAINING_DIR="Graphnli_baseline_experiments/2_FL_GraphNLI_model_training"

# Activate your environment here, e.g.:
# source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate <env>

echo "Job ${SLURM_JOB_ID} on $(hostname), started $(date)"
nvidia-smi

mkdir -p logs
python -u "${TRAINING_DIR}/train.py" "@${TRAINING_DIR}/args_for_training/${DATASET}.args"

echo "Finished $(date)"
