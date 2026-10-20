#!/bin/bash
# Train one seed per array task on the feature manifests in $FEATURES (space-separated).
#   sbatch --export=ALL,FEATURES="a/features.jsonl b/features.jsonl",RUN=... --array=0-2 boundary/slurm/train.sbatch
#SBATCH -J boundary-train
#SBATCH -A HARRISON-SL3-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH -o %x-%A_%a.out
set -euo pipefail
source "${REPO:-$HOME/transcPybe}/boundary/slurm/env.sh"   # sbatch runs a spooled copy, so not $0
SEED=${SLURM_ARRAY_TASK_ID:-0}
python -m boundary.train $FEATURES --out "$RUN/seed$SEED" --seed "$SEED" ${TRAIN_ARGS:-}
