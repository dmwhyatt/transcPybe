#!/bin/bash
# Log-mel, transcPybe f0 (CREPE full) and targets for every stem in $OUT/stems.jsonl.
#   sbatch --export=ALL,OUT=... boundary/slurm/features.sbatch
#SBATCH -J boundary-features
#SBATCH -A HARRISON-SL3-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH -o %x-%j.out
set -euo pipefail
source "${REPO:-$HOME/transcPybe}/boundary/slurm/env.sh"   # sbatch runs a spooled copy, so not $0
python -m boundary.features "$OUT/stems.jsonl" "$OUT/features" --device cuda
