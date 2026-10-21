#!/bin/bash
# Separate every mix in $OUT/mixes.jsonl with the pipeline's BS-RoFormer, from an Apptainer image ($SIF).
# Resumable: resubmit the same command if it runs out of time. Since 21 Oct the bulk data has no separator
# (prepare.sbatch); this is for the validation-size set of real separated audio.
#   sbatch --export=ALL,OUT=...,SIF=... boundary/slurm/separate.sbatch
#SBATCH -J boundary-separate
#SBATCH -A HARRISON-SL3-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH -o %x-%j.out
set -euo pipefail
source "${REPO:-$HOME/transcPybe}/boundary/slurm/env.sh"   # sbatch runs a spooled copy, so not $0
command -v apptainer >/dev/null || { echo "apptainer not found on $(hostname)"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m boundary.separate "$OUT/mixes.jsonl" "$OUT" --backend apptainer --sif "$SIF"
