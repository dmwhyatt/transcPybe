#!/bin/bash
# One part of a dataset per array task, from vocals to a feature shard (boundary/prepare.py), on one A100.
#   bash boundary/slurm/submit.sh prepare VOCALS=$DATA/m4singer.jsonl ACCOMP=$DATA/musdb18hq DEPTH=2 \
#       OUT=$DATA/m4-musdb NPARTS=16 --array=0-15
# The per-recording files live on the node's own disk, which CSD3 deletes when the job ends; only the shard goes to
# $OUT on hpc-work (https://docs.hpc.cam.ac.uk/hpc/user-guide/io_management.html). Each GPU comes with 32 cores
# (https://docs.hpc.cam.ac.uk/hpc/user-guide/a100.html), shared by the feature workers. SL3 jobs stop at 12 h
# (https://docs.hpc.cam.ac.uk/hpc/user-guide/policies.html): choose NPARTS so a part fits well inside that.
# A rerun skips finished parts.
#SBATCH -J boundary-prepare
#SBATCH -A HARRISON-SL3-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
set -euo pipefail
source "${REPO:-$HOME/transcPybe}/boundary/slurm/env.sh"   # sbatch runs a spooled copy, so not $0
PART=${SLURM_ARRAY_TASK_ID:-0}
LOCAL=/local
[ -w "$LOCAL" ] || LOCAL=/ramdisks
[ -w "$LOCAL" ] || LOCAL=${TMPDIR:-/tmp}
echo "scratch on $LOCAL: $(df -h "$LOCAL" | tail -1)"
python -m boundary.prepare "$VOCALS" "$OUT" --accomp "$ACCOMP" --accomp-depth "${DEPTH:-2}" \
    --variants "${VARIANTS:-1}" --seed "${SEED:-0}" --var-db ${VAR_DB:-20 35} --part "$PART/${NPARTS:-1}" \
    --scratch "$LOCAL/$USER-${SLURM_JOB_ID:-local}-$PART" --workers "${WORKERS:-8}" --device cuda
