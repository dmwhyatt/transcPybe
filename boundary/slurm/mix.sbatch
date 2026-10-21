#!/bin/bash
# Mix every vocal in $VOCALS over the accompaniments in $ACCOMP (CPU only). VAR_DB="20 35" for the no-separator
# stand-in. For the bulk data, prepare.sbatch mixes inside the GPU job instead: SL3 CPU jobs can queue for hours.
#   sbatch --export=ALL,VOCALS=...,ACCOMP=...,DEPTH=2,OUT=... boundary/slurm/mix.sbatch
#SBATCH -J boundary-mix
#SBATCH -A HARRISON-SL3-CPU
#SBATCH -p icelake
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 04:00:00
#SBATCH -o %x-%j.out
set -euo pipefail
source "${REPO:-$HOME/transcPybe}/boundary/slurm/env.sh"   # sbatch runs a spooled copy, so not $0
python -m boundary.mix "$VOCALS" "$OUT" --accomp "$ACCOMP" --accomp-depth "${DEPTH:-2}" \
    --variants "${VARIANTS:-1}" --seed "${SEED:-0}" --workers 32 ${KEEP_DRY:+--keep-dry} ${VAR_DB:+--var-db $VAR_DB}
