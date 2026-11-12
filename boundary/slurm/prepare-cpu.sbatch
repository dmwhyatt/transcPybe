#!/bin/bash
# prepare.sbatch's work on a CPU node, for a tracker that needs no GPU: PESTO (run on the CPU, which is also how the
# 4 November front-end comparison ran it) or pYIN. Mixing, spectrograms and packing were CPU work in round 1 as well;
# only CREPE used the GPU. One part per array task, as in prepare.sbatch:
#   bash boundary/slurm/submit.sh prepare-cpu VOCALS=... ACCOMP=... DEPTH=2 OUT=... NPARTS=45 F0_BACKEND=pyin \
#       WORKERS=24 --array=0-44
# 32 cores: as many as a part had beside its A100 in round 1, so mixing runs as wide. Memory: Ice Lake's default of
# 3,370 MB per core (more is charged as extra cores), about 105 GB here, against the 250 GB a GPU brought: ample for one
# part's audio and the workers. Time, not yet measured on a CPU node: round 1's parts took
# 5-8.5 min on an A100 node, CREPE on the GPU. The pitch tracking that replaces it is small: on 6 Nov, PESTO took
# 57.5 s on 8 cores for 17 min of audio (about 3 core-s per 7 s clip) and pYIN 64 s on one core (0.5 core-s per clip),
# so a part's 1,000 clips add 1-2 min on 32 cores. 30 min allows for slower cores; the trial part (retrain-tracker.sh)
# measures it, and the waiting parts can then be trimmed with scontrol.
# A rerun skips finished parts.
#SBATCH -J boundary-prepare-cpu
#SBATCH -A HARRISON-SL3-CPU
#SBATCH -p icelake
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 00:30:00
set -euo pipefail
source "${REPO:-$HOME/transcPybe}/boundary/slurm/env.sh"   # sbatch runs a spooled copy, so not $0
: "${F0_BACKEND:?prepare-cpu is for a tracker without a GPU: set F0_BACKEND=pesto or pyin}"
PART=${SLURM_ARRAY_TASK_ID:-0}
LOCAL=/local
[ -w "$LOCAL" ] || LOCAL=/ramdisks
[ -w "$LOCAL" ] || LOCAL=${TMPDIR:-/tmp}
echo "scratch on $LOCAL: $(df -h "$LOCAL" | tail -1)"
python -m boundary.prepare "$VOCALS" "$OUT" --accomp "$ACCOMP" --accomp-depth "${DEPTH:-2}" \
    --variants "${VARIANTS:-1}" --seed "${SEED:-0}" --var-db ${VAR_DB:-20 35} --part "$PART/${NPARTS:-1}" \
    --scratch "$LOCAL/$USER-${SLURM_JOB_ID:-local}-$PART" --workers "${WORKERS:-24}" --device cpu \
    --f0-backend "$F0_BACKEND"
