#!/bin/bash
# Build the vocals manifest for M4Singer (boundary/sources.py). It reads every clip's labels, tens of thousands of
# small MIDI and TextGrid files, so it runs as a job, not on a login node, which is "not intended for production
# workload" (https://docs.hpc.cam.ac.uk/hpc/user-guide/connecting.html).
#   bash boundary/slurm/submit.sh sources ROOT=$DATA/m4singer OUT=$DATA/m4singer-between.jsonl ONSETS=between
# 1 core, 30 min: single-threaded and light (38 MB). On a login node it spent over a minute mostly waiting on the
# shared filesystem at 8% of one core. Icelake's default 3,370 MB per core is ample
# (https://docs.hpc.cam.ac.uk/hpc/user-guide/icelake.html).
#SBATCH -J boundary-sources
#SBATCH -A HARRISON-SL3-CPU
#SBATCH -p icelake
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 00:30:00
set -euo pipefail
source "${REPO:-$HOME/transcPybe}/boundary/slurm/env.sh"   # sbatch runs a spooled copy, so not $0
python -m boundary.sources m4singer "$ROOT" "$OUT" --onsets "${ONSETS:-between}" --fraction "${FRACTION:-0.5}"
wc -l < "$OUT"
