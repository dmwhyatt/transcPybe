#!/bin/bash
# Submit one stage, logging to $DATA/logs/<stage>-<jobid>.out, and print the job id and log path.
#   bash boundary/slurm/submit.sh STAGE [NAME=VALUE ...] [--sbatch-option ...]
#   bash boundary/slurm/submit.sh train FEATURES=$DATA/m4-musdb/features/features.jsonl RUN=$DATA/runs/r1 --array=0-2
# STAGE is mix, separate, features or train. NAME=VALUE pairs are exported to the job. Set MAIL=you@cam.ac.uk to be
# emailed by Slurm when the job ends or fails.
set -euo pipefail
stage=$1; shift
REPO=${REPO:-$HOME/transcPybe}; DATA=${DATA:-/rds/user/$USER/hpc-work/boundary}
exports="ALL,REPO=$REPO,DATA=$DATA"; opts=()
for a in "$@"; do case $a in --*) opts+=("$a") ;; *=*) exports="$exports,$a" ;; *) echo "not NAME=VALUE or --option: $a" >&2; exit 2 ;; esac; done
[ -n "${MAIL:-}" ] && opts+=("--mail-type=END,FAIL" "--mail-user=$MAIL")
pattern="%x-%j.out"
for o in ${opts[@]+"${opts[@]}"}; do case $o in --array=*) pattern="%x-%A_%a.out" ;; esac; done
mkdir -p "$DATA/logs"; cd "$DATA"
id=$(sbatch --parsable --export="$exports" -o "$DATA/logs/$pattern" ${opts[@]+"${opts[@]}"} "$REPO/boundary/slurm/$stage.sbatch")
id=${id%%;*}
if [ "$pattern" = "%x-%j.out" ]; then log="boundary-$stage-$id.out"; else log="boundary-$stage-${id}_<task>.out"; fi
echo "submitted $stage as job $id; log: $DATA/logs/$log"
