#!/bin/bash
# Train on the feature manifests in $FEATURES: shards' .jsonl files or features.jsonl, space-separated; a glob such
# as "$DATA/m4-musdb/shards/*.jsonl" is expanded here, on the node.
# With SEEDS="0 1 2" the seeds train side by side on one GPU, each on its share of the job's cores and each loading
# the features into its own memory. The network is small: one run on an A100 went no faster than on musix's 3080,
# because the CPU sets the pace. GPUs are billed by the hour each is held
# (https://docs.hpc.cam.ac.uk/hpc/user-guide/a100.html). Without SEEDS, one seed per array task.
#   bash boundary/slurm/submit.sh train FEATURES="$DATA/m4-musdb/shards/*.jsonl" RUN=$DATA/runs/r1 SEEDS="0 1 2"
#   bash boundary/slurm/submit.sh train FEATURES=... RUN=... --array=0-2
#SBATCH -J boundary-train
#SBATCH -A HARRISON-SL3-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH -o %x-%A_%a.out
set -euo pipefail
source "${REPO:-$HOME/transcPybe}/boundary/slurm/env.sh"   # sbatch runs a spooled copy, so not $0
SEEDS=${SEEDS:-${SLURM_ARRAY_TASK_ID:-0}}
n=$(echo $SEEDS | wc -w)
export OMP_NUM_THREADS=$(( OMP_NUM_THREADS / n > 0 ? OMP_NUM_THREADS / n : 1 ))
export MKL_NUM_THREADS=$OMP_NUM_THREADS
mkdir -p "$RUN"
pids=()
for s in $SEEDS; do
    # each line of a run's output tagged with its seed, so status.sh can tell them apart in the one job log
    ( python -m boundary.train $FEATURES --out "$RUN/seed$s" --seed "$s" ${TRAIN_ARGS:-} 2>&1 | sed -u "s/^/[seed $s] /" ) &
    pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
exit $rc
