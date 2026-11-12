#!/bin/bash
# Round 2's landmark networks again, on another pitch tracker's features (TRACKER-RETRAIN-PLAN.md): three seeds side by
# side on one A100, with round 2's recipe and labels. Submitted by retrain-tracker.sh after the tracker's prepare parts.
#   TRACKER=pesto SHARDS=$DATA/m4gts-musdb-pesto/shards RUN=$DATA/runs/r2-lm-pesto LABELLED=$DATA/m4gts-musdb/all-lm.jsonl
#   M4=$DATA/m4gts-musdb/m4-lm.jsonl
# Steps: point round 2's labelled manifests at the new shards (swap_features, seconds); train, with the vowel targets
# placed by CREPE full's voicing (--label-voicing), so only the inputs differ from round 2's; meanwhile, fit the map from
# the tracker's confidence to CREPE's on the validation singers (voicing_map, minutes, beside the seeds).
# Time: round 2's three seeds took 1.92 h on 28 Oct (csd3-runs/r2-lm/seed0/log.jsonl); loading each recording's CREPE
# twin as well adds minutes, so 3 h, as train.sbatch asks. Memory: the GPU's 250 GB holds three seeds' features and the
# twins' shards while they load.
#SBATCH -J boundary-train-tracker
#SBATCH -A HARRISON-SL3-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 03:00:00
set -euo pipefail
source "${REPO:-$HOME/transcPybe}/boundary/slurm/env.sh"   # sbatch runs a spooled copy, so not $0
: "${TRACKER:?}" "${SHARDS:?}" "${RUN:?}" "${LABELLED:?}" "${M4:?}"
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI-http://musix.mus.cam.ac.uk:5000}
mkdir -p "$RUN"
python -m boundary.swap_features "$LABELLED" "$RUN/all-lm-$TRACKER.jsonl" --shards "$SHARDS" --expect-f0-backend "$TRACKER"
python -m boundary.swap_features "$M4" "$RUN/m4-lm-$TRACKER.jsonl" --shards "$SHARDS" --expect-f0-backend "$TRACKER"

SEEDS=${SEEDS:-0 1 2}
n=$(echo $SEEDS | wc -w)
cores=$OMP_NUM_THREADS
export OMP_NUM_THREADS=$(( cores / n > 0 ? cores / n : 1 ))
export MKL_NUM_THREADS=$OMP_NUM_THREADS
# round 2's arguments (csd3-runs/r2-lm/seed0/args.json), the rest at their defaults, plus --label-voicing
ARGS="--steps 30000 --batch 32 --crop-s 4 --lr 1e-3 --weight-decay 0.01 --warmup 500 --pos-weight 8 --eval-every 500
      --channels 32,64,96 --hidden 128 --dropout 0.2 --augment --onset-labels landmarks
      --label-voicing $LABELLED --extra-valid m4=$RUN/m4-lm-$TRACKER.jsonl"
pids=()
for s in $SEEDS; do
    ( python -m boundary.train "$RUN/all-lm-$TRACKER.jsonl" --out "$RUN/seed$s" --seed "$s" $ARGS ${TRAIN_ARGS:-} 2>&1 \
        | sed -u "s/^/[seed $s] /" ) &
    pids+=($!)
done
# on one core beside the seeds; a failure here costs the map, not the networks
OMP_NUM_THREADS=1 python -m boundary.voicing_map "$LABELLED" "$RUN/all-lm-$TRACKER.jsonl" "$RUN/voicing-map.json" \
    || echo "voicing_map failed (exit $?); the networks train on"
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
exit $rc
