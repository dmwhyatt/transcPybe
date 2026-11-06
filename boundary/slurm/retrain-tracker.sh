#!/bin/bash
# Retrain round 2's landmark networks on another pitch tracker's features (TRACKER-RETRAIN-PLAN.md). From a login node:
#   nice -n 19 bash boundary/slurm/retrain-tracker.sh pesto
#   nice -n 19 bash boundary/slurm/retrain-tracker.sh pyin
# It checks the setup, then submits, each waiting for the one before to succeed:
#   1. part 0 of the training data, a trial (round 1 did the same on 22 Oct);
#   2. the other parts;
#   3. training (train-tracker.sbatch).
# The mixes are round 1's: its vocals, accompaniment and number of parts come from its .done records, and mixing is
# seeded per recording, so only the pitch features differ. Outputs: $DATA/m4gts-musdb-TRACKER/shards and
# $DATA/runs/r2-lm-TRACKER. A rerun skips finished parts. MAIL=you@cam.ac.uk adds Slurm's end and failure emails.
set -euo pipefail
TRACKER=${1:?usage: retrain-tracker.sh pesto|pyin}
REPO=${REPO:-$HOME/transcPybe}; DATA=${DATA:-/rds/user/$USER/hpc-work/boundary}; VENV=${VENV:-$HOME/venvs/boundary}
TRANSCPYBE_SRC=${TRANSCPYBE_SRC:-$HOME/transcPybe-fe/src}   # claude/pitch-front-ends: the PESTO tracker
ROUND1=${ROUND1:-$DATA/m4gts-musdb}
OUT=${OUT:-$DATA/m4gts-musdb-$TRACKER}; RUN=${RUN:-$DATA/runs/r2-lm-$TRACKER}
LABELLED=${LABELLED:-$ROUND1/all-lm.jsonl}; M4=${M4:-$ROUND1/m4-lm.jsonl}   # round 2's (csd3-runs/r2-lm/seed0/args.json)
# Feature processes per part: 8 share the GPU for PESTO, as for CREPE; pYIN runs on the CPU, so 24 of the 32 cores.
case $TRACKER in pesto) WORKERS=${WORKERS:-8} ;; pyin) WORKERS=${WORKERS:-24} ;; *) echo "pesto or pyin, not $TRACKER" >&2; exit 2 ;; esac

for f in "$LABELLED" "$M4"; do [ -s "$f" ] || { echo "missing $f" >&2; exit 1; }; done
# transcPybe with the tracker (and, for PESTO, its chunking and TF32 fix of 4e23d5b), and pesto-pitch in the venv
PYTHONPATH="$REPO:$TRANSCPYBE_SRC" "$VENV/bin/python" - "$TRACKER" <<'PY'
import sys
import transcpybe, transcpybe.f0 as f0
tracker = sys.argv[1]
assert tracker in f0.f0_backends(), f"{transcpybe.__file__} has trackers {f0.f0_backends()}, not {tracker}"
if tracker == "pesto":
    assert hasattr(f0, "_full_float32") and hasattr(f0, "PESTO_CHUNK_S"), "transcPybe older than 4e23d5b"
    import pesto
    print("pesto-pitch from", pesto.__file__)
print("transcpybe from", transcpybe.__file__)
PY
# round 1's mixing arguments, from its .done records; prepare.sbatch's defaults must be what it used
mixing=$("$VENV/bin/python" - "$ROUND1/shards" <<'PY'
import json, pathlib, sys
done = sorted(pathlib.Path(sys.argv[1]).glob("part-*-of-*.done"))
if not done:
    sys.exit(f"no .done records in {sys.argv[1]}")
n = int(done[0].stem.split("-of-")[1])
keys = [json.loads(d.read_text()) for d in done]
if len(done) != n:
    sys.exit(f"round 1 has {len(done)} of {n} parts done")
k = keys[0]
same = ("vocals", "accomp", "accomp_depth", "var_db", "variants", "seed")
if any(any(x[f] != k[f] for f in same) for x in keys):
    sys.exit("round 1's parts were not all made with the same arguments")
if (k["var_db"], k["variants"], k["seed"]) != ([20.0, 35.0], 1, 0):
    sys.exit(f"round 1 used var_db {k['var_db']}, variants {k['variants']}, seed {k['seed']}: not prepare.sbatch's defaults")
print(k["vocals"], k["accomp"], k["accomp_depth"], n)
PY
)
read -r VOCALS ACCOMP DEPTH NPARTS <<< "$mixing"
echo "round 1: $NPARTS parts of $VOCALS over $ACCOMP (depth $DEPTH); $TRACKER features -> $OUT"

cd "$REPO"
job() { sed -n 's/^submitted [a-z-]* as job \([0-9][0-9]*\);.*/\1/p' | tail -1; }
common=(VOCALS="$VOCALS" ACCOMP="$ACCOMP" DEPTH="$DEPTH" OUT="$OUT" NPARTS="$NPARTS" F0_BACKEND="$TRACKER"
        WORKERS="$WORKERS" TRANSCPYBE_SRC="$TRANSCPYBE_SRC")
p0=$(bash boundary/slurm/submit.sh prepare "${common[@]}" --array=0 | tee /dev/stderr | job)
p1=$(bash boundary/slurm/submit.sh prepare "${common[@]}" --array=1-$((NPARTS - 1)) --dependency=afterok:$p0 \
     | tee /dev/stderr | job)
t=$(bash boundary/slurm/submit.sh train-tracker TRACKER="$TRACKER" SHARDS="$OUT/shards" RUN="$RUN" LABELLED="$LABELLED" \
    M4="$M4" TRANSCPYBE_SRC="$TRANSCPYBE_SRC" --dependency=afterok:$p0:$p1 | tee /dev/stderr | job)
[ -n "$p0" ] && [ -n "$p1" ] && [ -n "$t" ] || { echo "a submission failed; see above" >&2; exit 1; }
echo "$TRACKER: trial part $p0, parts $p1, training $t (each after the one before succeeds)"
