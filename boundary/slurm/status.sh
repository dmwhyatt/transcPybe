#!/bin/bash
# What is running, how far each stage has got, how training is going, and how many GPU hours have gone.
#   bash ~/transcPybe/boundary/slurm/status.sh            (on CSD3; Claude runs the same over the shared ssh connection)
DATA=${DATA:-/rds/user/$USER/hpc-work/boundary}
echo "=== $(date -u +%FT%TZ)  jobs"
squeue --me -o "%.14i %.22j %.9T %.10M %.10L %.4D %R" 2>/dev/null || echo "(squeue unavailable)"
echo; echo "=== latest progress of running jobs"
for id in $(squeue --me -h -t R -o "%i" 2>/dev/null); do
  log=$(ls -t "$DATA"/logs/*-"${id}".out "$DATA"/logs/*-"${id%%_*}"_*.out 2>/dev/null | head -1)
  [ -n "$log" ] && { echo "--- $id  $(basename "$log")"; grep -E "^[0-9]{4}-|batch [0-9]+:|Error|Traceback" "$log" | tail -n 2; }
done
echo; echo "=== stages (rows done / rows in the manifest)"
for d in "$DATA"/*/; do
  [ -f "$d/mixes.jsonl" ] || continue
  m=$(wc -l < "$d/mixes.jsonl"); s=$( [ -f "$d/stems.jsonl" ] && wc -l < "$d/stems.jsonl" || echo 0 )
  f=$( [ -f "$d/features/features.jsonl" ] && wc -l < "$d/features/features.jsonl" || echo 0 )
  printf "  %-30s mixes %7s   stems %7s   features %7s\n" "$(basename "$d")" "$m" "$s" "$f"
done
echo; echo "=== training runs (latest evaluation; best score)"
python3 - "$DATA" <<'PY'
import glob, json, os, sys
for log in sorted(glob.glob(os.path.join(sys.argv[1], "runs", "*", "log.jsonl"))):
    rows = [json.loads(l) for l in open(log) if l.strip()]
    if not rows: continue
    r = rows[-1]; best = max(x["score"] for x in rows)
    print(f"  {os.path.basename(os.path.dirname(log)):28s} step {r['step']:>6}  score {r['score']:.3f} (best {best:.3f})  onset F {r['onset_f']:.3f}"
          f"  timing {r.get('timing_median_ms', float('nan')):+.0f} ms  eta {r.get('eta_s', 0) // 60} min")
PY
echo; echo "=== GPU hours (ledger: finished jobs) and balance"
python3 - "$DATA/ledger.tsv" <<'PY'
import sys, re
used, n = 0.0, 0
try:
    for line in open(sys.argv[1]):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 7 and f[4] == "end":
            g = int(f[5].split("=")[1]); m = re.search(r"elapsed_s=(\d+)", f[6])
            if g and m: used += g * int(m.group(1)) / 3600; n += 1
except FileNotFoundError:
    pass
print(f"  {used:.1f} GPU hours over {n} finished GPU jobs (plan estimate: 10-15 A100 hours without separation, 21 Oct)")
PY
mybalance 2>/dev/null | grep -E "GPU|Account" || true
echo; echo "=== storage"
quota 2>/dev/null | grep -E "Filesystem|hpc-work|/rds-d6/user|/home" || true
