# Handoff: re-run the vocadito benchmark, transcPybe `main` vs this branch

Written to be handed to an agent on a GPU machine and followed start to
finish. Everything it needs is here; nothing assumes the conversation that
produced it.

## Objective

Score every system on vocadito in **one run, against one pinned set of
references**, and settle three questions with paired tests:

1. Does branch `claude/modest-bohr-4u67hw` beat `main` on vocadito?
2. Where does it sit against the nine other systems in the September table
   (`transcription-benchmarking/docs/transcpybe-v1.md`)?
3. Are the branch's two contestable defaults right on real data -- the release
   trim at 15 dB (`--no-release-trim` turns it off) and sung-pitch rounding
   (`--pitch-reference tuned` is the alternative)?

The deliverable is a report (format at the end), not code.

## Context

The branch changes note decoding (a third-of-a-semitone pitch grid), glide
handling, legato onset snapping, pitch rounding and note ends, and fixes a
CREPE `--fmin` bug. Its README section "What the realistic corpus says" has
the detail. A first vocadito run on a laptop (CPU, `--annotation-contains
notesA1`) gave:

| run | COn | COnP | COnPOff |
|---|---|---|---|
| main (`c1a016c`) | 0.716 | 0.579 | 0.424 |
| branch | 0.755 | 0.627 | 0.499 |
| branch, `--no-release-trim` | 0.756 | 0.628 | 0.481 |
| branch, `--pitch-reference tuned` | 0.754 | 0.600 | 0.479 |

The September table was scored on a different day and possibly with a
different annotation file per recording, which is why the other systems need
rescoring in the same run. Expect this run's transcPybe numbers within about
0.01 of the table above (GPU and CPU CREPE differ slightly); a gap larger
than 0.02 means something is wrong -- see Troubleshooting.

## Guardrails

- **Do not push** to either repository, and do not commit to `main` of either.
  Do not change any code in `transcription-benchmarking` or `transcpybe`.
- **Use a fresh work directory.** Never delete or reuse an existing
  `bench-work*` directory; the September run's outputs live in one.
- **Do not change transcPybe settings** beyond the three variants below. The
  point is to measure the defaults as shipped.
- Run long steps under `tmux` or `nohup` with output going to a log file.
- If a step fails twice for the same reason, stop and report rather than work
  around it. If one system cannot be built, fetched or run (ROSVOT's weights
  need a human to download), drop that system, carry on, and say so in the
  report.
- Every number in the report comes from a file this run wrote. Never fill a
  gap from memory or from the table above.
- Use `python -m <tool>` rather than bare console scripts where both exist. On
  machines with Anaconda, a bare `pytest` or `transcpybe` can resolve outside
  the venv.

## Prerequisites (check, then stop and ask if any are missing)

- NVIDIA GPU with 10 GB or more free (`nvidia-smi`), Docker with the NVIDIA
  Container Toolkit, about 40 GB free disk.
- Checkouts, both with git access to their (private) remotes:
  `~/transcpybe` and `~/transcription-benchmarking`. If the paths differ,
  adjust the variables below; nothing else depends on them.
- vocadito unpacked so that `$V/Audio` holds 40 `.wav` files and
  `$V/Annotations` holds the note annotations (`*notesA1*`, `*notesA2*`).
  It is on Zenodo, record 5578807, if it has to be downloaded.

## Step 0 -- environment

```shell
cd ~/transcpybe && git fetch origin
cd ~/transcription-benchmarking && git pull --ff-only
python -m venv .venv && . .venv/bin/activate
python -m pip install -e .                       # transcription-benchmark
python -m pip install -e "$HOME/transcpybe[crepe,dev]"   # transcpybe CLI, for step 4
cd orchestrator

export TRANSCPYBE_SRC=$HOME/transcpybe           # its default, ../transcpybe, is case-sensitive
V=$HOME/vocadito
W=$PWD/bench-work-$(date +%Y%m%d)
B="python ./bench.py --work $W"
OTHERS=game,omnizart,phoneme-informed,basic-pitch,crepe-notes,essentia,mirst500,rosvot,aubio

nvidia-smi
$B doctor
```

**Check:** `doctor` reports docker, GPU access from a container, and the
benchmark on PATH as OK, and enough disk. If any fails, stop and report its
output.

## Step 1 -- check the references before anything slow

```shell
transcription-benchmark --inspect $V/Annotations
transcription-benchmark $V/Audio -a $V/Annotations --annotation-contains notesA1 --dry-run
```

**Check:** the dry run pairs all 40 recordings, each with a `notesA1` file.
Save both outputs for the report.

## Step 2 -- build

```shell
$B --only $OTHERS build --prune
$B --only transcpybe --as transcpybe-main   --rev origin/main build
$B --only transcpybe --as transcpybe-branch --rev origin/claude/modest-bohr-4u67hw build
$B fetch
```

**Check:** each transcPybe build logs `source at <sha>`. Record both SHAs.
`main` should be `c1a016c` unless `main` has since moved; the branch should be
`e915d8a` or a descendant of it (`git -C ~/transcpybe merge-base
--is-ancestor e915d8a <sha>`). `fetch` prints what needs a human -- if that is
only ROSVOT, carry on without it.

## Step 3 -- smoke run, then transcribe

```shell
S=$PWD/bench-smoke-$(date +%Y%m%d)
python ./bench.py --work $S --only basic-pitch transcribe --audio $V/Audio --limit 2
python ./bench.py --work $S --only transcpybe --as transcpybe-branch transcribe --audio $V/Audio --limit 2
```

**Check:** each writes 2 `.mid` files under `$S/out/<name>/midi/` (both
systems carry their weights in the image, so the separate work directory
needs no fetch). Then the full run (about 70-90 minutes; omnizart is most of it):

```shell
$B --only $OTHERS transcribe --audio $V/Audio
$B --only transcpybe --as transcpybe-main   transcribe --audio $V/Audio
$B --only transcpybe --as transcpybe-branch transcribe --audio $V/Audio
```

**Check:** `ls $W/out/*/midi | head` -- 40 `.mid` files per system. Note any
system with fewer, and the reason from `$W/logs/<name>.log`.

## Step 4 -- the setting variants (outside Docker)

The orchestrator cannot pass extra flags into the transcPybe container, so
the variants run through the CLI on the host, writing where step 5 will score
them. The plain branch runs through the same path too, so each variant is
compared against a baseline from the same environment.

```shell
git -C $HOME/transcpybe checkout claude/modest-bohr-4u67hw
git -C $HOME/transcpybe pull --ff-only
python -m transcpybe --help | grep -q -- --pitch-reference || echo "WRONG CHECKOUT"

O=$W/out
python -m transcpybe $V/Audio -o $O/cli-branch/midi        --no-json --no-csv --overwrite
python -m transcpybe $V/Audio -o $O/cli-branch-notrim/midi --no-json --no-csv --overwrite --no-release-trim
python -m transcpybe $V/Audio -o $O/cli-branch-tuned/midi  --no-json --no-csv --overwrite --pitch-reference tuned
```

**Check:** 40 `.mid` files in each, and the log line `device: cuda...` (a
CPU run is valid but slow; note it).

## Step 5 -- score, ceiling, tables

```shell
$B score --audio $V/Audio -a $V/Annotations -- --annotation-contains notesA1
$B ceiling -a $V/Annotations

$B table --sort conpoff            > $W/table.txt
$B table --percent                 > $W/table-percent.txt
$B table --markdown --sort conpoff > $W/table.md
cat $W/table.txt
```

**Check:** the footer reads 40 recordings and 2237 reference notes, and
`grep -L '"n_items": 40' $R/*.json` lists no system's file. `transcpybe-branch` and `cli-branch` should
agree to within about 0.01 on every column; if not, see Troubleshooting.

## Step 6 -- paired comparisons

```shell
R=$W/results
pair() { echo "=== $1 -> $2"; transcription-benchmark --compare $R/$1.json $R/$2.json; }
{
  pair transcpybe-main   transcpybe-branch      # Q1
  pair phoneme-informed  transcpybe-branch      # Q2: best COnPOff in September
  pair omnizart          transcpybe-branch
  pair game              transcpybe-branch      # Q2: best COn in September
  pair cli-branch-notrim cli-branch             # Q3: the 15 dB trim
  pair cli-branch-tuned  cli-branch             # Q3: sung vs tuned rounding
} 2>&1 | tee $W/compare.txt
```

(Do not use `bench.py compare` here: it compares every pair, including the
ceiling file, which is 100+ comparisons nobody asked for.)

## Step 7 -- optional, if time allows

Second annotator, to check the ranking holds against the other references.
`score` overwrites results, so keep the first set:

```shell
cp -r $R $W/results-notesA1
$B score --audio $V/Audio -a $V/Annotations -- --annotation-contains notesA2
$B table --sort conpoff > $W/table-notesA2.txt
cp -r $R $W/results-notesA2
cp -r $W/results-notesA1/. $R/          # put the primary results back
```

ROSVOT's fair score needs the onset correction documented in
`docs/benchmarking-rosvot.md`; score it separately and keep both:

```shell
cp $R/rosvot.json $W/rosvot-uncorrected.json
$B --only rosvot score --audio $V/Audio -a $V/Annotations -- --annotation-contains notesA1 --onset-shift 0.041
```

## Report to hand back

Write `$W/REPORT.md` containing, in this order:

1. **Provenance**: date, GPU model, the two transcPybe SHAs from step 2, the
   benchmark repo's `git rev-parse HEAD`, and the annotation flag used.
2. **The table**: `table.md` as is, then `table-percent.txt`.
3. **The six comparisons** from `compare.txt`, each as one line: mean paired
   difference in COn, COnP and COnPOff with its 95% interval, and recordings
   better/worse -- then the raw output below.
4. **Answers** to the three questions, one sentence each, stating only what
   an interval excluding zero supports. "Indistinguishable" is a valid answer.
5. **Anything that went wrong**: systems dropped or short of 40 files, and
   why; any check above that did not pass.
6. Step 7's results, if run.

Then print the report and stop. Do not edit either repository's README or
docs; the figures go in by hand after review.

## Troubleshooting

- **`transcpybe-branch` and `cli-branch` differ by more than ~0.01**: the
  checkout used in step 4 is probably not the branch (the `grep` line in step
  4 catches the worst case), or the two ran on different devices. Report
  both; do not pick one.
- **This run's main/branch differ from the Context table by more than
  0.02**: check the SHAs from step 2, and that the dry run in step 1 paired
  `notesA1` files.
- **CUDA out of memory**: the card is probably shared (`nvidia-smi`). For
  transcPybe via the CLI, add `--crepe-batch-size 128`. For a container,
  wait for the card to be free rather than retrying in a loop.
- **`--rev` fails to resolve**: run `git -C $TRANSCPYBE_SRC fetch origin` and
  retry; use `origin/...` names, not local branch names, which can be stale.
- **`pytest` or `transcpybe` fails importing an unrelated package (e.g.
  `psynet`)**: an Anaconda install is shadowing the venv. Use `python -m`.
- **GAME scores badly on COnPOff**: known, and possibly this repository's
  adapter rather than GAME (see the caveat in `docs/transcpybe-v1.md`). Report
  it; do not tune it.
