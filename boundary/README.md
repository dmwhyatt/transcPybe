# boundary: a learned note-boundary detector for transcPybe

transcPybe tracks sung pitch well. Where it loses to GAME and MIR-ST500 on
separated pop is in placing note boundaries, especially telling a note re-sung on the same pitch
from a held note, and keeping short real notes. That was measured on RWC-P
(`bench-work-rwc-20261019` on musix, `protocol-amendment-1.md`). This folder
trains a small network that marks where notes start and end, frame by frame on
transcPybe's own 10 ms f0 grid, so its output can be fed to the existing
decoder as an `OnsetTrack`. Pitch tracking is left as it is.

This training pipeline is not part of the installed package, and nothing in `src/` depends on it. What it
trained ships there: the networks, their inference code and the two presets are `transcpybe.landmarks`
(`--preset acappella` or `pop`), with round 2's landmark networks (`r2-lm`) and the PESTO and pYIN voicing
maps as package data under CC BY-NC-SA 4.0. `model.py`, `features.py`, `infer.py` and `voicing_map.py` here
are where that code was copied from; a change to the inference code belongs in both.

## Stages

Each is `python -m boundary.<stage> --help`. Every stage reads the previous
stage's JSONL manifest and writes its own, adding fields rather than replacing them.

| stage | in | out | runs on |
|---|---|---|---|
| `sources` | labelled singing | `vocals.jsonl`: audio + notes + split | CPU |
| `mix` | vocals, multitrack accompaniment | `mixes/*.flac` 44.1 kHz stereo + `mixes.jsonl` | CPU |
| `separate` | mixes | `stems/*.wav` (BS-RoFormer lead vocal) + `stems.jsonl` | GPU |
| `features` | stems | `features/*.npz`: log-mel, transcPybe f0 features, targets | GPU (CREPE) |
| `pack` | feature manifest | one shard `.npz` + its `.jsonl` | CPU |
| `prepare` | vocals, accompaniment | `shards/part-*.npz`: mix, features and pack for one part, no separator | GPU |
| `train` | feature or shard manifests | `best.pt`, `last.pt`, `log.jsonl` | GPU |
| `infer` | checkpoint + audio | onset posterior and picked onsets per file | GPU/CPU |

The separator runs in the image built from audio-symbolic-pipeline (on musix:
`sep-bsroformer53:b24ab72`, from `~/sepbuild/Dockerfile`), through
`mss_bulk.py`. That script loads the model once per batch and reuses the
pipeline's own config patch, so chunk size and checkpointing match production.
On musix its output is time-aligned with the input sample for sample, and about
twice the input's level. Every feature is normalised per recording, so the level
does not matter.

Outputs are keyed by what made them. A mix id ends in a digest of its item,
notes, seed, variant, accompaniment set and mixing ranges. A feature file
records the stem, notes and f0 settings it was made from. A re-run therefore
reuses only what still matches and redoes the rest. Silence is never digital
zero: mixes carry a noise floor, and the features floor log-mel and level, so
the network never learns from values a separated stem cannot produce.

## Data

- **Vocals with notes.** M4Singer (on musix at `~/m4singer`: 20,896 clips, 20
  singers, 48 kHz). Its MIDI starts a syllable at its consonant;
  `--onsets vowel` moves starts to the vowel using the TextGrids. Four singers,
  one per voice type, are held out for validation (`sources.M4SINGER_VALID`).
  Other singers' recordings of the same songs are left out of training
  (`split: excluded`), so validation never scores a melody the network trained on.
- **GTSinger** (`sources gtsinger`; Hugging Face `GTSinger/GTSinger`, CC BY-NC-SA 4.0):
  20 singers in 9 languages, 3 of them English, each song sung with a technique and
  plainly. Speech recordings are skipped. Its notes also start at the consonant.
  The half-way labels move a note start towards the first vowel after it, past
  consonant clusters. Two cases keep the consonant start:
  - a held, pitched consonant longer than 0.25 s, such as the sung W of "word";
  - words whose phones look evenly divided rather than aligned. In a sample,
    none of the English or Mandarin words looked like this, but up to a quarter
    of the Japanese, Korean and Spanish ones did.

  Validation holds out EN-Alto-2 and KO-Soprano-2 (`sources.GTSINGER_VALID`).
  To add next: Opencpop. transcPybe's synthetic corpus is
  also a source, but it is synthesised at 16 kHz, so it is weak material for the
  separator.
- **Accompaniment.** MUSDB18-HQ (`--accomp-depth 2`) or MoisesDB (`--accomp-depth 1`):
  every stem except vocals, summed. `--accomp synthetic` is for tests only.
- **Never trained on:** RWC-P, Molina, vocadito. They are the evaluation sets.

## Evaluation

During training, validation logs precision, recall, onsets per reference note
and onset timing (median and quartiles of estimate minus reference, and the
share more than 25 ms early or late). Timing against the training convention
only checks alignment: a network that marks notes at the consonant looks
perfect against consonant labels. Add the other convention with
`--extra-valid vowel=…/features.jsonl` to see a mismatch while training. The
dry pilot put note starts 40-70 ms before the annotators of Molina and vocadito.

The decision is made on the RWC-P development half, against configuration B
from `dev/frozen.json` (COnP@150 0.659 on the development half, 0.650 on the test half): the
same replay harness (`dev/replay.py`), with the network's onsets passed to
`decode_notes(onsets=...)` in place of transcPybe's own. The test half stays
untouched until a frozen network is checked once, under a protocol amendment
written before that check.

## CSD3

The bulk work runs on CSD3 (Cambridge HPC), so that musix stays free for other
users. The account is `HARRISON-SL3-GPU` for the A100s (Wilkes-3, partition
`ampere`), with `HARRISON-SL3-CPU` for CPU work.

### How we use it

The choices below follow CSD3's documentation; each point names the page.

**Service level: SL3.**
- SL3 runs as QOS2: lower priority than paid levels, 12 hours at most per job,
  and at most 32 GPUs or 448 CPU cores per user at once
  ([policies](https://docs.hpc.cam.ac.uk/hpc/user-guide/policies.html)).
- Queue waits on 21 October:
  - a GPU job waited 1 h 26 min;
  - a CPU job was still waiting after more than 3 hours, at priority 104 with
    4,474 of 4,537 waiting CPU jobs ranked above it.
- Priority rises slowly with age. Idle nodes are held for higher-priority work
  and lent out only to jobs that fit the gaps, so ask for no more time than a
  job needs.

**What runs where.**
- **Login nodes:** editing, submitting, monitoring, the tests, small downloads.
  They are "not intended for production workload", and "any large or
  long-lasting job is liable to be terminated by an automatic script called
  watchdog" ([connecting](https://docs.hpc.cam.ac.uk/hpc/user-guide/connecting.html)).
- **GPU jobs:** almost everything else. CPU-only jobs at SL3 queue too long to
  rely on, so `prepare` mixes inside the GPU job that computes the features.
- **Heavier interactive work** (the Ampere page's example is recompiling; ours
  is building a container image): an interactive node, `sintr`
  ([Ampere](https://docs.hpc.cam.ac.uk/hpc/user-guide/a100.html)).

**Packing GPUs.**
- Each Ampere GPU comes with 32 cores, and "1 credit = 1 GPU for 1 hour"
  ([Ampere](https://docs.hpc.cam.ac.uk/hpc/user-guide/a100.html)).
- Measured on 21 October, one A100 against musix's RTX 3080:
  - training ran at the same speed;
  - CREPE features ran a little slower per process.

  The network is small, so the CPU sets the pace.
- So every GPU job keeps its GPU busy:
  - `prepare` runs 8 feature processes on one GPU;
  - `train.sbatch` trains several seeds side by side (`SEEDS="0 1 2"`).

**Where files live**
([file and I/O management](https://docs.hpc.cam.ac.uk/hpc/user-guide/io_management.html)).
- **`$DATA`** (default `/rds/user/$USER/hpc-work/boundary`) is on RDS, the
  Lustre filesystem where "active data to be read or written by jobs should be
  placed". It is built for large files. Its file quota is about a million
  files, and it is **not backed up**: copy checkpoints, logs and manifests
  somewhere that is once a campaign ends.
- **`/local` and `/ramdisks`** are each node's own disk and memory, "private
  to each compute node", and their files are "destroyed after completion of
  the current job". `prepare` keeps every per-recording file there and copies
  only one shard per part to `$DATA`: about 115 KB of compressed features per
  clip, a few hundred files for the whole plan.
- **Home** holds the Python environment and the code: "your home directory
  should not be used for I/O from running jobs", and jobs only read it at
  start-up.
- **Raw datasets:** download straight into `$DATA` from their sources. Relaying
  through a laptop ran at 1-9 MB/s.
- The login node's `/local` exists on that node only. It suits build scratch,
  nothing to keep.

**Jobs: ask for what the job needs, specifically and realistically.**
- Every script names its time, cores and memory, and says what measurement the
  numbers come from.
- Shorter, smaller jobs fit the gaps the scheduler fills between
  higher-priority work, so they start sooner. Asking for the maximum only makes
  a job wait longer.
- On Icelake, asking for more memory than 3,370 MB per core makes Slurm
  allocate, and charge for, more cores
  ([icelake](https://docs.hpc.cam.ac.uk/hpc/user-guide/icelake.html)).
- On Ampere, each GPU comes with 32 cores and 250 GB, charged as one GPU-hour
  per hour.
- To trim a job that is still waiting:
  `scontrol update JobId=N TimeLimit=... NumCPUs=... CPUsPerTask=...`.
  Memory per core takes plain megabytes (`MinMemoryCPU=3370`). The job keeps
  its place in the queue.

| Script | Asks for | Basis (21 October) |
|---|---|---|
| `prepare.sbatch` | 1 GPU, 2 h | one feature process managed 3 clips/s, so 10,000 clips take under 1 h; override `--time=` for bigger parts |
| `train.sbatch` | 1 GPU, 3 h | 2,000 steps with one evaluation took 105 s; 30,000 steps take about 40 min per seed, and 3 seeds share the GPU |
| `features.sbatch` | 1 GPU, 1 h | validation-size sets only; 8 workers |
| `separate.sbatch` | 1 GPU, 2 h | validation-size sets only; musix separated at 2.86x real time |
| `mix.sbatch` | 8 cores, 30 min | validation-size sets only; musix mixed 17,000 clips in 11 min on 16 cores |

- Arrays rather than a loop of submissions: "Array jobs should be prefered to
  multiple `sbatch` calls in a loop"
  ([batch](https://docs.hpc.cam.ac.uk/hpc/user-guide/batch.html)).
- Size each array task well inside the 12-hour limit. Every stage resumes after
  a time-out, so a time-out only means resubmitting.
- Quick experiments do not suit SL3 queues.

**Python** follows CSD3's
[PyTorch guide](https://docs.hpc.cam.ac.uk/hpc/software-packages/pytorch.html):
- a venv with pip, not conda;
- `rhel8/default-amp` loaded;
- `OMP_NUM_THREADS` and `MKL_NUM_THREADS` set to the job's cores.

**Not done yet**
- A pinned lock file for the environment.
- A read-only GitHub deploy key, so CSD3 can `git pull` the private repository.
  Until then, update it from a git bundle made on a machine with access.
- Checking that Wilkes-3 nodes have a writable `/local`. `prepare.sbatch`
  falls back to `/ramdisks`, then to `$TMPDIR`.

### Setting up

The job scripts are in `slurm/`. Each one sources `slurm/env.sh`, which:
- loads `rhel8/default-amp`;
- activates `$VENV` (default `~/venvs/boundary`);
- puts this repository on `PYTHONPATH`;
- sets one thread per allocated core (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`),
  as CSD3's PyTorch guide does;
- writes a line to the ledger when the job starts and when it ends.

The environment is a plain venv, as CSD3's PyTorch guide uses (no conda), on
the system's Python 3.11. The versions match musix, except torch's CUDA build:
the A100 nodes run driver 595, which takes the default CUDA 13 wheels.

```bash
/usr/bin/python3.11 -m venv ~/venvs/boundary
~/venvs/boundary/bin/python -m pip install --no-cache-dir torch==2.14.1 torchcrepe==0.0.24 numpy==2.2.6 \
    scipy==1.15.3 librosa==0.11.0 soundfile==0.14.0 soxr==1.1.0 numba==0.67.0 mir_eval==0.8.2 \
    pretty_midi==0.2.11.post0 pytest
```

Installing torch writes about 5 GB of small files into home, which takes
20-30 minutes; it happens once. The tests take about 95 s on a login node:
`python -m pytest boundary/tests -q`.

**The separator** (BS-RoFormer) is no longer part of the bulk job. On
21 October, mixing the backing in quietly with no separator recovered 77-89%
of what separation gained. It is still needed for a validation-size set of
separated audio. Its Apptainer image is rebuilt on CSD3 from the musix image's
recipe:
- every Python package is pinned to that image's `pip freeze`;
- the ZFTurbo commit and the weights' checksums are checked.

Two snags:
- Launchpad's Ubuntu PPA served intermittent 503s, so the CSD3 build starts
  from the official `python:3.9.25` image.
- Apptainer hides the image's `/tmp` while the build runs, so build files go
  elsewhere.

### Submitting

Use `submit.sh`. It writes each job's log to `$DATA/logs/<job name>-<job id>.out` and prints the job id and log path. Every stage is resumable: after a time-out, resubmit the same command.

The bulk recipe has no separator. `prepare` does one part of a dataset per
array task:
1. it mixes with the accompaniment 20-35 dB under the voice;
2. it computes the features;
3. it writes one shard.

Training then reads the shards, with three seeds on one GPU:

```bash
export DATA=/rds/user/$USER/hpc-work/boundary
bash boundary/slurm/submit.sh prepare VOCALS=$DATA/m4singer.jsonl ACCOMP=$DATA/musdb18hq DEPTH=2 \
    OUT=$DATA/m4-musdb NPARTS=16 --array=0-15
bash boundary/slurm/submit.sh train FEATURES="$DATA/m4-musdb/shards/*.jsonl" RUN=$DATA/runs/r1 SEEDS="0 1 2"
```

Choose `NPARTS` so that one part fits well inside 12 hours. The rate with 8
feature processes is not measured yet. At the single-process rate (about
3 clips/s on an A100), 10,000 clips take under an hour. So a few large parts
are enough, and each waits in the queue only once. Rerunning the same array
skips finished parts.

The validation-size set of real separated audio uses the stage-by-stage route:
`mix`, then `separate` with the Apptainer image (`SIF=...`), then `features`.

Add `MAIL=you@cam.ac.uk` before `bash` to get an email from Slurm when a job ends or fails.

### Following progress

The status command shows everything at once:

```bash
bash ~/transcPybe/boundary/slurm/status.sh
```

It reports:
- your queued and running jobs;
- the latest progress line of each running job;
- each dataset's mixes, stems and features done, or its `prepare` parts finished;
- each training run's latest validation (score, onset F, onset timing, ETA);
- GPU hours used so far from the ledger, against the plan's 10-15;
- `mybalance`;
- your quota.

Every long stage logs a timestamped line at least once a minute with its count, rate and ETA. Training logs one line per evaluation and keeps the full record in `<run>/log.jsonl`.

`$DATA/ledger.tsv` has one line per job start and end: time, job id, name, node, GPUs, exit code and elapsed seconds. A job killed at its time limit still gets its end line.

After you log in once with `ssh hpc` (password and phone code), the connection is shared for 8 hours (`ControlPersist` in `~/.ssh/config`). Claude can then run the same status command, read the logs and submit jobs over that shared connection. The connection runs through the VPN. If the VPN drops for more than a few minutes, the shared connection dies and needs `ssh hpc` again. Jobs, and anything started on CSD3 detached from the session, carry on regardless.

### Dashboard

Training runs send each evaluation to an MLflow server on musix as they go. You can follow them in a browser without
logging in to CSD3, and Claude can read the same numbers from musix without polling CSD3.

- **How runs report.** `train.sbatch` sets `MLFLOW_TRACKING_URI=http://musix.mus.cam.ac.uk:5000`. The CMS guide
  records that musix's firewall opens port 5000 to CSD3. Reporting uses only the standard library (`boundary/track.py`),
  so the venv needs nothing new.
- **Turning it off.** Pass `MLFLOW_TRACKING_URI=` (empty) to `submit.sh`. `MLFLOW_EXPERIMENT_NAME=...` groups runs
  under another name than `boundary`.
- **If musix is down.** A run says so once in its log and carries on. `<run>/log.jsonl` stays the record; the
  dashboard is only a view of it.
- **What a run shows.** Each run is named after its directory, such as `r1/seed0`. Its settings are parameters, and
  its Slurm job, node and commit are tags. Every evaluation logs score, onset F, re-sung recall, precision, recall,
  onsets per note, timing, loss and ETA, and `extra.NAME.*` for each `--extra-valid` set. A run stopped at its time
  limit is marked KILLED.

To view it, open a tunnel from your own machine, then browse to <http://localhost:5001> (macOS keeps port 5000 for
AirPlay):

```bash
ssh -N -L 5001:localhost:5000 musix
```

To compare seeds, tick their runs and choose Compare, or chart `score` against step.

The server runs as your user on musix. It is MLflow 3.17.2 with its login extra, installed in `~/venvs/mlflow` with
`pip install "mlflow[auth]==3.17.2"`. `boundary/mlflow_server.py` starts it, with its data in `~/mlflow`, and its
docstring gives the details:
- every request needs a login;
- it takes no file uploads;
- it runs none of MLflow's background workers.

To start it (once, and again after musix restarts), copy the script to musix and run it there:

```bash
scp boundary/mlflow_server.py musix:mlflow/server.py
```

```bash
python3 ~/mlflow/server.py start
```

The first start makes the passwords at random and keeps them in `~/mlflow`, which only you can read; nothing prints
them. It also makes the account the jobs use, `boundary`, and writes it to `~/mlflow/credentials`. To copy that to
CSD3 without it showing on screen, run this from your Mac:

```bash
ssh musix cat mlflow/credentials | ssh hpc 'umask 077; mkdir -p ~/.mlflow; cat > ~/.mlflow/credentials'
```

The browser asks for a login: use `boundary` and the password in that file. `python3 ~/mlflow/server.py status` and
`stop` do what they say.

Plain HTTP carries the jobs' login across the University network between CSD3 and musix. Viewing goes through the
SSH tunnel, which is encrypted.

## Tests

```bash
.venv/bin/python -m pytest boundary/tests -q
```

The last test runs the whole chain on three synthetic recordings, with a
synthetic accompaniment and no separator, in about 30 s.
