# boundary: a learned note-boundary detector for transcPybe

transcPybe tracks sung pitch well. Where it loses to GAME and MIR-ST500 on
separated pop is in placing note boundaries, especially telling a note re-sung on the same pitch
from a held note, and keeping short real notes. That was measured on RWC-P
(`bench-work-rwc-20261019` on musix, `protocol-amendment-1.md`). This folder
trains a small network that marks where notes start and end, frame by frame on
transcPybe's own 10 ms f0 grid, so its output can be fed to the existing
decoder as an `OnsetTrack`. Pitch tracking is left as it is.

Not part of the installed package, and nothing in `src/` depends on it.

## Stages

Each is `python -m boundary.<stage> --help`. Every stage reads the previous
stage's JSONL manifest and writes its own, adding fields rather than replacing them.

| stage | in | out | runs on |
|---|---|---|---|
| `sources` | labelled singing | `vocals.jsonl`: audio + notes + split | CPU |
| `mix` | vocals, multitrack accompaniment | `mixes/*.flac` 44.1 kHz stereo + `mixes.jsonl` | CPU |
| `separate` | mixes | `stems/*.wav` (BS-RoFormer lead vocal) + `stems.jsonl` | GPU |
| `features` | stems | `features/*.npz`: log-mel, transcPybe f0 features, targets | GPU (CREPE) |
| `train` | feature manifests | `best.pt`, `last.pt`, `log.jsonl` | GPU |
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
  To add next: GTSinger and Opencpop. transcPybe's synthetic corpus is
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
`ampere`), with `HARRISON-SL3-CPU` for CPU work. Both are SL3, the lowest
priority, so jobs queue: the first timing job waited 1 h 26 min.

### How we use it

Measured on 21 October with one A100 against musix's RTX 3080:
- training ran at the same speed;
- CREPE features ran a little slower per process.

The network is small, so the CPU side sets the pace, not the GPU. Each GPU
comes with 32 cores and 250 GB of memory. Billing is per GPU-hour.

**What runs where**
- **Login nodes:** editing, submitting, monitoring, the tests, small downloads.
  Large or long processes are killed by CSD3's watchdog.
- **CPU partitions:** mixing, data preparation, packing, scoring.
- **GPU partition:** CREPE features and training only. Pack each GPU:
  - several feature workers at once;
  - several seeds in one job.
- **Building a container image:** an interactive session (`sintr`), with the
  node's `/local` as scratch.

**Where files live**
- **`$DATA`** (default `/rds/user/$USER/hpc-work/boundary`): the shared
  filesystem (Lustre). It is fast for large files and slow for many small ones,
  and it is **not backed up**: copy checkpoints, logs and manifests somewhere
  that is once a campaign ends.
- **Raw datasets:** download straight into `$DATA` from their sources. Relaying
  through a laptop ran at 1-9 MB/s.
- **`/local` and `/ramdisks`:** every node has its own. On a compute node they
  belong to the job and are deleted when it ends. Per-clip working files belong
  there, with only packed results copied to `$DATA`. The login node's `/local`
  exists on that node only, and suits scratch for builds, nothing to keep.
- **Home:** the Python environment and the code. The CSD3 docs say home should
  not take job I/O; jobs only read the environment.

**How to submit**
- Few long jobs (SL3 allows up to 12 h) and arrays, submitted early or
  overnight: every queue wait is long, however short the job.
- Every stage resumes after a time-out, so a time-out only means resubmitting.
- Quick experiments do not suit SL3 queues; use an interactive session or a
  small CPU job.

**Not done yet**
- Mixing, features and packing in one job on a node's local disk, writing
  shards.
- Several training seeds per GPU.
- A pinned lock file for the environment.
- A read-only GitHub deploy key, so CSD3 can `git pull` the private repository.
  Until then, update it from a git bundle made on a machine with access.

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

```bash
export DATA=/rds/user/$USER/hpc-work/boundary
bash boundary/slurm/submit.sh mix VOCALS=$DATA/m4singer.jsonl ACCOMP=$DATA/musdb18hq DEPTH=2 OUT=$DATA/m4-musdb
bash boundary/slurm/submit.sh separate OUT=$DATA/m4-musdb SIF=$DATA/sep-bsroformer53.sif
bash boundary/slurm/submit.sh features OUT=$DATA/m4-musdb
bash boundary/slurm/submit.sh train FEATURES=$DATA/m4-musdb/features/features.jsonl RUN=$DATA/runs/r1 --array=0-2
```

Add `MAIL=you@cam.ac.uk` before `bash` to get an email from Slurm when a job ends or fails.

The example still includes separation. The bulk recipe now skips it: mix with
`--var-db 20 35`, then `separate --backend mix`, which needs no separator.
`mix.sbatch` and `separate.sbatch` do not pass those options yet.

### Following progress

The status command shows everything at once:

```bash
bash ~/transcPybe/boundary/slurm/status.sh
```

It reports:
- your queued and running jobs;
- the latest progress line of each running job;
- each dataset's mixes, stems and features done;
- each training run's latest validation (score, onset F, onset timing, ETA);
- GPU hours used so far from the ledger, against the plan's 10-15;
- `mybalance`;
- your quota.

Every long stage logs a timestamped line at least once a minute with its count, rate and ETA. Training logs one line per evaluation and keeps the full record in `<run>/log.jsonl`.

`$DATA/ledger.tsv` has one line per job start and end: time, job id, name, node, GPUs, exit code and elapsed seconds. A job killed at its time limit still gets its end line.

After you log in once with `ssh hpc` (password and phone code), the connection is shared for 8 hours (`ControlPersist` in `~/.ssh/config`). Claude can then run the same status command, read the logs and submit jobs over that shared connection. The connection runs through the VPN. If the VPN drops for more than a few minutes, the shared connection dies and needs `ssh hpc` again. Jobs, and anything started on CSD3 detached from the session, carry on regardless.

## Tests

```bash
.venv/bin/python -m pytest boundary/tests -q
```

The last test runs the whole chain on three synthetic recordings, with a
synthetic accompaniment and no separator, in about 30 s.
