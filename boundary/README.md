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

Jobs are in `slurm/`. Each sources `slurm/env.sh`, which loads
`rhel8/default-amp`, activates `$VENV` (default `~/venvs/boundary`) and puts
this repository on `PYTHONPATH`. Data goes under
`/rds/user/$USER/hpc-work/boundary`. One-off setup on CSD3:

```bash
python -m venv ~/venvs/boundary && source ~/venvs/boundary/bin/activate
pip install torch torchcrepe librosa soundfile pretty_midi mir_eval scipy
```

The separator image goes over from musix:

```bash
docker save sep-bsroformer53:b24ab72 -o sep.tar
```

Then, on CSD3:

```bash
apptainer build sep-bsroformer53.sif docker-archive://sep.tar
```

Typical sequence. Every stage is resumable: resubmit the same command after a
time-out and it carries on. First set `DATA` in the login shell. `env.sh` sets
it only inside jobs, and the lines below expand it before they are submitted.

```bash
export DATA=/rds/user/$USER/hpc-work/boundary
```

```bash
sbatch --export=ALL,VOCALS=$DATA/m4singer.jsonl,ACCOMP=$DATA/musdb18hq,DEPTH=2,OUT=$DATA/m4-musdb boundary/slurm/mix.sbatch
sbatch --export=ALL,OUT=$DATA/m4-musdb,SIF=$DATA/sep-bsroformer53.sif boundary/slurm/separate.sbatch
sbatch --export=ALL,OUT=$DATA/m4-musdb boundary/slurm/features.sbatch
sbatch --export=ALL,FEATURES=$DATA/m4-musdb/features/features.jsonl,RUN=$DATA/runs/r1 --array=0-2 boundary/slurm/train.sbatch
```

## Tests

```bash
.venv/bin/python -m pytest boundary/tests -q
```

The last test runs the whole chain on three synthetic recordings, with a
synthetic accompaniment and no separator, in about 30 s.
