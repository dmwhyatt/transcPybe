# transcPybe

Transcribes monophonic singing — typically a source-separated lead vocal stem —
into note events with **integer MIDI pitches (0–127)** and **floating-point
onset times in seconds**.

No GPU required and no checkpoint downloads: both pitch models ship their
weights inside their PyPI wheels.

```
audio ──▶ f0 + voicing ──▶ note HMM ──▶ glide handling ──▶ onset refinement ──▶ notes ──▶ .mid
          CREPE / pYIN      Viterbi      portamento         spectral flux        pitch     .json
                                                                                 measure   .csv
```

## How good is it?

On [vocadito](https://zenodo.org/records/5578807) — 40 excerpts of solo singing
in seven languages, by singers of varying training, recorded on whatever was to
hand — scored with `mir_eval` at the usual 50 ms / 50 cent tolerances:

| | COn | COnP | COnPOff |
|---|---|---|---|
| transcPybe | **0.707** | **0.589** | **0.407** |
| ROSVOT, convention-corrected | 0.472 | 0.355 | 0.261 |
| two human annotators | 0.817 | 0.740 | 0.642 |

Read those against the last row rather than against 1.0: two people annotating
the same recordings agree at 0.740 on onset and pitch, so most of the remaining
gap is a disagreement about what counts as a note. The comparison and how it
was made are in [docs/benchmarking-rosvot.md](docs/benchmarking-rosvot.md).

## Install

Not published. Install it from a checkout:

```shell
pip install -e .              # pYIN backend, no torch
pip install -e ".[crepe]"     # adds the CREPE backend, which is better
```

The `pip install transcpybe` form in this section is what it will look like if
this is ever released; `pyproject.toml` carries a classifier that makes PyPI
reject an upload until someone removes it on purpose.

CREPE is the default and the more accurate of the two, but it wants torch. The
plain install works without it; pass `--f0-backend pyin`.

## Use

```shell
# Transcribe a stem; writes lead-vocal.mid, .notes.json and .notes.csv
transcpybe lead-vocal.wav -o out/

# Print the whole transcription as a table
transcpybe lead-vocal.wav --print-notes

# Transcribe a whole folder, logging to a file you can tail
transcpybe stems/ -o midi/ --log-file runs/transcribe.log

# Fast pass (~14x quicker, less accurate)
transcpybe lead-vocal.wav --crepe-model tiny

# No torch installed
transcpybe lead-vocal.wav --f0-backend pyin

# Check the install against the built-in synthetic benchmark
transcpybe --self-test
```

## Monitoring a run

A four-minute vocal takes minutes; a few hundred songs takes hours. Both report
what they are doing as they go.

```
[2026-02-11 09:14:02] [7/143] Northern_Sky.wav (3:47)
[2026-02-11 09:14:02]   loading Northern_Sky.wav: done in 0:01
[2026-02-11 09:14:02]   device: cuda:0 (NVIDIA RTX A5000, 24.0 GiB) [auto-selected]
[2026-02-11 09:14:02]   transcribing 227.4s of audio, crepe backend
[2026-02-11 09:14:02]   CREPE full on cuda:0: started
[2026-02-11 09:14:09]   CREPE full on cuda:0   38.2%  ETA 0:11  87s / 227s audio
[2026-02-11 09:14:20]   CREPE full on cuda:0: done in 0:18 (0.08x realtime)
[2026-02-11 09:14:20]   22740 frames, 61.3% voiced
[2026-02-11 09:14:21]   224 notes after post-processing
[2026-02-11 09:14:21]   transcribed in 0:19 (0.09x realtime)
[2026-02-11 09:14:21]   224 notes in 0:19
[2026-02-11 09:14:21]   overall 7/143 files (5%), elapsed 2:41, ETA 51:22
```

Output adapts to where it goes. On a terminal, progress rewrites a single line
in place. Redirected to a file -- which is how anything long actually gets run,
under tmux or nohup -- it becomes a periodic timestamped line instead, because
carriage returns would make the log unreadable. Nothing needs configuring for
that; `--progress-interval` controls how often those lines appear.

| Flag | Purpose |
|---|---|
| `--log-file PATH` | Append timestamped logs to a file as well as the console. Survives a lost tmux session. |
| `--progress-interval N` | Seconds between progress lines when redirected (default 5). |
| `--no-progress` | Silence progress entirely. |
| `--gpu-memory-every N` | Log GPU memory every N CREPE batches, so you see headroom shrinking *before* an out-of-memory failure. |
| `-v` | Include debug detail. |

## Running on a GPU

`--device` defaults to `auto`: CUDA if available, then Apple MPS, then CPU. The
device that was chosen is logged at the start of every file, and recorded in the
output JSON, so a run can never quietly have used the CPU without saying so. An
explicit `--device cuda` is honoured as given rather than silently downgraded --
if CUDA is missing, torch raises, which is more useful than discovering days
later that a batch ran 30x slower than intended.

`--crepe-batch-size` (default 512 frames, about 5 seconds of audio) is the knob
for GPU memory against speed: raise it on a large card, lower it if memory is
tight. It also sets how finely progress is reported, since progress advances one
batch at a time.

## Batch mode

Point the CLI at a directory and it transcribes every audio file underneath.
Three properties matter for a run measured in hours, and all three hold:

- **Resumable.** Files whose outputs already exist are skipped, so an
  interrupted batch does not start over. `--overwrite` forces a redo.
- **Robust.** One unreadable file is logged and the batch carries on;
  `--stop-on-error` reverses that. Failures are listed again in the closing
  summary so they are not lost in the scrollback.
- **Honest ETA.** Remaining time is weighted by audio duration rather than file
  count, read cheaply from file headers, because songs are not all the same
  length.

```shell
tmux new -s transcribe
transcpybe /data/stems -o /data/midi --log-file /data/transcribe.log
# detach with Ctrl+B then D; later:
tail -f /data/transcribe.log
```

## Reproducibility

torchcrepe deliberately adds random triangular dither of +/-20 cents to every f0
estimate, to trade bin-quantization error for noise. It is drawn from numpy's
global RNG, so upstream the same audio gives a slightly different answer every
run. This module seeds it (`--seed`, default 0) and restores the previous RNG
state afterwards, so transcription is repeatable and importing it never disturbs
anyone else's random stream.

Measured on the synthetic suite, what the dither actually costs when unseeded:

| | Unseeded | Seeded |
|---|---|---|
| Note pitches | stable in testing | identical |
| Onset times | shift by a frame or so | identical |
| `pitch_midi_float` | varies by up to ~6 cents | identical |
| Benchmark note F | **0.9792, unchanged** | 0.9792 |

So seeding is about reproducibility, not accuracy: the differences sit well
inside the 50 ms and quarter-tone tolerances, and three repeats of the whole
16-case suite scored identically with and without it. It matters because these
transcriptions feed a feature dataset, where a file that changes between runs is
a problem regardless of whether the F-score moves. `--no-seed` restores
torchcrepe's stochastic behaviour.

From Python:

```python
from transcpybe import TranscriptionConfig, transcribe_file, write_midi

result = transcribe_file("lead-vocal.wav", TranscriptionConfig())
print(result.summary())

for note in result.notes:
    print(note.onset, note.offset, note.pitch)   # float, float, int

write_midi(result.notes, "lead-vocal.mid", result.config)
```

`result` also carries the intermediate `f0_track` and `onsets`, which is the
quickest way to see *why* a transcription came out the way it did.

## Output

`.mid` — one instrument track. Note times are **absolute seconds**; the tempo in
the header only sets the bar grid a DAW draws, and never quantises or moves a
note.

`.notes.csv` — `onset, offset, duration, pitch, velocity, confidence, pitch_hz,
pitch_midi_float`.

`.notes.json` — the same notes plus the full configuration that produced them,
so any run can be reproduced exactly.

`pitch_midi_float` is the unrounded measured pitch. The difference between it
and `pitch` is how far the singer was from equal temperament, which is useful
for intonation analysis and for spotting notes that only just rounded one way.

---

## How it works

**1. f0 and voicing.** CREPE estimates a fundamental and a periodicity
confidence every 10 ms. Confidence alone is not enough on separated stems, so it
is multiplied by an adaptive loudness gate set relative to the track's own 95th
percentile level. Without that gate, the residue separation leaves in silent
passages is transcribed as notes — it genuinely is periodic, so the pitch model
reports it with high confidence.

**2. Note decoding.** A Viterbi decode over one state per semitone plus a single
unvoiced state. Emissions score each semitone by how close it is to the measured
pitch, weighted by voicing; transitions make staying in a note cheap, changing
note expensive, and large leaps slightly more expensive than small ones. This
follows the note-tracking design of the Tony transcription system, simplified to
one state per semitone: vibrato is absorbed by the emission width and the
self-transition bias rather than by Tony's sub-semitone grid.

**3. Edge fragments.** Wide vibrato sheds a few frames a semitone below at the
end of a decaying note, which the decoder emits as a separate note. Fragments
that are very short, immediately against a neighbour many times their length,
and within a semitone or two of it, are folded back in.

**4. Glides.** Singers slide between pitches. That puts note starts late by the
length of the slide, and leaves brief "passing tone" notes on each semitone a
wide slide crosses. Both are corrected by recognising the transition region and
attributing it to the note being approached. The backtrack that does this
requires the frames it crosses to be audible, periodic, at the level of the note
they lead into, and moving steadily — because pitch trackers keep emitting a
smoothly drifting pitch through unvoiced noise while reporting near-zero
confidence, and following that drift puts onsets far too early.

**5. Onsets.** A SuperFlux onset envelope splits repeated notes at one pitch —
which have an attack but no pitch change, so the pitch HMM cannot see them — and
pulls note starts onto the audible attack. SuperFlux's max filter across
frequency is what stops vibrato registering as a stream of onsets.

**6. Pitch and cleanup.** Each note's pitch is the confidence-weighted median of
its f0, skipping the head of the note so scoops and consonants do not drag the
estimate, then rounded to the nearest integer. Finally, fragments of one note
are rejoined, notes below the duration floor are dropped, and notes far quieter
than the rest of the performance are discarded.

---

## Accuracy

Measured on the built-in synthetic benchmark (`--self-test`): 16 cases, 127
notes, scored with `mir_eval` at the MIREX 50 ms onset tolerance and quarter-tone
pitch tolerance.

| Backend | Cases perfect | Onset F | Note F (onset+pitch) | Note F (+offset) | Speed (4 CPU cores) |
|---|---|---|---|---|---|
| `crepe` (`full`, default) | **15 / 16** | 1.000 | **0.979** | 0.979 | ~1.6x realtime |
| `pyin` | 13 / 16 | 0.914 | 0.923 | 0.923 | ~0.34x realtime |
| `crepe` (`tiny`) | — | — | — | — | ~0.11x realtime |

The backends fail differently, which is why both are kept:

- **pYIN** finds *nothing* in the `breathy` case (broadband noise only 12 dB
  below the voice) and loses 3 of 8 notes in `low_register`. It is markedly less
  noise-robust, and struggles where harmonics are closely spaced.
- **CREPE** handles both, but confidently transcribes loud separation residue as
  notes, where pYIN correctly reports it unvoiced.

So: use CREPE on clean stems and anything breathy or low; if a stem is badly
separated and full of bleed, pYIN is worth a try as a second opinion.

### The one failing case, and why it is a dial rather than a bug

`loud_residue` puts separation residue only 18 dB below the vocal.
`dynamics` contains real singing 25 dB below the loudest notes in the same
phrase. These two cases are deliberately contradictory: **no level threshold
satisfies both.**

Sweeping `--note-level-floor`:

| `note_level_floor_db` | `dynamics` note F | `loud_residue` note F | |
|---|---|---|---|
| 20 | 0.667 | 1.000 | quiet singing lost |
| 24 | 0.857 | 0.778 | both degraded |
| **30 (default)** | **1.000** | 0.667 | residue kept as notes |

The default favours keeping real quiet singing, on the view that a spurious note
you can see and delete is less costly than a sung note that was never reported.
Lower it when working with a poorly separated stem.

### What these numbers do and do not mean

The benchmark is **synthesised**, not real singing. The synthesiser models
vibrato, portamento, scoops, jitter, consonant transients, breath noise and
separation residue, and each case was written to break a specific stage — but
real voices are harder than any of it. Read these scores as evidence that the
implementation does what it claims on the phenomena it models, not as an
estimate of accuracy on a real recording. For that, transcribe a stem you know
and read `--print-notes` against it.

---

## Benchmarking against real annotations

Everything under [Accuracy](#accuracy) is measured on signals this repository
generates, scored against ground truth this repository also generates. That
verifies the implementation does what it claims on the phenomena it models. It
is **not** a measure of how well it transcribes real singing, and no number from
it should be quoted as one.

`transcpybe/benchmark/` scores the transcriber against human
annotations and reports a figure that can be defended.

Check the layout first — it takes seconds and reads no audio:

```shell
transcpybe-benchmark path/to/dataset --dry-run
```

That reports which annotation file was paired with which audio, how each was
interpreted, and the note, time and pitch ranges it read.

Datasets often ship several annotations per recording — a note-level one, an f0
contour, lyrics, sometimes one per annotator. Audio is matched to any annotation
whose filename begins with its own, and where several qualify the note-level one
is chosen and the choice reported as `prefix (1 of 4)`. The dry run names the
file actually used, and `--annotation-contains notesB` pins a different one. The two ways a
benchmark silently reports nonsense are a mis-paired file and a misread
annotation, and both are visible here rather than after an hour of
transcription. If audio and annotations live in separate directories, pass
`--annotations`; the runner searches recursively, so pointing at the dataset
root usually works.

Then run it:

```shell
transcpybe-benchmark vocadito/audio \
    --annotations vocadito/annotations \
    --cache-dir .bench-cache \
    -o results/vocadito.json
```

`--limit N` scores only the first N items, which is worth doing once before
committing to the whole set. `--cache-dir` makes re-scoring instant, and is
keyed on the configuration so changing a parameter re-transcribes rather than
scoring the previous settings.

```text
40 recordings, 3211 reference notes, 3106 transcribed

metric                            P      R      F   95% CI on F
--------------------------------------------------------------------
COn      (onset)              0.871  0.842  0.856   [0.821-0.889]
COnP     (onset+pitch)        0.804  0.777  0.790   [0.749-0.828]
COnPOff  (+offset)            0.612  0.591  0.601   [0.558-0.643]

Macro average over recordings; 95% CI from 2000 bootstrap resamples.
Micro (pooled over all notes): COnP F = 0.798

Octave errors: 1.4% of notes whose onset was found
Median best time offset: +0.004s
```

*(illustrative shape, not a result — run it to get real numbers)*

### The metrics

**COn**, **COnP** and **COnPOff** are the note-transcription measures the
singing-transcription literature reports, so the numbers can sit beside
published ones. An onset matches within 50 ms, a pitch within 50 cents, an
offset within 50 ms or 20% of the reference note's duration. They nest: a note
counted in COnPOff is counted in COnP and in COn. **COnP is the headline.**

### Why there are error bars

A single number is not a measurement. The interval comes from a bootstrap
resampled over **recordings**, not notes, because notes within a recording are
not independent — a singer whose vibrato defeats the decoder defeats it for the
whole track. Resampling notes would report an interval several times too narrow
and imply a precision the benchmark does not have.

A benchmark of forty recordings supports a claim like "COnP 0.79, 95% CI
0.75–0.83". It does not support "0.79 versus 0.78 is an improvement".

### Two diagnostics worth reading first

**Median best time offset.** Reference annotations are often offset from the
audio by a constant. Scored against one, a perfect transcriber looks broken, and
no amount of tuning fixes it. The suite searches for the global shift that
maximises COnP and warns about any item that scores materially better under one.
**If that warning appears, fix the alignment before believing any score.**

**Octave errors** — notes whose onset was found but which landed a whole number
of octaves out. A high rate points at the pitch tracker rather than the note
decoder, and at `--fmin`/`--fmax` in particular.

### Annotation formats

Readers for CSV, TSV, TXT, MIDI, JSON and JAMS. Datasets disagree about whether
the second column is an offset or a duration and whether pitch is MIDI or Hz, so
both are inferred and **the inference is reported**. That matters: a duration
read as an offset gives notes that start correctly and end immediately, which
scores badly in a way that looks like a poor transcriber rather than a misread
file. Inference uses the proportion of rows supporting each reading, so a
handful of corrupt rows cannot reinterpret a whole file.

Audio is paired with annotations by stem, then by stem with a dataset tag such
as `_notes` stripped, then by substring. How each pair was matched is reported,
and unpaired audio is listed rather than quietly skipped.

### Reproducibility

Every result records the transcriber configuration, the seed, the versions of
torch, torchcrepe, librosa and mir_eval, the repository commit, and a SHA-256
digest of the dataset files. Change the data or a dependency and the digest or
the version string changes with it.

```shell
transcpybe-benchmark --synthetic     # check the harness itself
```

### One discipline worth keeping

If parameters are tuned until the benchmark score improves, that score is no
longer an estimate of performance on unseen material — it is an estimate of how
well the parameters fit that benchmark. Tune on one dataset and report on
another, or split a single one and keep the held-out half genuinely unseen.

## Tuning

| Symptom | Knob | Direction |
|---|---|---|
| Too many short spurious notes | `--min-note` | raise (e.g. 0.08) |
| Sustained notes split into several | `--p-self` | raise (e.g. 0.995) |
| Heavy vibrato still fragmenting | `--pitch-sigma` | raise (e.g. 0.7) |
| Genuine fast runs merged together | `--p-self` | lower (e.g. 0.97) |
| Notes appearing in silent passages | `--note-level-floor` | lower (e.g. 22) |
| Quiet real notes being dropped | `--note-level-floor` | raise, or `reject_quiet_notes=False` |
| Repeated same-pitch notes merged | keep onsets on; **lower** `onset_split_strength_ratio` |  |
| Onsets landing early on breathy audio | `glide_min_periodicity` | raise |
| Voice outside the tracked range | `--fmin` / `--fmax` | widen |

Every parameter in `config.py` is documented in place; the CLI exposes the ones
worth reaching for most often, and the rest are available through
`TranscriptionConfig`.

## Tests

```shell
pytest tests/                 # the full suite
pytest tests/ -m slow         # just the ones that exercise CREPE
transcpybe --self-test   # full synthetic benchmark
```

The CREPE tests include the one that matters most for the progress machinery:
transcription drives torchcrepe's `preprocess`/`infer`/`postprocess` directly
rather than calling `predict`, purely so progress can be reported between
batches, and `test_chunked_crepe_matches_torchcrepe_predict` asserts the two
remain bit-identical.

---

## Limitations

- **Monophonic only.** One pitch at a time. Harmonies or overlapping vocal
  layers in the same stem will be transcribed as one wandering line.
- **Offsets are softer than onsets.** Where a sung note ends is genuinely
  ambiguous, and the release is often breath rather than pitch.
- **Loud separation residue** is transcribed as notes at the default level
  floor — see the dial above.
- **Validated on synthetic audio.** See the caveat above.
- **Unquantised.** Output is in seconds, with no beat tracking or rhythmic
  quantisation. That is deliberate: quantising is lossy and belongs downstream.

---

## Credit

This module is an assembly of existing open-source research and software. The
substantive ideas and all pretrained weights come from the work below.

**Pitch estimation**

- **CREPE** — Jong Wook Kim, Justin Salamon, Peter Li, Juan Pablo Bello,
  "CREPE: A Convolutional Representation for Pitch Estimation", ICASSP 2018.
  <https://arxiv.org/abs/1802.06182>
- **torchcrepe** — Max Morrison's PyTorch port of CREPE, which supplies the
  pretrained weights used here. MIT License.
  <https://github.com/maxrmorrison/torchcrepe>
- **pYIN** — Matthias Mauch and Simon Dixon, "pYIN: A fundamental frequency
  estimator using probabilistic threshold distributions", ICASSP 2014. Used via
  librosa's implementation.

**Note segmentation**

- **Tony** — Matthias Mauch, Chris Cannam, Rachel Bittner, George Fazekas,
  Justin Salamon, Jiajie Dai, Juan Bello, Simon Dixon, "Computer-aided melody
  note transcription using the Tony software: Accuracy and efficiency",
  Proceedings of the First International Conference on Technologies for Music
  Notation and Representation, 2015. The pitch-state HMM here is a simplified
  version of Tony's note-tracking model.
  <https://code.soundsoftware.ac.uk/projects/tony>

**Onset detection**

- **SuperFlux** — Sebastian Böck and Gerhard Widmer, "Maximum filter vibrato
  suppression for onset detection", DAFx 2013. Used via librosa's
  `onset_strength(max_size=...)`.

**Libraries**

- **librosa** — Brian McFee et al., "librosa: Audio and music signal analysis in
  Python", SciPy 2015. ISC License. <https://librosa.org>
- **pretty_midi** — Colin Raffel and Daniel P. W. Ellis, "Intuitive analysis,
  creation and manipulation of MIDI data with pretty_midi", ISMIR 2014 Late
  Breaking. MIT License.
- **mir_eval** — Colin Raffel, Brian McFee, Eric J. Humphrey, Justin Salamon,
  Oriol Nieto, Dawen Liang, Daniel P. W. Ellis, "mir_eval: A transparent
  implementation of common MIR metrics", ISMIR 2014. MIT License.
- **NumPy** (BSD-3-Clause), **SciPy** (BSD-3-Clause), **soundfile**
  (BSD-3-Clause), **PyTorch** (BSD-3-Clause / Apache-2.0).

Licences were read from the installed distributions at the versions pinned in
`requirements.txt`.
