# transcPybe

Transcribes monophonic singing — typically a source-separated lead vocal stem —
into note events with **integer MIDI pitches (0–127)** and **floating-point
onset times in seconds**.

No GPU required and no checkpoint downloads: both pitch models ship their
weights inside their PyPI wheels.

```
audio ──▶ f0 + voicing ──▶ note HMM ──▶ glide handling ──▶ onset refinement ──▶ notes ──▶ .mid
          CREPE / pYIN      Viterbi      portamento         spectral flux        pitch     .json
                                                            (note starts)        measure   .csv
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
gap is a disagreement about what counts as a note.

Both systems were run on the same audio and scored against the same references
with the same tolerances, which is the only way that comparison means anything.
ROSVOT's row is corrected for a difference in what its references call the
start of a note; uncorrected it scores 0.359 / 0.262 / 0.184, and the
correction is stated rather than folded in. The procedure, and how to reproduce
either row, is in
[the benchmark repository](https://github.com/dmwhyatt/transcription-benchmarking/blob/main/docs/benchmarking-rosvot.md).

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

Measured on the built-in synthetic benchmark (`--self-test`): 22 cases, 174
notes, scored with `mir_eval` at the MIREX 50 ms onset tolerance and quarter-tone
pitch tolerance.

| Backend | Cases perfect | Onset F | Note F (onset+pitch) | Note F (+offset) | Speed (4 CPU cores) |
|---|---|---|---|---|---|
| `crepe` (`full`, default) | **17 / 22** | 1.000 | **0.968** | 0.862 | ~1.6x realtime |
| `pyin` | 16 / 22 | 0.938 | 0.944 | 0.853 | ~0.34x realtime |
| `crepe` (`tiny`) | — | — | — | — | ~0.11x realtime |

Four of those cases exist to keep the suite honest rather than to be passed,
and they were added because the suite could not see two defects the vocadito
run had already found. `release_tails` and `breath_ends` let a note keep
sounding after the annotation ends it, which is the only way the offset column
carries information at all -- before them it was identical to the note column
on every case, and 145 of 145 matched notes passed the offset tolerance.
`soft_rearticulation` and `soft_legato_repeats` re-articulate a pitch on a
vowel rather than behind a consonant, which is what singing mostly does and
what a spectral-flux detector finds hardest.

Read the first row against the suite it is measured on. Over these 22 cases
CREPE scored 0.943 / 0.936 / 0.814 before the onset envelope was fixed and
1.000 / 0.968 / 0.862 after; over the 18 cases that existed before, it scored
1.000 / 0.978 / 0.978 either way. That is the reason the four were added rather
than a point against them: a suite at its ceiling cannot rank two
configurations, and this one was.

**And read the whole table against vocadito, because it did not predict it.**
The onset work above is worth 0.057 onset F here and, on forty real
recordings, +0.005 with a 95% interval of [-0.006, +0.018] -- better on 19,
worse on 19, which is the shape of a change that rearranges rather than
improves. `release_tails` and `breath_ends` still score 0.000, because the
trim that fixed them is now off by default: it scored 1.000 on both and cost
0.161 COnPOff on every one of those forty recordings. Synthetic signals with
exact ground truth are the right place to find a defect and a poor place to
confirm a cure.

The backends fail differently, which is why both are kept:

- **pYIN** finds *nothing* in the `breathy` case (broadband noise only 12 dB
  below the voice) and loses 3 of 8 notes in `low_register`. It is markedly less
  noise-robust, and struggles where harmonics are closely spaced.
- **CREPE** tracks both, but confidently transcribes loud separation residue as
  notes, where pYIN correctly reports it unvoiced. It also over-segments
  `breathy` -- 11 notes where there are 7 -- because at that noise level the
  onset envelope is nearly flat, and the peaks a detector sensitive enough to
  find vowel onsets picks out of it are partly noise. `--onset-delta` raises
  the bar a peak must clear relative to that recording's own floor, and is the
  dial for it.

So: use CREPE on clean stems and anything low; if a stem is badly separated and
full of bleed, pYIN is worth a try as a second opinion, and on a very noisy one
raise `--onset-delta` before reaching for the backend switch.

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

## Measuring it against real annotations

Everything under [Accuracy](#accuracy) is measured on signals this repository
generates, scored against ground truth it also generates. That verifies the
implementation does what it claims on the phenomena it models. It is **not** a
measure of how well it transcribes real singing, and no number from it should
be quoted as one.

Scoring against human annotations lives in a separate package,
**[transcription-benchmarking](https://github.com/dmwhyatt/transcription-benchmarking)**,
and deliberately not here. A benchmark that ships with the system it measures
can only ever measure that one; kept apart, the same corpora, tolerances and
confidence intervals apply to this transcriber and to anything it is compared
against. Every figure in [How good is it?](#how-good-is-it) came from it, and
so did ROSVOT's.

```shell
pip install git+https://github.com/dmwhyatt/transcription-benchmarking.git
# (both repositories are private; from local checkouts,
#  pip install -e ../transcription-benchmarking does the same)

# Score a directory of transcriptions this package already wrote
transcpybe vocadito/audio --out-dir transcriptions/
transcription-benchmark vocadito/audio \
    --annotations vocadito/annotations \
    --estimates transcriptions/ \
    -o results/transcpybe.json
```

## Tuning

### By hand

| Symptom | Knob | Direction |
|---|---|---|
| Too many short spurious notes | `--min-note` | raise (e.g. 0.08) |
| Sustained notes split into several | `--p-self` | raise (e.g. 0.995) |
| Heavy vibrato still fragmenting | `--pitch-sigma` | raise (e.g. 0.7) |
| Genuine fast runs merged together | `--p-self` | lower (e.g. 0.97) |
| Notes appearing in silent passages | `--note-level-floor` | lower (e.g. 22) |
| Quiet real notes being dropped | `--note-level-floor` | raise, or `reject_quiet_notes=False` |
| Repeated same-pitch notes merged | keep onsets on; **lower** `onset_split_strength_ratio` |  |
| Sustained notes shredded into many | `--onset-max-size`, then `onset_split_strength_ratio` | raise |
| Noisy input over-segmented | `--onset-delta` | raise |
| Note ends late, running into the rest | `--release-drop` | lower |
| Note ends early, clipping the sung note | `--release-drop` | raise |
| Onsets landing early on breathy audio | `glide_min_periodicity` | raise |
| Voice outside the tracked range | `--fmin` / `--fmax` | widen |

Every parameter in `config.py` is documented in place; the CLI exposes the ones
worth reaching for most often, and the rest are available through
`TranscriptionConfig`.

### By search

`transcpybe-tune` fits the parameters on one part of a corpus and reports what
the fit is worth on a part no trial and no selection touched. It needs the
benchmark package for its metrics and its dataset loading:

```shell
pip install -e ".[tune]"

transcpybe-tune bimmuda/audio --annotations bimmuda/midi \
    --align --trials 60 --beta 1.5
```

Three sets, not two: one to fit on, one to choose the winner on, and one scored
once at the end. The fit and select figures are both biased upward by the
search itself, so the held-out number is the only one worth quoting — and if it
barely moves, the search found the fit set rather than the problem.

`--beta 1.5` favours recall, which is the safer direction when a reference may
be incomplete: a note the reference lacks is counted against the transcriber
whether or not it was sung. `--align` warps notated references onto the audio
first; tuning against an unaligned score optimises the mismatch, which on
BiMMuDa scores 0.06 however good the transcription is.

#### Corpora too small for a held-out fifth

A three-way split of forty recordings reports on eight, and eight cannot
distinguish any improvement a search of this kind produces. `--cv K` scores
every recording instead, each by a search that never saw it:

```shell
transcpybe-tune vocadito/audio --annotations vocadito/annotations \
    --annotation-contains notesA1 --cv 5 --trials 60
```

```text
Out-of-fold over all 40 recordings, onset F-beta at beta=1.5:
  baseline      0.518
  tuned         0.949
  change       +0.431   95% CI [+0.271, +0.586]   better on 12, worse on 0

What each fold chose:
  parameter                  default   median            range  moved  spread  agree
  onset_split_strength_ratio     1.6   0.4299  0.2373 to 0.6989    3/3     71%   100%
  pitch_sigma_semitones         0.55   0.5523  0.5322 to 0.5949    3/3     14%    67%
```

*(that table is from a test corpus with one knob deliberately detuned, which
is why the first row is so clear)*

Read the fold table before the score, and read **moved** before anything else
in it. A fold whose search beat nothing keeps the defaults and contributes no
value at all, so a row moved by one fold of five is one opinion — and one value
has zero range and agrees with itself perfectly, which is the most flattering
possible rendering of the weakest possible evidence. In a single run it would
have printed as a chosen value.

Among the folds that did move a parameter: **spread** is how much of the
searched range they covered, so near 100% is what drawing at random looks like
(undefined below two folds, and shown as `-` rather than as zero). **Agree** is
how many went the same way, so near 50% is disagreement about even the
direction. A parameter can agree on direction and still be scattered, so read
both.

Folds keeping the defaults is itself the clearest result the search can
produce, and it is reported above the interval. It happens during fitting, on
data the fold chose freely, so it says more than a held-out interval can: the
interval can only report that whatever was found failed to carry.

Cross-validation spends the corpus. Every recording has been fitted on in some
fold, so afterwards it cannot report an unseen number, and the folds are
several answers rather than one configuration to keep. Use it to find out
whether there is anything in the parameters at all, then fit properly on a
corpus you can afford to hold out from.

## Tests

```shell
pytest tests/                 # the full suite
pytest tests/ -m slow         # just the ones that exercise CREPE
transcpybe --self-test        # the synthetic suite
```

The tuning tests need the benchmark package and skip without it.

The CREPE tests include the one that matters most for the progress machinery:
transcription drives torchcrepe's `preprocess`/`infer`/`postprocess` directly
rather than calling `predict`, purely so progress can be reported between
batches, and `test_chunked_crepe_matches_torchcrepe_predict` asserts the two
remain bit-identical.

---

## Limitations

- **Monophonic only.** One pitch at a time. Harmonies or overlapping vocal
  layers in the same stem will be transcribed as one wandering line.
- **Offsets are softer than onsets, and nothing here refines them.** Every
  stage in the pipeline works on note starts; a note ends wherever the pitch
  decoder stopped hearing pitch. `--release-trim` ends it at the top of the
  decay instead, which is right on signals whose notes are annotated where the
  pitch stops and wrong on vocadito, where it cost 0.161 COnPOff across all
  forty recordings. It is off by default for that reason. Where a note ends is
  a convention, so fit `--release-drop` against your own corpus with
  `transcpybe-tune --objective offset` before trusting either answer.
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
