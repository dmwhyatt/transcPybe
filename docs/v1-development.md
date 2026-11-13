# transcPybe v1: development notes

These notes were moved from the README at version 2.0.0, unchanged. They record how transcPybe v1 (the rules
decoder, now `--preset rules`) was developed and measured, mostly with CREPE as its pitch tracker and on synthetic
recordings. They do not describe transcPybe v2. For the comparison with other systems on annotated singing, see the
README.

## Accuracy on synthetic recordings

Measured on the built-in synthetic benchmark (`--self-test`): 22 cases, 174
notes, scored with `mir_eval` at the MIREX 50 ms onset tolerance and quarter-tone
pitch tolerance.

| Backend | Cases perfect | Onset F | Note F (onset+pitch) | Note F (+offset) | Speed (4 CPU cores) |
|---|---|---|---|---|---|
| `crepe` (`full`, default) | **18 / 22** | 1.000 | **0.968** | **0.908** | ~1.6x realtime |
| `pyin` | 19 / 22 | 0.938 | 0.944 | 0.899 | ~0.34x realtime |
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
1.000 / 0.968 / 0.862 after (0.908 on the last, since note ends have been
trimmed); over the 18 cases that existed before, it scored 1.000 / 0.978 /
0.978 either way. That is the reason the four were added rather
than a point against them: a suite at its ceiling cannot rank two
configurations, and this one was.

**And read the whole table against vocadito, because it did not predict it.**
The onset work above is worth 0.057 onset F here and, on forty real
recordings, +0.005 with a 95% interval of [-0.006, +0.018] -- better on 19,
worse on 19, which is the shape of a change that rearranges rather than
improves. `release_tails` still scores 0.000 on the offset column. It
annotates each end at the top of a 180 ms decay, and the release trim that
reached it there, at 1 dB, cost 0.161 COnPOff on every one of those forty
recordings; the trim that ships, at 15 dB, ends notes further down the decay,
which fixes `breath_ends` (0.000 to 1.000) and leaves `release_tails` short of
the top. Synthetic signals with exact ground truth are the right place to find
a defect and a poor place to confirm a cure.

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

### What the tuning reference is worth

The suite cannot show this, by design: every case in it is sung on the A440
grid, so the estimate correctly declines on all 22 and the table above is
identical with it on and off. What it is worth shows up only on a performance
that is not on that grid.

Paired on the same audio, the same f0, and the same seeds — phrases sung a
fixed distance off the grid with per-note intonation scatter on top, scored as
note F (onset+pitch):

| Offset | Scatter | Forced A440 | Tuned | |
|---|---|---|---|---|
| 0 | 25c | 0.972 | 0.972 | estimate declines |
| 15c | 25c | 1.000 | 0.972 | nothing to win; one note lost |
| 25c | 25c | 0.833 | **0.917** | |
| 35c | 15c | 0.833 | **1.000** | |
| 35c | 25c | 0.750 | **0.917** | |

Better on 7 of those 15 recordings, worse on 1, unchanged on 7. The shape is
the point: it does nothing until the grid is demonstrably costing notes, and
a recording already near A440 never reaches the second decoding pass at all —
its output is the same object the single pass produced.

These are synthesised, and the caveat below applies to them with force. The
argument that this should carry to real singing is that a human annotator
hears a performance in its own tuning, so an estimate that does the same ought
to agree with them more often. That is an argument rather than a measurement,
and it wants a benchmark run before it is believed.

That table predates two changes, and is reproduced with
`--pitch-states-per-semitone 1 --pitch-reference tuned`, scored against the
pitches intended. A grid finer than a semitone now gives off-grid notes a state
of their own whether or not the estimate fires, and the default reports the
pitch sung rather than the pitch meant: against vocadito-style references, which
give the sung pitch, rounding on the recording's own grid could only lose notes.

### What the realistic corpus says

The suite above scores 1.000 on most of its cases, which is its job and also
its limit: it cannot rank two pipelines that both pass it, and it contains none
of what decides where a real note ends -- every case stops a note's audio at
the instant the reference ends it, in an anechoic room. `corpus.py` is the
other half: 80 randomised recordings of unaccompanied, amateur-ish singing in
a room -- reverberation, releases and falls, consonants that open voicing gaps
or dip the level, diphthongs, inhales, singers anywhere within 50 cents of
A440 with drift and scatter -- annotated by stated rules. Forty were used to
find and tune the changes below; the other forty were scored once, after.

Because where a note ends is a convention, every recording is scored under
several: release ends annotated a quarter, half or all the way down the decay;
consonant gaps closed up to the next onset; and pitch as sung (fractional, as
vocadito annotates) or as intended. On the 40 held-out recordings, against the
previous release:

| Annotation convention | COn | COnP | COnPOff | better / worse |
|---|---|---|---|---|
| release ends half way down | 0.807 → **0.853** | 0.768 → **0.826** | 0.586 → **0.704** | 35 / 3 |
| release ends a quarter down | | | 0.569 → **0.703** | 36 / 2 |
| release ends at the bottom | | | 0.598 → **0.700** | 35 / 4 |
| consonant gaps closed | | | 0.623 → **0.704** | 32 / 6 |
| intended pitch | | 0.725 → **0.751** | 0.547 → **0.631** | 31 / 6 |

COnPOff +0.118 [+0.084, +0.154] in the first row, with every interval in the
table excluding zero. In order of what they are worth, taking each out of the
shipped pipeline in turn:

- **Note ends trimmed at 15 dB** (`trim_note_releases`, `release_drop_db`):
  COnPOff -0.051 without, worse on 25 recordings and better on none. Under the
  closed-gap convention it is worth +0.009, not significantly -- the one
  convention under which ending a note at its consonant can cost anything.
- **Glide backtracking against the sung pitch, stopping at a plateau**: the
  largest gain in onsets. Added on its own, COn +0.044 [+0.027, +0.062],
  better on 28 and worse on 4: a note sung 40 cents off its semitone no longer
  has the next note's start walked 300 ms back through it.
- **Rounding to the sung pitch** (`pitch_reference`): COnP -0.023 [-0.051,
  -0.002] without, against sung-pitch references; +0.011 (not significant)
  against intended ones. That is the trade, and the default follows the
  benchmarks.
- **Shared-boundary snapping**: +0.012 COnPOff, better on 8, worse on none.
- **Third-of-a-semitone decoding** (`pitch_states_per_semitone`): -0.007
  without, inside the noise, because the tuning estimate covers most of the
  same ground; without either, -0.020 [-0.033, -0.008], worse on 14 and
  better on 1. It is kept because it handles what the estimate cannot --
  per-note scatter, drift, a performance 45 cents off -- and makes the
  estimate's second decoding pass nearly redundant.
- **Passing tones must touch their neighbours**: one recording in eighty
  moves; the change stops a real short note between two rests being deleted
  as a glide crossing.

The corpus also reproduces the one real-data result there is to check it
against. The release trim at its old 1 dB setting cost vocadito -0.161 [-0.191,
-0.134], worse on all forty recordings; on the held-out corpus with consonant
gaps closed it costs -0.173 [-0.212, -0.133], worse on 36. With gaps left open
it costs -0.018, not significantly. So vocadito most likely annotates notes up
to the next onset, and the closed-gap row is the one to read.

To reproduce, or to measure a change of your own:

```shell
python scripts/corpus_score.py --out branch.json
PYTHONPATH=../other-checkout/src python scripts/corpus_score.py --out other.json
python scripts/corpus_score.py --compare other.json branch.json
```

The first run spends a few CPU-minutes per minute of audio on CREPE; the cache
makes every later run, of any checkout, take seconds.

### What these numbers do and do not mean

The benchmark is **synthesised**, not real singing. The synthesiser models
vibrato, portamento, scoops, jitter, consonant transients, breath noise and
separation residue, and each case was written to break a specific stage — but
real voices are harder than any of it. Read these scores as evidence that the
implementation does what it claims on the phenomena it models, not as an
estimate of accuracy on a real recording. For that, transcribe a stem you know
and read `--print-notes` against it.


## What seeding CREPE's dither costs

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

## Reading a tuning search

Three sets, not two: one to fit on, one to choose the winner on, and one scored
once at the end. The fit and select figures are both biased upward by the
search itself, so the held-out number is the only one worth quoting — and if it
barely moves, the search found the fit set rather than the problem.

`--beta 1.5` favours recall, which is the safer direction when a reference may
be incomplete: a note the reference lacks is counted against the transcriber
whether or not it was sung. `--align` warps notated references onto the audio
first; tuning against an unaligned score optimises the mismatch, which on
BiMMuDa scores 0.06 however good the transcription is.

### Corpora too small for a held-out fifth

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
