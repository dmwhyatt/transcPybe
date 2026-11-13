# transcPybe

transcPybe transcribes monophonic singing into notes: onset and offset times in seconds, and pitches as MIDI note
numbers, with the measured pitch retained as a pitch bend. It is intended for solo vocal recordings and for lead vocals
separated from mixed recordings.

Version 2 places note onsets with three small neural networks, which detect where each syllable begins to sound and
where its vowel begins. Notes are then decoded from the pitch track by a hidden Markov model. Version 1, which relies on
the pitch track and a spectral-flux onset detector alone, remains available.

The code is released under the MIT licence. The weights of the version 2 networks are released under CC BY-NC-SA 4.0
(see [Licence](#licence)).

## Comparison to other models

| System | vocadito A1 | vocadito A2 | Molina | RWC-P dev | SSVD test | MIR-ST500 test | A cappella | Pop | Five-set mean [95% CI] | GPU s/min | CPU s/min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Phoneme-informed\* | 0.450 | 0.593 | 0.712 | 0.317 | 0.688 | 0.372 | 0.641 | 0.345 | **0.522** [0.507, 0.537] | 13.4 | 12.4 |
| transcPybe v2, CREPE | 0.473 | 0.573 | 0.670 | 0.347 | 0.568 | 0.354 | 0.587 | 0.350 | **0.492** [0.479, 0.505] | 3.0 | 65.5 |
| transcPybe v2, PESTO | 0.467 | 0.574 | 0.693 | 0.348 | 0.601 | 0.348 | 0.605 | 0.348 | **0.502** [0.488, 0.516] | 2.1 | 4.4 |
| transcPybe v2, pYIN | 0.481 | 0.580 | 0.675 | 0.339 | 0.609 | 0.391 | 0.605 | 0.365 | **0.509** [0.495, 0.523] | 10.6 | 10.9 |
| CTC-CE\* | 0.312 | 0.433 | 0.629 | 0.359 | 0.504 | 0.538 | 0.502 | 0.449 | **0.481** [0.465, 0.496] | 6.8 | 3.6 |
| MIR-ST500 model\* | 0.320 | 0.443 | 0.534 | 0.371 | 0.453 | 0.474 | 0.456 | 0.422 | **0.443** [0.427, 0.458] | 5.9 | 3.3 |
| Tony † | 0.686 | 0.654 | 0.466 | 0.266 | 0.582 | 0.276 | 0.524 | 0.271 | **0.397**† [0.383, 0.412] | – | 1.1 |
| YourMT3+\* | 0.275 | 0.350 | 0.464 | 0.328 | 0.410 | 0.426 | 0.395 | 0.377 | **0.388** [0.373, 0.403] | 3.0 | 21.8 |
| GAME | 0.264 | 0.364 | 0.506 | 0.348 | 0.332 | 0.406 | 0.384 | 0.377 | **0.381** [0.366, 0.396] | – | 83.6 |
| Omnizart | 0.440 | 0.494 | 0.512 | 0.180 | 0.533 | 0.178 | 0.504 | 0.179 | **0.374** [0.361, 0.387] | – | 134.6 |
| transcPybe v1 | 0.496 | 0.480 | 0.419 | 0.231 | 0.449 | 0.237 | 0.452 | 0.234 | **0.365** [0.353, 0.376] | 3.0 | 60.7 |
| SOME | 0.278 | 0.349 | 0.441 | 0.242 | 0.272 | 0.311 | 0.342 | 0.276 | **0.316** [0.304, 0.327] | 3.9 | 5.7 |
| Kum et al. | 0.219 | 0.247 | 0.251 | 0.224 | 0.229 | 0.248 | 0.238 | 0.236 | **0.237** [0.227, 0.248] | – | 6.8 |
| Basic Pitch | 0.307 | 0.325 | 0.345 | 0.133 | 0.254 | 0.137 | 0.305 | 0.135 | **0.237** [0.224, 0.250] | – | 4.6 |
| Essentia | 0.318 | 0.263 | 0.289 | 0.120 | 0.360 | 0.117 | 0.313 | 0.118 | **0.235** [0.224, 0.247] | – | 1.8 |
| CREPE Notes | 0.310 | 0.283 | 0.275 | 0.167 | 0.256 | 0.158 | 0.276 | 0.162 | **0.230** [0.219, 0.242] | – | 29.9 |
| ROSVOT | 0.184 | 0.204 | 0.189 | 0.154 | 0.082 | 0.109 | 0.155 | 0.132 | **0.146** [0.137, 0.155] | 1.6 | – |
| aubio | 0.061 | 0.060 | 0.086 | 0.019 | 0.106 | 0.026 | 0.084 | 0.022 | **0.059** [0.055, 0.064] | – | 0.4 |
| STARS ‡ | 0.064 | 0.076 | 0.068 | 0.038 | 0.084 | 0.088 | 0.074 | 0.063 | **0.070** [0.062, 0.078] | – | – |

The table compares transcPybe with fifteen published singing transcription systems on five annotated datasets.

- **Runs.** Each system ran in its own pinned container on identical audio.
- **Metric.** COnPOff@50 at whole semitones: a note counts when its onset is within 50 ms, its pitch within 50 cents,
  and its offset within 20% of the note's duration or 50 ms, whichever is larger.
- **Intervals.** The five-set mean has a 95% interval from a paired bootstrap over recordings, stratified by dataset.
- **Time.** Seconds of computation per minute of audio, on an NVIDIA RTX 3080 GPU and on eight CPU cores.

The methods, every system's reference and the per-recording scores are in the
[transcription-benchmarking](https://github.com/dmwhyatt/transcription-benchmarking#results) repository.

Notes to the table:

- \* Trained on other recordings from one of the evaluation datasets.
- † vocadito's reference annotations were produced by correcting Tony's output, so Tony's mean excludes vocadito.
- ‡ STARS requires lyrics. They were obtained by automatic speech recognition.
- transcPybe v2's settings were selected on vocadito, Molina and the RWC-P development set; SSVD test and MIR-ST500
  test were held out.

For reference, vocadito's two annotators agree with each other at 0.64 COnPOff@50.

## Installation

```shell
pip install transcpybe              # version 2 with the pYIN pitch tracker
pip install "transcpybe[crepe]"     # adds the CREPE pitch tracker
pip install "transcpybe[pesto]"     # adds the PESTO pitch tracker (LGPL-3.0)
```

transcPybe requires Python 3.9 or later. PyTorch is a dependency, because the version 2 networks require it. On a
machine without a GPU, installing PyTorch's CPU build first avoids downloading the CUDA libraries:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`. To install from a source checkout, use
`pip install -e ".[dev]"`.

## Usage

### Command line

```shell
# A lead vocal separated from a mix: writes lead-vocal.mid, .notes.json and .notes.csv
transcpybe lead-vocal.wav -o out/

# Solo singing, with nothing else in the recording
transcpybe solo.wav --preset acappella

# Version 1, without the networks
transcpybe lead-vocal.wav --preset rules

# Every audio file in a directory, with a log file
transcpybe stems/ -o midi/ --log-file transcribe.log

# Print the notes as a table
transcpybe lead-vocal.wav --print-notes

# Write whole equal-tempered semitones only, without pitch bends
transcpybe lead-vocal.wav --12tet

# Run the built-in synthetic test suite
transcpybe --self-test
```

`transcpybe --help` lists every option.

### Presets

| Preset | Version | Intended input | Evaluated on |
|---|---|---|---|
| `pop` (default) | 2 | a lead vocal separated from a pop mix | 82 MIR-ST500 test songs |
| `acappella` | 2 | solo singing, with nothing else in the recording | 127 SSVD test recordings |
| `rules` | 1 | either; no neural networks besides the pitch tracker | |

The two version 2 presets were fixed on development sets and then evaluated once on recordings that had not been used
for tuning. Each preset also sets the decoder parameters with which it was evaluated: `--min-note`, `--p-self`,
`--note-level-floor`, `--onset-delta`, `--onset-split-strength` and `--release-drop`. The command line warns if one of
these is given with a version 2 preset. With `--preset rules`, every parameter takes the value given.

### Pitch trackers

| Tracker | Option | Requires | Notes |
|---|---|---|---|
| pYIN | `--f0-backend pyin` (default) | | Version 2's best five-set mean; runs on the CPU |
| CREPE | `--f0-backend crepe` | `transcpybe[crepe]` | Fastest on a GPU, slowest on a CPU; `--crepe-model tiny` is about 14 times faster on a CPU, with lower accuracy |
| PESTO | `--f0-backend pesto` | `transcpybe[pesto]` | A much smaller network than CREPE |

The version 2 networks were trained on CREPE's pitch tracks. pYIN and PESTO report lower confidence than CREPE. Their
tracks therefore pass through a voicing map before decoding: a quantile map onto CREPE's confidence scale, fitted on
the training data. Version 1's thresholds were set on CREPE. Further trackers can be added with
`transcpybe.f0.register_f0_backend`.

### Python

```python
from transcpybe import TranscriptionConfig, transcribe_file, write_midi

result = transcribe_file("solo.wav", TranscriptionConfig(preset="acappella"))
for note in result.notes:
    print(note.onset, note.offset, note.pitch)          # seconds, seconds, MIDI note number

write_midi(result.notes, "solo.mid", result.config)
```

`TranscriptionConfig` documents every parameter. The result also contains the pitch track (`f0_track`) and the onset
track (`onsets`) from which the notes were decoded.

### Output

- **`.mid`:** one track. Note times are absolute seconds; the tempo in the header affects only the bar grid displayed
  by notation software. Each note's MIDI number is the semitone nearest the sung pitch. By default, the measured pitch
  is also written as a pitch bend at the start of the note, using General MIDI's two-semitone range. `--12tet` omits
  the bends.
- **`.notes.csv`:** onset, offset, duration, pitch, velocity, confidence, pitch in Hz, and the measured pitch as a
  fractional MIDI number (`pitch_midi_float`).
- **`.notes.json`:** the same notes, with the complete configuration that produced them.

`transcpybe.midi_io.read_midi_notes` reads a MIDI file back with its pitch bends applied.

## Method

### Version 2: note onsets from neural networks

Three networks with the same architecture were trained with different random seeds:

- **Architecture:** a convolutional front end over an 80-band log-mel spectrogram, then a two-layer bidirectional GRU,
  with 0.8 million parameters each.
- **Inputs:** the spectrogram and five features from the pitch track.
- **Outputs:** every 10 ms, the probability that a syllable begins to sound and the probability that its vowel
  begins.
- **Training data:** singing from M4Singer and GTSinger, mixed with accompaniment from MUSDB18-HQ.

The three networks' outputs are averaged. Each vowel onset is paired with the nearest preceding syllable onset, and the
note onset is placed between them: at 0.5 of the interval in the `pop` preset, and at 0.875 in the `acappella` preset.
The decoder described below then forms notes from the pitch track, with these onsets.

The networks run on the CPU. On an Apple M-series laptop with PyTorch's default four threads, the three together take
about 3.6 s per minute of audio. The code is in `transcpybe.landmarks`. The weights, the voicing maps and their
provenance are in `src/transcpybe/landmarks/weights/`.

### The note decoder

Both versions share the following stages. Version 1 takes its onsets from stage 5. Version 2 takes them from the
networks and uses its preset's parameters throughout.

1. **Pitch and voicing.** The pitch tracker estimates a fundamental frequency and a periodicity confidence every
   10 ms. The confidence is multiplied by a loudness gate set relative to the recording's 95th-percentile level, so
   that periodic residue left by source separation in quiet passages is not transcribed.
2. **Note decoding.** A Viterbi decoder runs over pitch states spaced a third of a semitone apart, plus an unvoiced
   state. Emissions favour states near the measured pitch, weighted by voicing. Transitions favour remaining on a
   note, and penalise larger intervals. The design follows the note tracker of Tony (Mauch et al., 2015). The
   sub-semitone states prevent a note sung between two semitones from alternating between them.
3. **Fragments.** Very short segments adjacent to a much longer note, within two semitones of it, are merged into
   it. Such segments typically arise from wide vibrato at the end of a note.
4. **Glides.** Where the singer slides into a note, the transition is attributed to the note being approached. This
   corrects the late onset the slide would otherwise produce, and removes the brief notes it would leave on each
   semitone crossed.
5. **Onsets.** A spectral-flux onset envelope with a frequency-domain maximum filter (SuperFlux) splits repeated
   notes at the same pitch, and aligns note onsets with audible attacks. The maximum filter prevents vibrato from
   registering as onsets.
6. **Pitch and clean-up.** Each note's pitch is the confidence-weighted median of its pitch track, excluding the
   start of the note, rounded to the nearest semitone. Notes shorter than the minimum duration, and notes far quieter
   than the rest of the performance, are removed.
7. **Note ends.** A note ends where its level has fallen a set amount below its own sustained level: 15 dB by default,
   20 dB in the `pop` preset. This excludes the release and room reverberation, which pitch trackers continue to
   report as periodic.
8. **Tuning reference.** If the notes of a first pass show that the performance is consistently offset from the
   A440 grid, decoding is repeated against the grid on which it was sung. Offsets beyond 40 cents are not applied,
   because scattered intonation is otherwise better explained by a grid a quarter tone away.
   `--no-tuning-estimate` disables the estimate.

## Processing large collections

Given a directory, `transcpybe` transcribes every audio file within it:

- **Existing outputs:** files whose outputs already exist are skipped, so an interrupted run resumes; `--overwrite`
  transcribes them again.
- **Failures:** a file that cannot be read is logged and the run continues; `--stop-on-error` stops it instead.
- **Remaining time:** estimated from audio duration rather than file count.

Progress is shown on a single line in a terminal, and as periodic timestamped lines when output is redirected. The
following options control logging and the GPU:

| Option | Purpose |
|---|---|
| `--log-file PATH` | Append timestamped logs to a file as well as the console. |
| `--progress-interval N` | Seconds between progress lines when output is redirected (default 5). |
| `--no-progress` | Disable progress reporting. |
| `--device` | Torch device for CREPE and PESTO: `auto` (default; CUDA, then Apple MPS, then CPU), or an explicit device such as `cuda:1`. The device used is logged and recorded in the JSON output. |
| `--crepe-batch-size N` | CREPE frames per forward pass (default 512); lower it if GPU memory is limited. |
| `--gpu-memory-every N` | Log GPU memory every N CREPE batches. |
| `-v` | Include debugging detail. |

## Reproducibility

The JSON output records the complete configuration used. CREPE adds random dither of ±20 cents to its pitch estimates.
transcPybe seeds it (`--seed`, default 0) and restores the global random state afterwards, so repeated runs give
identical output. `--no-seed` restores CREPE's unseeded behaviour.

## Adjusting version 1

With `--preset rules`, the decoder's parameters may be adjusted. The most commonly used are:

| Symptom | Option | Direction |
|---|---|---|
| Short spurious notes | `--min-note` | increase (e.g. 0.08) |
| Sustained notes split into several | `--p-self` | increase (e.g. 0.995) |
| Fast passages merged into single notes | `--p-self` | decrease (e.g. 0.97) |
| Vibrato fragmenting notes | `--pitch-sigma` | increase (e.g. 0.7) |
| Notes in silent passages | `--note-level-floor` | decrease (e.g. 22) |
| Quiet sung notes discarded | `--note-level-floor` | increase |
| Repeated notes at one pitch merged | `--onset-split-strength` | decrease |
| Noisy recordings over-segmented | `--onset-delta` | increase |
| Note ends too late | `--release-drop` | decrease (e.g. 10) |
| Note ends too early | `--release-drop` | increase, or `--no-release-trim` |
| Voice outside the tracked range | `--fmin`, `--fmax` | widen |

`transcpybe-tune` searches these parameters on part of an annotated corpus, and reports the result on a part that
neither the search nor the selection used. With `--cv K`, it reports a K-fold cross-validated estimate instead. It
requires the transcription-benchmarking package, which is installed from its repository:

```shell
pip install git+https://github.com/dmwhyatt/transcription-benchmarking.git
transcpybe-tune corpus/audio --annotations corpus/annotations --trials 60
```

## Tests

```shell
pytest tests/                 # the test suite
pytest tests/ -m slow         # the tests that run CREPE
transcpybe --self-test        # the synthetic suite
```

The tuning tests require the transcription-benchmarking package, and are skipped without it.

## Limitations

- **Monophonic input.** Harmonies or overlapping voices in the same recording are transcribed as a single line.
- **Note ends follow a convention.** Annotators differ on where a sung note ends. The release threshold
  (`--release-drop`) may need adjusting for a particular corpus.
- **Separation residue.** Loud residue from source separation can be transcribed as notes. With `--preset rules`,
  lowering `--note-level-floor` reduces this, at the cost of quiet singing.
- **Tuning.** A performance sung more than 40 cents from A440 is decoded on the A440 grid.
- **No quantisation.** Output times are in seconds, without beat tracking or rhythmic quantisation.

## Acknowledgements

**Pitch estimation**

- **CREPE:** J. W. Kim, J. Salamon, P. Li and J. P. Bello, "CREPE: A Convolutional Representation for Pitch
  Estimation", ICASSP 2018. Used through **torchcrepe** (M. Morrison; MIT licence), which supplies the weights.
- **pYIN:** M. Mauch and S. Dixon, "pYIN: A fundamental frequency estimator using probabilistic threshold
  distributions", ICASSP 2014. Used through librosa's implementation.
- **PESTO:** A. Riou, S. Lattner, G. Hadjeres and G. Peeters, "PESTO: Pitch Estimation with Self-supervised
  Transposition-equivariant Objective", ISMIR 2023. Used through the pesto-pitch package (LGPL-3.0).

**Note segmentation and onsets**

- **Tony:** M. Mauch, C. Cannam, R. Bittner, G. Fazekas, J. Salamon, J. Dai, J. Bello and S. Dixon, "Computer-aided
  melody note transcription using the Tony software: Accuracy and efficiency", TENOR 2015. The pitch-state hidden
  Markov model is a simplified form of Tony's note tracker.
- **SuperFlux:** S. Böck and G. Widmer, "Maximum filter vibrato suppression for onset detection", DAFx 2013. Used
  through librosa's `onset_strength`.

**Training data for the version 2 networks**

- **M4Singer:** L. Zhang et al., "M4Singer: A Multi-Style, Multi-Singer and Musical Score Provided Mandarin Singing
  Corpus", NeurIPS 2022 Datasets and Benchmarks. CC BY-NC-SA 4.0.
- **GTSinger:** Y. Zhang et al., "GTSinger: A Global Multi-Technique Singing Corpus with Realistic Music Scores for
  All Singing Tasks", NeurIPS 2024 Datasets and Benchmarks. CC BY-NC-SA 4.0.
- **MUSDB18-HQ:** Z. Rafii, A. Liutkus, F.-R. Stöter, S. I. Mimilakis and R. Bittner, 2019. Accompaniment only;
  educational and non-commercial use.

**Libraries**

- librosa (ISC licence)
- pretty_midi (MIT)
- mir_eval (MIT)
- NumPy (BSD-3-Clause)
- SciPy (BSD-3-Clause)
- soundfile (BSD-3-Clause)
- PyTorch (BSD-3-Clause)

## Licence

The code is released under the MIT licence (`LICENSE`).

The weights of the version 2 networks and the voicing maps, in `src/transcpybe/landmarks/weights/`, are released under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), because the singing on which they were trained
is licensed on those terms:

- **Attribution:** use requires attribution.
- **Non-commercial:** commercial use is not permitted.
- **Share-alike:** derived weights must be shared under the same licence.

That folder's `LICENSE` and `README.md` give the full terms and the provenance. The `pop` and `acappella` presets use
these weights; `--preset rules` does not. The pitch trackers' weights are distributed with their own packages, under
their own licences.
