# Benchmarking ROSVOT against this transcriber on vocadito

Comparing published figures compares datasets. ROSVOT reports COnPOff 77.4 on
M4Singer, which it was trained on; two human annotators on vocadito agree at
64.2. The only comparison that means anything is both systems over the same
audio, scored against the same references with the same tolerances.

vocadito is the right venue: neither system trained on it, it is annotated from
pitch rather than from phonemes, and its inter-annotator agreement is measured,
so both scores can be read against the same ceiling.

## 1. Run ROSVOT

```shell
./scripts/rosvot-transcribe.sh /path/to/vocadito
```

ROSVOT is not in this repository. It lives at `/app/rosvot` inside the pipeline
image, with its checkpoints bind-mounted, so this needs no ROSVOT environment on
the host -- and running it from this repository's root fails, because
`inference/rosvot.py` is a path inside that image.

The script builds the metadata, mounts the audio read-only, runs as the invoking
user so the output is not left owned by root, and reports how many MIDI files
were written against how many were expected. Checkpoints are expected under
`checkpoints/rosvot/` and `checkpoints/rwbd/`; set `CHECKPOINTS_DIR` if they are
elsewhere, and `NO_RWBD=1` if there is no `rwbd` checkpoint.

To drive ROSVOT yourself instead, the equivalent is:

```shell
python inference/rosvot.py \
    --metadata metadata.json \
    -o rosvot_out \
    --ckpt checkpoints/rosvot/model.pt \
    --apply_rwbd --wbd_ckpt checkpoints/rwbd/model.pt \
    --verbose
```

with metadata from `python scripts/rosvot_metadata.py <audio_dir> -o metadata.json`.
Item names come from the audio stem, which is what lets `--estimates` pair the
MIDI back to the same recordings.

Two flags are not optional, for reasons that are not obvious from the help text.

**`--verbose`.** Failures are otherwise silent. In `save_result`, the branch
that regulates note intervals against word boundaries ends its `except` with a
bare `return`, so an item that raises is skipped with no MIDI written and
nothing printed.

**`--apply_rwbd`, or word boundaries, or `infer_regulate_real_note_itv=False`.**
That regulation step needs `word_durs` in the metadata. vocadito has no
syllable annotations, so it raises for every item and the output directory
comes out empty. `--apply_rwbd` takes the other branch, which writes note
intervals directly. With no `rwbd` checkpoint, set
`infer_regulate_real_note_itv: False` in the hparams instead.

Check the count before going further -- one MIDI per recording, or something was
skipped. The launcher does this for you. A short run is worse than a failed one,
because the recordings a system skips are its hardest, and dropping them
flatters it.

## 2. Score both, and compare

```shell
python -m transcpybe.benchmark /path/to/vocadito \
    --estimates data/work/rosvot/midi -o rosvot.json

python -m transcpybe.benchmark /path/to/vocadito \
    --cache-dir .bench-cache-2 -o ours.json

python -m transcpybe.benchmark --compare rosvot.json ours.json
```

The comparison is paired over the recordings both runs cover: the difficulty of
a recording is shared between the two systems, and comparing separate
confidence intervals counts it twice.

For the ceiling on the same data:

```shell
python -m transcpybe.benchmark /path/to/vocadito --agreement notesA1 notesA2
```

## The result, 11 September 2026

40 recordings, 2237 reference notes. COnP is the headline; COnPOff also
requires the note end.

| system | COn | COnP | COnPOff |
|---|---|---|---|
| ROSVOT, as it runs | 0.359 | 0.262 | 0.184 |
| ROSVOT, `--onset-shift 0.041` | 0.472 | 0.355 | 0.261 |
| this transcriber | **0.707** | **0.589** | **0.407** |
| two human annotators | 0.817 | 0.740 | 0.642 |

Paired over the 40 shared recordings, against the corrected ROSVOT run:
COn +0.236 [+0.201, +0.272], COnP +0.234 [+0.204, +0.265], COnPOff +0.146
[+0.113, +0.180]. Better on 40 recordings of 40 for onsets and pitch, 36 of
40 for offsets. Every interval excludes zero.

### The correction, and why it is applied

ROSVOT's notes arrived about 66 ms wider than the reference: starting 41 ms
earlier and ending 25 ms later. That is M4Singer's convention, where a note
spans the syllable including its leading consonant, carried into a dataset
annotated from pitch. Scoring it uncorrected would penalise ROSVOT for a
difference in what the two sides call a note, which is the same artefact
this transcriber suffered when scored against M4Singer.

`--onset-shift 0.041` aligns the starts and leaves the ends alone. After it,
the measured start difference is -0.001 s, so the correction is the right
size. It recovers about a third of the gap -- COnP 0.262 to 0.355 -- and the
remaining 0.234 is not convention.

### What was ruled out

Language. ROSVOT scores 0.247 on the Mandarin tracks, the language it was
trained on, against 0.254 English, 0.231 French and 0.276 Tagalog: flat
within noise, so linguistic domain shift does not explain the gap. Mandarin
is two tracks, so this is evidence against a large effect rather than proof
of none.

Still untested: acoustic conditions. M4Singer is 20 professionals in a
studio, vocadito is untrained singers on whatever device was to hand.
`--by-metadata ...:singer_id` gives the same breakdown by singer.

### What this claims

On solo singing that is multilingual, not studio-recorded, sung by singers
of varying training, and annotated from pitch, this transcriber is
substantially more accurate than ROSVOT, after correcting for the annotation
convention ROSVOT brings with it.

It does not claim ROSVOT is a worse model in general. vocadito is out of
distribution for ROSVOT, which reports 77.4 COnPOff on M4Singer, and its own
paper notes it degrades out of domain. The comparison is fair in that
neither system trained on vocadito, and asymmetric in that a system with no
training data cannot be out of domain anywhere -- which is the trade being
measured, and it runs the other way on M4Singer.

Both numbers should be read against the ceiling. Two annotators agree at
COnP 0.740, so 0.589 is 80% of the distance a transcriber could travel, and
0.355 is 48%.

## Reading the result

**Check the alignment warning on ROSVOT's run first.** ROSVOT learned
M4Singer's convention, where a note begins at the syllable including its
leading consonant; vocadito annotates from pitch. If that convention travels
with the model, its onsets will land systematically before vocadito's
references and the report will say so and suggest an `--onset-shift`. That is
the mirror of what this transcriber suffered on M4Singer, and leaving it
uncorrected would be as unfair to ROSVOT here as M4Singer was to us. Score it
both ways and report both.

**Neither system is at home here.** ROSVOT is trained on Mandarin studio
recordings; vocadito is seven languages recorded on whatever was to hand, and
ROSVOT's paper concedes it degrades out of domain. That is the fair test for
the use this pipeline puts it to, but it is not the number its authors would
quote, and saying so is part of reporting it honestly.

**The ceiling is the context.** Two annotators agree at COnP 0.740 and COnPOff
0.642 on this data. A system at 0.6 COnP is closer to the ceiling than the raw
figure suggests, and one reporting far above 0.74 has a scoring fault rather
than an exceptional model.
