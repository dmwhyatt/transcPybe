# Pitch plan: measuring a note's pitch after its start has moved

Written 26 October 2026, after the round-1 note-error diagnostic (`arms/diag-notes.txt` on musix) and before any decode with these options. Scoring follows `protocol-amendment-2.md`, as before. Nothing here changes it.

## Why

Round 1's best network (M4Singer only, B-plain at threshold 0.7, every onset may cut) finds 527 more of the development half's reference notes than preset B. Almost all of them are notes B had merged into the note before. But **213 of the 527 (40%) get the wrong pitch**, against 20% for the notes B already finds:
- they are off by one semitone (+134) or two (+66), not by octaves;
- more of them start over 50 ms before the reference note (+69);
- more of them take the previous note's pitch (+58);
- short notes are not the cause (+3 under 150 ms).

## The cause, in transcPybe's decoder

Two code paths in the spurious-notes tree (c7d3ab2) let a note's pitch be measured over frames still on the previous pitch:
1. **Snapping a start earlier** (`onsets.snap_segment_starts`). Where two notes join legato, the decoder places the boundary where the pitch has crossed over. Snapping pulls the start back onto the attack, which comes earlier. The moved note is then measured from its new start. Only the first 20% of the note, at most 60 ms, is skipped (`attack_trim_frac`, `attack_trim_max_s`). So the measurement can include the previous note's pitch.
2. **Cutting a note at an onset** (`onsets.split_segments_on_onsets`). Each part is measured from the cut, with the same short skip. The part after the cut can include the transition into its pitch.

The weighted median then rounds to the wrong semitone when enough of those frames are included.

## The change

Two options in transcPybe's configuration, both off by default, so that preset B and the shipped settings are unchanged:
- **`measure_after_snap`.** When a start is snapped earlier, the note is measured from where the decoder had placed it, where the pitch crossed over (`measure_from`, which the code already supports). The note still starts at the attack. This mirrors what the code already does when a start may not move earlier.
- **`cut_settle_s`.** A note created by an onset cut skips this many seconds after the cut before measuring, on top of the usual skip. If no frames are left, it falls back as now.

They are added on the `claude/spurious-notes` branch of transcPybe, with tests. That branch is not pushed without the go-ahead.

## What is decoded

On the RWC-P development half (50 songs, 48 at note level), at six settings:

| Setting | `measure_after_snap` | `cut_settle_s` |
|---|---|---|
| `p0` (current) | off | 0 |
| `p1` | on | 0 |
| `p2` | off | 0.05 |
| `p3` | off | 0.10 |
| `p4` | on | 0.05 |
| `p5` | on | 0.10 |

Each setting is decoded twice:
- **with round 1's network:** the M4Singer-only chosen checkpoints, 3 seeds, B-plain at onset threshold 0.7, every onset may cut (round 1's best hand-over). `p0` reproduces round 1's 0.6653.
- **without a network:** preset B as it stands, with transcPybe's own onsets (SuperFlux). This is reported only.

## Decision rules, fixed now

Each comparison takes each song's COnP@150 averaged over the 3 seeds, pairs it over songs, and uses a 95% bootstrap interval.

1. **Against B (amendment 2).** A candidate (the network at `p1`–`p5`) passes only if all of these hold:
   - it beats B by at least 0.01, with the interval excluding zero;
   - at least 2 of its 3 seeds beat B on their own;
   - frame overall accuracy falls by no more than 0.005;
   - notes in silence rise by no more than 0.5 points.

   If several pass, the one with the highest development score is frozen and hashed (checkpoints, setting, both code commits). It is then checked once on the test half: the first of the two test checks amendment 2 allows. Claude asks before spending it.
2. **Without a network.** If a setting helps preset B on its own, that is a separate change to transcPybe's preset. It decides nothing here, and would need its own plan and test check.
3. **If no candidate passes,** there is no test check. The next target is the second leak, the extra notes, with a plan of its own (stricter cutting).

**Reported, deciding nothing:**
- COnP@50 and COn@150;
- the note-error breakdown of `diag_notes.py` for every setting: reference notes found with the wrong pitch, and among those, how many start over 50 ms early and how many take the previous note's pitch. The change is working if these fall towards B's rate;
- every setting without the network.

## Steps

1. **transcPybe** (`claude/spurious-notes`, local):
   - add the two options to `TranscriptionConfig`, and use them in `snap_segment_starts` and `split_segments_on_onsets`;
   - a test: a legato pair whose second note is snapped earlier is measured from where its pitch settles;
   - commit, without pushing.
2. **boundary** (PR #4): `rwc_dev.py` gains `--config-overrides KEY=VALUE ...`, added to the decoder configuration, with the setting's name in the run name.
3. **Musix:**
   - copy both commits;
   - replay preset B at the six settings with `dev/replay.py`;
   - decode the 3 M4Singer-only checkpoints at the six settings, 8 decodes at a time on the CPU;
   - score, apply the rules with a short script (`pitch.txt`), and run the breakdown for each setting.
4. **Cost:** no CSD3 time. About an hour on musix.
