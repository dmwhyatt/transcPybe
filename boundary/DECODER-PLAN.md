# Decoder plan: cutting notes only where the re-sung output fires

Written 26 October 2026, after training round 1 was scored and before any decode with this hand-over. Scoring follows `protocol-amendment-2.md` (RWC work directory on musix), as round 1 did. Nothing here changes it.

## Why

In round 1 the network found note starts clearly better than transcPybe's own detector. Note starts regardless of pitch (COn@150) rose by 0.017 over preset B, but the note score (COnP@150) rose by only 0.0065.

The loss is in the hand-over. Today every strong network onset may cut a note. The pitch decoder already starts a note wherever the pitch changes, so a cut there can only add a note. The fourth output, trained in round 1 but not yet used, marks only the boundaries the pitch cannot explain: a note sung again on the previous note's pitch, with no gap.

## The new hand-over

transcPybe's decoder takes two kinds of onset:
- onsets with no strength, which move a nearby note start but never cut a note;
- split-only times, which cut a note and block the same-pitch merge, but never move a note start.

The new hand-over uses each output for one job:
- **Every peak of the onset output** at threshold 0.7 (round 1's primary) is passed with no strength, so it moves note starts and cuts nothing.
- **Every peak of the fourth output** at threshold *s* is passed as a split-only time, so it cuts the note it falls in, subject to the decoder's existing gap and voicing guards.

The decoder configuration is **B-plain**: preset B without its timbre and voicing-break splits. The fourth output then does the cutting that those hand-made rules did. Variant B is left out: its timbre splits are computed only when no split-only times are passed, so it would mix two cutting rules unpredictably.

## What is decoded

On the RWC-P development half (50 songs, 48 at note level), at six settings per checkpoint:

| Setting | Notes cut at | Role |
|---|---|---|
| `rs0.5`, `rs0.7`, `rs0.9` | fourth-output peaks at *s* = 0.5, 0.7, 0.9 | candidates |
| `snap` | nowhere: onsets only move note starts | control: the value of snapping alone |
| `oracle` | the reference's re-sung note starts (`train.resung` on the RWC-P reference notes) | control: the most this hand-over could give |
| round 1's `B-plain-t0.7` | every onset-output peak, already scored | the comparison |

The checkpoints are round 1's chosen ones (`best.pt`):
- **candidates** come only from the M4Singer-only arm, the recipe round 1 kept (rules 1 and 2);
- the two combined arms are decoded at the same settings and reported only.

The oracle control uses reference labels, so it is development-only and selects nothing.

## Decision rules, fixed now

Each comparison takes each song's COnP@150 averaged over the 3 seeds, pairs it over songs, and uses a 95% bootstrap interval.

1. **Against B (amendment 2).** A candidate (M4Singer-only, `rs0.5`, `rs0.7` or `rs0.9`) passes only if all of these hold:
   - it beats B by at least 0.01, with the interval excluding zero;
   - at least 2 of its 3 seeds beat B on their own;
   - frame overall accuracy falls by no more than 0.005;
   - notes in silence rise by no more than 0.5 points.

   If several pass, the one with the highest development score is frozen and hashed (checkpoints, threshold, code commit). It is then checked once on the test half: the first of the two test checks amendment 2 allows. Claude asks before spending it.
2. **Against round 1's hand-over.** The new hand-over becomes the default for later work if its best candidate beats round 1's M4Singer-only `B-plain-t0.7` (0.6653) with the interval excluding zero. Picking the best of three thresholds makes this lenient; it chooses a direction for later work, not what ships.
3. **If no candidate passes rule 1,** there is no test check. The next step is the deeper change: giving the onset probability to the pitch decoder itself, so that it weighs pitch and onset evidence together. That needs its own plan.

**Reported, deciding nothing:**
- COnP@50 and COn@150;
- notes per song, and the error types `score.py` counts, especially same-pitch merges (`mrgSame`) and same-pitch splits (`splSame`);
- the `snap` and `oracle` controls, and the combined arms at every setting;
- the fourth output's precision and recall against the RWC-P reference's re-sung notes, at 50 ms and 150 ms, for each threshold.

## Steps

1. **Code** (transcPybe PR #4, `boundary/`):
   - `infer.onset_track` gains the policy: the onset output's peaks with no strength, and the fourth output's peaks as split-only times;
   - `rwc_dev.py` gains `--resung-thresholds 0.5 0.7 0.9`, `--snap-only` and `--oracle-resung`, with run names `<name>-B-plain-t0.7-rs<s>`, `-snap` and `-oracle`;
   - tests for each.
2. **Musix:**
   - copy the commit to `code-<commit>`;
   - decode 9 checkpoints × 5 new settings with `decode.sh`'s parallel layout, 8 at a time on the CPU (about 45 minutes);
   - score with `dev/score.py` and apply the rules with a short script, writing `decoder.txt`.
3. **Cost:** no CSD3 time. About an hour on musix.
4. **Report:** write the results into the report, every setting listed.
