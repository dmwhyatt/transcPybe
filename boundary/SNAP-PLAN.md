# Snap plan: letting note starts reach the network's onsets

Written 27 October 2026, after the start-timing diagnostic (`start-timing-plan.md`, sha 96874d25…; `arms/start-timing.txt` and `start-window.txt` on musix), and before any decode with these settings. Judged by protocol amendment 3 (COnPOff@50).

## Why

The diagnostic looked at frozen candidate 2's near misses on the development half: notes with the right pitch whose start is 50–150 ms off the reference.
- **Share:** 25% of the notes matched within 150 ms, against 16% for GAME.
- **No shift helps:** they go both ways, so a constant shift does not help (−0.0009, cross-validated).
- **The network had the right time for 51% of them:** it fired a peak within 50 ms of the reference start, but the decoder did not use it. Fixing these perfectly is worth +0.049 COnPOff@50.
  - Most of that (+0.041) is a peak more than 50 ms from the decoder's start, beyond the snap window (`onset_snap_window_s` = 0.05).
  - +0.007 is a peak later than the start, within the window, that snapping only earlier (`onset_snap_direction` = "earlier") would not take.
- **The network's own timing was off for 43% (+0.036).** That is for a later retraining, not this plan.

Cutting a note does not reach these peaks either: B forbids a cut within 150 ms of an existing boundary (`onset_split_min_gap_s` 0.15).

Of the near misses whose right peak lies beyond the window:
- 62% sit after it, by a median of 90 ms. The pitch-change boundary lags the attack.
- 38% sit before it, by a median of 80 ms.

Within 150 ms, that peak is also the nearest one to the start for about three-quarters of them. So a wider window would usually choose it.

The risk is the other side: a wider window also reaches wrong peaks from starts that are right now. Moving a start later, as "both" allows, also leaves the frames before it unvoiced. Both show up in the scores below.

## The change

**No code changes.** Both options exist in transcPybe (3361803, frozen candidate 2's tree), and `rwc_dev.py` (boundary 7432d9a) passes them through `--config-overrides`:
- **`onset_snap_window_s`:** a start moves to the nearest onset within this window.
  - Moving it earlier across a legato boundary takes the previous note's end with it, as long as that note keeps `min_note_duration_s`.
  - In B, the window is used nowhere else: the late-arrival search, which also reads it, is off (`syllable_lead_s` = 0).
- **`onset_snap_direction`:** "earlier" (as frozen) or "both".

## What is decoded

On the RWC-P development half, using frozen candidate 2's hand-over:
- the m4gts-aug chosen checkpoints, each seed with its own outputs;
- B-plain, onset threshold 0.5, every onset may cut;
- `sustain_merge_above` 0.99.

It is decoded at five settings:

| Setting | `onset_snap_window_s` | `onset_snap_direction` |
|---|---|---|
| `w05e` | 0.05 | earlier (frozen candidate 2) |
| `w10e` | 0.10 | earlier |
| `w15e` | 0.15 | earlier |
| `w10b` | 0.10 | both |
| `w15b` | 0.15 | both |

This makes 15 decodes. Before any rule, `w05e` must reproduce frozen candidate 2 exactly: per-song COnPOff@50, COnP@150 and OA, with COnPOff@50 0.2930.

## Decision rules, fixed now

Each comparison takes each song's score averaged over the 3 seeds, pairs it over songs, and uses a 95% bootstrap interval (10,000 resamples, seed 0).

1. **Against B:** each candidate (`w10e`, `w15e`, `w10b`, `w15b`) is judged by protocol amendment 3's conditions:
   - COnPOff@50 at least +0.01 over B, with the interval excluding zero;
   - at least 2 of 3 seeds above B;
   - OA falls by at most 0.005;
   - notes in silence rise by at most 0.5 points;
   - COnP@150 falls by at most 0.005.
2. **Against frozen candidate 2:** the highest passing candidate replaces frozen candidate 2 as the one to test only if it also beats it on COnPOff@50 with the 95% interval excluding zero. Otherwise frozen candidate 2 stays. A replacement is frozen and hashed (`frozen-candidate-3.json`) before anything else.
3. **No test check yet:** none is spent without the user's go-ahead. One remains.

**Reported, deciding nothing:**
- each setting against `w05e`, on COnPOff@50, COnP@50, COnPOff@150 and COnP@150;
- frame accuracy minus B, split into pitch (RPA) and voicing (VR, VFA);
- for each setting, all seeds:
  - the share of notes matched within 150 ms whose start is within 50 ms;
  - the near-miss share, as in the diagnostic's M1;
  - of the notes whose start was within 50 ms at `w05e`, how many are no longer within 50 ms (starts the change broke).

## Steps

1. **Musix, before any decode:** this plan is hashed (`snap-plan.md` and `.sha256` in `~/boundary-work-20261020`).
2. **Decode:** `arms/decode-snap.sh` runs the 15 decodes, 12 at a time on the CPU, as the ends plan did. It then scores them with `dev/score.py` (`snap-rwcdev.txt`).
3. **Rules:** `arms/snap.py` applies them (`snap.txt`).
4. **Cost:** no GPU and no CSD3 time, about 30 minutes on musix.
