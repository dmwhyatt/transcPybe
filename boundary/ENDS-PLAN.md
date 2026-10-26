# Ends plan: placing note ends at the network's offset peaks

Written 26 October 2026, after the ends diagnostic (`arms/diag-ends.log` on musix) and before any decode with this change. Judged by protocol amendment 3 (COnPOff@50).

## Why

COnPOff is the user's priority, and note ends are transcPybe's largest gap on it. On the development half, transcPybe scores 0.26–0.29 on COnPOff@50 against GAME's 0.35, while on COnP@150 it is level with GAME or ahead.

The ends diagnostic took the notes whose onset (within 50 ms) and pitch are right, as COnP@50 counts them. For preset B, round 1's M4Singer-only network and frozen candidate 1:
- Only **54–57%** of those notes also have the right end: within 20% of the reference note's length, or 50 ms, whichever is larger. Wrong ends are early and late about equally often (20–24% each).
  - **Late ends** are a median of 121–141 ms late. 55–60% of them run straight into the next note, and in about half the reference leaves a gap of over 50 ms there.
  - **Early ends** are a median of 120–125 ms early, and most stop before a rest.
- The network's **offset output** (peaks of at least 0.5) has a peak within the tolerance of the true end:
  - for 47–51% of the wrong ends, and 36–37% where the note can reach it (after its onset, before the next note starts);
  - for 83–87% of the right ends, and 68–69% within reach.
- The **sounding output** marks ends poorly (its first fall lands at the true end for 8–36%), and is not used here.

About 3,500 notes on the development half have the right onset and pitch but the wrong end. Each one fixed raises COnPOff without touching anything else.

## The change

**In transcPybe,** one option, off by default:
- `end_snap_window_s` (default 0, off). After everything else (pitch measured, notes merged, releases trimmed, short notes dropped), each note's end moves to the nearest of the onset track's `end_times` within this window either side of it. It may not:
  - pass the next note's onset;
  - come within `min_note_duration_s` of its own onset.

  With no such time in reach, the end stays where it is.
- Onsets and pitches do not change, so COnP is unchanged exactly. Frame accuracy changes only through voicing.
- The end times travel on the `OnsetTrack`, as a new optional `end_times` array, passed through every rebuild as `sounding` is.

**In boundary:**
- `onset_track` takes the end times.
- `rwc_dev.py` sets them to the offset output's peaks at 0.5 (`pick_peaks`, as for onsets: local maxima, at least 50 ms apart).

## What is decoded

On the RWC-P development half (48 songs at note level), frozen candidate 2's hand-over: the m4gts-aug chosen checkpoints, B-plain, onset threshold 0.5, every onset may cut, `sustain_merge_above` 0.99. Each seed uses its own outputs, at four settings:

| Setting | `end_snap_window_s` |
|---|---|
| `e0` | 0 (off) |
| `e10` | 0.10 |
| `e20` | 0.20 |
| `e30` | 0.30 |

This makes 12 decodes. Before any rule, `e0` must reproduce frozen candidate 2 exactly (COnPOff@50 0.2930).

## Decision rules, fixed now

Each comparison takes each song's score averaged over the 3 seeds, pairs it over songs, and uses a 95% bootstrap interval.

1. **Against B:** each candidate (`e10`, `e20`, `e30`) is judged by protocol amendment 3's four conditions.
2. **Against frozen candidate 2:** the highest passing candidate replaces frozen candidate 2 as the one to test only if it also beats it on COnPOff@50 with the 95% interval excluding zero. Otherwise frozen candidate 2 stays.
3. **No test check yet:** none is spent without the user's go-ahead.

**Reported, deciding nothing:**
- each setting against `e0`;
- COnPOff@150, and frame accuracy split into pitch and voicing;
- the ends diagnostic's breakdown (right, early and late ends) for each setting, on seed 0.

## Steps

1. **transcPybe** (`claude/spurious-notes`, local):
   - add `end_snap_window_s`, with validation;
   - add the `end_times` array to `OnsetTrack`, passed through the rebuilds;
   - add `snap_note_ends`, called at the end of `_decode_pass`;
   - tests:
     - a late end moves back to an end time, and an early one forward;
     - an end never passes the next onset, nor comes within the shortest note of its own onset;
     - with the option off, nothing changes;
   - commit, without pushing.
2. **boundary** (PR #4 branch):
   - `onset_track(..., end_times=None)`;
   - `rwc_dev.py` passes the offset peaks;
   - a test;
   - commit, without pushing.
3. **Musix:**
   - copy both commits;
   - decode the 12 runs;
   - score them, and apply the rules (`ends.txt`).
4. **Cost:** no CSD3 time, and about 30 minutes on musix.
