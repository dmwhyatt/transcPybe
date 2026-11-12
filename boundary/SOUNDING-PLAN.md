# Sounding plan: merging notes the network hears as one

Written 26 October 2026, after the offset and sounding diagnostic (`arms/diag-outputs.log` and `diag-outputs2.log` on musix) and before any decode with this change. Scoring follows `protocol-amendment-2.md`, as before. Nothing here changes it.

## Why

The wrong-pitch diagnostic found that transcPybe measures pitch correctly, and that wrong pitches come from note spans. The commonest fault is a reference note cut in two, typically at a scoop or slide into it: the first part gets the wrong pitch, and the second part is an extra note.

The offset and sounding diagnostic looked at every boundary where one of transcPybe's notes runs into the next (within 30 ms), on the development half:
- About 28% of these boundaries are false: they fall inside a reference note. 84–88% of the false ones cut one reference note in two; the rest start a note about 140 ms late.
- The network's **sounding output** separates real boundaries from false ones better than its other outputs. At a false boundary it stays at about 0.98; at a real one it dips to 0.3–0.45. The separation score (AUC) is 0.85 for preset B's notes and 0.89 for the network decode's, against 0.79 for the onset output.
- Where the sounding output stays above 0.98 within 30 ms either side of the boundary:

  | Notes from | Cut-in-two boundaries caught | Real boundaries caught |
  |---|---|---|
  | Network decode (round 1, seed 0) | 54% (about 860) | 4% (about 200) |
  | Preset B | 51% (about 730) | 4% (about 175) |

  At 0.95 it catches 64–67% of the cut-in-two boundaries and 9–11% of the real ones.
- About 40% of the late starts would be caught as well (about 90 for the network decode). Merging those is wrong, and counts against the change.

transcPybe's decoder uses none of the sounding output today.

The diagnostic averaged the three seeds' outputs. Here each seed is decoded with its own outputs, so the counts will differ a little.

## Why not the offset output

Its peaks find 79% of the reference note ends within 100 ms, against 63–65% for transcPybe's own ends. But COnP@150 does not score note ends; they matter only through the pitch a note's span takes in. The late starts, where an end would matter most, are 12–16% of the false boundaries. The offset output is left for a later plan.

## The change

**In transcPybe:** one option, off by default, so preset B and the shipped settings are unchanged.
- `sustain_merge_above` (default 0, off). After cutting, snapping and late-arrival attachment, two segments that abut (the second starts within 30 ms of the first's end) are merged into one when the sounding track stays above this value over every frame within 30 ms of the second's start.
- The merged segment runs from the first's start to the second's end, and keeps the first's measuring point. Its pitch is measured from the f0 track as usual. A note cut in three can merge back to one.
- The sounding track travels on the `OnsetTrack`, as a new optional `sounding` array on its envelope's time grid. Every place that rebuilds an `OnsetTrack` passes it on. (The pitch plan's settle option was lost exactly that way, in a rebuild.)

**In boundary:**
- `infer.onset_track` takes the network's sounding output and puts it on the track.
- `rwc_dev.py` gains a hand-over called `own`: transcPybe's own onset detector (preset B's), with only the network's sounding output added. The network then changes nothing but the merges.

## What is decoded

On the RWC-P development half (50 songs, 48 at note level), the three M4Singer-only chosen checkpoints from round 1, each with its own outputs. Two hand-overs, at four settings each:

| Hand-over | Decoder | Onsets |
|---|---|---|
| network | B-plain, onset threshold 0.7, every onset may cut (round 1's best) | the network's |
| own | preset B as it stands (B-full) | transcPybe's own |

| Setting | `sustain_merge_above` |
|---|---|
| `s0` | 0 (off) |
| `s95` | 0.95 |
| `s98` | 0.98 |
| `s99` | 0.99 |

This makes 24 decodes. Two checks come before any rule:
- the network at `s0` reproduces round 1 (0.6653);
- `own` at `s0` reproduces preset B (0.6589).

If either fails, nothing is judged until the cause is found.

## Decision rules, fixed now

Each comparison takes each song's COnP@150 averaged over the 3 seeds, pairs it over songs, and uses a 95% bootstrap interval.

1. **Against B (amendment 2).** The six candidates are both hand-overs at `s95`, `s98` and `s99`. A candidate passes only if all of these hold:
   - it beats B by at least 0.01, with the interval excluding zero;
   - at least 2 of its 3 seeds beat B on their own;
   - frame overall accuracy falls by no more than 0.005;
   - notes in silence rise by no more than 0.5 points.

   If several pass, the one with the highest development score is frozen and hashed (checkpoints, hand-over, setting, both code commits). It is then checked once on the test half: the first of the two test checks amendment 2 allows. Claude asks before spending it.
2. **If no candidate passes,** there is no test check. Claude reports, and the next step gets a plan of its own.

**Reported, deciding nothing:**
- each setting against its own `s0` (the merge's effect alone);
- COnP@50 and COn@150;
- notes transcribed, and the note-error breakdown of `diag_notes.py` (wrong pitch, extra notes, merged reference notes);
- the boundary diagnostic rerun on every setting's notes: cut-in-two boundaries left, and real boundaries lost.

## Steps

1. **transcPybe** (`claude/spurious-notes`, local):
   - add `sustain_merge_above` to `TranscriptionConfig`, with validation;
   - add the `sounding` array to `OnsetTrack`, and pass it through every rebuild in `refine_segments`;
   - add `merge_sustained_boundaries`, and call it in `_decode_pass` after `attach_late_arrivals`;
   - tests:
     - two abutting segments merge when the sounding stays high, and stay apart when it dips;
     - a note cut in three merges back to one;
     - the sounding array survives the voicing-break and timbre rebuilds;
     - with the option off, nothing changes;
   - commit, without pushing.
2. **boundary** (PR #4 branch):
   - `onset_track(..., sounding=None)`;
   - `rwc_dev.py` passes the sounding output, and gains the `own` hand-over;
   - a test;
   - commit, without pushing.
3. **Musix:**
   - copy both commits;
   - decode the 24 runs, 8 at a time on the CPU;
   - score them, and apply the rules with a short script (`sounding.txt`);
   - run the breakdown and the boundary diagnostic for each setting.
4. **Cost:** no CSD3 time, and about an hour on musix.
